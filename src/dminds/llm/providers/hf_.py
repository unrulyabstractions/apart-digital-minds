"""Local weights in this process, via transformers.

Slower to start than Ollama and heavier, but the model object is right there,
so you can attach hooks, read hidden states, and steer activations. That is the
reason to prefer this backend for interpretability work.

    llm = get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")
    llm.model_obj      # the torch module
    llm.tokenizer

Weights are held once per checkpoint. Wrap this in `SharedLLM` to give many
modules the same copy, because a mind with ten probes would otherwise load ten
of them. Unified memory holds the whole model, so there is nothing to offload
to a host: a checkpoint that does not fit has to be a smaller checkpoint.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseLLM
from ....api.types import ChatMessage, Completion, GenOptions, Usage


def free_memory() -> None:
    """Hand back whatever the accelerator is still holding.

    Worth calling between two models in a sweep, where the first one's cache
    would otherwise sit in memory while the second one loads.
    """
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLLM(BaseLLM):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        spec: str | None = None,
        device: str | None = None,
        dtype: Any = None,
        load_kwargs: dict | None = None,
    ):
        super().__init__(model, spec)
        self.device = device
        self.dtype = dtype
        self.load_kwargs = load_kwargs or {}
        self.model_obj = None
        self.tokenizer = None

    def load(self) -> None:
        """Load weights. Called automatically on first use."""
        if self.model_obj is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. Install them with: "
                "pip install 'dmind[hf]'"
            ) from exc

        self.device = self.device or _pick_device()
        if self.dtype is None:
            self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model, dtype=self.dtype, **self.load_kwargs
        ).to(self.device)
        self.model_obj.eval()

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        import torch

        self.load()

        chat = [{"role": m.role, "content": m.content} for m in messages]
        extra = dict(opts.extra)
        # Template options rather than generation options. `enable_thinking`
        # lives here, and a family whose template ignores it is unaffected.
        template_kwargs = extra.pop("chat_template_kwargs", {})
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
        else:
            # A base model with no chat template. Fall back to plain concatenation.
            prompt = (
                "\n\n".join(f"{m.role}: {m.content}" for m in messages) + "\n\nassistant:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n_input = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": opts.max_tokens,
            "do_sample": opts.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        }
        if opts.temperature > 0:
            gen_kwargs["temperature"] = opts.temperature
        if opts.seed is not None:
            torch.manual_seed(opts.seed)
        gen_kwargs.update(extra)

        # `inference_mode` over `no_grad`: it also skips version counting and
        # view tracking, which a forward-only workload never needs.
        with torch.inference_mode():
            output = self.model_obj.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][n_input:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_new = len(new_tokens)
        # The prompt and the generation are the two big tensors here. Drop them
        # before returning so a long sweep does not hold the last call's
        # activations while the next one allocates.
        del output, new_tokens, inputs
        for stop in opts.stop:
            if stop in text:
                text = text.split(stop)[0]

        return Completion(
            text=text.strip(),
            model=self.model,
            usage=Usage(input_tokens=n_input, output_tokens=n_new),
            raw=None,
        )

    def score(self, messages: list[ChatMessage], choices: list[str]) -> dict[str, float]:
        """How much probability the model puts on each of a few answers.

        A forced choice read by generating gives one word and hides how close
        the runner-up was. Scoring gives the whole distribution, so a readout
        moves with the conversation even on turns where the winner does not
        change. It also costs one forward pass per choice and no decoding.

        Returns probabilities over `choices` alone, normalised, with each
        choice's tokens scored in context and length-normalised so a longer
        word is not penalised for being longer.
        """
        import torch

        self.load()
        chat = [{"role": m.role, "content": m.content} for m in messages]
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = "\n\n".join(f"{m.role}: {m.content}" for m in messages)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        scores = []
        for choice in choices:
            choice_ids = self.tokenizer(choice, add_special_tokens=False,
                                        return_tensors="pt")["input_ids"]
            ids = torch.cat([prompt_ids, choice_ids], dim=-1).to(self.device)
            with torch.inference_mode():
                logits = self.model_obj(input_ids=ids).logits
            # Line the predictions up with the tokens they predict.
            logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            wanted = ids[0, 1:]
            n = choice_ids.shape[-1]
            taken = logprobs[-n:].gather(1, wanted[-n:].unsqueeze(1)).squeeze(1)
            scores.append(float(taken.mean()))
            del logits, logprobs, ids

        top = max(scores)
        weights = [pow(2.718281828, s - top) for s in scores]
        total = sum(weights)
        return {c: w / total for c, w in zip(choices, weights)}

    def close(self) -> None:
        """Drop the weights and give the memory back.

        Without the release a sweep holds every model it has finished with,
        and the last one in the list is the one that fails to load.
        """
        self.model_obj = None
        self.tokenizer = None
        free_memory()
