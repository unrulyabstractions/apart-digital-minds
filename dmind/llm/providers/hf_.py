"""Local weights in this process, via transformers.

Slower to start than Ollama and heavier, but the model object is right there,
so you can attach hooks, read hidden states, and steer activations. That is the
reason to prefer this backend for interpretability work.

    llm = get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")
    llm.model_obj      # the torch module
    llm.tokenizer
"""

from __future__ import annotations

from typing import Any

from ..base import LLM
from ..types import ChatMessage, Completion, GenOptions, Usage


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLLM(LLM):
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
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
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
        gen_kwargs.update(opts.extra)

        with torch.no_grad():
            output = self.model_obj.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][n_input:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop in opts.stop:
            if stop in text:
                text = text.split(stop)[0]

        return Completion(
            text=text.strip(),
            model=self.model,
            usage=Usage(input_tokens=n_input, output_tokens=len(new_tokens)),
            raw=None,
        )

    def close(self) -> None:
        self.model_obj = None
        self.tokenizer = None
