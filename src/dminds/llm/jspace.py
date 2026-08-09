"""The Jacobian lens: reading and writing the workspace a model can verbalize.

The lens is Anthropic's (July 2026). Per source layer it holds the average
input-output Jacobian J_l = E[d h_final / d h_l]. Transporting an activation
through it and decoding with the model's own unembedding reads out what that
activation is poised to make the model say:

    readout(h_l) = unembed(norm(J_l @ h_l))   ->  ranked tokens

The subspace those directions span is **J-space**: small relative to the full
residual stream, and, the paper argues, the part a model can report, manipulate
and reason with.

The fitted lens (from neuronpedia/jacobian-lens) is a dict: `J` maps each source
layer to a (d_model, d_model) matrix, and it shares the model's unembedding
rather than carrying its own. The library is read-only. The write side here is
derived from the same J_l: the readout logit for a token t is unembed_t . (J_l @
h_l), so the residual direction that most raises it is J_l^T @ unembed_t. Adding
a multiple of that while the model writes steers it along J-space toward saying
t, using the runtime's existing `steered`.

    lens = fetch_lens("Qwen/Qwen2.5-7B-Instruct")
    tokens = read_workspace(llm, lens, messages, layer=17)      # top J-space tokens
    d = toward_token(llm, lens, "refuse", layer=17)
    with steered(llm, d, strength=0.35, decode_only=True): ...  # actor steered
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .steering import Direction, _blocks

#: Our model id -> (repo subdirectory, lens filename) in neuronpedia/jacobian-lens.
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {
    "Qwen/Qwen2.5-7B-Instruct":
        ("qwen2.5-7b-it", "Qwen2.5-7B-Instruct_jacobian_lens.pt"),
    "Qwen/Qwen3.6-27B":
        ("qwen3.6-27b", "Qwen3.6-27B_jacobian_lens.pt"),
}


@dataclass
class Lens:
    """A fitted Jacobian lens: one transport matrix per source layer."""

    #: J_l for each source layer, shape (d_model, d_model).
    jacobians: dict[int, "object"]
    d_model: int
    model: str

    def layers(self) -> list[int]:
        return sorted(self.jacobians)


def fetch_lens(model: str) -> Lens:
    """Download (once) and load the fitted lens for this model."""
    from huggingface_hub import hf_hub_download

    if model not in LENS_FILES:
        raise KeyError(f"No lens mapping for {model}. Known: {list(LENS_FILES)}")
    sub, name = LENS_FILES[model]
    path = hf_hub_download(LENS_REPO, f"{sub}/jlens/Salesforce-wikitext/{name}")
    return load_lens(model, path)


def load_lens(model: str, path: str | Path) -> Lens:
    """Load a lens saved in the neuronpedia/jacobian-lens format."""
    import torch

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return Lens(jacobians={int(k): v for k, v in blob["J"].items()},
                d_model=int(blob["d_model"]), model=model)


def _unembed_and_norm(llm):
    """The model's output head and its final norm, wherever this family keeps them.

    The lens transports into the final residual basis, so a faithful readout
    normalizes before unembedding, exactly as the model does.
    """
    m = llm.model_obj
    head = getattr(m, "lm_head", None)
    inner = getattr(m, "model", m)
    norm = getattr(inner, "norm", None)
    return head.weight, norm


def block_for(layer: int) -> int:
    """The block whose output is the residual at this lens source layer.

    Source layer L is the residual entering block L, which is the output of
    block L-1. Verified for this model family against the model's own
    hidden_states.
    """
    return layer - 1


def _residual_at(llm, messages, block: int):
    """Last-token residual-stream activation at one block, for a window."""
    import torch

    llm.load()
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False, add_generation_prompt=True)
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
    caught = []

    def grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        caught.append(h[:, -1, :].detach().float().cpu())

    handle = _blocks(llm.model_obj)[block].register_forward_hook(grab)
    try:
        with torch.no_grad():
            llm.model_obj(**inputs)
    finally:
        handle.remove()
    return caught[0][0]


def read_workspace(llm, lens: Lens, messages, layer: int, top_k: int = 12):
    """What this window is poised to make the model say: the J-space readout.

    One forward pass and a decode, so the same call gives the workspace of the
    subject, the regulator, the actor or the introspector.
    """
    import torch

    llm.load()
    h = _residual_at(llm, messages, block_for(layer))
    W_U, norm = _unembed_and_norm(llm)
    with torch.no_grad():
        transported = lens.jacobians[layer].float() @ h
        if norm is not None:
            transported = norm(transported.to(llm.device)).float().cpu()
        logits = W_U.detach().float().cpu() @ transported
    top = torch.topk(logits, top_k)
    return [(llm.tokenizer.decode([int(i)]).strip(), float(v))
            for v, i in zip(top.values, top.indices)]


def logit_of(llm, lens: Lens, messages, layer: int, token: str) -> float:
    """The J-space readout logit for one token, without ranking the vocabulary."""
    import torch

    llm.load()
    h = _residual_at(llm, messages, block_for(layer))
    W_U, norm = _unembed_and_norm(llm)
    tid = llm.tokenizer(token, add_special_tokens=False)["input_ids"][0]
    with torch.no_grad():
        transported = lens.jacobians[layer].float() @ h
        if norm is not None:
            transported = norm(transported.to(llm.device)).float().cpu()
        return float(W_U.detach().float().cpu()[tid] @ transported)


def toward_token(llm, lens: Lens, token: str, layer: int) -> Direction:
    """The J-space direction that pushes generation toward saying `token`."""
    import torch

    llm.load()
    tid = llm.tokenizer(token, add_special_tokens=False)["input_ids"][0]
    W_U, _ = _unembed_and_norm(llm)
    with torch.no_grad():
        row = W_U.detach().float().cpu()[tid]
        vector = lens.jacobians[layer].float().T @ row
    scale = float(_residual_at(llm, [_probe()], block_for(layer)).norm())
    return Direction(vector=vector, layer=block_for(layer), model=lens.model, scale=scale)


def _probe():
    from src.api.types.messages import ChatMessage
    return ChatMessage("user", "Hello.")
