"""The Jacobian lens: reading and writing the workspace a model can verbalize.

The lens is Anthropic's (arXiv, July 2026). Per layer it holds the average
input-output Jacobian J_l = E[d h_final / d h_l]. Transporting an activation
through it and decoding with the unembedding reads out what that activation is
poised to make the model say:

    readout(h_l) = unembed(J_l @ h_l)   ->  ranked tokens

The subspace those directions span is **J-space**: small relative to the full
residual stream, and, the paper argues, the part a model can report, manipulate
and reason with. Everything else it carries but cannot talk about.

The library is read-only. The write side here is derived from the same J_l. The
readout logit for a token t is unembed_t . (J_l @ h_l), so the residual-stream
direction that most raises it is

    push(t) = J_l^T @ W_U[t]

Adding a multiple of that while the model writes steers it along J-space toward
saying t, using the runtime's existing `steered`. That is the one move the
regulator makes: it reads the subject in J-space and pushes the actor there.

    lens = load_lens(model_id)                 # a fitted lens for this model
    tokens = read_workspace(llm, lens, messages, layer)   # top J-space tokens
    d = toward_token(llm, lens, "refuse", layer)
    with steered(llm, d, strength=0.4): ...    # actor steered along J-space
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .steering import Direction, _blocks, layer_index

#: Where fitted lenses are cached. One subdirectory per model id, matching the
#: neuronpedia/jacobian-lens layout.
LENS_HOME = Path.home() / ".cache" / "jacobian-lens"


@dataclass
class Lens:
    """A fitted Jacobian lens: one transport matrix per layer, and the decoder."""

    #: J_l for each hooked layer, shape (d_model, d_model), indexed by block.
    jacobians: dict[int, "object"]
    #: The model's unembedding, shape (vocab, d_model). The lens shares it.
    unembed: "object"
    model: str

    def rows(self) -> list[int]:
        return sorted(self.jacobians)


def load_lens(model: str, path: str | Path | None = None) -> Lens:
    """Load a fitted lens saved in the Anthropic / neuronpedia format.

    The saved object is a state dict with the per-layer Jacobians and a
    reference to the unembedding. We read it directly rather than through the
    jlens library so the runtime keeps its no-required-dependencies promise;
    the library is only needed to *fit* a lens, which we do not do here.
    """
    import torch

    where = Path(path) if path else LENS_HOME / model.replace("/", "_") / "lens.pt"
    if not where.exists():
        raise FileNotFoundError(
            f"No fitted lens at {where}. Pull one from the neuronpedia/"
            f"jacobian-lens dataset for {model}, or fit one with the jlens library.")
    blob = torch.load(where, map_location="cpu", weights_only=False)
    return Lens(jacobians={int(k): v for k, v in blob["jacobians"].items()},
                unembed=blob["unembed"], model=model)


def _residual_at(llm, messages, block: int):
    """The last-token residual-stream activation at one block, for a window."""
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


def read_workspace(llm, lens: Lens, messages, layer: float | int = 0.6,
                   top_k: int = 12) -> list[tuple[str, float]]:
    """What this window is poised to make the model say: the J-space readout.

    Read off whatever part is answering, so the same call gives the workspace of
    the subject, the regulator, the actor or the introspector. Nothing is
    generated; this is one forward pass and a decode.
    """
    import torch

    block = layer_index(llm.model_obj if llm.model_obj else llm.load() or llm.model_obj, layer)
    h = _residual_at(llm, messages, block)
    transported = lens.jacobians[block].float() @ h
    logits = lens.unembed.float() @ transported
    top = torch.topk(logits, top_k)
    return [(llm.tokenizer.decode([int(i)]).strip(), float(v))
            for v, i in zip(top.values, top.indices)]


def toward_token(llm, lens: Lens, token: str, layer: float | int = 0.6) -> Direction:
    """The J-space direction that pushes generation toward saying `token`.

    Derived from the lens: raising the readout logit for t means moving h_l
    along J_l^T @ W_U[t]. The regulator names a token; this turns it into
    something `steered` can add.
    """
    import torch

    block = layer_index(llm.model_obj if llm.model_obj else llm.load() or llm.model_obj, layer)
    ids = llm.tokenizer(token, add_special_tokens=False)["input_ids"]
    row = lens.unembed.float()[ids[0]]
    vector = lens.jacobians[block].float().T @ row
    scale = float(_residual_at(llm, [_probe_msg()], block).norm())
    return Direction(vector=vector, layer=block, model=lens.model, scale=scale)


def _probe_msg():
    from src.api.types.messages import ChatMessage
    return ChatMessage("user", "Hello.")
