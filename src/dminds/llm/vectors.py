"""Steering vectors extracted elsewhere, loaded as directions this runtime steers with.

Two libraries are read, both produced by the persona-EM project and both stored
at every layer, so which depth to steer at is a choice made here rather than at
extraction time.

    emotions   171 emotion vectors and the `default emotion`, the grand mean
               that sits at the neutral centre of the space
    roles      the 25-role persona cast, plus the assistant axis and the evil
               and misalignment directions

The two families are not on the same scale. A role vector's norm is about ten
times an emotion vector's, so a raw strength means two different things. Every
direction here carries the layer's own activation norm as its scale, and
`steered(..., relative=True)` expresses strength as a fraction of it, which puts
both families in the same units.

    pack = load_pack(EMOTION_VECTORS, ROLE_VECTORS, row=34)
    directions = as_directions(pack, llm, model="Qwen/Qwen2.5-14B-Instruct")
"""

from __future__ import annotations

from pathlib import Path

from .steering import Direction, _blocks

#: Where the persona-EM project keeps its exports.
LIBRARY = Path.home() / "work" / "bluedot-tais-project-2026"

#: The extraction stores one row per layer plus a row 0 for the embedding, so
#: the vector for layer L sits at row L and is added at block L-1. Mixing the
#: two silently steers the wrong depth, which is why `check_rows` exists.
def block_of(row: int) -> int:
    return row - 1


def load_emotions(path: str | Path) -> dict:
    """The emotion library: named vectors, the default emotion, and the neutral mean."""
    import torch

    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {"vectors": data["vectors"],
            "default": data["default_emotion"],
            "neutral": data["neutral_mean"],
            "base": data["base"],
            "rows": int(data["layers"])}


def load_roles(path: str | Path) -> dict:
    """The persona cast, and the axes fitted alongside it."""
    import torch

    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {"vectors": data["role_vectors"],
            "assistant_axis": data["vA"],
            "evil": data["v_evil"],
            "misalignment": data["vEM"],
            "row": int(data["lstar"])}


def layer_scale(llm, row: int, prompts: list[str]) -> float:
    """The typical activation norm at this depth, measured on this model.

    Steering strength is a fraction of it. Measuring rather than assuming keeps
    one strength meaning one size of nudge across models and across the two
    vector families.
    """
    import torch

    llm.load()
    block = _blocks(llm.model_obj)[block_of(row)]
    seen = []

    def grab(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seen.append(hidden[:, -1, :].detach().float().cpu())

    handle = block.register_forward_hook(grab)
    try:
        for text in prompts:
            chat = [{"role": "user", "content": text}]
            prompt = llm.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
            inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                llm.model_obj(**inputs)
    finally:
        handle.remove()
    return float(torch.cat(seen, dim=0).norm(dim=-1).mean())


def check_convention(llm, prompt: str, row: int) -> dict:
    """Prove that the row we steer at is the row the vectors were taken from.

    The extraction saved `hidden_states[L]`, which is the output of block L-1.
    Steering the wrong block would be silent, so this compares the two tensors
    directly rather than trusting the arithmetic: they are the same tensor or
    the convention is wrong.
    """
    import torch

    llm.load()
    block = _blocks(llm.model_obj)[block_of(row)]
    caught = []

    def grab(_module, _inputs, output):
        caught.append((output[0] if isinstance(output, tuple) else output).detach().float().cpu())

    handle = block.register_forward_hook(grab)
    try:
        chat = [{"role": "user", "content": prompt}]
        text = llm.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        inputs = llm.tokenizer(text, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            out = llm.model_obj(**inputs, output_hidden_states=True)
    finally:
        handle.remove()

    hooked = caught[0]
    stated = out.hidden_states[row].detach().float().cpu()
    gap = float((hooked - stated).abs().max())
    return {"row": row, "block": block_of(row), "blocks": len(_blocks(llm.model_obj)),
            "hidden_states_rows": len(out.hidden_states),
            "max_abs_difference": gap, "matches": gap < 1e-3}


def as_direction(vector, row: int, model: str, scale: float) -> Direction:
    """One saved row, as something `steered` accepts."""
    return Direction(vector=vector[row].float(), layer=block_of(row),
                     model=model, scale=scale)
