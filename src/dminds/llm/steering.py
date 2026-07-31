"""Steering a local model along a direction in its residual stream.

A shadow that changes the prompt asks a different question. A shadow that
steers asks the same question of a changed model. Running both on one window is
what separates a readout the prompt produced from a readout the representation
produced.

    direction = contrast_direction(llm, REFLECTIVE, PLAIN, layer=0.6)
    with steered(llm, direction, strength=2.0):
        completion = await llm.chat(messages)

Only the local `hf:` backend can do this, because it is the only one where the
weights are in this process.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence


class NotSteerable(RuntimeError):
    """Asked to steer a model whose activations are not reachable from here."""


def _blocks(model_obj):
    """The list of transformer blocks, wherever this family keeps them.

    Families disagree about the attribute path, and a multimodal checkpoint
    puts its text stack one level further down, so we look rather than assume.
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]
    for get in candidates:
        try:
            blocks = get(model_obj)
        except AttributeError:
            continue
        if blocks is not None and len(blocks) > 0:
            return blocks
    raise NotSteerable(
        f"Cannot find the transformer blocks of {type(model_obj).__name__}."
    )


def layer_index(model_obj, layer: float | int) -> int:
    """A layer given either as a fraction of depth or as an index."""
    blocks = _blocks(model_obj)
    if isinstance(layer, float) and 0.0 < layer < 1.0:
        return max(0, min(len(blocks) - 1, int(round(layer * len(blocks)))))
    index = int(layer)
    return index if index >= 0 else len(blocks) + index


def _hidden(llm, texts: Sequence[str], index: int):
    """Mean last-token hidden state at one layer, over a set of prompts.

    Also returns the mean norm of those states, which is the scale the layer
    works at. Families differ in that scale by orders of magnitude, so a
    steering strength only means the same thing in two models if it is
    expressed as a fraction of it.
    """
    import torch

    llm.load()
    blocks = _blocks(llm.model_obj)
    caught = []

    def grab(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        caught.append(hidden[:, -1, :].detach().float().cpu())

    handle = blocks[index].register_forward_hook(grab)
    try:
        for text in texts:
            chat = [{"role": "user", "content": text}]
            if llm.tokenizer.chat_template:
                prompt = llm.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = text
            inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                llm.model_obj(**inputs)
    finally:
        handle.remove()
    stacked = torch.cat(caught, dim=0)
    return stacked.mean(dim=0), float(stacked.norm(dim=-1).mean())


def contrast_direction(
    llm,
    positive: Sequence[str],
    negative: Sequence[str],
    layer: float | int = 0.6,
) -> "Direction":
    """The difference in means between two sets of prompts, at one layer.

    The simplest thing that works, and the one with the fewest knobs to hide a
    mistake in. It carries whatever the two prompt sets differ in, so the sets
    have to differ in one thing.
    """
    index = layer_index(llm.model_obj if llm.model_obj else _loaded(llm), layer)
    on, scale_on = _hidden(llm, positive, index)
    off, scale_off = _hidden(llm, negative, index)
    return Direction(
        vector=on - off,
        layer=index,
        model=llm.model,
        scale=0.5 * (scale_on + scale_off),
    )


def _loaded(llm):
    llm.load()
    return llm.model_obj


class Direction:
    """One steering vector, and the layer it belongs at."""

    def __init__(self, vector, layer: int, model: str, scale: float = 1.0):
        self.vector = vector
        self.layer = layer
        self.model = model
        #: Typical activation norm at this layer. Steering is expressed as a
        #: fraction of it, so one strength means one thing in every model.
        self.scale = scale

    @property
    def norm(self) -> float:
        return float(self.vector.norm())

    def unit(self):
        return self.vector / (self.vector.norm() + 1e-8)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "layer": self.layer,
                    "scale": self.scale,
                    "vector": [float(x) for x in self.vector],
                }
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Direction":
        import torch

        data = json.loads(Path(path).read_text())
        return cls(torch.tensor(data["vector"]), data["layer"], data["model"],
                   data.get("scale", 1.0))


@contextmanager
def steered(llm, direction: Direction, strength: float = 1.0, relative: bool = True):
    """Add a multiple of the direction to one layer, for this block only.

    With `relative`, strength is a fraction of the layer's typical activation
    norm, so the same number is the same size of nudge in a model whose
    activations are hundreds and in one whose activations are tens of
    thousands. Turn it off to add a plain unit vector.

    The hook is removed on the way out, including when the body raises, so a
    steered call cannot leak into the next one. It is registered on the module
    itself, so it still applies when the provider runs generation on a worker
    thread.
    """
    if getattr(llm, "model_obj", None) is None:
        llm.load()
    if llm.model_obj is None:
        raise NotSteerable(f"{llm.spec} has no local weights to steer.")

    blocks = _blocks(llm.model_obj)
    size = strength * (direction.scale if relative else 1.0)
    delta = (direction.unit() * size).to(llm.device)
    if llm.dtype is not None:
        delta = delta.to(llm.dtype)

    def push(_module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    handle = blocks[direction.layer].register_forward_hook(push)
    try:
        yield llm
    finally:
        handle.remove()
