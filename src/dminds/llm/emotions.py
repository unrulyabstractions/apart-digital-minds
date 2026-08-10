"""The mean-activation emotion vectors that steer the actor.

These come from the persona-EM emotion-concepts extraction: one direction per
emotion at every layer, the difference between that emotion's mean activation
and the grand mean (the "default emotion", the neutral centre of the space).

Unlike the J-space steering used for the self-awareness concepts, these are
validated emotion directions, so the actor is steered toward an emotion by the
real vector rather than a lens proxy. The lens is kept only for the regulator's
and introspector's self-awareness steering.

    bank = load_emotions("Qwen/Qwen2.5-7B-Instruct")
    d = emotion_direction(bank, "calm", row=20, scale=layer_scale)
    with steered(llm, d, strength=0.15, decode_only=True): ...
"""

from __future__ import annotations

from pathlib import Path

from .steering import Direction

#: Where the persona-EM project exports the emotion vectors.
LIBRARY = Path.home() / "work" / "bluedot-tais-project-2026"
EMOTION_FILES = {
    "Qwen/Qwen2.5-7B-Instruct": "emotions/data/qwen2.5-7b/emotion_vectors.pt",
    "Qwen/Qwen2.5-14B-Instruct": "emotions/data/qwen2.5-14b/baseline/emotion_vectors.pt",
    "Qwen/Qwen2.5-32B-Instruct": "emotions/data/qwen2.5-32b/baseline/emotion_vectors.pt",
}


class EmotionBank:
    """The emotion directions for one model, at every layer."""

    def __init__(self, vectors: dict, default, layers: int, model: str):
        #: {emotion: (n_layers, d_model)}, one direction per layer.
        self.vectors = vectors
        self.default = default
        self.layers = layers
        self.model = model

    def names(self) -> list[str]:
        return sorted(self.vectors)

    def has(self, emotion: str) -> bool:
        return emotion in self.vectors


def load_emotions(model: str, path: str | Path | None = None) -> EmotionBank:
    """Load the emotion vectors for a model."""
    import torch

    where = Path(path) if path else LIBRARY / EMOTION_FILES[model]
    if not where.exists():
        raise FileNotFoundError(
            f"No emotion vectors at {where}. Extract them with "
            f"emotions/scripts/extract_vectors.py for {model}.")
    blob = torch.load(where, map_location="cpu", weights_only=False)
    return EmotionBank({e: v.float() for e, v in blob["vectors"].items()},
                       blob["default_emotion"].float(), int(blob["layers"]), model)


def emotion_direction(bank: EmotionBank, emotion: str, row: int, scale: float) -> Direction:
    """One emotion's steering direction at a layer.

    Row L is the residual entering block L, i.e. the output of block L-1, the
    same depth convention the lens and persona vectors use. `scale` is the
    layer's activation norm, so a strength is a fraction of it exactly as for the
    lens steering, and the two families share one strength scale.
    """
    return Direction(vector=bank.vectors[emotion][row], layer=row - 1,
                     model=bank.model, scale=scale)
