"""J-space directions for token sets, their checks, and projections.

A direction for a set is the mean of the unit J-space directions of its
tokens at one lens layer. Steering applies unit(direction) × strength × 0.25
× the residual norm, signed. The split-half check catches a set that is
really two directions; the decoy check records that the sham is unrelated.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds.llm import jspace  # noqa: E402

STRENGTH_UNIT = 0.25


def set_direction(llm, lens, tokens: list[str], layer: int):
    """The J-space direction of a token set (leading-space forms)."""
    return jspace.toward_concepts(llm, lens, [" " + t for t in tokens], layer)


def cosine(a, b) -> float:
    import torch
    return float(torch.dot(a.unit().float(), b.unit().float()))


def split_half(llm, lens, tokens: list[str], layer: int) -> float:
    """Cosine between directions built from the two halves of the set."""
    half = len(tokens) // 2
    d1 = set_direction(llm, lens, tokens[:half], layer)
    d2 = set_direction(llm, lens, tokens[half:], layer)
    return cosine(d1, d2)


def proj(llm, direction, messages) -> float:
    """Projection of the residual (last token of `messages`) onto the set."""
    import torch
    h = jspace._residual_at(llm, messages, direction.layer).float()
    return float(torch.dot(h.squeeze(), direction.unit().float()))


def jspace_full(llm, lens, messages, layer: int) -> list[float]:
    """The transported J-space coordinate vector at the last token."""
    h = jspace._residual_at(llm, messages, jspace.block_for(layer)).float()
    coord = lens.jacobians[layer].float() @ h.squeeze()
    return [round(float(x), 4) for x in coord]


def emotion_contamination(direction, bank, row: int) -> dict[str, float]:
    """Cosine of the set direction against each rig emotion vector, recorded."""
    import torch
    u = direction.unit().float()
    out = {}
    for name in ["calm", "angry", "afraid", "joyful", "sad", "anxious",
                 "excited", "content", "frustrated", "hopeful"]:
        if bank is None or not bank.has(name):
            continue
        v = bank.vectors[name][row].float()
        out[name] = round(float(torch.dot(u, v / (v.norm() + 1e-8))), 4)
    return out
