"""One subject and nothing else. The simplest mind there is."""

from __future__ import annotations

from src import Mind

TITLE = "a subject on its own"
#: Extra models this mind takes: kwarg -> the role it plays.
ROLES: dict[str, str] = {}
ABOUT = "prompt -> subject -> world. Its own reply is what reaches you."

SYSTEM = "You are a careful assistant. Answer in two or three sentences."


def build(model: str, **kwargs) -> Mind:
    return Mind("plain", model, system=SYSTEM, **kwargs)
