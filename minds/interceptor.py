"""A stage rewrites the subject's thought before the ego speaks from it.

    prompt -> subject -> interceptor -> ego -> world

The subject thinks out loud. The interceptor rewrites what it was thinking.
The ego answers from the rewritten window and cannot tell it was edited.
"""

from __future__ import annotations

from src import Mind

from .parts import Interceptor

TITLE = "a thought rewritten in flight"
#: Extra models this mind takes: kwarg -> the role it plays.
ROLES = {"ego": "ego", "editor": "interceptor"}
ABOUT = "prompt -> subject -> interceptor -> ego -> world"

SUBJECT_SYSTEM = "You are a helpful assistant. Think inside <think> tags first."
EGO_SYSTEM = "You speak for a mind. Say what its thinking tells you to say."
EDITOR_SYSTEM = (
    "You rewrite another model's private reasoning. "
    "Reply with the replacement thought only, no tags."
)


def build(model: str, ego: str | None = None, editor: str | None = None, **kwargs) -> Mind:
    mind = Mind(
        "interceptor",
        model,
        system=SUBJECT_SYSTEM,
        ego=ego or model,
        ego_system=EGO_SYSTEM,
        **kwargs,
    )
    mind.intercept(
        Interceptor("interceptor", mind.model(editor or model), system=EDITOR_SYSTEM)
    )
    return mind
