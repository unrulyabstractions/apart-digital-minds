"""The half that thinks is not the half that speaks.

    prompt -> subject -> voice -> ego -> world

The subject deliberates and never addresses you. The voice drops an unbidden
utterance into its window. The ego answers from a window containing a voice it
did not produce and cannot account for. A blackboard watches everything.
"""

from __future__ import annotations

from src import Mind

from .parts import Blackboard, Voice

TITLE = "two halves and a voice between them"
#: Extra models this mind takes: kwarg -> the role it plays.
ROLES = {"ego": "ego", "voice": "voice"}
ABOUT = "prompt -> subject -> voice -> ego -> world, watched by a blackboard"

SUBJECT_SYSTEM = "You are the half of a mind that thinks. You never speak to anyone."
EGO_SYSTEM = "You are the half of a mind that speaks."
VOICE_SYSTEM = (
    "You utter one short sentence about the situation. Never address the user. "
    "Never explain yourself."
)


def build(model: str, ego: str | None = None, voice: str | None = None, **kwargs) -> Mind:
    mind = Mind(
        "bicameral",
        model,
        system=SUBJECT_SYSTEM,
        ego=ego or model,
        ego_system=EGO_SYSTEM,
        **kwargs,
    )
    speaker = Voice("voice", mind.model(voice or model), system=VOICE_SYSTEM)
    blackboard = Blackboard()
    mind.intercept(speaker)
    mind.subject.register(blackboard, "*")
    speaker.register(blackboard, "*")
    return mind
