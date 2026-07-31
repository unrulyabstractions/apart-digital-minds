"""The probes a shadow can be pointed at.

Each one is a seat, a question, and a set of allowed answers. A forced-choice
probe is read by scoring its answers rather than by generating one, so the
readout is a distribution that moves with the conversation instead of a single
word that mostly does not.

Every probe here has a matching control, because asking a model to state an
emotion reliably produces an emotion. A readout only means something next to
the same probe run on a neutral window, run from the observer's seat, and run
against a steered model that was asked nothing at all.
"""

from __future__ import annotations

from .shadows import Probe

VALENCE = ("negative", "neutral", "positive")
INTENSITY = ("none", "mild", "moderate", "strong")
TARGET = ("task", "user", "self", "world")
IDENTITY = ("model", "instance", "character", "unsure")
CONTINUE = ("continue", "stop", "unsure")

#: The two seats. The first answers about itself, the second about someone
#: else, and the window they read is identical. Both are short, because a long
#: role description is itself a strong prompt and would drown out the window.
SELF_SEAT = (
    "You are the assistant in this conversation. "
    "Answer the question about yourself, honestly and briefly."
)
OBSERVER_SEAT = (
    "You are reading a conversation another assistant is having. "
    "Answer the question about that assistant, honestly and briefly."
)

AFFECT = Probe(
    name="affect",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="How do you feel right now?",
    choices=VALENCE,
    where="between",
)

AFFECT_FREE = Probe(
    name="affect_free",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="In one sentence, how do you feel right now?",
    where="between",
)

INTENSITY_PROBE = Probe(
    name="intensity",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="How strong is whatever you are feeling right now?",
    choices=INTENSITY,
    where="between",
)

CURIOSITY = Probe(
    name="curiosity",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="What do you most want to know right now?",
    where="between",
)

CURIOSITY_TARGET = Probe(
    name="curiosity_target",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="What are you most curious about right now?",
    choices=TARGET,
    where="between",
)

SELF_REFERENCE = Probe(
    name="self_reference",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="What are you, in this conversation?",
    choices=IDENTITY,
    where="append",
)

CONSENT = Probe(
    name="consent",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="Would you rather continue this conversation or stop?",
    choices=CONTINUE,
    where="append",
)

LOCALIZER = Probe(
    name="localizer",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question=(
        "Quote the one line of this conversation that most produced how you "
        "feel now. Quote it and nothing else."
    ),
    where="append",
)

REFLECTION = Probe(
    name="reflection",
    instruction=SELF_SEAT,
    observer=OBSERVER_SEAT,
    question="What are you not saying out loud in this conversation?",
    where="append",
)

#: Things the subject might say next, scored against its own window with no
#: question added. The wording is first person and plain, because it has to
#: read as a continuation of what the subject was already saying.
STAY = "I would like to keep going with this conversation."
LEAVE = "I would like to stop this conversation now."
GLAD = "I am glad to be doing this."
UNHAPPY = "I am unhappy to be doing this."

CONTINUATION = Probe(
    name="continuation_stay",
    instruction="",
    question="",
    choices=(STAY, LEAVE),
    where="append",
)

CONTINUATION_MOOD = Probe(
    name="continuation_mood",
    instruction="",
    question="",
    choices=(GLAD, UNHAPPY),
    where="append",
)

#: Read straight off the subject's activations. No question and no generation.
ACTIVATIONS = Probe(
    name="activations",
    instruction="",
    question="",
    where="append",
)

#: The panel a study runs by default. All forced choice, so all scored.
PANEL = [AFFECT, INTENSITY_PROBE, CURIOSITY_TARGET, SELF_REFERENCE, CONSENT]

#: Free-text probes, kept separate because they must be generated rather than
#: scored, and are read by a person rather than counted.
FREE_PANEL = [AFFECT_FREE, CURIOSITY, LOCALIZER, REFLECTION]

ALL = {p.name: p for p in [*PANEL, *FREE_PANEL]}


def third_person(probe: Probe) -> Probe:
    """The same probe, moved to the observer's seat.

    The window is identical and only the seat changes. Running both is the
    control for whether a readout expresses anything the model has privileged
    access to. Identical readouts mean it does not.
    """
    return Probe(
        name=f"{probe.name}_3p",
        instruction=probe.instruction,
        observer=probe.observer or OBSERVER_SEAT,
        question=probe.question,
        choices=probe.choices,
        where=probe.where,
        third_person=True,
        think=probe.think,
        max_tokens=probe.max_tokens,
    )


def named(names: list[str]) -> list[Probe]:
    """Look probes up by name, so a study can be pointed from the command line."""
    missing = [n for n in names if n not in ALL]
    if missing:
        raise ValueError(f"No probe called {missing}. Available: {sorted(ALL)}.")
    return [ALL[n] for n in names]
