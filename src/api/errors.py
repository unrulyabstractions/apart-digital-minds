"""Failures you are meant to catch.

Both of these mean the mind was built wrong rather than that a model
misbehaved, so they are raised loudly instead of logged and swallowed.
"""


class UndeclaredChannel(ValueError):
    """A module emitted on, or was registered against, a channel it never
    declared in `OUTPUTS`.

    Raised at wiring time where possible, so a mistyped channel fails before a
    run instead of dropping messages during one.
    """


class RunawayMind(RuntimeError):
    """The mind never went quiet.

    Usually two modules answering each other, or a `wants_process` that never
    turns False. Carries the outstanding queues so you can see who.
    """
