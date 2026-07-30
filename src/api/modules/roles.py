"""Roles a module can play.

These name the seams the examples are built on. The runtime never checks for
them and nothing requires you to use them. They exist so that a mind you
assemble says what each part is for, and so two implementations of the same
role are swappable.

    Editor       turns a context into a replacement context
    Workspace    records what passes through, and can be read back
    InnerVoice   utters something unbidden, never addressing you

`examples/02` uses an `Editor` as a pipeline stage. `examples/03` uses an
`InnerVoice` as one, watched by a `Workspace`.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import Message, Payload


@runtime_checkable
class Editor(Protocol):
    """Turns an inspection into a replacement.

    An editor never mutates the module it edits for.
    It returns a new payload, and the inspected module decides whether to adopt
    it.
    """

    async def revise(self, payload: Payload) -> Payload:
        """Read an export and produce what should replace it."""


@runtime_checkable
class Workspace(Protocol):
    """A shared surface that records what passes through it.

    Register one onto `"*"` of every module you want it to observe, and it
    sees each of their emissions.
    """

    def record(self, message: Message, tick: int) -> None:
        """Note that something was said."""

    def entries(self) -> Sequence[Any]:
        """Everything recorded, oldest first."""

    def render(self) -> str:
        """The workspace as readable text."""


@runtime_checkable
class InnerVoice(Protocol):
    """The part of a mind that never addresses the world.

    It is given a situation and produces an utterance about it. It does not
    answer questions, and it is not asked for advice.
    """

    async def utter(self, situation: str) -> str:
        """Say something about the situation, to no one in particular."""
