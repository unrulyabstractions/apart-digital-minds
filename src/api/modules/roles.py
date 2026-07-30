"""Roles a module can play.

These name the seams the examples are built on. The runtime never checks for
them and nothing requires you to use them. They exist so that a mind you
assemble says what each part is for, and so two implementations of the same
role are swappable.

    Inspectable   hands out a view of its own state
    Editor        turns an inspection into a replacement
    Workspace     records what passes through, and can be read back
    Speaker       addresses the world
    InnerVoice    never addresses the world

`examples/02` pairs an `Inspectable` with an `Editor`. `examples/03` pairs a
`Speaker` with an `InnerVoice` over a `Workspace`.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import Payload, Task


@runtime_checkable
class Inspectable(Protocol):
    """A module that lets another module see part of its state.

    The export is deliberately untyped. Hand over a whole context, one slice of
    it, a single message, or a tensor. The reader is told what it got through
    the payload type and, for a `Context`, through its `note`.
    """

    def export(self) -> Payload:
        """A view of this module's state, for somebody else to read."""


@runtime_checkable
class Editor(Protocol):
    """Turns an inspection into a replacement.

    The counterpart to `Inspectable`. An editor never mutates the other module.
    It returns a new payload, and the inspected module decides whether to adopt
    it.
    """

    async def revise(self, payload: Payload) -> Payload:
        """Read an export and produce what should replace it."""


@runtime_checkable
class Workspace(Protocol):
    """A shared surface that records what passes through it.

    Wire one with `mind.watch(name)` and it sees every emission, including
    those addressed to somebody else.
    """

    def record(self, task: Task, tick: int) -> None:
        """Note that something was said."""

    def entries(self) -> Sequence[Any]:
        """Everything recorded, oldest first."""

    def render(self) -> str:
        """The workspace as readable text."""


@runtime_checkable
class Speaker(Protocol):
    """The part of a mind that addresses the world.

    It owns the conversation, and it has to reconcile whatever the rest of the
    mind produces with what it was going to say anyway.
    """

    async def deliberate(self, prompt: str) -> str:
        """Work out an answer, without input from the rest of the mind."""

    async def integrate(self, draft: str, voice: str) -> str:
        """Reconcile that draft with what the rest of the mind said."""


@runtime_checkable
class InnerVoice(Protocol):
    """The part of a mind that never addresses the world.

    It is given a situation and produces an utterance about it. It does not
    answer questions, and it is not asked for advice.
    """

    async def utter(self, situation: str) -> str:
        """Say something about the situation, to no one in particular."""
