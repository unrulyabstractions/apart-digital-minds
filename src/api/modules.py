"""The interface a participant in a mind satisfies.

A module holds a queue and handles one task at a time. It never calls another
module. It emits, and the scheduler delivers on the next tick.

Implemented in `src.dminds.module` by `BaseModule`, which supplies the queue and
routes each task to an `on_<kind>` method. Subclass `BaseModule`, or `Agent` if
you want a model attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Sequence

from .observability import Logger
from .types import Payload, Task

if TYPE_CHECKING:  # pragma: no cover
    from .runtime import Host


class Ctx(Protocol):
    """What a handler is given: where it is, what it can say, how to log.

    Constructed by the scheduler, one per handled task. You implement this only
    if you are writing an alternative runtime.
    """

    tick: int
    task: Task
    module: "Module"
    mind: "Host"
    log: Logger

    def emit(
        self,
        kind: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
        cause: str | None = None,
    ) -> list[Task]:
        """Send a task. It is delivered at the start of the next tick.

        `to` names one or more modules. Leave it out to use the wired routes.
        Use `to="world"` to hand something back to the caller.
        """

    def reply(self, kind: str, payload: Payload) -> list[Task]:
        """Emit straight back to whoever sent the current task."""


class Module(ABC):
    """Something with a name, a queue, and a way to handle a task."""

    name: str

    # -- lifecycle -----------------------------------------------------

    @abstractmethod
    def attach(self, mind: "Host") -> None:
        """Called once when the module joins a mind. Set up here."""

    def close(self) -> None:
        """Called when the mind shuts down. Release resources here."""

    # -- queue ---------------------------------------------------------

    @abstractmethod
    def receive(self, task: Task) -> None:
        """Accept a delivered task. The scheduler calls this, not you."""

    @abstractmethod
    def next_task(self) -> Task | None:
        """Take one task, in arrival order. Return None when empty."""

    @property
    @abstractmethod
    def pending(self) -> int:
        """How many tasks are queued. The scheduler reads this to find work."""

    # -- dispatch ------------------------------------------------------

    @abstractmethod
    async def process(self, task: Task, ctx: Ctx) -> None:
        """Handle exactly one task.

        Called at most once per tick. Emitting is done through `ctx`, never by
        touching another module.
        """


class Handler(Protocol):
    """The shape of an `on_<kind>` method."""

    async def __call__(self, task: Task, ctx: Ctx) -> None: ...


__all__ = ["Ctx", "Module", "Handler"]
