"""The interface a participant in a mind satisfies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from ..types import Task

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Host
    from .context import Ctx


class Module(ABC):
    """Something with a name, a queue, and a way to handle a task.

    A module holds a queue and handles one task per tick. It never calls
    another module. It emits, and the scheduler delivers on the next tick.

    `BaseModule` in `src.dminds` supplies the queue and routes each task to an
    `on_<kind>` method. Subclass that unless you want none of the machinery.
    """

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
    async def process(self, task: Task, ctx: "Ctx") -> None:
        """Handle exactly one task.

        Called at most once per tick. Emitting is done through `ctx`, never by
        touching another module.
        """


class Handler(Protocol):
    """The shape of an `on_<kind>` method."""

    async def __call__(self, task: Task, ctx: "Ctx") -> None: ...
