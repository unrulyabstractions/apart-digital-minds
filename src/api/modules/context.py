"""What a handler is given."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from ..observability import Logger
from ..types import Payload, Task

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Host
    from .module import Module


class Ctx(Protocol):
    """Where a handler is, what it can say, and how it logs.

    Constructed by the scheduler, one per handled task. Implement this only if
    you are writing an alternative runtime.
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
