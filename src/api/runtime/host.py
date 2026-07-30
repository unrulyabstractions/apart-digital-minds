"""Where modules live."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..modules import Module
from ..observability import Tracer
from ..types import Payload, Task
from .router import Router


class Host(Protocol):
    """The assembly a module belongs to. Reached through `ctx.mind`.

    Implementations own the module registry, the router, the tracer, and the
    task-id counter. `Mind` is the one you get.
    """

    modules: dict[str, Module]
    bus: Router
    tracer: Tracer
    entry: str | None

    def stage(
        self,
        src: str,
        kind: str,
        payload: Payload,
        to: str | Sequence[str] | None,
        cause: str | None,
        outbox: list[Task],
    ) -> list[Task]:
        """Turn one emission into one task per destination.

        Appends to `outbox` rather than delivering. The scheduler delivers at
        the end of the tick, which is what keeps a tick's emissions invisible
        until the next one.
        """

    def deliver(self, task: Task) -> bool:
        """Put a task in its target queue. False if it had nowhere to go."""
