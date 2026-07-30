"""Internal: the mind as the runtime sees it.

`api.Mind` is what you drive. This is the narrower view the scheduler and the
module machinery need, plus the factories a mind is built from. None of it is
part of the external surface, which is why it lives here and not in `src.api`.

A module receives a `Host` through `ctx.mind`. It can stage an emission and
look up who else is here. It cannot add modules, relay out the pipeline, or
drive the clock, and that is what keeps the tick discipline enforceable rather
than merely advised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..api.modules import Module
from ..api.observability import Tracer
from ..api.types import Message, Payload

if TYPE_CHECKING:  # pragma: no cover
    from .scheduler import Scheduler


class Host(Protocol):
    """The assembly a module belongs to. Reached through `ctx.mind`.

    Routing is absent. A host does not decide where a message goes; the
    emitting module's own registrations do. The host only turns one emission
    into messages and delivers them when the tick ends.
    """

    modules: dict[str, Module]
    tracer: Tracer

    def stage(
        self,
        src: Module,
        channel: str,
        payload: Payload,
        cause: str | None,
        outbox: list[Message],
    ) -> list[Message]:
        """Turn one emission into one message per registered consumer.

        Appends to `outbox` rather than delivering. The scheduler delivers at
        the end of the tick, which is what keeps a tick's emissions invisible
        until the next one.
        """

    def deliver(self, message: Message) -> bool:
        """Put a message in its target queue. False if it had nowhere to go."""


class SchedulerFactory(Protocol):
    """Builds a scheduler bound to the host it will drive.

    A scheduler needs its host, so it arrives as a factory rather than as a
    finished object.
    """

    def __call__(self, host: Host) -> "Scheduler": ...
