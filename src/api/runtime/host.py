"""Where modules live, as a module sees it."""

from __future__ import annotations

from typing import Protocol

from ..modules import Module
from ..observability import Tracer
from ..types import Message, Payload


class Host(Protocol):
    """The assembly a module belongs to. Reached through `ctx.mind`.

    Narrow on purpose. A module can stage an emission, look up who else is
    here, and log. It cannot register links on other modules' behalf, add
    modules, or drive the clock.

    Note that routing is absent. A host does not decide where a message goes;
    the emitting module's own registrations do. The host only turns one
    emission into messages and delivers them when the tick ends.
    """

    modules: dict[str, Module]
    tracer: Tracer

    def stage(
        self,
        src: "Module",
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
