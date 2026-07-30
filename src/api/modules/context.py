"""What a handler is given."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..observability import Logger
from ..types import Message, Payload

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Host
    from .module import Module


class Ctx(Protocol):
    """Where a module is in time, what it can emit, and how it logs.

    Constructed by the scheduler, one per turn. Implement this only if you are
    writing an alternative runtime.
    """

    tick: int
    module: "Module"
    mind: "Host"
    log: Logger

    def emit(self, channel: str, payload: Payload, cause: str | None = None) -> list[Message]:
        """Emit on one of this module's declared channels.

        Goes to whoever registered onto that channel, and to nobody otherwise.
        There is no address here on purpose: a module says what it produced,
        and the registrations decide who hears it.

        Delivered at the start of the next tick, never within this one.
        """
