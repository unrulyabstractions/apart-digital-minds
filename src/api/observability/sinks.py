"""Where events go."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Event


@runtime_checkable
class Sink(Protocol):
    """A destination for events. Implement these two methods and attach it.

        mind.tracer.add_sink(MySink())

    Shipped implementations write JSONL, split per module, print to a stream,
    or keep events in a list.
    """

    def write(self, event: Event) -> None:
        """Record one event. Called for every event, in order."""

    def close(self) -> None:
        """Flush and release. Called once when the mind shuts down."""
