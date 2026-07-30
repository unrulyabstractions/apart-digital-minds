"""Who hears what."""

from __future__ import annotations

from typing import Protocol

from ..types import Route
from .constants import WILDCARD


class Router(Protocol):
    """The routing table.

    A **route** decides where an unaddressed emission goes and may rename the
    kind on the wire. An **observer** gets a copy of matching traffic however
    it was addressed.

    The two are separate because an explicit `to=` bypasses routes, and a
    monitor still has to see traffic addressed elsewhere.
    """

    routes: list[Route]
    observers: list[Route]

    def wire(
        self, src: str, kind: str, dst: str, as_kind: str | None = None
    ) -> Route:
        """Connect one emitter to one receiver."""

    def observe(
        self,
        dst: str,
        kind: str = WILDCARD,
        src: str = WILDCARD,
        as_kind: str | None = None,
    ) -> Route:
        """Copy matching traffic to `dst`, however it was addressed."""

    def resolve(self, src: str, kind: str) -> list[tuple[str, str]]:
        """Addressed destinations as `(module, kind_as_seen)` pairs."""

    def observers_for(self, src: str, kind: str) -> list[tuple[str, str]]:
        """Destinations that get a copy regardless of address."""

    def describe(self) -> str:
        """The table, for printing."""
