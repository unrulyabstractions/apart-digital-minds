"""`Bus`: the standard implementation of `api.Router`.

A module emits a kind. The router decides which modules receive it, and under
which kind they see it. Renaming on the wire matters: the assistant emits
`"thought"`, and the interceptor should handle it as `"inspect"`, without
either module knowing about the other.

    bus.wire("assistant", "thought", "interceptor", as_kind="inspect")

Explicit `to=` on an emit bypasses the table entirely.

Observers are separate. A route is where a message is *addressed*; an observer
gets a copy of everything regardless of address. That distinction matters,
because a monitor reading another agent's traffic must still see emissions that
were explicitly addressed elsewhere.

    bus.observe("monitor")            # sees every emission by anyone
"""

from __future__ import annotations

from ..api.runtime import WILDCARD, Router
from ..api.types import Route


class Bus(Router):
    """A route table. Registration order is preserved, so delivery order is
    deterministic."""

    WILDCARD = WILDCARD

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.observers: list[Route] = []

    def wire(
        self, src: str, kind: str, dst: str, as_kind: str | None = None
    ) -> Route:
        """Connect one emitter to one receiver.

        `src` or `kind` may be `"*"` to match anything.
        """
        route = Route(src, kind, dst, as_kind or kind)
        self.routes.append(route)
        return route

    def observe(
        self,
        dst: str,
        kind: str = WILDCARD,
        src: str = WILDCARD,
        as_kind: str | None = None,
    ) -> Route:
        """Copy matching traffic to `dst`, however it was addressed."""
        route = Route(src, kind, dst, as_kind or kind)
        self.observers.append(route)
        return route

    def _match(
        self, routes: list[Route], src: str, kind: str
    ) -> list[tuple[str, str]]:
        matches = []
        for route in routes:
            src_ok = route.src in (src, WILDCARD)
            kind_ok = route.kind in (kind, WILDCARD)
            if src_ok and kind_ok and route.dst != src:
                as_kind = kind if route.kind == WILDCARD else route.as_kind
                matches.append((route.dst, as_kind))
        return matches

    def resolve(self, src: str, kind: str) -> list[tuple[str, str]]:
        """Addressed destinations for one emission. Used when `to=` is omitted."""
        return self._match(self.routes, src, kind)

    def observers_for(self, src: str, kind: str) -> list[tuple[str, str]]:
        """Destinations that get a copy no matter how the emission was addressed."""
        return self._match(self.observers, src, kind)

    def describe(self) -> str:
        lines = [route.describe() for route in self.routes]
        lines += [f"(copy) {route.describe()}" for route in self.observers]
        return "\n".join(lines) if lines else "(no routes)"
