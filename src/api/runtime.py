"""Interfaces for the runtime itself: routing, the clock, and the host.

These are the three seams you replace when you want the mind to behave
differently at the level of mechanism rather than content.

    Router     who hears what
    Scheduler  what "one step" means
    Host       where modules live and how emissions become deliveries

Implemented in `src.dminds` by `Bus`, `TickScheduler`, and `Mind`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

from .modules import Module
from .observability import Tracer
from .types import Payload, Route, Task

#: The virtual module that stands for the caller, outside the mind.
WORLD = "world"

#: Matches any source or any kind in a route.
WILDCARD = "*"


class Router(Protocol):
    """The routing table.

    A **route** decides where an unaddressed emission goes and may rename the
    kind on the wire. An **observer** gets a copy of matching traffic however
    it was addressed. The two are separate because an explicit `to=` bypasses
    routes, and a monitor still has to see traffic addressed elsewhere.
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


class Host(Protocol):
    """Where modules live. A module reaches it through `ctx.mind`.

    Implementations own the module registry, the router, the tracer, and the
    task-id counter.
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


class Scheduler(ABC):
    """The clock. This is where "one step" is defined.

    The shipped implementation runs every module holding work concurrently, one
    task each, then delivers everything at once. Replace it to get a different
    notion of time, such as priority queues or real-time deadlines.
    """

    #: The tick about to run, or being run.
    t: int

    @abstractmethod
    async def tick(self) -> int:
        """Run one tick. Returns how many tasks were handled."""

    @abstractmethod
    def is_idle(self) -> bool:
        """True when no module has queued work."""

    @abstractmethod
    async def run_until_idle(self, max_ticks: int | None = None) -> int:
        """Tick until every queue is empty. Returns the number of ticks run.

        This is the guarantee behind `Mind.prompt`: an external input is fully
        digested before control returns to the caller.
        """


class RunawayMind(RuntimeError):
    """The mind never went quiet. Usually two modules answering each other."""
