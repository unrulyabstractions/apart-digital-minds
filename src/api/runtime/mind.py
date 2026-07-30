"""The interface you drive.

`Host` and `Mind` are deliberately two things.

    Host   what a module needs from the assembly it lives in. Narrow on
           purpose: a module can stage an emission and nothing else. It
           cannot add modules, rewire the graph, or drive the clock.
    Mind   what you need to build and run one. Everything in Host, plus
           assembly and driving.

A module that only ever sees `Host` cannot reach around the scheduler, which is
what keeps the tick discipline enforceable rather than merely advised.

Implemented in `src.dminds.mind` by `Mind`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..modules import Module
from ..types import Payload, Route, Task
from .constants import WILDCARD
from .host import Host
from .scheduler import Scheduler


class Mind(Host):
    """Assemble modules, wire them, drive the clock, read what came out."""

    name: str
    run_id: str
    scheduler: Scheduler

    #: Tasks addressed to `world`, oldest first.
    outbox: list[Task]

    # -- assembly ------------------------------------------------------

    @abstractmethod
    def add(self, *modules: Module) -> Module:
        """Register modules. Returns the last one, so you can inline it.

        The first module added becomes `entry` unless you set it yourself.
        """

    @abstractmethod
    def wire(
        self, src: str, kind: str, dst: str, as_kind: str | None = None
    ) -> Route:
        """Connect an emitter to a receiver, optionally renaming the kind."""

    @abstractmethod
    def watch(
        self, dst: str, kind: str = WILDCARD, src: str = WILDCARD
    ) -> Route:
        """Give one module a copy of matching traffic, however it was addressed."""

    @abstractmethod
    def validate(self) -> list[str]:
        """Every problem with the assembly, as readable lines.

        Routes are named by string, so a typo would otherwise fail silently by
        dropping messages. Implementations run this before the first tick.
        """

    # -- models --------------------------------------------------------

    @abstractmethod
    def model(self, spec: str, **kwargs) -> "object":
        """Build a model through this mind's factory.

        Going through the mind rather than calling `get_llm` directly is what
        lets one argument tape, throttle, or redirect every model in the run.
        """

    # -- driving -------------------------------------------------------

    @abstractmethod
    def send(
        self,
        kind: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
        src: str = "world",
    ) -> list[Task]:
        """Inject an external input. Delivered immediately, not next tick."""

    @abstractmethod
    async def run(self, max_ticks: int | None = None) -> int:
        """Tick until quiet. Returns the number of ticks run."""

    @abstractmethod
    async def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        kind: str = "user_prompt",
        max_ticks: int | None = None,
    ) -> list[Task]:
        """Say something, wait for the mind to settle, take what it produced.

        Returns only the tasks addressed to `world` during this prompt.
        """

    # -- lifecycle -----------------------------------------------------

    @abstractmethod
    def describe(self) -> str:
        """Modules, routes, and where the trace is going."""

    @abstractmethod
    def close(self) -> None:
        """Close every module and every sink."""
