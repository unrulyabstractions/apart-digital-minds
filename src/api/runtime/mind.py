"""The interface you drive.

`Host` and `Mind` are deliberately two things.

    Host   what a module needs from the assembly it lives in. Narrow: stage an
           emission, and nothing else. A module cannot add modules or drive
           the clock.
    Mind   what you need to build and run one.

Wiring is absent from both. Modules register consumers on each other, so a
mind holds modules and time, and has no opinion about who talks to whom.

    mind = Mind("demo")
    a, b = Agent("a", ...), Agent("b", ...)
    a.register(mind.world, "reply")     # a joins, because world is already here
    a.register(b, "draft")              # b joins, because a is here now

There is no separate step for adding a module. Wiring it in is adding it.

Implemented in `src.dminds.mind` by `Mind`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..modules import Module
from ..types import Message, Payload
from .host import Host
from .scheduler import Scheduler


class Mind(Host):
    """Hold modules, drive the clock, read what came out."""

    name: str
    run_id: str
    scheduler: Scheduler

    #: The module standing for you, outside the mind. Register onto a channel
    #: with it as the consumer and whatever arrives lands in `outbox`.
    world: Module

    #: Where `prompt` delivers. Defaults to the first module added.
    entry: str | None

    #: Messages that reached `world`, oldest first.
    outbox: list[Message]

    # -- assembly ------------------------------------------------------

    @abstractmethod
    def adopt(self, module: Module) -> Module:
        """Take one module into this mind. Called by `Module.register`.

        Registering a module against something already here brings it in, so
        wiring and populating are one act and `register` is the only verb you
        normally need.
        """

    @abstractmethod
    def add(self, *modules: Module) -> Module:
        """Take modules in explicitly. Only needed for a module wired to
        nothing, such as one that runs on `wants_process` alone."""

    @abstractmethod
    def validate(self) -> list[str]:
        """Every problem with the assembly, as readable lines.

        Catches links pointing at modules that were never added, and an entry
        that names nothing. Implementations run this before the first tick.
        """

    @abstractmethod
    def links(self) -> list:
        """Every registration in the mind, gathered from its modules."""

    # -- models --------------------------------------------------------

    @abstractmethod
    def model(self, spec: str, **kwargs) -> object:
        """Build a model through this mind's factory.

        Going through the mind rather than calling `get_llm` directly is what
        lets one argument tape, throttle, or redirect every model in the run.
        """

    # -- driving -------------------------------------------------------

    @abstractmethod
    def send(
        self,
        channel: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
    ) -> list[Message]:
        """Inject an external input. Delivered immediately, not next tick."""

    @abstractmethod
    async def run(self, max_ticks: int | None = None) -> int:
        """Tick until quiet. Returns the number of ticks run."""

    @abstractmethod
    async def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        channel: str = "user_prompt",
        max_ticks: int | None = None,
    ) -> list[Message]:
        """Say something, wait for the mind to settle, take what it produced.

        Returns only the messages that reached `world` during this prompt.
        """

    # -- lifecycle -----------------------------------------------------

    @abstractmethod
    def describe(self) -> str:
        """Modules, their channels, their links, and where the trace goes."""

    @abstractmethod
    def close(self) -> None:
        """Close every module and every sink."""
