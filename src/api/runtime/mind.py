"""The interface you drive.

`Host` and `Mind` are deliberately two things.

    Host   what a module needs from the assembly it lives in. Narrow: stage an
           emission, and nothing else. A module cannot add modules or drive
           the clock.
    Mind   what you need to build and run one.

Wiring is absent from both. Modules register consumers on each other, so a
mind holds modules and time, and has no opinion about who talks to whom.

A mind is one target model, its **subject**, plus whatever you attach to it.

    mind = Mind("demo", "openai:gpt-5", system="Be terse.")
    mind.subject.register(mind.world, "reply")     # hear what it says
    mind.subject.register(monitor, "context")      # read what it remembers

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

    #: The target model, as a module. The thing a mind is built around, and
    #: the thing an experiment is about. None if the mind was given no model.
    subject: Module | None

    #: The part that speaks, if there is one. Without an ego the subject's own
    #: reply is what reaches you.
    ego: Module | None

    #: The module standing for you, outside the mind. Register onto a channel
    #: with it as the consumer and whatever arrives lands in `outbox`.
    world: Module

    #: Where `prompt` delivers. Defaults to the first module added.
    entry: str | None

    #: Messages that reached `world`, oldest first.
    outbox: list[Message]

    # -- assembly ------------------------------------------------------

    @abstractmethod
    def pipeline(self, *stages: Module) -> list[Module]:
        """Lay out `prompt -> subject -> stages -> ego -> world`.

        The mind owns this because subject, stages, and ego are its own anatomy.
        Each stage takes a `context` and passes a `context` along, so an
        interceptor is just a stage.
        """

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
    def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        channel: str = "prompt",
    ) -> list[Message]:
        """Say something. Delivers immediately and returns, without running.

        Driving a mind is three steps, kept apart on purpose: put something in,
        let it think, read what came out.

            mind.prompt("hello")
            await mind.process()
            replies = mind.get_replies()
        """

    @abstractmethod
    async def process_one(self) -> int:
        """Run exactly one tick. Returns how many modules took a turn.

        Step through an experiment with this, reading state between ticks.
        Zero means nothing had work and the mind is settled.
        """

    @abstractmethod
    async def process(self, max_ticks: int | None = None) -> int:
        """Tick until nothing has work anywhere. Returns the number of ticks."""

    @abstractmethod
    def get_replies(self) -> list[Message]:
        """Everything that reached `world` since you last asked. Reading drains."""

    # -- lifecycle -----------------------------------------------------

    @abstractmethod
    def describe(self) -> str:
        """Modules, their channels, their links, and where the trace goes."""

    @abstractmethod
    def close(self) -> None:
        """Close every module and every sink."""
