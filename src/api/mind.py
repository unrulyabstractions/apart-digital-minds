"""What you build and drive.

A mind is one target model, its **subject**, plus whatever you attach to it.

    mind = Mind("demo", "openai:gpt-5", system="Be terse.")
    mind.prompt("hello")
    await mind.process()
    print(texts(mind.get_replies()))

The subject is the front door in both directions: `prompt` reaches it because
it is the entry, and its `reply` reaches you because the mind connected it. Put
a module in between to preprocess either side.

This is the whole external surface of a mind. The machinery underneath, the
scheduler and the narrow view a module gets of its host, lives in `src.dminds`
because you do not call it.

Implemented in `src.dminds.mind` by `Mind`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .models import LLM
from .modules import Module
from .observability import Tracer
from .types import Link, Message


class Mind(ABC):
    """Hold a subject, attach parts to it, drive the clock, read what came out."""

    name: str
    run_id: str
    tracer: Tracer

    #: Every module in this mind, by name.
    modules: dict[str, Module]

    #: The target model, as a module. The thing an experiment is about. None
    #: if the mind was given no model.
    subject: Module | None

    #: The part that speaks, if there is one. Without an ego the subject's own
    #: reply is what reaches you.
    ego: Module | None

    #: The module standing for you, outside the mind. Register onto a channel
    #: with it as the consumer and whatever arrives lands in `outbox`.
    world: Module

    #: Where `prompt` delivers. Defaults to the subject.
    entry: str | None

    #: Messages that reached `world`, oldest first.
    outbox: list[Message]

    # -- assembly ------------------------------------------------------

    @abstractmethod
    def intercept(self, *stages: Module) -> list[Module]:
        """Put `stages` between the subject and whoever speaks.

        A stage reads `context` and writes `revision`; no module consumes what
        it produces. Shorthand for `register` calls and nothing else. With one
        stage and an ego it is exactly:

            mind.subject.register(stage, "context")
            stage.register(mind.ego, "revision", as_channel="context")
            mind.ego.register(mind.world, "reply")

        Write those yourself when the shape is not a line.
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
    def links(self) -> list[Link]:
        """Every registration in the mind, gathered from its modules."""

    @abstractmethod
    def validate(self) -> list[str]:
        """Every problem with the assembly, as readable lines.

        Implementations run this before the first tick and refuse to start if
        anything comes back.
        """

    # -- models --------------------------------------------------------

    @abstractmethod
    def model(self, spec: str, **kwargs) -> LLM:
        """Build a model through this mind's factory.

        Going through the mind rather than calling `get_llm` directly is what
        lets one argument tape, throttle, or redirect every model in the run.
        """

    # -- driving -------------------------------------------------------

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
