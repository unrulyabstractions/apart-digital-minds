"""The interface a participant in a mind satisfies.

A module declares the channels it can emit on, registers consumers onto those
channels itself, and handles a turn in two steps.

    OUTPUTS      what this module can emit, and what each channel carries
    INPUTS       what it expects to receive, for the reader
    register     attach a consumer to one of my channels
    on_input     something arrived
    on_process   take a step

Wiring lives here, not on the mind. A module owns its output channels, so a
consumer is attached by the module that will do the emitting:

    subject.register(monitor, "thought", as_channel="overheard")

When `subject` emits on `"thought"`, `monitor.on_input` is called with a message
whose channel reads `"overheard"`. Neither module names the other's
vocabulary.

Keep channels atomic: one name, one direction, one meaning. A module that both
consumes and produces the same channel name can be wired into a cycle, and a
cycle in a mind is something a human has to reason about at every tick. The
shipped `Subject` reads `prompt` and writes `context`, so it cannot loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Mapping

from ..types import Link, Message

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime import Host
    from .context import Ctx


class Module(ABC):
    """Something with a name, a queue, declared outputs, and a turn.

    A turn is two steps, run once per tick:

        1. every message that arrived goes through `on_input`, in order
        2. `on_process` runs once

    Splitting them means a module absorbs everything that reached it before it
    acts, so it never decides on half the picture. A module that wants to
    handle a message the instant it lands can still do the work in `on_input`.
    """

    #: Channel name -> what it carries. Emitting on an undeclared channel is
    #: an error, and registering onto one is too, so a typo fails at wiring
    #: time instead of silently dropping messages at run time.
    OUTPUTS: ClassVar[Mapping[str, str]] = {}

    #: Channel name -> what this module expects. Documentation for the reader;
    #: the runtime does not enforce it, because a module may be wired to
    #: anything.
    INPUTS: ClassVar[Mapping[str, str]] = {}

    name: str

    # -- wiring --------------------------------------------------------

    @abstractmethod
    def register(
        self, consumer: "Module", channel: str, as_channel: str | None = None
    ) -> Link:
        """Attach `consumer` to one of my output channels.

        `channel` must be in `OUTPUTS`, or `"*"` to forward everything, which
        is how a monitor or a workspace attaches.
        """

    @abstractmethod
    def links(self) -> list[Link]:
        """Every registration made on this module, in the order they were made."""

    @abstractmethod
    def unregister(self, link: Link) -> bool:
        """Undo one registration. True if it was there."""

    @abstractmethod
    def unregister_all(self) -> None:
        """Drop every consumer of this module. Membership is unaffected."""

    @abstractmethod
    def consumers(self, channel: str) -> list[tuple["Module", str]]:
        """`(consumer, channel_as_heard)` for everything listening to `channel`."""

    # -- lifecycle -----------------------------------------------------

    @abstractmethod
    def attach(self, mind: "Host") -> None:
        """Called once when the module joins a mind. Set up here."""

    def close(self) -> None:
        """Called when the mind shuts down. Release resources here."""

    # -- queue ---------------------------------------------------------

    @abstractmethod
    def receive(self, message: Message) -> None:
        """Accept a delivered message. The scheduler calls this, not you."""

    @abstractmethod
    def drain(self) -> list[Message]:
        """Take everything queued, in arrival order, and empty the queue."""

    @property
    @abstractmethod
    def pending(self) -> int:
        """How many messages are queued. The scheduler reads this to find work."""

    def wants_process(self) -> bool:
        """True if `on_process` should run even with nothing in the queue.

        Default is False, so an idle module costs nothing. Return True for a
        module that acts on its own, such as a voice that speaks unprompted.
        """
        return False

    # -- the turn ------------------------------------------------------

    @abstractmethod
    async def on_input(self, message: Message, ctx: "Ctx") -> None:
        """One message arrived. Called once per message, in arrival order."""

    @abstractmethod
    async def on_process(self, ctx: "Ctx") -> None:
        """Take one step. Called once per tick, after every `on_input`."""
