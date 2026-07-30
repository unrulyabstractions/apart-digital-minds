"""`BaseModule`: the standard implementation of `api.Module`.

Declare what you emit, register who hears it, and write a turn.

    class Critic(BaseModule):
        OUTPUTS = {"verdict": "whether the draft holds up"}
        INPUTS = {"draft": "something to judge"}

        async def on_input(self, message, ctx):
            self.inputs.append(message)

        async def on_process(self, ctx):
            for message in self.take_inputs():
                ctx.emit("verdict", judge(message.payload))

    writer.register(critic, "draft")
    critic.register(writer, "verdict")

The default `on_input` routes to `on_<channel>` when you have written such a
method, and buffers into `self.inputs` when you have not. So a module can react
per channel or absorb everything and act once. `Ctx` here is the concrete
object handed to a turn; it satisfies `api.Ctx`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

from ..api.modules import Ctx as CtxProtocol
from ..api.modules import Module
from ..api.observability import Logger
from ..api.constants import WILDCARD
from ..api.errors import UndeclaredChannel
from ..api.types import Link, Message, Payload

if TYPE_CHECKING:  # pragma: no cover
    from .host import Host


def handler_name(channel: str) -> str:
    """`"inspect.context"` -> `"on_inspect_context"`."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in channel)
    return f"on_{safe}"


@dataclass
class Ctx(CtxProtocol):
    """Where a module is in time, what it can emit, and how it logs."""

    tick: int
    module: Module
    mind: "Host"
    log: Logger
    outbox: list[Message] = field(default_factory=list)

    def emit(
        self, channel: str, payload: Payload, cause: str | None = None
    ) -> list[Message]:
        """Emit on one of this module's declared channels.

        Goes to whoever registered onto it. Delivered at the start of the next
        tick, never within this one.
        """
        return self.mind.stage(
            src=self.module,
            channel=channel,
            payload=payload,
            cause=cause,
            outbox=self.outbox,
        )


class BaseModule(Module):
    """A queue, declared output channels, and a two-step turn."""

    OUTPUTS: ClassVar[Mapping[str, str]] = {}
    INPUTS: ClassVar[Mapping[str, str]] = {}

    def __init__(self, name: str):
        self.name = name
        self.inbox: deque[Message] = deque()
        #: Messages the default `on_input` buffered, waiting for `on_process`.
        self.inputs: list[Message] = []
        self._links: list[Link] = []
        self._consumers: dict[str, list[tuple[Module, str]]] = {}
        self.mind: "Host | None" = None
        self.log: Logger | None = None
        self.turns = 0

    # -- wiring --------------------------------------------------------

    def register(
        self, consumer: Module, channel: str, as_channel: str | None = None
    ) -> Link:
        """Attach `consumer` to one of my output channels.

        `channel` must be declared in `OUTPUTS`, or be `"*"` to forward
        everything I emit. Declaring outputs is what turns a typo here into an
        error rather than a message that silently goes nowhere.

        Registering also settles membership. Whichever of the two modules
        already belongs to a mind pulls the other one in, so wiring a mind and
        populating it are the same act. There is nothing else to call.
        """
        if channel != WILDCARD and channel not in self.OUTPUTS:
            declared = ", ".join(sorted(self.OUTPUTS)) or "none"
            raise UndeclaredChannel(
                f"{type(self).__name__} {self.name!r} has no output channel "
                f"{channel!r}. Declared channels: {declared}. Add it to OUTPUTS, "
                f'or register on "*" to receive everything.'
            )

        host = self.mind or getattr(consumer, "mind", None)
        if host is None:
            raise ValueError(
                f"Neither {self.name!r} nor {consumer.name!r} belongs to a mind yet, "
                f"so there is nothing to join. Wire one of them to `mind.world` "
                f"first, or call `mind.add(...)` for a module that emits to nobody."
            )
        for module in (self, consumer):
            if module.mind is None:
                host.adopt(module)

        link = Link(self.name, channel, consumer.name, as_channel or channel)
        self._links.append(link)
        self._consumers.setdefault(channel, []).append((consumer, link.as_channel))
        return link

    def links(self) -> list[Link]:
        return list(self._links)

    def unregister(self, link: Link) -> bool:
        """Undo one registration. True if it was there.

        Removing exactly what you put down matters: a caller that tore down
        every link to relay out a pipeline would also destroy the monitor
        somebody else attached.
        """
        if link not in self._links:
            return False
        self._links.remove(link)
        listeners = self._consumers.get(link.channel, [])
        for i, (consumer, as_channel) in enumerate(listeners):
            if consumer.name == link.dst and as_channel == link.as_channel:
                listeners.pop(i)
                break
        if not listeners:
            self._consumers.pop(link.channel, None)
        return True

    def unregister_all(self) -> None:
        """Drop every consumer of this module. Membership is unaffected."""
        self._links.clear()
        self._consumers.clear()

    def consumers(self, channel: str) -> list[tuple[Module, str]]:
        """Everyone listening to `channel`, plus everyone listening to `"*"`.

        Registration order is preserved, and a consumer registered twice for
        one emission is only told once.
        """
        found: list[tuple[Module, str]] = []
        seen: set[str] = set()
        for consumer, as_channel in self._consumers.get(channel, []):
            if consumer.name not in seen:
                seen.add(consumer.name)
                found.append((consumer, as_channel))
        for consumer, as_channel in self._consumers.get(WILDCARD, []):
            if consumer.name not in seen:
                seen.add(consumer.name)
                # A wildcard listener hears the real channel name.
                found.append((consumer, channel))
        return found

    def declares(self, channel: str) -> bool:
        return channel in self.OUTPUTS

    # -- lifecycle -----------------------------------------------------

    def attach(self, mind: "Host") -> None:
        """Called once when the module joins a mind. Override to set up."""
        self.mind = mind
        self.log = mind.tracer.bind(self.name)

    def close(self) -> None:
        """Called when the mind shuts down. Override to release resources."""

    # -- queue ---------------------------------------------------------

    def receive(self, message: Message) -> None:
        self.inbox.append(message)

    @property
    def pending(self) -> int:
        return len(self.inbox)

    def drain(self) -> list[Message]:
        """Take everything queued, in arrival order, and empty the queue."""
        taken = list(self.inbox)
        self.inbox.clear()
        return taken

    def take_inputs(self) -> list[Message]:
        """Take what `on_input` buffered, and clear the buffer.

        The usual first line of `on_process`.
        """
        taken = self.inputs
        self.inputs = []
        return taken

    def wants_process(self) -> bool:
        """True if `on_process` should run with nothing queued.

        False by default, so an idle module costs nothing. Return True for a
        module that acts on its own.
        """
        return False

    # -- the turn ------------------------------------------------------

    def find_handler(self, channel: str) -> Callable[..., Any] | None:
        """The `on_<channel>` method, if you wrote one."""
        return getattr(self, handler_name(channel), None)

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        """One message arrived.

        Routes to `on_<channel>` when such a method exists. Otherwise buffers
        into `self.inputs` for `on_process` to pick up.
        """
        handler = self.find_handler(message.channel)
        if handler is not None:
            await handler(message, ctx)
        else:
            self.inputs.append(message)

    async def on_process(self, ctx: Ctx) -> None:
        """Take one step. Does nothing unless you override it."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} pending={self.pending}>"
