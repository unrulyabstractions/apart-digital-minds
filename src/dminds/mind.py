"""`Mind`: the standard implementation of `api.Mind`.

It holds modules and time. It does not wire them, and it has no routing table.
A module registers consumers onto its own channels, so the mind never decides
who hears what.

    mind = Mind("demo")
    assistant = mind.add(Agent("assistant", mind.model("echo:")))
    assistant.register(mind.world, "reply")
    replies = await mind.prompt("hello")

`prompt` injects one external input and runs until every queue is empty, so
when it returns the mind has finished thinking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from ..api.models import LLM
from ..api.modules import Module
from ..api.observability import TASK_DELIVER, TASK_EMIT, Sink, Tracer
from ..api.runtime import WORLD, ModelFactory, Scheduler, SchedulerFactory
from ..api.runtime import Mind as MindInterface
from ..api.types import Link, Message, Payload, Text
from .llm import get_llm
from .module import BaseModule
from .scheduler import TickScheduler
from .trace import ConsoleSink, JsonlSink, MemorySink, PerModuleSink, RunTracer


def _fresh_run_id(run_dir: str | Path | None) -> str:
    """A run id that never collides with an existing run directory.

    Two experiments started in the same second must not write into one trace
    file, so the id carries milliseconds and is checked against the directory.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    if run_dir is None:
        return stamp
    base = Path(run_dir)
    candidate, n = stamp, 1
    while (base / candidate).exists():
        candidate = f"{stamp}-{n}"
        n += 1
    return candidate


class World(BaseModule):
    """You, from the mind's point of view.

    A sink. Register it onto any channel and whatever is emitted there lands
    in `mind.outbox`, which is what `prompt` returns.

        assistant.register(mind.world, "reply")
    """

    INPUTS = {"*": "anything a module wants to hand back to the caller"}

    def __init__(self, outbox: list[Message], name: str = WORLD):
        super().__init__(name)
        self._outbox = outbox

    async def on_input(self, message: Message, ctx) -> None:
        self._outbox.append(message)


class Mind(MindInterface):
    """The default composition, with every part replaceable.

    The scheduler, the tracer, and the model factory are arguments. Left alone
    they are `TickScheduler`, `RunTracer`, and `get_llm`.

        Mind("fast",  scheduler=lambda host: MyScheduler(host))
        Mind("taped", model_factory=taped("runs/tape.jsonl"))
    """

    def __init__(
        self,
        name: str = "mind",
        run_id: str | None = None,
        run_dir: str | Path | None = "runs",
        console: bool = True,
        keep_events: bool = True,
        strict: bool = True,
        max_ticks: int = 200,
        sinks: Sequence[Sink] | None = None,
        scheduler: SchedulerFactory | None = None,
        tracer: Tracer | None = None,
        model_factory: ModelFactory | None = None,
    ):
        self.name = name
        self.run_id = run_id or _fresh_run_id(run_dir)
        self.modules: dict[str, Module] = {}
        self.outbox: list[Message] = []
        self.model_factory: ModelFactory = (
            model_factory if model_factory is not None else get_llm
        )
        self._message_counter = 0
        self._validated = False

        #: Where `prompt` delivers, unless you say otherwise. Defaults to the
        #: first module added. Set it directly to change the front door.
        self.entry: str | None = None

        self.tracer: Tracer = tracer if tracer is not None else RunTracer(self.run_id)
        self.events: MemorySink | None = None
        self.run_path: Path | None = None

        if keep_events:
            self.events = MemorySink()
            self.tracer.add_sink(self.events)
        if run_dir is not None:
            self.run_path = Path(run_dir) / self.run_id
            self.tracer.add_sink(JsonlSink(self.run_path / "trace.jsonl"))
            self.tracer.add_sink(PerModuleSink(self.run_path / "modules"))
        if console:
            self.tracer.add_sink(ConsoleSink())
        for sink in sinks or []:
            self.tracer.add_sink(sink)

        #: The caller, as a module. Never scheduled; it only receives.
        self.world = World(self.outbox)
        self.world.attach(self)

        self.scheduler: Scheduler = (
            scheduler(self)
            if scheduler is not None
            else TickScheduler(self, strict=strict, max_ticks=max_ticks)
        )

    # -- models ----------------------------------------------------------

    def model(self, spec: str, **kwargs) -> LLM:
        """Build a model through this mind's factory.

        Going through the mind rather than calling `get_llm` directly is what
        lets one constructor argument tape, throttle, or redirect every model
        in the run.
        """
        return self.model_factory(spec, **kwargs)

    # -- assembly ------------------------------------------------------

    def adopt(self, module: Module) -> Module:
        """Take a module into this mind. Called by `Module.register`.

        You rarely call this. Wiring a module to something already here brings
        it in, so `register` is normally the only verb you need.
        """
        if module is self.world:
            return module
        if module.name in self.modules:
            if self.modules[module.name] is module:
                return module
            raise ValueError(f"A different module is already named {module.name!r}.")
        if module.name == WORLD:
            raise ValueError(
                f"{WORLD!r} is reserved. Use `mind.world` to hear a channel."
            )
        self.modules[module.name] = module
        module.attach(self)
        if self.entry is None:
            self.entry = module.name
        return module

    def add(self, *modules: Module) -> Module:
        """Take modules in explicitly. Returns the last one, so you can inline it.

        Only needed for a module that is wired to nothing, such as one that
        runs on `wants_process` alone. Everything else joins by being
        registered.
        """
        last = None
        for module in modules:
            last = self.adopt(module)
        if last is None:
            raise ValueError("add() needs at least one module.")
        return last

    def links(self) -> list[Link]:
        """Every registration in the mind, gathered from its modules."""
        return [link for module in self.modules.values() for link in module.links()]

    def validate(self) -> list[str]:
        """Every problem with the assembly, as readable lines.

        Registrations name modules by object, so a bad name is impossible, but
        a consumer can still have been left out of the mind. `run` calls this
        before the first tick and refuses to start if anything comes back.
        """
        problems = []
        known = set(self.modules) | {WORLD}

        for module in self.modules.values():
            for link in module.links():
                if link.dst not in known:
                    listed = ", ".join(sorted(self.modules)) or "none"
                    problems.append(
                        f"link {link.describe()!r} sends to {link.dst!r}, which was "
                        f"never added to this mind. Added modules: {listed}."
                    )

        if self.entry is not None and self.entry not in self.modules:
            problems.append(
                f"entry is {self.entry!r}, which is not a module. "
                f"Added modules: {', '.join(sorted(self.modules)) or 'none'}."
            )
        return problems

    def _check_once(self) -> None:
        if self._validated:
            return
        self._validated = True
        problems = self.validate()
        if problems:
            raise ValueError("This mind is wired wrong:\n  " + "\n  ".join(problems))

    # -- message plumbing -------------------------------------------------

    def _next_id(self) -> str:
        self._message_counter += 1
        return f"M{self._message_counter:04d}"

    def stage(
        self,
        src: Module,
        channel: str,
        payload: Payload,
        cause: str | None,
        outbox: list[Message],
    ) -> list[Message]:
        """Turn one emission into one message per registered consumer.

        Called by `Ctx.emit`. Nothing is delivered here; the scheduler does
        that at the end of the tick.
        """
        if channel not in src.OUTPUTS:
            declared = ", ".join(sorted(src.OUTPUTS)) or "none"
            raise ValueError(
                f"{type(src).__name__} {src.name!r} emitted on {channel!r}, which is "
                f"not one of its output channels. Declared: {declared}. "
                f"Add it to OUTPUTS."
            )

        targets = src.consumers(channel)
        if not targets:
            self.tracer.emit(
                src.name,
                TASK_EMIT,
                {
                    "channel": channel,
                    "summary": f"{channel} has no listeners, dropped",
                    "dropped": True,
                },
                cause=cause,
            )
            return []

        messages = []
        for consumer, as_channel in targets:
            message = Message(
                id=self._next_id(),
                channel=as_channel,
                payload=payload,
                src=src.name,
                dst=consumer.name,
                t_created=self.scheduler.t,
                t_deliver=self.scheduler.t + 1,
                cause=cause,
            )
            outbox.append(message)
            messages.append(message)
            self.tracer.emit(
                src.name,
                TASK_EMIT,
                {"channel": as_channel, "dst": consumer.name,
                 "summary": message.describe()},
                task_id=message.id,
                cause=cause,
            )
        return messages

    def deliver(self, message: Message) -> bool:
        """Put a message in its target queue. False if it had nowhere to go."""
        target = self.world if message.dst == WORLD else self.modules.get(message.dst)
        if target is None:
            self.tracer.emit(
                "runtime",
                TASK_DELIVER,
                {
                    "summary": f"no module named {message.dst!r}, dropped "
                    f"{message.channel}",
                    "dropped": True,
                },
                task_id=message.id,
            )
            return False

        if target is self.world:
            # World is never scheduled, so it absorbs on delivery.
            self.outbox.append(message)
        else:
            target.receive(message)

        self.tracer.emit(
            message.dst,
            TASK_DELIVER,
            {"channel": message.channel, "src": message.src,
             "summary": message.describe()},
            task_id=message.id,
            cause=message.cause,
        )
        return True

    # -- driving ------------------------------------------------------

    def send(
        self,
        channel: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
    ) -> list[Message]:
        """Inject an external input. Delivered immediately, not next tick."""
        names = [to] if isinstance(to, str) else list(to or ([self.entry] if self.entry else []))
        messages = []
        for name in names:
            message = Message(
                id=self._next_id(),
                channel=channel,
                payload=payload,
                src=WORLD,
                dst=name,
                t_created=self.scheduler.t,
                t_deliver=self.scheduler.t,
            )
            self.tracer.emit(
                WORLD,
                TASK_EMIT,
                {"channel": channel, "dst": name, "summary": message.describe()},
                task_id=message.id,
            )
            if self.deliver(message):
                messages.append(message)
        return messages

    async def run(self, max_ticks: int | None = None) -> int:
        """Tick until quiet. Returns the number of ticks run.

        Validates the assembly before the first tick, so a consumer that was
        never added fails loudly instead of silently dropping messages.
        """
        self._check_once()
        return await self.scheduler.run_until_idle(max_ticks)

    async def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        channel: str = "user_prompt",
        max_ticks: int | None = None,
    ) -> list[Message]:
        """Say something, wait for the mind to settle, take what it produced.

        Delivers to `to`, or to `self.entry` when you do not say. Returns only
        the messages that reached `world` during this prompt.
        """
        mark = len(self.outbox)
        self.send(channel, Text(text), to=to)
        await self.run(max_ticks)
        return self.outbox[mark:]

    # -- inspection --------------------------------------------------------

    def describe(self) -> str:
        lines = [f"Mind {self.name!r} (run {self.run_id})", f"  entry: {self.entry}"]
        lines.append("  modules:")
        for module in self.modules.values():
            channels = ", ".join(sorted(module.OUTPUTS)) or "no outputs"
            lines.append(f"    {module.name:<16} {type(module).__name__:<14} {channels}")
        links = self.links()
        lines.append("  links:")
        for link in links:
            lines.append(f"    {link.describe()}")
        if not links:
            lines.append("    (nothing registered)")
        if self.run_path:
            lines.append(f"  trace: {self.run_path}/trace.jsonl")
        return "\n".join(lines)

    def close(self) -> None:
        for module in self.modules.values():
            module.close()
        self.tracer.close()

    def __enter__(self) -> "Mind":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def texts(messages: Iterable[Message]) -> list[str]:
    """Pull the plain text out of a batch of messages, skipping anything else."""
    out = []
    for message in messages:
        payload = message.payload
        if isinstance(payload, Text):
            out.append(payload.text)
        elif isinstance(payload, str):
            out.append(payload)
    return out
