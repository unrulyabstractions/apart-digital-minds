"""`Mind`: the standard implementation of `api.Mind`.

A mind is one target model, called its **subject**, plus whatever you attach to
it. It holds modules and time. It does not wire them and has no routing table,
because a module registers consumers onto its own channels.

    mind = Mind("demo", "openai:gpt-5", system="Be terse.")
    mind.prompt("hello")
    await mind.process()
    print(texts(mind.get_replies()))

The subject is wired to you in both directions already: `prompt` reaches it
because it is the entry, and its `reply` reaches you because the mind
connected it. Put a module in between to preprocess either side.

`mind.subject` publishes `context`, `reply`, and `thought`. Register anything you
like onto those, and register back onto its `context` to rewrite what it
remembers.

Driving is three calls: `prompt` delivers, `process` ticks until nothing has
work, `get_replies` drains what reached you.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..api.models import LLM
from ..api.modules import Module
from ..api.observability import TASK_DELIVER, TASK_EMIT, Tracer
from ..api.constants import WORLD
from ..api.mind import Mind as MindInterface
from ..api.models import ModelFactory
from .host import SchedulerFactory
from .scheduler import Scheduler, TickScheduler
from ..api.types import Link, Message, Payload, Text
from .llm import get_llm
from .module import BaseModule
from .subject import Ego, Subject
from .trace import ConsoleSink, JsonlSink, MemorySink, PerModuleSink, RunTracer


def _build_subject(spec, llm: LLM, system: str | None) -> Module:
    """The module at the centre: the default, a subclass, or a factory."""
    if spec is None:
        return Subject("subject", llm, system=system)
    if isinstance(spec, type):
        return spec("subject", llm, system=system)
    return spec(llm)


def _build_ego(spec, system: str | None, model: Callable[[str], LLM]) -> Module:
    """The module that speaks. A finished module, or a model to build one from."""
    if isinstance(spec, Module):
        return spec
    llm = spec if isinstance(spec, LLM) else model(spec)
    return Ego("ego", llm, system=system)


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

    A sink, and the one module that is never scheduled. Register it as the
    consumer of any channel and whatever is emitted there lands in
    `mind.outbox`, which is what `prompt` returns.

        assistant.register(mind.world, "reply")

    It absorbs on delivery rather than on a turn, so it overrides `receive`
    instead of `on_input`. Waiting for a turn would cost a tick and would make
    the mind look like it was still thinking when it had already answered.
    """

    INPUTS = {"*": "anything a module wants to hand back to the caller"}
    OUTPUTS: dict[str, str] = {}

    def __init__(self, outbox: list[Message], name: str = WORLD):
        super().__init__(name)
        self.outbox = outbox

    def receive(self, message: Message) -> None:
        """Take the message straight out of the mind. Never queued."""
        self.outbox.append(message)


class Mind(MindInterface):
    """The default composition, with every part replaceable.

    Args:
        model: the target model, as a spec string or a built `LLM`. The mind
            wraps it in a subject and holds it as `mind.subject`. Leave it out for a
            mind that is only a graph of modules.
        system: the subject's system prompt.
        subject: something other than the default `Subject` at the centre. A
            subclass, or a callable taking the `LLM` and returning a module.
        autowire: connect the subject's `reply` channel to `world`, so a mind
            answers you without any wiring at all. Turn it off to put a filter
            in between.

    The scheduler, the tracer, and the model factory are arguments too. Left
    alone they are `TickScheduler`, `RunTracer`, and `get_llm`.

        Mind("study", "openai:gpt-5", system="Think first.")
        Mind("halves", "ollama:qwen3:8b", subject=Outer)
        Mind("taped", "echo:", model_factory=taped("runs/tape.jsonl"))
    """

    def __init__(
        self,
        name: str = "mind",
        model: str | LLM | None = None,
        system: str | None = None,
        subject: type[Module] | Callable[[LLM], Module] | None = None,
        ego: str | LLM | Module | None = None,
        ego_system: str | None = None,
        autowire: bool = True,
        run_id: str | None = None,
        run_dir: str | Path | None = "runs",
        console: bool = True,
        max_ticks: int = 200,
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
        self._read_mark = 0
        self._validated = False

        #: Where `prompt` delivers, unless you say otherwise. Defaults to the
        #: first module added. Set it directly to change the front door.
        self.entry: str | None = None

        self.tracer: Tracer = tracer if tracer is not None else RunTracer(self.run_id)
        #: Every event of this run, in order. What the tests read.
        self.events = MemorySink()
        self.tracer.add_sink(self.events)
        self.run_path: Path | None = None
        if run_dir is not None:
            self.run_path = Path(run_dir) / self.run_id
            self.tracer.add_sink(JsonlSink(self.run_path / "trace.jsonl"))
            self.tracer.add_sink(PerModuleSink(self.run_path / "modules"))
        if console:
            self.tracer.add_sink(ConsoleSink())

        #: The caller, as a module. Never scheduled; it only receives.
        self.world = World(self.outbox)
        self.world.attach(self)

        self.scheduler: Scheduler = (
            scheduler(self)
            if scheduler is not None
            else TickScheduler(self, max_ticks=max_ticks)
        )

        #: The target model, as a module. None if the mind was given no model.
        self.subject: Module | None = None
        #: The part that speaks, if there is one. Otherwise the subject speaks.
        self.ego: Module | None = None
        self._stages: list[Module] = []
        self._auto_links: list[tuple[Module, Link]] = []

        if model is not None:
            llm = model if isinstance(model, LLM) else self.model(model)
            self.subject = self.adopt(_build_subject(subject, llm, system))
        if ego is not None:
            self.ego = self.adopt(_build_ego(ego, ego_system, self.model))
        if autowire:
            self.intercept()  # with no stages: subject -> ego -> world

    # -- models ----------------------------------------------------------

    def model(self, spec: str, **kwargs) -> LLM:
        """Build a model through this mind's factory.

        Going through the mind rather than calling `get_llm` directly is what
        lets one constructor argument tape, throttle, or redirect every model
        in the run.
        """
        return self.model_factory(spec, **kwargs)

    # -- interception ----------------------------------------------------

    def intercept(self, *stages: Module) -> list[Module]:
        """Put `stages` between the subject and whoever speaks.

        A stage reads `subject_context` and writes `ego_input`, named for the
        two ends of the path. With one stage the names already line up, so
        nothing is renamed and the wiring reads exactly as it runs.

        Shorthand for `register` calls and nothing else: with one stage and
        an ego, `mind.intercept(editor)` is exactly

            mind.subject.register(editor, "subject_context")
            editor.register(mind.ego, "ego_input")
            mind.ego.register(mind.world, "reply")

        With no stage the subject is renamed straight onto the ego, and with
        two or more the middle links are renamed, since a second stage still
        reads `subject_context`. `describe` shows every rename.

        Write those yourself when the shape is not a line, or pass
        `autowire=False` and wire everything with the one verb.

        Calling it again discards the previous layout and only that: links you
        made by hand are left alone. `describe` marks what came from here.
        """
        if self.subject is None:
            return []

        # Undo exactly what a previous layout put down, and nothing else. A
        # monitor somebody registered by hand has to survive a relayout.
        for module, link in self._auto_links:
            module.unregister(link)
        self._auto_links = []

        self._stages = list(stages)
        for stage in self._stages:
            self.adopt(stage)

        chain: list[Module] = [self.subject, *self._stages]
        speaker = self.ego if self.ego is not None else self.subject
        if self.ego is not None:
            chain.append(self.ego)

        for producer, consumer in zip(chain, chain[1:]):
            # Each end uses its own name. With one stage the two already agree,
            # so nothing is renamed; a longer chain renames the middle links.
            emits = "subject_context" if producer is self.subject else "ego_input"
            hears = "ego_input" if consumer is self.ego else "subject_context"
            self._auto_links.append(
                (producer, producer.register(consumer, emits, as_channel=hears))
            )
        self._auto_links.append(
            (speaker, speaker.register(self.world, "reply"))
        )
        return self._stages

    @property
    def stages(self) -> list[Module]:
        """The modules currently sitting between the subject and the ego."""
        return list(self._stages)

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

        # Uniform: every module, including world, takes it through `receive`.
        # World's override drops it in the outbox instead of a queue.
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

    def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        channel: str = "prompt",
    ) -> list[Message]:
        """Say something. Delivers immediately and returns.

        This does not run the mind. Call `process` for that, then
        `get_replies` to read what came out.

            mind.prompt("hello")
            await mind.process()
            print(texts(mind.get_replies()))

        Delivers to `to`, or to `self.entry` when you do not say.
        """
        return self.send(channel, Text(text), to=to)

    async def process_one(self) -> int:
        """Run exactly one tick. Returns how many modules took a turn.

        Step through an experiment with this, reading state between ticks.
        Zero means nothing had work and the mind is settled.
        """
        self._check_once()
        return await self.scheduler.tick()

    async def process(self, max_ticks: int | None = None) -> int:
        """Tick until nothing has work anywhere. Returns the number of ticks.

        Validates the assembly before the first tick, so a consumer that was
        never added fails loudly instead of silently dropping messages.
        """
        self._check_once()
        return await self.scheduler.run_until_idle(max_ticks)

    def get_replies(self) -> list[Message]:
        """Everything that reached `world` since you last asked.

        Reading drains, so a prompt-process-read loop sees each reply once.
        `mind.outbox` keeps the whole history if you want it.
        """
        new = self.outbox[self._read_mark :]
        self._read_mark = len(self.outbox)
        return new

    # -- inspection --------------------------------------------------------

    def describe(self) -> str:
        lines = [f"Mind {self.name!r} (run {self.run_id})", f"  entry: {self.entry}"]
        lines.append("  modules:")
        for module in self.modules.values():
            channels = ", ".join(sorted(module.OUTPUTS)) or "no outputs"
            lines.append(f"    {module.name:<16} {type(module).__name__:<14} {channels}")
        links = self.links()
        laid_out = {id(link) for _, link in self._auto_links}
        lines.append("  links:")
        for link in links:
            mark = "  [auto]" if id(link) in laid_out else ""
            lines.append(f"    {link.describe()}{mark}")
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
