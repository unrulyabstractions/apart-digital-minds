"""The assembly point: modules, routes, a clock, and a trace.

    mind = Mind("demo")
    mind.add(Agent("assistant", get_llm("echo:")))
    replies = await mind.prompt("hello")

`prompt` injects one external input and then runs until every queue is empty,
so when it returns, the mind has finished thinking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .bus import Bus, Route
from .messages import Payload, Task, Text
from .module import Module
from .scheduler import Scheduler
from .trace import (
    TASK_DELIVER,
    TASK_EMIT,
    ConsoleSink,
    JsonlSink,
    MemorySink,
    PerModuleSink,
    Sink,
    Tracer,
)

#: The virtual module that stands for you, outside the mind.
WORLD = "world"


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


class Mind:
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
    ):
        self.name = name
        self.run_id = run_id or _fresh_run_id(run_dir)
        self.modules: dict[str, Module] = {}
        self.bus = Bus()
        self.outbox: list[Task] = []
        self._task_counter = 0

        #: Where `prompt` delivers, unless you say otherwise. Defaults to the
        #: first module added. Set it directly to change the front door.
        self.entry: str | None = None

        self.tracer = Tracer(self.run_id)
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

        self.scheduler = Scheduler(self, strict=strict, max_ticks=max_ticks)

    # -- assembly ------------------------------------------------------

    def add(self, *modules: Module) -> Module:
        """Register modules. Returns the last one, so you can inline it."""
        last = None
        for module in modules:
            if module.name in self.modules:
                raise ValueError(f"A module named {module.name!r} is already added.")
            if module.name == WORLD:
                raise ValueError(f"{WORLD!r} is reserved for the outside caller.")
            self.modules[module.name] = module
            module.attach(self)
            if self.entry is None:
                self.entry = module.name
            last = module
        if last is None:
            raise ValueError("add() needs at least one module.")
        return last

    def wire(
        self, src: str, kind: str, dst: str, as_kind: str | None = None
    ) -> Route:
        """Connect an emitter to a receiver. See `dmind.bus.Bus.wire`."""
        return self.bus.wire(src, kind, dst, as_kind)

    def watch(
        self, dst: str, kind: str = Bus.WILDCARD, src: str = Bus.WILDCARD
    ) -> Route:
        """Give one module a copy of matching traffic, however it was addressed.

        This is how you attach a monitor that reads what other modules say to
        each other without being in the middle of the conversation.
        """
        return self.bus.observe(dst, kind, src)

    # -- task plumbing ---------------------------------------------------

    def _next_id(self) -> str:
        self._task_counter += 1
        return f"T{self._task_counter:04d}"

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

        Called by `Ctx.emit`. Nothing is delivered here; the scheduler does that
        at the end of the tick.
        """
        if to is None:
            targets = self.bus.resolve(src, kind)
        elif isinstance(to, str):
            targets = [(to, kind)]
        else:
            targets = [(name, kind) for name in to]

        # Observers get a copy even when the emission was addressed elsewhere.
        addressed = {dst for dst, _ in targets}
        for dst, as_kind in self.bus.observers_for(src, kind):
            if dst not in addressed:
                targets.append((dst, as_kind))
                addressed.add(dst)

        if not targets:
            self.tracer.emit(
                src,
                TASK_EMIT,
                {
                    "kind": kind,
                    "summary": f"{kind} went nowhere (no route, no explicit to=)",
                    "dropped": True,
                },
                cause=cause,
            )
            return []

        tasks = []
        for dst, as_kind in targets:
            task = Task(
                id=self._next_id(),
                kind=as_kind,
                payload=payload,
                src=src,
                dst=dst,
                t_created=self.scheduler.t,
                t_deliver=self.scheduler.t + 1,
                cause=cause,
            )
            outbox.append(task)
            tasks.append(task)
            self.tracer.emit(
                src,
                TASK_EMIT,
                {"kind": as_kind, "dst": dst, "summary": task.describe()},
                task_id=task.id,
                cause=cause,
            )
        return tasks

    def deliver(self, task: Task) -> bool:
        """Put a task in its target queue. Returns False if it had nowhere to go."""
        if task.dst == WORLD:
            self.outbox.append(task)
            self.tracer.emit(
                WORLD,
                TASK_DELIVER,
                {"kind": task.kind, "summary": task.describe()},
                task_id=task.id,
                cause=task.cause,
            )
            return True

        module = self.modules.get(task.dst)
        if module is None:
            self.tracer.emit(
                "runtime",
                TASK_DELIVER,
                {
                    "summary": f"no module named {task.dst!r}, dropped {task.kind}",
                    "dropped": True,
                },
                task_id=task.id,
            )
            return False

        module.receive(task)
        self.tracer.emit(
            task.dst,
            TASK_DELIVER,
            {"kind": task.kind, "src": task.src, "summary": task.describe()},
            task_id=task.id,
            cause=task.cause,
        )
        return True

    # -- driving it ------------------------------------------------------

    def send(
        self,
        kind: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
        src: str = WORLD,
    ) -> list[Task]:
        """Inject an external input. Delivered immediately, not next tick."""
        staged: list[Task] = []
        tasks = self.stage(src, kind, payload, to, cause=None, outbox=staged)
        for task in staged:
            task.t_deliver = self.scheduler.t
            self.deliver(task)
        return tasks

    async def run(self, max_ticks: int | None = None) -> int:
        """Tick until quiet. Returns the number of ticks run."""
        return await self.scheduler.run_until_idle(max_ticks)

    async def prompt(
        self,
        text: str,
        to: str | Sequence[str] | None = None,
        kind: str = "user_prompt",
        max_ticks: int | None = None,
    ) -> list[Task]:
        """Say something, wait for the mind to settle, take what it produced.

        Delivers to `to`, or to `self.entry` when you do not say. Returns only
        the tasks addressed to `world` during this prompt.
        """
        mark = len(self.outbox)
        self.send(kind, Text(text), to=to if to is not None else self.entry)
        await self.run(max_ticks)
        return self.outbox[mark:]

    # -- inspection --------------------------------------------------------

    def describe(self) -> str:
        lines = [f"Mind {self.name!r} (run {self.run_id})"]
        lines.append(f"  entry: {self.entry}")
        lines.append("  modules:")
        for module in self.modules.values():
            lines.append(f"    {module.name:<16} {type(module).__name__}")
        lines.append("  routes:")
        for line in self.bus.describe().splitlines():
            lines.append(f"    {line}")
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


def texts(tasks: Iterable[Task]) -> list[str]:
    """Pull the plain text out of a batch of tasks, skipping anything else."""
    out = []
    for task in tasks:
        payload = task.payload
        if isinstance(payload, Text):
            out.append(payload.text)
        elif isinstance(payload, str):
            out.append(payload)
    return out
