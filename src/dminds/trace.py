"""Instrumentation. Every module and every model call is logged, always.

The trace is append-only and totally ordered by `seq`, which makes it an event
log you can diff between two runs. Each event carries both a logical tick and a
wall-clock timestamp. Logic must only ever read the tick. Wall time is for you.

A run writes three things under `runs/<run_id>/`:

    trace.jsonl              every event, in order
    modules/<name>.jsonl     the same events, split per module
    console                  a readable stream, if you enable ConsoleSink
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

# Event kinds the runtime emits. Yours can be any string.
TASK_EMIT = "task.emit"
TASK_DELIVER = "task.deliver"
HANDLE_START = "handle.start"
HANDLE_END = "handle.end"
HANDLE_ERROR = "handle.error"
LLM_REQUEST = "llm.request"
LLM_RESPONSE = "llm.response"
LLM_ERROR = "llm.error"
MEMORY_WRITE = "memory.write"
TICK_START = "tick.start"
TICK_END = "tick.end"
NOTE = "note"


@dataclass(slots=True)
class Event:
    seq: int
    tick: int
    wall: str
    elapsed_s: float
    module: str
    kind: str
    data: dict = field(default_factory=dict)
    duration_s: float | None = None
    task_id: str | None = None
    cause: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        """One readable line."""
        head = f"t={self.tick:<3} {self.elapsed_s:7.3f}s {self.module:<14} {self.kind:<14}"
        detail = self.data.get("summary") or _compact(self.data)
        if self.duration_s is not None:
            detail = f"{detail}  ({self.duration_s * 1000:.0f}ms)"
        return f"{head} {detail}"


def _compact(data: dict, width: int = 90) -> str:
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        text = value if isinstance(value, str) else repr(value)
        text = " ".join(str(text).split())
        if len(text) > 40:
            text = text[:39] + "…"
        parts.append(f"{key}={text}")
    joined = " ".join(parts)
    return joined if len(joined) <= width else joined[: width - 1] + "…"


class Sink(Protocol):
    def write(self, event: Event) -> None: ...
    def close(self) -> None: ...


class MemorySink:
    """Keeps events in a list. Useful in tests and notebooks."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def write(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def of_module(self, module: str) -> list[Event]:
        return [e for e in self.events if e.module == module]


class JsonlSink:
    """Appends every event to one file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a")

    def write(self, event: Event) -> None:
        self._handle.write(json.dumps(event.to_dict(), default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class PerModuleSink:
    """Splits the trace into one file per module, so you can read one mind."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}

    def write(self, event: Event) -> None:
        name = event.module or "runtime"
        if name not in self._handles:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            self._handles[name] = (self.directory / f"{safe}.jsonl").open("a")
        handle = self._handles[name]
        handle.write(json.dumps(event.to_dict(), default=str) + "\n")
        handle.flush()

    def close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
        self._handles.clear()


class ConsoleSink:
    """Prints a readable stream.

    `only` and `exclude` filter by event kind so you can watch one thing.
    """

    def __init__(
        self,
        stream: Any = None,
        only: set[str] | None = None,
        exclude: set[str] | None = None,
        predicate: Callable[[Event], bool] | None = None,
    ):
        self.stream = stream or sys.stderr
        self.only = only
        self.exclude = exclude or {TASK_DELIVER, TICK_START}
        self.predicate = predicate

    def write(self, event: Event) -> None:
        if self.only is not None and event.kind not in self.only:
            return
        if event.kind in self.exclude:
            return
        if self.predicate is not None and not self.predicate(event):
            return
        print(event.line(), file=self.stream)

    def close(self) -> None:
        pass


class Tracer:
    """Owns the sequence counter and fans events out to sinks."""

    def __init__(self, run_id: str, sinks: list[Sink] | None = None):
        self.run_id = run_id
        self.sinks: list[Sink] = list(sinks or [])
        self.seq = 0
        self.tick = 0
        self._start = time.perf_counter()

    def add_sink(self, sink: Sink) -> "Tracer":
        self.sinks.append(sink)
        return self

    def emit(
        self,
        module: str,
        kind: str,
        data: dict | None = None,
        duration_s: float | None = None,
        task_id: str | None = None,
        cause: str | None = None,
    ) -> Event:
        event = Event(
            seq=self.seq,
            tick=self.tick,
            wall=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            elapsed_s=time.perf_counter() - self._start,
            module=module,
            kind=kind,
            data=data or {},
            duration_s=duration_s,
            task_id=task_id,
            cause=cause,
        )
        self.seq += 1
        for sink in self.sinks:
            sink.write(event)
        return event

    def bind(self, module: str) -> "ModuleLog":
        """A logger stamped with one module's name."""
        return ModuleLog(self, module)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class ModuleLog:
    """What a module writes to. Every event is tagged with its name and tick."""

    def __init__(self, tracer: Tracer, module: str):
        self.tracer = tracer
        self.module = module

    @property
    def tick(self) -> int:
        return self.tracer.tick

    def event(self, kind: str, **data: Any) -> Event:
        return self.tracer.emit(self.module, kind, data)

    def note(self, summary: str, **data: Any) -> Event:
        """A free-form log line from your own code."""
        return self.tracer.emit(self.module, NOTE, {"summary": summary, **data})

    @contextmanager
    def span(self, kind: str, **data: Any) -> Iterator[dict]:
        """Time a block and record its duration.

        Mutate the yielded dict to attach results to the closing event.

            with log.span("llm.call") as span:
                span["tokens"] = 42
        """
        start = time.perf_counter()
        self.tracer.emit(self.module, f"{kind}.start", dict(data))
        extra: dict = {}
        try:
            yield extra
        except Exception as exc:
            self.tracer.emit(
                self.module,
                f"{kind}.error",
                {**data, **extra, "error": repr(exc)},
                duration_s=time.perf_counter() - start,
            )
            raise
        self.tracer.emit(
            self.module,
            f"{kind}.end",
            {**data, **extra},
            duration_s=time.perf_counter() - start,
        )

    def child(self, suffix: str) -> "ModuleLog":
        """A logger for a subagent, named `parent/child`."""
        return ModuleLog(self.tracer, f"{self.module}/{suffix}")


def read_trace(path: str | Path) -> list[Event]:
    """Load a trace back off disk, for diffing two runs."""
    events = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            events.append(Event(**json.loads(line)))
    return events


def causal_chain(events: list[Event], task_id: str) -> list[Event]:
    """Every event descending from one task. Answers "why did this happen"."""
    wanted = {task_id}
    chain = []
    for event in events:
        if event.task_id in wanted or event.cause in wanted:
            chain.append(event)
            if event.task_id:
                wanted.add(event.task_id)
    return chain
