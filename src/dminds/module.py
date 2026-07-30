"""`BaseModule`: the standard implementation of `api.Module`.

Subclass it and write one method per task kind. A task of kind `"user_prompt"`
is handled by `on_user_prompt`. That is the whole convention.

    class Critic(BaseModule):
        async def on_inspect(self, task, ctx):
            ctx.emit("verdict", "looks wrong", to="assistant")

Handlers never call another module directly. They emit, and the scheduler
delivers on the next tick.

`Ctx` here is the concrete object handed to handlers. It satisfies `api.Ctx`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ..api.modules import Ctx as CtxProtocol
from ..api.modules import Module
from ..api.observability import Logger
from ..api.types import Payload, Task

if TYPE_CHECKING:  # pragma: no cover
    from ..api.runtime import Host


def handler_name(kind: str) -> str:
    """`"inspect.context"` -> `"on_inspect_context"`."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in kind)
    return f"on_{safe}"


@dataclass
class Ctx(CtxProtocol):
    """What a handler is given: where it is, what it can say, how to log."""

    tick: int
    task: Task
    module: Module
    mind: "Host"
    log: Logger
    outbox: list[Task] = field(default_factory=list)

    def emit(
        self,
        kind: str,
        payload: Payload,
        to: str | Sequence[str] | None = None,
        cause: str | None = None,
    ) -> list[Task]:
        """Send a task. It is delivered at the start of the next tick.

        `to` names one or more modules. Leave it out to use the routes wired on
        the mind. Use `to="world"` to hand something back to the caller.
        """
        return self.mind.stage(
            src=self.module.name,
            kind=kind,
            payload=payload,
            to=to,
            cause=cause if cause is not None else self.task.id,
            outbox=self.outbox,
        )

    def reply(self, kind: str, payload: Payload) -> list[Task]:
        """Emit straight back to whoever sent the current task."""
        return self.emit(kind, payload, to=self.task.src)


class BaseModule(Module):
    """A queue plus `on_<kind>` dispatch. The usual thing to subclass."""

    def __init__(self, name: str):
        self.name = name
        self.inbox: deque[Task] = deque()
        self.mind: "Host | None" = None
        self.log: Logger | None = None
        self.handled = 0

    # -- lifecycle -----------------------------------------------------

    def attach(self, mind: "Host") -> None:
        """Called once when the module joins a mind. Override to set up."""
        self.mind = mind
        self.log = mind.tracer.bind(self.name)

    def close(self) -> None:
        """Called when the mind shuts down. Override to release resources."""

    # -- queue ---------------------------------------------------------

    def receive(self, task: Task) -> None:
        self.inbox.append(task)

    @property
    def pending(self) -> int:
        return len(self.inbox)

    def next_task(self) -> Task | None:
        """Take one task, in arrival order."""
        return self.inbox.popleft() if self.inbox else None

    # -- dispatch ------------------------------------------------------

    def find_handler(self, kind: str) -> Callable[..., Any]:
        """Route a kind to a method. Override for your own routing scheme."""
        return getattr(self, handler_name(kind), self.on_default)

    async def process(self, task: Task, ctx: Ctx) -> None:
        """The single entry point. Routes to `on_<kind>`.

        Override this if you want dispatch to work some other way.
        """
        await self.find_handler(task.kind)(task, ctx)

    async def on_default(self, task: Task, ctx: Ctx) -> None:
        """Reached when no `on_<kind>` method exists. Logs and drops."""
        ctx.log.note(
            f"no handler for {task.kind!r}, dropped",
            kind=task.kind,
            expected_method=handler_name(task.kind),
            src=task.src,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} pending={self.pending}>"


class FnModule(BaseModule):
    """Wraps a plain function as a module, for glue and quick probes.

        FnModule("counter", lambda task, ctx: ctx.emit("n", 1, to="world"))

    The function may be sync or async and is called for every task kind.
    """

    def __init__(self, name: str, fn: Callable[[Task, Ctx], Any]):
        super().__init__(name)
        self.fn = fn

    async def process(self, task: Task, ctx: Ctx) -> None:
        result = self.fn(task, ctx)
        if hasattr(result, "__await__"):
            await result
