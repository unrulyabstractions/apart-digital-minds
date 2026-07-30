"""`TickScheduler`: the standard implementation of `api.Scheduler`.

This file defines what "one step" means. Replace it to get a different notion
of time.

One tick has two phases.

    Phase 1, act.     Every module holding at least one task pops exactly one
                      and handles it. Modules run concurrently, so two agents
                      think at the same wall-clock moment.
    Phase 2, deliver. Everything emitted during phase 1 lands in its target
                      queue, all at once.

Nothing emitted at tick t is visible before tick t+1. That single rule is what
makes a run deterministic: a module can never observe how far another module
got inside the same tick, so concurrency cannot change the outcome. Each module
writes to a private outbox and those outboxes are concatenated in registration
order, so the delivered sequence is fixed too.

The lock-step this produces is exactly the target/interceptor protocol:

    t=0  target answers, emits its context.   interceptor idle.
    t=1  target idle.                         interceptor rewrites, emits back.
    t=2  target consumes the rewrite.         interceptor idle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..api.modules import Module
from ..api.observability import (
    HANDLE_END,
    HANDLE_ERROR,
    HANDLE_START,
    TICK_END,
    TICK_START,
)
from ..api.runtime import RunawayMind, Scheduler
from ..api.types import Task
from .module import Ctx

if TYPE_CHECKING:  # pragma: no cover
    from ..api.runtime import Host


class TickScheduler(Scheduler):
    """Runs every module holding work concurrently, one task each per tick."""

    def __init__(self, mind: "Host", strict: bool = True, max_ticks: int = 200):
        self.mind = mind
        self.strict = strict
        self.max_ticks = max_ticks
        self.t = 0

    # -- one tick ------------------------------------------------------

    async def tick(self) -> int:
        """Run one tick. Returns how many tasks were handled."""
        mind = self.mind
        mind.tracer.tick = self.t

        active = [m for m in mind.modules.values() if m.pending]
        if not active:
            return 0

        mind.tracer.emit(
            "runtime",
            TICK_START,
            {"active": [m.name for m in active], "summary": f"{len(active)} active"},
        )

        # Phase 1: act. Private outbox per module keeps ordering deterministic
        # even though the handlers run concurrently.
        outboxes: list[list[Task]] = [[] for _ in active]
        await asyncio.gather(
            *(
                self._handle_one(module, outbox)
                for module, outbox in zip(active, outboxes)
            )
        )

        # Phase 2: deliver, in module registration order.
        delivered = 0
        for outbox in outboxes:
            for task in outbox:
                task.t_deliver = self.t + 1
                if mind.deliver(task):
                    delivered += 1

        mind.tracer.emit(
            "runtime",
            TICK_END,
            {
                "handled": len(active),
                "delivered": delivered,
                "summary": f"handled {len(active)}, delivered {delivered}",
            },
        )
        self.t += 1
        return len(active)

    async def _handle_one(self, module: Module, outbox: list[Task]) -> None:
        task = module.next_task()
        if task is None:
            return

        ctx = Ctx(
            tick=self.t,
            task=task,
            module=module,
            mind=self.mind,
            log=module.log,
            outbox=outbox,
        )
        tracer = self.mind.tracer
        tracer.emit(
            module.name,
            HANDLE_START,
            {"kind": task.kind, "src": task.src, "summary": task.describe()},
            task_id=task.id,
            cause=task.cause,
        )

        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            await module.process(task, ctx)
        except Exception as exc:
            tracer.emit(
                module.name,
                HANDLE_ERROR,
                {"kind": task.kind, "error": repr(exc), "summary": f"FAILED {exc!r}"},
                duration_s=loop.time() - start,
                task_id=task.id,
            )
            if self.strict:
                raise
            return

        module.handled += 1
        tracer.emit(
            module.name,
            HANDLE_END,
            {"kind": task.kind, "emitted": len(outbox)},
            duration_s=loop.time() - start,
            task_id=task.id,
        )

    # -- running to quiescence ------------------------------------------

    def is_idle(self) -> bool:
        return not any(m.pending for m in self.mind.modules.values())

    async def run_until_idle(self, max_ticks: int | None = None) -> int:
        """Tick until every queue is empty. Returns the number of ticks run.

        This is the guarantee behind `mind.prompt`: an external input is fully
        digested before control comes back to you.
        """
        limit = self.max_ticks if max_ticks is None else max_ticks
        ticks = 0
        while not self.is_idle():
            if ticks >= limit:
                backlog = {
                    m.name: m.pending for m in self.mind.modules.values() if m.pending
                }
                raise RunawayMind(
                    f"Still busy after {limit} ticks. Outstanding queues: {backlog}. "
                    f"Raise max_ticks if this is expected, or check for two modules "
                    f"replying to each other forever."
                )
            await self.tick()
            ticks += 1
        return ticks
