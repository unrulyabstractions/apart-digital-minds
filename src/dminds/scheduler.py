"""`TickScheduler`: the standard implementation of `api.Scheduler`.

This file defines what "one step" means. Replace it to get a different notion
of time.

One tick has two phases.

    Phase 1, act.     Every module with something to do takes a turn. A turn
                      is `on_input` for each message that arrived, in order,
                      then one `on_process`. Modules take their turns
                      concurrently, so two agents think at the same moment.
    Phase 2, deliver. Everything emitted during phase 1 lands in its target
                      queue, all at once.

Nothing emitted at tick t is visible before tick t+1. That single rule is what
makes a run deterministic: a module can never observe how far another module
got inside the same tick, so concurrency cannot change the outcome. Each module
writes to a private outbox and those outboxes are concatenated in registration
order, so the delivered sequence is fixed too.

Absorbing every input before processing is why the two phases of a turn are
separate. A module never decides on half of what reached it.

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
from ..api.types import Message
from .module import Ctx

if TYPE_CHECKING:  # pragma: no cover
    from ..api.runtime import Host


class TickScheduler(Scheduler):
    """Runs every module that has work, concurrently, one turn each per tick."""

    def __init__(self, mind: "Host", strict: bool = True, max_ticks: int = 200):
        self.mind = mind
        self.strict = strict
        self.max_ticks = max_ticks
        self.t = 0

    def active(self) -> list[Module]:
        """Modules with queued input, or that asked to run anyway."""
        return [
            m
            for m in self.mind.modules.values()
            if m.pending or m.wants_process()
        ]

    # -- one tick ------------------------------------------------------

    async def tick(self) -> int:
        """Run one tick. Returns how many modules took a turn."""
        mind = self.mind
        mind.tracer.tick = self.t

        active = self.active()
        if not active:
            return 0

        mind.tracer.emit(
            "runtime",
            TICK_START,
            {"active": [m.name for m in active], "summary": f"{len(active)} active"},
        )

        # Phase 1: act. Private outbox per module keeps ordering deterministic
        # even though the turns run concurrently.
        outboxes: list[list[Message]] = [[] for _ in active]
        await asyncio.gather(
            *(
                self._take_turn(module, outbox)
                for module, outbox in zip(active, outboxes)
            )
        )

        # Phase 2: deliver, in module registration order.
        delivered = 0
        for outbox in outboxes:
            for message in outbox:
                message.t_deliver = self.t + 1
                if mind.deliver(message):
                    delivered += 1

        mind.tracer.emit(
            "runtime",
            TICK_END,
            {
                "turns": len(active),
                "delivered": delivered,
                "summary": f"{len(active)} turns, delivered {delivered}",
            },
        )
        self.t += 1
        return len(active)

    async def _take_turn(self, module: Module, outbox: list[Message]) -> None:
        """One module's turn: every input, then one process."""
        arrived = module.drain()
        ctx = Ctx(
            tick=self.t,
            module=module,
            mind=self.mind,
            log=module.log,
            outbox=outbox,
        )
        tracer = self.mind.tracer
        summary = (
            f"{len(arrived)} in: "
            + ", ".join(m.channel for m in arrived[:3])
            + ("…" if len(arrived) > 3 else "")
            if arrived
            else "no input, wants_process"
        )
        tracer.emit(
            module.name,
            HANDLE_START,
            {"inputs": len(arrived), "summary": summary},
            task_id=arrived[0].id if arrived else None,
        )

        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            for message in arrived:
                await module.on_input(message, ctx)
            await module.on_process(ctx)
        except Exception as exc:
            tracer.emit(
                module.name,
                HANDLE_ERROR,
                {"error": repr(exc), "summary": f"FAILED {exc!r}"},
                duration_s=loop.time() - start,
            )
            if self.strict:
                raise
            return

        module.turns += 1
        tracer.emit(
            module.name,
            HANDLE_END,
            {"inputs": len(arrived), "emitted": len(outbox)},
            duration_s=loop.time() - start,
        )

    # -- running to quiescence ------------------------------------------

    def is_idle(self) -> bool:
        return not self.active()

    async def run_until_idle(self, max_ticks: int | None = None) -> int:
        """Tick until nothing has work. Returns the number of ticks run.

        This is the guarantee behind `mind.prompt`: an external input is fully
        digested before control comes back to you.
        """
        limit = self.max_ticks if max_ticks is None else max_ticks
        ticks = 0
        while not self.is_idle():
            if ticks >= limit:
                backlog = {m.name: m.pending for m in self.active()}
                raise RunawayMind(
                    f"Still busy after {limit} ticks. Outstanding: {backlog}. "
                    f"Raise max_ticks if this is expected, check for two modules "
                    f"replying to each other forever, or for a module whose "
                    f"wants_process never turns False."
                )
            await self.tick()
            ticks += 1
        return ticks
