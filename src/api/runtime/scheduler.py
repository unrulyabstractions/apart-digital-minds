"""What "one step" means."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RunawayMind(RuntimeError):
    """The mind never went quiet. Usually two modules answering each other."""


class Scheduler(ABC):
    """The clock.

    The shipped implementation, `TickScheduler`, runs every module holding work
    concurrently, one task each, then delivers everything at once. Nothing
    emitted at tick t is visible before t+1, which is what makes a run
    deterministic despite the concurrency.

    Replace it for a different notion of time: priority queues, real-time
    deadlines, or one module per tick.
    """

    #: The tick about to run, or being run.
    t: int

    @abstractmethod
    async def tick(self) -> int:
        """Run one tick. Returns how many tasks were handled."""

    @abstractmethod
    def is_idle(self) -> bool:
        """True when no module has queued work."""

    @abstractmethod
    async def run_until_idle(self, max_ticks: int | None = None) -> int:
        """Tick until every queue is empty. Returns the number of ticks run.

        This is the guarantee behind `Mind.prompt`: an external input is fully
        digested before control returns to the caller.
        """
