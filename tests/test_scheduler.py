"""The scheduling rule, which everything else depends on."""

from __future__ import annotations

import asyncio

from src.api import Ctx, Mind, Module, RunawayMind, Task, Text


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Recorder(Module):
    """Notes the tick on which it handled each task."""

    def __init__(self, name: str, forward_to: str | None = None, limit: int = 1):
        super().__init__(name)
        self.seen: list[tuple[int, str]] = []
        self.forward_to = forward_to
        self.limit = limit

    async def process(self, task: Task, ctx: Ctx) -> None:
        self.seen.append((ctx.tick, task.kind))
        if self.forward_to and len(self.seen) <= self.limit:
            ctx.emit("ping", task.payload, to=self.forward_to)


def test_emission_is_not_visible_in_the_same_tick():
    async def run():
        mind = quiet_mind()
        a = Recorder("a", forward_to="b")
        b = Recorder("b")
        mind.add(a, b)
        mind.send("ping", Text("x"), to="a")
        await mind.run()
        mind.close()
        return a, b

    a, b = asyncio.run(run())
    assert a.seen == [(0, "ping")]
    # b must not have run at tick 0, however fast a was.
    assert b.seen == [(1, "ping")]


def test_run_until_idle_drains_a_chain():
    async def run():
        mind = quiet_mind()
        mind.add(
            Recorder("a", forward_to="b"),
            Recorder("b", forward_to="c"),
            Recorder("c"),
        )
        mind.send("ping", Text("x"), to="a")
        ticks = await mind.run()
        idle = mind.scheduler.is_idle()
        mind.close()
        return ticks, idle

    ticks, idle = asyncio.run(run())
    assert ticks == 3
    assert idle


def test_modules_in_one_tick_run_concurrently():
    """Two slow modules in the same tick take one duration, not two."""

    class Slow(Module):
        async def process(self, task: Task, ctx: Ctx) -> None:
            await asyncio.sleep(0.05)

    async def run():
        mind = quiet_mind()
        mind.add(Slow("a"), Slow("b"), Slow("c"))
        for name in ("a", "b", "c"):
            mind.send("go", Text("x"), to=name)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await mind.run()
        elapsed = loop.time() - start
        mind.close()
        return elapsed

    elapsed = asyncio.run(run())
    assert elapsed < 0.12, f"three 50ms handlers took {elapsed:.3f}s, so they serialised"


def test_runaway_is_caught():
    async def run():
        mind = quiet_mind(max_ticks=10)
        # Two modules that answer each other forever.
        mind.add(Recorder("a", forward_to="b", limit=999))
        mind.add(Recorder("b", forward_to="a", limit=999))
        mind.send("ping", Text("x"), to="a")
        try:
            await mind.run()
        except RunawayMind as exc:
            mind.close()
            return str(exc)
        mind.close()
        return None

    message = asyncio.run(run())
    assert message is not None
    assert "10 ticks" in message


def test_one_task_per_module_per_tick():
    """A module with three queued tasks handles them one tick at a time."""

    async def run():
        mind = quiet_mind()
        a = Recorder("a")
        mind.add(a)
        for _ in range(3):
            mind.send("ping", Text("x"), to="a")
        await mind.run()
        mind.close()
        return a.seen

    seen = asyncio.run(run())
    assert [tick for tick, _ in seen] == [0, 1, 2]


def test_prompt_returns_only_this_prompts_output():
    class Talker(Module):
        async def process(self, task: Task, ctx: Ctx) -> None:
            ctx.emit("reply", Text(f"re:{task.payload.text}"), to="world")

    async def run():
        mind = quiet_mind()
        mind.add(Talker("t"))
        first = await mind.prompt("one")
        second = await mind.prompt("two")
        mind.close()
        return first, second

    first, second = asyncio.run(run())
    assert len(first) == 1 and first[0].payload.text == "re:one"
    assert len(second) == 1 and second[0].payload.text == "re:two"
