"""The scheduling rule, and the shape of a turn."""

from __future__ import annotations

import asyncio

from src import BaseModule, Ctx, Message, Mind, RunawayMind, Text


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Recorder(BaseModule):
    """Notes the tick of every input it saw, and of every process step."""

    OUTPUTS = {"ping": "a forwarded ping"}

    def __init__(self, name: str, forward: bool = False, limit: int = 1):
        super().__init__(name)
        self.inputs_seen: list[tuple[int, str]] = []
        self.processed: list[int] = []
        self.forward = forward
        self.limit = limit

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.inputs_seen.append((ctx.tick, message.channel))

    async def on_process(self, ctx: Ctx) -> None:
        self.processed.append(ctx.tick)
        if self.forward and len(self.inputs_seen) <= self.limit:
            ctx.emit("ping", Text("x"))


def test_emission_is_not_visible_in_the_same_tick():
    async def run():
        mind = quiet_mind()
        a, b = Recorder("a", forward=True), Recorder("b")
        mind.add(a, b)
        a.register(b, "ping")
        mind.send("ping", Text("x"), to="a")
        await mind.run()
        mind.close()
        return a, b

    a, b = asyncio.run(run())
    assert a.inputs_seen == [(0, "ping")]
    # b must not have run at tick 0, however fast a was.
    assert b.inputs_seen == [(1, "ping")]


def test_a_turn_is_every_input_then_one_process():
    """Three messages arrive together: three on_input, one on_process."""

    async def run():
        mind = quiet_mind()
        a = Recorder("a")
        mind.add(a)
        for _ in range(3):
            mind.send("ping", Text("x"), to="a")
        await mind.run()
        mind.close()
        return a

    a = asyncio.run(run())
    assert len(a.inputs_seen) == 3
    assert [t for t, _ in a.inputs_seen] == [0, 0, 0], "all absorbed in one turn"
    assert a.processed == [0], "process runs once, after every input"


def test_run_until_idle_drains_a_chain():
    async def run():
        mind = quiet_mind()
        a, b, c = Recorder("a", forward=True), Recorder("b", forward=True), Recorder("c")
        mind.add(a, b, c)
        a.register(b, "ping")
        b.register(c, "ping")
        mind.send("ping", Text("x"), to="a")
        ticks = await mind.run()
        idle = mind.scheduler.is_idle()
        mind.close()
        return ticks, idle

    ticks, idle = asyncio.run(run())
    assert ticks == 3
    assert idle


def test_modules_in_one_tick_run_concurrently():
    """Three slow modules in the same tick take one duration, not three."""

    class Slow(BaseModule):
        async def on_input(self, message: Message, ctx: Ctx) -> None:
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
    assert elapsed < 0.12, f"three 50ms turns took {elapsed:.3f}s, so they serialised"


def test_runaway_is_caught():
    async def run():
        mind = quiet_mind(max_ticks=10)
        a = Recorder("a", forward=True, limit=999)
        b = Recorder("b", forward=True, limit=999)
        mind.add(a, b)
        a.register(b, "ping")
        b.register(a, "ping")
        mind.send("ping", Text("x"), to="a")
        try:
            await mind.run()
        except RunawayMind as exc:
            mind.close()
            return str(exc)
        mind.close()
        return None

    message = asyncio.run(run())
    assert message is not None and "10 ticks" in message


def test_wants_process_lets_a_module_act_unprompted():
    """A module with an empty queue can still take a turn."""

    class Spontaneous(BaseModule):
        def __init__(self, name: str, times: int):
            super().__init__(name)
            self.times = times
            self.ran = 0

        def wants_process(self) -> bool:
            return self.ran < self.times

        async def on_process(self, ctx: Ctx) -> None:
            self.ran += 1

    async def run():
        mind = quiet_mind()
        s = Spontaneous("s", times=3)
        mind.add(s)
        ticks = await mind.run()
        mind.close()
        return s.ran, ticks

    ran, ticks = asyncio.run(run())
    assert ran == 3, "it should have acted three times with an empty queue"
    assert ticks == 3


def test_an_idle_module_costs_nothing():
    class Quiet(BaseModule):
        pass

    async def run():
        mind = quiet_mind()
        mind.add(Quiet("q"))
        ticks = await mind.run()
        mind.close()
        return ticks

    assert asyncio.run(run()) == 0


def test_prompt_returns_only_this_prompts_output():
    class Talker(BaseModule):
        OUTPUTS = {"reply": "an answer"}

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            ctx.emit("reply", Text(f"re:{message.payload.text}"))

    async def run():
        mind = quiet_mind()
        t = mind.add(Talker("t"))
        t.register(mind.world, "reply")
        first = await mind.prompt("one")
        second = await mind.prompt("two")
        mind.close()
        return first, second

    first, second = asyncio.run(run())
    assert len(first) == 1 and first[0].payload.text == "re:one"
    assert len(second) == 1 and second[0].payload.text == "re:two"
