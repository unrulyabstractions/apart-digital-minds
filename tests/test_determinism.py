"""Determinism: the property the whole runtime is arranged around."""

from __future__ import annotations

import asyncio
import random

from dmind import Agent, Cassette, Ctx, Mind, Module, Task, Text, get_llm


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Jitter(Module):
    """Emits after an unpredictable delay, to try to break ordering."""

    def __init__(self, name: str, targets: list[str]):
        super().__init__(name)
        self.targets = targets

    async def process(self, task: Task, ctx: Ctx) -> None:
        await asyncio.sleep(random.random() * 0.01)
        for target in self.targets:
            ctx.emit("note", Text(f"{self.name}->{target}"), to=target)


class Collector(Module):
    def __init__(self, name: str):
        super().__init__(name)
        self.got: list[str] = []

    async def process(self, task: Task, ctx: Ctx) -> None:
        self.got.append(task.payload.text)


def test_delivery_order_survives_random_handler_latency():
    """Handlers finish in random order; delivery order must not change."""

    async def run():
        mind = quiet_mind()
        mind.add(
            Jitter("a", ["sink"]),
            Jitter("b", ["sink"]),
            Jitter("c", ["sink"]),
            Collector("sink"),
        )
        for name in ("a", "b", "c"):
            mind.send("go", Text("x"), to=name)
        await mind.run()
        got = list(mind.modules["sink"].got)
        mind.close()
        return got

    runs = [asyncio.run(run()) for _ in range(8)]
    # Registration order, every single time.
    assert all(r == ["a->sink", "b->sink", "c->sink"] for r in runs), runs


def test_task_ids_are_stable_across_runs():
    async def run():
        mind = quiet_mind()
        mind.add(Jitter("a", ["sink", "sink2"]), Collector("sink"), Collector("sink2"))
        mind.send("go", Text("x"), to="a")
        await mind.run()
        ids = [e.task_id for e in mind.events.events if e.kind == "task.emit"]
        mind.close()
        return ids

    first = asyncio.run(run())
    second = asyncio.run(run())
    assert first == second
    # T0001 is the injected prompt; T0002/T0003 are the fan-out, in target order.
    assert first == ["T0001", "T0002", "T0003"]


def test_two_runs_of_the_same_mind_have_the_same_trace_shape():
    async def run(tape):
        mind = quiet_mind()
        llm = Cassette(get_llm("echo:"), tape, mode="auto")
        mind.add(Agent("agent", llm))
        await mind.prompt("hello")
        shape = [(e.tick, e.module, e.kind) for e in mind.events.events]
        mind.close()
        return shape

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "tape.jsonl"
        first = asyncio.run(run(tape))
        second = asyncio.run(run(tape))
    assert first == second


def test_cassette_makes_a_random_model_reproducible():
    def unpredictable(messages, opts):
        return f"n={random.randint(0, 10**9)}"

    async def run(tape, mode):
        mind = quiet_mind()
        llm = Cassette(get_llm("echo:", rule=unpredictable), tape, mode=mode)
        mind.add(Agent("agent", llm))
        replies = await mind.prompt("hello")
        mind.close()
        return replies[0].payload.text

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "tape.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        replays = [asyncio.run(run(tape, "replay")) for _ in range(3)]

    assert all(r == recorded for r in replays), (recorded, replays)
