"""Determinism: the property the whole runtime is arranged around."""

from __future__ import annotations

import asyncio
import random
import tempfile
from pathlib import Path

from src import Agent, BaseModule, Cassette, Ctx, Message, Mind, Text, get_llm


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Jitter(BaseModule):
    """Emits after an unpredictable delay, to try to break ordering."""

    OUTPUTS = {"note": "a note with this module's name on it"}

    async def on_process(self, ctx: Ctx) -> None:
        await asyncio.sleep(random.random() * 0.01)
        ctx.emit("note", Text(f"{self.name}->sink"))


class Collector(BaseModule):
    def __init__(self, name: str):
        super().__init__(name)
        self.got: list[str] = []

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.got.append(message.payload.text)


def test_delivery_order_survives_random_handler_latency():
    """Turns finish in random order; delivery order must not change."""

    async def run():
        mind = quiet_mind()
        sink = Collector("sink")
        mind.add(sink)
        for name in ("a", "b", "c"):
            Jitter(name).register(sink, "note")
        for name in ("a", "b", "c"):
            mind.send("go", Text("x"), to=name)
        await mind.run()
        got = list(sink.got)
        mind.close()
        return got

    runs = [asyncio.run(run()) for _ in range(8)]
    # Join order, every single time.
    assert all(r == ["a->sink", "b->sink", "c->sink"] for r in runs), runs


def test_message_ids_are_stable_across_runs():
    async def run():
        mind = quiet_mind()
        sink, sink2 = Collector("sink"), Collector("sink2")
        mind.add(sink)
        a = Jitter("a")
        a.register(sink, "note")
        a.register(sink2, "note")
        mind.send("go", Text("x"), to="a")
        await mind.run()
        ids = [e.task_id for e in mind.events.events if e.kind == "task.emit"]
        mind.close()
        return ids

    first, second = asyncio.run(run()), asyncio.run(run())
    assert first == second
    # M0001 is the injected prompt; the fan-out follows in registration order.
    assert first == ["M0001", "M0002", "M0003"]


def test_two_runs_of_the_same_mind_have_the_same_trace_shape():
    async def run(tape):
        mind = quiet_mind()
        agent = Agent("agent", Cassette(get_llm("echo:"), tape, mode="auto"))
        agent.register(mind.world, "reply")
        await mind.prompt("hello")
        shape = [(e.tick, e.module, e.kind) for e in mind.events.events]
        mind.close()
        return shape

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "tape.jsonl"
        first, second = asyncio.run(run(tape)), asyncio.run(run(tape))
    assert first == second


def test_cassette_makes_a_random_model_reproducible():
    def unpredictable(messages, opts):
        return f"n={random.randint(0, 10**9)}"

    async def run(tape, mode):
        mind = quiet_mind()
        agent = Agent(
            "agent", Cassette(get_llm("echo:", rule=unpredictable), tape, mode=mode)
        )
        agent.register(mind.world, "reply")
        replies = await mind.prompt("hello")
        mind.close()
        return replies[0].payload.text

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "tape.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        replays = [asyncio.run(run(tape, "replay")) for _ in range(3)]

    assert all(r == recorded for r in replays), (recorded, replays)
