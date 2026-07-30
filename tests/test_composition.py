"""Dependency injection, wiring validation, and the model factory.

These are the tests that make the interfaces real rather than decorative. If
`Mind` constructed its own router and scheduler, the contracts next door would
describe nothing.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from src import Agent, BaseModule, Bus, Ctx, Mind, Task, Text, get_llm, taped, texts
from src.api import Host
from src.api import Mind as MindInterface
from src.dminds import RunTracer, TickScheduler


def quiet(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Echoer(BaseModule):
    async def process(self, task: Task, ctx: Ctx) -> None:
        ctx.emit("reply", Text("ok"), to="world")


# -- the interface ---------------------------------------------------------


def test_mind_satisfies_both_its_interfaces():
    assert MindInterface in Mind.__mro__
    assert Host in Mind.__mro__


def test_host_is_narrower_than_mind():
    """A module gets a Host, which cannot add modules or drive the clock."""
    host_only = {n for n in dir(Host) if not n.startswith("_")}
    mind_only = {n for n in dir(MindInterface) if not n.startswith("_")}
    assert "stage" in host_only and "deliver" in host_only
    for name in ("add", "wire", "run", "prompt"):
        assert name in mind_only, name
        assert name not in host_only, f"{name} should not be reachable from a module"


# -- injection -------------------------------------------------------------


def test_router_can_be_injected():
    class CountingBus(Bus):
        def __init__(self):
            super().__init__()
            self.resolved = 0

        def resolve(self, src, kind):
            self.resolved += 1
            return super().resolve(src, kind)

    class Unaddressed(BaseModule):
        """Emits with no `to=`, so the router has to decide where it goes."""

        async def process(self, task: Task, ctx: Ctx) -> None:
            ctx.emit("reply", Text("ok"))

    async def run():
        bus = CountingBus()
        mind = quiet(router=bus)
        mind.add(Unaddressed("a"))
        mind.wire("a", "reply", "world")
        out = texts(await mind.prompt("hi"))
        mind.close()
        return bus, out

    bus, out = asyncio.run(run())
    assert bus.resolved > 0, "the injected router was never consulted"
    assert out == ["ok"], "the injected router did not deliver"


def test_scheduler_can_be_injected():
    seen = {}

    class LoudScheduler(TickScheduler):
        async def tick(self):
            seen["ticked"] = seen.get("ticked", 0) + 1
            return await super().tick()

    async def run():
        mind = quiet(scheduler=lambda host: LoudScheduler(host, max_ticks=5))
        mind.add(Echoer("a"))
        await mind.prompt("hi")
        assert isinstance(mind.scheduler, LoudScheduler)
        mind.close()

    asyncio.run(run())
    assert seen["ticked"] == 1


def test_scheduler_factory_receives_the_host():
    got = {}

    def factory(host):
        got["host"] = host
        return TickScheduler(host)

    mind = quiet(scheduler=factory)
    assert got["host"] is mind
    mind.close()


def test_tracer_can_be_injected():
    tracer = RunTracer("custom-run")
    mind = quiet(tracer=tracer)
    assert mind.tracer is tracer
    mind.close()


def test_defaults_are_used_when_nothing_is_injected():
    mind = quiet()
    assert isinstance(mind.bus, Bus)
    assert isinstance(mind.scheduler, TickScheduler)
    assert isinstance(mind.tracer, RunTracer)
    assert mind.model_factory is get_llm
    mind.close()


# -- validation ------------------------------------------------------------


def test_validate_catches_a_mistyped_route_target():
    mind = quiet()
    mind.add(Echoer("assistant"))
    mind.wire("assistant", "thought", "intercepter")  # typo
    problems = mind.validate()
    mind.close()
    assert len(problems) == 1
    assert "intercepter" in problems[0]
    assert "assistant" in problems[0], "the message should list the known modules"


def test_validate_catches_a_mistyped_observer():
    mind = quiet()
    mind.add(Echoer("a"))
    mind.watch("blackbord")
    problems = mind.validate()
    mind.close()
    assert len(problems) == 1 and "blackbord" in problems[0]


def test_validate_catches_a_bad_entry():
    mind = quiet()
    mind.add(Echoer("a"))
    mind.entry = "nobody"
    problems = mind.validate()
    mind.close()
    assert len(problems) == 1 and "nobody" in problems[0]


def test_validate_accepts_world_and_wildcard():
    mind = quiet()
    mind.add(Echoer("a"), Echoer("b"))
    mind.wire("a", "reply", "world")
    mind.watch("b")
    problems = mind.validate()
    mind.close()
    assert problems == []


def test_run_refuses_to_start_a_mind_wired_wrong():
    async def run():
        mind = quiet()
        mind.add(Echoer("a"))
        mind.wire("a", "reply", "ghost")
        try:
            await mind.prompt("hi")
        except ValueError as exc:
            mind.close()
            return str(exc)
        mind.close()
        return None

    message = asyncio.run(run())
    assert message is not None, "a typo'd route must not fail silently"
    assert "wired wrong" in message and "ghost" in message


def test_validation_runs_once_not_every_tick():
    mind = quiet()
    mind.add(Echoer("a"))
    calls = {"n": 0}
    original = mind.validate

    def counting():
        calls["n"] += 1
        return original()

    mind.validate = counting

    async def run():
        await mind.prompt("one")
        await mind.prompt("two")

    asyncio.run(run())
    mind.close()
    assert calls["n"] == 1


# -- the model factory -----------------------------------------------------


def test_mind_model_goes_through_the_factory():
    seen = []

    def factory(spec, **kwargs):
        seen.append(spec)
        return get_llm(spec, **kwargs)

    mind = quiet(model_factory=factory)
    mind.model("echo:")
    mind.model("echo:")
    mind.close()
    assert seen == ["echo:", "echo:"]


def test_taped_makes_a_whole_mind_reproducible():
    """One factory argument tapes every agent, not one model at a time."""
    counter = {"n": 0}

    def jittery(messages, opts):
        counter["n"] += 1
        return f"answer {counter['n']}"

    async def run(tape, mode):
        mind = Mind(
            "t", run_dir=None, console=False, model_factory=taped(tape, mode=mode)
        )
        mind.add(
            Agent("a", mind.model("echo:", rule=jittery), reply_to=None),
            Agent("b", mind.model("echo:", rule=jittery), reply_to=None),
        )
        mind.wire("a", "reply", "world")
        mind.wire("b", "reply", "world")
        out = texts(await mind.prompt("hello", to=["a", "b"]))
        mind.close()
        return out

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "tape.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        calls_after_record = counter["n"]
        replayed = asyncio.run(run(tape, "replay"))

    assert len(recorded) == 2
    assert replayed == recorded
    assert counter["n"] == calls_after_record, "replay must not call any model"


def test_taped_record_mode_does_not_clobber_across_models():
    """Two cassettes share one tape. The second must not erase the first."""
    counter = {"n": 0}

    def jittery(messages, opts):
        counter["n"] += 1
        return f"answer {counter['n']}"

    async def run(tape, mode):
        factory = taped(tape, mode=mode)
        a, b = factory("echo:", rule=jittery), factory("echo:", rule=jittery)
        from src import user as u

        return [
            (await a.chat([u("one")])).text,
            (await b.chat([u("two")])).text,
        ]

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "shared.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        lines = [ln for ln in tape.read_text().splitlines() if ln.strip()]
        replayed = asyncio.run(run(tape, "replay"))

    assert len(lines) == 2, f"both calls should be on the tape, found {len(lines)}"
    assert replayed == recorded
