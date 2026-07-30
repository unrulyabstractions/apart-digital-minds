"""Dependency injection, membership, validation, and the model factory.

These are the tests that make the interfaces real rather than decorative. If
`Mind` constructed its own scheduler, the contract next door would describe
nothing.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from src import Agent, BaseModule, Ctx, Message, Mind, Text, get_llm, taped, texts
from src.api import Host
from src.api import Mind as MindInterface
from src.dminds import RunTracer, TickScheduler


def quiet(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Echoer(BaseModule):
    OUTPUTS = {"reply": "an answer"}

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        ctx.emit("reply", Text("ok"))


# -- the interface ---------------------------------------------------------


def test_mind_satisfies_both_its_interfaces():
    assert MindInterface in Mind.__mro__
    assert Host in Mind.__mro__


def test_host_is_narrower_than_mind():
    """A module gets a Host, which cannot add modules or drive the clock."""
    host_names = {n for n in dir(Host) if not n.startswith("_")}
    mind_names = {n for n in dir(MindInterface) if not n.startswith("_")}
    assert "stage" in host_names and "deliver" in host_names
    for name in ("add", "adopt", "run", "prompt", "validate"):
        assert name in mind_names, name
        assert name not in host_names, f"{name} should not be reachable from a module"


def test_host_has_no_routing():
    """Routing left the runtime entirely. Modules own their own links."""
    assert not hasattr(Host, "bus")
    assert not any("rout" in n.lower() for n in dir(Host))


# -- membership ------------------------------------------------------------


def test_registering_brings_both_modules_into_the_mind():
    """One verb. Wiring a module in is adding it."""
    mind = quiet()
    a, b = Echoer("a"), Echoer("b")
    a.register(mind.world, "reply")   # a joins: world is already here
    a.register(b, "reply")            # b joins: a is here now
    names = sorted(mind.modules)
    mind.close()
    assert names == ["a", "b"]


def test_first_module_to_join_becomes_the_entry():
    mind = quiet()
    a, b = Echoer("a"), Echoer("b")
    a.register(mind.world, "reply")
    a.register(b, "reply")
    entry = mind.entry
    mind.close()
    assert entry == "a"


def test_registering_two_orphans_says_what_to_do():
    mind = quiet()
    a, b = Echoer("a"), Echoer("b")
    try:
        a.register(b, "reply")
    except ValueError as exc:
        mind.close()
        assert "neither" in str(exc).lower()
        assert "mind.world" in str(exc)
    else:
        mind.close()
        raise AssertionError("expected a ValueError")


def test_add_is_only_needed_for_a_module_wired_to_nothing():
    class Loner(BaseModule):
        def __init__(self):
            super().__init__("loner")
            self.ran = 0

        def wants_process(self) -> bool:
            return self.ran < 1

        async def on_process(self, ctx: Ctx) -> None:
            self.ran += 1

    async def run():
        mind = quiet()
        loner = mind.add(Loner())
        await mind.run()
        mind.close()
        return loner.ran

    assert asyncio.run(run()) == 1


def test_adopting_the_same_module_twice_is_harmless():
    mind = quiet()
    a = Echoer("a")
    a.register(mind.world, "reply")
    mind.adopt(a)
    count = len(mind.modules)
    mind.close()
    assert count == 1


# -- injection -------------------------------------------------------------


def test_scheduler_can_be_injected():
    seen = {}

    class LoudScheduler(TickScheduler):
        async def tick(self):
            seen["ticked"] = seen.get("ticked", 0) + 1
            return await super().tick()

    async def run():
        mind = quiet(scheduler=lambda host: LoudScheduler(host, max_ticks=5))
        Echoer("a").register(mind.world, "reply")
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
    assert isinstance(mind.scheduler, TickScheduler)
    assert isinstance(mind.tracer, RunTracer)
    assert mind.model_factory is get_llm
    mind.close()


# -- validation ------------------------------------------------------------


def test_validate_catches_a_bad_entry():
    mind = quiet()
    Echoer("a").register(mind.world, "reply")
    mind.entry = "nobody"
    problems = mind.validate()
    mind.close()
    assert len(problems) == 1 and "nobody" in problems[0]


def test_a_clean_mind_validates():
    mind = quiet()
    a, b = Echoer("a"), Echoer("b")
    a.register(mind.world, "reply")
    a.register(b, "reply")
    problems = mind.validate()
    mind.close()
    assert problems == []


def test_validation_runs_once_not_every_tick():
    mind = quiet()
    Echoer("a").register(mind.world, "reply")
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
        for name in ("a", "b"):
            Agent(name, mind.model("echo:", rule=jittery)).register(
                mind.world, "reply"
            )
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


def test_taped_shares_one_cursor_across_models():
    """Two agents, same model, same question. Each must get its own answer."""
    counter = {"n": 0}

    def jittery(messages, opts):
        counter["n"] += 1
        return f"answer {counter['n']}"

    async def run(tape, mode):
        from src import user as u

        factory = taped(tape, mode=mode)
        a, b = factory("echo:", rule=jittery), factory("echo:", rule=jittery)
        return [(await a.chat([u("same")])).text, (await b.chat([u("same")])).text]

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "shared.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        lines = [ln for ln in tape.read_text().splitlines() if ln.strip()]
        replayed = asyncio.run(run(tape, "replay"))

    assert len(lines) == 2, f"both calls should be on the tape, found {len(lines)}"
    assert recorded[0] != recorded[1], "the two calls differ"
    assert replayed == recorded, "and replay must keep them in order"
