"""Routing, renaming on the wire, and observers."""

from __future__ import annotations

import asyncio

from src import Ctx, Mind, BaseModule, Task, Text


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Emitter(BaseModule):
    def __init__(self, name: str, kind: str, to=None):
        super().__init__(name)
        self.kind = kind
        self.to = to

    async def process(self, task: Task, ctx: Ctx) -> None:
        ctx.emit(self.kind, Text("payload"), to=self.to)


class Sink(BaseModule):
    def __init__(self, name: str):
        super().__init__(name)
        self.kinds: list[str] = []

    async def process(self, task: Task, ctx: Ctx) -> None:
        self.kinds.append(task.kind)


def drive(build) -> Mind:
    async def run():
        mind = quiet_mind()
        build(mind)
        mind.send("go", Text("x"), to="src")
        await mind.run()
        return mind

    return asyncio.run(run())


def test_wire_renames_the_kind_on_delivery():
    def build(mind):
        mind.add(Emitter("src", "thought"), Sink("dst"))
        mind.wire("src", "thought", "dst", as_kind="inspect")

    mind = drive(build)
    assert mind.modules["dst"].kinds == ["inspect"]
    mind.close()


def test_explicit_to_bypasses_routes():
    def build(mind):
        mind.add(Emitter("src", "thought", to="other"), Sink("dst"), Sink("other"))
        mind.wire("src", "thought", "dst")

    mind = drive(build)
    assert mind.modules["dst"].kinds == []
    assert mind.modules["other"].kinds == ["thought"]
    mind.close()


def test_observer_still_sees_explicitly_addressed_traffic():
    """The reason observers are not just wildcard routes."""

    def build(mind):
        mind.add(Emitter("src", "thought", to="other"), Sink("other"), Sink("spy"))
        mind.watch("spy")

    mind = drive(build)
    assert mind.modules["other"].kinds == ["thought"]
    assert mind.modules["spy"].kinds == ["go", "thought"]
    mind.close()


def test_observer_does_not_receive_its_own_emissions():
    def build(mind):
        mind.add(Emitter("src", "thought", to="sink"), Sink("sink"))
        mind.watch("src")

    mind = drive(build)
    assert mind.modules["sink"].kinds == ["thought"]
    mind.close()


def test_unrouted_emission_is_logged_not_raised():
    def build(mind):
        mind.add(Emitter("src", "thought"))

    mind = drive(build)
    dropped = [e for e in mind.events.events if e.data.get("dropped")]
    assert len(dropped) == 1
    assert "went nowhere" in dropped[0].data["summary"]
    mind.close()


def test_unknown_destination_is_dropped_with_a_trace():
    def build(mind):
        mind.add(Emitter("src", "thought", to="ghost"))

    mind = drive(build)
    dropped = [e for e in mind.events.events if e.data.get("dropped")]
    assert any("ghost" in e.data["summary"] for e in dropped)
    mind.close()


def test_fan_out_to_several_destinations():
    def build(mind):
        mind.add(Emitter("src", "thought", to=["a", "b"]), Sink("a"), Sink("b"))

    mind = drive(build)
    assert mind.modules["a"].kinds == ["thought"]
    assert mind.modules["b"].kinds == ["thought"]
    mind.close()


def test_handler_routing_by_kind():
    class Router(BaseModule):
        def __init__(self):
            super().__init__("r")
            self.calls: list[str] = []

        async def on_user_prompt(self, task, ctx):
            self.calls.append("user_prompt")

        async def on_inspect_context(self, task, ctx):
            self.calls.append("inspect_context")

        async def on_default(self, task, ctx):
            self.calls.append(f"default:{task.kind}")

    async def run():
        mind = quiet_mind()
        router = Router()
        mind.add(router)
        mind.send("user_prompt", Text("x"), to="r")
        mind.send("inspect.context", Text("x"), to="r")  # dots become underscores
        mind.send("unheard_of", Text("x"), to="r")
        await mind.run()
        mind.close()
        return router.calls

    assert asyncio.run(run()) == [
        "user_prompt",
        "inspect_context",
        "default:unheard_of",
    ]
