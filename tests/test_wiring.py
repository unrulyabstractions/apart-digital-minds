"""Registration: declared channels, renaming, wildcards, and what fails loudly.

Wiring lives on modules. The mind has no routing table, so these tests are
about what a module will and will not agree to.
"""

from __future__ import annotations

import asyncio

from src import BaseModule, Ctx, Message, Mind, Text, UndeclaredChannel


def quiet_mind(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


class Emitter(BaseModule):
    OUTPUTS = {"thought": "a thought", "reply": "an answer"}

    def __init__(self, name: str, channel: str = "thought"):
        super().__init__(name)
        self.channel = channel

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        ctx.emit(self.channel, Text("payload"))


class Listener(BaseModule):
    def __init__(self, name: str):
        super().__init__(name)
        self.heard: list[str] = []

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.heard.append(message.channel)


def drive(build) -> Mind:
    async def run():
        mind = quiet_mind()
        build(mind)
        mind.send("go", Text("x"), to="src")
        await mind.process()
        return mind

    return asyncio.run(run())


# -- declared channels -----------------------------------------------------


def test_registering_on_an_undeclared_channel_fails_immediately():
    mind = quiet_mind()
    src, dst = Emitter("src"), Listener("dst")
    mind.add(src, dst)
    try:
        src.register(dst, "thougth")  # typo
    except UndeclaredChannel as exc:
        mind.close()
        assert "thougth" in str(exc)
        assert "thought" in str(exc), "the error should list the real channels"
    else:
        mind.close()
        raise AssertionError("expected an UndeclaredChannel")


def test_emitting_on_an_undeclared_channel_fails():
    class Rogue(BaseModule):
        OUTPUTS = {"declared": "the only one"}

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            ctx.emit("undeclared", Text("x"))

    async def run():
        mind = quiet_mind()
        mind.add(Rogue("r"))
        mind.send("go", Text("x"), to="r")
        try:
            await mind.process()
        except ValueError as exc:
            mind.close()
            return str(exc)
        mind.close()
        return None

    message = asyncio.run(run())
    assert message is not None
    assert "undeclared" in message and "declared" in message


def test_outputs_are_readable_without_reading_the_body():
    assert Emitter.OUTPUTS == {"thought": "a thought", "reply": "an answer"}
    assert Emitter("x").declares("thought")
    assert not Emitter("x").declares("nope")


# -- delivery --------------------------------------------------------------


def test_register_delivers_on_the_named_channel():
    def build(mind):
        src, dst = Emitter("src"), Listener("dst")
        mind.add(src, dst)
        src.register(dst, "thought")

    mind = drive(build)
    assert mind.modules["dst"].heard == ["thought"]
    mind.close()


def test_as_channel_renames_on_the_wire():
    def build(mind):
        src, dst = Emitter("src"), Listener("dst")
        mind.add(src, dst)
        src.register(dst, "thought", as_channel="inspect")

    mind = drive(build)
    assert mind.modules["dst"].heard == ["inspect"]
    mind.close()


def test_nobody_registered_means_nobody_hears():
    def build(mind):
        mind.add(Emitter("src"), Listener("dst"))

    mind = drive(build)
    assert mind.modules["dst"].heard == []
    dropped = [e for e in mind.events.events if e.data.get("dropped")]
    assert any("no listeners" in e.data["summary"] for e in dropped)
    mind.close()


def test_one_channel_can_have_several_listeners():
    def build(mind):
        src, a, b = Emitter("src"), Listener("a"), Listener("b")
        mind.add(src, a, b)
        src.register(a, "thought")
        src.register(b, "thought")

    mind = drive(build)
    assert mind.modules["a"].heard == ["thought"]
    assert mind.modules["b"].heard == ["thought"]
    mind.close()


def test_wildcard_hears_every_channel_under_its_real_name():
    def build(mind):
        src, spy = Emitter("src", channel="reply"), Listener("spy")
        mind.add(src, spy)
        src.register(spy, "*")

    mind = drive(build)
    assert mind.modules["spy"].heard == ["reply"], "a wildcard hears the real name"
    mind.close()


def test_a_wildcard_listener_is_not_told_twice():
    def build(mind):
        src, spy = Emitter("src"), Listener("spy")
        mind.add(src, spy)
        src.register(spy, "thought")
        src.register(spy, "*")

    mind = drive(build)
    assert mind.modules["spy"].heard == ["thought"], "registered twice, told once"
    mind.close()


def test_a_module_can_register_itself():
    """Self-registration is how a module schedules its own next tick."""

    class SelfPoker(BaseModule):
        OUTPUTS = {"again": "a note to self"}

        def __init__(self, name: str):
            super().__init__(name)
            self.turns_taken = 0

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            self.turns_taken += 1
            if self.turns_taken < 3:
                ctx.emit("again", Text("x"))

    async def run():
        mind = quiet_mind()
        p = mind.add(SelfPoker("p"))
        p.register(p, "again")
        mind.send("go", Text("x"), to="p")
        await mind.process()
        mind.close()
        return p.turns_taken

    assert asyncio.run(run()) == 3


# -- validation ------------------------------------------------------------


def test_an_orphan_consumer_is_impossible():
    """Registering brings the consumer in, so it can never be left out."""
    mind = quiet_mind()
    src = Emitter("src")
    mind.add(src)
    src.register(Listener("newcomer"), "thought")
    names = sorted(mind.modules)
    problems = mind.validate()
    mind.close()
    assert names == ["newcomer", "src"], "the consumer joined by being wired"
    assert problems == []


def test_validate_still_catches_a_bad_entry():
    mind = quiet_mind()
    mind.add(Emitter("src"))
    mind.entry = "nobody"
    problems = mind.validate()
    mind.close()
    assert len(problems) == 1 and "nobody" in problems[0]


def test_world_is_always_a_valid_consumer():
    mind = quiet_mind()
    src = mind.add(Emitter("src"))
    src.register(mind.world, "reply")
    problems = mind.validate()
    mind.close()
    assert problems == []


def test_links_are_gathered_from_the_modules():
    mind = quiet_mind()
    a, b = Emitter("a"), Listener("b")
    mind.add(a, b)
    a.register(b, "thought", as_channel="inspect")
    links = mind.links()
    mind.close()
    assert len(links) == 1
    assert links[0].describe() == "a --thought--> b as inspect"


# -- default on_input dispatch --------------------------------------------


def test_on_input_routes_to_a_per_channel_method_when_one_exists():
    class Router(BaseModule):
        def __init__(self):
            super().__init__("r")
            self.calls: list[str] = []

        async def on_user_prompt(self, message, ctx):
            self.calls.append("user_prompt")

        async def on_inspect_context(self, message, ctx):
            self.calls.append("inspect_context")

    async def run():
        mind = quiet_mind()
        r = mind.add(Router())
        mind.send("user_prompt", Text("x"), to="r")
        mind.send("inspect.context", Text("x"), to="r")  # dots become underscores
        mind.send("unheard_of", Text("x"), to="r")
        await mind.process()
        mind.close()
        return r.calls, r.inputs

    calls, buffered = asyncio.run(run())
    assert calls == ["user_prompt", "inspect_context"]
    assert [m.channel for m in buffered] == ["unheard_of"], "the rest buffers"


def test_take_inputs_drains_the_buffer():
    class Batcher(BaseModule):
        def __init__(self):
            super().__init__("b")
            self.batches: list[int] = []

        async def on_process(self, ctx: Ctx) -> None:
            self.batches.append(len(self.take_inputs()))

    async def run():
        mind = quiet_mind()
        b = mind.add(Batcher())
        for _ in range(3):
            mind.send("x", Text("x"), to="b")
        await mind.process()
        mind.close()
        return b.batches, b.inputs

    batches, leftover = asyncio.run(run())
    assert batches == [3], "one turn absorbed all three"
    assert leftover == []
