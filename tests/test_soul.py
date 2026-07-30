"""The soul: the target model a mind is built around, and how you drive it."""

from __future__ import annotations

import asyncio

from src import Ctx, Message, Mind, Soul, Text, get_llm, texts
from src.api import Agent as AgentInterface
from src.api import Module


def quiet(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


# -- the soul exists and is wired --------------------------------------------


def test_a_mind_with_a_model_has_a_soul():
    mind = quiet(model="echo:")
    soul = mind.soul
    mind.close()
    assert isinstance(soul, Soul)
    assert soul.name == "soul"


def test_a_mind_without_a_model_has_no_soul():
    mind = quiet()
    assert mind.soul is None
    mind.close()


def test_soul_is_a_module_and_an_agent():
    assert Module in Soul.__mro__
    assert AgentInterface in Soul.__mro__


def test_the_soul_is_the_entry_and_speaks_without_wiring():
    """No register call anywhere. A mind answers out of the box."""

    async def run():
        mind = quiet(model=get_llm("echo:", script=["hello back"]))
        mind.prompt("hi")
        await mind.process()
        out = texts(mind.get_replies())
        entry, links = mind.entry, [ln.describe() for ln in mind.links()]
        mind.close()
        return out, entry, links

    out, entry, links = asyncio.run(run())
    assert out == ["hello back"]
    assert entry == "soul"
    assert links == ["soul --reply--> world"]


def test_autowire_can_be_turned_off():
    mind = quiet(model="echo:", autowire=False)
    links = mind.links()
    mind.close()
    assert links == []


def test_a_built_model_can_be_passed_instead_of_a_spec():
    llm = get_llm("echo:", script=["from a built model"])

    async def run():
        mind = quiet(model=llm)
        mind.prompt("x")
        await mind.process()
        out = texts(mind.get_replies())
        mind.close()
        return out

    assert asyncio.run(run()) == ["from a built model"]


# -- what the soul publishes -------------------------------------------------


def test_the_soul_publishes_context_reply_and_thought():
    assert set(Soul.OUTPUTS) == {"context", "reply", "thought"}


def test_context_carries_the_whole_window():
    seen = {}

    class Watcher(Soul):
        pass

    from src import BaseModule

    class Reader(BaseModule):
        async def on_input(self, message: Message, ctx: Ctx) -> None:
            seen["n"] = len(message.payload.messages)
            seen["note"] = message.payload.note

    async def run():
        mind = quiet(model=get_llm("echo:", script=["ok"]), system="rules")
        mind.soul.register(Reader("reader"), "context")
        mind.prompt("hi")
        await mind.process()
        mind.close()

    asyncio.run(run())
    # system, user, assistant
    assert seen["n"] == 3
    assert "after answer" in seen["note"]


def test_thought_is_published_only_when_there_is_one():
    from src import BaseModule

    heard = []

    class Reader(BaseModule):
        async def on_input(self, message: Message, ctx: Ctx) -> None:
            heard.append(message.channel)

    async def run(script):
        mind = quiet(model=get_llm("echo:", script=script))
        mind.soul.register(Reader("reader"), "thought")
        mind.prompt("hi")
        await mind.process()
        mind.close()

    asyncio.run(run(["plain answer"]))
    assert heard == [], "no think block, no thought channel"
    asyncio.run(run(["<think>planning</think>answer"]))
    assert heard == ["thought"]


def test_reply_has_the_reasoning_stripped_out():
    async def run():
        mind = quiet(model=get_llm("echo:", script=["<think>secret</think>Visible."]))
        mind.prompt("hi")
        await mind.process()
        out = texts(mind.get_replies())
        mind.close()
        return out

    assert asyncio.run(run()) == ["Visible."]


def test_a_replacement_context_is_adopted_wholesale():
    from src import BaseModule, Context, user

    class Rewriter(BaseModule):
        OUTPUTS = {"context": "a replacement"}

        def __init__(self):
            super().__init__("rewriter")
            self.done = False

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            if self.done:
                return
            self.done = True
            ctx.emit("context", Context([user("a totally different history")]))

    async def run():
        mind = quiet(model=get_llm("echo:", script=["first", "second"]))
        rewriter = Rewriter()
        mind.soul.register(rewriter, "context")   # soul -> rewriter
        rewriter.register(mind.soul, "context")   # and back again
        mind.prompt("hi")
        await mind.process()
        history = [m.content for m in mind.soul.transcript]
        out = texts(mind.get_replies())
        mind.close()
        return history, out

    history, out = asyncio.run(run())
    assert history[0] == "a totally different history", "the old window is gone"
    assert out == ["first", "second"], "it answered again after adopting"


def test_a_custom_soul_class_goes_at_the_centre():
    class Terse(Soul):
        OUTPUTS = {"reply": "just the answer"}

    mind = quiet(model="echo:", soul=Terse)
    soul = mind.soul
    mind.close()
    assert isinstance(soul, Terse)
    assert soul.name == "soul"


def test_a_soul_factory_is_also_accepted():
    class Named(Soul):
        pass

    mind = quiet(model="echo:", soul=lambda llm: Named("soul", llm, system="via factory"))
    content = mind.soul.transcript[0].content
    mind.close()
    assert content == "via factory"


# -- driving -----------------------------------------------------------------


def test_prompt_does_not_run_the_mind():
    mind = quiet(model="echo:")
    mind.prompt("hi")
    assert mind.scheduler.t == 0, "prompt only delivers"
    assert mind.soul.pending == 1, "and the soul is holding it"
    mind.close()


def test_process_one_runs_exactly_one_tick():
    async def run():
        mind = quiet(model=get_llm("echo:", script=["a"]))
        mind.prompt("hi")
        turns = await mind.process_one()
        settled = await mind.process_one()
        mind.close()
        return turns, settled

    turns, settled = asyncio.run(run())
    assert turns == 1
    assert settled == 0, "nothing left to do"


def test_process_runs_until_everyone_is_empty():
    from src import BaseModule

    class Relay(BaseModule):
        OUTPUTS = {"on": "pass it along"}

        def __init__(self, name, hops):
            super().__init__(name)
            self.hops = hops

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            if self.hops > 0:
                self.hops -= 1
                ctx.emit("on", Text("x"))

    async def run():
        mind = quiet()
        r = mind.add(Relay("r", hops=3))
        r.register(r, "on")
        mind.send("on", Text("x"), to="r")
        ticks = await mind.process()
        mind.close()
        return ticks

    assert asyncio.run(run()) == 4


def test_get_replies_drains():
    async def run():
        mind = quiet(model=get_llm("echo:", script=["one", "two"]))
        mind.prompt("a")
        await mind.process()
        first = texts(mind.get_replies())
        again = mind.get_replies()
        mind.prompt("b")
        await mind.process()
        second = texts(mind.get_replies())
        history = len(mind.outbox)
        mind.close()
        return first, again, second, history

    first, again, second, history = asyncio.run(run())
    assert first == ["one"]
    assert again == [], "reading drains"
    assert second == ["two"]
    assert history == 2, "outbox keeps everything"
