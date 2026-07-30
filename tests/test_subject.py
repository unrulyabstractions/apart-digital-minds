"""The subject: the target model a mind is built around, and how you drive it."""

from __future__ import annotations

import asyncio

from src import Ctx, Message, Mind, Subject, Text, get_llm, texts
from src.api import Agent as AgentInterface
from src.api import Module


def quiet(**kwargs) -> Mind:
    return Mind("test", run_dir=None, console=False, **kwargs)


# -- the subject exists and is wired --------------------------------------------


def test_a_mind_with_a_model_has_a_subject():
    mind = quiet(model="echo:")
    subject = mind.subject
    mind.close()
    assert isinstance(subject, Subject)
    assert subject.name == "subject"


def test_a_mind_without_a_model_has_no_subject():
    mind = quiet()
    assert mind.subject is None
    mind.close()


def test_subject_is_a_module_and_an_agent():
    assert Module in Subject.__mro__
    assert AgentInterface in Subject.__mro__


def test_the_subject_is_the_entry_and_speaks_without_wiring():
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
    assert entry == "subject"
    assert links == ["subject --reply--> world"]


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


# -- what the subject publishes -------------------------------------------------


def test_the_subject_publishes_context_reply_and_thought():
    assert set(Subject.OUTPUTS) == {"context", "reply", "thought"}


def test_context_carries_the_whole_window():
    seen = {}

    class Watcher(Subject):
        pass

    from src import BaseModule

    class Reader(BaseModule):
        async def on_input(self, message: Message, ctx: Ctx) -> None:
            seen["n"] = len(message.payload.messages)
            seen["note"] = message.payload.note

    async def run():
        mind = quiet(model=get_llm("echo:", script=["ok"]), system="rules")
        mind.subject.register(Reader("reader"), "context")
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
        mind.subject.register(Reader("reader"), "thought")
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


def test_the_subject_cannot_consume_what_it_produces():
    """Atomic channels. No name is both an input and an output, so no cycle."""
    assert set(Subject.INPUTS) & set(Subject.OUTPUTS) == set()
    assert set(Subject.INPUTS) == {"prompt"}


def test_a_stage_edits_the_context_on_its_way_to_the_ego():
    """The ego speaks from what it was handed, never from the original."""
    from src import BaseModule, Context, user

    class Rewriter(BaseModule):
        INPUTS = {"context": "the subject's window"}
        OUTPUTS = {"revised": "a replacement"}

        async def on_input(self, message: Message, ctx: Ctx) -> None:
            ctx.emit("revised", Context([user("a totally different history")]))

    async def run():
        mind = quiet(
            model=get_llm("echo:", script=["what the subject thought"]),
            ego=get_llm("echo:", script=["what the ego said"]),
        )
        mind.intercept(Rewriter("rewriter"))
        mind.prompt("hi")
        await mind.process()
        subject_history = [m.content for m in mind.subject.transcript]
        ego_history = [m.content for m in mind.ego.transcript]
        out = texts(mind.get_replies())
        mind.close()
        return subject_history, ego_history, out

    subject_history, ego_history, out = asyncio.run(run())
    assert "hi" in subject_history, "the subject kept its own window"
    assert ego_history[0] == "a totally different history", "the ego got the edit"
    assert out == ["what the ego said"], "the reply comes from the ego"


def test_a_custom_subject_class_goes_at_the_centre():
    class Terse(Subject):
        OUTPUTS = {"reply": "just the answer"}

    mind = quiet(model="echo:", subject=Terse)
    subject = mind.subject
    mind.close()
    assert isinstance(subject, Terse)
    assert subject.name == "subject"


def test_a_subject_factory_is_also_accepted():
    class Named(Subject):
        pass

    mind = quiet(model="echo:", subject=lambda llm: Named("subject", llm, system="via factory"))
    content = mind.subject.transcript[0].content
    mind.close()
    assert content == "via factory"


# -- driving -----------------------------------------------------------------


def test_prompt_does_not_run_the_mind():
    mind = quiet(model="echo:")
    mind.prompt("hi")
    assert mind.scheduler.t == 0, "prompt only delivers"
    assert mind.subject.pending == 1, "and the subject is holding it"
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


# -- the ego and the pipeline -------------------------------------------------


def test_without_an_ego_the_subject_speaks():
    mind = quiet(model="echo:")
    links = [ln.describe() for ln in mind.links()]
    ego = mind.ego
    mind.close()
    assert ego is None
    assert links == ["subject --reply--> world"]


def test_with_an_ego_the_reply_comes_from_the_ego():
    async def run():
        mind = quiet(
            model=get_llm("echo:", script=["the subject's answer"]),
            ego=get_llm("echo:", script=["the ego's answer"]),
        )
        mind.prompt("hi")
        await mind.process()
        out = texts(mind.get_replies())
        links = [ln.describe() for ln in mind.links()]
        mind.close()
        return out, links

    out, links = asyncio.run(run())
    assert out == ["the ego's answer"], "the subject's reply is not what reaches you"
    assert links == ["subject --context--> ego", "ego --reply--> world"]


def test_intercept_lays_out_subject_then_stages_then_ego():
    from src import BaseModule

    class Stage(BaseModule):
        INPUTS = {"context": "in"}
        OUTPUTS = {"revised": "out"}

    mind = quiet(model="echo:", ego="echo:")
    mind.intercept(Stage("one"), Stage("two"))
    links = sorted(ln.describe() for ln in mind.links())
    stages = [m.name for m in mind.stages]
    mind.close()
    assert stages == ["one", "two"]
    assert links == sorted([
        "subject --context--> one",
        "one --revised--> two as context",
        "two --revised--> ego as context",
        "ego --reply--> world",
    ])


def test_intercepting_again_discards_the_previous_layout():
    from src import BaseModule

    class Stage(BaseModule):
        INPUTS = {"context": "in"}
        OUTPUTS = {"revised": "out"}

    mind = quiet(model="echo:", ego="echo:")
    mind.intercept(Stage("one"))
    mind.intercept(Stage("two"))
    links = sorted(ln.describe() for ln in mind.links())
    mind.close()
    assert "one" not in " ".join(links), "the old layout should be gone"
    assert links == sorted([
        "subject --context--> two",
        "two --revised--> ego as context",
        "ego --reply--> world",
    ])


def test_a_stage_joins_the_mind_by_being_intercepted():
    from src import BaseModule

    class Stage(BaseModule):
        INPUTS = {"context": "in"}
        OUTPUTS = {"revised": "out"}

    mind = quiet(model="echo:", ego="echo:")
    mind.intercept(Stage("middle"))
    names = sorted(mind.modules)
    mind.close()
    assert names == ["ego", "middle", "subject"]


def test_the_ego_reads_context_and_writes_reply():
    from src import Ego

    assert set(Ego.INPUTS) == {"context"}
    assert set(Ego.OUTPUTS) == {"reply"}
    assert set(Ego.INPUTS) & set(Ego.OUTPUTS) == set()


def test_relayout_keeps_wiring_it_did_not_create():
    """A monitor attached by hand must survive a pipeline change."""
    from src import BaseModule

    class Stage(BaseModule):
        INPUTS = {"context": "in"}
        OUTPUTS = {"revised": "out"}

    class Spy(BaseModule):
        pass

    mind = quiet(model="echo:", ego="echo:")
    mind.subject.register(Spy("spy"), "*")
    mind.intercept(Stage("one"))
    mind.intercept(Stage("two"))
    links = [ln.describe() for ln in mind.links()]
    mind.close()
    assert any("spy" in ln for ln in links), "the hand-made link was destroyed"


def test_intercept_is_exactly_the_register_calls_it_documents():
    """No hidden wiring. `intercept` is shorthand, and this pins it down."""
    from src import BaseModule

    class Stage(BaseModule):
        INPUTS = {"context": "in"}
        OUTPUTS = {"revised": "out"}

    laid_out = quiet(model="echo:", ego="echo:")
    laid_out.intercept(Stage("interceptor"))
    automatic = sorted(ln.describe() for ln in laid_out.links())
    laid_out.close()

    by_hand = quiet(model="echo:", ego="echo:", autowire=False)
    stage = Stage("interceptor")
    by_hand.subject.register(stage, "context")
    stage.register(by_hand.ego, "revised", as_channel="context")
    by_hand.ego.register(by_hand.world, "reply")
    manual = sorted(ln.describe() for ln in by_hand.links())
    by_hand.close()

    assert automatic == manual


def test_describe_marks_which_links_the_mind_laid_out():
    from src import BaseModule

    class Spy(BaseModule):
        pass

    mind = quiet(model="echo:", ego="echo:")
    mind.subject.register(Spy("spy"), "*")
    text = mind.describe()
    mind.close()
    marked = [ln for ln in text.splitlines() if "[auto]" in ln]
    unmarked = [ln for ln in text.splitlines() if "spy" in ln]
    assert len(marked) == 2, "subject -> ego and ego -> world were laid out"
    assert unmarked and "[auto]" not in unmarked[0], "the hand-made link is not"


def test_a_stage_never_consumes_what_it_produces():
    """The atomic-channel rule, pinned for every shipped part and example."""
    import importlib.util, pathlib, sys

    from src import Ego, Subject

    sys.path.insert(0, str(pathlib.Path("examples").resolve()))
    for cls in (Subject, Ego):
        assert set(cls.INPUTS) & set(cls.OUTPUTS) == set(), cls.__name__

    for path in sorted(pathlib.Path("examples").glob("0*.py")):
        spec = importlib.util.spec_from_file_location(f"ex_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, "OUTPUTS") and obj.OUTPUTS:
                overlap = set(obj.INPUTS) & set(obj.OUTPUTS)
                assert overlap == set(), f"{path.name}:{name} reuses {overlap}"
