"""Memory stores, think-tag helpers, and the Agent defaults."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dmind import (
    Agent,
    Journal,
    Mind,
    Scratchpad,
    Transcript,
    assistant,
    get_llm,
    has_think,
    replace_think,
    split_think,
    strip_think,
    system,
    user,
)


# -- transcript ------------------------------------------------------------


def test_window_always_keeps_system_messages():
    t = Transcript([system("rules"), user("a"), assistant("b"), user("c")])
    window = t.window(2)
    assert window[0].role == "system"
    assert [m.content for m in window[1:]] == ["b", "c"]


def test_replace_all_swaps_the_whole_history():
    t = Transcript([system("rules"), user("a")])
    t.replace_all([user("edited")])
    assert len(t) == 1 and t[0].content == "edited"


def test_replace_all_copies_so_the_source_is_not_aliased():
    source = [user("a")]
    t = Transcript()
    t.replace_all(source)
    t[0].content = "changed"
    assert source[0].content == "a"


def test_tagged_finds_messages_by_meta():
    t = Transcript([user("a"), assistant("b", stage="draft"), assistant("c")])
    assert [m.content for m in t.tagged("stage", "draft")] == ["b"]


def test_edit_text_keeps_role_and_meta():
    t = Transcript([assistant("hello", stage="draft")])
    t.edit_text(0, str.upper)
    assert t[0].content == "HELLO"
    assert t[0].role == "assistant"
    assert t[0].meta["stage"] == "draft"


# -- scratchpad -------------------------------------------------------------


def test_scratchpad_bump_and_render():
    s = Scratchpad(mood="flat")
    s.bump("count")
    s.bump("count", 2)
    assert s["count"] == 3
    assert "mood: flat" in s.as_text()


# -- journal ---------------------------------------------------------------


def test_journal_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        first = Journal(path=path)
        first.remember("octopuses have nine brains", source="a")
        second = Journal(path=path)
        assert len(second) == 1
        assert second.episodes[0].source == "a"


def test_recall_prefers_overlap_then_recency():
    j = Journal()
    j.remember("cats are aloof")
    j.remember("octopuses are curious")
    j.remember("octopuses solve puzzles")
    hits = j.recall("octopuses", k=2)
    assert len(hits) == 2
    assert all("octopus" in e.text for e in hits)
    # Ties on overlap break toward the more recent memory.
    assert hits[0].text == "octopuses solve puzzles"


def test_recall_returns_nothing_when_nothing_matches():
    j = Journal()
    j.remember("cats are aloof")
    assert j.recall("quantum chromodynamics") == []


def test_custom_scorer_replaces_the_default():
    j = Journal(scorer=lambda q, e: len(e.text))
    j.remember("short")
    j.remember("a much longer memory")
    assert j.recall("anything", k=1)[0].text == "a much longer memory"


# -- think tags -------------------------------------------------------------


def test_split_think_separates_thought_from_output():
    thoughts, visible = split_think("<think>plan</think>\nAnswer.")
    assert thoughts == ["plan"]
    assert visible == "Answer."


def test_split_think_handles_several_blocks_and_newlines():
    thoughts, visible = split_think("<think>one\ntwo</think>mid<think>three</think>end")
    assert thoughts == ["one\ntwo", "three"]
    assert visible == "midend"


def test_replace_think_keeps_the_visible_text():
    out = replace_think("<think>old</think>\nAnswer.", "new")
    assert "<think>new</think>" in out
    assert "Answer." in out
    assert "old" not in out


def test_replace_think_prepends_when_there_is_no_block():
    out = replace_think("Just an answer.", "inserted")
    assert out.startswith("<think>inserted</think>")
    assert "Just an answer." in out


def test_replace_think_targets_one_block_by_index():
    out = replace_think("<think>a</think><think>b</think>", "B", index=1)
    assert "<think>a</think>" in out and "<think>B</think>" in out


def test_think_helpers_on_text_without_tags():
    assert not has_think("plain")
    assert strip_think("plain") == "plain"
    assert split_think("plain") == ([], "plain")


# -- agent -----------------------------------------------------------------


def test_agent_default_handler_answers_the_world():
    async def run():
        mind = Mind("t", run_dir=None, console=False)
        mind.add(Agent("a", get_llm("echo:", script=["hi there"]), system="be terse"))
        replies = await mind.prompt("hello")
        agent = mind.modules["a"]
        mind.close()
        return replies, agent

    replies, agent = asyncio.run(run())
    assert replies[0].payload.text == "hi there"
    # system, user, assistant
    assert [m.role for m in agent.transcript] == ["system", "user", "assistant"]


def test_max_context_limits_what_is_sent_not_what_is_kept():
    seen = {}

    def capture(messages, opts):
        seen["n"] = len(messages)
        return "ok"

    async def run():
        mind = Mind("t", run_dir=None, console=False)
        agent = Agent(
            "a", get_llm("echo:", rule=capture), system="rules", max_context=2
        )
        mind.add(agent)
        for _ in range(4):
            await mind.prompt("hello")
        mind.close()
        return seen["n"], len(agent.transcript)

    sent, kept = asyncio.run(run())
    assert sent == 3, "one system plus the last two messages"
    assert kept == 9, "the transcript keeps everything"


def test_memory_writes_land_in_the_agents_own_log():
    async def run():
        mind = Mind("t", run_dir=None, console=False)
        mind.add(Agent("a", get_llm("echo:")))
        await mind.prompt("hello")
        events = [
            e for e in mind.events.events if e.kind == "memory.write" and e.module == "a"
        ]
        mind.close()
        return events

    events = asyncio.run(run())
    assert len(events) == 2  # the user turn and the reply
    assert all(e.data["store"] == "transcript" for e in events)
