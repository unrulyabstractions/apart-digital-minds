"""Provider specs, the echo stand-in, and the cassette."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dmind import Cassette, GenOptions, get_llm, user
from dmind.llm import CassetteMiss, parse_spec


def test_parse_spec_splits_on_the_first_colon_only():
    assert parse_spec("openai:gpt-5") == ("openai", "gpt-5")
    assert parse_spec("ollama:qwen3:8b") == ("ollama", "qwen3:8b")
    assert parse_spec("hf:Qwen/Qwen3-4B-Instruct-2507") == (
        "hf",
        "Qwen/Qwen3-4B-Instruct-2507",
    )


def test_bare_provider_uses_its_default_model():
    provider, model = parse_spec("openai:")
    assert provider == "openai" and model == "gpt-5"


def test_aliases_expand():
    assert parse_spec("qwen") == ("ollama", "qwen3:8b")
    assert parse_spec("gemma") == ("ollama", "gemma3:4b")
    assert parse_spec("claude")[0] == "anthropic"


def test_unknown_provider_names_the_alternatives():
    try:
        get_llm("notaprovider:x")
    except ValueError as exc:
        assert "openai" in str(exc) and "register_provider" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_spec_without_a_provider_is_rejected():
    try:
        parse_spec("gpt-5")
    except ValueError as exc:
        assert "provider prefix" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_lazy_import_means_unconfigured_providers_still_construct():
    """Building an OpenAI model must not need a key. Only calling it does."""
    llm = get_llm("openai:gpt-5", api_key=None)
    assert llm.spec == "openai:gpt-5"


def test_echo_is_deterministic():
    async def run():
        llm = get_llm("echo:")
        first = await llm.chat([user("hello")])
        llm2 = get_llm("echo:")
        second = await llm2.chat([user("hello")])
        return first.text, second.text

    first, second = asyncio.run(run())
    assert first == second


def test_echo_script_is_consumed_in_order():
    async def run():
        llm = get_llm("echo:", script=["one", "two"])
        return [
            (await llm.chat([user("x")])).text,
            (await llm.chat([user("x")])).text,
            (await llm.chat([user("x")])).text,
        ]

    out = asyncio.run(run())
    assert out[:2] == ["one", "two"]
    assert "exhausted" in out[2]


def test_completion_is_timed_and_named():
    async def run():
        return await get_llm("echo:").chat([user("hi")])

    completion = asyncio.run(run())
    assert completion.latency_s >= 0
    assert completion.model == "echo"


def test_cassette_records_then_replays():
    counter = {"n": 0}

    def counting(messages, opts):
        counter["n"] += 1
        return f"call {counter['n']}"

    async def run(tape, mode):
        llm = Cassette(get_llm("echo:", rule=counting), tape, mode=mode)
        return (await llm.chat([user("hello")])).text

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "t.jsonl"
        recorded = asyncio.run(run(tape, "record"))
        replayed = asyncio.run(run(tape, "replay"))
        assert recorded == replayed == "call 1"
        assert counter["n"] == 1, "replay must not call the model"


def test_cassette_replay_miss_is_explicit():
    async def run(tape):
        llm = Cassette(get_llm("echo:"), tape, mode="replay")
        return await llm.chat([user("never recorded")])

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "empty.jsonl"
        tape.write_text("")
        try:
            asyncio.run(run(tape))
        except CassetteMiss as exc:
            assert "Re-run with mode='auto'" in str(exc)
        else:
            raise AssertionError("expected a CassetteMiss")


def test_cassette_distinguishes_repeated_identical_calls():
    """Same request twice, different answers, replayed in the right order."""
    counter = {"n": 0}

    def counting(messages, opts):
        counter["n"] += 1
        return f"call {counter['n']}"

    async def two(tape, mode):
        llm = Cassette(get_llm("echo:", rule=counting), tape, mode=mode)
        return [
            (await llm.chat([user("same")])).text,
            (await llm.chat([user("same")])).text,
        ]

    with tempfile.TemporaryDirectory() as tmp:
        tape = Path(tmp) / "t.jsonl"
        recorded = asyncio.run(two(tape, "record"))
        replayed = asyncio.run(two(tape, "replay"))
    assert recorded == ["call 1", "call 2"]
    assert replayed == recorded


def test_options_reach_the_provider():
    seen = {}

    def capture(messages, opts):
        seen.update(temperature=opts.temperature, max_tokens=opts.max_tokens)
        return "ok"

    async def run():
        llm = get_llm("echo:", rule=capture)
        await llm.chat([user("x")], GenOptions(temperature=0.1, max_tokens=7))

    asyncio.run(run())
    assert seen == {"temperature": 0.1, "max_tokens": 7}
