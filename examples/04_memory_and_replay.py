"""Memory that outlives the process, and runs that reproduce exactly.

Three things, in order:

    1. A journal survives the mind that wrote it. Build a second mind over the
       same file and it remembers.
    2. A cassette pins a non-deterministic model. Record once, replay forever.
    3. Two replayed runs produce the same trace, event for event.

Point 3 is the payoff. The model call is the only place non-determinism can
enter this runtime, so capturing it at that boundary makes the whole run
reproducible, and two traces can be diffed to find what a change did.

Run it:
    python examples/04_memory_and_replay.py
"""

from __future__ import annotations

import asyncio
import random
import shutil
from pathlib import Path

from src import (
    Ctx,
    Journal,
    Message,
    Mind,
    Soul,
    Text,
    get_llm,
    taped,
    texts,
    user,
)

WORK = Path("runs/example04")


class Remembering(Soul):
    """An agent that consults its journal before answering, and writes to it after."""

    OUTPUTS = {"reply": "the answer, informed by what it remembered"}
    INPUTS = {"prompt": "a question, as Text"}

    def __init__(self, *args, journal_path=None, **kwargs):
        super().__init__(*args, journal=Journal(path=journal_path), **kwargs)

    async def on_prompt(self, message: Message, ctx: Ctx) -> None:
        prompt = message.payload.text

        recalled = self.journal.recall(prompt, k=3)
        if recalled:
            ctx.log.note(f"recalled {len(recalled)} episodes", query=prompt)
            context = "What you remember:\n" + self.journal.as_text(recalled)
        else:
            context = "You remember nothing relevant."

        completion = await self.think(
            messages=[*self.transcript.messages, user(f"{context}\n\nUser: {prompt}")],
            tag="answer",
        )

        self.journal.remember(
            f"user asked about {prompt}", t=ctx.tick, source=self.name
        )
        ctx.emit("reply", Text(completion.text))


def remembering_mind(journal_path: Path, llm, **kwargs) -> Mind:
    """A mind whose soul consults a journal before answering."""
    return Mind(
        model=llm,
        system="You are an assistant with a long memory.",
        soul=lambda built: Remembering(
            "soul", built, system="You are an assistant with a long memory.",
            journal_path=journal_path,
        ),
        console=False,
        **kwargs,
    )


def jittery_rule(messages, opts) -> str:
    """A model that answers differently every time. Nothing is seeded."""
    memory_line = next(
        (line for m in messages for line in m.content.splitlines() if line.startswith("- ")),
        None,
    )
    seen = " I recall we spoke before." if memory_line else ""
    return f"Answer #{random.randint(1000, 9999)}.{seen}"


async def part1_memory_survives() -> Path:
    print("=" * 70)
    print("1. a journal outlives its mind")
    print("=" * 70)

    journal_path = WORK / "journal.jsonl"

    mind_a = remembering_mind(journal_path, get_llm("echo:", rule=jittery_rule),
                              name="session-a", run_dir=None)
    mind_a.prompt("what are octopuses like?")
    await mind_a.process()
    print("session A:", texts(mind_a.get_replies())[0])
    print(f"  journal now holds {len(mind_a.soul.journal)} episodes")
    mind_a.close()

    # A brand new process would do the same thing: the file is the memory.
    mind_b = remembering_mind(journal_path, get_llm("echo:", rule=jittery_rule),
                              name="session-b", run_dir=None)
    print(f"session B loaded {len(mind_b.soul.journal)} episodes from disk")
    mind_b.prompt("tell me about octopuses")
    await mind_b.process()
    print("session B:", texts(mind_b.get_replies())[0])
    print(f"  recall hit: {[e.text for e in mind_b.soul.journal.recall('octopuses')]}")
    mind_b.close()
    return journal_path


async def run_once(tape: Path, mode: str, run_id: str) -> tuple[str, list]:
    """One full run against a tape. Returns the answer and the trace.

    The tape is attached with `model_factory`, not by wrapping a model by hand.
    Every model this mind builds goes through it, so a mind with twenty agents
    on five providers is made reproducible by this one argument.
    """
    factory = taped(tape, mode=mode)
    mind = remembering_mind(
        WORK / f"journal-{run_id}.jsonl",
        factory("echo:", rule=jittery_rule),
        name=run_id,
        run_id=run_id,
        run_dir=WORK / "traces",
        model_factory=factory,
    )
    mind.prompt("what are octopuses like?")
    await mind.process()
    replies = mind.get_replies()
    events = list(mind.events.events)
    mind.close()
    return texts(replies)[0], events


async def part2_replay(journal_path: Path) -> None:
    print()
    print("=" * 70)
    print("2. a cassette pins a non-deterministic model")
    print("=" * 70)

    # Without a cassette the answer moves every time.
    loose = [
        (await run_once(WORK / "throwaway.jsonl", "record", f"loose-{i}"))[0]
        for i in range(2)
    ]
    print("no cassette:", loose, "->", "differ" if loose[0] != loose[1] else "same")

    tape = WORK / "tape.jsonl"
    recorded, trace_1 = await run_once(tape, "record", "record")
    print(f"recorded   : {recorded}   (tape: {tape})")

    replayed, trace_2 = await run_once(tape, "replay", "replay-1")
    _, trace_3 = await run_once(tape, "replay", "replay-2")
    print(f"replayed   : {replayed}")
    print(f"match      : {recorded == replayed}")

    print()
    print("=" * 70)
    print("3. two replays produce the same event sequence")
    print("=" * 70)

    def shape(events):
        """The parts of a trace that are logic, not timing."""
        return [(e.tick, e.module, e.kind) for e in events]

    same = shape(trace_2) == shape(trace_3)
    print(f"events per run : {len(trace_2)} vs {len(trace_3)}")
    print(f"identical      : {same}")
    if not same:
        for a, b in zip(shape(trace_2), shape(trace_3)):
            if a != b:
                print(f"  first divergence: {a} != {b}")
                break

    # And it differs from the recording run only where you would expect.
    drift = sum(1 for a, b in zip(shape(trace_1), shape(trace_2)) if a != b)
    print(f"record vs replay differences: {drift}")


async def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    journal_path = await part1_memory_survives()
    await part2_replay(journal_path)

    print(f"\nartifacts under {WORK}/")


if __name__ == "__main__":
    asyncio.run(main())
