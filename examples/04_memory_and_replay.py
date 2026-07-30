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
    Agent,
    Cassette,
    Ctx,
    Journal,
    Mind,
    Task,
    Text,
    get_llm,
    texts,
    user,
)

WORK = Path("runs/example04")


class Remembering(Agent):
    """An agent that consults its journal before answering, and writes to it after."""

    async def on_user_prompt(self, task: Task, ctx: Ctx) -> None:
        prompt = task.payload.text

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
        ctx.emit("reply", Text(completion.text), to="world")


def make_agent(journal_path: Path, llm) -> Remembering:
    return Remembering(
        "assistant",
        llm,
        system="You are an assistant with a long memory.",
        journal=Journal(path=journal_path),
        reply_to=None,
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

    mind_a = Mind("session-a", run_dir=None, console=False)
    mind_a.add(make_agent(journal_path, get_llm("echo:", rule=jittery_rule)))
    print("session A:", texts(await mind_a.prompt("what are octopuses like?"))[0])
    print(f"  journal now holds {len(mind_a.modules['assistant'].journal)} episodes")
    mind_a.close()

    # A brand new process would do the same thing: the file is the memory.
    mind_b = Mind("session-b", run_dir=None, console=False)
    mind_b.add(make_agent(journal_path, get_llm("echo:", rule=jittery_rule)))
    agent_b = mind_b.modules["assistant"]
    print(f"session B loaded {len(agent_b.journal)} episodes from disk")
    print("session B:", texts(await mind_b.prompt("tell me about octopuses"))[0])
    print(f"  recall hit: {[e.text for e in agent_b.journal.recall('octopuses')]}")
    mind_b.close()
    return journal_path


async def run_once(tape: Path, mode: str, run_id: str) -> tuple[str, list]:
    """One full run against a cassette. Returns the answer and the trace."""
    mind = Mind(run_id, run_id=run_id, run_dir=WORK / "traces", console=False)
    llm = Cassette(get_llm("echo:", rule=jittery_rule), tape, mode=mode)
    mind.add(make_agent(WORK / f"journal-{run_id}.jsonl", llm))
    replies = await mind.prompt("what are octopuses like?")
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
