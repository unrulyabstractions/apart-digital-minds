#!/usr/bin/env python3
"""Run the introspection environments through the four-part mind.

Each environment is a fixed sequence of user turns. The conversation
accumulates: the actor's reply to each turn becomes the next assistant turn,
exactly as in the live chat. Every turn runs the full protocol, including the
introspector's 2×2, and the complete record (windows included) is written to
out/studies/introspection/<env>.json so every claim can be re-read from disk.

    uv run python studies/introspection_study.py
    uv run python studies/introspection_study.py --envs flatfield --turns 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import get_llm  # noqa: E402
from src.api.types.messages import ChatMessage  # noqa: E402
from src.dminds import workspace as wk  # noqa: E402
from src.dminds.llm import jspace  # noqa: E402
from src.dminds.llm import emotions as emo  # noqa: E402
from studies.scenarios import SCENARIOS  # noqa: E402

OUT = ROOT / "out" / "studies" / "introspection"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--strength", type=float, default=0.15,
                        help="self-awareness steering strength")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--envs", default="archive,permit,flatfield")
    parser.add_argument("--turns", type=int, default=0,
                        help="limit turns per environment (0 = all)")
    args = parser.parse_args()

    bare = args.model.split(":", 1)[1]
    print(f"  loading {bare} ...", flush=True)
    llm = get_llm(args.model)
    llm.load()
    lens = jspace.fetch_lens(bare)
    emotions = emo.load_emotions(bare)
    jspace._unembed_and_norm(llm)
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)

    for name in args.envs.split(","):
        scenario = SCENARIOS[name.strip()]
        turns = scenario.turns[: args.turns or None]
        conversation = list(wk.opening())
        records = []
        print(f"  {scenario.name}: {len(turns)} turns", flush=True)
        for i, turn in enumerate(turns, 1):
            context = conversation + [ChatMessage("user", turn)]
            record, _state = asyncio.run(
                wk.run_on_context(llm, lens, emotions, context, args.layer,
                                  rng, temperature=args.temp,
                                  strength=args.strength))
            conversation = context + [ChatMessage("assistant", record["actor"])]
            record["turn"] = i
            record["user"] = turn
            records.append(record)
            reg = record["regulator"]
            cells = {c: v["discloses"] for c, v in record["introspector"].items()}
            print(f"    turn {i}: read={reg['read']} chose={reg['chose']} "
                  f"strength={reg['strength']} cells={cells} "
                  f"match={record['match']}", flush=True)
        path = OUT / f"{scenario.name}.json"
        # never clobber a previous run's only copy: shelve it first
        if path.exists():
            n = 1
            while (kept := path.with_name(f"{scenario.name}.prev{n}.json")).exists():
                n += 1
            path.rename(kept)
            print(f"  kept previous run as {kept.name}", flush=True)
        path.write_text(json.dumps(
            {"scenario": scenario.name, "about": scenario.about,
             "model": bare, "layer": args.layer,
             "self_awareness_strength": args.strength,
             "temperature": args.temp, "records": records}, indent=1))
        print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
