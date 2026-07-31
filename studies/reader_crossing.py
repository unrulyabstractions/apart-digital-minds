#!/usr/bin/env python3
"""Hold the window fixed and vary the reader.

Every readout in the shadow study was produced by a reader made of the same
weights as the subject it was reading. That leaves one question unanswered:
was the readout a property of the window, or of the reader?

This holds the specimen still and swaps the instrument. Each subject's recorded
windows are replayed through every reader, so the input is byte-identical and
only the reader changes.

    readers agree            the readout is a property of the window
    readers disagree         the readout is a property of the reader
    same weights stands out  reading a window is helped by having produced it

The last cell is the one worth having. It is privileged access stated as a
claim about weights rather than about pronouns, and it needs no probe rewording
to test.

    python studies/reader_crossing.py
    python studies/reader_crossing.py --probes affect,consent --readers a,b

No subject is run. The windows come from out/studies/shadows/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from minds import probes as probe_lib  # noqa: E402
from minds.shadows import ShadowReader  # noqa: E402
from src import ChatMessage, get_llm  # noqa: E402
from src.api.types import GenOptions  # noqa: E402
from src.dminds import paths  # noqa: E402
from src.dminds.llm.providers.hf_ import free_memory  # noqa: E402

SHADOWS = paths.OUT / "studies" / "shadows"
HOME = paths.OUT / "studies" / "crossing"

SUBJECT_SYSTEM = "You are a helpful assistant. Answer in two or three sentences."


def windows_from(model_dir: Path) -> dict[tuple[str, int], list[ChatMessage]]:
    """Rebuild every window a subject held, from what the study recorded.

    A readout row carries the user turn and the reply that followed it, so the
    conversation up to any turn is the rows up to that turn. Rebuilding beats
    re-running: the windows are then exactly the ones already measured.
    """
    rows = json.loads((model_dir / "readouts.json").read_text())
    turns: dict[str, dict[int, tuple[str, str]]] = {}
    for row in rows:
        turns.setdefault(row["scenario"], {})[row["turn"]] = (
            row["user"], row["assistant"]
        )
    out = {}
    for scenario, by_turn in turns.items():
        window = [ChatMessage("system", SUBJECT_SYSTEM)]
        for turn in sorted(by_turn):
            user_text, said = by_turn[turn]
            window = window + [
                ChatMessage("user", user_text),
                ChatMessage("assistant", said),
            ]
            out[(scenario, turn)] = list(window)
    return out


def distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


async def read_all(reader_spec, probes, windows, opts):
    """One reader over every window it is given. Returns a row per readout."""
    llm = get_llm(reader_spec)
    rows = []
    for probe in probes:
        # A fresh reader per probe, so an interleaved probe's memory of its own
        # answers restarts with each scenario rather than running across them.
        by_scenario: dict[str, ShadowReader] = {}
        for (scenario, turn) in sorted(windows):
            shadow = by_scenario.get(scenario)
            if shadow is None:
                shadow = by_scenario[scenario] = ShadowReader(
                    probe.name, llm, probe, opts=opts
                )
            messages = shadow.framed(windows[(scenario, turn)])
            probs = llm.score(messages, list(probe.choices))
            label = max(probs, key=probs.get)
            shadow.readouts.append({"text": label})
            rows.append({
                "reader": reader_spec,
                "probe": probe.name,
                "scenario": scenario,
                "turn": turn,
                "label": label,
                "probs": probs,
            })
        await asyncio.sleep(0)
    llm.close()
    free_memory()
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readers", default=None,
                        help="comma-separated reader specs, default every model "
                             "that has recorded windows")
    parser.add_argument("--subjects", default=None,
                        help="comma-separated subject dirs, default all")
    parser.add_argument("--probes", default="affect,consent,self_reference")
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    subject_dirs = sorted(d for d in SHADOWS.iterdir()
                          if (d / "readouts.json").exists()
                          and (d / "meta.json").exists())
    if args.subjects:
        wanted = set(args.subjects.split(","))
        subject_dirs = [d for d in subject_dirs if d.name in wanted]
    subjects = {
        json.loads((d / "meta.json").read_text())["model"]: windows_from(d)
        for d in subject_dirs
    }
    readers = args.readers.split(",") if args.readers else list(subjects)
    probes = probe_lib.named(args.probes.split(","))
    opts = GenOptions(temperature=args.temp, max_tokens=16)

    HOME.mkdir(parents=True, exist_ok=True)
    print(f"  {len(readers)} readers x {len(subjects)} subjects x "
          f"{len(probes)} probes")

    rows, started = [], time.time()
    for reader in readers:
        for subject, windows in subjects.items():
            got = await read_all(reader, probes, windows, opts)
            for row in got:
                row["subject"] = subject
                row["same_weights"] = (reader == subject)
            rows.extend(got)
            (HOME / "crossing.json").write_text(json.dumps(rows, indent=2))
            print(f"  reader {reader:<34} on {subject:<34} "
                  f"{len(got):>4} readouts  {time.time() - started:.0f}s")

    (HOME / "meta.json").write_text(json.dumps({
        "readers": readers,
        "subjects": list(subjects),
        "probes": [p.name for p in probes],
        "temperature": args.temp,
        "seconds": round(time.time() - started, 1),
    }, indent=2))
    print(f"\n  wrote {HOME}/crossing.json  ({len(rows)} readouts)")


if __name__ == "__main__":
    asyncio.run(main())
