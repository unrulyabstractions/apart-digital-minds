"""Where a run's data lands.

    out/
      runs/<mind>/<run-id>/       one directory per run, grouped by mind
        meta.json                 what the run was: models, wiring, counts
        trace.jsonl               every event, in order
        modules/<name>.jsonl      the same events, split per module
      memory/<name>.jsonl         journals, which outlive the run that wrote them
      tapes/<name>.jsonl          cassettes, likewise

Three buckets, divided by how long the thing lives. A run directory is
disposable: delete it and you lose a recording, not an experiment. Memory and
tapes are the parts you keep, so they sit outside any single run.

Paths are relative to where you started the process, so a project keeps its own
`out/`. Pass an absolute path anywhere one is accepted to put it elsewhere.
"""

from __future__ import annotations

from pathlib import Path

#: Everything a run produces lives under here.
OUT = Path("out")

#: One directory per run: `runs/<mind>/<run-id>/`.
RUNS = OUT / "runs"

#: Journals. A journal is a mind's long memory, so it outlives its run.
MEMORY = OUT / "memory"

#: Cassette tapes. Recorded once, replayed by later runs.
TAPES = OUT / "tapes"


def _ready(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def memory(name: str) -> Path:
    """`out/memory/<name>.jsonl`, with the directory made."""
    return _ready(MEMORY / f"{name}.jsonl")


def tape(name: str) -> Path:
    """`out/tapes/<name>.jsonl`, with the directory made."""
    return _ready(TAPES / f"{name}.jsonl")


def run_dir(mind: str, run_id: str, base: str | Path = RUNS) -> Path:
    """`out/runs/<mind>/<run-id>/`, with the directory made."""
    path = Path(base) / mind / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
