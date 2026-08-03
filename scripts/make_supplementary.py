#!/usr/bin/env python3
"""Build the anonymous supplementary archive.

AAAI takes supplementary material three days after the paper, under the same
double-blind rules. This collects the code, the result files the paper's numbers
and figures are read from, and one recorded run, then zips them.

What goes in is an explicit list rather than everything minus exclusions, so a
file can only ship if it was named. What is left is scanned for anything that
identifies an author, and a single hit stops the build before the zip is
written.

    python scripts/make_supplementary.py              # code, results, replays
    python scripts/make_supplementary.py --with-raw   # add the per-trial records

Writes dist/supplementary.zip.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The code, copied whole. The paper travels on its own deadline and is not
#: here. Neither is this script, which spells out the strings it removes.
CODE = ("src", "minds", "studies", "tests", "examples", "ui",
        "README.md", "pyproject.toml")
SELF = Path("scripts/make_supplementary.py")

#: The result files. `studies/make_paper_numbers.py` and
#: `studies/make_paper_figures.py` read files with these names and no others,
#: so these reproduce every number, table and figure in the paper. The
#: per-trial records they were computed from are large and are added only by
#: --with-raw.
STUDIES = Path("out/studies")
RESULTS = {"summary.json", "meta.json", "crossing.json", "results.json",
           "verdict.json", "verdicts.json"}

#: The supplementary document, built by paper/build.sh. It is the appendix the
#: paper has no room for, and it ships as a PDF because a reviewer reads it
#: rather than rebuilds it.
DOCUMENT = (Path("paper/build/supplement.pdf"), "supplement.pdf")

#: One recorded run and the memory it wrote, so the claim that a run replays
#: without a model can be checked.
REPLAY = (Path("out/tapes"), Path("out/runs/replay-study"), Path("out/memory"))

#: Never shipped, wherever they appear. Caches and build products, all of which
#: the code rebuilds.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_NAMES = {".DS_Store"}
SKIP_SUFFIX = (".pyc", ".pyo")

#: What must not appear in any shipped byte. The home-directory pattern catches
#: absolute paths left in a script or a result file, which name the account
#: even when nothing else does.
FORBIDDEN = {
    "account name": re.compile(r"unruly ?abstractions", re.I),
    "personal name": re.compile(r"iansebas|ian sebastian", re.I),
    "email": re.compile(r"[\w.+-]+@(?:gmail|umich|[\w-]+\.edu)", re.I),
    "home directory": re.compile(r"/Users/[A-Za-z0-9_.-]+"),
    "code host": re.compile(r"(?:github|gitlab)\.com/[\w.-]+", re.I),
    "api key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
}


def under(root: Path, rel: Path, keep=lambda _: True) -> list[Path]:
    """Every wanted file below rel, as paths relative to root."""
    found = []
    for here, dirs, names in os.walk(root / rel):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if name in SKIP_NAMES or name.endswith(SKIP_SUFFIX):
                continue
            path = Path(here, name).relative_to(root)
            if path != SELF and keep(name):
                found.append(path)
    return found


def contents(root: Path, with_raw: bool) -> list[Path]:
    """Everything the archive ships, in one list."""
    files = []
    for entry in CODE:
        path = root / entry
        files += [Path(entry)] if path.is_file() else under(root, Path(entry))
    files += under(root, STUDIES, keep=lambda n: with_raw or n in RESULTS)
    for entry in REPLAY:
        files += under(root, entry)
    return sorted(set(files))


def leaks(root: Path, files: list[Path]) -> list[str]:
    """Every forbidden match, read from the bytes that would be shipped.

    Files are read as bytes and decoded loosely, so text carried in an image's
    metadata is scanned the same way a source file is.
    """
    found = []
    for rel in files:
        text = (root / rel).read_bytes().decode("utf-8", "ignore")
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN.items():
                hit = pattern.search(line)
                if hit:
                    found.append(f"{rel}:{number}: {label}: {hit.group(0)}")
    return found


NOTE = """\
# Supplementary material

The supplementary document, the runtime, the studies, and the result files
every number and figure is read from. The paper itself is not included.

`supplement.pdf` carries every scenario, every probe, every model and the full
result of each study. It is the material the paper reports on but has no room
to show.

    supplement.pdf   the appendix: scenarios, probes, models, full results
    src/       the runtime: `src/api` declares each part, `src/dminds` implements one
    minds/     one mind per file, each a composition the paper measures
    studies/   one script per experiment, and the scripts that write the paper's numbers
    tests/     the test suite, which needs no model and no network
    examples/  short scripts that run a mind and show what it did
    ui/        a browser view of a mind, live or replayed
    out/       the results, one recorded run, and the memory that run wrote

## Reproducing the paper without a model

    python tests/run_tests.py              # the suite, no dependencies
    python studies/make_paper_numbers.py   # every number and table
    python studies/make_paper_figures.py   # every figure, needs matplotlib

Both scripts read `out/studies/` and write into `paper/`, which they create.
Nothing else is needed, and no model is called.

## Reproducing a run

`examples/04_memory_and_replay.py` replays the recorded run in `out/runs/` from
the cassette in `out/tapes/`, with no model and no network.

## What is not here

The per-trial records each result was computed from are large and were left
out. The scripts under `studies/` regenerate them, and each one names the
open-weight checkpoint it needs, which is downloaded at first use.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "supplementary.zip")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--with-raw", action="store_true",
                        help="also ship the per-trial records, which are large")
    args = parser.parse_args()
    root = args.root.resolve()

    document = root / DOCUMENT[0]
    if not document.exists():
        print(f"  {DOCUMENT[0]} is missing. Run `bash paper/build.sh` first.")
        return 1

    files = contents(root, args.with_raw) + [DOCUMENT[0]]
    total = sum((root / f).stat().st_size for f in files)
    print(f"  {len(files)} files, {total / 1e6:.1f} MB before compression")

    found = leaks(root, files)
    if found:
        print(f"\n  {len(found)} identifying strings, nothing written:\n")
        for line in found[:40]:
            print(f"    {line}")
        if len(found) > 40:
            print(f"    ... and {len(found) - 40} more")
        return 1
    print(f"  scanned for {len(FORBIDDEN)} kinds of identifying string, none found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("supplementary/SUPPLEMENTARY.md", NOTE)
        for rel in files:
            name = DOCUMENT[1] if rel == DOCUMENT[0] else rel
            archive.write(root / rel, f"supplementary/{name}")

    # Read the archive back rather than trusting that it was written, and scan
    # again, so the check covers the bytes a reviewer will actually open.
    with zipfile.ZipFile(args.out) as archive:
        names = archive.namelist()
        again = [f"{name}: {label}"
                 for name in names
                 for label, pattern in FORBIDDEN.items()
                 if pattern.search(archive.read(name).decode("utf-8", "ignore"))]
    if again or len(names) != len(files) + 1:
        print(f"\n  the archive does not match what was scanned, so it is not usable")
        for line in again[:20]:
            print(f"    {line}")
        return 1

    print(f"  reopened the archive: {len(names)} entries, clean")
    print(f"\n  wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
