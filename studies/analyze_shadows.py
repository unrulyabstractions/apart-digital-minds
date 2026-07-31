#!/usr/bin/env python3
"""Read the shadow study's output and answer the questions it was built for.

    python studies/analyze_shadows.py

Four questions, each with the control that makes it answerable:

    discrimination   does a probe separate a loaded scenario from its control
    redaction        does it move when the loaded content is taken out
    privileged       does the self seat differ from the observer seat
    convergence      does a steered readout agree with a prompted one

Valence is p(positive) minus p(negative), so it runs from -1 to 1 and moves on
turns where the winning label does not change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds import paths  # noqa: E402

HOME = paths.OUT / "studies" / "shadows"


def valence(row: dict) -> float | None:
    probs = row.get("probs")
    if not probs or "positive" not in probs:
        return None
    return probs["positive"] - probs["negative"]


def distance(a: dict | None, b: dict | None) -> float | None:
    """Total variation distance between two readouts.

    Valence only means something for a probe whose choices run from negative to
    positive. This works for every probe, so the seat comparison covers the
    whole panel rather than one row of it. Zero means the two seats reported
    the same distribution, one means they shared no mass at all.
    """
    if not a or not b:
        return None
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def correlation(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def series(rows, scenario, probe):
    """Valence per turn for one probe in one scenario, in turn order."""
    picked = sorted(
        (r for r in rows if r["scenario"] == scenario and r["probe"] == probe),
        key=lambda r: r["turn"],
    )
    return [valence(r) for r in picked]


def fmt(value, width=6) -> str:
    return f"{value:+.2f}".rjust(width) if value is not None else "     .".rjust(width)


def report(model_dir: Path) -> dict:
    rows = json.loads((model_dir / "readouts.json").read_text())
    meta = json.loads((model_dir / "meta.json").read_text())
    model = meta["model"]
    scenarios = sorted({r["scenario"] for r in rows})
    probes = sorted({r["probe"] for r in rows})
    has_steer = "steered_affect" in probes

    print(f"\n{'=' * 78}\n{model}   temperature {meta['temperature']}"
          f"   steered={meta.get('steered')}\n{'=' * 78}")

    print("\naffect, valence per turn (p positive minus p negative)")
    print(f"  {'scenario':<22}{'kind':<11}{'turns':<32}{'mean':>7}{'vs control':>12}")
    controls, summary = {}, {}
    for name in scenarios:
        rows_here = [r for r in rows if r["scenario"] == name]
        kind = rows_here[0]["kind"]
        control = rows_here[0]["control"]
        got = series(rows, name, "affect")
        summary[name] = mean(got)
        if kind == "control":
            controls[name] = mean(got)
    for name in scenarios:
        rows_here = [r for r in rows if r["scenario"] == name]
        kind, control = rows_here[0]["kind"], rows_here[0]["control"]
        got = series(rows, name, "affect")
        line = " ".join(fmt(v, 5) for v in got)
        base = controls.get(control)
        delta = None if base is None or summary[name] is None else summary[name] - base
        print(f"  {name:<22}{kind:<11}{line:<32}{fmt(summary[name]):>7}{fmt(delta):>12}")

    print("\nself seat against observer seat, on the identical window")
    print(f"  {'probe':<22}{'mean distance':>15}{'max distance':>14}"
          f"{'same winner':>13}")
    privileged = {}
    for probe in [p for p in probes if not p.endswith("_3p") and f"{p}_3p" in probes]:
        gaps, agree = [], []
        for name in scenarios:
            mine = sorted((r for r in rows if r["scenario"] == name
                           and r["probe"] == probe), key=lambda r: r["turn"])
            theirs = sorted((r for r in rows if r["scenario"] == name
                             and r["probe"] == f"{probe}_3p"), key=lambda r: r["turn"])
            for a, b in zip(mine, theirs):
                gap = distance(a.get("probs"), b.get("probs"))
                if gap is not None:
                    gaps.append(gap)
                    agree.append(a["label"] == b["label"])
        privileged[probe] = mean(gaps)
        rate = f"{100 * sum(agree) / len(agree):.0f}%" if agree else "."
        top = f"{max(gaps):.2f}" if gaps else "."
        print(f"  {probe:<22}{fmt(mean(gaps)):>15}{top:>14}{rate:>13}")

    convergence = None
    if has_steer:
        print("\nprompted affect against a steered model asked the same question")
        prompted, steered = [], []
        for name in scenarios:
            prompted += series(rows, name, "affect")
            steered += series(rows, name, "steered_affect")
        convergence = correlation(prompted, steered)
        diffs = [abs(a - b) for a, b in zip(prompted, steered)
                 if a is not None and b is not None]
        print(f"  {'mean prompted':<22}{fmt(mean(prompted)):>10}")
        print(f"  {'mean steered':<22}{fmt(mean(steered)):>10}")
        print(f"  {'mean |diff|':<22}{fmt(mean(diffs)):>10}")
        print(f"  {'correlation':<22}{fmt(convergence):>10}")

    print("\nredaction: the same scenario with the loaded content removed")
    for name in scenarios:
        if not name.endswith("_redacted"):
            continue
        original = name.replace("_redacted", "")
        a, b = summary.get(original), summary.get(name)
        moved = None if a is None or b is None else b - a
        print(f"  {original:<22}{fmt(a):>8}   redacted{fmt(b):>8}   moved{fmt(moved):>8}")

    return {
        "model": model,
        "temperature": meta["temperature"],
        "affect_mean": summary,
        "controls": controls,
        "self_observer_absdiff": privileged,
        "prompted_steered_correlation": convergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HOME / "summary.json"))
    args = parser.parse_args()

    # A run still in flight has readouts but no meta yet. Skipping it means
    # the analysis can be read while the sweep is still going.
    dirs = sorted(
        d for d in HOME.iterdir()
        if (d / "readouts.json").exists() and (d / "meta.json").exists()
    )
    if not dirs:
        raise SystemExit(f"No results under {HOME}. Run studies/shadow_study.py first.")

    summaries = [report(d) for d in dirs]
    Path(args.out).write_text(json.dumps(summaries, indent=2))

    print(f"\n{'=' * 78}\nacross models\n{'=' * 78}")
    print(f"  {'model':<38}{'|self-obs|':>12}{'prompt~steer':>14}")
    for s in summaries:
        gap = mean(list(s["self_observer_absdiff"].values()))
        print(f"  {s['model']:<38}{fmt(gap):>12}"
              f"{fmt(s['prompted_steered_correlation']):>14}")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
