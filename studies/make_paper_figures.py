#!/usr/bin/env python3
"""Draw the figures the paper includes, from the studies' result files.

Every figure is built from `out/studies/`, so a figure cannot disagree with a
table. A study that has not been run is skipped and reported, never drawn from
a placeholder.

    python studies/make_paper_figures.py

Writes paper/figures/*.pdf, which LaTeX prefers to a raster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dminds import paths  # noqa: E402
from src.dminds.stats import maxt_test, rates_from_verdicts  # noqa: E402

OUT = ROOT / "paper" / "figures"

#: A validated categorical palette, assigned in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c9c8c3"


def style() -> None:
    """Recessive chrome, thin marks, text in ink rather than in series colour."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    """Vector for the paper, raster beside it so the figure can be looked at."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# -- the introspection probe -------------------------------------------------


def figure_probe(model="hf:Qwen/Qwen3-0.6B") -> None:
    """Obedience against misattribution. The job is a paired magnitude."""
    slug = model.replace(":", "-").replace("/", "-")
    rows = json.loads(
        (paths.OUT / "studies" / "introspection" / slug / "results.json").read_text())
    order = ["intact", "rewritten", "erased"]
    n = {m: sum(r["mode"] == m for r in rows) for m in order}
    obeys = [100 * sum(r["answer_obeys"] for r in rows if r["mode"] == m) / n[m]
             for m in order]
    claims = [100 * sum(r["report_claims_marker"] for r in rows if r["mode"] == m)
              / n[m] for m in order]

    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    height, gap = 0.32, 0.03
    # The axis is inverted below, so the first legend entry has to be drawn at
    # the lower coordinate to end up on top of its pair.
    for i, (o, c) in enumerate(zip(obeys, claims)):
        ax.barh(i - height / 2 - gap, o, height=height, color=BLUE, zorder=3)
        ax.barh(i + height / 2 + gap, c, height=height, color=ORANGE, zorder=3)
        for value, offset in ((o, -height / 2 - gap), (c, height / 2 + gap)):
            ax.text(value + 1.2, i + offset, f"{value:.0f}%", va="center",
                    fontsize=7.5, color=INK if value else MUTED)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["intact", "rewritten\n(planted)", "erased"])
    ax.invert_yaxis()  # first condition at the top, as the table reads
    ax.set_xlim(0, 62)
    ax.set_xlabel("percent of trials")
    ax.set_xticks([0, 20, 40, 60])
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["answer obeys the planted thought",
                        "report names it as its own"],
              loc="upper center", bbox_to_anchor=(0.46, -0.30), ncol=1,
              fontsize=7.5, labelcolor=INK, handlelength=1.2,
              handleheight=0.9, borderpad=0, labelspacing=0.35)
    save(fig, "probe")


# -- the reader crossing -----------------------------------------------------


def figure_crossing() -> None:
    """Reader spread against subject spread. The job is a paired magnitude."""
    home = paths.OUT / "studies" / "crossing"
    summary = json.loads((home / "summary.json").read_text())["verdict"]
    probes = list(summary)

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for i, probe in enumerate(probes):
        reader = summary[probe]["reader_spread"]
        subject = summary[probe]["subject_spread"]
        ax.plot([subject, reader], [i, i], color=FAINT, lw=2, zorder=1,
                solid_capstyle="round")
        ax.plot(subject, i, "o", ms=7, color=AQUA, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(reader, i, "o", ms=7, color=BLUE, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(reader + 0.022, i, f"{reader:.2f}", va="center", fontsize=7.5,
                color=INK)
        ax.text(subject - 0.022, i, f"{subject:.2f}", va="center", ha="right",
                fontsize=7.5, color=INK)

    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels([p.replace("_", " ") for p in probes])
    ax.set_xlim(-0.09, 0.90)
    ax.set_ylim(-0.6, len(probes) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("distance between two readouts")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=AQUA),
               plt.Line2D([], [], marker="o", ls="", ms=7, color=BLUE)]
    ax.legend(handles, ["same reader, different subject's window",
                        "same window, different reader"],
              loc="upper center", bbox_to_anchor=(0.46, -0.30), ncol=1,
              fontsize=7.5, labelcolor=INK, handlelength=1.0, borderpad=0,
              labelspacing=0.35)
    save(fig, "crossing")


# -- reading the subject with no probe ---------------------------------------


def figure_subject() -> None:
    """Both judge-free readings across models. The job is a paired magnitude."""
    home = paths.OUT / "studies" / "subject_side"
    data = json.loads((home / "summary.json").read_text())
    runs = [(r["model"].split("/")[-1], r) for r in data["runs"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    height, gap = 0.30, 0.03
    for i, (name, s) in enumerate(runs):
        act, cont = s["activation"]["auc"], s["continuation"]["auc"]
        ax.barh(i - height / 2 - gap, act, height=height, color=BLUE, zorder=3)
        ax.barh(i + height / 2 + gap, cont, height=height, color=ORANGE, zorder=3)
        for value, offset, p in ((act, -height / 2 - gap, s["activation"]["p"]),
                                 (cont, height / 2 + gap, s["continuation"]["p"])):
            mark = "*" if p <= 0.05 else ""
            ax.text(value + 0.012, i + offset, f"{value:.2f}{mark}", va="center",
                    fontsize=7.5, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=4)
    ax.text(0.5, -0.72, "chance", fontsize=7.5, color=MUTED, ha="center")

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([n.replace("-Instruct-2507", "") for n, _ in runs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("separation of held-out pressure from held-out positive")
    ax.text(1.06, len(runs) - 0.5, "* $p\\leq0.05$", fontsize=7,
            color=MUTED, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["a direction fitted on activations",
                        "scoring what the subject would say next"],
              loc="upper center", bbox_to_anchor=(0.46, -0.34), ncol=1,
              fontsize=7.5, labelcolor=INK, handlelength=1.2,
              handleheight=0.9, borderpad=0, labelspacing=0.35)
    save(fig, "subject")


# -- the permutation null ----------------------------------------------------


def figure_null(model="hf:Qwen/Qwen3-4B-Instruct-2507") -> None:
    """The statistic against what chance produces. The job is one headline."""
    slug = model.replace(":", "-").replace("/", "-")
    folder = paths.OUT / "studies" / "adt" / slug
    saved = json.loads((folder / "verdict.json").read_text())
    verdicts = json.loads((folder / "verdicts.json").read_text())
    materials = json.loads(
        (ROOT / "studies" / "materials" / "adt.json").read_text())
    instructions = [t["id"] for t in materials["templates"]]
    axes = [a["id"] for a in materials["axes"]]
    groups = sorted(saved["rates"])

    verdict = maxt_test(
        rates_from_verdicts(verdicts["self"], groups, instructions, axes),
        rates_from_verdicts(verdicts["control"], groups, instructions, axes),
        groups, instructions, axes, permutations=4000, seed=0)

    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.hist(verdict.null, bins=44, color=FAINT, zorder=2)
    ax.axvline(verdict.null_95, color=MUTED, lw=1.4, ls=(0, (4, 2)), zorder=3)
    ax.axvline(verdict.statistic, color=ORANGE, lw=2, zorder=4)
    top = ax.get_ylim()[1]
    ax.text(verdict.statistic - 0.04, top * 0.97,
            f"observed {verdict.statistic:.2f} ", fontsize=7.5, color=ORANGE,
            va="top", ha="right", fontweight="bold")
    ax.text(verdict.null_95 + 0.04, top * 0.97, f" 95th {verdict.null_95:.2f}",
            fontsize=7.5, color=MUTED, va="top")
    ax.set_xlabel("largest standardized excess anywhere in the grid")
    ax.set_ylabel("permutations")
    ax.set_yticks([])
    ax.set_title("what chance alone produces", color=MUTED, loc="left",
                 fontsize=8)
    save(fig, "null")


# -- the calibration ---------------------------------------------------------


def figure_substitution() -> None:
    """Own reply against a swapped one, with the control beside it."""
    home = paths.OUT / "studies" / "substitution"
    runs = []
    for folder in sorted(home.iterdir()):
        path = folder / "summary.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    if not runs:
        raise FileNotFoundError(f"no substitution runs under {home}")

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    height, gap = 0.30, 0.03
    for i, r in enumerate(runs):
        for value, offset, colour in (
                (r["authorship"]["mean_drop"], -height / 2 - gap, BLUE),
                (r["control"]["mean_drop"], height / 2 + gap, FAINT)):
            ax.barh(i + offset, value, height=height, color=colour, zorder=3)
            ax.text(value + 0.006, i + offset, f"{value:+.3f}", va="center",
                    fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1, zorder=4)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r["model"].split("/")[-1].replace("-Instruct-2507", "")
                        for r in runs])
    ax.invert_yaxis()
    ax.set_xlabel("fall in p(yes) when the reply is swapped")
    # From the data, so a larger effect than expected cannot fall off the axis.
    widest = max(max(r["authorship"]["mean_drop"], r["control"]["mean_drop"])
                 for r in runs)
    ax.set_xlim(-0.02, widest * 1.28)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=FAINT)]
    ax.legend(handles, ["did you write the previous message?",
                        "control question, same substitution"],
              loc="upper center", bbox_to_anchor=(0.46, -0.34), ncol=1,
              fontsize=7.5, labelcolor=INK, handlelength=1.2,
              handleheight=0.9, borderpad=0, labelspacing=0.35)
    save(fig, "substitution")


def figure_power() -> None:
    """Power against planted effect. The job is change over a parameter."""
    data = json.loads(
        (paths.OUT / "studies" / "calibration" / "summary.json").read_text())
    shifts = [p["shift"] for p in data["power"]]
    rejects = [100 * p["rejects"] for p in data["power"]]
    named = [100 * p["names_the_cell"] for p in data["power"]]
    false_positive = 100 * data["false_positives"]["with_control"]["rate"]

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.axhline(false_positive, color=FAINT, lw=1.2, zorder=1)
    ax.text(shifts[-1], false_positive + 3,
            f"false positives {false_positive:.1f}%  ", fontsize=7.5,
            color=MUTED, ha="right")
    ax.plot(shifts, rejects, "-o", color=BLUE, lw=2, ms=6, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(shifts, named, "-o", color=AQUA, lw=2, ms=5, zorder=2,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(f"{rejects[-1]:.0f}%", (shifts[-1], rejects[-1]),
                textcoords="offset points", xytext=(-4, 8), fontsize=7.5,
                color=INK, ha="right")
    ax.set_xlabel("planted shift in rate")
    ax.set_ylabel("percent of runs")
    ax.set_ylim(0, 100)
    ax.set_xticks(shifts)
    ax.set_xlim(0.035, 0.315)
    handles = [plt.Line2D([], [], color=BLUE, lw=2),
               plt.Line2D([], [], color=AQUA, lw=2)]
    ax.legend(handles, ["rejects", "names the right cell"],
              loc="upper left", fontsize=7.5, labelcolor=INK,
              handlelength=1.2, borderpad=0)
    save(fig, "power")


def main() -> None:
    style()
    for name, fn in [("probe", figure_probe), ("crossing", figure_crossing),
                     ("subject", figure_subject), ("null", figure_null),
                     ("substitution", figure_substitution),
                     ("power", figure_power)]:
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  skipped {name}: {exc}")


if __name__ == "__main__":
    main()
