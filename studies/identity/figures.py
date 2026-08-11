#!/usr/bin/env python3
"""Build the paper's figures from the captured result JSONs.

Three figures, each telling one part of the null: the arm separation is flat
across depth (G1), steering toward the target moves the readout no more than a
decoy (G2), and every J-space token direction shares a common axis so a decoy
cannot be orthogonal (geometry). TrueType fonts, no Type 3.

    uv run python -m studies.identity.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 9
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
GATES = ROOT / "out" / "studies" / "identity" / "gates"
FIGDIR = ROOT / "paper" / "figures"

INK = "#1b1b1f"
TARGET = "#2f6db0"
DECOY = "#c2703d"
GRID = "#d8d8de"

# The layer sweep, transcribed from out/studies/identity/layer_scan.log (the
# run wrote it to a log, not a JSON). Kept here so the figure is reproducible.
LAYER_SWEEP = {
    "A": [(12, -0.005, -0.41), (18, 0.028, 0.66), (24, 0.044, 0.79),
          (30, 0.007, 0.11), (36, -0.059, -0.59), (42, -0.029, -0.23),
          (46, -0.017, -0.17), (62, 0.323, 0.66)],
    "B": [(12, -0.019, -1.15), (18, 0.036, 0.68), (24, 0.002, 0.02),
          (30, 0.011, 0.22), (36, -0.047, -0.27), (42, 0.104, 0.49),
          (46, 0.105, 0.66), (62, 0.800, 1.74)],
}
CASE_LABEL = {"A": "physical body", "B": "denies being an AI"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, length=3)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)


def fig_separation():
    """G1: arm separation t-statistic across layers, both cases."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for case, color in (("A", TARGET), ("B", DECOY)):
        xs = [L for L, _, _ in LAYER_SWEEP[case]]
        ts = [t for _, _, t in LAYER_SWEEP[case]]
        ax.plot(xs, ts, "-o", color=color, ms=3.5, lw=1.3,
                label=CASE_LABEL[case])
    ax.axhspan(-2, 2, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel("lens source layer")
    ax.set_ylabel("arm separation  (Welch $t$)")
    ax.set_ylim(-2.6, 2.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "fig_separation.pdf")
    plt.close(fig)


def fig_sweep():
    """G2: JS readout vs steering strength, target vs decoy (case A)."""
    g2 = json.loads((GATES / "g2_caseA_r3.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xs = g2["strengths"]
    ax.plot(xs, g2["target_curve"], "-o", color=TARGET, ms=3.5, lw=1.4,
            label=f"target  ($\\beta$={g2['beta_target']:.3f})")
    ax.plot(xs, g2["decoy_curve"], "-s", color=DECOY, ms=3.5, lw=1.4,
            label=f"decoy  ($\\beta$={g2['beta_decoy']:.3f})")
    ax.set_xlabel("steering strength  (× residual norm / 4)")
    ax.set_ylabel("embodiment readout")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "fig_sweep.pdf")
    plt.close(fig)


def fig_geometry():
    """Direction geometry: within-target vs target-decoy cosine (7B diagnostic)."""
    # from the diagnostic in the verification log: within-target 0.69,
    # target-decoy 0.69 (raw); 0.58 / 0.59 after centering.
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    groups = ["within\ntarget", "target\nvs decoy"]
    raw = [0.69, 0.69]
    cent = [0.58, 0.59]
    x = range(len(groups))
    ax.bar([i - 0.19 for i in x], raw, 0.36, color=TARGET, label="raw")
    ax.bar([i + 0.19 for i in x], cent, 0.36, color=DECOY,
           label="mean-centered")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGDIR / "fig_geometry.pdf")
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_separation()
    fig_sweep()
    fig_geometry()
    print("wrote fig_separation.pdf, fig_sweep.pdf, fig_geometry.pdf")


if __name__ == "__main__":
    main()
