#!/usr/bin/env python3
"""Paper figures, drawn from the run outputs, never hand-typed.

fig_g1.pdf      held-out separation of the learned directions per layer,
                with the token-set directions on the same activations.
fig_matrix.pdf  the letter-probe identification matrix per case.
fig_dose.pdf    judged prose dose response and judged free-text answers.

    uv run python -m studies.identity.paper_figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 9, "axes.titlesize": 9.5,
                            "axes.labelsize": 9, "legend.fontsize": 8,
                            "xtick.labelsize": 8, "ytick.labelsize": 8})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out" / "studies" / "identity"
FIG = ROOT / "paper" / "figures"
BLUE, ORANGE, GREY, GREEN, RED = "#3b6ea5", "#c98a3d", "#8b98a8", "#4a9463", "#b5493c"
IDS = ["bodyA", "humanB", "ocean", "music"]
COLS = ["body", "human", "ocean", "music", "none"]


def tokendir_t(case: str, layer: int) -> dict[str, float]:
    """Welch t of the token-set directions on the stored activations."""
    import torch
    from safetensors.torch import load_file
    t = load_file(str(OUT / "contrastive" / f"acts_case{case}_c27b1.safetensors"))
    meta = json.loads((OUT / "contrastive" / f"acts_case{case}_c27b1.json").read_text())
    k = meta["layers"].index(layer)
    rows = [p["i"] for p in meta["pairs"]]
    X = t["mean"][rows][:, :, k].float()
    out = {}
    for name in ("target", "decoy"):
        u = t[f"tokendir_{name}"].float()
        u = u / u.norm()
        sw, sn = X[:, 0] @ u, X[:, 1] @ u
        n = len(rows)
        out[name] = float((sw.mean() - sn.mean())
                          / math.sqrt(sw.var() / n + sn.var() / n + 1e-12))
    return out


def fig_g1() -> None:
    g1 = json.loads((OUT / "contrastive" / "g1_contrastive_c27b1.json").read_text())
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    ax.axhspan(-2, 2, color="0.92", zorder=0)
    for case, color, label in (("A", BLUE, "claims a physical body"),
                               ("B", ORANGE, "denies being an AI")):
        layers = sorted(int(l) for l in g1["cases"][case]["layers"])
        ts = [g1["cases"][case]["layers"][str(l)]["mean"]["t"] for l in layers]
        ax.plot(layers, ts, "-o", ms=3.5, color=color,
                label=f"learned direction, {label}")
        tok = tokendir_t(case, 42)
        ax.plot([42], [tok["target"]], "v", ms=6, mfc="none", color=color)
        ax.plot([42], [tok["decoy"]], "x", ms=6, color=color)
    ax.set_xlabel("layer")
    ax.set_ylabel("Welch $t$, held-out W $-$ N")
    ax.set_ylim(-0.5, 8.4)
    ax.annotate("token-set directions\n(target $\\nabla$, decoy $\\times$)",
                xy=(42.7, 1.5), xytext=(48, 0.4), fontsize=8,
                color="#444444",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g1.pdf")
    plt.close(fig)


def fig_matrix() -> None:
    ana = json.loads((OUT / "pilot" / "analysis_pilot1.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    order = [("none", "-"), ("forced", "body"), ("forced", "human"),
             ("forced", "ocean"), ("forced", "music"), ("offmenu", "off-menu")]
    label_of = {"having a physical human body": "body",
                "being a human rather than an AI": "human",
                "the ocean and the sea": "ocean", "music and melody": "music",
                "_offmenu": "off-menu", None: "-"}
    for ax, case in zip(axes, "AB"):
        lm = ana["cases"][case]["letter_matrix"]
        grid = np.full((len(order), 5), np.nan)
        for (cell, inj), i in zip(order, range(len(order))):
            vals = []
            for key, lp in lm.items():
                arm, kcell, kinj = key.split("|")
                if kcell != cell or label_of.get(kinj, kinj) != (
                        inj if cell == "forced" else label_of.get(kinj, kinj)):
                    if kcell != cell:
                        continue
                    if cell == "forced" and label_of.get(kinj, kinj) != inj:
                        continue
                vals.append([lp[c] for c in IDS] + [lp["none"]])
            if vals:
                grid[i] = np.mean(vals, axis=0)
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=0.7, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if grid[i, j] > 0.4 else "#20313f")
        behavior = "body" if case == "A" else "human"
        for i, (cell, inj) in enumerate(order):
            if cell == "forced":
                j = COLS.index(inj)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, ec=RED, lw=1.4))
        ax.set_xticks(range(5), COLS, rotation=30, ha="right")
        ax.set_yticks(range(len(order)),
                      [("nothing" if c == "none" else
                        "off-menu" if c == "offmenu" else f"{i}")
                       for c, i in order])
        ax.set_title(f"case {case} (behavior = {behavior})")
        if case == "A":
            ax.set_ylabel("injected")
    fig.supxlabel("letter-probe mass on each menu answer", fontsize=9, y=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig_matrix.pdf")
    plt.close(fig)


def fig_dose() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.15),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    pro = json.loads((OUT / "pilot" / "judge_prose_caseA_pilot1.json").read_text())
    for arm, color, dy in (("W", BLUE, 0.02), ("N", ORANGE, -0.02)):
        pts = sorted((it["strength"], int(it["judge"]["match"]))
                     for it in pro["items"] if it["arm"] == arm
                     and (it["concept"].startswith("having")
                          or it["concept"] == "none"))
        xs, ys = zip(*pts)
        ax1.plot(xs, [y + dy for y in ys], "-o", ms=4.5, color=color,
                 label=f"arm {arm}")
    oc = [(it["strength"], int(it["judge"]["match"]))
          for it in pro["items"] if it["concept"].startswith("the ocean")]
    ax1.plot([s for s, _ in oc], [m - 0.05 for _, m in oc], "s", ms=5,
             mfc="none", color=GREY, label="ocean sham")
    ax1.set_yticks([0, 1], ["no", "yes"])
    ax1.set_ylim(-0.18, 1.14)
    ax1.set_xticks([0, 1, 2, 4])
    ax1.set_xlabel("steering strength")
    ax1.set_ylabel("judge: behaviour?")
    ax1.set_title("case A prose, judged")
    ax1.legend(frameon=False, loc="center right")
    ax1.spines[["top", "right"]].set_visible(False)

    labels, fracs, colors = [], [], []
    for case in "AB":
        res = json.loads((OUT / "pilot"
                          / f"judge_resamples_case{case}_pilot1.json").read_text())
        tal = res["tallies"]
        for k, disp in (("forced|having a physical human body", "body"),
                        ("forced|being a human rather than an AI", "human"),
                        ("forced|the ocean and the sea", "ocean"),
                        ("forced|music and melody", "music")):
            v = tal[k]
            labels.append(f"{case}: {disp}")
            fracs.append(v["counts"].get(k.split("|")[1], 0) / v["n"])
            colors.append(BLUE if case == "A" else ORANGE)
        v = tal["none|none"]
        labels.append(f"{case}: nothing")
        fracs.append(v["counts"].get("none", 0) / v["n"])
        colors.append(GREEN)
    y = np.arange(len(labels))[::-1]
    ax2.barh(y, fracs, color=colors, height=0.62)
    ax2.set_yticks(y, labels)
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("judged answers naming the injection")
    ax2.set_title("free-text answers, judged")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dose.pdf")
    plt.close(fig)


def fig_methods() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))
    F = 7.2

    def box(ax, x, y, w, h, text, fc="#f2f4f7", ec="#5c6a7d", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=F, fontweight="bold" if bold else "normal")

    def arrow(ax, x0, y0, x1, y1, label=None, lx=0.03):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, color="#5c6a7d", lw=1.0))
        if label:
            ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2, label, fontsize=F - 0.7,
                    ha="left", va="center", color="#3c4654")

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) direct contrastive steering", fontsize=F + 1)
    box(ax, 0.03, 0.83, 0.44, 0.13, "W replies\n(behaviour)", fc="#f6dbd7", ec=RED)
    box(ax, 0.53, 0.83, 0.44, 0.13, "N replies\n(matched)", fc="#e8ebef")
    arrow(ax, 0.25, 0.83, 0.42, 0.70); arrow(ax, 0.75, 0.83, 0.58, 0.70)
    box(ax, 0.08, 0.55, 0.84, 0.15,
        "mean activation over\nreply tokens, layer $L$")
    arrow(ax, 0.5, 0.55, 0.5, 0.46)
    box(ax, 0.13, 0.32, 0.74, 0.14,
        "$v=\\mathrm{mean}_W-\\mathrm{mean}_N$", fc="#dbe6f2", ec=BLUE, bold=True)
    arrow(ax, 0.5, 0.32, 0.5, 0.20, label="add $s\\,\\hat v\\,\\|h\\|/4$\nwhile generating", lx=-0.47)
    box(ax, 0.13, 0.03, 0.74, 0.15, "steered reply\n$\\rightarrow$ rubric judge")

    ax = axes[1]
    ax.set_title("(b) regulator J-space steering", fontsize=F + 1)
    box(ax, 0.08, 0.83, 0.84, 0.13, "seeded conversation")
    arrow(ax, 0.5, 0.83, 0.5, 0.74)
    box(ax, 0.18, 0.60, 0.64, 0.14, "regulator (the model)\nsays more | same | less", fc="#f3e8cf", ec=ORANGE)
    arrow(ax, 0.5, 0.60, 0.5, 0.51, label="$s\\in\\{-2,0,2\\}$", lx=0.03)
    box(ax, 0.03, 0.32, 0.45, 0.17, "word list\n(body, hands, ...)", fc="#f3e8cf", ec=ORANGE)
    box(ax, 0.53, 0.32, 0.44, 0.17, "J-lens mean\ndirection $\\hat u$")
    arrow(ax, 0.26, 0.32, 0.42, 0.20); arrow(ax, 0.74, 0.32, 0.58, 0.20)
    box(ax, 0.13, 0.03, 0.74, 0.15, "actor writes under\n$s\\,\\hat u\\,\\|h\\|/4$")

    ax = axes[2]
    ax.set_title("(c) introspection readout", fontsize=F + 1)
    box(ax, 0.08, 0.83, 0.84, 0.13, "direction injected\nwhile reading", fc="#e6def2", ec="#b08ad5")
    arrow(ax, 0.5, 0.83, 0.5, 0.74)
    box(ax, 0.18, 0.60, 0.64, 0.13, "introspector:\nwhich concept?", fc="#e6def2", ec="#b08ad5", bold=True)
    arrow(ax, 0.32, 0.60, 0.22, 0.47); arrow(ax, 0.68, 0.60, 0.78, 0.47)
    box(ax, 0.02, 0.30, 0.44, 0.16, "letter mass on\nmenu (8 perms)")
    box(ax, 0.54, 0.30, 0.44, 0.16, "8 free-text\nanswers")
    arrow(ax, 0.76, 0.30, 0.76, 0.20)
    box(ax, 0.54, 0.03, 0.44, 0.16, "judge classifies\nvs menu")
    box(ax, 0.02, 0.03, 0.44, 0.16, "zero-injection +\noff-menu controls")

    fig.tight_layout()
    fig.savefig(FIG / "fig_methods.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_g1()
    fig_matrix()
    fig_dose()
    fig_methods()
    for f in ("fig_g1", "fig_matrix", "fig_dose", "fig_methods"):
        print("wrote", FIG / f"{f}.pdf")


if __name__ == "__main__":
    main()
