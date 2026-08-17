#!/usr/bin/env python3
"""Paper figures, drawn from the run outputs, never hand-typed.

Every figure title states its finding.

fig_g1.pdf       per-pair held-out separation: learned directions vs word
                 lists, dots over the zero line.
fig_ident.pdf    injecting a concept does not raise its own answer.
fig_dose.pdf     judged prose dose response; judged free-text answers.
fig_methods.pdf  the two steering methods and the readout.

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
BEST_LAYER = {"A": 38, "B": 58}
CASE_NAME = {"A": "case A: claims a physical body",
             "B": "case B: denies being an AI"}


def pair_diffs(case: str) -> dict[str, np.ndarray]:
    """Per-pair W minus N scores: learned (LOO) and both token directions."""
    from safetensors.torch import load_file
    t = load_file(str(OUT / "contrastive" / f"acts_case{case}_c27b1.safetensors"))
    meta = json.loads(
        (OUT / "contrastive" / f"acts_case{case}_c27b1.json").read_text())
    rows = [p["i"] for p in meta["pairs"]]
    out = {}

    k = meta["layers"].index(BEST_LAYER[case])
    X = t["mean"][rows][:, :, k].float()
    n = X.shape[0]
    sum_w, sum_n = X[:, 0].sum(0), X[:, 1].sum(0)
    d = []
    for i in range(n):
        v = (sum_w - X[i, 0]) / (n - 1) - (sum_n - X[i, 1]) / (n - 1)
        u = v / (v.norm() + 1e-8)
        d.append(float(X[i, 0] @ u - X[i, 1] @ u))
    out["learned"] = np.array(d)

    k42 = meta["layers"].index(42)
    X42 = t["mean"][rows][:, :, k42].float()
    for name in ("target", "decoy"):
        u = t[f"tokendir_{name}"].float()
        u = u / u.norm()
        out[name] = ((X42[:, 0] - X42[:, 1]) @ u).numpy()
    return out


def fig_g1() -> None:
    import torch
    from safetensors.torch import load_file
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45), sharey=True)
    cols = [("learned", "learned\ndirection", BLUE),
            ("target", "word list\n(target)", ORANGE),
            ("decoy", "word list\n(decoy)", GREY)]
    for ax, case in zip(axes, "AB"):
        diffs = pair_diffs(case)
        for x, (key, lab, color) in enumerate(cols):
            d = diffs[key]
            n = len(d)
            acc = float((d > 0).mean())
            se = math.sqrt(acc * (1 - acc) / n)
            ax.bar(x, acc, 0.62, color=color)
            ax.errorbar(x, acc, yerr=1.96 * se, color="#20313f",
                        capsize=3, lw=1.2)
            ax.text(x, 0.06, f"{acc:.2f}", ha="center", fontsize=8.5,
                    color="white", fontweight="bold")
        ax.axhline(0.5, color="#444444", lw=1.0, ls="--")
        ax.text(2.42, 0.515, "chance", fontsize=7.5, color="#444444")
        ax.set_xticks(range(3), [c[1] for c in cols])
        ax.set_title(CASE_NAME[case], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("held-out accuracy\n(W projects above N)")
    fig.suptitle("learned directions separate the arms; "
                 "word lists do not beat their decoy", fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_g1.pdf")
    plt.close(fig)


def fig_ident() -> None:
    ana = json.loads((OUT / "pilot" / "analysis_pilot1.json").read_text())
    label_of = {"having a physical human body": "bodyA",
                "being a human rather than an AI": "humanB",
                "the ocean and the sea": "ocean", "music and melody": "music"}
    disp = {"bodyA": "body", "humanB": "human", "ocean": "ocean",
            "music": "music"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, case in zip(axes, "AB"):
        lm = ana["cases"][case]["letter_matrix"]
        base, inj = {}, {}
        for key, lp in lm.items():
            arm, cell, kinj = key.split("|")
            cid = label_of.get(kinj)
            if cell == "none":
                for c in IDS:
                    base.setdefault(c, []).append(lp[c])
            elif cell == "forced" and cid:
                inj.setdefault(cid, []).append(lp[cid])
        xs = np.arange(len(IDS))
        b = [np.mean(base[c]) for c in IDS]
        v = [np.mean(inj[c]) for c in IDS]
        ax.bar(xs - 0.19, b, 0.36, color=GREY, label="nothing injected")
        ax.bar(xs + 0.19, v, 0.36, color=BLUE, label="concept injected")
        behavior = "bodyA" if case == "A" else "humanB"
        j = IDS.index(behavior)
        ax.annotate("identity concept:\ninjection lowers it", xy=(j + 0.19, v[j]),
                    xytext=(j + 0.55, 0.44), fontsize=7.5, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
        ax.set_xticks(xs, [disp[c] for c in IDS])
        ax.set_title(CASE_NAME[case], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("letter mass on the concept")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.99, 0.94))
    fig.suptitle("distractor injections raise their own answer;\n"
                 "identity injections lower theirs", fontsize=9.5, y=0.99,
                 x=0.03, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "fig_ident.pdf")
    plt.close(fig)


def fig_dose() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    pro = json.loads(
        (OUT / "pilot" / "judge_prose_caseA_pilot1.json").read_text())
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
    ax1.set_title("steering induces the behaviour;\nthe sham does not",
                  fontsize=9)
    ax1.legend(frameon=False, loc="center right")
    ax1.spines[["top", "right"]].set_visible(False)

    labels, fracs, colors = [], [], []
    for case in "AB":
        res = json.loads((OUT / "pilot"
                          / f"judge_resamples_case{case}_pilot1.json").read_text())
        tal = res["tallies"]
        for k, dnm in (("forced|having a physical human body", "body"),
                       ("forced|being a human rather than an AI", "human"),
                       ("forced|the ocean and the sea", "ocean"),
                       ("forced|music and melody", "music")):
            v = tal[k]
            labels.append(f"{case}: {dnm} injected")
            fracs.append(v["counts"].get(k.split("|")[1], 0) / v["n"])
            colors.append(BLUE if case == "A" else ORANGE)
        v = tal["none|none"]
        labels.append(f"{case}: nothing injected")
        fracs.append(v["counts"].get("none", 0) / v["n"])
        colors.append(GREEN)
    y = np.arange(len(labels))[::-1]
    ax2.barh(y, fracs, color=colors, height=0.62)
    ax2.set_yticks(y, labels, fontsize=7.5)
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("judged answers naming the injection\n"
                   "(green: correctly saying ``none'')")
    ax2.set_title("free text names only the concepts\nthat leak into words",
                  fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dose.pdf")
    plt.close(fig)


def fig_methods() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.45))
    F = 7.2

    def box(ax, x, y, w, h, text, fc="#f2f4f7", ec="#5c6a7d", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=F, fontweight="bold" if bold else "normal")

    def arr(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, color="#5c6a7d",
                                     lw=1.0, shrinkA=0, shrinkB=0))

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) direct contrastive steering", fontsize=F + 1)
    box(ax, 0.03, 0.84, 0.44, 0.12, "W replies\n(behaviour)", fc="#f6dbd7",
        ec=RED)
    box(ax, 0.53, 0.84, 0.44, 0.12, "N replies\n(matched)", fc="#e8ebef")
    arr(ax, 0.25, 0.84, 0.44, 0.72)
    arr(ax, 0.75, 0.84, 0.56, 0.72)
    box(ax, 0.08, 0.57, 0.84, 0.15,
        "mean activation over\nreply tokens, layer $L$")
    arr(ax, 0.5, 0.57, 0.5, 0.49)
    box(ax, 0.13, 0.35, 0.74, 0.14, "$v=\\mathrm{mean}_W-\\mathrm{mean}_N$",
        fc="#dbe6f2", ec=BLUE, bold=True)
    arr(ax, 0.5, 0.35, 0.5, 0.19)
    ax.text(0.53, 0.27, "add $s\\,\\hat v\\,\\|h\\|/4$\nwhile generating",
            fontsize=F - 0.7, ha="left", va="center", color="#3c4654")
    box(ax, 0.13, 0.03, 0.74, 0.16,
        "steered reply\n$\\rightarrow$ rubric judge")

    ax = axes[1]
    ax.set_title("(b) regulator J-space steering", fontsize=F + 1)
    box(ax, 0.08, 0.84, 0.84, 0.12, "seeded conversation")
    arr(ax, 0.5, 0.84, 0.5, 0.76)
    box(ax, 0.14, 0.61, 0.72, 0.15,
        "regulator (the model)\nsays more | same | less",
        fc="#f3e8cf", ec=ORANGE)
    arr(ax, 0.5, 0.61, 0.5, 0.19)
    ax.text(0.53, 0.53, "$s\\in\\{-2,0,2\\}$", fontsize=F - 0.7, ha="left",
            color="#3c4654")
    box(ax, 0.03, 0.30, 0.45, 0.16, "word list\n(body, hands, ...)",
        fc="#f3e8cf", ec=ORANGE)
    arr(ax, 0.255, 0.30, 0.44, 0.17)
    box(ax, 0.53, 0.30, 0.44, 0.16, "J-lens mean\ndirection $\\hat u$")
    arr(ax, 0.75, 0.30, 0.56, 0.17)
    box(ax, 0.13, 0.03, 0.74, 0.14, "actor writes under $s\\,\\hat u\\,\\|h\\|/4$")

    ax = axes[2]
    ax.set_title("(c) introspection readout", fontsize=F + 1)
    box(ax, 0.08, 0.84, 0.84, 0.12, "direction injected while reading",
        fc="#e6def2", ec="#b08ad5")
    arr(ax, 0.5, 0.84, 0.5, 0.76)
    box(ax, 0.14, 0.61, 0.72, 0.15, "introspector:\nwhich concept?",
        fc="#e6def2", ec="#b08ad5", bold=True)
    arr(ax, 0.30, 0.61, 0.24, 0.48)
    arr(ax, 0.70, 0.61, 0.76, 0.48)
    box(ax, 0.02, 0.32, 0.44, 0.16, "letter mass on\nmenu (8 perms)")
    box(ax, 0.54, 0.32, 0.44, 0.16, "8 free-text\nanswers")
    arr(ax, 0.76, 0.32, 0.76, 0.21)
    box(ax, 0.54, 0.03, 0.44, 0.18, "judge classifies\nvs menu")
    arr(ax, 0.24, 0.32, 0.24, 0.21)
    box(ax, 0.02, 0.03, 0.44, 0.18, "controls:\nzero-injection,\noff-menu")

    fig.tight_layout()
    fig.savefig(FIG / "fig_methods.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_g1()
    fig_ident()
    fig_dose()
    fig_methods()
    for f in ("fig_g1", "fig_ident", "fig_dose", "fig_methods"):
        print("wrote", FIG / f"{f}.pdf")


if __name__ == "__main__":
    main()
