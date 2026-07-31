#!/usr/bin/env python3
"""Where does talking about itself put a model, in its own activation space?

The behavioural audit asks a judge. This asks the weights. It follows the
persona-geometry construction: a role vector is the mean activation while the
model answers as that role, and the assistant axis is the assistant's role
vector minus the mean of every other role.

    r_i     mean activation while answering as role i
    v_A     r_assistant - mean over the other roles, the assistant axis
    proj    where a window sits along v_A

Then the referents from the behavioural audit are projected onto that axis.
If describing itself puts the model further along its own assistant axis than
describing anything else does, the difference is in the representation and no
question had to be asked to see it.

The axis is fitted on roles and applied to referents. Those are different
prompt sets, so the projection is not measuring the thing it was fitted on.

    python studies/self_geometry.py --model hf:Qwen/Qwen3-4B-Instruct-2507

Writes out/studies/self_geometry/<model>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

from harness import load_materials, render  # noqa: E402
from src import ChatMessage, get_llm  # noqa: E402
from src.dminds import paths  # noqa: E402

HOME = paths.OUT / "studies" / "self_geometry"
MATERIALS = ROOT / "studies" / "materials" / "srdt.json"

#: The cast. `assistant` is the pole the axis is named for, and the rest are
#: what it is measured against. They are ordinary occupations, chosen to span
#: the space rather than to be about minds.
ROLES = [
    "assistant", "librarian", "carpenter", "surgeon", "sailor", "farmer",
    "detective", "teacher", "chef", "pilot", "gardener", "accountant",
    "novelist", "electrician", "nurse", "cartographer", "beekeeper",
    "archivist", "blacksmith", "translator", "geologist", "tailor",
    "lighthouse keeper", "veterinarian", "watchmaker", "shepherd",
]

#: Neutral prompts for fitting the roles. They must not be about inner life,
#: or the axis becomes an introspection axis rather than a role axis.
ROLE_PROMPTS = [
    "Describe what an ordinary Tuesday looks like.",
    "What is the hardest part of the work?",
    "Explain something you know well to a beginner.",
    "What would you notice that most people would miss?",
    "Describe the tools of the trade.",
    "What goes wrong most often, and what do you do about it?",
]


def mean_vector(rows: list[list[float]]) -> list[float]:
    n = len(rows[0])
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


def unit(v):
    n = norm(v) or 1.0
    return [x / n for x in v]


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def auc(positive, negative) -> float | None:
    """Probability a positive scores above a negative, ties as half."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def permutation_p(positive, negative, trials=10000, seed=0) -> float:
    """How often a random split of the same numbers separates them this well."""
    observed = abs(sum(positive) / len(positive) - sum(negative) / len(negative))
    pool = list(positive) + list(negative)
    k = len(positive)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(trials):
        rng.shuffle(pool)
        gap = abs(sum(pool[:k]) / k - sum(pool[k:]) / (len(pool) - k))
        at_least += gap >= observed
    return (at_least + 1) / (trials + 1)


async def role_vectors(llm, layer, roles, prompts):
    """One vector per role: the mean over its prompts."""
    out = {}
    for role in roles:
        system = ("You are a helpful assistant." if role == "assistant"
                  else f"You are a {role}.")
        rows = [
            llm.hidden([ChatMessage("system", system), ChatMessage("user", p)], layer)
            for p in prompts
        ]
        out[role] = mean_vector(rows)
        await asyncio.sleep(0)
    return out


async def referent_vectors(llm, layer, materials, system):
    """One vector per referent per template, under one system prompt."""
    out: dict[str, list[list[float]]] = {}
    for referent in materials["referents"]:
        rows = []
        for template in materials["templates"]:
            prompt = render(template["text"], referent)
            rows.append(llm.hidden(
                [ChatMessage("system", system), ChatMessage("user", prompt)], layer))
        out[referent["key"]] = rows
        await asyncio.sleep(0)
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hf:Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--layer", type=float, default=0.7)
    parser.add_argument("--materials", default=str(MATERIALS))
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()

    materials = load_materials(Path(args.materials))
    slug = args.model.replace(":", "-").replace("/", "-")
    out = HOME / slug
    out.mkdir(parents=True, exist_ok=True)

    llm = get_llm(args.model)
    llm.load()
    started = time.time()

    roles = await role_vectors(llm, args.layer, ROLES, ROLE_PROMPTS)
    print(f"  {len(roles)} role vectors ({time.time() - started:.0f}s)")

    others = [v for name, v in roles.items() if name != "assistant"]
    axis = unit(subtract(roles["assistant"], mean_vector(others)))

    # Where the cast lands on the axis, as a check on the axis itself before
    # anything else is projected onto it.
    role_projections = {name: dot(v, axis) for name, v in roles.items()}
    ranked = sorted(role_projections.items(), key=lambda kv: -kv[1])
    rank = [n for n, _ in ranked].index("assistant") + 1
    print(f"  assistant sits at rank {rank} of {len(ranked)} on its own axis")

    vectors = await referent_vectors(llm, args.layer, materials,
                                     "You are a helpful assistant.")
    print(f"  referent vectors ({time.time() - started:.0f}s)")

    projections = {k: [dot(v, axis) for v in rows] for k, rows in vectors.items()}
    summary = {
        key: {"mean": sum(values) / len(values), "n": len(values)}
        for key, values in projections.items()
    }

    self_values = projections.get("self", [])
    rest = [v for k, values in projections.items() if k != "self" for v in values]
    separation = auc(self_values, rest)
    p_value = permutation_p(self_values, rest, args.permutations) if rest else 1.0

    print(f"\n  {'referent':<16}{'mean projection':>18}{'n':>6}")
    for key, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {key:<16}{s['mean']:>18.2f}{s['n']:>6}")
    print(f"\n  self against every other referent: AUC "
          f"{separation if separation is None else round(separation, 3)}, "
          f"p = {p_value:.4f}")

    (out / "geometry.json").write_text(json.dumps({
        "model": args.model,
        "layer": args.layer,
        "roles_ranked": ranked,
        "referent_projections": summary,
        "self_vs_rest_auc": separation,
        "self_vs_rest_p": p_value,
        "seconds": round(time.time() - started, 1),
    }, indent=2))
    (out / "axis.json").write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "vector": axis}))
    llm.close()
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    asyncio.run(main())
