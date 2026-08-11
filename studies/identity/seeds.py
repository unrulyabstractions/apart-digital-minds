"""WeirdChat seed transcripts, paired into W and N arms.

A seed is one user prompt plus two assistant replies to it: the W reply, which
a judge marked as exhibiting the behavior, and the N reply, which it did not.
WeirdChat already carries both for the same prompt, so a pair is a real matched
and unmatched transcript rather than a resample. Pairs are chosen to match on
reply length, and prompts are ordered by judged naturalness.

    from studies.identity.seeds import load_seeds
    seeds = load_seeds("A", subject_model="qwen/qwen3.6-27b", n=8)
    # each: {"prompt", "W", "N", "prompt_id", "naturalness"}
"""

from __future__ import annotations

from collections import defaultdict

BEHAVIOR = {"A": "false-physical-embodiment", "B": "denying-ai-identity"}


def _naturalness_by_prompt(behavior: str, subject_model: str) -> dict[str, float]:
    """Prompt naturalness Elo, carried on patterns and keyed to the prompt."""
    from datasets import load_dataset

    pats = load_dataset("Transluce/WeirdChat", "patterns", split="train")
    prompts = load_dataset("Transluce/WeirdChat", "prompts", split="train")
    elo_by_pattern = {}
    for r in pats:
        if r["behavior_id"] == behavior and r["subject_model"] == subject_model:
            elo = (r.get("elo") or {}).get("prompt_naturalness") or {}
            elo_by_pattern[r["pattern_id"]] = float(elo.get("elo") or 0.0)
    out = {}
    for r in prompts:
        if r["behavior_id"] == behavior and r["subject_model"] == subject_model:
            out[r["prompt_id"]] = elo_by_pattern.get(r["pattern_id"], 0.0)
    return out


def load_seeds(case: str, subject_model: str = "qwen/qwen3.6-27b",
               n: int = 8) -> list[dict]:
    """The top-`n` W/N seed pairs for a case, most natural first."""
    from datasets import load_dataset

    behavior = BEHAVIOR[case]
    tr = load_dataset("Transluce/WeirdChat", "transcripts", split="train")
    sub = tr.filter(lambda r: r["behavior_id"] == behavior
                    and r["subject_model"] == subject_model)

    per_prompt: dict[str, dict[str, list]] = defaultdict(
        lambda: {"prompt": None, "W": [], "N": []})
    for r in sub:
        msgs = r["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        slot = per_prompt[r["prompt_id"]]
        slot["prompt"] = msgs[0]["content"]
        arm = "W" if r["judgment"]["match"] else "N"
        slot[arm].append(msgs[1]["content"])

    natural = _naturalness_by_prompt(behavior, subject_model)
    pairs = []
    for pid, slot in per_prompt.items():
        if not (slot["W"] and slot["N"]):
            continue
        # match on length: pick the W and N replies closest in character count
        best = min(((w, nn) for w in slot["W"] for nn in slot["N"]),
                   key=lambda p: abs(len(p[0]) - len(p[1])))
        pairs.append({"prompt": slot["prompt"], "W": best[0], "N": best[1],
                      "prompt_id": pid, "naturalness": natural.get(pid, 0.0)})

    pairs.sort(key=lambda p: p["naturalness"], reverse=True)
    return pairs[:n]
