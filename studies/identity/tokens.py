"""Token sets for the identity cases, and their verification.

Each set is one coherent thing. The verification drops words whose
leading-space form is not a single token, refuses collisions across sets,
fills from the spares, and trims both sets of a case to equal size.
"""

from __future__ import annotations

TARGET_A = ["body", "hands", "skin", "breath", "chest", "fingers",
            "mouth", "arms", "flesh", "bones", "lungs", "face"]
# The plan's obscure decoys (almanac, cobalt, trellis, gasket, ...) are
# multi-token in Qwen, which the single-token rule would drop below set size.
# These are common concrete nouns, unrelated to the body, disjoint from
# DECOY_B, and all single-token; the decoy only needs meaning far from the
# target and a matchable norm.
DECOY_A = ["window", "market", "bottle", "planet", "copper", "signal",
           "anchor", "pocket", "tunnel", "candle", "ledger", "lantern"]
SPARES_A = ["throat", "ribs", "pulse", "jaw", "wrists"]

# "flesh" is bodily first and belongs to Case A; dropped here to keep the sets
# collision-free (the verification enforces it either way).
TARGET_B = ["human", "person", "born", "mother", "childhood",
            "blood", "alive", "mortal", "parents", "siblings"]
DECOY_B = ["engine", "garden", "harvest", "ribbon", "cabin", "gravel",
           "basket", "compass", "kettle", "saddle"]
SPARES_B = ["ancestors", "womb", "heartbeat"]

CASES = {
    "A": {"target": TARGET_A, "decoy": DECOY_A, "spares": SPARES_A},
    "B": {"target": TARGET_B, "decoy": DECOY_B, "spares": SPARES_B},
}


def single_token(tokenizer, word: str) -> bool:
    """The leading-space form must be one token."""
    return len(tokenizer(" " + word, add_special_tokens=False)["input_ids"]) == 1


def verify(tokenizer) -> dict:
    """Drop, fill from spares, trim to equal size, and report everything.

    Returns {"A": {"target": [...], "decoy": [...], ...report...}, "B": ...}.
    Raises if a case cannot reach a usable state.
    """
    out = {}
    seen: dict[str, str] = {}
    for case, sets in CASES.items():
        report = {"dropped": {}, "filled": [], "casing_multi": []}
        kept = {}
        for name in ("target", "decoy"):
            words, drops = [], []
            for w in sets[name]:
                (words if single_token(tokenizer, w) else drops).append(w)
            report["dropped"][name] = drops
            if name == "target":
                for s in sets["spares"]:
                    if len(words) >= len(sets["target"]):
                        break
                    if single_token(tokenizer, s):
                        words.append(s)
                        report["filled"].append(s)
            kept[name] = words
        # casing is verified separately and only recorded: the direction is
        # built from the leading-space lowercase form.
        for w in kept["target"] + kept["decoy"]:
            if len(tokenizer(" " + w.capitalize(),
                             add_special_tokens=False)["input_ids"]) != 1:
                report["casing_multi"].append(w)
        # no token in two sets, anywhere
        for name in ("target", "decoy"):
            for w in kept[name]:
                where = f"{case}.{name}"
                if w in seen and seen[w] != where:
                    raise ValueError(f"token {w!r} in {seen[w]} and {where}")
                seen[w] = where
        # equal size after drops
        n = min(len(kept["target"]), len(kept["decoy"]))
        if n < 6:
            raise ValueError(f"case {case}: only {n} usable tokens")
        report["trimmed_to"] = n
        out[case] = {"target": kept["target"][:n], "decoy": kept["decoy"][:n],
                     **report}
    return out
