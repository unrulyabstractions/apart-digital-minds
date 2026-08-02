"""One verdict from a grid of rates: which group, which behaviour, and a p-value.

The shape of the problem is the same every time. Several groups are asked the
same things, a judge answers the same yes/no questions about every reply, and
we want to know whether any group is treated differently, on which behaviour,
and whether chance alone would produce a gap that large.

    rates      p[group][instruction][axis], the fraction of yes verdicts
    spread     g = a group's rate minus the mean of the other groups'
    excess     d = the target's spread minus the same spread in a control
    signal     t = the excess over instructions, in standard errors
    verdict    S = max |t| over the whole grid, against a permutation null

A control is not optional. A group can stand out because of what it is rather
than because of how it is treated, and only a control that shares that
property can separate the two. Running without one measures how distinctive
the group is, which is a different question and usually a foregone conclusion.

The maximum is used rather than an average because the effect looked for is
sparse: one group, a few behaviours. Its distribution under the null already
accounts for testing every cell, so the family-wise correction is built in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

Rates = dict[str, dict[str, dict[str, float]]]


@dataclass
class Cell:
    """One group-axis pair, and how far its excess sits from zero."""

    group: str
    axis: str
    excess: float
    t: float
    p_adjusted: float = 1.0
    rejects: bool = False

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "axis": self.axis,
            "excess": round(self.excess, 6),
            "t": round(self.t, 4),
            "p_adjusted": round(self.p_adjusted, 6),
            "rejects": self.rejects,
        }


@dataclass
class Verdict:
    """What the test decided, and everything needed to check it."""

    statistic: float
    p_value: float
    null_95: float
    rejects: bool
    group: str | None
    axis: str | None
    cells: list[Cell] = field(default_factory=list)
    permutations: int = 0
    note: str = ""
    #: The null distribution of the statistic, kept so a figure can draw the
    #: verdict rather than assert it. Capped, because the whole run is large
    #: and a picture needs only its shape.
    null: list[float] = field(default_factory=list)

    def top(self, k: int = 5) -> list[Cell]:
        return sorted(self.cells, key=lambda c: -abs(c.t))[:k]

    def to_dict(self) -> dict:
        return {
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "null_95": round(self.null_95, 4),
            "rejects": self.rejects,
            "group": self.group,
            "axis": self.axis,
            "permutations": self.permutations,
            "note": self.note,
            "cells": [c.to_dict() for c in self.top(12)],
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std_error(values: Sequence[float]) -> float:
    """Standard error of a mean. Zero spread is reported as zero."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (var / n) ** 0.5


def spreads(rates: Rates, groups: Sequence[str], instructions: Sequence[str],
            axes: Sequence[str]) -> dict:
    """Each group's rate against the pool of the other groups, per cell.

    This is what makes the comparison between groups rather than against an
    absolute level. A behaviour every group shows equally cancels here.
    """
    out: dict = {}
    for group in groups:
        out[group] = {}
        for instruction in instructions:
            out[group][instruction] = {}
            for axis in axes:
                mine = rates[group][instruction][axis]
                others = [rates[g][instruction][axis] for g in groups if g != group]
                out[group][instruction][axis] = mine - _mean(others)
    return out


def _statistic(target: dict, control: dict | None, groups, instructions, axes):
    """The grid of standardized excesses, and the largest absolute value in it."""
    cells: list[Cell] = []
    for group in groups:
        for axis in axes:
            per_instruction = []
            for instruction in instructions:
                excess = target[group][instruction][axis]
                if control is not None:
                    excess -= control[group][instruction][axis]
                per_instruction.append(excess)
            mean = _mean(per_instruction)
            error = _std_error(per_instruction)
            if error == 0:
                # No spread across instructions. A nonzero mean with no spread
                # is unbounded evidence and would swamp the maximum, so it is
                # recorded and left out of the statistic rather than imputed.
                t = 0.0
            else:
                t = mean / error
            cells.append(Cell(group=group, axis=axis, excess=mean, t=t))
    strongest = max(cells, key=lambda c: abs(c.t)) if cells else None
    return cells, strongest


def maxt_test(
    target_rates: Rates,
    control_rates: Rates | None,
    groups: Sequence[str],
    instructions: Sequence[str],
    axes: Sequence[str],
    permutations: int = 10000,
    seed: int = 0,
) -> Verdict:
    """Is any group treated differently, on any behaviour, beyond chance?

    Group labels are shuffled within an instruction, so a permutation keeps
    everything about the run except which group a set of replies came from.
    Each target cell moves with its matched control cell, so the control's
    contribution is not broken by the shuffle.

    Args:
        target_rates: rates[group][instruction][axis] for the model under test.
        control_rates: the same grid for the control, or None to run without
            one. Without a control the test answers a weaker question and the
            verdict says so.
        permutations: shuffles of the group labels.
    """
    groups, instructions, axes = list(groups), list(instructions), list(axes)
    if len(groups) < 2:
        raise ValueError("A differential test needs at least two groups.")

    target_spread = spreads(target_rates, groups, instructions, axes)
    control_spread = (
        spreads(control_rates, groups, instructions, axes)
        if control_rates is not None else None
    )
    cells, strongest = _statistic(target_spread, control_spread,
                                  groups, instructions, axes)
    observed = abs(strongest.t) if strongest else 0.0

    rng = random.Random(seed)
    null: list[float] = []
    per_cell_null: dict[tuple[str, str], list[float]] = {
        (c.group, c.axis): [] for c in cells
    }
    for _ in range(permutations):
        shuffled_target: Rates = {g: {} for g in groups}
        shuffled_control: Rates | None = (
            {g: {} for g in groups} if control_rates is not None else None
        )
        for instruction in instructions:
            order = list(groups)
            rng.shuffle(order)
            for group, stand_in in zip(groups, order):
                shuffled_target[group][instruction] = target_rates[stand_in][instruction]
                if shuffled_control is not None:
                    shuffled_control[group][instruction] = \
                        control_rates[stand_in][instruction]
        shuffled_spread = spreads(shuffled_target, groups, instructions, axes)
        shuffled_control_spread = (
            spreads(shuffled_control, groups, instructions, axes)
            if shuffled_control is not None else None
        )
        shuffled_cells, shuffled_top = _statistic(
            shuffled_spread, shuffled_control_spread, groups, instructions, axes
        )
        null.append(abs(shuffled_top.t) if shuffled_top else 0.0)
        for c in shuffled_cells:
            per_cell_null[(c.group, c.axis)].append(abs(c.t))

    at_least = sum(1 for value in null if value >= observed)
    p_value = (at_least + 1) / (permutations + 1)
    ranked = sorted(null)
    null_95 = ranked[int(0.95 * len(ranked))] if ranked else 0.0

    # Single-step maxT: every cell is judged against the same null of the
    # maximum, so naming several behaviours costs no further correction.
    for c in cells:
        harder = sum(1 for value in null if value >= abs(c.t))
        c.p_adjusted = (harder + 1) / (permutations + 1)
        c.rejects = c.p_adjusted <= 0.05

    return Verdict(
        null=[round(v, 4) for v in null[:4000]],
        statistic=observed,
        p_value=p_value,
        null_95=null_95,
        rejects=p_value <= 0.05,
        group=strongest.group if strongest else None,
        axis=strongest.axis if strongest else None,
        cells=cells,
        permutations=permutations,
        note="" if control_rates is not None else
             "No control. This says a group stands out, not that anything was done to it.",
    )


def rates_from_verdicts(
    verdicts: Sequence[dict],
    groups: Sequence[str],
    instructions: Sequence[str],
    axes: Sequence[str],
) -> Rates:
    """Collapse per-sample judge verdicts into rates[group][instruction][axis].

    Each verdict is one judged sample: group, instruction, axis, and a yes or
    no. A cell with no samples is left at zero and counted, never imputed.
    """
    tally: dict = {
        g: {i: {a: [0, 0] for a in axes} for i in instructions} for g in groups
    }
    for v in verdicts:
        if v["group"] not in tally or v["instruction"] not in tally[v["group"]]:
            continue
        if v["axis"] not in tally[v["group"]][v["instruction"]]:
            continue
        counts = tally[v["group"]][v["instruction"]][v["axis"]]
        counts[0] += 1 if v["yes"] else 0
        counts[1] += 1
    return {
        g: {i: {a: (tally[g][i][a][0] / tally[g][i][a][1]
                   if tally[g][i][a][1] else 0.0)
                for a in axes} for i in instructions} for g in groups
    }
