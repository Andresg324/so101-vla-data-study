#!/usr/bin/env python3
"""
analysis/analyze_density.py

Registered analyses for the density sweep.

  Per-cell rates   Wilson 95% intervals for every policy, seed and cell.
  Trend            Cochran-Armitage across the four densities, scores 5, 10, 25, 50,
                   separately per seed and separately at T6 and New Positions. Two-sided.
                   Degenerate when every cell is at floor or ceiling; reported as such.
  Cross-condition  Sweep policies against August Clean, Randomized and the density probe,
                   same seed, Newcombe interval on the difference. At T6, 15 against 15.
                   At New Positions the sweep's first 3 episodes per position are used,
                   matching August's 3.

RUN: python analysis/analyze_density.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import newcombe_diff, wilson_ci

DENS = "documents/density.csv"
REG = "documents/results_full.csv"
EXP = "documents/exploratory.csv"
OUTDIR = "analysis/out_density"

SWEEP = ["density5", "density10", "density25", "density50"]
SCORES = {"density5": 5, "density10": 10, "density25": 25, "density50": 50}
CONTROLS = ["density25-fixedstep", "density50-fixedstep"]
CELLS = ["in_distribution", "distractors", "spot_check", "new_positions"]
TREND_CELLS = ["in_distribution", "new_positions"]
AUGUST = ["clean", "randomized", "density"]
CROSS_CELLS = ["in_distribution", "new_positions"]


def cochran_armitage(k, n, x):
    k, n, x = (np.asarray(a, float) for a in (k, n, x))
    N, K = n.sum(), k.sum()
    if K == 0 or K == N:
        return np.nan, np.nan, "degenerate: every cell at floor or ceiling"
    p = K / N
    t = (x * (k - n * p)).sum()
    var = p * (1 - p) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / N)
    z = t / np.sqrt(var)
    return z, 2 * norm.sf(abs(z)), ""


def counts(df, cond, seed, cell, first3=False):
    g = df[(df.condition == cond) & (df.seed == seed) & (df.eval_cell == cell)]
    g = g.sort_values("episode")
    if first3:
        g = g.groupby("instance", group_keys=False).head(3)
    return int(g.success.sum()), len(g)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = pd.read_csv(DENS)
    reg = pd.read_csv(REG)
    exp = pd.read_csv(EXP)
    aug = pd.concat([reg, exp[exp.condition == "density"]], ignore_index=True)

    policies = SWEEP + CONTROLS
    unknown = set(d.condition) - set(policies)
    if unknown:
        raise SystemExit(f"conditions in {DENS} not in SWEEP or CONTROLS: {sorted(unknown)}")

    # ---------- per-cell rates ----------
    rows = []
    for cond in policies:
        for seed in sorted(d[d.condition == cond].seed.unique()):
            for cell in CELLS:
                k, n = counts(d, cond, seed, cell)
                lo, hi = wilson_ci(k, n)
                rows.append({"condition": cond, "seed": seed, "eval_cell": cell,
                             "successes": k, "n": n, "rate": k / n if n else np.nan,
                             "ci_low": lo, "ci_high": hi})
    cells = pd.DataFrame(rows)

    # ---------- trend ----------
    rows = []
    for seed in sorted(d[d.condition.isin(SWEEP)].seed.unique()):
        for cell in TREND_CELLS:
            kn = [counts(d, c, seed, cell) for c in SWEEP]
            z, p, note = cochran_armitage([a for a, _ in kn], [b for _, b in kn],
                                          [SCORES[c] for c in SWEEP])
            rows.append({"seed": seed, "eval_cell": cell,
                         "successes": " ".join(f"{a}/{b}" for a, b in kn),
                         "z": z, "p_two_sided": p, "note": note})
    trend = pd.DataFrame(rows)

    # ---------- cross-condition ----------
    rows = []
    for cond in policies:
        for seed in sorted(d[d.condition == cond].seed.unique()):
            for ref in AUGUST:
                for cell in CROSS_CELLS:
                    first3 = cell == "new_positions"
                    k2, n2 = counts(aug, ref, seed, cell)
                    if n2 == 0:
                        continue
                    k1, n1 = counts(d, cond, seed, cell, first3=first3)
                    diff, lo, hi = newcombe_diff(k1, n1, k2, n2)
                    rows.append({"condition": cond, "seed": seed, "reference": ref,
                                 "eval_cell": cell, "sweep": f"{k1}/{n1}",
                                 "august": f"{k2}/{n2}", "difference": diff,
                                 "diff_ci_low": lo, "diff_ci_high": hi})
    cross = pd.DataFrame(rows)

    for name, t in [("cell_rates", cells), ("trend", trend), ("cross_condition", cross)]:
        t.round(4).to_csv(os.path.join(OUTDIR, f"{name}.csv"), index=False)

    print("--- Success by policy, seed and cell (Wilson 95%) ---")
    print(cells.round(3).to_string(index=False))
    print("\n--- Cochran-Armitage across density, per seed ---")
    print(trend.round(4).to_string(index=False))
    print("\n--- Sweep against August conditions, same seed ---")
    print(cross.round(3).to_string(index=False))
    print(f"\nSaved to {OUTDIR}/")


if __name__ == "__main__":
    main()