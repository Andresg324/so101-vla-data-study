#!/usr/bin/env python3
"""
analysis/analyze_followup.py

Follow-up session on the rebuilt bench, September 24 (PROTOCOL.md §8.35 to §8.38).

  Reach probe     Success at (19.0, 12.7) by density and seed, Wilson 95%. The §8.38
                  prediction is checked at seed 2000.
  Randomized      August Randomized on the rebuilt bench against August (§8.36), and the
                  RunPod retrain against the §8.37 cutoff and August's 6/15. Newcombe
                  interval on every difference. The retrain against the rebuilt run is
                  reported but was not registered.
  Failure modes   Randomized seed 1000, August against rebuilt, exploratory.

RUN: python analysis/analyze_followup.py
"""

import os
import sys
import numpy as np

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import newcombe_diff, wilson_ci

FOL = "documents/followup.csv"
REG = "documents/results_full.csv"
OUTDIR = "analysis/out_followup"
REACH = ["density5", "density10", "density25", "density50"]
KNOWN = set(REACH) | {"randomized", "randomized-runpod"}


def counts(df, cond, seed, cell):
    g = df[(df.condition == cond) & (df.seed == seed) & (df.eval_cell == cell)]
    if g.empty:
        raise SystemExit(f"no rows for {cond}, seed {seed}, {cell}")
    return int(g.success.sum()), len(g)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    f = pd.read_csv(FOL)
    aug = pd.read_csv(REG)

    unknown = set(f.condition) - KNOWN
    if unknown:
        raise SystemExit(f"conditions in {FOL} not expected: {sorted(unknown)}")

    # Reach probe, §8.35 and §8.38
    rows = []
    for seed in (1000, 2000):
        for c in REACH:
            k, n = counts(f, c, seed, "reach_e5")
            lo, hi = wilson_ci(k, n)
            rows.append({"condition": c, "seed": seed, "successes": k, "n": n,
                         "ci_low": lo, "ci_high": hi})
    reach = pd.DataFrame(rows)
    s2 = reach[reach.seed == 2000].set_index("condition").successes
    held = bool((s2[["density25", "density50"]] >= 3).all()
                and (s2[["density5", "density10"]] <= 1).all())

    # Randomized follow-up, §8.36 and §8.37
    rb_t6 = counts(f, "randomized", 1000, "in_distribution_rebuilt")
    rt_t6 = counts(f, "randomized-runpod", 1000, "in_distribution_rebuilt")
    au_t6 = counts(aug, "randomized", 1000, "in_distribution")
    comps = [("§8.36 rebuilt vs August, T6", rb_t6, au_t6),
             ("§8.37 retrain vs August, T6", rt_t6, au_t6),
             ("not registered: retrain vs rebuilt, T6", rt_t6, rb_t6),
             ("rebuilt vs August, New Positions",
              counts(f, "randomized", 1000, "new_positions_rebuilt"),
              counts(aug, "randomized", 1000, "new_positions"))]
    rows = []
    for name, (k1, n1), (k2, n2) in comps:
        diff, lo, hi = newcombe_diff(k1, n1, k2, n2)
        rows.append({"comparison": name, "a": f"{k1}/{n1}", "b": f"{k2}/{n2}",
                     "difference": diff, "diff_ci_low": lo, "diff_ci_high": hi})
    rand = pd.DataFrame(rows)

    # Failure modes, exploratory
    both = pd.concat([
        aug[(aug.condition == "randomized") & (aug.seed == 1000)
            & aug.eval_cell.isin(["in_distribution", "new_positions"])],
        f[f.condition.isin(["randomized", "randomized-runpod"])]], ignore_index=True)
    fm = pd.crosstab([both.condition, both.eval_cell], both.failure_mode)

    reach.round(4).to_csv(os.path.join(OUTDIR, "reach.csv"), index=False)
    rand.round(4).to_csv(os.path.join(OUTDIR, "randomized_followup.csv"), index=False)
    fm.to_csv(os.path.join(OUTDIR, "randomized_failure_modes.csv"))

    # Failure modes per policy at the held-out positions and the reach point
    LABEL = "failure_mode"   # set from the csv header
    den = pd.read_csv("documents/density.csv")
    fol = pd.read_csv("documents/followup.csv")
    rows = pd.concat([den[den.eval_cell == "new_positions"], fol[fol.eval_cell == "reach_e5"]])
    fm_fol = (rows.groupby(["condition", "seed", "eval_cell", "instance", LABEL])
            .size().rename("n").reset_index())
    fm_fol.to_csv(os.path.join(OUTDIR, "failure_modes_by_policy.csv"), index=False)
    print(fm_fol[fm_fol.instance.isin(["E3", "E5", "reach_e5"])].to_string(index=False))

    # Policy-level exact test for the reach split, and the per-seed limit
    from scipy.stats import permutation_test
    high, low = [4, 4, 3, 3], [0, 0, 0, 0]
    res = permutation_test((high, low), lambda a, b: np.mean(a) - np.mean(b),
                        permutation_type="independent", alternative="greater",
                        n_resamples=np.inf)
    print(f"reach split, 8 policies, exact one-sided p = {res.pvalue:.4f} (1/70 = {1/70:.4f})")
    print(f"per seed, 2 vs 2, smallest attainable p = {1/6:.3f}")

    print("--- Reach probe, Wilson 95% ---")
    print(reach.round(3).to_string(index=False))
    print(f"\n§8.38 seed 2000 prediction held: {held}")
    print(f"§8.37 cutoff: retrain {rt_t6[0]}/{rt_t6[1]}, "
          f"{'10 or more, training environment' if rt_t6[0] >= 10 else '9 or fewer, data'}")
    print("\n--- Randomized follow-up, Newcombe ---")
    print(rand.round(3).to_string(index=False))
    print("\n--- Randomized failure modes, seed 1000, exploratory ---")
    print(fm.to_string())
    print(f"\nSaved to {OUTDIR}/")


if __name__ == "__main__":
    main()