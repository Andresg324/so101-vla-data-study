#!/usr/bin/env python3
"""
tools/make_subsets.py

Builds the nested subsample index for the density sweep (PROTOCOL.md §8.32).

The pool is 520 demonstrations, 52 passes of the 10 training positions cycled in
fixed order, so position = index % 10 + 1 and pass = index // 10 + 1. Passes 10 and
30 are held out in full and enter no training subset.

For each position the 50 remaining passes are shuffled under a single
numpy.random.default_rng(1000) and the first k taken, giving nested subsets
5 ⊂ 10 ⊂ 25 ⊂ 50 per position. That seed governs data selection only and is
distinct from the training seeds 1000 and 2000.

Writes analysis/subsets.json and prints the checks.

usage:
    python tools/make_subsets.py
"""

import json
import os
from collections import Counter

import numpy as np

N_PASSES = 52
N_POSITIONS = 10
HELD_OUT_PASSES = [10, 30]          # 1-indexed, PROTOCOL.md §8.32
BUDGETS = [5, 10, 25, 50]
SEED = 1000
OUT = "analysis/subsets.json"


def episode(pass_idx, position):
    """1-indexed pass and position to 0-indexed episode."""
    return (pass_idx - 1) * N_POSITIONS + (position - 1)


def main():
    available = [p for p in range(1, N_PASSES + 1) if p not in HELD_OUT_PASSES]
    assert len(available) == 50, len(available)

    rng = np.random.default_rng(SEED)
    order = {}                       # position -> shuffled pass order
    for pos in range(1, N_POSITIONS + 1):
        order[pos] = rng.permutation(available).tolist()

    subsets = {}
    for k in BUDGETS:
        eps = []
        for pos in range(1, N_POSITIONS + 1):
            eps += [episode(p, pos) for p in order[pos][:k]]
        subsets[k] = sorted(eps)

    held_out = sorted(episode(p, pos)
                      for p in HELD_OUT_PASSES
                      for pos in range(1, N_POSITIONS + 1))

    # ---- checks ----
    print(f"seed {SEED}, held-out passes {HELD_OUT_PASSES}\n")
    for k in BUDGETS:
        eps = subsets[k]
        by_pos = Counter(e % N_POSITIONS + 1 for e in eps)
        assert len(eps) == k * N_POSITIONS, (k, len(eps))
        assert len(set(eps)) == len(eps), "duplicate episode"
        assert set(by_pos.values()) == {k}, by_pos
        assert not set(eps) & set(held_out), "held-out episode leaked in"
        passes = sorted({e // N_POSITIONS + 1 for e in eps})
        print(f"{k:>2}/position: {len(eps):>3} episodes, "
              f"{k} at every position, passes {min(passes)} to {max(passes)}")

    for a, b in zip(BUDGETS, BUDGETS[1:]):
        assert set(subsets[a]) <= set(subsets[b]), f"{a} not nested in {b}"
        print(f"nested: {a} subset of {b}")

    print(f"\nheld out: {len(held_out)} episodes, "
          f"{len(held_out)//N_POSITIONS} per position")

    # Practice confound check: mean pass index should sit near the middle of the
    # range for every budget, not near the start.
    print("\nmean pass index per budget (50 passes available, midpoint 26.5):")
    for k in BUDGETS:
        m = np.mean([e // N_POSITIONS + 1 for e in subsets[k]])
        print(f"  {k:>2}/position: {m:.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"seed": SEED,
                   "held_out_passes": HELD_OUT_PASSES,
                   "held_out_episodes": held_out,
                   "subsets": {str(k): v for k, v in subsets.items()}},
                  f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()