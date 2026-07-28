"""
Creates a fake results.csv to build and test the enalyze results.py before the real data is collected.
The numbers encode a hypothesis, nto a real result - these will not be reflected in the paper.

RUN:
    python analysis/make_synthetic_results.py
    python analysis/analyze_results.py analysis/results_synthetic.csv --outdir analysis/out
"""

import os
import numpy as np
import pandas as pd

os.makedirs("analysis/out", exist_ok=True)

rng = np.random.default_rng(0) # seeded generater, makes it so every run gives the same synthetic data

conditions = ["clean", "randomized", "recovery", "color"]
cells = ["in_distribution", "new_positions", "new_lighting", "different_object", "distractors"]
seeds = [0]
episodes_per_cell = 15

# Hypothesized success probability per condition and cell; matched conditions expected to do better on their cell
# Numbers below are geenerated through Claude Opus 5, July 2026)
p = {
    "clean":      {"in_distribution": .90, "new_positions": .20, "new_lighting": .40, "different_object": .30, "distractors": .45},
    "randomized": {"in_distribution": .85, "new_positions": .70, "new_lighting": .40, "different_object": .30, "distractors": .45},
    "recovery":   {"in_distribution": .88, "new_positions": .40, "new_lighting": .45, "different_object": .35, "distractors": .55},
    "color":      {"in_distribution": .85, "new_positions": .25, "new_lighting": .45, "different_object": .70, "distractors": .45},
}

rows = []
for c in conditions:
    for cell in cells:
        for s in seeds:
            for e in range(episodes_per_cell):
                success = int(rng.random() < p[c][cell]) # Success has probability p, filling in column with 1, else 0
                rows.append({"condition": c, "eval_cell": cell, "seed": s, "episode": e, "success": success})

pd.DataFrame(rows).to_csv("analysis/out/results_synthetic.csv", index=False)
print(f"wrote analysis/out/results_synthetic.csv ({len(rows)} rows)")