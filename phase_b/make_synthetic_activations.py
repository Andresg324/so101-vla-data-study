#!/usr/bin/env python3
"""
Fake 'activations' to build and test the probe before the real data.
Encodes a hypothesis (derived by Claude Opus 5, July 2026), not a result
"""

import numpy as np
import os

os.makedirs("phase_b/out", exist_ok=True)

rng = np.random.default_rng(0) # Sets same seed so reruns are the same numbers

conditions = ["clean", "randomized", "recovery", "color"]
cells = ["in_distribution", "new_positions", "new_lighting", "different_object", "distractors"]
episodes_per_cell = 15
steps_per_episode = 30
dim = 64 # assuming the hidden actvations are a 64-dimensional vector, to validate once data is collected

decodability = {"clean": 0.10, "randomized": 0.16, "recovery": 0.3, "color": 0.13}
cell_base_rate = {"in_distribution": 0.85, "new_positions": 0.30, "new_lighting": 0.45,
                  "different_object": 0.35, "distractors": 0.55}

success_dir = rng.standard_normal(dim)
cell_dir = {c: rng.standard_normal(dim) for c in cells}

X, cond, cell_col, epi, succ, tfe = [], [], [], [], [], []
ep_id = 0
for c in conditions:
    for cell in cells:
        for _ in range(episodes_per_cell):
            outcome = int(rng.random() < cell_base_rate[cell])
            for t in range(steps_per_episode):
                noise = rng.standard_normal(dim)
                strength = decodability[c] * (t / steps_per_episode)
                vec = (noise + strength * (1 if outcome else -1) * success_dir + 0.3 * cell_dir[cell]) # Adding a cell identity leak (confound)
                X.append(vec); cond.append(c); cell_col.append(cell); epi.append(ep_id)
                succ.append(outcome); tfe.append(steps_per_episode - 1 - t)
            ep_id += 1

np.savez("phase_b/out/activations_synthetic.npz", X=np.array(X), condition=np.array(cond), eval_cell=np.array(cell_col),
         episode=np.array(epi), success=np.array(succ), t_from_end=np.array(tfe))
print(f"Wrote phase_b/out/activations_synthetic.npz ({len(X)} samples)")