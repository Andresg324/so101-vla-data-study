#!/usr/bin/env python3
"""
tools/motion_stats.py
Checks the recordings for motion speed and corrective sub-movements.

Episode duration alone doesn't separate these two. Mean per-step joint displacement
can: if two conditions move at the same speed but one finishes sooner, the
shorter one simply contains less hesitation and re-adjustment.

Usage:
    python tools/motion_stats.py cube-pickup-clean_20260809_105745 \
                                 cube-pickup-color_20260809_183224
"""

import os
import sys

import numpy as np
import glob
import pandas as pd

CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/Andresg324")
FPS = 30

def load_actions(root):
    #Actions per frame and the episode ID of each frame
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}/data")

    df = pd.concat(
        [pd.read_parquet(f, columns=["action", "episode_index", "frame_index"]) for f in files], ignore_index=True,
    )

    df = df.sort_values(["episode_index", "frame_index"], kind="stable")
    # Read straight from the underlying table, not the video

    actions = np.stack(df["action"].to_numpy())
    episodes = df["episode_index"].to_numpy()
    return actions, episodes

def main():
    print(f"{'dataset':45s} {'sec/demo':>9s} {'deg/step':>9s} {'deg/sec':>9s}")
    for name in sys.argv[1:]:
        root = name if os.path.isdir(name) else os.path.join(CACHE, name)
        actions, episodes = load_actions(root)

        per_step, durations = [], []
        for ep in np.unique(episodes):
            a = actions[episodes == ep]
            if len(a) < 2:
                continue
            # Mean absolute change per joint, per timestep, averaged over the joints
            # This is the speed the operator moved at, independent of how long the episodes ran
            per_step.append(np.abs(np.diff(a, axis = 0)).mean())
            durations.append(len(a)/FPS)

        sec = float(np.mean(durations))
        step = float(np.mean(per_step))

        print(f"{os.path.basename(root):45s} {sec:9.1f} {step:9.4f} {step*FPS:9.3f}")

if __name__ == "__main__":
    main()