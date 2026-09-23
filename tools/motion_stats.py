#!/usr/bin/env python3
"""
tools/motion_stats.py
Checks the recordings for motion speed and corrective sub-movements.

Episode duration alone doesn't separate these two. Mean per-step joint displacement
can: if two conditions move at the same speed but one finishes sooner, the
shorter one simply contains less hesitation and re-adjustment.

deg/step here is the mean absolute per-frame change across all six commanded joints, 
gripper included, since the question is how fast the teleoperator moved. 
rollout_motion.py reports a same-named statistic over the five arm joints only. Do not 
compare the two directly.

Usage:
    python tools/motion_stats.py cube-pickup-clean_20260809_105745 \
                                 cube-pickup-color_20260809_183224
"""

import os
import sys
import json

import numpy as np
import glob
import pandas as pd

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/Andresg324"))
OUTDIR = "analysis/out_pace"

DEPART_DEG = 20.0   # Defined threshold for max deviation of a joint from frame 0


DATASETS = {
    "clean":         "cube-pickup-clean_20260809_105745",
    "randomized":    "cube-pickup-randomized_20260809_115825",
    "recovery":      "cube-pickup-recovery_20260809_141725",
    "color":         "cube-pickup-color_20260809_183224",
    "slowpace":      "cube-pickup-color_20260809_130649",
    "density_probe": "cube-pickup-density_20260822_194111",
    "density_sweep": "cube-pickup-densitypool_20260908_125700",
}

def subset_episodes(budget):
    """Episode indices for density data subsets."""
    s = json.load(open("analysis/subsets.json"))
    keep = s["subsets"][str(budget)]
    held = set(s["held_out_episodes"])
    assert not (set(keep) & held), f"budget {budget} overlaps the held-out passes"
    return keep

def dataset_fps(root, default=30):
    p = os.path.join(root, "meta", "info.json")
    if os.path.exists(p):
        return float(json.load(open(p))["fps"])
    print(f" note: no meta/info.json under {root}, assuming {default} fps")
    return default

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
    # if len(sys.argv) < 2:
    #     raise SystemExit(__doc__)
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []

    args = [a for a in sys.argv[1:] if not a.startswith("--subset=")]
    budgets = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--subset=")]
    names = args or list(DATASETS)
    for key in names:
        name = DATASETS.get(key, key)
        root = name if os.path.isdir(name) else os.path.join(CACHE, name)
        actions, episodes = load_actions(root)
        fps = dataset_fps(root)

        for budget in (budgets if (budgets and key == "density_sweep") else [None]):
            if budget is None:
                a_all, e_all, label = actions, episodes, key
            else:
                keep = subset_episodes(budget)
                m = np.isin(episodes, keep)
                a_all, e_all = actions[m], episodes[m]
                label = f"{key}_{budget}"

            # Episode 0 is kept. The warm-up convention applies to rollouts only; in a training
            # dataset episode 0 is demonstration T1, which calibrate_pose.py maps by index.
            per_step, durations, depart, per_step_post = [], [], [], []
            for ep in np.unique(e_all):
                a = a_all[e_all == ep]
                if len(a) < 2:
                    continue

                per_step.append(np.abs(np.diff(a, axis=0)).mean())
                durations.append(len(a)/fps)
                dev = np.abs(a-a[0]).max(axis=1)
                hit = np.flatnonzero(dev > DEPART_DEG)
                d0 = int(hit[0]) if len(hit) else len(a)
                depart.append(d0 / fps)
                post = a[d0:]
                per_step_post.append(np.abs(np.diff(post, axis=0)).mean() if len(post) > 1 else np.nan)

            frames = int(len(a_all))
            rows.append({
                "dataset": label,
                "episodes": len(durations),
                "fps": fps,
                "frames": frames,
                "frames_per_ep": frames / len(durations),
                "sec_per_ep": float(np.mean(durations)),
                "deg_per_step": float(np.mean(per_step)),
                "deg_per_sec": float(np.mean(per_step)) * fps,
                "total_path_deg": float(np.mean(per_step)) * (frames / len(durations)),
                "depart_s": float(np.mean(depart)),
                "depart_sd": float(np.std(depart)),
                "sec_per_ep_sd": float(np.std(durations)),
                "post_depart_s": float(np.mean(durations)) - float(np.mean(depart)),
                "deg_per_step_post": float(np.nanmean(per_step_post)),
            })

    t = pd.DataFrame(rows)
    base = t.loc[t.dataset == "clean", "sec_per_ep"]
    if len(base):
        t["vs_clean_pct"] = (t.sec_per_ep / float(base.iloc[0]) - 1) * 100
    else:
        print(" note: no 'clean' dataset in this run, vs_clean_pct omitted from pace.csv")
    t.round(4).to_csv(os.path.join(OUTDIR, "pace.csv"), index=False)
    print(t.round(3).to_string(index=False))
    print(f"\nsaved to {OUTDIR}/pace.csv")

if __name__ == "__main__":
    main()