#!/usr/bin/env python3
"""
tools/check_norm_stats.py

Were sweep normalizers fit on the training subset or on the whole pool? Each sweep
policy trained on a subset via --dataset.episodes. If its saved action statistics
match the pool's dataset-wide metadata, the normalizer saw all 520 demonstrations,
including the 20 held out (PROTOCOL.md §8.32). Statistics only, not training signal.

The subset column is a plain population std over the subset's frames, so expect it
close to LeRobot's computation rather than identical. The pool comparison decides.

RUN: python tools/check_norm_stats.py
"""

import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deterministic_loss import POLICIES, ALL_SWEEP, DATASETS, HF_USER, OUTDIR
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

arr = lambda x: np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=float)

def budget(key):
    return key.replace("density", "").split("_")[0]

def main():
    pool_repo = f"{HF_USER}/{DATASETS['pool']}"
    meta = LeRobotDatasetMetadata(pool_repo)
    pool_std = arr(meta.stats["action"]["std"])

    root = str(meta.root)
    pattern = os.path.join(root, "data", "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        from huggingface_hub import snapshot_download
        snapshot_download(pool_repo, repo_type="dataset",
                          allow_patterns=["data/**"], local_dir=root)
        files = glob.glob(pattern, recursive=True)
    if not files:
        raise SystemExit(f"no pool parquet under {root}")

    df = pd.concat(pd.read_parquet(f, columns=["action", "episode_index"]) for f in files)
    subsets = json.load(open("analysis/subsets.json"))["subsets"]
    
    rows = []
    for key in ALL_SWEEP:
        repo = f"{HF_USER}/{POLICIES[key]}"
        pol = SmolVLAPolicy.from_pretrained(repo)
        pre, _ = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=repo)
        n = [s for s in pre.steps if type(s).__name__ == "NormalizerProcessorStep"][0]
        ckpt = arr(n.stats["action"]["std"])
        keep = subsets[budget(key)]
        sub = np.stack(df[df.episode_index.isin(keep)]["action"].to_numpy()).std(axis=0)
        rows.append({"policy": key, "budget": budget(key),
                     "max_diff_vs_pool": float(np.abs(ckpt - pool_std).max()),
                     "max_diff_vs_subset": float(np.abs(ckpt - sub).max())})
        del pol

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(OUTDIR, "norm_stats_check.csv"), index=False)
    print(f"\nwrote {OUTDIR}/norm_stats_check.csv")

if __name__ == "__main__":
    main()