#!/usr/bin/env python3
"""
tools/verify_pool_mapping.py

Verifies the index-to-position mapping for a cycled collection (PROTOCOL.md §3, §8.19)
without refitting anything. Applies the position assigned by episode index and reports
the within-position joint spread; the ten positions span about 155 degrees of bearing,
so a one-episode offset anywhere would blow up the pan column at the affected positions.

Does not touch tools/calibrate_pose.py, whose TRAIN dataset is the calibration itself
and must not be repointed.

usage:
    python tools/verify_pool_mapping.py <dataset root>
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grasp import grasp_pose
from calibrate_pose import T, JOINTS, azimuth, azimuth_fit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="local dataset root")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable")

    rows, skipped = [], []
    for ep, g in df.groupby("episode_index"):
        A = np.asarray([np.asarray(a, float) for a in g["action"].to_numpy()])
        pose, _, _ = grasp_pose(A)
        if pose is None:
            skipped.append(int(ep))
            continue
        rows.append(dict(zip(JOINTS, pose), pos=(int(ep) % 10) + 1, ep=int(ep)))

    total = df.episode_index.nunique()
    print(f"{len(skipped)} of {total} episodes had no detectable grasp"
          + (f": {skipped}" if skipped else ""))

    d = pd.DataFrame(rows)
    fit, _, _ = azimuth_fit()
    d["fit_az"] = fit.predict(d[["pan"]].to_numpy())
    d["true_az"] = [azimuth(*T[p]) for p in d.pos]
    d["err"] = d.fit_az - d.true_az

    print("\ncount, bearing through the fit, and error by position:")
    s = d.groupby("pos").agg(n=("pan", "size"), pan_mean=("pan", "mean"),
                             pan_std=("pan", "std"), fit_az=("fit_az", "mean"),
                             true_az=("true_az", "first"), err=("err", "mean"))
    print(s.round(2).to_string())

    print("\nwithin-position spread (std across all demonstrations at each position):")
    print(d.groupby("pos")[JOINTS].std().round(2).to_string())

    counts = s["n"].unique()
    if len(counts) == 1:
        print(f"\nall positions have {counts[0]} demonstrations")
    else:
        print(f"\nWARNING: uneven counts per position: {s['n'].to_dict()}")

    worst = s["pan_std"].max()
    print(f"largest pan spread: {worst:.2f} deg at position "
          f"{int(s['pan_std'].idxmax())}. A one-episode offset would produce tens "
          f"of degrees, so the mapping holds if this is single digits.")


if __name__ == "__main__":
    main()