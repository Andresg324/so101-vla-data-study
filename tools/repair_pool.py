#!/usr/bin/env python3
"""
tools/repair_pool.py

Repairs a LeRobotDataset whose parquet holds rows that no episode claims.

A demonstration interrupted mid-episode can have its actions flushed to the data
parquet without a corresponding video segment or episode metadata row. The
symptom is `total_frames` in meta/info.json reading lower than the parquet row
count, with one episode holding roughly double its neighbours' frames and a
`frame_index` that resets to 0 partway through.

This breaks the episode-subset sampler rather than anything obvious: it reads
absolute row offsets from `dataset_from_index` and `dataset_to_index` in the
episode metadata, so the orphaned rows push every later episode's offsets out of
alignment and the sampler raises a KeyError for a frame index it has no entry
for.

To fix this it takes two passes. First drop the orphaned rows and renumber `index` contiguously.
Then rebuild every episode's row offsets from the per-episode frame counts,
rather than shifting the affected ones.

Video files and their timestamps are untouched, because the video side already
excludes the orphaned frames. Only the data parquet and the episode metadata
change, so a push after repair uploads a few megabytes rather than the whole
dataset.

Back up the dataset before running. Verify afterwards with the checks printed at
the end.

usage:
    # find the orphaned block
    python tools/repair_pool.py <root> --diagnose

    # repair it
    python tools/repair_pool.py <root> --drop 74639 75248
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def data_files(root):
    return sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"),
                            recursive=True))


def meta_file(root):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    if len(files) != 1:
        raise SystemExit(f"expected one episode metadata file, found {len(files)}")
    return files[0]


def diagnose(root):
    """Locate the orphaned block. Prints the index range to pass to --drop."""
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    df = pd.concat([pd.read_parquet(p, columns=["index", "episode_index", "frame_index"])
                    for p in data_files(root)], ignore_index=True)
    diff = len(df) - info["total_frames"]
    print(f"meta total_frames {info['total_frames']}, data rows {len(df)}, diff {diff}")
    if diff == 0:
        print("nothing to repair")
        return

    m = pd.read_parquet(meta_file(root))
    m["meta_n"] = m["stats/timestamp/count"].apply(lambda v: int(v[0]))
    actual = df.groupby("episode_index").size().rename("data_n")
    j = m.set_index("episode_index")[["meta_n"]].join(actual)
    bad = j[j.meta_n != j.data_n]
    print(f"\nepisodes where metadata and data disagree: {len(bad)}")
    print(bad.to_string())

    for ep in bad.index:
        g = df[df.episode_index == ep].sort_values("index")
        resets = np.flatnonzero(g.frame_index.values == 0)
        print(f"\nepisode {ep}: {len(g)} rows, frame_index resets to 0 at {resets}")
        if len(resets) > 1:
            k = int(resets[1])
            lo = int(g["index"].iloc[0])
            hi = int(g["index"].iloc[k - 1])
            print(f"  orphaned block is the first {k} rows: --drop {lo} {hi}")
            print(f"  retained block is the remaining {len(g) - k} rows")


def repair(root, lo, hi):
    n_bad = hi - lo + 1
    print(f"dropping rows with index {lo} to {hi} inclusive ({n_bad} rows)\n")

    # ---- data parquet: drop the block, renumber index contiguously ----
    before = 0
    for p in data_files(root):
        df = pd.read_parquet(p)
        before += len(df)
        keep = ~df["index"].between(lo, hi)
        df = df[keep].copy()
        df["index"] = np.where(df["index"] > hi, df["index"] - n_bad, df["index"])
        df.to_parquet(p, index=False)
        print(f"  {os.path.basename(p)}: dropped {int((~keep).sum())}")

    after = pd.concat([pd.read_parquet(p, columns=["index"]) for p in data_files(root)])
    print(f"\nrows {before} -> {len(after)}")
    contiguous = after["index"].max() - after["index"].min() + 1 == len(after)
    print(f"index contiguous: {contiguous}, max {after['index'].max()}")
    if not contiguous:
        raise SystemExit("index is not contiguous after the drop; do not push this")

    # ---- episode metadata: rebuild offsets from the frame counts ----
    # Rebuilt rather than shifted. Shifting requires deciding whether each of an
    # episode's two offsets falls before or after the deleted block, and the
    # episode containing the block has one on each side.
    mp = meta_file(root)
    m = pd.read_parquet(mp).sort_values("episode_index").reset_index(drop=True)
    counts = m["stats/timestamp/count"].apply(lambda v: int(v[0])).to_numpy()
    m["dataset_from_index"] = np.concatenate([[0], np.cumsum(counts)[:-1]])
    m["dataset_to_index"] = np.cumsum(counts)
    m.to_parquet(mp, index=False)
    print(f"rebuilt offsets for {len(m)} episodes, last ends at {int(counts.sum())}")

    # ---- info.json ----
    ip = os.path.join(root, "meta", "info.json")
    info = json.load(open(ip))
    info["total_frames"] = int(counts.sum())
    json.dump(info, open(ip, "w"), indent=4)
    print(f"info.json total_frames -> {info['total_frames']}")

    print("\nverify with:")
    print(f"  python tools/repair_pool.py {root} --diagnose")
    print(f"  python tools/verify_pool_mapping.py {root}")
    print("  then load a subset through LeRobotDataset(episodes=[...]) before pushing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="local dataset root")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--drop", nargs=2, type=int, metavar=("LO", "HI"),
                    help="inclusive index range of the orphaned block")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    if args.diagnose or not args.drop:
        diagnose(root)
        return
    repair(root, args.drop[0], args.drop[1])


if __name__ == "__main__":
    main()