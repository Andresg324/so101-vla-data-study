#!/usr/bin/env python3
"""
tools/norm_sweep.py

PROTOCOL.md §8.31, step 1. Measures ||theta_ft - theta_base|| per parameter group
for every fine-tuned checkpoint against the SmolVLA base, relative to the base
weight norm of the same group.

Establishes two things. First, which parameter groups actually moved during
fine-tuning: the paper claims a frozen vision encoder and language backbone with
roughly 100M of 450M trainable, and that claim currently rests on config flags
rather than on the weights. Second, a per-checkpoint scalar for how far each
dataset pushed the model, as context for the loss matrix.

Two sources of false positives are handled. Normalization statistics are state
dict buffers and are per dataset, so ten checkpoints trained on six datasets
differ there with the backbone genuinely frozen; excluded by name. And a
checkpoint saved in a different dtype than the base shows a small delta on every
tensor from round-tripping, hence a relative threshold rather than a test against
zero, and a printed dtype column.

Reading the output:
    relative norm > 1e-2   the group trained
    relative norm < 1e-4   the group is frozen
    in between             check the dtype column; matching dtypes there is a
                           real finding and should be investigated

Two sanity checks: parameters summed over the moved groups should land near
100M, and all ten checkpoints should show the same moved groups. A checkpoint
that differs means a training config drifted between runs.

usage:
    python tools/norm_sweep.py
    python tools/norm_sweep.py --depth 5 --only clean_s1000 randomized_s1000
"""

import argparse
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

BASE_REPO = "lerobot/smolvla_base"
WEIGHT_FILE = "model.safetensors"      # confirm against the inventory listing

CKPTS = {
    "clean_s1000":      "Andresg324/smolvla-cube-clean",
    "clean_s2000":      "Andresg324/smolvla-cube-clean-seed2000",
    "randomized_s1000": "Andresg324/smolvla-cube-randomized",
    "randomized_s2000": "Andresg324/smolvla-cube-randomized-seed2000",
    "recovery_s1000":   "Andresg324/smolvla-cube-recovery",
    "recovery_s2000":   "Andresg324/smolvla-cube-recovery-seed2000",
    "color_s1000":      "Andresg324/smolvla-cube-color",
    "color_s2000":      "Andresg324/smolvla-cube-color-seed2000",
    "color_slowpace":   "Andresg324/smolvla-cube-color-slowpace",
    "density":          "Andresg324/smolvla-cube-density",
}

BUFFER_MARKERS = (
    "normalize_inputs", "normalize_targets", "unnormalize_outputs",
    "running_mean", "running_var", "num_batches_tracked",
    "_mean", "_std", "_min", "_max", "buffer_",
)

MOVED, FROZEN = 1e-2, 1e-4


def is_buffer(key):
    return any(m in key for m in BUFFER_MARKERS)


def sweep(tag, repo, base_path, depth):
    ckpt_path = hf_hub_download(repo, WEIGHT_FILE)
    agg = defaultdict(lambda: {"d2": 0.0, "b2": 0.0, "n": 0, "dtypes": set()})
    skipped, mismatched = 0, []

    with safe_open(base_path, framework="pt", device="cpu") as fb, \
         safe_open(ckpt_path, framework="pt", device="cpu") as fc:
        bk, ck = set(fb.keys()), set(fc.keys())
        missing, extra = sorted(bk - ck), sorted(ck - bk)
        for k in sorted(bk & ck):
            if is_buffer(k):
                skipped += 1
                continue
            b, c = fb.get_tensor(k), fc.get_tensor(k)
            if b.shape != c.shape:
                mismatched.append((k, tuple(b.shape), tuple(c.shape)))
                continue
            g = agg[".".join(k.split(".")[:depth])]
            g["dtypes"].add((str(b.dtype), str(c.dtype)))
            bf, cf = b.float(), c.float()
            d = cf - bf
            g["d2"] += float(torch.sum(d * d))
            g["b2"] += float(torch.sum(bf * bf))
            g["n"] += d.numel()

    print(f"\n=== {tag}  ({repo}) ===")
    print(f"  buffers excluded: {skipped}")
    if missing:
        print(f"  in base, absent from ckpt: {len(missing)}  e.g. {missing[:3]}")
    if extra:
        print(f"  in ckpt, absent from base: {len(extra)}  e.g. {extra[:3]}")
    if mismatched:
        print(f"  SHAPE MISMATCH: {mismatched[:3]}")

    print(f"  {'group':<50}{'params':>12}{'||d||':>13}{'rel':>12}  verdict  dtypes")
    moved = 0
    for p in sorted(agg, key=lambda p: -agg[p]["d2"]):
        g = agg[p]
        dn = g["d2"] ** 0.5
        rel = dn / (g["b2"] ** 0.5) if g["b2"] > 0 else float("nan")
        if rel > MOVED:
            verdict = "MOVED  "
            moved += g["n"]
        elif rel < FROZEN:
            verdict = "frozen "
        else:
            verdict = "AMBIG  "
        dt = "same" if all(a == b for a, b in g["dtypes"]) else sorted(g["dtypes"])
        print(f"  {p:<50}{g['n']:>12,}{dn:>13.4e}{rel:>12.3e}  {verdict} {dt}")

    total = sum(g["n"] for g in agg.values())
    print(f"  moved: {moved:,} of {total:,} ({100 * moved / total:.1f}%)")
    return {p: agg[p]["d2"] ** 0.5 for p in agg}, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    base_path = hf_hub_download(BASE_REPO, WEIGHT_FILE)
    tags = args.only or list(CKPTS)

    groups, counts = {}, {}
    for tag in tags:
        norms, moved = sweep(tag, CKPTS[tag], base_path, args.depth)
        groups[tag] = {p for p, v in norms.items() if v > 0}
        counts[tag] = moved

    print(f"\n{'=' * 70}\nCROSS-CHECKPOINT CONSISTENCY\n{'=' * 70}")
    ref = tags[0]
    for tag in tags[1:]:
        diff = groups[tag] ^ groups[ref]
        print(f"  {tag:<20} vs {ref:<20} "
              f"{'same groups' if not diff else f'DIFFERS: {sorted(diff)[:5]}'}")
    print("\n  moved-parameter counts:")
    for tag in tags:
        print(f"    {tag:<20} {counts[tag]:>12,}")


if __name__ == "__main__":
    main()