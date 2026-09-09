#!/usr/bin/env python3
"""
inventory.py

Inventory Hugging Face repos to confirm what datasets and rollouts are recorded. Provides file 
listings with sizes, and the safetensors header (parameter names, shapes, dtypes) read without
downloading the weights.

Run this first whenever a script is about to hardcode a filename or a parameter
key, so the hardcoded value comes from the repo rather than from memory.

Usage:
    python inventory.py lerobot/smolvla_base Andresg324/smolvla-cube-clean
    python inventory.py --keys 40 Andresg324/smolvla-cube-density
"""

import argparse
from collections import Counter

from huggingface_hub import HfApi


def human(nbytes):
    if nbytes is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024


def inventory(repo_id, n_keys, api):
    print(f"\n{'=' * 70}\n{repo_id}\n{'=' * 70}")

    try:
        info = api.repo_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"  could not reach repo: {e}")
        return

    print("  files:")
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        print(f"    {s.rfilename:<52} {human(s.size):>10}")

    try:
        meta = api.get_safetensors_metadata(repo_id)
    except Exception as e:
        print(f"\n  no safetensors header available ({type(e).__name__}: {e})")
        return

    tensors = meta.files_metadata
    all_tensors = {}
    for fname, fmeta in tensors.items():
        for k, t in fmeta.tensors.items():
            all_tensors[k] = t

    print(f"\n  tensors: {len(all_tensors)}")
    print(f"  parameters by dtype: {dict(meta.parameter_count)}")

    depth2 = Counter(".".join(k.split(".")[:2]) for k in all_tensors)
    print("\n  top-level structure (depth 2):")
    for prefix, count in sorted(depth2.items()):
        print(f"    {prefix:<48} {count:>5} tensors")

    print(f"\n  first {n_keys} keys:")
    for k in sorted(all_tensors)[:n_keys]:
        t = all_tensors[k]
        print(f"    {k:<58} {str(tuple(t.shape)):<20} {t.dtype}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repos", nargs="+")
    p.add_argument("--keys", type=int, default=20,
                   help="how many parameter keys to print per repo")
    args = p.parse_args()

    api = HfApi()
    for repo in args.repos:
        inventory(repo, args.keys, api)


if __name__ == "__main__":
    main()