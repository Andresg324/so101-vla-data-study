#!/usr/bin/env python3
"""
Loss increase along random directions at matched step size. A solution in a narrow
basin degrades faster than one in a flat basin, which is the property deployment
exposes and the per-frame loss does not: chunks execute open loop for 50 steps and
flow-matching inference is stochastic.

Only the action expert is perturbed. §8.31's update-norm check established that
vlm.model and vlm.lm_head are bit-identical to the base in all ten checkpoints,
perturbing them would move weights no training touched.

Step size is matched across policies as a fraction of each policy's own trainable
weight norm, since absolute step size is not comparable between checkpoints.

Sharpness is a ratio within one policy on one fixed episode subset, so the subset
choice does not enter the comparison the way it would for an absolute loss value.

RHO is chosen on clean_s1000 alone, before other policies are scored: the smallest of
0.05 and 0.1 at which its mean ratio exceeds 1.1. 0.02 gave 1.006, too flat to
discriminate. August policies are scored on their own training episodes and sweep
policies on the 20 held-out demonstrations, so the two groups are reported separately
and not compared to each other.

RUN: python tools/sharpness.py --device cuda
"""

import json, argparse, os
import numpy as np
import torch

from deterministic_loss import score, POLICIES, ALL_SWEEP, OUTDIR
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

HF_USER = "Andresg324"
EPISODES = list(range(10))   # fixed subset; sharpness is a ratio, see docstring
RHO = 0.02                   # direction norm as a fraction of trainable weight norm
N_DIRECTIONS = 5
DIR_SEED = 1000

DIAGONAL = {
    "clean_s1000": "clean", "clean_s2000": "clean",
    "randomized_s1000": "randomized", "randomized_s2000": "randomized",
    "recovery_s1000": "recovery", "recovery_s2000": "recovery",
    "color_s1000": "color", "color_s2000": "color",
    "color_slowpace": "slowpace", "density": "density",
    **{p: "pool" for p in ALL_SWEEP},
}

SWEEP_EPISODES = json.load(open("analysis/subsets.json"))["held_out_episodes"]

def trainable(policy):
    """Parameters the fine-tune moved: outside the frozen VLM, and excluding
    lm_expert.norm, which §8.31 found bit-identical to the base across checkpoints."""
    return [(n, p) for n, p in policy.named_parameters()
            if "vlm_with_expert.vlm." not in n and ".lm_expert.norm." not in n]

def sharpness(key, device, rho):
    prepo = f"{HF_USER}/{POLICIES[key]}"
    eps = SWEEP_EPISODES if key in ALL_SWEEP else EPISODES
    ds = DIAGONAL[key]

    policy = SmolVLAPolicy.from_pretrained(prepo)
    policy.to(device).eval()

    params = trainable(policy)
    base = {n: p.detach().clone() for n, p in params}
    wnorm = torch.sqrt(sum((v ** 2).sum() for v in base.values()))

    ref = score(key, ds, device, episodes=eps, policy=policy, verbose=False)["masked"]

    g = torch.Generator(device="cpu").manual_seed(DIR_SEED)
    ratios = []
    for d in range(N_DIRECTIONS):
        noise = {n: torch.randn(v.shape, generator=g).to(v.device) for n, v in base.items()}
        nnorm = torch.sqrt(sum((v ** 2).sum() for v in noise.values()))
        scale = rho * wnorm / nnorm
        with torch.no_grad():
            for n, p in params:
                p.copy_(base[n] + noise[n] * scale)
        r = score(key, ds, device, episodes=eps, policy=policy, verbose=False)["masked"]
        ratios.append(r / ref)
        print(f"  {key} dir {d}: {r:.5f} / {ref:.5f} = {r/ref:.3f}")

    with torch.no_grad():
        for n, p in params:
            p.copy_(base[n])
    return {"policy": key, "dataset": ds, "n_episodes": len(eps),
            "ref_loss": ref, "mean_ratio": float(np.mean(ratios)),
            "sd_ratio": float(np.std(ratios)), "ratios": [float(x) for x in ratios]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policy", help="one policy key; default all")
    ap.add_argument("--rho", type=float, default=RHO)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    keys = [args.policy] if args.policy else list(DIAGONAL)
    rows = []
    for k in keys:
        rows.append(sharpness(k, args.device, args.rho))
        with open(f"{OUTDIR}/sharpness_rho{args.rho:g}.json", "w") as f:
            json.dump(rows, f, indent=1)
    import pandas as pd
    t = pd.DataFrame(rows).drop(columns=["ratios"])
    print("\n" + t.round(4).to_string(index=False))
    print(f"\nwrote {OUTDIR}/sharpness_rho{args.rho:g}.json")


if __name__ == "__main__":
    main()