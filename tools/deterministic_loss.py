#!/usr/bin/env python3
"""
tools/deterministic_loss.py

PROTOCOL.md §8.31 step 2: scores every fine-tuned checkpoint on every training
dataset, to separate a property of each condition's target distribution from a
property of its solution.

Determinism is exact rather than seeded. SmolVLAPolicy.forward accepts the
flow-matching noise and timestep as arguments, so both are generated once per
frame from a fixed seed and reused across every checkpoint-by-dataset cell. Two
identical calls therefore return bit-identical values, and the noise contribution
cancels in any cell-to-cell comparison.

Two loss quantities are recorded per call:

  reported   loss_dict["losses_after_rm_padding"], the key W&B logs and Table 7
             reports. Padded timesteps are zeroed but remain in the denominator,
             so this is deflated in proportion to how much padding a dataset
             produces: with chunk_size 50, an episode of length L pads 24.5/L of
             its chunk-step pairs, about 4% at these episode lengths. Kept for
             comparability with Table 7, not because it is the right number.

  masked     the reduction="none" per-sample loss, which divides by the count of
             unpadded timesteps. This is the correct masked mean.

Uncertainty is a bootstrap over episodes, not over noise draws. Frames within an
episode are correlated, so the episode is the sampling unit; and with tens of
thousands of frames each carrying its own timestep draw, the Monte Carlo error in
the noise expectation is already negligible next to the error from evaluating a
subset of episodes.

usage:
    # how many episodes are enough
    python tools/deterministic_loss.py --calibrate --policy clean_s1000

    # one cell
    python tools/deterministic_loss.py --policy clean_s1000 --dataset clean

    # the full matrix
    python tools/deterministic_loss.py --all
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tqdm import tqdm

HF_USER = "Andresg324"

POLICIES = {
    "clean_s1000":      "smolvla-cube-clean",
    "clean_s2000":      "smolvla-cube-clean-seed2000",
    "randomized_s1000": "smolvla-cube-randomized",
    "randomized_s2000": "smolvla-cube-randomized-seed2000",
    "recovery_s1000":   "smolvla-cube-recovery",
    "recovery_s2000":   "smolvla-cube-recovery-seed2000",
    "color_s1000":      "smolvla-cube-color",
    "color_s2000":      "smolvla-cube-color-seed2000",
    "color_slowpace":   "smolvla-cube-color-slowpace",
    "density":          "smolvla-cube-density",
}

DATASETS = {
    "clean":      "cube-pickup-clean_20260809_105745",
    "randomized": "cube-pickup-randomized_20260809_115825",
    "recovery":   "cube-pickup-recovery_20260809_141725",
    "color":      "cube-pickup-color_20260809_183224",
    "slowpace":   "cube-pickup-color_20260809_130649",
    "density":    "cube-pickup-density_20260822_194111",
}

NOISE_SEED   = 1000        # fixed across every cell; see the module docstring
EPISODE_SEED = 1000        # which episodes form the fixed evaluation subset
N_EPISODES = 40            # set from the --calibrate run before any cell is scored
BATCH_SIZE = 16
N_BOOTSTRAP = 2000
OUTDIR = "analysis/out_loss"


def pick_episodes(repo_id, n, seed=EPISODE_SEED):
    """A fixed subset of complete episodes. Complete rather than loose frames,
    because frames within an episode are correlated and the episode is the unit
    the bootstrap resamples."""
    meta = LeRobotDatasetMetadata(repo_id)
    rng = np.random.default_rng(seed)
    eps = rng.permutation(meta.total_episodes)[:n]
    return sorted(int(e) for e in eps), meta


def make_noise(n_frames, chunk, dim, device, seed=NOISE_SEED):
    """One (noise, time) pair per frame, generated on CPU from a fixed seed so
    frame i receives identical values in every cell regardless of batching or
    device."""
    torch.manual_seed(seed)
    noise = torch.randn(n_frames, chunk, dim)
    beta = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time = beta.sample((n_frames,)) * 0.999 + 0.001
    return noise.to(device), time.to(device)


def score(policy_key, dataset_key, device, n_episodes=N_EPISODES,
          norm_from=None, verbose=True):
    """One cell of the matrix. norm_from names the dataset whose normalization
    statistics to use; None means the checkpoint's own, which is the primary
    condition in §8.31."""
    prepo = f"{HF_USER}/{POLICIES[policy_key]}"
    drepo = f"{HF_USER}/{DATASETS[dataset_key]}"

    policy = SmolVLAPolicy.from_pretrained(prepo)
    policy.to(device).eval()

    stats = None
    if norm_from is not None:
        stats = LeRobotDatasetMetadata(f"{HF_USER}/{DATASETS[norm_from]}").stats
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=prepo,
        dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    episodes, meta = pick_episodes(drepo, n_episodes)
    delta = {"action": [i / meta.fps for i in policy.config.action_delta_indices]}
    ds = LeRobotDataset(drepo, delta_timestamps=delta, episodes=episodes)

    chunk = policy.config.chunk_size
    dim = policy.config.max_action_dim
    noise_all, time_all = make_noise(len(ds), chunk, dim, device)

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    rows = []
    offset = 0
    for batch in tqdm(dl, desc=f"{policy_key}/{dataset_key}"):
        b = batch["action"].shape[0]
        noise = noise_all[offset:offset + b]
        time = time_all[offset:offset + b]
        offset += b

        batch = pre(batch)
        with torch.no_grad():
            per_sample, loss_dict = policy.forward(
                batch, noise=noise, time=time, reduction="none")

        rows.append({
            "episode": batch["episode_index"].cpu().numpy(),
            "masked": per_sample.cpu().numpy(),
            "reported": np.full(b, loss_dict["losses_after_rm_padding"]),
        })

    ep = np.concatenate([r["episode"] for r in rows])
    masked = np.concatenate([r["masked"] for r in rows])
    reported = np.concatenate([r["reported"] for r in rows])

    # Per-episode means, which are the bootstrap unit.
    uniq = np.unique(ep)
    per_ep_masked = np.array([masked[ep == e].mean() for e in uniq])
    per_ep_reported = np.array([reported[ep == e].mean() for e in uniq])

    rng = np.random.default_rng(0)
    boot = np.array([
        per_ep_masked[rng.integers(0, len(uniq), len(uniq))].mean()
        for _ in range(N_BOOTSTRAP)
    ])

    out = {
        "policy": policy_key,
        "dataset": dataset_key,
        "norm_from": norm_from or policy_key,
        "n_episodes": len(uniq),
        "n_frames": int(len(ds)),
        "masked": float(per_ep_masked.mean()),
        "masked_se": float(boot.std()),
        "masked_ci": [float(np.percentile(boot, 2.5)),
                      float(np.percentile(boot, 97.5))],
        "reported": float(per_ep_reported.mean()),
        "per_episode": {int(e): float(v) for e, v in zip(uniq, per_ep_masked)},
    }
    if verbose:
        print(f"{policy_key:>18} on {dataset_key:<11} "
              f"masked {out['masked']:.5f} ± {out['masked_se']:.5f}   "
              f"reported {out['reported']:.5f}   "
              f"({out['n_episodes']} eps, {out['n_frames']} frames)")
    del policy
    return out


def calibrate(policy_key, device):
    """How many episodes are enough. Runs the diagonal cell at increasing episode
    counts; the answer is where the value stops moving relative to the 10%
    criterion registered in §8.31."""
    dataset_key = policy_key.split("_")[0]
    if dataset_key not in DATASETS:
        dataset_key = "clean"
    print(f"calibrating on {policy_key} / {dataset_key}\n")
    prev = None
    for n in (5, 10, 20, 40, 80):
        r = score(policy_key, dataset_key, device, n_episodes=n, verbose=False)
        delta = "" if prev is None else f"   change {100*(r['masked']-prev)/prev:+.2f}%"
        print(f"  {n:>3} episodes: masked {r['masked']:.5f} "
              f"± {r['masked_se']:.5f}{delta}")
        prev = r["masked"]
    print("\nPick the smallest count where the change is well inside 10%, "
          "the §8.31 criterion, and the standard error is a small fraction of it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--dataset")
    ap.add_argument("--norm-from", help="score with this dataset's normalization "
                                        "statistics instead of the checkpoint's own")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    if args.calibrate:
        calibrate(args.policy or "clean_s1000", args.device)
        return

    if args.all:
        results = []
        for p, d in itertools.product(POLICIES, DATASETS):
            results.append(score(p, d, args.device, args.episodes))
            with open(f"{OUTDIR}/loss_matrix.json", "w") as f:
                json.dump(results, f, indent=1)
            
        print(f"\nwrote {OUTDIR}/loss_matrix.json ({len(results)} cells)")
        return

    if not (args.policy and args.dataset):
        ap.error("--policy and --dataset, or --all, or --calibrate")
    r = score(args.policy, args.dataset, args.device, args.episodes, args.norm_from)
    print(json.dumps({k: v for k, v in r.items() if k != "per_episode"}, indent=1))


if __name__ == "__main__":
    main()