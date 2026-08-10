#!/usr/bin/env python3
"""
phase_b/extract_activations.py

Replay saved evaluation rollouts through a trained SmolVLA policy, capture one
layer's hidden activations, and write the .npz that train_probes.py consumes.

Three modes:
  1. Find a layer to hook:
        python phase_b/extract_activations.py --policy clean --list-layers
  2. Extract for one policy across its eval cells:
        python phase_b/extract_activations.py \
            --policy clean --layer <name from step 1> \
            --results results.csv --device cuda
  3. Merge the four per-condition files into the one train_probes.py reads:
        python phase_b/extract_activations.py --merge phase_b/out/activations_*.npz

Mode 2 to be run in Colab
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from huggingface_hub import HfApi

HF_USER = "Andresg324"
CELLS = ["in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors"]

TASK = "Pick up the cube and place it in the cup"  # This needs to be verbatim to the training

# ------------------------------------------------
# Imports that LeRobot moved between releases
# ------------------------------------------------

def _import_first(paths, attr):
    errors = []
    for path in paths:
        try:
            module = __import__(path, fromlist=[attr])
            return getattr(module, attr)
        except Exception as exc:
            errors.append(f" {path}: {exc}")
    raise ImportError(f"could not import {attr}:\n" + "\n".join(errors))

def load_policy(condition, device):
    cls = _import_first(
        ["lerobot.policies.smolvla.modeling_smolvla",
         "lerobot.common.policies.smolvla.modeling_smolvla"],
         "SmolVLAPolicy",
    )

    policy = cls.from_pretrained(f"{HF_USER}/smolvla-cube-{condition}")
    policy.to(device)
    policy.eval()
    return policy

def load_dataset(repo_id):
    cls = _import_first(
        ["lerobot.datasets.lerobot_dataset",
         "lerobot.common.datasets.lerobot_dataset"],
         "LeRobotDataset",
    )
    return cls(repo_id)

# ------------------------------------
# Layer discovery and hooking
# ------------------------------------

def list_layers(policy):
    # Print every submodule name, pipe through grep to narrow it down
    for name, module in policy.named_modules():
        if name:
            print(f"{name:75s} {type(module).__name__}")


def get_module(policy, name):
    # Looks up module by the full dotted name
    modules = dict(policy.named_modules())
    if name not in modules:
        raise KeyError(f"layer '{name}' not found. Run --list-layers to see options.")
    return modules[name]

# One dict shared with the hook. The hook cannot return a value, so it stashes
# the activation here and the main loop reads it back out.
captured = {}

def hook_fn(module, inputs, output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        # (batch, tokens, hidden), these are averaged over tokesn to get one vector
        tensor = tensor.mean(dim=1)
    elif tensor.ndim > 3:
        tensor = tensor.flatten(2).mean(dim=2)
    captured["act"] = tensor[0].cpu().numpy() # batch size is always 1 (e.g., tensor[0])


# ----------------------------------------
# Data plumbing
# ----------------------------------------

def build_observation(item, device):
    # Turns one dataset row into the batch dict select_action() expects

    obs = {}
    for key, value in item.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            obs[key] = value.unsqueeze(0).to(device) # adds the batch dimension
    obs["task"] = [item.get("task", TASK)]
    return obs


def episode_index_columns(ds):
    # Finds an episode ID for every row, without decoding the video frames
    
    try:
        return np.asarray(ds.hf_dataset["episode_index"])
    except Exception:
        return np.asarray([int(ds[i]["episode_index"]) for i in range(len(ds))])

def load_labels(path):
    # {(condition, eval_cell, episode): success} from the scored results
    df = pd.read_csv(path)
    return {(r.condition, r.eval_cell, int(r.episode)): int(r.success) for r in df.itertuples()}

# ----------------------------------------
# Extractions
# ----------------------------------------

def extract_cell(policy, condition, cell, labels, device, repo_override=None, limit_episodes=None, next_uid=0):
    #Replays one (policy, cell) rollout dataset and returns parallel lists.

    #repo = repo_override or f"{HF_USER}/rollout_{condition}_{cell}"
    if repo_override:
        repo = repo_override
    else:
        # LeRobot rollouts have a timestamp attached, so we need to find the file with
        # the right prefix
        prefix = f"{HF_USER}/rollout_{condition}_{cell}_"
        matches = [d.id for d in HfApi().list_datasets(author=HF_USER) if d.id.startswith(prefix)]
        if not matches:
            raise FileNotFoundError(f"No datasets match {prefix}")
        if len(matches) > 1:
            raise ValueError(f"several files match {prefix}*")
        repo = matches[0]

    print(f" {repo}")
    ds = load_dataset(repo)
    ep_col = episode_index_columns(ds)

    out = {"X": [], "episode": [], "success": [], "t_from_end": []}
    episodes = sorted({int(e) for e in ep_col})
    if limit_episodes:
        episodes = episodes[:limit_episodes]

    for ep in episodes:
        rows = np.flatnonzero(ep_col == ep) # Already in frame order
        last = len(rows) - 1

        key = (condition, cell, ep)
        if key not in labels:
            print(f" episode {ep}: no row in results.csv, skipped")
            continue
        y = labels[key]

        policy.reset() # Clears the action queue so episodes don't bleed

        for t, row in enumerate(rows):
            obs = build_observation(ds[int(row)], device)

            captured.pop("act", None) # Allows us to tell whether the hook fired
            with torch.no_grad():
                policy.select_action(obs)

                # SmolVLA predicts a chunk of actions, then pops from a queue for the next n steps
                # without running the model, so the hook fires at the chunk boundaries and this is
                # where we want to record; Forcing every pass would be a significant more compute and 
                # not how the policy behaves

                if "act" in captured:
                    out["X"].append(captured["act"])
                    out["episode"].append(next_uid)
                    out["success"].append(y)
                    out["t_from_end"].append(last - t)

        next_uid += 1

    print(f" {len(out['X'])} activations from {len(episodes)} episodes")
    return out, next_uid

def merge(paths, out_path):
    keys = ["X", "condition", "eval_cell", "episode", "success", "t_from_end"]
    parts = [np.load(p, allow_pickle=True) for p in paths]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    np.savez(out_path, **merged)
    print(f"merged {len(paths)} files to {out_path} ({len(merged['X'])} rows)")

# ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="clean | randomized | recovery | color")
    ap.add_argument("--layer", help="module name from --list-layers")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu")
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--rollout-repo", help="override the dataset name (gate test)")
    ap.add_argument("--limit-episodes", type=int, help="Stop after N per cell")
    ap.add_argument("--outdir", default="phase_b/out")
    ap.add_argument("--list-layers", action="store_true")
    ap.add_argument("--merge", nargs="*")
    args = ap.parse_args()

    if args.merge:
        os.makedirs(args.outdir, exist_ok=True)
        paths = [p for pat in args.merge for p in glob.glob(pat)]
        merge(sorted(paths), os.path.join(args.outdir, "activations_real.npz"))
        return

    policy = load_policy(args.policy, args.device)

    if args.list_layers:
        list_layers(policy)
        return

    get_module(policy, args.layer).register_forward_hook(hook_fn)
    labels = load_labels(args.results)

    X, cond, cells, eps, succ, tfe = [], [], [], [], [], []
    uid = 0
    for cell in args.cells:
        part, uid = extract_cell(policy, args.policy, cell, labels, args.device, args.rollout_repo, args.limit_episodes, uid)
        n = len(part["X"])
        X.extend(part["X"])
        cond.extend([args.policy] * n)
        cells.extend([cell] * n)
        eps.extend(part["episode"])
        succ.extend(part["success"])
        tfe.extend(part["t_from_end"])

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"activations_{args.policy}.npz")
    np.savez(
        out_path,
        X=np.stack(X),
        condition=np.array(cond),
        eval_cell=np.array(cells),
        episode=np.array(eps, dtype=int),
        success=np.array(succ, dtype=int),
        t_from_end=np.array(tfe, dtype=int),
    )

    print(f"\nwrote {out_path} X = {np.stack(X).shape}")

if __name__ == "__main__":
    main()