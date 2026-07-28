#!/usr/bin/env python3
"""
PRELIMINARY VERSION
Extract hidden activations by replaying saved eval episodes through the policy.
For each eval episode: load its saved observations, run the policy forward with a hook that
grabs one layer's activation per step, and save (activation, condition, episode, success, t_from_end).
Output feeds train_probes.py. 
Still need to verify marks API calls to confirm on your machine.
"""
import numpy as np
import torch
import pandas as pd

# 1) load your labels (which episode succeeded) from your results CSV
labels = pd.read_csv("results.csv")   # condition, eval_cell, seed, episode, success

# 2) load the trained policy  [VERIFY exact import and class name]
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy # <- to update this
# policy = SmolVLAPolicy.from_pretrained("Andresg324/smolvla-cube-<condition>").eval() # <- to update this

# 3) register a forward hook to capture one layer's output each forward pass
captured = {}
def hook(module, inp, out):
    captured["act"] = out.detach().float().mean(dim=1).cpu().numpy()  # [VERIFY shape; maybe pool over tokens
# target_layer = policy.model.<...>          # [VERIFY: print(policy) to find a good layer, e.g. last VLM block]
# target_layer.register_forward_hook(hook)

# 4) load the saved eval rollout dataset (the observations lerobot recorded)  [VERIFY]
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Andresg324/rollout_cube_<condition>")

# ---- OUTPUT CONTRACT: train_probes.py reads these exact keys ----
# X          : (n_steps, hidden_dim) float  - activations, one row per timestep
# condition  : (n_steps,) str   - clean / randomized / recovery / color
# eval_cell  : (n_steps,) str   - in_distribution / new_positions / ...
# episode    : (n_steps,) int   - episode id (used to GROUP splits; no leakage)
# success    : (n_steps,) int   - 1/0 episode outcome, repeated for every step in the episode
# t_from_end : (n_steps,) int   - steps remaining until the episode ends (0 = last step)
# ------ These are required for the lead-time curve ---------------------
#
# X, cond, eval_cell, success, tfe = [], [], [], [], []
# for each episode in ds:
#     frames = the episode's ordered observations
#     for t, obs in enumerate(frames):
#        with torch.no_grad():
#            _ = policy.select_action(obs)      # [VERIFY call]; this triggers the hook
#        X.append(captured["act"][0]); ... append condition/episode/success/(len-1-t)
#  np.savez("phase_b/activations_real.npz", X=..., condition=..., episode=..., success=..., t_from_end=...)