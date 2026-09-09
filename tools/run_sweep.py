#!/usr/bin/env python3
"""Density sweep, sequential. PROTOCOL.md §8.32."""

import json
import os
import subprocess
import sys

from huggingface_hub import HfApi

HF_USER = "Andresg324"
POOL = f"{HF_USER}/cube-pickup-densitypool_20260908_125700"
RENAME = ('{"observation.images.overhead": "observation.images.camera1", '
          '"observation.images.wrist": "observation.images.camera2"}')

with open("subsets.json") as f:
    SUBSETS = json.load(f)["subsets"]

# (per-position budget, steps, warmup, save_freq)
CELLS = [(5, 10000, 333, 2000),
         (10, 20000, 667, 4000),
         (25, 50000, 1667, 10000),
         (50, 100000, 3333, 20000)]
SEEDS = [1000, 2000]

api = HfApi()

for budget, steps, warmup, save_freq in CELLS:
    episodes = json.dumps(SUBSETS[str(budget)])
    for seed in SEEDS:
        name = f"smolvla_density{budget}_seed{seed}"
        outdir = f"outputs/train/{name}"
        suffix = "" if seed == 1000 else f"-seed{seed}"
        modelrepo = f"{HF_USER}/smolvla-cube-density{budget}{suffix}"

        if os.path.isdir(f"{outdir}/checkpoints/{steps:06d}/pretrained_model"):
            print(f"skip {name}, already complete")
            continue

        cmd = (
            "lerobot-train"
            " --policy.path=lerobot/smolvla_base"
            " --policy.push_to_hub=false"
            f" --dataset.repo_id={POOL}"
            f" --dataset.episodes='{episodes}'"
            f" --rename_map='{RENAME}'"
            " --batch_size=32"
            f" --steps={steps}"
            f" --save_freq={save_freq}"
            f" --policy.scheduler_warmup_steps={warmup}"
            f" --policy.scheduler_decay_steps={steps}"
            f" --seed={seed}"
            f" --output_dir={outdir}"
            f" --job_name={name}"
            " --policy.device=cuda"
            " --wandb.enable=true"
            " --wandb.disable_artifact=true"
        )
        print(f"\n=== {name} ===\n{cmd}\n", flush=True)
        subprocess.run(cmd, shell=True, check=True)

        ckpt = f"{outdir}/checkpoints/{steps:06d}/pretrained_model"
        assert os.path.isdir(ckpt), f"no step-{steps} checkpoint at {ckpt}"

        # Verify the schedule landed before uploading anything.
        c = json.load(open(f"{ckpt}/train_config.json"))
        assert c["policy"]["scheduler_warmup_steps"] == warmup, c["policy"]["scheduler_warmup_steps"]
        assert c["policy"]["scheduler_decay_steps"] == steps, c["policy"]["scheduler_decay_steps"]
        assert len(c["dataset"]["episodes"]) == budget * 10, len(c["dataset"]["episodes"])

        api.create_repo(modelrepo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=ckpt, repo_id=modelrepo, repo_type="model")
        print(f"uploaded {modelrepo}", flush=True)

print("\nall cells complete")