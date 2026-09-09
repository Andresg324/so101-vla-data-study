#!/usr/bin/env python3
"""
tools/run_controls.py

Fixed-step control runs for the density sweep (PROTOCOL.md §8.32).

The sweep holds epochs constant, so larger datasets receive proportionally more
gradient steps and data quantity is confounded with optimization amount. These two
runs train the 250 and 500 demonstration cells at 10,000 steps instead, the August
configuration, so the confound can be measured. This is done for just one seed each.

Everything else matches the corresponding sweep cell: same episode list, same
rename map, same batch size, same §4.7 settings. Only steps, warmup and decay
differ, and they revert to what every August run used.

Run from the repository root:
    python tools/run_controls.py 2>&1 | tee controls.log
"""

import json
import os
import shutil
import subprocess

from huggingface_hub import HfApi

HF_USER = "Andresg324"
POOL = f"{HF_USER}/cube-pickup-densitypool_20260908_125700"
RENAME = ('{"observation.images.overhead": "observation.images.camera1", '
          '"observation.images.wrist": "observation.images.camera2"}')

with open("analysis/subsets.json") as f:
    SUBSETS = json.load(f)["subsets"]

# The August configuration: 10,000 steps, warmup 333, decay 10,000, save every 2,000.
STEPS, WARMUP, SAVE_FREQ = 10000, 333, 2000
BUDGETS = [25, 50]
SEED = 1000

api = HfApi()

for budget in BUDGETS:
    episodes = json.dumps(SUBSETS[str(budget)])
    name = f"smolvla_density{budget}_seed{SEED}_fixedstep"
    outdir = f"outputs/train/{name}"
    final = f"{outdir}/checkpoints/{STEPS:06d}/pretrained_model"
    modelrepo = f"{HF_USER}/smolvla-cube-density{budget}-fixedstep"

    if os.path.isdir(final):
        print(f"skip {name}, already complete", flush=True)
        continue
    if os.path.isdir(outdir):
        print(f"clearing partial {outdir}", flush=True)
        shutil.rmtree(outdir)

    cmd = (
        "lerobot-train"
        " --policy.path=lerobot/smolvla_base"
        " --policy.push_to_hub=false"
        f" --dataset.repo_id={POOL}"
        f" --dataset.episodes='{episodes}'"
        f" --rename_map='{RENAME}'"
        " --batch_size=32"
        f" --steps={STEPS}"
        f" --save_freq={SAVE_FREQ}"
        f" --policy.scheduler_warmup_steps={WARMUP}"
        f" --policy.scheduler_decay_steps={STEPS}"
        f" --seed={SEED}"
        f" --output_dir={outdir}"
        f" --job_name={name}"
        " --policy.device=cuda"
        " --wandb.enable=true"
        " --wandb.disable_artifact=true"
    )
    print(f"\n=== {name} ===\n{cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)

    assert os.path.isdir(final), f"no step-{STEPS} checkpoint at {final}"

    c = json.load(open(f"{final}/train_config.json"))
    assert c["policy"]["scheduler_warmup_steps"] == WARMUP
    assert c["policy"]["scheduler_decay_steps"] == STEPS
    assert c["steps"] == STEPS
    assert len(c["dataset"]["episodes"]) == budget * 10
    assert c["seed"] == SEED

    api.create_repo(modelrepo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=final, repo_id=modelrepo, repo_type="model")
    print(f"uploaded {modelrepo}", flush=True)

print("\nboth controls complete", flush=True)