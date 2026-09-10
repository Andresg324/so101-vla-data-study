#!/usr/bin/env python3
"""
tools/run_sweep.py

Density sweep training runs (PROTOCOL.md §8.32). Trains four per-position budgets
at two seeds each, sequentially, both seeds of a budget before moving to the next.

Subsets are episode index lists from analysis/subsets.json, passed to
--dataset.episodes, so no physical dataset copies are made. Steps scale with
dataset size to hold epochs constant, and decay steps equal the step count in
every cell, because CosineDecayWithWarmupSchedulerConfig rescales warmup and
decay whenever training steps fall below the configured decay (§8.29). Warmup is
the 1/30 ratio the August runs resolved to.

The resolved config is checked after each run and before upload: a silently
dropped scheduler override would train every cell on the wrong schedule shape,
so the chain stops rather than propagating it.

Run from the repository root:
    python tools/run_sweep.py 2>&1 | tee sweep.log
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
        final = f"{outdir}/checkpoints/{steps:06d}/pretrained_model"
        suffix = "" if seed == 1000 else f"-seed{seed}"
        modelrepo = f"{HF_USER}/smolvla-cube-density{budget}{suffix}"

        try:
            api.repo_info(modelrepo, repo_type="model")
            print(f"skip {name}, already on Hub", flush=True)
            continue
        except Exception:
            pass
        if os.path.isdir(outdir):
            # Partial run from an interrupted attempt. lerobot-train refuses to
            # write into an existing output dir unless --resume is set, and a
            # partial run is not worth resuming at these step counts.
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

        assert os.path.isdir(final), f"no step-{steps} checkpoint at {final}"

        c = json.load(open(f"{final}/train_config.json"))
        assert c["policy"]["scheduler_warmup_steps"] == warmup, \
            f"warmup {c['policy']['scheduler_warmup_steps']} != {warmup}"
        assert c["policy"]["scheduler_decay_steps"] == steps, \
            f"decay {c['policy']['scheduler_decay_steps']} != {steps}"
        assert c["steps"] == steps, f"steps {c['steps']} != {steps}"
        assert len(c["dataset"]["episodes"]) == budget * 10, \
            f"{len(c['dataset']['episodes'])} episodes != {budget * 10}"
        assert c["seed"] == seed, f"seed {c['seed']} != {seed}"

        api.create_repo(modelrepo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=final, repo_id=modelrepo, repo_type="model")
        print(f"uploaded {modelrepo}", flush=True)

print("\nall cells complete", flush=True)