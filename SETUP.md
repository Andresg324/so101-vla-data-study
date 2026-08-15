# Setup & Reproduction
 
## Environment
- **Data collection + inference:** macOS (MacBook Air).
- **Training:** cloud GPU (Google Colab, A100).
- Python 3.12 via Miniforge/conda.
```bash
conda create -n lerobot python=3.12
conda activate lerobot
pip install "lerobot[feetech,smolvla,dataset]"
pip install numpy scipy pandas scikit-learn statsmodels matplotlib openpyxl
```
 
## Hardware bring-up (SO-ARM101)
Power: **leader = 5V, follower = 12V**
 
1. **Find serial ports:** `lerobot-find-port` (run once per arm; note each port).
2. **Assign motor IDs:** `lerobot-setup-motors` (connect one motor at a time).
3. **Calibrate each arm** (clamp them down first):
> Replace `<follower_port>` / `<leader_port>` with your own from `lerobot-find-port`
```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=<follower_port> --robot.id=my_follower_arm
   lerobot-calibrate --teleop.type=so101_leader   --teleop.port=<leader_port>  --teleop.id=my_leader_arm
```
4. **Verify teleoperation + cameras:** `bash scripts/check_cameras_live.sh`
## Cameras
- Two USB cameras (overhead and wrist) via a hub. Current mapping: **overhead = index 1,
  wrist = index 0**.
- **macOS flag:** OpenCV camera indices shuffle between sessions, and **index 2 is the
  MacBook's own built-in camera** which must not be passed to LeRobot. Run
  `python tools/check_cameras.py` at the start of every session and confirm that
  `tools/preview_overhead.jpg` actually shows the board from above. Update the `_IDX` values in
  `scripts/record_dataset.sh`, `scripts/run_inference.sh`, `scripts/check_cameras_live.sh` and
  `tools/check_cameras.py` if they have moved. All four must agree.
- `check_cameras.py` also reports the measured frame rate. Both cameras must sustain 30 fps at
  640 x 480; a camera that silently drops to 5 or 15 fps corrupts the recorded timing and
  invalidates every pace and rate figure derived from it.
- `bash scripts/preview_cameras.sh` is the fastest check that both cameras are alive and framed.
  It addresses them **by name** rather than by index, so it survives the index shuffle; use
  `check_cameras_live.sh` when you also need teleoperation running.
## Data collection
```bash
hf auth login   # write token, once
bash scripts/record_dataset.sh clean 50   # <condition> <num_episodes>
```
`lerobot-record` appends a timestamp to the dataset repo_id (e.g.
`cube-pickup-clean_YYYYMMDD_HHMMSS`). See [RUN_SHEET.md](RUN_SHEET.md) for the per-session
checklist and the datasets as actually recorded.
 
## Training (Colab GPU)
Fine-tune from `lerobot/smolvla_base`. The `--rename_map` maps the dataset's
camera keys (`overhead`, `wrist`) to the policy's expected keys (`camera1`, `camera2`):
> Replace `<user>/cube-pickup-clean_<timestamp>` with your Hugging Face dataset name.
 
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=<user>/cube-pickup-clean_<timestamp> \
  --rename_map='{"observation.images.overhead":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size=32 --steps=10000 --save_freq=2000 --seed=1000 \
  --output_dir=outputs/train/smolvla_clean \
  --policy.device=cuda --wandb.enable=true
```
 
Training does not push to the Hub. The final checkpoint is uploaded afterwards so the name
follows the convention in PROTOCOL.md §7:
 
```bash
hf upload <user>/smolvla-cube-clean \
  outputs/train/smolvla_clean/checkpoints/010000/pretrained_model
```
 
These hyperparameters are fixed by the protocol (PROTOCOL.md §4.7) and are identical for all four
conditions within a replication. The seed is the only setting varied between replications:
`--seed=1000` for the primary run and `--seed=2000` for the replication, with
`--output_dir=outputs/train/smolvla_{condition}_seed{seed}` and the `-seed2000` suffix on the
uploaded model. The checkpoint evaluated is the final one at step 10000. Do not tune anything per
condition, as doing so breaks the comparison the study is built on.
 
`so101_Data_Project.ipynb` is the notebook these runs were executed from; it lists all nine runs
and records the installed package versions.
 
## Inference (autonomous)
```bash
bash scripts/run_inference.sh <policy> <cell>
```
Uses `lerobot-rollout` with `--strategy.type=episodic`,
`--strategy.reset_to_initial_position=true` and `--policy.device=mps`, matching PROTOCOL.md §4.12.
The camera keys must be `camera1` (overhead) and `camera2` (wrist) to match training. Keep the
leader parked in frame so the view matches the training data.
 
Each cell records 16 episodes. The first (index 0) is a warmup and is never scored, because the
first forward pass in a process pays a one-off Metal kernel compilation cost that later passes do
not. The 15 scored episodes are indices 1 to 15. See PROTOCOL.md §4.13.
 
## Reviewing episodes
LeRobot v3 packs many episodes into one mp4 per camera, so an episode is a time range rather than
a file. `tools/show_episode.py` reads the episode boundaries from the dataset metadata and plays
the slice:
 
```bash
python tools/show_episode.py <dataset_name> <episode_index> [camera_key]
 
# overhead view of scored episode 3
python tools/show_episode.py rollout_randomized_in_distribution_20260810_121255 3 observation.images.camera1
```
 
Omit the camera key for the default view, or pass `observation.images.camera2` for the wrist.
The episode index in the file is the scored episode number directly, since index 0 is the
discarded warmup.
 
## Analysis
 
Rollout scores live in `documents/results_raw_two_seeds.xlsx`, the only file edited by hand.
Everything else regenerates from it.
 
```bash
python tools/export_results.py                 # rebuild the derived CSVs; refuses to write on validation failure
python tools/audit_labels.py                   # label screens, recording-window measurement
 
# analyze_results refuses multi-seed input: PROTOCOL.md §4.7 doesn't allow pooling seeds
python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
python analysis/analyze_exploratory.py         # displacement and demonstration-pace probes
python analysis/seed_variance.py               # the same condition compared across seeds
 
# grasp poses, once per policy, then board coordinates onto the endpoint files
python tools/endpoints.py --policy clean --cells in_distribution new_positions \
    reduced_lighting different_object distractors
python tools/calibrate_pose.py --apply
 
python tools/rollout_motion.py                 # latency, velocity, no_departure validation
python tools/drops.py                          # detector calibration, release events, drop locations
python tools/azimuth_analysis.py               # calibration, envelope, aiming error, aim invariance
python tools/motion_stats.py <dataset> [...]   # demonstration pace, frame counts, epochs
 
python analysis/make_figures.py                # writes figures/
```
 
Order matters in two places: `endpoints.py` before `calibrate_pose.py --apply`, and both before
`azimuth_analysis.py`. `analysis/README.md` maps each script to the numbers it produces and
carries the instrument-validation figures.
 
Probing requires extracted activations, which are gitignored for size. Regenerate with
`python probing/extract_activations.py`, then `python probing/probe_position.py`.
`probing/train_probes.py` is kept for provenance but is not reportable on this data; see
`analysis/README.md`.