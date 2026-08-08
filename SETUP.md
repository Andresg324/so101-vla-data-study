# Setup & Reproduction

## Environment
- **Data collection + inference:** macOS (MacBook Air).
- **Training:** cloud GPU (Google Colab, A100).
- Python 3.12 via Miniforge/conda.

```bash
conda create -n lerobot python=3.12
conda activate lerobot
pip install "lerobot[feetech,smolvla,dataset]"
```

## Hardware bring-up (SO-ARM101)
Power: **leader = 5V, follower = 12V**

1. **Find serial ports:** `lerobot-find-port` (run once per arm; note each port).
2. **Assign motor IDs:** `lerobot-setup-motors` (connect one motor at a time).
   - `tools/` contains helper scripts used during bring-up (motor ID scan/swap).
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
  640 x 480; a camera that silently drops to 5 or 15 fps corrupts the recorded timing.

## Data collection
```bash
hf auth login   # write token, once
bash scripts/record_dataset.sh clean 50   # <condition> <num_episodes>
```
Note: `lerobot-record` appends a timestamp to the dataset repo_id (e.g. `cube-pickup-clean_YYYYMMDD_HHMMSS`).

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

These hyperparameters are fixed by the protocol (PROTOCOL.md §4) and are identical for all four
conditions, including the seed. The checkpoint evaluated is the final one at step 10000. Do not
tune them per condition, as doing so breaks the comparison the study is built on.

## Inference (autonomous)
```bash
bash scripts/run_inference.sh
```
Uses `lerobot-rollout` with `--strategy.type=sentry` (records the run) and
`--policy.device=mps`. The camera keys must be `camera1` (overhead) / `camera2` (wrist)
to match training. Keep the leader parked in frame so the view matches the training data.