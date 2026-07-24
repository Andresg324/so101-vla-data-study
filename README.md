# SO-101 × SmolVLA: How Demonstration-Collection Strategy Shapes Generalization

A controlled empirical study on a self-built low-cost robot arm (Seeed SO-ARM101,
LeRobot platform) using the SmolVLA vision-language-action model.

## The question
A VLA learns to map camera images and language instruction directly to robot motion,
end-to-end from demonstrations (no hand-coded perception or inverse kinematics needed).
This project asks: how does the way you collect demonstrations affect how well the 
learned policy generalizes to conditions it never saw?

## DEMO
![First working loop](media/overhead_demo.gif)

The arm initially misses the cube, as it was misplaced from it's original starting point 
(denoted by an x), however, once the cube is moved to the correct place, the arm autonomously
picks up the cube and places it in the cup, running a SmolVLA policy fine-tuned on 50 
teleoperated demonstrations. No teleoperation required as the policy drives the follower
from the two camera feeds.

Full clip: [Recording](https://drive.google.com/file/d/1-PAdS0wvpLscmjM5CJIJJoptIiziuqE7/view?usp=sharing)

## Design (see [PROTOCOL.md](PROTOCOL.md) for the full pre-registered protocol)
Fix the model (SmolVLA), the task (pick-up a cube and place it in a cup), and the demo budget (50 episodes).
Vary only the data-collection strategy across four conditions: clean, position-randomized,
recovery (failed grasps and teleoperated recovery), and visual diversity.
Evaluate all four on held-out conditions none of them saw, measuring task success rate.

## Hardware
- Seeed SO-ARM101 Pro (leader 5V / follower 12V), Feetech STS3215 servos
- Two cameras: Seeed 1080p (wrist/side) and Logitech C270 720p (workspace view)
- MacBook Air (data collection and inference); cloud GPU (Colab) for training

## Pipeline
1. `scripts/record_dataset.sh` — teleoperate and record synchronized camera and joint data
2. Train SmolVLA on a cloud GPU (fine-tune from `lerobot/smolvla_base`), log to Weights & Biases
3. `scripts/run_inference.sh` — trained policy drives the arm autonomously

## Tools
`tools/check_cameras.py` — a headless-safe camera probe that saves a frame from
each camera (overhead and wrist) so you can verify the framing before recording. 
Used during camera setup; written because lerobot's OpenCV build is headless and can't
open a live preview window.

## Status
- [x] Hardware assembled and calibrated
- [x] Teleoperation verified (leader to follower mirroring)
- [x] First clean dataset recorded
- [x] SmolVLA trained; first autonomous pick
- [ ] Four-condition study + analysis

## Results (from first working loop)
Closed the full pipeline end-to-end: teleoperated data collection to SmolVLA
fine-tuning (10k steps, single A100) → autonomous inference on the real arm.
The policy reliably picks and places when the cube is at the trained position
(4 consecutive successes).

When the cube was moved off the trained position, the policy reached and missed
repeatedly — a direct, observed instance of the generalization gap that the
four-condition study is designed to measure. The clean-data policy nails the
in-distribution pose and degrades off it.


  <img src="media/loss_curve.png" width="500" alt="Training loss">


Training loss (`train/losses_after_rm_padding`) fell from ~0.19 to ~0.045 over
10k steps, plateauing around step 6k — the policy converged well within the run.

**Artifacts:** [dataset](https://huggingface.co/datasets/Andresg324/cube-pickup-clean_20260723_151726) · [trained model](https://huggingface.co/Andresg324/smolvla-cube-clean)

## Reproducing
Requires the LeRobot environment (Python 3.12). See [SETUP.md](SETUP.md).