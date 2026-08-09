# SO-101 × SmolVLA: How Demonstration-Collection Strategy Shapes Generalization

A controlled empirical study on a self-built low-cost robot arm (Seeed SO-ARM101,
LeRobot platform) using the SmolVLA vision-language-action model.

<p align="center">
  <img src="media/overhead_demo.gif" width="600" alt="Autonomous cube pick-and-place">
</p>

## The question
A VLA learns to map camera images and a language instruction directly to robot motion,
end-to-end from demonstrations, with no hand-coded perception or inverse kinematics. This
project asks: how does the way you collect demonstrations affect how well the learned policy
generalizes to conditions it never saw?

## Demo

The clip above is the **overhead camera view** during autonomous inference. The leader is
disconnected and the policy is driving the follower arm on its own. It receives only the two
camera feeds (overhead and wrist) and the instruction *"pick up the cube and place it in the
cup,"* and outputs motion directly, without hand-coded perception, planning, or teleoperation.

At the start, the arm reaches and **misses**: the cube has been placed off the position it was
trained on, so the policy is out of distribution and can't localize the grasp. Once the cube is
returned to its trained spot, the policy **picks it up and drops it in the cup**, and repeats
this reliably (4 consecutive successes).

That contrast, reliable in distribution, failing just outside it, is the generalization gap
this project sets out to measure, across four separate factors that shift the distribution.

The policy shown is SmolVLA trained on 50 teleoperated demonstrations of the *clean* condition,
collected during the pilot (see below).

Full clip: [Recording](https://youtu.be/m15W2h3b3i8)

## Design
See [PROTOCOL.md](PROTOCOL.md) for the full pre-registered protocol.

Fix the model (SmolVLA), the task (pick up a cube and place it in a cup), the demo budget
(50 episodes per condition), and the training hyperparameters. Vary only the data-collection
strategy, one factor at a time:

| Condition | What varies |
|---|---|
| **Clean** | nothing: red cube, one fixed position, every demo a first-try success |
| **Randomized** | cube start position, cycled over 10 marked positions |
| **Recovery** | 20 of 50 demos include a deliberate mid-carry drop and re-grasp |
| **Color-varied** | cube color, cycled over five colors; green held out for evaluation |

<p align="center">
 <img src="media/cube_starting_positions.png" width="520" alt="Cube starting positions">
</p>

All four policies are then evaluated on the same five cells: one in-distribution reference plus
four held-out (new positions, reduced lighting, unseen object color, distractors), at 15
episodes each, 300 rollouts total. Held-out positions are split into interpolation and
extrapolation relative to the convex hull of the training positions and reported separately.

The protocol was pre-registered before any study data was collected. Amendments made on the
rebuilt workstation are listed, dated and justified in
[§8](PROTOCOL.md#8-amendments-to-the-original-pre-registration).

## Hardware
- Seeed SO-ARM101 Pro (leader 5 V / follower 12 V), Feetech STS3215 servos
- Two USB cameras: Logitech C270 overhead on a fixed gantry, Seeed webcam at the wrist, both
  operated at **640 × 480 @ 30 fps**
- Work surface in flat gray primer; two 1000 lm / 4000 K clamp LEDs bounced off the ceiling
- MacBook Air for collection and inference; Colab A100 for training

<p align="center">
 <img src="media/overhead_baseline_lighting.jpg" width="380" alt="Overhead camera view">
 <img src="media/wrist_view.jpg" width="380" alt="Wrist camera view">
</p>

*The only two inputs the policy receives, at the resolution it receives them: overhead (left) and
wrist (right), both 640 x 480.*

## Pipeline
1. `scripts/record_dataset.sh`: teleoperate and record synchronized camera and joint data
2. Train SmolVLA on a cloud GPU (fine-tune from `lerobot/smolvla_base`), log to Weights & Biases
3. `scripts/run_inference.sh`: trained policy drives the arm autonomously

## Tools
`tools/check_cameras.py`: a headless-safe camera probe that saves a frame from each camera
(overhead and wrist) so you can verify framing before recording. Used during camera setup;
written because LeRobot's OpenCV build is headless and can't open a live preview window.

## Status
- [x] Hardware assembled and calibrated
- [x] Teleoperation verified (leader → follower mirroring)
- [x] Pilot dataset recorded, SmolVLA fine-tuned, first autonomous pick
- [x] Workstation rebuilt; protocol pre-registered and amended before collection
- [ ] Four-condition data collection (200 demonstrations)
- [ ] Four training runs
- [ ] Evaluation grid (300 rollouts) and analysis

## Pilot results (superseded by the study)

> **These results come from the previous workstation and are not part of the study.** The bench
> was rebuilt in August 2026 with different camera geometry and a gray work surface in place of
> blue tape, so the pilot data is no longer distribution-matched and is **not** mixed with study
> data. It is kept here because it established that the pipeline works end to end, and because
> the failure it exposed is what motivated the study.

Closed the full pipeline end-to-end: teleoperated data collection, SmolVLA fine-tuning
(10k steps, single A100), autonomous inference on the real arm. The policy reliably picks and
places when the cube is at the trained position (4 consecutive successes).

When the cube was moved off the trained position, the policy reached and missed repeatedly. This was a
direct, observed instance of the generalization gap that the four-condition study is designed to
measure. The clean-data policy nails the in-distribution pose and degrades off it.

<p align="center">
 <img src="media/loss_curve.png" width="500" alt="Training loss">
</p>

Training loss (`train/losses_after_rm_padding`) fell from ~0.19 to ~0.045 over 10k steps,
plateauing around step 6k, the policy converged well within the run.

**Pilot artifacts:** [dataset](https://huggingface.co/datasets/Andresg324/cube-pickup-clean_20260723_151726) · [trained model](https://huggingface.co/Andresg324/smolvla-cube-clean-pilot) (superseded; study artifacts will be linked here as they are published)

## Limitations & observed failure modes
1. **No position generalization:** trained only on the clean (fixed-position) condition, the
   policy reliably fails to grasp when the cube starts outside its trained pose. This is the
   motivating observation for the four-condition study.
2. **Setup-tied:** the policy is bound to the exact camera framing, lighting, background and work
   surface it trained on. This is not incidental, it is why the pilot data was discarded rather
   than reused when the workstation was rebuilt, and why all four conditions are collected on the
   same bench without moving the camera.
3. **Scope:** one arm, one task, 50 demonstrations per condition; results are suggestive, not
   conclusive.
4. **Single training seed per condition** unless time permits more. Per-cell confidence intervals
   capture episode-level uncertainty, not training-run variance. See
   [PROTOCOL.md §9](PROTOCOL.md#9-known-limitations-stated-up-front) for the full list.

## Reproducing
Requires the LeRobot environment (Python 3.12). See [SETUP.md](SETUP.md).