# SO-101 × SmolVLA: How Demonstration-Collection Strategy Shapes Generalization

A controlled empirical study on a self-built low-cost robot arm (Seeed SO-ARM101,
LeRobot platform) using the SmolVLA vision-language-action model.

## The question
A VLA learns to map camera images and language instruction directly to robot motion,
end-to-end from demonstrations (no hand-coded perception or inverse kinematics needed).
This project asks: how does the way you collect demonstrations affect how well the
learned policy generalizes to conditions it never saw?

## Design (see PROTOCOL.md for the full pre-registered protocol)
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
`tools/` holds the hardware bring-up scripts (motor ID scan/repair, camera probe) written
during assembly and calibration.

## Status
- [x] Hardware assembled and calibrated
- [x] Teleoperation verified (leader to follower mirroring)
- [ ] First clean dataset recorded
- [ ] SmolVLA trained; first autonomous pick
- [ ] Four-condition study + analysis

## Reproducing
Requires the LeRobot environment (Python 3.12). See `SETUP.md`.