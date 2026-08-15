# Analysis map

Every number in `PROTOCOL.md`, `README.md` and the paper is produced by a script in this
repository and read back from a CSV, never retyped. This file maps each script to the numbers it
produces, and records the validation of the two measuring instruments the study depends on.

The only file edited by hand is `documents/results_raw_two_seeds.xlsx`, one row per scored
episode. Everything else regenerates from it.

## Run order

```bash
python tools/export_results.py                 # rebuild the derived CSVs
python tools/audit_labels.py                   # label screens, recording-window measurement

python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
python analysis/analyze_exploratory.py
python analysis/seed_variance.py

python tools/endpoints.py --policy clean --cells in_distribution new_positions \
    reduced_lighting different_object distractors
python tools/calibrate_pose.py --apply

python tools/rollout_motion.py
python tools/drops.py
python tools/azimuth_analysis.py
python tools/motion_stats.py <dataset> [...]

python analysis/make_figures.py
```

Order matters in two places only: `endpoints.py` before `calibrate_pose.py --apply`, and both
before `azimuth_analysis.py`. Everything else is independent.

`analyze_results.py` refuses to run on a file containing more than one training seed.
PROTOCOL.md §4.7 and §8.14 forbid pooling seeds, so the guard is a hard exit rather than a
warning.

## Script to number map

| script | writes | numbers it carries |
|---|---|---|
| `tools/export_results.py` | `documents/results_seed1000.csv`, `results_seed2000.csv`, `results_all.csv`, `exploratory.csv` | the canonical derived scores; refuses to write if validation fails |
| `tools/audit_labels.py` | `analysis/out_audit/` | the three label screens; the recording-window measurement (1151 frames, 38.4 s, 23 chunks, 288 ms per forward pass) |
| `analysis/analyze_results.py` | `analysis/out_seed1000/`, `analysis/out_seed2000/` | per-cell success rates, Wilson intervals, the two matched comparisons (Newcombe interval plus Fisher exact), generalization gap |
| `analysis/seed_variance.py` | `analysis/out_seed_variance/` | the same condition compared across seeds |
| `analysis/analyze_exploratory.py` | `analysis/out_exploratory/` | displacement curve, pace comparison, failure-mode breakdowns for both probes |
| `tools/endpoints.py` | `analysis/out_endpoints/` | grasp pose per policy per cell |
| `tools/calibrate_pose.py --apply` | `analysis/out_azimuth/calibration.csv` | the pan-to-bearing fit and its leave-one-out error |
| `tools/rollout_motion.py` | `analysis/out_motion/` | departure latency, deg/step, episode duration, `no_departure` validation, final grip |
| `tools/drops.py` | `analysis/out_drops/` | detector calibration on demonstrations, release events, drop locations, demonstrated release point |
| `tools/azimuth_analysis.py` | `analysis/out_azimuth/` | aim by cell, aim IQR by cell, the clean trajectory envelope, Randomized's aiming error |
| `tools/motion_stats.py` | `analysis/out_pace/pace.csv` | demonstration pace, frame counts, epochs at 10k steps |
| `probing/extract_activations.py` | `probing/out/` (gitignored for size) | action-expert hidden states |
| `probing/probe_position.py` | `analysis/out_probe/position_probe.csv` | bearing decoding, by episode and leave-one-position-out |
| `probing/verify_replay.py` | stdout | the replay-fidelity check behind PROTOCOL.md §9 |
| `analysis/make_figures.py` | `figures/` | Table 1 and the five charts, all read from the CSVs above |

## Instrument validation

### Pan-to-bearing calibration

`azimuth = 0.9485 x pan + 3.669`, fitted on the 50 Randomized training grasps, where the true
cube location is known from the recorded start position.

- Leave-one-out median absolute error **0.86 deg**, 90th percentile 2.22, R2 0.9990.
- Validated at two independent known locations that were not used to fit it. T6 has a true
  bearing of 24.23 and Clean grasps at 23.32 and 24.27. The cup has a true bearing of −25.60 and
  all 332 successes release at a median 2.6 deg from its centre, IQR 1.9 to 3.3.

The cup half-width used to separate a delivery from a drop, 7.2 deg, is derived from the cup's
physical radius and its distance from the arm base. It is not a tuned threshold.

### Release detector

A gripper opening of 12 deg sustained for 5 frames, with a 10 deg azimuth-travel filter.

- On the recovery demonstrations, where the drop is deliberate and its location is known:
  **20/20 recall at 0/30 false positives**.
- On rollouts: 43/43 `deliberate_drop`, 6/6 `success_after_drop`, 6/10 `grasp_drop`.
- All four misses are travel-filtered rather than undetected openings. The largest azimuth travel
  among them is 6.6 deg against the 10 deg threshold, and all four were confirmed as genuine
  drops on video.

**Known blind spot:** the detector cannot see a drop that happens within 10 degrees of the grasp,
which biases the drop-location estimate toward the cup. Every reported drop location should be
read with that bias in mind.

### `no_departure` label

50 episodes carry the label. Scored against a 20 degree arm-joint departure threshold computed
independently from telemetry, there are **0 disagreements across all 662 episodes**.

### Label audit

Three independent telemetry screens over all 662 scored episodes.

- Duration flagged 10 successes near the recording ceiling and 2 failures ending early. All were
  reviewed and explained.
- Cup release in successes: **332 of 332** contain one.
- Cup release in failures: 19 contain one, and all 19 ran the full window, which §6.7 permits.
- Review queue after the audit: **none**.

One transposed pair was found and corrected, `color / in_distribution` episodes 8 and 9. Because
the swap exchanges one success for another inside the same cell, no rate, interval or test
statistic anywhere in the study changed.

Failure-mode distribution across 662 episodes: success 320, timeout_other 163,
contact_no_grasp 63, no_departure 50, deliberate_drop 43, grasp_drop 10, success_after_drop 6,
success_after_missed_grasp 6, cube_out_of_bounds 1, cup_knocked 0.

## What is in the repository but not reportable

- **`probing/train_probes.py`.** Activations were extracted only for the New Positions cell,
  where every policy scored 0/15, so `success` has a single class and the outcome probe is
  undefined on this data. The script is kept for provenance. Use `probe_position.py` instead.
- **The distractor hijack hypothesis.** Randomized's aim does not land on the distractors. What
  the data supports is destabilisation, a fivefold increase in aiming variance under clutter,
  which is the weaker claim.
- **Any telemetry test for "cube still held at the buzzer."** The `action` column is commanded
  joint position, not achieved, so a gripper blocked by a cube still commands full closure. Those
  claims rest on the live notes and the retained video.
- **The grasp-pose comparison across displacement cells,** retired as phase-confounded: the
  gripper closes around frame 215 in distribution and around frame 470 when the cube is
  displaced. The trajectory envelope replaces it.

## Conventions worth knowing before editing anything here

- Training datasets use the camera keys `observation.images.overhead` and
  `observation.images.wrist`; rollout datasets use `camera1` and `camera2`. The rename happens at
  training time via `--rename_map`.
- PROTOCOL.md §3 numbers demonstration passes from 1. Dataset `episode_index` is 0-indexed. The
  two do not line up and have caused errors before.
- Cell identifiers are `eval_cell` everywhere except `drop_locations.csv` and
  `release_events.csv`, which use `cell`. Rename on join or the merge key silently degrades and
  matches across cells.
- Durations computed from recorded data are execution time, not elapsed time. Frames are recorded
  only while an action chunk executes, so 45 s of wall clock is at most 38.4 s of recorded motion.