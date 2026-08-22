# Analysis map

Every number in `PROTOCOL.md`, `README.md` and the paper is produced by a script in this
repository and read back from a CSV, never retyped. This file maps each script to the numbers it
produces, and records the validation of the measuring instruments the study depends on.

The only file edited by hand is `documents/results_raw_two_seeds.xlsx`, one row per scored
episode. Everything else regenerates from it.


## Run order

```bash
python tools/export_results.py             # rebuild the derived CSVs, refuses to write on failure
python tools/audit_labels.py               # label screens, recording-window measurement

python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
python analysis/analyze_exploratory.py
python analysis/seed_variance.py

python tools/endpoints.py --policy clean --cells in_distribution new_positions \
    reduced_lighting different_object distractors near_1in near_2in
python tools/calibrate_pose.py --apply

python tools/rollout_motion.py
python tools/drops.py
python tools/azimuth_analysis.py
python tools/motion_stats.py \
  cube-pickup-clean_20260809_105745 \
  cube-pickup-randomized_20260809_115825 \
  cube-pickup-recovery_20260809_141725 \
  cube-pickup-color_20260809_183224 \
  cube-pickup-color_20260809_130649

for p in clean clean-seed2000 color color-seed2000 \
         randomized randomized-seed2000 recovery recovery-seed2000; do
  python probing/extract_activations.py --policy "$p" \
    --layer model.vlm_with_expert.lm_expert.norm --device mps --seed 0
done

python probing/probe_position.py
python probing/probe_success.py --sweep

python tools/annotate_bench.py media/bench_wide.jpeg   # manual, only when the photo changes
python analysis/make_figures.py               # last: reads everything above
```

Order matters in three places only: `endpoints.py` before `calibrate_pose.py --apply`, both
before `azimuth_analysis.py`, and `make_figures.py` last. Everything else is independent.

`analyze_results.py` refuses to run on a file containing more than one training seed.
PROTOCOL.md §4.7 and §8.14 forbid pooling seeds, so the guard is a hard exit rather than a
warning.

`endpoints.py` rebuilds only the cells named on the command line and keeps the rest of the file
untouched, so regenerate every cell of every policy after any change to the grasp detector.
`near_1in` and `near_2in` exist for the Clean policies only.

### One input that does not regenerate from the spreadsheet

`documents/training_loss.csv` is exported once from Weights and Biases and committed. It is
fetched **by run id**, not by run name, because the display names are ambiguous: three runs are
called `smolvla_clean` and two `smolvla_color`. The run id for each condition is in
`configs/train_config_<condition>.json` under `wandb.run_id`. The W&B project also holds a tenth
run, the pilot from the old bench, which is not part of the study and is not exported.

```bash
python - <<'EOF'
import glob, json, os
import pandas as pd, wandb
api = wandb.Api(); rows = []
for p in sorted(glob.glob("configs/train_config_*.json")):
    cond = os.path.basename(p)[len("train_config_"):-len(".json")]
    wb = json.load(open(p)).get("wandb") or {}
    rid, ent, proj = wb.get("run_id"), wb.get("entity"), wb.get("project")
    r = api.run(f"{ent}/{proj}/{rid}" if ent else f"{proj}/{rid}")
    h = r.history(keys=["train/losses_after_rm_padding"], pandas=True, samples=5000)
    h["run"] = cond
    rows.append(h)
pd.concat(rows).to_csv("documents/training_loss.csv", index=False)
EOF
```

W&B marks all nine runs "crashed" at step 9,800. That is a Colab artifact: the session ended
without a clean `wandb.finish()`, and 9,800 is the last logged point at a 200 step interval.
Training ran to 10,000 and the checkpoints exist.

## Script to number map

| script | writes | numbers it carries |
|---|---|---|
| `tools/export_results.py` | `documents/results_full.csv`, `results.csv`, `results_seed1000.csv`, `results_seed2000.csv`, `exploratory.csv`, `results_all.csv` | the canonical derived scores; validates both the registered grid and the exploratory sheet against the ten-term vocabulary, cell sizes and success/label agreement, and refuses to write anything if a check fails |
| `tools/audit_labels.py` | `analysis/out_audit/` | the three label screens; the recording window measurement (1151 frames, 38.4 s, 23 chunks, 288 ms per forward pass) |
| `analysis/analyze_results.py` | `analysis/out_seed1000/`, `analysis/out_seed2000/` | per cell success rates, Wilson intervals, the two matched comparisons (Newcombe interval plus Fisher exact), generalization gap |
| `analysis/seed_variance.py` | `analysis/out_seed_variance/` | the same condition compared across seeds |
| `analysis/analyze_exploratory.py` | `analysis/out_exploratory/` | displacement curve, pace comparison, failure mode breakdowns for both probes |
| `tools/endpoints.py` | `analysis/out_endpoints/endpoints_<policy>.csv` | grasp pose per policy per cell, and the count of episodes with no close event |
| `tools/calibrate_pose.py` | stdout; `--apply` adds `grasp_x` and `grasp_y` to the endpoint CSVs | the joint to board coordinate model and its leave-one-out error, and the within-position spread table behind §8.19 |
| `tools/rollout_motion.py` | `analysis/out_motion/` | departure latency, deg/step, episode duration, `no_departure` validation |
| `tools/drops.py` | `analysis/out_drops/` | detector calibration on demonstrations, release events, drop locations, demonstrated release point |
| `tools/azimuth_analysis.py` | `analysis/out_azimuth/` | the pan to bearing calibration and its leave-one-out error, aim by cell, aim IQR by cell, the clean trajectory envelope, Randomized's aiming error |
| `tools/motion_stats.py` | `analysis/out_pace/pace.csv` | demonstration pace, frame counts, epochs at 10k steps |
| `probing/extract_activations.py` | `probing/out_np/` (gitignored for size) | action expert hidden states from `lm_expert.norm`, one row per forward pass |
| `probing/probe_position.py` | `analysis/out_probe/position_probe.csv` | bearing decoding, by episode and leave-one-position-out, with cluster bootstrap intervals |
| `probing/probe_success.py` | `analysis/out_probe/success_probe.csv`, `success_sweep.csv` | outcome decoding at an information cutoff swept across the episode, with a full episode control and a within cell permutation null |
| `probing/verify_replay.py` | stdout | the replay fidelity check behind PROTOCOL.md §9 |
| `analysis/make_synthetic_results.py` | `analysis/out/` | synthetic scores used to exercise the analysis path before real data existed. Not a study output. |
| `probing/make_synthetic_activations.py` | `probing/out_synthetic/` (gitignored) | synthetic activations used to exercise the probe path. Not a study output. |
| `analysis/make_figures.py` | `figures/` | Tables 1 and 2, the training loss table, and the charts, all read from the CSVs above |
| `tools/annotate_bench.py` | `figures/figure_A1_annotated.png` | the labeled apparatus photo for Figure A1. Manual and outside the pipeline: run it, look at the output, nudge the fractional label positions, repeat. `make_figures.py` does **not** regenerate it, so a clean of `figures/` loses it until this is re-run. |

`calibrate_pose.py` reports a leave-one-out error of
2.79 inches, which belongs to the joint to board coordinate model used by `--apply` and is not
the instrument the paper reports. The paper's instrument is `azimuth_fit()`, whose leave-one-out
error of 0.86 degrees is printed by `azimuth_analysis.py` and written to
`analysis/out_azimuth/calibration.csv`.

`analyze_results.py` and `seed_variance.py` compute Wilson, Newcombe and Fisher on pooled
episodes, which treats them as independent. They are not: episodes within a cell share a scene.
Per-cell intervals at n = 15 are unaffected, but the pooled 75-episode intervals in
`seed_variance.py` are narrower than a cluster-corrected version would be. That script bounds
seed sensitivity; it is not a reportable test.

The last point on the displacement curve of `analysis/analyze_exploratory.py` is a range rather than a single distance:
`new_positions` spans E1 to E5 at 3.5 to 14 inches from T6, plotted as one point. Read the
curve as three measured displacements plus a far field bound, not as four equally spaced steps.

### Label vocabulary

`export_results.py` canonicalizes two labels written at scoring time:

```
success_after_recovery  ->  success_after_missed_grasp
success_after_regrasp   ->  success_after_drop
```

The spreadsheet deliberately keeps the as-scored vocabulary; the derived CSVs carry the
canonical one. Do not rewrite the spreadsheet to match, and expect the two to differ when
comparing them directly.

`results_all.csv` concatenates the 600 registered and 62 exploratory rows and carries a
`source` column marking which is which. Filter on it before treating that file as the grid.

## Instrument validation

### Activation extraction

Hidden states come from `model.vlm_with_expert.lm_expert.norm`, the action expert's final
RMSNorm, 720 wide, captured once per forward pass. About 23 per full length episode, matching
the 23 action chunks. This is the final pre-decoding hidden state, the same locus SAFE
(Gu et al., NeurIPS 2025) reads for failure detection.

**The activations are one draw, not a fixed function of the input.** SmolVLA samples action
noise through a 10 step flow matching sampler (`num_steps: 10` in the train config), so
replaying the same episode twice gives different hidden states. Two runs on the same device
correlate at **0.998** (maximum element difference 0.42 against a reference standard deviation
of 0.95); a CUDA run against an MPS run correlates at **0.997**. Those are the same magnitude,
so device choice is irrelevant beside the sampling, and extraction can run anywhere.

`extract_activations.py --seed` seeds per (cell, episode), so a rerun reproduces and a partial
extraction matches the same cells inside a full run. The activations behind the first committed
`position_probe.csv` predate seeding and cannot be reproduced exactly; they were superseded by a
seeded run rather than corrected, since nothing was wrong with them.

`probe_success.py` skips any policy with too few episodes in the minority outcome, so the sweep
covers six of the eight: Clean at seed 1000 has 5 episodes in the 
minority outcome and Color at seed 2000 has 4, so both are skipped 
rather than fitted. Of the six, only Randomized at seed 1000 clears ts
permutation null before the grasp (p = 0.046), and it does not replicate at seed 2000
(p = 0.176).

**Hook the norm, not a decoder layer.** LeRobot's SmolVLA runs a custom interleaved forward that
calls each layer's submodules directly, so a forward hook on `lm_expert.layers.N` never fires.
`--list-layers` shows the layer and `get_module` resolves it, because the module
exists; the only symptom is "0 activations from N episodes".

**Only the action expert is trained.** `train_config.json` records `freeze_vision_encoder: true`
and `train_expert_only: true`, and LeRobot's `set_requires_grad()` puts the whole VLM in eval
mode with `requires_grad=False`. All eight policies therefore share an identical perceptual
backbone; the probed module is the only part fine-tuning modifies.

### Pan to bearing calibration

`azimuth = 0.9485 x pan + 3.669`, fitted on the 50 Randomized training grasps, where the true
cube location is known from the recorded start position.

- Leave-one-out median absolute error **0.86 deg**, 90th percentile 2.22, R2 0.9990.
- Validated at two independent known locations that were not used to fit it. T6 has a true
  bearing of 24.23 and Clean grasps at 23.32 and 24.27. The cup has a true bearing of −25.64 and
  all 332 successes release at a median 2.6 deg from its center, IQR 1.9 to 3.3.

The cup half width used to separate a delivery from a drop, 7.2 deg, is derived from the cup's
physical radius and its distance from the arm base. It is not a tuned threshold.

`calibrate_pose.load_training` also prints the spread of commanded pan within each training
position. That table is the check behind PROTOCOL.md §8.19: the Randomized recording session
crashed and resumed, so the episode to position mapping is reconstructed rather than logged, and
within-position spreads of 0.38 to 1.93 degrees of pan confirm the assumed mapping held. A one
step offset would have scattered the affected grasps across positions tens of degrees apart.

### Grasp event detection

`tools/grasp.py` is shared by `endpoints.py` and `calibrate_pose.py`. It finds the first
sustained gripper opening and takes the pose at the last frame before the gripper begins to
close, which is robust to the gripper stalling on the cube rather than reaching the fully closed
threshold.

An episode whose first opening runs to the last recorded frame has no close event and returns no
pose. That is not a detector failure: it is the arm reaching out, finding nothing, and hovering
open until the window expires. Those episodes are reported per run as "no grasp detected" and
are excluded from every grasp pose statistic, which is why per cell counts in
`endpoints_<policy>.csv` are below 15 in the cells where policies fail most.

### Release detector

A gripper opening of 12 deg sustained for 5 frames, with a 10 deg azimuth travel filter. This is
a different threshold from the grasp event detector above, calibrated for a different question.

- On the recovery demonstrations, where the drop is deliberate and its location is known:
  **20/20 recall at 0/30 false positives**.
- On rollouts: 43/43 `deliberate_drop`, 6/6 `success_after_drop`, 6/10 `grasp_drop`.
- All four misses are travel filtered rather than undetected openings. The largest azimuth
  travel among them is 6.6 deg against the 10 deg threshold, and all four were confirmed as
  genuine drops on video.

**The detector finds gripper openings, not cube releases.** It was calibrated on demonstrations,
where the teleoperator always has the cube in hand. On rollouts a policy can close on nothing and
open again. Of the episodes with at least one detected drop, 55 carry a drop consistent label
(`deliberate_drop` 43, `grasp_drop` 6, `success_after_drop` 6) and 34 do not
(`contact_no_grasp` 20, `timeout_other` 14). The fourteen `timeout_other` episodes were reviewed
on the overhead and wrist cameras; none showed a grasp, and all were retained as timeouts rather
than recoded.

This does not reach the release analysis. `drop_locations.csv` is restricted to episodes
independently labeled as drops, so all 53 recovery rows come from `deliberate_drop` (43),
`success_after_drop` (6) or `grasp_drop` (4), and none from `timeout_other` or
`contact_no_grasp`. The median release bearing is −11.7 deg, IQR −14.4 to −9.4.

**`n_events` does not screen false positives.** All fourteen audited episodes fired exactly once.
The detector logs one event per episode regardless of how many times the gripper cycles.

**Known blind spot:** the detector cannot see a drop that happens within 10 degrees of the grasp,
which biases the drop location estimate toward the cup. Every reported drop location should be
read with that bias in mind.

### `no_departure` label

50 episodes carry the label. Scored against a 20 degree arm joint departure threshold computed
independently from telemetry, there are **0 disagreements across all 662 episodes**
(`analysis/out_motion/no_departure_disagreements.csv` is empty).

The threshold was chosen by sweeping candidate values against the labels. The two groups do not
overlap: the largest maximum joint deviation among `no_departure` episodes is 15.9 degrees and
the smallest among departing episodes is 59.5 degrees, so every threshold from 16 to 59 degrees
reproduces all 662 labels exactly. 20 is the conservative end of that band, and the `departed`
column in `rollout_motion_episodes.csv` is exactly `max_dev_deg > 20`.

### Label audit

Three independent telemetry screens over all 662 scored episodes. Each looks for a specific
kind of mislabel, and every flagged episode was reviewed on video.

- **Duration.** A success that runs to the recording ceiling, or a failure that ends early,
  contradicts the usual shape of each outcome. Ten successes and two failures were flagged;
  all twelve were reviewed and explained, and no label changed.
- **Cup release in successes.** A success claims the cube ended in the cup, so the gripper
  must have opened while pointing at the cup at some point in the episode. All **332**
  successes contain such an opening.
- **Cup release in failures.** The mirror check: a failure containing an opening at the cup
  looks, from telemetry alone, like a delivery scored wrong. Nineteen do. None is a misscore.
  The detector sees the gripper open, not the cube leave, so an empty gripper opening over the
  cup produces the same signature; and all nineteen ran the full window, meaning the arm kept
  working after that opening rather than stopping.
- **Review queue after the audit: none.** No episode was flagged by more than one screen, and
  every singly flagged episode was reviewed on video and explained.

One transposed pair was found and corrected, `color / in_distribution` episodes 8 and 9. Because
the swap exchanges one success for another inside the same cell, no rate, interval or test
statistic anywhere in the study changed.

### Seed 1000 `timeout_other` census

Failure modes at seed 2000 were coded live; seed 1000 codes were assigned retrospectively from
contemporaneous notes and video, so `timeout_other` at seed 1000 is coarser: 57.7% of seed 1000
failures against 41.3% at seed 2000. All 94 seed 1000 `timeout_other` episodes were audited.

- Every one ran to the recording ceiling (median 38.3 s) with substantial joint motion (maximum
  deviation 92 to 192 deg), confirming the timeout label in all cases.
- 85 were re-scored from video, using the wrist camera for the reduced lighting cells and for
  every episode with a detected gripper opening. **Seven were recoded as `contact_no_grasp`.**
- All 37 New Positions episodes were confirmed as no contact timeouts, consistent with the
  scoring convention that contact was recorded at the time.
- The remaining 9 carry contemporaneous notes identifying the failure and were not re-watched.

Failure mode distribution across 662 episodes, after the seven recodes: success 320,
timeout_other 156, contact_no_grasp 70, no_departure 50, deliberate_drop 43, grasp_drop 10,
success_after_drop 6, success_after_missed_grasp 6, cube_out_of_bounds 1, cup_knocked 0.
Regenerate this line from `analysis/out_audit/`.

### Local dataset census

`tools/rollout_paths.discover()` finds 46 rollout datasets, which is exactly 4 conditions x
5 cells x 2 seeds (40), plus Clean at `near_1in` and `near_2in` at both seeds (4), plus the two
slow pace cells (2). It warns about any `rollout_*` directory that fails the naming pattern
rather than skipping it silently, since a name missing its timestamp is invisible to every tool.

## What is in the repository but not reportable

- **`probing/train_probes.py`.** Activations were extracted only for the New Positions cell,
  where every policy scored 0/15, so `success` has a single class and the outcome probe is
  undefined on this data. The script is kept for provenance. Use `probe_position.py` instead.
- **The distractor hijack hypothesis.** Randomized's aim does not land on the distractors. What
  the data supports is destabilization: its commanded bearing shifts by 8.4 degrees under
  clutter at seed 2000 and 2.3 at seed 1000, against at most 2.3 for every other condition, and
  its interquartile spread under clutter is 8.4 degrees against at most 1.6 elsewhere. That is
  the weaker and better supported claim.
- **Any telemetry test for "cube still held at the buzzer."** The `action` column is commanded
  joint position, not achieved, so a gripper blocked by a cube still commands full closure. Those
  claims rest on the live notes and the retained video.
- **The grasp pose comparison across displacement cells,** retired as phase confounded: the
  gripper closes around frame 215 in distribution and around frame 470 when the cube is
  displaced. The trajectory envelope replaces it.

## Conventions worth knowing before editing anything here

- Training datasets use the camera keys `observation.images.overhead` and
  `observation.images.wrist`; rollout datasets use `observation.images.camera1` and
  `observation.images.camera2`, where camera1 is the overhead gantry and camera2 the wrist. The
  rename happens at training time via `--rename_map`. Code that picks a camera by the substring
  `overhead` or `camera1` resolves to the overhead view in both.
- PROTOCOL.md §3 numbers demonstration passes from 1. Dataset `episode_index` is 0 indexed. The
  two do not line up and have caused errors before.
- Rollout readers drop episode 0 as the protocol warmup; training readers keep it, because in a
  demonstration dataset episode 0 is the first demonstration and `calibrate_pose.py` maps
  positions by index. Both conventions are correct and neither should be harmonized to the other.
- Cell identifiers are `eval_cell` everywhere except `drop_locations.csv`, `release_events.csv`,
  `rollout_motion_episodes.csv` and `endpoints_<policy>.csv`, which use `cell`. Rename on join,
  or the merge key silently degrades and matches across cells.
- Two datasets share the `cube-pickup-color_` prefix: `_20260809_130649` is the superseded slow
  pace collection and `_20260809_183224` is the registered Color condition. Never select
  either by prefix.
- Durations computed from recorded data are execution time, not elapsed time. Frames are recorded
  only while an action chunk executes, so 45 s of wall clock is at most 38.4 s of recorded motion.
  The policy executes the full 50 action chunk before re-observing, an open loop chunk rather
  than a receding horizon, which is why the decision rate is 0.600 Hz.
- `motion_stats.py` reports deg/step across all six commanded joints including the gripper;
  `rollout_motion.py` reports a statistic of the same name across the five arm joints only. They
  are not comparable.
- The probe layer is `model.vlm_with_expert.lm_expert.norm` and is recorded in every `.npz` as
  `layer`. Numbers from different layers are not comparable, so all eight policies in a probe
  figure must come from one extraction run.
- Figure sizes in `make_figures.py` are set to the width each panel is printed at, so text inside
  a figure matches the caption size. Changing a `figsize` without changing the
  `\includegraphics` width in the paper rescales the text.