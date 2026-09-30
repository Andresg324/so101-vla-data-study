# Analysis map

Every number in `PROTOCOL.md`, `README.md` and the two papers is produced by a script in this
repository and read back from a CSV. This file maps each script to the numbers it
produces, and records the validation of the measuring instruments the study depends on.

The only file edited by hand is `documents/results_raw_two_seeds.xlsx`, one row per scored
episode, in three sheets: the August grid and its exploratory probes (707 episodes), the density
sweep (`density`, 610), and the September 24 follow-up session (`density_followup`, 95).
Everything else regenerates from it.

Two papers read from this map. The four-condition study (PROTOCOL.md §1 to §8.30) uses the
August block below. The density sweep paper (PROTOCOL.md §8.32 to §8.38) uses the sweep block,
plus `endpoints.py`, `calibrate_pose.py`, `rollout_motion.py`, `drops.py`,
`azimuth_analysis.py` and `motion_stats.py`, which serve both.

## Run order

### Shared first steps

```bash
python tools/export_results.py             # rebuild every derived CSV; refuses to write on failure
python tools/audit_labels.py               # label screens over all 1412 scored episodes, recording-window measurement
```

### August grid and probes

```bash
python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
python analysis/analyze_exploratory.py     # displacement, demonstration pace and sampling density probes
python analysis/seed_variance.py

for p in clean-seed2000 color color-seed2000 randomized randomized-seed2000 \
         recovery recovery-seed2000; do
  python tools/endpoints.py --policy "$p" --cells in_distribution new_positions \
      reduced_lighting different_object distractors
done
python tools/endpoints.py --policy clean --cells in_distribution new_positions \
    reduced_lighting different_object distractors near_1in near_2in
python tools/endpoints.py --policy density --cells in_distribution trained_t2 new_positions
```

### Density sweep and follow-up session

```bash
python analysis/analyze_density.py         # registered: Wilson per cell, Cochran-Armitage per seed, Newcombe against August
python analysis/analyze_followup.py        # reach probe, §8.38 prediction, rebuilt and retrain Newcombe, failure modes, exact test

# grasp poses for the ten sweep policies, the reach probe, and the two rebuilt-bench cells
for p in density5 density10 density25 density50 \
         density5-seed2000 density10-seed2000 density25-seed2000 density50-seed2000; do
  python tools/endpoints.py --policy "$p" --cells in_distribution distractors spot_check \
      new_positions reach_e5
done
for p in density25-fixedstep density50-fixedstep; do
  python tools/endpoints.py --policy "$p" --cells in_distribution distractors spot_check new_positions
done
python tools/endpoints.py --policy randomized --cells in_distribution_rebuilt new_positions_rebuilt
python tools/endpoints.py --policy randomized-runpod --cells in_distribution_rebuilt
```

`endpoints.py` rebuilds only the cells named on the command line, so the two `randomized` calls
above and in the August block add cells to one file rather than replacing each other.

### Instruments, motion and figures (both papers)

```bash
python tools/calibrate_pose.py --apply     # board coordinates onto every endpoint file
python tools/rollout_motion.py             # latency, velocity, start pose, no_departure validation
python tools/drops.py                      # detector calibration, release events, drop locations
python tools/azimuth_analysis.py           # calibration, aim, reach probe aim, geometry table
python tools/motion_stats.py \
  cube-pickup-clean_20260809_105745 \
  cube-pickup-randomized_20260809_115825 \
  cube-pickup-recovery_20260809_141725 \
  cube-pickup-color_20260809_183224 \
  cube-pickup-color_20260809_130649 \
  cube-pickup-density_20260822_194111 \
  cube-pickup-densitypool_20260908_125700

python tools/verify_pool_mapping.py ~/.cache/huggingface/lerobot/<hf-user>/cube-pickup-densitypool_20260908_125700
python tools/check_norm_stats.py           # sweep normalizers against the pool statistics
python tools/bench_compare.py              # rebuilt bench against the August reference frame; interactive clicks

python analysis/make_figures_density.py    # the density sweep paper's figures; reads the endpoint files and trackers
python analysis/make_figures.py            # the four-condition paper's figures; last
```

### Probes (four-condition paper only)

```bash
for p in clean clean-seed2000 color color-seed2000 \
         randomized randomized-seed2000 recovery recovery-seed2000; do
  python probing/extract_activations.py --policy "$p" \
    --layer model.vlm_with_expert.lm_expert.norm --device mps --seed 0
done
python probing/probe_position.py
python probing/probe_success.py --sweep
python tools/annotate_bench.py media/bench_wide.jpeg   # manual, only when the photo changes
```

Order matters in four places: `export_results.py` first; every `endpoints.py` call before
`calibrate_pose.py --apply`; both before `azimuth_analysis.py`; and the two figure scripts last.
Everything else is independent.

`analyze_results.py` refuses to run on a file containing more than one training seed.
PROTOCOL.md §4.7 and §8.14 forbid pooling seeds, so the guard is a hard exit rather than a
warning. `analyze_density.py` runs the Cochran-Armitage trend separately per seed for the same
reason.

`endpoints.py` requires `--policy` and rebuilds only the cells named on the command line,
keeping the rest of the file untouched, so regenerate every cell of every policy after any
change to the grasp detector. There are twenty endpoint files: the eight registered August
policies, `density`, the ten sweep policies, and `randomized-runpod`. `near_1in` and `near_2in`
exist for the Clean policies only, `trained_t2` for `density` only, `reach_e5` for the eight
epoch-matched sweep policies only, and the `_rebuilt` cells for `randomized` and
`randomized-runpod` only. `color-slowpace` has no endpoint file because no azimuth analysis uses
it.

### One input that does not regenerate from the spreadsheet

`documents/training_loss.csv` is exported once from Weights and Biases and committed. It is
fetched **by run id**, not by run name, because the display names are ambiguous: three runs are
called `smolvla_clean` and two `smolvla_color`. The run id for each condition is in
`configs/train_config_<condition>.json` under `wandb.run_id`. The W&B project holds the ten
August policies, the pilot from the old bench (not part of any study and not exported), the ten
sweep policies and the retrain.

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

W&B marks every August run "crashed" at step 9,800. That is a Colab artifact: the session ended
without a clean `wandb.finish()`, and 9,800 is the last logged point at a 200 step interval.
Training ran to 10,000 and the checkpoints exist. The sweep runs on RunPod finished cleanly and
log through their final step.

Two definitions of "final loss" are in circulation and they differ by about 0.001. `table_loss`
reports the last logged value at step 9,800; `fig_loss` draws a three-point rolling mean. The
figure annotation is taken from the raw last value so the table and the figure agree. The sweep
paper's Table 3 reports the last logged `train/losses_after_rm_padding` at each run's final step.

## Script to number map

| script | writes | numbers it carries |
|---|---|---|
| `tools/export_results.py` | `documents/results_full.csv`, `results.csv`, `results_seed1000.csv`, `results_seed2000.csv`, `exploratory.csv`, `results_all.csv`, `density.csv`, `followup.csv` | the canonical derived scores; validates the registered grid, the exploratory sheet, the sweep sheet (610 rows, four cells per policy) and the follow-up sheet (95 rows) against the vocabulary, cell sizes and success/label agreement, and refuses to write anything if a check fails |
| `tools/audit_labels.py` | `analysis/out_audit/` | the three label screens over all scored episodes; the recording window measurement (1151 frames, 38.4 s, 23 chunks, 288 ms per forward pass); the release detector coverage by threshold, August against sweep |
| `analysis/analyze_results.py` | `analysis/out_seed1000/`, `analysis/out_seed2000/` | per cell success rates, Wilson intervals, the two matched comparisons (Newcombe interval plus Fisher exact), generalization gap |
| `analysis/seed_variance.py` | `analysis/out_seed_variance/` | the same condition compared across seeds |
| `analysis/analyze_exploratory.py` | `analysis/out_exploratory/` | displacement curve, pace comparison, density probe rates against Clean seed 1000, and failure mode breakdowns for all three probes |
| `analysis/analyze_density.py` | `analysis/out_density/cell_rates.csv`, `trend.csv`, `cross_condition.csv` | the sweep paper's Table 1 with Wilson intervals; Cochran-Armitage across density per seed at T6 and New Positions (registered); Newcombe intervals for every sweep policy against Clean, Randomized and Density at the same seed, at T6 on 15 against 15 and at New Positions on the first 3 episodes per position |
| `analysis/analyze_followup.py` | `analysis/out_followup/reach.csv`, `randomized_followup.csv`, `randomized_failure_modes.csv`, `failure_modes_by_policy.csv` | reach probe success with Wilson intervals and the §8.38 prediction check; Newcombe intervals for the rebuilt-bench re-evaluation (§8.36) and the retrain (§8.37); failure modes per policy at E3, E5 and the reach point (the sweep paper's Table 6); the policy-level exact test on the reach split, printed to stdout |
| `tools/endpoints.py` | `analysis/out_endpoints/endpoints_<policy>.csv` | grasp pose per policy per cell, and the count of episodes with no close event |
| `tools/calibrate_pose.py` | stdout; `--apply` adds `grasp_x` and `grasp_y` to the endpoint CSVs | the joint to board coordinate model and its leave-one-out error, and the within-position spread table behind §8.19 |
| `tools/rollout_motion.py` | `analysis/out_motion/` | departure latency, deg/step, episode duration, start pose per cell, `no_departure` validation |
| `tools/drops.py` | `analysis/out_drops/` | detector calibration on demonstrations, release events, drop locations, demonstrated release point |
| `tools/azimuth_analysis.py` | `analysis/out_azimuth/` | the pan to bearing calibration and its leave-one-out error (`calibration.csv`); aim by cell and aim IQR by cell; the clean trajectory envelope; Randomized's aiming error; the density probe's `density_aim.csv`; for the sweep, grasp coverage at held-out positions, `density_aiming.csv` (aim by policy and held-out position, with the nearest trained bearing), `sweep_trained_aim.csv` (aim at T6 and the spot check), `settled_new_positions.csv`, `reach_aim.csv`, `randomized_rebuilt_e5.csv`, `randomized_t6_aim.csv`, and `geometry.csv` (bearing, radius, bracket, nearest trained position and hull membership for every held-out position and the reach point) |
| `tools/motion_stats.py` | `analysis/out_pace/pace.csv` | demonstration pace, frame counts, departure latency and its spread, epochs at 10k steps, for the six August datasets and the 520-demonstration pool; the four-condition paper's velocity to completion-time rank correlation (n = 5) uses the five registered datasets |
| `tools/verify_pool_mapping.py` | stdout | the pool's index-to-position mapping check: bearing through the August fit and within-position base rotation spread at all ten positions (§8.32) |
| `tools/check_norm_stats.py` | `norm_stats_check.csv` | every sweep checkpoint's normalizer against the pool statistics and against its own subset's statistics; the pool match is the `10^-6` claim in the sweep paper's Appendix B |
| `tools/bench_compare.py` | stdout; frames archived to `media/bench_rebuild/` | overhead luminosity by patch and geometry shift from three surface marks, rebuilt bench against the August reference; the geometry step is interactive |
| `probing/extract_activations.py` | `probing/out_np/` (gitignored for size) | action expert hidden states from `lm_expert.norm`, one row per forward pass |
| `probing/probe_position.py` | `analysis/out_probe/position_probe.csv` | bearing decoding, by episode and leave-one-position-out, with cluster bootstrap intervals; see the baseline note below |
| `probing/probe_success.py` | `analysis/out_probe/success_probe.csv`, `success_sweep.csv` | outcome decoding at an information cutoff swept across the episode, with a full episode control and a within cell permutation null |
| `probing/verify_replay.py` | stdout | the replay fidelity check behind PROTOCOL.md §9 |
| `analysis/make_synthetic_results.py` | `analysis/out/` | synthetic scores used to exercise the analysis path before real data existed. Not a study output. |
| `probing/make_synthetic_activations.py` | `probing/out_synthetic/` (gitignored) | synthetic activations used to exercise the probe path. Not a study output. |
| `analysis/make_figures.py` | `figures/` | the four-condition paper's grid and matched-comparison tables, the training loss table, and its nine figure PDFs, all read from the CSVs above |
| `analysis/make_figures_density.py` | `figures/aim_error_density`, `aim_error_fixedstep`, `aim_error_randomized`, `positions_two_axes`, `reach_probe` (png and pdf); `analysis/out_azimuth/aim_error_*.csv`, `reach_probe_success.csv`, `reach_probe_aim.csv`; `figures/table_aim_error_*.md` | the sweep paper's Figures 1, 2, 3, 5 and 6, from the endpoint files, the trackers and the position geometry |
| `tools/annotate_bench.py` | `figures/figure_A1_annotated.png` | the labeled apparatus photo for the four-condition paper. Manual and outside the pipeline: run it, look at the output, nudge the fractional label positions, repeat. `make_figures.py` does **not** regenerate it, so a clean of `figures/` loses it until this is re-run. |

`calibrate_pose.py` reports a leave-one-out error of 2.79 inches, which belongs to the joint to
board coordinate model used by `--apply` and is not the instrument either paper reports. The
papers' instrument is `azimuth_fit()`, whose leave-one-out error of 0.86 degrees is printed by
`azimuth_analysis.py` and written to `analysis/out_azimuth/calibration.csv`. The per-axis errors
of the board model, 1.21 in along the frame's x and 2.28 in along its y, are why radius is never
read from telemetry.

`analyze_results.py` and `seed_variance.py` compute Wilson, Newcombe and Fisher on pooled
episodes, which treats them as independent. They are not: episodes within a cell share a scene.
Per-cell intervals at n = 15 are unaffected, but the pooled 75-episode intervals in
`seed_variance.py` are narrower than a cluster-corrected version would be. That script bounds
seed sensitivity; it is not a reportable test.

The exact test in `analyze_followup.py` treats each of the eight epoch-matched sweep policies as
one unit, since density varies at the policy level, and enumerates all 70 assignments of four
policies to the high-density group. The split between 10 and 25 per position was chosen after
the seed 1000 results, so the test is exploratory; the registered evidence at seed 2000 is the
§8.38 per-policy prediction.

The last point on the displacement curve of `analysis/analyze_exploratory.py` is a range rather
than a single distance: `new_positions` spans E1 to E5 at 3.5 to 14 inches from T6, plotted as
one point. Read the curve as three measured displacements plus a far field bound, not as four
equally spaced steps.

`aim_by_cell.csv` is written to three decimals rather than two so that a difference taken from
it and then rounded for display matches the value computed at full precision. Rounding twice
moved one figure annotation by 0.1 degrees before this was fixed.

The `density` row of `aim_by_cell.csv` is partial: that policy ran three cells, none of them
`reduced_lighting`, `different_object` or `distractors`, so its same-target spread is computed
from a single value and comes out as 0.0, and its all-cell spread of about 52 degrees is just
the T6 to T2 separation. Neither number is meaningful and neither belongs in a table. The same
caveat applies to `color-slowpace`, which ran two of the four same-target cells, and to the
sweep policies, which ran `in_distribution` and `distractors` but neither `reduced_lighting` nor
`different_object`; their same-target spread is the T6 to distractor difference only.

### Label vocabulary

`export_results.py` canonicalizes two labels written at scoring time:

```
success_after_recovery  ->  success_after_missed_grasp
success_after_regrasp   ->  success_after_drop
```

The spreadsheet deliberately keeps the as-scored vocabulary; the derived CSVs carry the
canonical one. Do not rewrite the spreadsheet to match, and expect the two to differ when
comparing them directly.

`results_all.csv` concatenates the 600 registered and 107 exploratory August rows and carries a
`source` column marking which is which. Filter on it before treating that file as the grid. The
sweep and follow-up rows live in `density.csv` and `followup.csv` and are never concatenated
with the August files.

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
covers six of the eight: Clean at seed 1000 has 5 episodes in the minority outcome and Color at
seed 2000 has 4, so both are skipped rather than fitted. Of the six, only Randomized at seed 1000
clears its permutation null before the grasp (p = 0.046), which does not survive Bonferroni
correction for six tests (α = 0.0083), and it does not replicate at seed 2000 (p = 0.176).

**Position probe baseline, under revision.** `probe_position.py` reports chance as the median
absolute error of the mean bearing, 42 degrees. That is not the strongest constant predictor
under the median absolute error the probe is scored by, so it overstates how far the by-episode
decoding beats chance. The leave-one-position-out split needs a different correction, since its
constant must be fit on the four training positions only. The script is being changed to fit the
best constant per fold on the training folds under the same metric, and the chance comparison in
the four-condition paper's Figure 13 will be restated from that output. No claim in the density
sweep paper uses the probe.

**Hook the norm, not a decoder layer.** LeRobot's SmolVLA runs a custom interleaved forward that
calls each layer's submodules directly, so a forward hook on `lm_expert.layers.N` never fires.
`--list-layers` shows the layer and `get_module` resolves it, because the module
exists; the only symptom is "0 activations from N episodes".

**Only the action expert is trained.** `train_config.json` records `freeze_vision_encoder: true`
and `train_expert_only: true`, and LeRobot's `set_requires_grad()` puts the whole VLM in eval
mode with `requires_grad=False`. All twenty-one policies therefore share an identical perceptual
backbone; the probed module is the only part fine-tuning modifies. `tools/norm_sweep.py`
confirmed this at the weights for the ten August checkpoints (PROTOCOL.md §8.31).

### Pan to bearing calibration

`azimuth = 0.9485 x pan + 3.669`, fitted on the 50 Randomized training grasps, where the true
cube location is known from the recorded start position.

- Leave-one-out median absolute error **0.86 deg**, 90th percentile 2.22, R2 0.9990.
- Validated at two independent known locations that were not used to fit it. T6 has a true
  bearing of 24.23 and Clean grasps at 23.32 and 24.27. The cup has a true bearing of −25.64 and
  every success releases at a median 2.6 deg from its center, IQR 2.0 to 3.3.
- The density probe grasps at T2 at a median −32.55 against a true −30.96. T2 is one of the ten
  positions the fit was made on, so this is a consistency check rather than independent
  validation, but it confirms the model holds 55 degrees away from T6 for a policy that never
  saw the fitting data.
- The fit is not repointed at the 520-demonstration pool. `verify_pool_mapping.py` runs all 520
  grasps through it: every position's mean bearing lands within 2.3 degrees of its true bearing,
  with a within-position standard deviation of 0.75 to 2.10 degrees of base rotation. That is
  also the check that the pool's index-to-position mapping survived its five documented
  interruptions (RUN_SHEET.md Part E).

The cup half width used to separate a delivery from a drop, 7.2 deg, is derived from the cup's
physical radius and its distance from the arm base. It is not a tuned threshold.

`calibrate_pose.load_training` also prints the spread of commanded pan within each training
position. That table is the check behind PROTOCOL.md §8.19: the Randomized recording session
crashed and resumed, so the episode to position mapping is reconstructed rather than logged, and
within-position spreads of 0.38 to 1.93 degrees of pan confirm the assumed mapping held. A one
step offset would have scattered the affected grasps across positions tens of degrees apart.

### Geometry

`azimuth_analysis.py` writes `analysis/out_azimuth/geometry.csv`, one row per held-out position
and the reach point: bearing and radius from the base at (11.0, 0), the two trained bearings
that bracket it, the nearest trained bearing and its gap, the nearest trained position and its
distance, the radius margin against the furthest trained radius (15.21 in at T3), whether the
point lies inside the convex hull of the ten training positions, and the cube's angular
half-width at that radius. Every geometric number in the sweep paper reads from this file. The
held-out coordinates are also declared in `probing/probe_position.py` and
`analysis/make_figures_density.py`; the three copies must agree.

### Grasp event detection

`tools/grasp.py` is shared by `endpoints.py` and `calibrate_pose.py`. It finds the first
sustained gripper opening and takes the pose at the last frame before the gripper begins to
close, which is robust to the gripper stalling on the cube rather than reaching the fully closed
threshold.

An episode whose first opening runs to the last recorded frame has no close event and returns no
pose. That is not a detector failure: it is the arm reaching out, finding nothing, and hovering
open until the window expires. Those episodes are reported per run as "no grasp detected" and
are excluded from every grasp pose statistic, which is why per cell counts in
`endpoints_<policy>.csv` are below 15 in the cells where policies fail most. In the sweep,
coverage at held-out positions is 237 of 250 episodes, lowest at 5 per position (0.80 and 0.84),
so aim at the lowest budget is summarized over the episodes that closed the gripper.

### Release detector

A gripper opening of 12 deg sustained for 5 frames, with a 10 deg azimuth travel filter. This is
a different threshold from the grasp event detector above, calibrated for a different question.

- On the recovery demonstrations, where the drop is deliberate and its location is known:
  **20/20 recall at 0/30 false positives**.
- On rollouts: 43/43 `deliberate_drop`, 6/6 `success_after_drop`, 6/10 `grasp_drop`.
- All four misses are travel filtered rather than undetected openings. The largest azimuth
  travel among them is 6.6 deg against the 10 deg threshold, and all four were confirmed as
  genuine drops on video.

**The screen is uninformative for the density probe's `trained_t2` cell.** T2 sits at a bearing
of −31.0 degrees and the cup at −25.6, a separation of 5.4 degrees that falls inside the cup's
7.2 degree half-width, so every gripper opening at the cube registers as a cup release. That is
also why eight `density` / `new_positions` rows appear in the audit's failures-with-a-release
list: those are openings near T2, not near-misses at the cup. The same 5.4 degrees is below the
10 degree `MIN_TRAVEL` filter, which is why `drops.py` reports no density events at all.
Density's successes at T2 rest on visual scoring under PROTOCOL.md §6.6 alone, without the
telemetry corroboration every other cell receives.

**The detector finds gripper openings, not cube releases.** It was calibrated on demonstrations,
where the teleoperator always has the cube in hand. On rollouts a policy can close on nothing and
open again. Of the episodes with at least one detected drop, 55 carry a drop consistent label
(`deliberate_drop` 43, `grasp_drop` 6, `success_after_drop` 6) and 34 do not
(`contact_no_grasp` 20, `timeout_other` 14). The fourteen `timeout_other` episodes were reviewed
on the overhead and wrist cameras; none showed a grasp, and all were retained as timeouts rather
than recoded.

**It under-detects the sweep policies.** The threshold was calibrated on human demonstrations
that open the gripper fully; the sweep policies open less. At the registered threshold of 12
degrees, 18 of 271 September successes contain no detected release against 0 of 362 in August,
and the count falls to 9 at a threshold of 10 and 4 at 8 (`audit_labels.py`). No label depends
on the detector; this is a coverage limit on the second audit screen.

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

50 episodes carry the label, all of them in the August grid; no sweep or follow-up episode
failed to depart. Scored against a 20 degree arm joint departure threshold computed
independently from telemetry, there are **0 disagreements across all scored episodes**
(`analysis/out_motion/no_departure_disagreements.csv` is empty).

The threshold was chosen by sweeping candidate values against the labels. The two groups do not
overlap: the largest maximum joint deviation among `no_departure` episodes is 15.9 degrees and
the smallest among departing episodes is 59.5 degrees, so every threshold from 16 to 59 degrees
reproduces every label exactly. 20 is the conservative end of that band, and the `departed`
column in `rollout_motion_episodes.csv` is exactly `max_dev_deg > 20`.

### Start pose

`rollout_motion.py` reports the median commanded base rotation, wrist flex and gripper at the
first frame of every cell. The home pose was not controlled across sessions. Median start
rotation ranges from about 0.5 to 7.8 degrees across sweep cells and from 4.8 to 7.8 within the
September 24 follow-up session; on the rebuilt bench the August Randomized policy started about
6 degrees of base rotation and 8 degrees of wrist flex from its August start pose. These are the
numbers behind the start-pose limitation in the sweep paper.

### Label audit

Three independent telemetry screens over every scored episode: 662 on August 14, extended to 707
when the density probe closed on August 22, to 1317 when the sweep closed, and to 1412 when the
follow-up session closed. Each looks for a specific kind of mislabel, and every flagged episode
was reviewed on video.

- **Duration.** A success that runs to the recording ceiling, or a failure that ends early,
  contradicts the usual shape of each outcome. In the August grid ten successes and two failures
  were flagged, reviewed and explained, and no label changed. In the sweep, 22 successes sit at
  the ceiling because successful episodes were left running until the arm returned home
  (RUN_SHEET.md Part E, deviation 3), and the density5 New Positions episodes at seed 1000 ended
  3 to 8 s early because the record loop slowed during that cell; all 25 were failures.
- **Cup release in successes.** A success claims the cube ended in the cup, so the gripper
  must have opened while pointing at the cup at some point in the episode. All 362 August
  successes contain such an opening; 24 sweep successes do not, which is the under-detection
  described above, and each was reviewed on video.
- **Cup release in failures.** The mirror check: a failure containing an opening at the cup
  looks, from telemetry alone, like a delivery scored wrong. Forty-nine do across both
  collections. None is a misscore. The detector sees the gripper open, not the cube leave, so an
  empty gripper opening over the cup produces the same signature; 48 of the 49 ran the full
  window, meaning the arm kept working after that opening rather than stopping; and eight of the
  August ones are density episodes at held-out positions where the arm parked near T2, which is
  angularly indistinguishable from the cup (see the release detector above).
- **Review queue after the audit.** Two episodes were flagged by more than one screen, one in
  the sweep and one in the follow-up session; both were reviewed on video and explained.

In the August grid one transposed pair was found and corrected, `color / in_distribution`
episodes 8 and 9. Because the swap exchanges one success for another inside the same cell, no
rate, interval or test statistic anywhere in that study changed. In the sweep, 21 live-flagged
and 6 audit-flagged episodes were re-scored from video; one label changed (density50-fixedstep
distractors episode 12, `timeout_other` to `contact_no_grasp`) and no success value changed.

### Seed 1000 `timeout_other` census (August grid)

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

Failure mode distribution across all 707 August episodes, after the seven recodes: success 350,
timeout_other 171, contact_no_grasp 70, no_departure 50, deliberate_drop 43, grasp_drop 10,
success_after_drop 6, success_after_missed_grasp 6, cube_out_of_bounds 1, cup_knocked 0. The
density probe contributes 30 successes and 15 timeouts to that total. Regenerate this line from
`documents/results_all.csv`. Sweep failure modes were coded live at both seeds and are tabulated
per policy in `analysis/out_followup/failure_modes_by_policy.csv`.

### Local dataset census

`tools/rollout_paths.discover()` finds 100 scored rollout datasets: 4 conditions x 5 cells x 2
seeds (40), Clean at `near_1in` and `near_2in` at both seeds (4), the two slow pace cells (2),
the three density probe cells (3), the ten sweep policies x 4 cells (40), the reach probe for
the eight epoch-matched sweep policies (8), the August Randomized policy's two rebuilt-bench
cells (2) and the retrain's T6 cell (1). The bench confirmation rollout
`rollout_clean_in_distribution_20260908_110904` is a 101st directory that no analysis scores.
`discover()` warns about any `rollout_*` directory that fails the naming pattern rather than
skipping it silently, since a name missing its timestamp is invisible to every tool.

Rollout dataset names carry a date and time suffix, which is how session order is recovered:
the sweep's New Positions cells ran in ascending density at both seeds (September 10 to 11), and
both reach probe sessions ran 50, 25, 10, 5 (September 24, 15:00 to 15:19 and 15:55 to 16:16).

## What is in the repository but not reportable

- **`probing/train_probes.py`.** Activations were extracted only for the New Positions cell,
  where every August policy scored 0/15, so `success` has a single class and the outcome probe is
  undefined on this data. The script is kept for provenance. Use `probe_position.py` instead.
- **The distractor hijack hypothesis.** Randomized's aim does not land on the distractors. What
  the data supports is destabilization: its commanded bearing shifts by 8.4 degrees under
  clutter at seed 2000 and 2.3 at seed 1000, against at most 2.3 for every other condition, and
  its interquartile spread under clutter is 8.4 degrees against at most 1.6 elsewhere. That is
  the weaker and better supported claim.
- **Any telemetry test for "cube still held at the buzzer."** The `action` column is commanded
  joint position, not achieved, so a gripper blocked by a cube still commands full closure. Those
  claims rest on the live notes and the retained video.
- **Any telemetry measure of radius.** The board model's 2.3 in error along the frame's depth is
  larger than the radius differences the sweep paper turns on, so how far the arm extended at E5
  rests on the failure labels and the retained video, never on `grasp_y`.
- **The grasp pose comparison across displacement cells,** retired as phase confounded: the
  gripper closes around frame 215 in distribution and around frame 470 when the cube is
  displaced. The trajectory envelope replaces it.
- **The density probe's same-target aim spread**, for the reason given under the script map: it
  ran one of the four cells that metric is defined over.
- **The loss-landscape scripts** `tools/deterministic_loss.py`, `tools/norm_sweep.py` and
  `tools/sharpness.py`, with their outputs in `analysis/out_loss` and `analysis/out_retrain`.
  They belong to PROTOCOL.md §8.31, which is reported separately from both papers; nothing in
  either paper reads from them except the frozen-backbone confirmation noted above.
- **The position probe on the sweep policies.** `extract_activations.py --match-cell` and
  `probe_position.py --cell` can run on the sweep and rebuilt-bench rollouts, but no result is
  reported until the baseline correction above is in.

## Conventions worth knowing before editing anything here

- Training datasets use the camera keys `observation.images.overhead` and
  `observation.images.wrist`; rollout datasets use `observation.images.camera1` and
  `observation.images.camera2`, where camera1 is the overhead gantry and camera2 the wrist. The
  rename happens at training time via `--rename_map`. Code that picks a camera by the substring
  `overhead` or `camera1` resolves to the overhead view in both. The policy config also declares
  a third slot, `camera3`, inherited from `smolvla_base` and filled by no dataset in the study
  (PROTOCOL.md §8.29).
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
  either by prefix. The same trap applies to condition names: `"color-slowpace".startswith("color")`
  is true, `"density5".startswith("density")` is true, and `density` is its own August policy, so
  match conditions exactly.
- Sweep policy names carry the seed and the control as suffixes: `density25`,
  `density25-seed2000`, `density25-fixedstep`. `parse_policy()` in `tools/rollout_paths.py`
  splits them into the `condition` and `seed` columns the trackers use.
- `trained_t2` is a cell that exists only for the density policy, `reach_e5` only for the eight
  epoch-matched sweep policies, and the `_rebuilt` cells only for `randomized` and
  `randomized-runpod`. Any allowlist of evaluation cells has to include them, and any code that
  assumes a fixed number of cells per policy does not hold.
- The sweep trackers use `instance` for the position within a cell (`E1` to `E5`, `T1`, `T3`,
  `T10`, `reach_e5`). New Positions episodes 1 to 5 are E1, 6 to 10 E2, and so on to 21 to 25
  at E5; the spot check runs T1, T3, T10 in that order, two episodes each.
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
- Figure sizes in `make_figures.py` and `make_figures_density.py` are set to the width each
  panel is printed at, so text inside a figure matches the caption size. Changing a `figsize`
  without changing the `\includegraphics` width in the paper rescales the text.