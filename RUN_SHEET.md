# Collection and Evaluation Run Sheet

Companion to PROTOCOL.md. PROTOCOL.md is the source of truth if the two disagree.

**Status: reopened September 8, 2026** for the density sweep (PROTOCOL.md §8.32, Part E below).

- Collection: August 9 to 10, 2026
- Training: August 9 (seed 1000) and August 9 to 10 (seed 2000), single A100 per run
- Seed 1000 evaluation grid: August 10, 2026
- Seed 2000 evaluation grid: August 11, 2026
- Exploratory probes: August 11, 2026
- Analysis closed: August 14, 2026
- Sampling density probe, collected, trained and evaluated: August 22, 2026
- Analysis reclosed: August 22, 2026
- Bench rebuild and verification: September 8, 2026
- Density sweep collection: September 8, 2026
- Density sweep training: September 9, 2026, single A100 per run
- Loss matrix: September 9, 2026, single A40

---

## Every session, before touching data

Run every item at the start of every collection and evaluation session.

- [x] Power: leader 5V, follower 12V. Both clamped.
- [x] Continuity Camera off on the iPhone.
- [x] Wave test via `bash scripts/check_cameras_live.sh`. Confirm which index is overhead and
      which is wrist.
- [x] Set `OVERHEAD_IDX` and `WRIST_IDX` for `record_dataset.sh`, `run_inference.sh`,
      `check_cameras_live.sh` and `check_cameras.py`. All four must agree.
- [x] Confirm both cameras sustain 30 fps at 640 x 480. A camera that silently drops to 5 or
      15 fps corrupts the recorded timing and invalidates every pace and rate figure.
- [x] Leader parked in frame, matching the training view.
- [x] Baseline lighting: blinds closed, room lights off, both clamp LEDs on.
- [x] Work surface clear of stray objects (pencils, erasers, cable ties). A stray object in
      frame is a re-record trigger under PROTOCOL.md §8.20.
- [x] Arm returns to home pose: fully retracted, base forward, joints folded, gripper visible
      in the overhead frame.
- [x] `hf auth whoami` returns the correct username. Write token active.
- [x] Laptop plugged in, sleep disabled, disk space checked.

---

## Part A: collect four datasets

Command: `bash scripts/record_dataset.sh <condition> 50`

The episode window is a ceiling of **45 seconds** (PROTOCOL.md §4.10), not a target. End each
demonstration with the right arrow as soon as the cube is resting in the cup.

Every demonstration is a first try clean success. Anything less, redo it with the left arrow.
**When you redo a demonstration, redo it at the same position or color**, so that the episode
index still maps to the factor level. Re-record only for the reasons listed in
PROTOCOL.md §8.20, never on the basis of task outcome.

| # | Condition | Cube | Position | Instruction | Done |
|---|---|---|---|---|---|
| 1 | clean | red | T6 (15.5, 10.0) every demo | nothing varies | [x] |
| 2 | randomized | red | cycle T1 to T10, five passes | demo 1 at T1, demo 2 at T2 ... demo 11 back at T1 | [x] |
| 3 | recovery | red | T6 every demo | within each group of 5 demos, demos 2 and 4 are drop and recover | [x] |
| 4 | color | cycle red, orange, yellow, blue, purple | T6 every demo | ten full passes through the color sequence, no green | [x] |

**Randomized cycle order:** T1 (2.0, 2.5), T2 (6.5, 7.5), T3 (8.5, 15.0), T4 (12.0, 14.0),
T5 (15.5, 2.5), T6 (15.5, 10.0), T7 (15.5, 14.25), T8 (20.5, 2.5), T9 (20.5, 6.5),
T10 (20.5, 10.0). Repeat five times.

**Recovery drop pattern:** demonstrations 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29, 32, 34,
37, 39, 42, 44, 47, 49. Twenty in total. Drop during the carry from 4 to 5 inches above the
surface, then re-grasp from wherever it lands and complete the task. Specified as
"approximately the midpoint"; measured post hoc at roughly 65% of the carry
(PROTOCOL.md §8.24).

**Color cycle order:** red, orange, yellow, blue, purple. Repeat ten times.

### Datasets as recorded

| Condition | Hub slug | Episodes | Frames | Frames/ep | Sec/ep | Status |
|---|---|---|---|---|---|---|
| clean | `Andresg324/cube-pickup-clean_20260809_105745` | 50 | 27,947 | 558.9 | 18.6 | retained |
| randomized | `Andresg324/cube-pickup-randomized_20260809_115825` | 50 | 31,682 | 633.6 | 21.1 | retained |
| recovery | `Andresg324/cube-pickup-recovery_20260809_141725` | 50 | 32,170 | 643.4 | 21.4 | retained |
| color (superseded) | `Andresg324/cube-pickup-color_20260809_130649` | 50 | 34,511 | 690.2 | 23.0 | superseded, §8.18 |
| color (retained) | `Andresg324/cube-pickup-color_20260809_183224` | 50 | 24,847 | 496.9 | 16.6 | retained |
| density (exploratory, §8.30) | `Andresg324/cube-pickup-density_20260822_194111` | 50 | 26,349 | 527.0 | 17.6 | retained, Part D |

Regenerate the table, with deg/step and epochs, using `python tools/motion_stats.py <dataset> [...]`.

Note that both Color collections keep the `cube-pickup-color_` prefix. The timestamp is the
only thing distinguishing them, and anything matching that prefix by glob picks up both.

The frames per condition are not equal by design; the budget is fixed in episodes, not frames
(PROTOCOL.md §9). Recovery is the longest condition and is therefore trained for the fewest
passes over its own data at a fixed 10,000 steps.

### Deviations during collection

1. **Randomized session crashed after the 41st recorded episode** (August 10) and was resumed
   at the next index with the cube at T2, environment unchanged. Logged as PROTOCOL.md §8.19.
   Verification: the video recordings were reviewed and the index to position mapping was confirmed downstream, since the azimuth
   calibration groups all 50 grasps by derived position and finds a within-position base
   rotation spread of 0.38 to 1.93 degrees at all ten positions. A one-step offset after the
   resume would have scattered the last nine grasps across positions tens of degrees apart, so
   the tight clustering rules it out. `tools/calibrate_pose.py` prints that table on every run.
2. **Demonstrations re-recorded.** §3 requires every demonstration to be a first try clean
   success, so a demonstration judged not clean at the time was discarded and re-recorded.
   Logged as PROTOCOL.md §8.20.
3. **Color-varied re-collected** after the first collection came out 23% slower per
   demonstration than Clean. Logged as PROTOCOL.md §8.18. The superseded dataset was retained
   and reused as the slow pace probe (§8.16). It was never a designed condition: the pace
   difference was unintentional as the teleoperator was trying to match clean's pace and accidentally moved slower. The re-collected version was done naturally, without accounting for clock, and comes out faster due to more experience (this was collected last). The slower collection was
   repurposed after the fact, which is why it is exploratory and why
   `scripts/record_dataset.sh` deliberately refuses `color-slowpace` as a condition.

### After each condition finished

- [x] Hugging Face upload allowed to complete before the next recording started.
- [x] Episode count on the Hub confirmed as 50 for each retained dataset.
- [x] Frame count recorded (table above).
- [x] Colab training run launched for the condition.

---

## Between A and B: train the policies

Identical settings for all conditions within a replication, fixed by PROTOCOL.md §4.7. Not
tuned per condition.

`batch_size=32`, `steps=10000`, `save_freq=2000`, `policy.path=lerobot/smolvla_base`,
`policy.device=cuda`, single A100, LeRobot default optimizer and learning rate schedule. The
evaluated checkpoint is the final one at step 10000. The seed is the only setting that varies
between replications.

The resolved configuration for each run is committed at
`configs/train_config_<condition>.json`. Diffing them shows that only `output_dir`, `seed`,
`dataset.repo_id`, `job_name` and `wandb.run_id` differ. The default schedule is a cosine decay
with 1000 warmup steps and `scheduler_decay_steps: 30000`, so at 10,000 steps the final
checkpoint sits near 79% of peak learning rate rather than fully annealed.

### Seed 1000 (primary), trained August 9

- [x] `Andresg324/smolvla-cube-clean`
- [x] `Andresg324/smolvla-cube-randomized`
- [x] `Andresg324/smolvla-cube-recovery`
- [x] `Andresg324/smolvla-cube-color`

### Seed 2000 (replication), trained August 9 to 10

- [x] `Andresg324/smolvla-cube-clean-seed2000`
- [x] `Andresg324/smolvla-cube-randomized-seed2000`
- [x] `Andresg324/smolvla-cube-recovery-seed2000`
- [x] `Andresg324/smolvla-cube-color-seed2000`

### Exploratory, outside the grid

- [x] `Andresg324/smolvla-cube-color-slowpace`, trained on the superseded color collection,
      seed 1000, otherwise identical settings.
- [x] `Andresg324/smolvla-cube-density`, trained August 22 on the two-position collection,
      seed 1000, otherwise identical settings. See Part D.

Ten policies in total, nine of them trained in August 9 to 10 and the tenth on August 22. The
W&B project holds eleven runs: these ten plus the pilot from the old bench, which is not part of
the study. The pilot's curve is the one in `media/loss_curve.png`, which is therefore not one of
the ten.

---

## Part B: evaluate on five cells

Command: `bash scripts/run_inference.sh <policy> <cell>`

16 episodes recorded per cell, index 0 discarded as warmup, 15 scored. 45 second window, scored
live. Batched by cell across policies so each scene is configured once.

| Cell | Cube | Position | Lighting | Distractors |
|---|---|---|---|---|
| in_distribution | red | T6 | baseline, both LEDs | none |
| new_positions | red | E1 to E5, 3 episodes each, in order | baseline, both LEDs | none |
| reduced_lighting | red | T6 | left LED off, right LED on | none |
| different_object | green | T6 | baseline, both LEDs | none |
| distractors | red | T6 | baseline, both LEDs | four objects, see below |

**Held-out positions:** E1 (2.0, 7.5), E2 (6.5, 2.5), E3 (12.0, 10.0), E4 (15.5, 6.5),
E5 (19.5, 13.5). Three episodes at each. The position identifier is recorded per episode. E2,
E3 and E4 fall inside the convex hull of the ten training positions; E1 and E5 fall outside it.

**Distractor placement, identical for all 15 episodes:** crumpled paper at T2 (6.5, 7.5),
penny at T4 (12.0, 14.0), battery at E4 (15.5, 6.5), screw at T8 (20.5, 2.5).

### Progress grid, seed 1000 (August 10)

| Policy | in_distribution | new_positions | reduced_lighting | different_object | distractors |
|---|---|---|---|---|---|
| clean | [x] | [x] | [x] | [x] | [x] |
| randomized | [x] | [x] | [x] | [x] | [x] |
| recovery | [x] | [x] | [x] | [x] | [x] |
| color | [x] | [x] | [x] | [x] | [x] |

### Progress grid, seed 2000 (August 11)

Cell order fixed in advance per PROTOCOL.md §8.14: In-Distribution, Different Object,
Distractors, Reduced Lighting, New Positions. The order did not depend on any observed outcome.

| Policy | in_distribution | different_object | distractors | reduced_lighting | new_positions |
|---|---|---|---|---|---|
| clean-seed2000 | [x] | [x] | [x] | [x] | [x] |
| randomized-seed2000 | [x] | [x] | [x] | [x] | [x] |
| recovery-seed2000 | [x] | [x] | [x] | [x] | [x] |
| color-seed2000 | [x] | [x] | [x] | [x] | [x] |

**Registered total: 600 scored rollouts** (2 seeds x 4 policies x 5 cells x 15), plus 40
discarded warmup episodes.

### Success criterion

Success is the cube released and resting in the cup, cup upright, within 45 seconds.

- If the cube misses the cup, the arm may retry inside the window.
- Knocking the cup over is an immediate failure.
- A cube knocked out of the reachable and visible area is a failure.
- Score 1 or 0 live, into the tracker, at the time of the rollout.
- Flag anything ambiguous and re-score it from video before analysis.

### Deviations during evaluation

**Approximately five rollouts were re-recorded**, fewer than ten. The exact count was not
logged and is not recoverable, since a re-record replaces the discarded take. One was because a
pencil had been left in the overhead frame, so the scene did not match the cell specification.
One (randomized, seed 1000, in_distribution, episode 2) was re-recorded because the arm did not
depart and the experimenter was unsure whether that counted as an episode; that one was not
outcome independent and is logged separately as PROTOCOL.md §8.26. The rest were control
misfires: a right arrow press as an episode ended made the harness prompt for a recording and a
reset at once, so the next episode never started, no inference ran, and the arm was completely
inert. That inertness is how a misfire is told apart at the time from a scored `no_departure`
episode, in which the policy runs and the arm vibrates slightly without departing.

**One transposed label pair was found and corrected** during an audit: color /
in_distribution / seed 1000 / episodes 8 and 9 carried each other's labels. Both were re-scored
from video, episodes 10 and 11 were checked and match, and because the swap exchanges one
success for another inside the same cell no reported rate changed. Logged as
PROTOCOL.md §8.27.

---

## Part C: exploratory probes (August 11, after the seed 2000 grid closed)

Both were declared exploratory in writing before they were run. Neither is pooled into the four
condition grid.

### C1. Demonstration pace probe (PROTOCOL.md §8.16)

`smolvla-cube-color-slowpace` evaluated on In-Distribution and Different Object, 15 scored
episodes each, **30 rollouts**, compared against the retained Color policy at 16.6 s and
0.3466 deg/step against the slow pace 23.0 s and 0.2481 deg/step. The two datasets are separate
collection sessions, so pace is the measured and manipulated difference but not the only
difference between them.

- [x] in_distribution
- [x] different_object

### C2. Displacement probe (PROTOCOL.md §8.15)

Clean at both seeds evaluated at P1 (15.5, 9.0) and P2 (15.5, 8.0), on the line from T6 to E4.
9 episodes each, index 0 discarded, 8 scored, **32 rollouts** (2 positions x 2 seeds x 8).

- [x] clean seed 1000 at P1, P2
- [x] clean seed 2000 at P1, P2
- [x] P1 and P2 marked in erasable pencil only after all 600 registered rollouts and the 30
      slow pace rollouts were complete
- [x] Marks erased afterward so the surface matches its prior state. No registered episode was
      recorded with the marks present, and the marks are not visible in the appendix figure.
- [x] Board photographed before marking, while marked, and after erasing:
      `media/before_marking.jpg`, `media/marked_board.jpg`, `media/overhead_marks_erased.jpg`.

**Part C total: 62 rollouts.**

---

## Part D: sampling density probe (August 22, PROTOCOL.md §8.30)

Declared in writing before any demonstration was recorded. A fifth collection condition and a
tenth policy, exploratory, single seed, never pooled into the four condition grid.

### Bench

Unchanged from Parts A to C. Camera framing verified against
`media/overhead_baseline_lighting.jpg` before the first episode. One USB device node had changed
since the August grid and the port strings in `scripts/record_dataset.sh` and
`scripts/run_inference.sh` were updated to match; no change to arms, cameras, lighting or work
surface.

### D1. Collection

`Andresg324/cube-pickup-density_20260822_194111`, 50 demonstrations at two positions, 25 at
T6 (15.5, 10.0) and 25 at T2 (6.5, 7.5), alternating by zero-based index, even at T2 and odd at
T6. Baseline lighting, red cube, every demonstration a first try clean success. 26,349 frames,
17.57 s per demonstration, inside the 16.6 to 21.4 s range the four registered collections span,
and 12.15 passes over its own data at 10,000 steps, inside the 9.9 to 12.9 span.

- [x] `scripts/record_dataset.sh` extended to accept `density`
- [x] 50 episodes recorded, position parity checked at episode 24
- [x] Uploaded to the Hub and episode count confirmed

**Deviation:** two aborted starts at 19:37 and 19:38 created empty dataset directories with no
episodes. Both were deleted locally and neither reached the Hub, so the retained collection
begins at episode index 0 as declared.

### D2. Training

`Andresg324/smolvla-cube-density`, seed 1000, settings identical to §4.7 apart from
`dataset.repo_id`, `output_dir` and `job_name`. Final training loss 0.0559 at step 9800, the highest of the ten runs, against Randomized's
0.0357 at the bottom. See `figures/table_loss.md`.

### D3. Evaluation

Three cells at 16 episodes each, index 0 discarded, **45 scored rollouts**. Cell order fixed by
§8.30 in advance and unchanged.

| Cell | Cube | Position | Result |
|---|---|---|---|
| in_distribution | red | T6 | [x] 15/15 |
| trained_t2 | red | T2 (6.5, 7.5) | [x] 15/15 |
| new_positions | red | E1 to E5, 3 episodes each, in order | [x] 0/15 |

All fifteen held-out failures scored `timeout_other`; none reached the cube. At four of the five
held-out positions the arm settled within 1.4 to 3.2 degrees of one of the two trained bearings
while 22 to 64 degrees from the target.

**Part D total: 45 rollouts. Grand total recorded and scored: 707.**

---

## After Parts B, C and D

- [x] Tracker exported to `documents/results_raw_two_seeds.xlsx` and to the derived CSVs, with
      columns `condition, eval_cell, seed, episode, instance, success, flagged, failure_mode, notes`.
- [x] Failure modes normalized against the fixed vocabulary (PROTOCOL.md §8.17, §8.21, §8.28).
      `tools/export_results.py` canonicalizes the two legacy labels and refuses to write
      anything if the vocabulary, the cell sizes or the success and label agreement fail.
- [x] Ambiguous episodes re-scored from retained video.
- [x] All 707 scored episodes screened by `tools/audit_labels.py` against two independent
      telemetry criteria; one transposed pair found and corrected (§8.27). The screens ran over
      662 episodes on August 14 and were extended to 707 when Part D closed.
- [x] All 94 seed 1000 `timeout_other` episodes audited, since that code was assigned
      retrospectively at seed 1000 and covers a larger share of failures there than at
      seed 2000. Every one ran to the recording ceiling with substantial joint motion,
      confirming the timeout label; 85 were re-scored from video and **seven were recoded as
      `contact_no_grasp`**. The derived CSVs were regenerated afterward.
- [x] Departure labels checked against telemetry in all 707 episodes, zero disagreements. The
      20 degree threshold separates the two label groups completely: the largest maximum joint
      deviation among `no_departure` episodes is 15.9 degrees and the smallest among departing
      episodes is 59.5 degrees.
- [x] Release detector re-calibrated against the recovery demonstrations as part of
      `tools/drops.py` on every run.
- [x] Every rollout dataset confirmed present on the Hub. The probing analysis replays these.
      49 rollout datasets in total: 4 conditions x 5 cells x 2 seeds, plus Clean at
      `near_1in` and `near_2in` at both seeds, plus the two slow pace cells, plus the three
      density cells.
- [x] No rollout dataset deleted.
- [x] Bench left standing and the gantry mounted.

Regenerate every derived CSV and every reported number from the workbook:

```bash
python tools/export_results.py                 # rebuild the derived CSVs; refuses to write on validation failure
python tools/audit_labels.py                   # label screens, recording-window measurement

# analyze_results refuses multi-seed input: PROTOCOL.md §4.7 does not allow pooling seeds
python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
python analysis/analyze_exploratory.py         # displacement, demonstration pace and sampling density probes
python analysis/seed_variance.py               # the same condition compared across seeds

# grasp poses, once per policy, then board coordinates onto the endpoint files
for p in clean-seed2000 color color-seed2000 randomized randomized-seed2000 \
         recovery recovery-seed2000; do
  python tools/endpoints.py --policy "$p" --cells in_distribution new_positions \
      reduced_lighting different_object distractors
done
python tools/endpoints.py --policy clean --cells in_distribution new_positions \
    reduced_lighting different_object distractors near_1in near_2in
python tools/endpoints.py --policy density --cells in_distribution trained_t2 new_positions
python tools/calibrate_pose.py --apply

python tools/rollout_motion.py                 # latency, velocity, no_departure validation
python tools/drops.py                          # detector calibration, release events, drop locations
python tools/azimuth_analysis.py               # calibration, envelope, aiming error, aim invariance, density bearings
python tools/motion_stats.py \
  cube-pickup-clean_20260809_105745 \
  cube-pickup-randomized_20260809_115825 \
  cube-pickup-recovery_20260809_141725 \
  cube-pickup-color_20260809_183224 \
  cube-pickup-color_20260809_130649 \
  cube-pickup-density_20260822_194111

# activations and probes; the slowest stage, and unchanged by the density probe
for p in clean clean-seed2000 color color-seed2000 \
         randomized randomized-seed2000 recovery recovery-seed2000; do
  python probing/extract_activations.py --policy "$p" \
    --layer model.vlm_with_expert.lm_expert.norm --device mps --seed 0
done
python probing/probe_position.py
python probing/probe_success.py --sweep

python tools/annotate_bench.py media/bench_wide.jpeg   # manual, only when the photo changes
python analysis/make_figures.py                        # last: reads everything above
```

---

## Part E: density sweep (September 8, PROTOCOL.md §8.32)

Registered before any demonstration was recorded. One pool of 520 demonstrations, subsampled
into four per-position budgets, two seeds each.

### Bench rebuild

The workbench was disassembled and rebuilt between the August grid and this collection, in a
different room with a higher ceiling. Three checks before the first demonstration, all logged
in PROTOCOL.md §8.32.

| Check | August | Rebuilt | Verdict |
|---|---|---|---|
| Geometry, mean shift from T1/T3/T8 | reference | 0.05 in x, 0.03 in y | below the 0.06 to 0.07 in clicking precision |
| T1-to-T3 pixel distance | reference | −0.1% | camera height and angle unchanged |
| Frame mean luminosity | 144.8 | 144.5 | −0.2%, against the 144.8 to 101.0 that defines Reduced Lighting |
| Patch luminosity | reference | +1.3% upper right, −3.7% mid left | redistribution consistent with a higher ceiling |
| Furthest commanded bearing at T6 | 27.04 deg | 26.70 deg | shift is one standard error (0.35 deg); episode sd 0.88 and 1.06 |
| Median bearing at gripper release | −23.00 deg | −22.47 deg | 0.53 deg, no standard error quoted |
| In-distribution success | 15/15 | 15/15 | 0 of 15 no-departures both |

Geometry read from three permanent surface marks rather than the cup or cube, which carry
placement and center-estimation error. An earlier attempt using cup center, cube center and
gripper base gave a T1-to-T3 equivalent of −0.9%, all of it estimation error.

- [x] `tools/bench_compare.py` written and run, frames archived to `media/bench_rebuild/`
- [x] Base clamp measured against the front edge by the August method, reads unchanged
- [x] `rollout_clean_in_distribution_20260908_110904` recorded and retained

An unresolved common-mode bearing offset of roughly half a degree remains, logged in
PROTOCOL.md §9. Both stability checks moved the same direction by about that amount, which
excludes a slope change in the pan-to-bearing fit but not a constant offset.

### E1. Collection

Command: `bash scripts/record_dataset.sh densitypool <cumulative total>`

52 passes of T1 through T10, cycled in fixed order, 520 demonstrations. Position is recoverable
from episode index by the §3 rule. Passes 10 and 30 held out in full and named in advance.

`--dataset.root` is mandatory: resume in LeRobot 0.5.2 refuses to write into the Hub snapshot
cache, so the root must be set on the first session as well as on resumes.

| Field | Value |
|---|---|
| Hub slug | `Andresg324/cube-pickup-densitypool_20260908_125700` |
| Demonstrations | 520 |
| Frames | 298,201 |
| Frames per demonstration | 573.5 |
| Seconds per demonstration | 19.1 |

573.5 frames per demonstration is 9.5% shorter than Randomized at 633.6 and shorter than every
August condition except the retained Color, consistent with a more practiced teleoperator. The
realized epoch count per sweep cell is computed from this rather than from the August figures.

- [x] `scripts/record_dataset.sh` extended to accept `densitypool` and to pass `--dataset.root`
- [x] Resume tested on a throwaway dataset before the first real session
- [x] 520 demonstrations recorded, all ten positions at exactly 52
- [x] `tools/verify_pool_mapping.py` run: pan spread 0.75 to 2.10 degrees, largest at T3, against
      August's 0.38 to 1.93 on 5 demonstrations per position. Mean bearing at every position
      matches its true bearing through the August fit to within 2.3 degrees.
- [x] Uploaded to the Hub, 2.35 GB, episode and frame counts confirmed

**Deviation: five documented interruptions**, at episodes 2, 35, 128, 207 and 369, from a lost
teleoperation link, an accidental disconnection, and operator use of the escape key. A small
number of further interruptions may have gone undocumented where recording resumed immediately;
but the total was below eight. Each resumed at the next index with the cube at the position the
cycle assigns. Verified two ways: the within-position pan spread above, which a one-step offset
would have blown up, and a frame-length check at every boundary. Episodes at the five boundaries
run 518 to 669 frames against a pool mean of 573.5, all inside the interquartile range, and the
ten shortest demonstrations in the pool (442 to 471) include none at a boundary, so nothing was
truncated.

**Deviation: position error at episodes 367 and 368.** Two demonstrations were recorded at
previously used positions rather than the T8 and T9 the cycle assigns, from a misread column in
the position list. Caught at episode 369. The session was stopped and episodes 367, 368 and 369
deleted with `delete_episodes` from `lerobot.datasets.dataset_tools`, which writes a new dataset
rather than mutating the source. Recording resumed at index 367 with the cube at T8. Confirmed
by overhead video at the recorded timestamps; episodes 2, 35, 128 and 418 were also checked and
match the cycle. 369 was a correct T10 but was deleted too for a clean restart at 367.

**Deviation: recording continued past 520.** The operator let the session run into episode 520,
the 521st, to confirm the 520th had been written, and left the cube in the cup for it. That
episode was deleted the same way.

**Deviation: 610 orphaned action frames at episode 129.** The pool as first written held 298,811
parquet rows against a `total_frames` of 298,201. The excess was 610 action rows carried under
episode index 129 with no video segment and no episode metadata, at the boundary of the
documented resume at episode 128. Both the orphaned block and the retained episode grasp at
T10, the position the cycle assigns, so the interrupted demonstration was re-recorded at the
same position and no demonstration was lost; position counts stayed at 52 everywhere.

Present in all four pre-deletion backups, so it originated at recording rather than in
`delete_episodes`. Found only because the episode-subset sampler raised `KeyError: 168053`
during the first training run: the sampler reads absolute row offsets from
`dataset_from_index` and `dataset_to_index`, so the orphaned rows pushed every later episode's
offsets out of alignment.

Repaired with `tools/repair_pool.py`: the 610 rows dropped, `index` renumbered contiguously,
and every episode's offsets rebuilt from the per-episode frame counts. Video files and
timestamps unchanged and untouched. Re-verified for frame counts, offset chaining and
index-to-position mapping, and a five-episode subset loaded successfully through
`LeRobotDataset(episodes=[...])` before the corrected pool was pushed.

### E2. Subsampling

`tools/make_subsets.py`, written and run before the first demonstration. 5 ⊂ 10 ⊂ 25 ⊂ 50 per
position, drawn by shuffling the 50 available passes at each position under
`numpy.random.default_rng(1000)` and taking the first k. Realized lists committed to
`analysis/subsets.json`.

| Budget | Demonstrations | Passes spanned | Mean pass index |
|---|---|---|---|
| 5/position | 50 | 2 to 51 | 24.8 |
| 10/position | 100 | 1 to 52 | 26.0 |
| 25/position | 250 | 1 to 52 | 26.3 |
| 50/position | 500 | 1 to 52 | 26.8 |

Mean pass index near the midpoint of 26.5 at every budget, so no budget is drawn
disproportionately from the least practiced passes.

- [x] Nesting verified: each budget is a subset of the next
- [x] No held-out pass leaked into any budget
- [x] Every position at exactly k demonstrations in every budget

### E3. Training

`tools/run_sweep.py`, sequential, both seeds of a budget before the next. Trained on RunPod
rather than Colab, single A100 per run. `--dataset.episodes` takes the subset index lists
directly, so no physical dataset copies were made.

Environment differs from the August grid and is recorded because it is not reproducible from
the resolved configs alone: LeRobot at commit `6a788fbd`, the same build as August, but torch
2.8.0 and torchcodec 0.7.0 against Colab's unrecorded versions. The container needed
`apt-get install ffmpeg`, a torchcodec downgrade to match torch 2.8, and a one-line patch to
`wandb_utils.py` line 122 replacing `wandb.run.get_url()` with `wandb.run.url`, which recent
wandb removed.

| Cell | Steps | Warmup | Decay | Final loss (s1000) | Final lr |
|---|---|---|---|---|---|
| 5/position | 10,000 | 333 | 10,000 | 0.0434 | 2.5318e-6 |
| 10/position | 20,000 | 667 | 20,000 | 0.0279 | 2.508e-6 |
| 25/position | 50,000 | 1,667 | 50,000 | 0.0470 | 2.5013e-6 |
| 50/position | 100,000 | 3,333 | 100,000 | | |

Every run anneals to `decay_lr` of 2.5e-6, matching August, and no auto-scaling line appeared,
so decay equalled steps in every cell as intended. The script asserts warmup, decay, steps,
episode count and seed against the resolved config before uploading each checkpoint, so a
silently dropped override would stop the chain rather than propagate.

Loss is not monotone in density. Each cell fits a different dataset, so the losses are scored
against different targets, which is the confound §8.31 exists to untangle.

- [x] Smoke test at 20 steps confirmed `--dataset.episodes` reaches the resolved config
- [x] 5/position, both seeds
- [x] 10/position, both seeds
- [x] 25/position, both seeds
- [ ] 50/position, both seeds
- [ ] Fixed-step controls at 250 and 500 demonstrations, one seed each

### E4. Loss matrix (PROTOCOL.md §8.31)

`tools/deterministic_loss.py` on a single A40, separate pod, concurrent with training.

- [x] `tools/norm_sweep.py`: frozen-backbone claim confirmed at the weights. `vlm.model` and
      `vlm.lm_head` bit-identical to base in all ten checkpoints, 99,880,240 of 450,046,176
      trainable at 22.2%, expert layers moved 1.071e-1 to 1.119e-1 of base norm across all ten.
- [x] Determinism verified: two identical calls on the Clean diagonal returned
      0.048411693423986435 both times, bit-identical through the dataloader.
- [x] Calibration: the Clean diagonal moved up to 9.3% between adjacent episode counts, so
      subsampling was rejected and every cell uses all 50 demonstrations.
- [x] Clean diagonal reproduces Table 7: reported 0.04755 against 0.0482, a 1.3% gap between
      two different estimators of the same quantity.
- [ ] Ten diagonal cells, the validity gate
- [ ] Full 60-cell matrix
- [ ] Normalization control on a subset of cells
- [ ] Validation loss on the 20 held-out demonstrations, per sweep checkpoint

### E5. Evaluation

Not yet run. Four cells per policy per §8.32: 15 at T6, 5 at each of E1 to E5, 15 with
distractors, and 2 each at T1, T3 and T10 as a descriptive spot check. One warmup discarded per
cell, so 65 recorded and 61 scored per policy, 520 recorded and 488 scored across the eight
primary runs.

---

## Media captured

- [x] `media/cube_starting_positions.png`: overhead frame, all 15 marks, annotated with IDs
- [x] `media/distractor_layout.png`: overhead frame with the four distractors placed
- [x] `media/distractors_overhead_live.jpg`: the distractor cell as recorded
- [x] `media/starting_position.jpg`: overhead frame of the arm home pose
- [x] `media/overhead_baseline_lighting.jpg` and `media/overhead_reduced_lighting.jpg`
- [x] `media/wrist_view.jpg`
- [x] `media/overhead_all_colors.jpg`: all six cubes on the gray primer
- [x] `media/before_marking.jpg`, `media/marked_board.jpg`,
      `media/overhead_marks_erased.jpg`: the displacement probe marks before, during and after
- [x] `media/loss_curve.png`: the pilot run on the old bench, not one of the ten study runs.
      The ten are in `documents/training_loss.csv` and in the training loss figure.
- [x] `media/overhead_demo.gif`: a successful autonomous rollout (Color policy, seed 1000,
      in-distribution, recorded on the study bench)
- [x] `media/bench_wide.jpeg` wide shot of the bench showing both LEDs and the gantry

### Failure mode clips (August 22, for the README)

Cut from the retained rollout video with `tools/make_gif.py`, overhead camera, real time unless
noted. Each is one scored episode.

- [x] `media/clip1_clean_success.gif`: clean, in_distribution, episode 9
- [x] `media/clip2_offset_2in.gif`: clean, near_2in, episode 5
- [x] `media/clip3_no_motion.gif`: randomized-seed2000, different_object, episode 6, trimmed to 15 s
- [x] `media/clip4_drop_no_recovery.gif`: recovery-seed2000, in_distribution, episode 14
- [x] `media/clip5_rare_regrasp.gif`: recovery-seed2000, in_distribution, episode 5
- [x] `media/clip6_correct_aim_no_grasp.gif`: randomized-seed2000, new_positions, episode 8
- [x] `media/clip7_density_selector.gif`: density, new_positions, episode 12 (E4)