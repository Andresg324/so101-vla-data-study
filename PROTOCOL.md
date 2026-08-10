# Pre-registered Protocol for SO-101 Experiment

*Amended August 8th, 2026 on the rebuilt workstation, prior to any data collection. All coordinate,
lighting, and object specifications below were fixed before the first training episode was
recorded. Changes from the original pre-registration are listed in §8.*

## 1. Apparatus

- **Arm:** SO-ARM101 Pro (Feetech STS3215 servos); follower 12 V, leader 5 V teleoperation.
- **Cameras:** Overhead Logitech C270 and wrist Seeed USB webcam, both 640 x 480 @ 30 fps. The
  overhead camera is rigidly mounted on a fixed gantry and captures a **22 x 17 in** area of the
  work surface. Camera position is locked for the duration of the study.
- **Work surface:** Plywood finished in flat gray primer.
- **Cube:** 1 inch foam cube (Teacher Created Resources). All colors are the same product and are
  identical in size, shape, mass and surface finish, only the color differs.
- **Cup:** Plastic cup, 4.5 in tall, 2.5 in diameter at the base, 3.5 in at the rim. Fixed at
  **(5.0, 12.5)** for all conditions and all evaluation cells.
- **Coordinate frame:** Positions are given in inches from the front-left corner of the
  overhead camera frame.
- **Arm base:** The follower is clamped to the front edge (y = 0), centered at **x = 11.0**.
- **Arm home pose:** Every episode begins with the arm fully retracted into its most compact
  configuration, base oriented forward toward the workspace, all joints folded back over
  themselves. The gripper is visible in the overhead frame in this pose.
- **Lighting (baseline):** blinds fully closed, room lights off. Two clamp LEDs
  (**1000 lm each, 4000 K, CRI 80**) bounced off the ceiling, clamped to both sides of the table.
  Measured at approximately **197 lux** at the work surface. This is the lighting for all training
  demonstrations and for four of the five evaluation cells.

The two views the policy receives, at the resolution it receives them:

<p align="center">
 <img src="media/overhead_baseline_lighting.jpg" width="380" alt="Overhead view">
 <img src="media/wrist_view.jpg" width="380" alt="Wrist view">
</p>

Arm home pose:

<p align="center">
 <img src="media/starting_position.jpg" width="380" alt="Arm home pose">
</p>

## 2. Cube positions (locked)

<p align="center">
 <img src="media/cube_starting_positions.png" width="500" alt="Cube Starting Positions">
</p>

**Training positions (10)** used by the Randomized condition, 5 demos each:

| ID | (X, Y) | | ID | (X, Y) |
|----|--------|---|----|--------|
| T1 | 2.0, 2.5 | | T6 | 15.5, 10.0 **Clean fixed position** |
| T2 | 6.5, 7.5 | | T7 | 15.5, 14.25 |
| T3 | 8.5, 15.0 | | T8 | 20.5, 2.5 |
| T4 | 12.0, 14.0 | | T9 | 20.5, 6.5 |
| T5 | 15.5, 2.5 | | T10 | 20.5, 10.0 |

**Held-out positions (5)** used only by the New Positions evaluation cell, 3 episodes each:

| ID | (X, Y) | Relation to training convex hull |
|----|--------|----------------------------------|
| E1 | 2.0, 7.5 | **outside** (extrapolation) |
| E2 | 6.5, 2.5 | on the boundary (front edge, between T1 and T5) |
| E3 | 12.0, 10.0 | inside (interpolation) |
| E4 | 15.5, 6.5 | inside (interpolation) |
| E5 | 19.5, 13.5 | **outside** (extrapolation) |

The convex hull of the training positions has vertices (2, 2.5), (20.5, 2.5), (20.5, 10),
(15.5, 14.25), (8.5, 15). Interpolation and extrapolation results are reported separately.

All 15 positions are marked identically on the work surface and are never redrawn.

## 3. Independent variables (conditions change)

Data-collection strategy, across four conditions. Each varies exactly one factor; all other
factors are held identical to the Clean condition.

1. **Clean:** All 50 demos have identical conditions: red cube at **T6 (15.5, 10)**, baseline
   lighting, and each demo is a first-try success.
2. **Randomized (position):** Only the cube's start position varies. The cube starts at the 10
   training positions listed in §2, which lie in an irregular region bounded by the cup at the
   back-left and the arm base at the front-center. Positions are cycled T1 through T10 in five
   complete passes, giving 5 demos per position and 50 episodes total. Cycling rather than
   blocking prevents drift in teleoperator skill over the session from being confounded with
   position. The Clean position (T6) is one of the 10.
3. **Recovery:** Conditions held identical to Clean, except 20 of the 50 demos include the robot
   dropping the cube, recovering it, and then releasing it in the cup. Recovery demonstrations use
   a deliberate gripper release rather than a naturally occurring grasp failure. The recovery behavior
   is therefore learned from a clean drop, whose dynamics may differ from an unintended slip during a 
   real failed grasp. These demos are distributed evenly across the session: within each consecutive
   group of 5 demos, demos 2 and 4 are recovery demos. The drop is performed at approximately the midpoint
   of the carry between the pick location and the cup, from a height of **4 to 5 inches** above the surface. 
   The robot then re-grasps the cube from wherever it lands and completes the task. Distributing the drops
   rather than blocking them at the end of the session prevents recovery behavior from being
   confounded with teleoperator fatigue.
4. **Color-varied:** Only cube color varies. Five colors are used for 10 episodes each: red,
   orange, yellow, blue, purple, for 50 episodes total. Colors are cycled in the order red,
   orange, yellow, blue, purple, repeated ten times, for the same reason positions are cycled in
   the Randomized condition. Green is *not* used in training, as it is reserved for evaluation,
   and the green cube is removed from the workspace entirely during all training collection.

<p align="center">
 <img src="media/overhead_all_colors.jpg" width="500" alt="All six cube colors on the gray work surface">
</p>

*The five training colors plus the held-out green, as seen by the overhead camera. All six are
clearly separable against the gray primer, which is why the background was changed from blue tape
(see §8.3).*

Because positions and colors are cycled deterministically, the factor level of every episode is
recoverable from its index without separate logging: in the Randomized condition episode *i* uses
position T[((i-1) mod 10) + 1], and in the Color-varied condition episode *i* uses the
((i-1) mod 5)+1'th color of the sequence. When a demonstration is discarded and re-recorded, it is
re-recorded at the same position or color, so this correspondence is preserved.

## 4. Fixed variables (confound control)

1. **Model:** SmolVLA, fine-tuned from lerobot/smolvla_base.
2. **Task:** Pick up a cube and place it in a cup.
3. **Demo budget:** 50 training episodes per condition; 15 evaluation episodes per cell.
4. **Hardware:** Camera position and framing, gripper, control frequency.
5. **Arm home pose** and **baseline lighting** as defined in §1.
6. **Reset procedure:** Between episodes the cube is replaced by hand onto its marked position.
7. **Training hyperparameters:** All four conditions are trained with identical settings:
   batch_size=32, steps=10000, save_freq=2000, seed=1000, LeRobot's default optimizer and
   learning-rate schedule for SmolVLA, policy.device=cuda, and the same rename_map
   (observation.images.overhead becomes camera1, observation.images.wrist becomes camera2).
8. **Checkpoint selection:** The final checkpoint at step 10000 is the one evaluated, for every
   condition. No best-loss or early-stopped checkpoint is used, so checkpoint selection cannot
   vary across conditions.
9. **Compute:** All training runs execute on a single A100.
10. **Recording window:** episode_time_s = 45 and reset_time_s = 15 for all four conditions and
    all five evaluation cells. During collection the episode window is a ceiling, not a target:
    recording is ended manually as soon as the cube is resting in the cup. The ceiling is set
    high enough to accommodate recovery demonstrations without truncation, and matches the
    45-second evaluation window defined in §6.7.
11. **Language instruction:** The instruction string is verbatim identical for every training
    demonstration and every evaluation rollout: "Pick up the cube and place it in the cup".
12. **Inference settings:** All evaluation rollouts run with identical settings:
    policy.device=mps, strategy.type=episodic, strategy.reset_to_initial_position=true,
    chunk_size=50, n_action_steps=50, dataset.fps=30, episode_time_s=45,
    reset_time_s=15. SmolVLA inference on this hardware takes approximately
    320 ms per forward pass, so the action-update rate is roughly 3 Hz while
    frames are recorded at approximately 25 fps. This applies identically to
    every policy and every cell.
13. **Warmup episodes:** Each evaluation cell records 16 episodes. Episode index 0
    is discarded as a warmup and is never scored; the 15 scored episodes are
    indices 1 through 15. The first policy forward pass in a process incurs a
    one-off Metal kernel compilation of roughly 15 seconds, which would
    otherwise systematically penalise the first episode of every cell. The
    discard is unconditional and independent of the episode's outcome.

Changing any of these mid-study invalidates the comparison. Held-out rule: evaluation instances
(positions, colors) differ from those seen in any training condition.

## 5. Evaluation

All four trained policies are evaluated across five cells: one in-distribution (reference) and
four held-out.

1. **In-distribution (reference):** Only the cup and red cube on the testbed, with baseline
   lighting and the cube at T6. Same across all 15 episodes.
2. **New Positions:** The cube starts at the 5 held-out positions, in order. 3 episodes each, 15
   total. Results are reported overall and split by interpolation (E2, E3, E4) versus
   extrapolation (E1, E5). E2 lies exactly on the front edge of the training convex hull and is
   grouped with interpolation. The held-out position identifier is recorded for every episode.
3. **Reduced Lighting:** The left-side LED lamp is switched off, and the remaining LED is
   unchanged in position, orientation and output. Blinds stay closed and room lights stay off, as
   in baseline. Scene illumination falls from approximately 197 lux to approximately 108 lux, and
   mean grayscale intensity of the overhead frame falls from **144.8 to 101.0**, a 30% reduction
   in the recorded image. The change is therefore both a reduction in illumination and a shift
   from symmetric to one-sided lighting, which alters the direction and depth of shadows across
   the workspace. The cube is at T6 for all 15 episodes, and the same fixture is switched off for
   all 15 episodes and for all four policies.

<p align="center">
 <img src="media/overhead_baseline_lighting.jpg" width="380" alt="Baseline lighting">
 <img src="media/overhead_reduced_lighting.jpg" width="380" alt="Reduced lighting">
</p>

*Baseline (both LEDs, left) and Reduced Lighting (one LED, right), as recorded by the overhead
camera.*

4. **Different object (new color):** The red cube is replaced with a green cube of the same 1 inch
   foam type, at T6, with baseline lighting and background for 15 episodes.
5. **Distractors:** Four distractor objects are placed at fixed marked positions, identical
   across all 15 episodes:

<p align="center">
 <img src="media/distractor_layout.png" width="420" alt="Distractor Layout Positions">
 <img src="media/distractors_overhead_live.jpg" width="360" alt="Distractor cell as recorded">
</p>

   | Object | Position |
   |--------|----------|
   | crumpled paper | T2 (6.5, 7.5) |
   | penny | T4 (12.0, 14.0) |
   | battery | E4 (15.5, 6.5) |
   | screw | T8 (20.5, 2.5) |

   The red cube starts at T6 with baseline lighting, only the distractors change. The objects
   span a range of size, shape, color and material (flat and copper, thin and metallic, etc.).
   Three of the four distractors occupy positions in the Randomized condition's training set
   (T2, T4, T8), making this cell adversarial specifically for the Randomized policy; for the
   other three conditions, which trained only at T6, they are simply unfamiliar objects in the
   scene. This asymmetry is intentional and is accounted for when interpreting the cell.

## 6. Pre-committed metrics

1. Task success rate per (condition x eval cell).
2. Per-cell success rate reported with Wilson 95% confidence intervals.
3. Each evaluation cell is 15 episodes.
4. Primary comparison is the **generalization gap:** In-distribution success minus mean held-out
   success.
5. **Matched-axis comparisons (key results):** Randomized vs Clean on New Positions, and
   Color-varied vs Clean on Different-object. Each is reported as a difference in success rate
   with a confidence interval on the difference and a Fisher exact test. Reduced Lighting and
   Distractors are cross-transfer cells that no condition trained on; for these we report whether
   any condition generalizes to an unseen axis.
6. **Success** occurs when the robot picks up the cube and releases it into the cup where it
   rests, with the cup remaining upright.
7. An **episode** is a 45-second window in which the arm may complete the task. If the success
   criterion (#6) is met, the episode is scored a success and terminated; otherwise, after 45
   seconds it is scored a failure and reset. If the cube misses the cup, the arm may retry within
   the allotted time. Knocking the cup over is an immediate failure (the policy is not trained to
   correct it). If the cube is knocked outside the testbed or out of the reachable-and-visible
   area such that the task can no longer be completed, the episode is scored a failure and reset.
8. **Scoring:** Episodes are scored live by the experimenter against criterion #6 at the time of
   the rollout. All rollout video is retained; any episode judged ambiguous at the time is flagged
   and re-scored from video before analysis. For New Positions episodes the held-out position
   identifier (E1 to E5) is recorded alongside the score.
9. **Seeds:** One training seed per condition (seed=1000) is run first and reported. Additional
   seeds (up to three per condition) will be added if time permits. The number of seeds actually
   run is reported for every condition. If only one seed is used, this will be stated as a
   limitation: per-cell confidence intervals capture episode-level uncertainty, not training-run
   variance. No analysis or claim is conditioned on how many seeds are obtained.

## 7. Dataset naming (locked)

cube-pickup-{clean,randomized,recovery,color}

Rollout datasets are named rollout_{policy}_{cell} and are retained for offline analysis.

## 8. Amendments to the original pre-registration

All made **before any training data was collected** on the rebuilt workstation.

1. **Workspace geometry.** The reachable, in-frame region is irregular, bounded by the cup at
   back-left and the arm base at front-center, so a rectangular grid was not achievable.
   "4 corners and 6 interior spots" is replaced by 10 explicitly listed training positions, and
   "4 edge-midpoints and one interior spot" by 5 explicitly listed held-out positions.
   Interpolation/extrapolation status is now reported, which the original did not specify.
2. **Color-varied condition** changed from red x14 and purple/yellow/blue x12 to five colors x10
   each, adding orange. This change was made to have a balanced design, and orange was available
   in the same cube set.
3. **Background** changed from blue masking tape to flat gray primer since a blue cube against a
   blue background gave poor contrast, which would have confounded the Color-varied condition.
4. **New Lighting** cell restructured into a single **Reduced Lighting** level (15 episodes). The
   original specified three levels (normal, dim, bright) reduced to one level because no light
   source brighter than baseline was available, and because "normal" is already measured by the
   in-distribution cell. The reduced level is implemented as one of the two LEDs switched off, not
   both: with blinds closed and room lights off, switching both off leaves the workspace at
   approximately 17 lux, effectively unlit, which would measure the policy's response to absent
   visual input rather than its robustness to a lighting change. The one-LED level was verified
   before collection to produce a 30% reduction in recorded image brightness (§5.3), confirming
   the manipulation survives the cameras' automatic exposure compensation.
5. **Scoring** changed from post-hoc video review to live scoring with video retained for
   ambiguous cases. Scoring live was deemed easier than re-reviewing the videos, and the videos
   will be saved to audit the scoring or score ambiguous cases.
6. **Seeds** wording clarified to commit to reporting the number run rather than to a number, and
   the seed value is now fixed at 1000 for every condition.
7. **Recovery condition** given an operational definition (drop point and height), which the
   original left unspecified.
8. **Demo ordering specified:** Randomized positions are cycled T1 through T10 in five passes, and
   Color-varied colors are cycled through the five-color sequence ten times. The original
   pre-registration left demo ordering unspecified; blocking demos by position or by color would
   have confounded the manipulated factor with drift in teleoperator skill over a single
   collection session.
9. **Recovery demo placement specified** as demos 2 and 4 within each consecutive group of 5. The
   original run-sheet placed all 20 recovery demos at the end of the session (demos 31 to 50),
   which would have confounded the recovery behavior with operator fatigue.
10. **Training hyperparameters, seed and checkpoint selection added to the fixed-variable list
    (§4):** The original pre-registration constrained the model, task and demo budget but left
    batch size, step count, random seed and which checkpoint to evaluate unspecified.
11. **Recording window and language instruction added to the fixed-variable list (§4):** The
    original did not specify the per-episode recording ceiling, which risked truncating recovery
    demonstrations, and did not fix the instruction string verbatim even though it is a model
    input.
12. **Matched-axis comparisons** now specify a confidence interval on the difference and a Fisher
    exact test. The original committed only to per-cell Wilson intervals; overlapping intervals
    are not a valid test of a difference between two rates.
13. A 16th warmup episode per cell was added and pre-committed to be discarded. This changes was
   made before any evaluation episode was recorded.

## 9. Existing Limitations

- Single training seed per condition unless time permits more.
- Extrapolation held-out positions lie a few inches beyond the training envelope, still within
  the arm's reach and camera frame; this is modest extrapolation, not a domain shift.
- The interpolation/extrapolation split rests on 9 and 6 episodes respectively and is reported as
  descriptive, not as a formal test.
- At 15 episodes per cell, the study is powered to detect large differences only. A shift from
  roughly 0.2 to roughly 0.7 is detectable; differences of 0.1 to 0.2 are not, and null results
  are reported as inconclusive rather than as evidence of no effect.
- Illumination was measured with a phone light meter at approximately 197 lux (both LEDs), 108 lux
  (one LED) and 17 lux (both off) at the work surface. These readings fluctuated and the
  instrument is not calibrated, so they are reported as approximate. Lighting is specified
  primarily by fixture type, rated output, position and on/off state, and secondarily by the
  measured change in recorded image brightness (§5.3), which is the quantity the policy actually
  sees.
- The cameras auto-expose. Halving scene illumination produced a 30% rather than a 50% reduction
  in recorded image brightness, so the Reduced Lighting cell tests a combination of reduced
  brightness, one-sided shadow structure, and exposure-related sensor noise rather than
  illumination alone.
- LED color rendering index is 80 (moderate). Lighting is constant throughout, but color fidelity
  is not studio-grade, which is relevant to the Color-varied condition.
- Distractors are placed at trained cube positions rather than randomly, making that cell
  adversarial by construction, and adversarial to a different degree for the Randomized condition
  than for the other three.
- Scoring is not blinded. Episodes are scored live by the experimenter, who knows which
  condition's policy is running. The success criterion is binary and physically unambiguous (cube
  released and resting in an upright cup within 45 seconds) which limits the room for
  interpretation, and all rollout video is retained so any episode can be re-scored or
  independently audited. Blinded scoring was not feasible for a single-operator study.
- The demo budget is fixed in episodes, not frames. Every condition contributes 50 episodes, but
  recovery demonstrations run longer than the others, so the Recovery dataset contains more frames
  than the rest. At a fixed 10,000 training steps this means Recovery is trained for fewer passes
  over its own data. Episode count was chosen as the fixed budget because it is the quantity an
  experimenter actually controls when deciding how much data to collect. Per-condition frame
  counts are reported alongside the results.
- At the three right-edge training positions (T8, T9, T10 at x = 20.5), the arm partially exits the 
  overhead frame during the reach, though the cube remains fully visible at episode start. Any position 
  effect at those spots is therefore partly a visibility effect.