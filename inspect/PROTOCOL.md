# PROTOCOL: Porting trajectory measures into Inspect Robots

Companion to the main study's PROTOCOL.md, registered there as §8.39. Measure definitions
are taken from this repository's code, cited by file, and are not redefined here.
Frozen October 20, 2026, before any comparison is run.

## 0. Context

In the main study, success rate reported no difference between collection strategies,
while telemetry from the commanded joint positions showed whether each policy aimed at
the cube, whether it left its home pose, and where it failed. Inspect Robots scores real
hardware episodes with an operator's success call, recorded as `termination_reason` and
read by `success_at_end`. It has no trajectory-level scorer.

## 1. Question

Can the two trajectory measures from the density sweep, grasp bearing error and departure,
be implemented as Inspect Robots scorers that reproduce what my analysis scripts compute?
Run on new episodes, do they recover the sweep's E5 result, where success is zero while
aim is within a few degrees of the cube?

## 2. Scope

In scope: two scorers, validated on archival rollouts and then on new episodes from one
existing policy at two existing positions, plus a calibration spot check at three trained
positions.

Out of scope: additional training, new policies, new positions, new embodiments, and any
new claim about policy behavior. The replication in §7 is exploratory and does not update
the paper's results.

## 3. Setup

- inspect-robots 0.60.0, inspect-robots-so101 0.4.0, Python 3.12
- LeRobot 0.5.2, editable install at commit 6a788fbd (cloned June 26, 2026, never
  pulled), the same code used for every rollout in the main study
- Units: degrees for the five arm joints, 0 to 100 for the gripper. LeRobot's SO follower
  defaults to `use_degrees=True` at this commit, the main study's recording script did
  not override it, and `use_degrees=True` is set explicitly on both `SOArmConfig` and
  `LeRobotPolicyConfig`. The plugin passes policy outputs to the motors unconverted and
  its compatibility check does not compare units, so this is set by hand and checked on
  the warm-up episode (§5).
- Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  (plugin `packing.py`), the same order as the `action` column in the archival datasets.
- Motor calibration: `robot_id="my_follower_arm"`. The cached LeRobot calibration file is
  identical to `inspect/my_follower_arm.json`; the arm has not been recalibrated since
  the sweep.
- Parity with the main study's rollouts: control at 30 Hz; `max_steps` 1,151, the longest
  archival episode (the 45 s window divided by 1.955 s per chunk, which is 50 actions at
  30 Hz plus the 288 ms forward pass, gives 23 chunks or 1,150 actions, and 1,151 recorded
  frames); chunks of 50 executed open loop (`chunk_size=50`, controller
  `replan_interval=None`); both cameras at 640x480.
- Cameras: `camera1` is the overhead C270 and `camera2` is the wrist camera, matching the
  training `rename_map` and `scripts/run_inference.sh`, set on both `SOArmConfig` and
  `LeRobotPolicyConfig`. Indices confirmed with `tools/check_cameras.py` before the run.
- Home pose, sent before every episode: pan 3.69, lift −97.62, elbow 96.75, wrist_flex
  67.86, wrist_roll −1.10, gripper 3.51, the median first-frame state across the 66
  density50 seed 1000 archival episodes.
- `max_relative_target`: the plugin requires it with `home_pose`, and LeRobot applies it
  to every command, not only homing. The main study set no limit. It is set before the
  warm-up to 1.25 times the largest absolute gap between commanded action and measured
  state, on any joint including the gripper, across every frame of the density50 seed
  1000 archival rollouts, rounded up. At that value it limits the homing move without
  clipping any policy action seen in the sweep. The measured gap and the value are
  logged in §8 before the warm-up episode.
- Policy: `Andresg324/smolvla-cube-density50`, seed 1000, revision
  a7889110fe89e93c6546289fc4eab3aecfb708cf
- Pan to azimuth calibration: `tools/calibrate_pose.azimuth_fit()`,
  azimuth = 0.9485 × pan + 3.669, leave-one-out median absolute error 0.86 deg.
  Specific to this bench.
- Positions as in `tools/azimuth_analysis.py`: T6 (15.5, 10.0), E5 (19.5, 13.5).

## 4. Measurements

Both measures are computed from the logged policy actions. Inspect Robots writes each
step's policy action before the embodiment's safety clamp (`eval._write_action_log`),
and the main study set no `max_relative_target`, so its recorded `action` column equals
the policy action whether the recorder stores the policy output or the sent command.
The two logs hold the same quantity. The first episode of every run is a warm-up and is
excluded.

**Grasp bearing error (primary).** As in `tools/grasp.py`, `tools/endpoints.py` and
`tools/azimuth_analysis.py` (§8.32). The gripper command (index 5) counts as open above
15. Open runs separated by 3 frames or fewer are merged, and only runs of at least 5
frames count. The grasp frame is the last frame of the first sustained open run. There
is no grasp if the gripper never opens or is still open on the last frame. Released
means a second sustained open run follows. The commanded pan at the grasp frame is
converted to azimuth with the calibration above, and the cube's azimuth
(`calibrate_pose.azimuth(x, y)`, with x and y stored in the scene parameters) is
subtracted. Output per episode: grasp frame, azimuth, signed error in degrees, released.
Episodes with no grasp are reported as coverage (grasped / scored), as in the sweep.

**Settled bearing (secondary, E5 only).** As in §8.32 and `settled_new_positions`: the
median commanded azimuth over the last quarter of the episode, computed for every
unsuccessful E5 episode and reported in its own column beside grasp bearing. The two
are never pooled. Not computed at T6, where the window is dominated by delivery to the
cup (§8.34).

**Departure.** As in `tools/rollout_motion.py`: an episode departs if any of the five arm
joints deviates more than 20 degrees from its first-frame value. Output per episode:
departed, maximum deviation, and latency to the first departing step, counted in steps
and divided by 30, the archival convention, so it measures execution time and excludes
inference pauses.

## 5. Validation

**Stage A, archival data, no hardware.** Each measure is written as a function of an
action array (plus cube position for aim), separate from the Inspect Robots wrapper,
and run on:
- density50 seed 1000: in_distribution, spot_check and new_positions, compared with
  `analysis/out_azimuth/density_aiming.csv`, `sweep_trained_aim.csv` and
  `settled_new_positions.csv`
- randomized and randomized-seed2000, all cells, compared with
  `analysis/out_motion/rollout_motion_episodes.csv`, which includes the labeled
  no-departure episodes, so the positive case is tested
Pass: the same grasp frame, departure flag and latency for every episode, and the same
bearing values after the same rounding the stored CSVs apply.

**Warm-up check, before any scored episode.** On the warm-up episode of the first run
(the spot check), two things are checked at the prompt for the next episode. First, the
arm has returned to the home pose. Second, the first logged action in
`actions/<run>/<trial>.jsonl` is within 10 on every joint of the median first-frame
commanded action across the density50 seed 1000 archival episodes, logged in §8 with the
`max_relative_target` value. That tolerance covers archival start-to-start variation and
is far narrower than a unit or joint-order error would produce. If either check fails,
the run stops and an amendment is logged.

**Stage B, Inspect Robots.** On every new episode, scorer outputs are compared with the
same measures recomputed independently from the raw action log
(`actions/<run>/<trial>.jsonl`). This tests the wrapper: field names, joint order,
episode boundaries and warm-up exclusion.
Pass: identical outputs for every episode.

## 6. Run plan

**Policy:** density50, seed 1000.

**Order:** calibration spot check, then T6, then E5. All video retained.

**Calibration spot check:** 3 scored episodes each at T1, T3 and T10, plus one warm-up at
T1. A passing score would be the median grasp bearing error within 3.5 deg at each
position, the bound the sweep met at every trained position. If any position fails, T6
and E5 are not run until the calibration is redone and an amendment is logged.

**Cells:** T6 (in_distribution) and E5, 15 scored episodes each plus one warm-up. Prior
counts for this policy were 15 at T6 and 5 at E5. In total, 42 episodes recorded and 39
scored.

**Success label:** binary, same criterion as the main study (§6.6): the cube is released
into the cup and rests there, cup upright. The operator presses Enter to end each
episode and answers y/N, which the plugin records as `termination_reason`. An episode
that is not ended by the operator runs to `max_steps` and scores as a failure. Failure
modes use the main study's vocabulary. The trajectory measures are reported beside
success, not merged into it.

## 7. Predictions

Committed before Stage A is run.

1. Stage A: agreement on every episode as defined in §5.
2. Stage B: identical outputs between the scorers and the recomputation.
3. Replication, exploratory. At E5, 0 of 15 successes, with median grasp bearing error
   within 3 deg of the cube and failures recorded as contact without a grasp, the arm
   aimed at the cube but not extending far enough. At T6, at least 12 of 15 successes
   and median grasp bearing error within 3.5 deg. Every episode at both cells departs.

## 8. Amendments

Dated entries, same format as the main protocol. Any change to a definition, threshold,
cell or count after the freeze is logged here before the data it affects is collected.

- *October 19, 2026, setup checks before freeze.* LeRobot is at commit 6a788fbd, the
  commit used for every main-study rollout. The SO follower defaults to degrees at that
  commit and the recording script did not override it, so the archival actions are in
  degrees. The cached motor calibration matches `inspect/my_follower_arm.json`.