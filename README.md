# SO-101 × SmolVLA: How Demonstration-Collection Strategy Shapes Generalization

A controlled empirical study on a self-built low-cost robot arm (Seeed SO-ARM101,
LeRobot platform) using the SmolVLA vision-language-action model.

<p align="center">
  <img src="media/overhead_demo.gif" width="520" alt="Autonomous cube pick-and-place">
</p>

## The question

A VLA learns to map camera images and a language instruction directly to robot motion,
end-to-end from demonstrations, with no hand-coded perception or inverse kinematics. This
project asks: how does the way you collect demonstrations affect how well the learned policy
generalizes to conditions it never saw?

## Demo

The clip above is the **overhead camera view** during autonomous inference. The leader is
disconnected and the policy is driving the follower arm on its own. It receives only the two
camera feeds (overhead and wrist) and the instruction *"Pick up the cube and place it in the
cup."* and outputs motion directly, with no hand-coded perception, planning, or teleoperation.

This is one in-distribution rollout from the Color-varied policy at seed 1000, played at 2x.
The cube is at the trained position T6 under baseline lighting, and the policy picks it up and
releases it into the cup inside the 45-second window. The four policies already separate here,
before anything is held out: across both seeds Clean succeeds 30 of 30 in distribution, Color 25
of 30, Recovery 17 of 30 and Randomized 12 of 30. The question the study asks is what happens
when any one thing changes, and the answer differs sharply by how the demonstrations were
collected.

## Design

See [PROTOCOL.md](PROTOCOL.md) for the full pre-registered protocol and
[RUN_SHEET.md](RUN_SHEET.md) for the as-run record.

Fix the model (SmolVLA), the task (Pick up the cube and place it in the cup), the demo budget
(50 episodes per condition), and the training hyperparameters. Vary only the data-collection
strategy, one factor at a time:

| Condition | What varies |
|---|---|
| **Clean** | nothing: red cube, one fixed position, every demo a first-try success |
| **Randomized** | cube start position, cycled over 10 marked positions |
| **Recovery** | 20 of 50 demos include a deliberate drop during the carry |
| **Color-varied** | cube color, cycled over five colors; green held out for evaluation |

<p align="center">
 <img src="figures/fig_positions.png" width="440" alt="Training and held-out cube positions">
</p>

*The ten training positions (T) and the five held-out evaluation positions (E), with the convex
hull of the training set drawn. E2, E3 and E4 fall inside it, E1 and E5 outside.*

All four policies are then evaluated on the same five cells: one in-distribution reference plus
four held-out (new positions, reduced lighting, unseen object color, distractors), at 15 scored
episodes each (16 recorded, index 0 discarded as warmup), for 300 rollouts per seed and 600
registered rollouts across two training seeds. A further 62 rollouts were recorded for two
exploratory probes (displacement and demonstration pace) and are reported separately, never
pooled into the grid. Held-out positions are split into interpolation and extrapolation relative
to the convex hull of the training positions and reported separately.

The protocol was pre-registered before any study data was collected. Amendments made on the
rebuilt workstation are listed, dated and justified in
[§8](PROTOCOL.md#8-amendments-to-the-original-pre-registration).

## Results

Every quantity below regenerates from the raw scores with the commands in
[SETUP.md](SETUP.md#analysis); the analysis map is in
[analysis/README.md](analysis/README.md).

Across 600 registered rollouts **both pre-registered comparisons are null, for opposite
reasons**. New Positions is 0/15 for every policy at both seeds, 0 of 120, so that
comparison sits on a floor; on the held-out color both compared policies sat at or near
ceiling. Success rate reported no effect anywhere, but the telemetry did.

**Six of the eight policies aim at the trained cube position even in the cell where the
cube is somewhere else.** Their median commanded bearing moves by 0.2 to 2.3 degrees
across all five evaluation cells. Only the position-randomized policy's aim moves with
the cube, by 9.1 and 22.0 degrees, and inside the training hull it localizes to within a
fifth of a cube width.

**The clean-data policy's reach is a fixed sweep.** It turns to about 27 degrees whatever is in
front of it. Reaching the cube requires 24.2 degrees at the trained position, 26.6 at a one-inch
displacement and 29.4 at two inches, so the sweep clears the first comfortably, grazes the second
and cannot reach the third. Episodes that get far enough to touch the cube: 15 of 15 in
distribution, then 6 of 8 and 4 of 8 at one inch across the two seeds, then 0 of 8 at two inches
at both. The policy is not failing to localize the cube, it is not looking for it.

<p align="center">
 <img src="figures/fig_aim_invariance.png" width="420" alt="Commanded bearing by evaluation cell">
 <img src="figures/fig_envelope.png" width="420" alt="Furthest bearing reached, clean policy">
</p>

*Left: median commanded bearing per policy across the five evaluation cells. Seven of the eight
lines are flat. Right: the clean policy's reach against the bearing each cube position requires.*

**Clutter destabilizes the position-randomized policy's aim.** Its commanded bearing under
distractors scatters to an interquartile range of 8.4 degrees at both seeds, against at most 1.6
for every other policy in every cell. The median moves too, though only decisively at seed 2000,
by 8.4 degrees against at most 1.7 elsewhere. The aim does not land on the distractors, so this
is destabilization rather than capture: the one policy trained to look around is the one clutter
unsettles.

**Position is in the representation either way.** A linear probe reads cube bearing from
every policy's final hidden state at 8.6 to 11.4 degrees against a 42-degree chance
floor, including the policies whose aim never moves. Holding out whole positions rather
than episodes collapses decoding to 26 to 42 degrees with intervals spanning chance, so
the code is interpolative. What the collection strategy changed is not whether position
enters the representation, but whether the readout to action uses it. Only the action
expert is fine-tuned, so the backbone supplying that representation is identical across
all eight policies.

<p align="center">
 <img src="figures/fig_probe.png" width="420" alt="Bearing decoded from hidden states">
 <img src="figures/fig_aiming_error.png" width="420" alt="Aiming error, interpolation vs extrapolation">
</p>

*Left: bearing decoded from each policy's final hidden state, by episode and with whole positions
held out. Right: the randomized policy's aiming error at held-out positions, split by whether the
position falls inside the training hull.*

**The outcome is unreadable until the gripper closes.** A probe swept across the episode does
not clear its own within-cell permutation null before the grasp in any policy that replicates
across seeds, while a full-episode control reaches 0.991 to 1.000. Randomized clears it at seed
1000 (p = 0.046) but not at seed 2000 (p = 0.176), so the one apparent early signal does not
replicate. Two policies are not testable: Clean and Color at seed 2000 have only 5 and 4
episodes in the minority outcome.

<p align="center">
 <img src="figures/fig_success_sweep.png" width="480" alt="Outcome decoding swept across the episode">
</p>

**Position diversity cost execution outright.** The randomized policy never left the home pose in
11 and 18 of 75 episodes across the two seeds, against 4 and 7 of 91 for Clean, and posted the
lowest grid totals in the study at 12/75 and 13/75. Fifty demonstrations spread across ten
positions left too few at each to specify an action confidently.

**Training loss does not order the policies the way evaluation does.** All nine runs
converge by roughly step 6,000. Randomized reaches the lowest final loss of any run and
scores worst in the study; Recovery reaches the highest and scores three times better.
The collapse is not a training failure.

<p align="center">
 <img src="figures/fig_loss.png" width="480" alt="Training loss for all nine fine-tuning runs">
</p>

**Recovery inherits the release point exactly.** Its dropped cubes land at a median
bearing of -11.7 degrees against the demonstrated -10.8, one degree apart against a
calibration accurate to 0.86 degrees, both about 65% of the way from cube to cup. It
inherits the first half of the demonstrated behavior and not the second: 6 of 53
releases were followed by the re-grasp that completes every demonstration.

<p align="center">
 <img src="figures/fig_release.png" width="480" alt="Release bearing, demonstrations against rollouts">
</p>

**The demonstrator's tempo transfers.** Completion time on successes is perfectly rank-ordered
with demonstration velocity across all five datasets (Spearman rho = -1.0, n = 5, p = 0.017,
the floor at this n), and the four datasets with policies at both seeds reproduce the ordering.
The slow-pace control policy finishes in 22.4 s against the retained Color policy's 13.4 and
13.5 s.

Tables in `figures/table1.md`, `figures/table2.md` and `figures/table_loss.md`; every figure
above is regenerated by `analysis/make_figures.py` in both PNG and PDF. Raw scores in
`documents/results_raw_two_seeds.xlsx` (662 scored episodes, one row each).

## Hardware

- Seeed SO-ARM101 Pro (leader 5 V / follower 12 V), Feetech STS3215 servos
- Two USB cameras: Logitech C270 overhead on a fixed gantry, Seeed webcam at the wrist, both
  operated at **640 × 480 @ 30 fps**
- Work surface in flat gray primer; two 1000 lm / 4000 K clamp LEDs bounced off the ceiling
- MacBook Air for collection and inference; Colab A100 for training

<p align="center">
 <img src="media/bench_wide.jpeg" width="600" alt="The study bench">
</p>

*The bench: leader and follower arms, the overhead gantry, both clamp LEDs, the cup and the
marked work surface. Camera position is locked for the duration of the study.*

<p align="center">
 <img src="media/overhead_baseline_lighting.jpg" width="380" alt="Overhead camera view">
 <img src="media/wrist_view.jpg" width="380" alt="Wrist camera view">
</p>

*The only two visual inputs the policy receives, at the resolution it receives them: overhead (left) and
wrist (right), both 640 x 480.*

## Pipeline

1. `scripts/record_dataset.sh`: teleoperate and record synchronized camera and joint data
2. Train SmolVLA on a cloud GPU (fine-tune from `lerobot/smolvla_base`), log to Weights & Biases
3. `scripts/run_inference.sh`: trained policy drives the arm autonomously
4. `tools/export_results.py` then `analysis/`: regenerate every reported number from the scores

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | collection, inference and camera bring-up shell scripts |
| `tools/` | camera checks, episode playback, and the motion and geometry analyses |
| `analysis/` | pre-committed statistics, exploratory analyses, figure generation |
| `probing/` | activation extraction and linear probes (extracted activations are gitignored) |
| `configs/` | the resolved training configuration for each of the nine runs, as written by LeRobot |
| `documents/` | raw scores and the derived CSVs |
| `figures/` | generated tables and figures (`make_figures.py`) |
| `media/` | photographs and camera frames referenced by the docs |

`analysis/README.md` maps each script to the numbers it produces. Two tools worth knowing about
outside the pipeline: `tools/check_cameras.py`, a headless-safe camera probe that saves a frame
from each camera so framing can be verified before recording, written because LeRobot's OpenCV
build cannot open a live preview window; and `tools/show_episode.py`, which plays a single
episode out of LeRobot v3's chunked video files.

## Status

- [x] Hardware assembled and calibrated
- [x] Teleoperation verified (leader → follower mirroring)
- [x] Pilot dataset recorded, SmolVLA fine-tuned, first autonomous pick
- [x] Workstation rebuilt; protocol pre-registered and amended before collection
- [x] Four-condition data collection (200 demonstrations retained; Color re-collected, see
      [PROTOCOL.md §8.18](PROTOCOL.md#8-amendments-to-the-original-pre-registration))
- [x] Eight training runs (four conditions × two seeds), plus one exploratory policy
- [x] Evaluation grid complete (600 registered rollouts across two seeds) and analyzed
- [x] Label audit across all 662 scored episodes, three telemetry screens

## Pilot results (superseded by the study)

> **These results come from the previous workstation and are not part of the study.** The bench
> was rebuilt in August 2026 with different camera geometry and a gray work surface in place of
> blue tape, so the pilot data is no longer distribution-matched and is **not** mixed with study
> data. It is kept here because it established that the pipeline works end to end.

Closed the full pipeline end-to-end: teleoperated data collection, SmolVLA fine-tuning
(10k steps, single A100), autonomous inference on the real arm. The policy reliably picks and
places when the cube is at the trained position (4 consecutive successes).

When the cube was moved off the trained position, the policy reached and missed repeatedly. This
was a direct, observed instance of the generalization gap that the four-condition study is
designed to measure. The clean-data policy nails the in-distribution pose and degrades off it.

<p align="center">
 <img src="media/loss_curve.png" width="500" alt="Pilot training loss">
</p>

Training loss (`train/losses_after_rm_padding`) fell from ~0.19 to ~0.045 over 10k steps,
plateauing around step 6k; the policy converged well within the run. **This curve is the pilot
run only.** The nine study runs are exported to `documents/training_loss.csv` and plotted
separately by `analysis/make_figures.py`.

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
4. **Two training seeds per condition** (1000 and 2000), reported separately and never pooled.
   Two seeds bound training-run variance loosely; per-cell confidence intervals capture
   episode-level uncertainty, not training-run variance.
5. **Only the action expert is fine-tuned.** The vision encoder and language backbone are frozen
   by the base model's defaults, so all eight policies share an identical perceptual front end and
   the study measures what collection strategy does to the action readout, not to perception. See
   [PROTOCOL.md §9](PROTOCOL.md#9-known-limitations) for the full list.

## Reproducing

Requires the LeRobot environment (Python 3.12). See [SETUP.md](SETUP.md).