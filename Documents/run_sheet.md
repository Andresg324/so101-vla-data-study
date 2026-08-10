# Collection and Evaluation Run Sheet

Matches PROTOCOL.md as amended August 8, 2026. If this sheet and the protocol ever disagree,
the protocol wins and this sheet is wrong.

Out of Date, to be updated with collection results

---

## Every session, before touching data

- [ ] Power: leader 5V, follower 12V. Both clamped.
- [ ] Continuity Camera off on the iPhone.
- [ ] Wave test via `bash scripts/check_cameras_live.sh`. Confirm which index is overhead and
      which is wrist.
- [ ] Set `OVERHEAD_IDX` and `WRIST_IDX` in `record_dataset.sh`, `run_inference.sh`,
      `check_cameras_live.sh` and `check_cameras.py`. All four must agree.
- [ ] Leader parked in frame, matching the training view.
- [ ] Baseline lighting: blinds closed, room lights off, both clamp LEDs on.
- [ ] Arm returns to home pose: fully retracted, base forward, joints folded, gripper visible
      in the overhead frame.
- [ ] `hf auth whoami` returns correct username. Write token active.
- [ ] Laptop plugged in, sleep disabled, disk space checked.

---

## Part A: collect four datasets

Command: `bash scripts/record_dataset.sh <condition> 50`

The episode window is a ceiling of 40 seconds, not a target. End each demo with the right arrow
as soon as the cube is resting in the cup.

Every demo is a first try clean success. Anything less, redo it with the left arrow.
**When you redo a demo, redo it at the same position or color**, so that episode index still
maps to factor level.

| # | Condition | Cube | Position | Instruction | Done |
|---|---|---|---|---|---|
| 1 | clean | red | T6 (15.5, 10.0) every demo | nothing varies | [ ] |
| 2 | randomized | red | cycle T1 to T10, five passes | demo 1 at T1, demo 2 at T2 ... demo 11 back at T1 | [ ] |
| 3 | recovery | red | T6 every demo | within each group of 5 demos, demos 2 and 4 are drop and recover | [ ] |
| 4 | color | cycle red, orange, yellow, blue, purple | T6 every demo | ten full passes through the color sequence, no green | [ ] |

**Randomized cycle order:** T1 (2.0, 2.5), T2 (6.5, 7.5), T3 (8.5, 15.0), T4 (12.0, 14.0),
T5 (15.5, 2.5), T6 (15.5, 10.0), T7 (15.5, 14.25), T8 (20.5, 2.5), T9 (20.5, 6.5),
T10 (20.5, 10.0). Repeat five times.

**Recovery drop pattern:** demos 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29, 32, 34, 37, 39,
42, 44, 47, 49. Twenty total. Drop at roughly the midpoint of the carry, from 4 to 5 inches
above the surface, then re-grasp from wherever it lands and complete the task.

**Color cycle order:** red, orange, yellow, blue, purple. Repeat ten times.

### After each condition finishes

- [ ] Let the Hugging Face upload finish completely. Do not start the next recording during it.
- [ ] Open the dataset on the Hub and confirm the episode count is 50.
- [ ] Write the real timestamped slug here, exactly as it appears:

```
clean       ______________________________________________
randomized  ______________________________________________
recovery    ______________________________________________
color       ______________________________________________
```

- [ ] Note the total frame count for the condition (needed for the frames vs episodes
      limitation): ______________
- [ ] Launch the Colab training run for this condition, then start the next collection.

---

## Between A and B: train four policies

Identical settings for all four, fixed by PROTOCOL.md section 4. Do not tune per condition.

`batch_size=32`, `steps=10000`, `save_freq=2000`, `policy.path=lerobot/smolvla_base`,
`policy.push_to_hub=false`, `policy.device=cuda`, single A100, LeRobot default optimizer and
learning rate schedule. The evaluated checkpoint is the final one at step 10000.

Models uploaded to `[USER]/smolvla-cube-<condition>`:

- [ ] clean
- [ ] randomized
- [ ] recovery
- [ ] color

Before committing 75 rollouts to any policy, run 3 sanity rollouts. If the arm flails or scores
0 for 3, that is a training problem. Retrain overnight rather than burning eval time.

---

## Part B: evaluate four policies on five cells

Command: `bash scripts/run_inference.sh <policy> <cell>`

15 episodes per cell, 45 second window, scored live. Batch by cell across every policy that is
available, so each scene is configured once.

| Cell | Cube | Position | Lighting | Distractors |
|---|---|---|---|---|
| in_distribution | red | T6 | baseline, both LEDs | none |
| new_positions | red | E1 to E5, 3 episodes each, in order | baseline, both LEDs | none |
| reduced_lighting | red | T6 | left LED off, right LED on | none |
| different_object | green | T6 | baseline, both LEDs | none |
| distractors | red | T6 | baseline, both LEDs | four objects, see below |

**Held-out positions:** E1 (2.0, 7.5), E2 (6.5, 2.5), E3 (12.0, 10.0), E4 (15.5, 6.5),
E5 (19.5, 13.5). Three episodes at each.

**Distractor placement, identical for all 15 episodes:** crumpled paper at T2 (6.5, 7.5),
penny at T4 (12.0, 14.0), battery at E4 (15.5, 6.5), screw at T8 (20.5, 2.5).

### Progress grid

| Policy | in_distribution | new_positions | reduced_lighting | different_object | distractors |
|---|---|---|---|---|---|
| clean | [ ] | [ ] | [ ] | [ ] | [ ] |
| randomized | [ ] | [ ] | [ ] | [ ] | [ ] |
| recovery | [ ] | [ ] | [ ] | [ ] | [ ] |
| color | [ ] | [ ] | [ ] | [ ] | [ ] |

### Success criterion

Success is the cube released and resting in the cup, cup upright, within 45 seconds.

- If the cube misses the cup, the arm may retry inside the window.
- Knocking the cup over is an immediate failure.
- Cube knocked out of the reachable and visible area is a failure.
- Score 1 or 0 live, into the tracker, at the time of the rollout.
- Flag anything ambiguous and re-score it from video before analysis.

---

## After Part B

- [ ] Export the tracker to `results.csv` with columns
      `condition, eval_cell, seed, episode, success`.
- [ ] Run `python analysis/analyze_results.py results.csv --outdir analysis/out`.
- [ ] Confirm every rollout dataset is on the Hub. Phase B replays these.
- [ ] Do not delete any rollout dataset.
- [ ] Leave the bench standing and the gantry mounted.

---

## Media to capture, once, in baseline state

- [ ] `media/cube_starting_positions.png`: overhead frame, all 15 marks, annotated with IDs
- [ ] `media/distractor_layout.png`: overhead frame with the four distractors placed
- [ ] Overhead frame of the arm home pose
- [ ] Overhead frame under reduced lighting, to pair with the baseline frame
- [ ] All six cubes on the gray primer, five training colors plus green
- [ ] Wide shot of the bench showing both LEDs and the gantry
- [ ] `media/overhead_demo.gif`: a successful autonomous rollout

---

## The rule that overrides the sheet

If demo quality is slipping because you are tired and pushing to finish a condition, stop.
A rushed dataset costs a September redo. A slipped day costs a day.



# Info from runs (to write in the write spot after)

clean: 
Andresg324/cube-pickup-clean_20260809_105745    
50 episodes
27,947 frames

randomized:
Andresg324/cube-pickup-randomized_20260809_115825 
50 episodes
31,682 frames