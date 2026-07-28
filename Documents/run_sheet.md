# Collection & Evaluation Run-Sheet

## Every session (before touching data)
- [ ] Power: leader 5V, follower 12V. Clamp checked.
- [ ] Cameras: run rerun wave-test. Confirm camera1 = overhead, camera2 = wrist. Fix _IDX if shuffled.
- [ ] Leader parked in frame (matches training view).
- [ ] hf auth login active (write token).

## Part A — Collect 4 datasets (50 demos each, push_to_hub=true)
Command each time:  bash scripts/record_dataset.sh <condition> 50

| Condition   | Cube color(s)                  | Position(s)                              | Special instruction                                  |
|-------------|--------------------------------|------------------------------------------|------------------------------------------------------|
| clean       | red                            | fixed start spot                         | every demo a clean first-try success                 |
| randomized  | red                            | 10 spots (4 corners + 6 interior)        | 5 demos per spot; else identical to clean            |
| recovery    | red                            | fixed start spot                         | demos 31–50: deliberately drop, recover, then place  |
| color       | red×14, purple/yellow/blue×12  | fixed start spot                         | NO green; cycle colors; else identical to clean      |

Record the actual pushed dataset name (with timestamp) for each: ______________________

## Part B — Evaluate each of the 4 policies on all 5 cells (15 episodes each)
Reset the scene per the cell, run the policy autonomously, score 1/0 from video (criterion below).

| Cell               | Cube      | Position(s)                          | Lighting            | Distractors |
|--------------------|-----------|--------------------------------------|---------------------|-------------|
| in_distribution    | red       | fixed trained spot                   | normal              | none        |
| new_positions      | red       | 4 edge-midpoints + 1 new interior (3 ea) | normal          | none        |
| new_lighting       | red       | fixed trained spot                   | 5 normal/5 dim/5 bright | none    |
| different_object   | GREEN     | fixed trained spot                   | normal              | none        |
| distractors        | red       | fixed trained spot                   | normal              | marbles/paper/balls |

## Success criterion (score each episode)
Cube picked up and released into the cup, resting there, cup upright, within 45s.
Cup knocked over or cube leaves reachable/visible area = failure.