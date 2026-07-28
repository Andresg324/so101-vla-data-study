# Pre-registered Protocol for SO-101 Experiment

## Independent variables (conditions change)
Data-collection strategy, across four conditions. Each varies exactly one factor; all
other factors are held identical to the Clean condition.

1. **Clean:** All 50 demos have identical conditions: red cube, constant lighting (normal room lighting),
   fixed start position, and each demo is a first-try success.
2. **Randomized (position):** Only the cube's start position varies (lighting, color, etc.
   held as Clean). The cube starts across 10 locations (the 4 corners and 6 interior spots, including 
   the Clean start position), with 5 demos per location for a total of 50 training episodes.
3. **Recovery:** Conditions held identical to Clean, except 20 of the 50 demos include the
   robot dropping the cube, recovering it, and then releasing it in the cup.
4. **Color-varied:** Only cube color varies (lighting, position held as Clean). A red cube will be used
   for 14 episodes; purple, yellow, and blue will be used for 12 episodes each, for a total of 50 episodes.
   Green will *not* be used in training, as it will be used for the evaluation episodes.

## Fixed variables (confound control)
1. **Model:** SmolVLA
2. **Task:** Pick up a cube and place it in a cup
3. **Demo budget:** 50 training episodes per condition; 15 evaluation episodes per evaluation cell
4. **Hardware:** Camera positions, gripper, control frequency

Changing any of these mid-study invalidates the comparison. Held-out rule: evaluation instances
(positions, colors) are chosen to differ from those seen in any training condition.

## Evaluation
Evaluate all four trained policies across five cells: one in-distribution (reference) and four
held-out (new positions, new lighting, different-colored object, distractors).

## Pre-committed metrics
1. Task success rate per (condition × eval cell).
2. Report per-cell success rate with Wilson 95% confidence intervals. Seeds are run
   incrementally, one training seed per condition first (one evaluation day), adding seeds (up to 3)
   if time permits. The number of seeds used is reported; a single seed is a stated limitation
   (per-cell CIs capture episode-level uncertainty, not training-run variance).
3. Each evaluation cell is 15 episodes.
4. Primary comparison will be defined as the generalization gap, measured by the in-distribution 
   success minus held-out success.
5. Matched-axis comparisons (key results): Randomized vs Clean on New Positions, and
   Color-varied vs Clean on Different-object. New Lighting and Distractors are cross-transfer
   cells that no condition trained on; we report whether any condition generalizes to these
   unseen axes.
6. Success occurs when the robot picks up the cube and releases it into the cup where it rests, 
   with the cup remaining upright.
7. An "episode" is a 45-second window in which the arm may complete the task. If the success
   criterion (#6) is met, the episode is scored a success and terminated; otherwise, after 45s
   it is scored a failure and reset. If the cube misses the cup, the arm may retry within the
   allotted time window. Knocking the cup over is an immediate failure (the policy is not trained 
   to correct it). If the cube is knocked outside the testbed or out of the reachable-and-visible 
   area such that the task can no longer be completed, the episode is scored a failure and reset.
8. Episodes are scored by the experimenter from the recorded video against criterion #6.

## Evaluation cells defined
1. **In-distribution (reference):** Only the cup and red cube on the testbed; normal lighting;
   cube at the fixed trained position; same across all 15 episodes.
2. **New Positions:** The cube starts at 5 positions distinct from the Randomized condition's
   training positions. The cube will be at the 4 edge-midpoints and one distinct interior spot,
   and there will be 3 episodes per location, for a total of 15 evaluation episodes.
3. **New Lighting:** Room lighting varies across normal lighting, dim and bright. Each will be used 
   for 5 episodes. Bright lighting will use extra room lights or an iPhone flashlight; dim lighting 
   will use room lights lowered.
4. **Different object (new color):** The red cube is replaced with a green cube (used in no
   training condition), at the fixed trained position with in-distribution lighting/background,
   for 15 episodes. Tests generalization to an unseen object appearance.
5. **Distractors:** Small items (marbles, crumpled paper, stress balls) are added to the testbed;
   the red cube starts at the fixed trained position with in-distribution lighting. Only the
   distractors change. 15 episodes.

## Dataset naming (locked)
cube-pickup-{clean,randomized,recovery,color}