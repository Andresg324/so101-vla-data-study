# Pre-registered Protocol for SO-101 Experiment

## Independent variables (conditions change)
Data-collection strategy, across four conditions:
1. **clean** — same object position, same lighting, near-identical demos
2. **randomized** — object starts in different positions each demo
3. **recovery** — deliberately include failed grasps and teleoperated recovery
4. **visual** — vary lighting, add distractor objects, change background, different colored cube

## Fixed variables (confound control)
1. **model** - SmolVLA
2. **task** - pick up a cube and place it in a cup
3. **demo budget** - 50 episodes
4. **hardware** - camera positions, gripper, control frequency

Changing any of these mid-study invalidates the comparison.

## Evaluation
Test all four trained policies on held-out conditions none were trained on:
novel object positions, new lighting, unseen distractors.

## Pre-committed metrics
1. Task success rate per (condition × eval cell)
2. Report per-cell base rates and confidence intervals over >=3 seeds
3. Primary comparison: generalization gap = (in-distribution success) - (held-out success)

## Dataset naming (locked)
cube-pickup-{clean,randomized,recovery,visual}