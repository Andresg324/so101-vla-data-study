#!/usr/bin/env bash
# scripts/run_inference.sh
# Run the trained SmolVLA policy autonomously and record the result.
# The recorded video IS your inference footage.
#
# Camera keys MUST match the policy's training names (from the rename_map):
#   camera1 = birds-eye (overhead) view
#   camera2 = wrist (gripper) view
# Verify which index is which RIGHT NOW (they shuffle) and set below.
#
# Run:  conda activate lerobot && bash scripts/run_inference.sh
set -e

FOLLOWER_PORT=/dev/tty.usbmodem5B415324451   # 12V arm, drives itself

OVERHEAD_IDX=1 # Index of overhead camera
WRIST_IDX=0      # Index of wrist camera

lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=my_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: ${WRIST_IDX}, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: ${OVERHEAD_IDX}, width: 640, height: 480, fps: 30}}" \
    --policy.path=Andresg324/smolvla-cube-clean \
    --policy.device=mps \
    --strategy.type=sentry \
    --task="Pick up the cube and place it in the cup" \
    --dataset.repo_id=Andresg324/rollout_cube_clean \
    --dataset.single_task="Pick up the cube and place it in the cup" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --display_data=true
