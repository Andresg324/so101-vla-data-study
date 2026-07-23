#!/usr/bin/env bash
# Record a teleoperated dataset for one condition of the study.
# Usage:  bash scripts/record_dataset.sh <condition> <num_episodes>
# Example: bash scripts/record_dataset.sh clean 50
set -e
CONDITION=${1:-clean} # Conditions are 'clean' | 'randomized' | 'recovery' | 'visual'
NUM_EPISODES=${2:-5}

# ---- Hardware information ----
FOLLOWER_PORT=/dev/tty.usbmodem5B415324451   # 12V arm that executes
LEADER_PORT=/dev/tty.usbmodem5B415328441     # 5V arm - moved manually
HF_USER=Andresg324

#Two cameras -> dataset keys 'wrist' (Seeed idx0) and 'overhead' (C270 idx1)
# dataset name encodes the condition; instruction is the same across all conditions
# push proof runs local (false); push real runs to the hub (ture)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=my_follower_arm \
    --robot.cameras="{ overhead: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/cube-pickup-${CONDITION} \
    --dataset.single_task="Pick up the cube and place it in the cup" \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.fps=30 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false
