#!/usr/bin/env bash
# scripts/check_cameras_live.sh
# Live camera preview using LeRobot's built-in 'rerun' viewer (works with
# headless OpenCV, unlike cv2.imshow). Runs teleoperation WITH the cameras
# attached, so you frame the cameras and confirm they work in the real
# record pipeline at the same time.
#
# Run:  conda activate lerobot && bash scripts/check_cameras_live.sh
# Stop: Ctrl + C

set -e

# ---- Hardware information ----
FOLLOWER_PORT=/dev/tty.usbmodem5B415324451   # 12V arm that executes
LEADER_PORT=/dev/tty.usbmodem5B415328441     # 5V arm - moved manually

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=my_follower_arm \
    --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=my_leader_arm \
    --display_data=true