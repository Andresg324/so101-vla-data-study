#!/usr/bin/env bash
# scripts/preview_cameras.sh
# Fast live preview of both cameras, addressed BY NAME rather than by index, so it
# survives the macOS index shuffle. Use this to confirm the cameras are alive and framed;
# use check_cameras_live.sh when you also need teleoperation running.
# Overhead = Logitech C270 (gantry). Wrist = Seeed "Web Camera".
# Quit: Ctrl-C, or close both windows.

set -e
trap 'kill 0' EXIT INT TERM

ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "C270 HD WEBCAM" &
ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "Web Camera" &
wait