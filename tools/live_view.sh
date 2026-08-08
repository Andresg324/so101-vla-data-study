#!/usr/bin/env bash
# Live preview of both cameras before recording.
# Addressed BY NAME, not index — names survive the macOS index shuffle.
# Overhead = Logitech C270 (gantry).  Wrist = Seeed "Web Camera".
# Quit: close both windows, or `pkill ffplay`.

ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "C270 HD WEBCAM" &
ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "Web Camera" &
wait