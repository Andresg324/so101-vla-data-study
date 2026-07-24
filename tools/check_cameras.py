#!/usr/bin/env python3
"""
tools/check_cameras.py
Live preview of the two STUDY cameras at the exact resolution the dataset
records (640x480), so you can frame them before recording. What you see here
is literally what the policy will 'see' — frame accordingly.

Cameras (must match record_dataset.sh):
  wrist    = index 0  (Seeed, hand-mounted)
  overhead = index 1  (C270, birds-eye)

Run:  conda activate lerobot && python tools/check_cameras.py
Press 'q' in a preview window to quit.
"""

import cv2

# Role -> OpenCV index; keep this identical to record_dataset.sh
CAMERAS = {"wrist": 0, "overhead": 1}
WIDTH, HEIGHT = 640, 480 # Same as the recording so that the previews match the experiments


for name, idx in CAMERAS.items():
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    for _ in range(10):
        ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"tools/preview_{name}.jpg", frame)
        print(f"saved tools/preview_{name}.jpeg")
    else:
        print(f"could not read from camera index {idx}")
    cap.release()