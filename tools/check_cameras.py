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

def main():
    # Open each camera for live preview
    caps = {}
    for name, idx in CAMERAS.items():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        caps[name] = cap

    print("Live feeds up, adjust cameras as needed, then press 'q' to quit.")

    while True:
        for name, cap in caps.items():
            ok, frame = cap.read() # Grab one frame
            if ok:
                cv2.imshow(name, frame) # one window per camera, titled by the role
        if cv2.waitKey(1) & 0xFF == ord("q"): # keep the windows live, quit when 'q' is pressed
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()