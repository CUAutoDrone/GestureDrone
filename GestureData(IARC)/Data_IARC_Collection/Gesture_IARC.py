"""
    # 1) Go to the repo root
cd ~/Desktop/CUAD/GestureDrone
# Move into the project folder first so all relative paths work.

# 2) Check which Python you are currently using
which python3
python3 --version
# See whether this computer is using system Python, conda Python, Homebrew Python, etc.

# 3) If conda/base is active, turn it off
conda deactivate
# Prevents mixing conda packages with the project environment.

# 4) Remove any old project venv if you want a completely clean setup
rm -rf .venv
# Optional, but useful if the previous environment was broken or used the wrong Python version.

# 5) Create a fresh venv using Python 3.10 if available
/usr/local/bin/python3 -m venv .venv
# Creates a project-local virtual environment with Python 3.10.x for consistency.
# If /usr/local/bin/python3 does not exist, first run:
# which python3
# and use that full path instead.

# 6) Activate the venv
source .venv/bin/activate
# Makes this repo use its own isolated Python and packages.

# 7) Verify the active interpreter
which python
python --version
# Should point to .../GestureDrone/.venv/bin/python and show the intended Python version.

# 8) Upgrade package tools
python -m pip install --upgrade pip setuptools wheel
# Ensures package installation works cleanly across different machines.

# 9) Install required libraries
pip install opencv-python mediapipe numpy
# Installs the core dependencies for camera capture, hand landmarks, and array processing.

# 10) Verify the libraries import correctly
python -c "import cv2, mediapipe, numpy; print('cv2 OK'); print('mediapipe OK'); print('numpy OK')"
# Quick sanity check before trying to run the scripts.

# 11) Move into the IARC data collection folder
cd "GestureData(IARC)/Data_IARC_Collection"
# This folder contains Gesture_IARC.py, DataProcessing.py, and hand_landmarker.task.

# 12) Confirm the model file is present
ls -lh hand_landmarker.task
# The MediaPipe Tasks API needs this file in the same folder as Gesture_IARC.py.

# 13) Run the raw data collection script
python Gesture_IARC.py
# Starts the webcam-based gesture collection pipeline.
"""

# Collect raw landmark data for 4 command gestures:
# 1. Arm_Drone
# 2. Takeoff
# 3. Start_Search
# 4. Land

import csv
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GESTURE_LABELS = [
    "Arm_Drone",
    "Takeoff",
    "Start_Search",
    "Land",
]

SAVE_PATH = "gesture_data_command4_raw.csv"
MODEL_PATH = "hand_landmarker.task"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: could not find {MODEL_PATH}")
        print("Put hand_landmarker.task in this same folder.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    current_label = None
    data = []

    print("Press number keys (1-4) to select gesture label.")
    print("Press 's' to save current frame, 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            landmarks = None
            handedness_label = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                handedness_label = (
                    result.handedness[0][0].category_name
                    if result.handedness else "Unknown"
                )

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    dtype=np.float32
                ).flatten()

            if current_label:
                cv2.putText(
                    frame,
                    f"Label: {current_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            if handedness_label:
                cv2.putText(
                    frame,
                    f"Hand: {handedness_label}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            cv2.imshow("Collecting 4 Command Gestures", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord(str(i + 1)) for i in range(len(GESTURE_LABELS))]:
                current_label = GESTURE_LABELS[int(chr(key)) - 1]
                print(f"Selected label: {current_label}")

            elif key == ord("s"):
                if current_label and landmarks is not None:
                    sample = list(landmarks) + [handedness_label, current_label]
                    data.append(sample)
                    print(f"Saved {current_label} sample ({len(data)} total)")
                    time.sleep(0.2)
                else:
                    print("No hand detected or no label selected.")

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    if data:
        with open(SAVE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(21 * 3)] + ["handedness", "label"]
            writer.writerow(header)
            writer.writerows(data)
        print(f"Saved {len(data)} samples to {SAVE_PATH}")
    else:
        print("No samples captured, not writing CSV.")


if __name__ == "__main__":
    main()