# Collect raw landmark data for 4 command gestures:
# 1. Arm_Drone
# 2. Takeoff
# 3. Start_Search
# 4. Land
#
# Saves:
# - 21 landmarks * (x, y, z)
# - handedness ("Left" or "Right")
# - label
#
# Uses MediaPipe Tasks API (HandLandmarker), which works with current mediapipe builds.

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
MODEL_PATH = "hand_landmarker.task"  # put this file in the same folder as this script


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: could not find {MODEL_PATH}")
        print("Download MediaPipe hand_landmarker.task and place it in this folder.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
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

            # Convert OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            landmarks = None
            handedness_label = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                handedness_label = result.handedness[0][0].category_name if result.handedness else "Unknown"

                # draw landmarks manually
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