#use "python GestureData2.py"

#This file collects data for pre-implemented gestures
#Two data files will be merged

import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# ONLY NEW GESTURES YOU HAVEN'T COLLECTED YET
# number keys: 1..5
GESTURE_LABELS = [
    "Open_Palm",     # hover / stop
    "Victory",       # speed up
    "Thumb_Up",      # pitch forward
    "Thumb_Down",    # pitch backward
    "Pointing_Up"    # throttle up
]

SAVE_PATH = "gesture_data2.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

current_label = None
data = []

print("Press number keys (1–5) to select gesture label.")
print("Press 's' to save current frame, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # mirror webcam
    frame = cv2.flip(frame, 1)

    # mediapipe needs RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # if we see a hand, draw it and grab landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # flatten 21 (x,y,z) triplets -> length 63 feature vector
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()
    else:
        landmarks = None

    # show which label is active
    if current_label:
        cv2.putText(
            frame,
            f"Label: {current_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Collecting NEW Gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    # pick which gesture you're labeling
    if key in [ord(str(i+1)) for i in range(len(GESTURE_LABELS))]:
        current_label = GESTURE_LABELS[int(chr(key)) - 1]
        print(f"Selected label: {current_label}")

    # save current frame
    elif key == ord('s'):
        if current_label and landmarks is not None:
            sample = np.append(landmarks, current_label)
            data.append(sample)
            print(f"Saved {current_label} sample ({len(data)} total)")
            time.sleep(0.2)  # debounce so it doesn't spam
        else:
            print("No hand detected or no label selected.")

    # quit out
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# dump to CSV
if data:
    with open(SAVE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21 * 3)] + ["label"]
        writer.writerow(header)
        writer.writerows(data)
    print(f"✅ Saved {len(data)} samples to {SAVE_PATH}")
else:
    print("No samples captured, not writing CSV.")
