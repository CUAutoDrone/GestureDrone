#use "python GestureCollection.py"

import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# --- configuration ---
GESTURE_LABELS = [
    "Thumb_Left",
    "Thumb_Right",
    "Palm_Left",
    "Palm_Right",
    "Pointing_Down",
    "Victory_Inverted",
    "Stop_Palm",
    "Closed_Fist"
]
SAVE_PATH = "gesture_data.csv"

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

print("Press number keys (1–8) to select gesture label.")
print("Press 's' to save current frame, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # draw and maybe save
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # extract raw landmarks (x,y,z in [0,1] normalized to image, z is depth-ish)
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten()

    # UI text overlay
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

    cv2.imshow('Collecting Gestures', frame)

    # single key read for this frame
    key = cv2.waitKey(1) & 0xFF

    # choose gesture label 1-8
    if key in [ord(str(i+1)) for i in range(len(GESTURE_LABELS))]:
        current_label = GESTURE_LABELS[int(chr(key)) - 1]
        print(f"Selected label: {current_label}")

    # save this frame's landmarks with the active label
    elif key == ord('s'):
        if current_label and results.multi_hand_landmarks:
            sample = np.append(landmarks, current_label)
            data.append(sample)
            print(f"Saved {current_label} sample ({len(data)} total)")
            time.sleep(0.2)  # small pause so you don't spam 60 samples/sec
        else:
            print("No hand detected or no label selected.")

    # quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# dump to CSV at the end
if data:
    with open(SAVE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21 * 3)] + ["label"]
        writer.writerow(header)
        writer.writerows(data)

    print(f"Saved {len(data)} samples to {SAVE_PATH}")
else:
    print("No samples captured, not writing CSV.")
