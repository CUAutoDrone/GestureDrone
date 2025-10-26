#use python GestureInference.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os

# --- paths relative to this file ---
MODEL_PATH   = "../GestureData/gesture_classifier.joblib"
ENCODER_PATH = "../GestureData/label_encoder.joblib"

# load trained model + label encoder
clf = joblib.load(MODEL_PATH)
le  = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_prediction = None          # last raw predicted class label
stable_prediction = None        # the one we "trust"
same_count = 0
STABILITY_FRAMES = 5            # require N consistent frames before we accept

print("Press 'q' to quit.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw landmarks on the preview
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # same feature extraction you used for training:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten().reshape(1, -1)

            # model -> class index -> human label
            class_idx = clf.predict(landmarks)[0]
            class_label = le.inverse_transform([class_idx])[0]

            # temporal smoothing so commands don't flicker
            if class_label == last_prediction:
                same_count += 1
            else:
                same_count = 0
                last_prediction = class_label

            if same_count >= STABILITY_FRAMES:
                stable_prediction = class_label

            # overlay text
            cv2.putText(
                frame,
                f"live: {class_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )
            if stable_prediction:
                cv2.putText(
                    frame,
                    f"stable: {stable_prediction}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2
                )

    cv2.imshow("Realtime Gesture Prediction", frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
