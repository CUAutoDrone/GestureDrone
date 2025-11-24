import os
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
#pip install torch torchvision torchaudio
import torch.nn as nn
#pip install opencv-python mediapipe numpy

#get into GestureTraining/ and run python3 GestureInference.py


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../GestureData/gesture_classifier.pt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "../GestureData/label_map.json")


with open(LABEL_MAP_PATH, "r") as f:
    LABELS = json.load(f)  # list: idx -> label string
num_classes = len(LABELS)

# -------------------------
# Model definition
# -------------------------
class GestureNet(nn.Module):
    def __init__(self, input_dim=63, num_classes=num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = GestureNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print(f"Loaded model from {MODEL_PATH}")
print("Labels:", LABELS)

# -------------------------
# Mediapipe setup
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------
# Gesture -> drone command mapping
# (update names to match your CSV labels)
# -------------------------
COMMAND_MAP = {
    "Thumb_Up":        "PITCH_FORWARD",
    "Thumb_Down":      "PITCH_BACKWARD",
    "Pointing_Up":     "THROTTLE_UP",
    "Pointing_Down":   "THROTTLE_DOWN",
    "Thumb_Left":      "YAW_LEFT",
    "Thumb_Right":     "YAW_RIGHT",
    "Palm_Left":       "ROLL_LEFT",
    "Palm_Right":      "ROLL_RIGHT",
    "Open_Palm":       "HOLD_POSITION",
    "Victory":         "SPEED_UP",
    "Victory_Inverted":"SPEED_DOWN",
    "Closed_Fist":     "KILL"
}

# For smoothing predictions
PRED_HISTORY_LEN = 5
pred_history = []

last_command = None

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)  # Pi camera; adjust index if needed

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed, exiting.")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_label = "NO_HAND"
    command_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 * 3 = 63 normalized landmarks
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            ).flatten()  # (63,)

            # Torch inference
            with torch.no_grad():
                x = torch.from_numpy(landmarks).unsqueeze(0)  # (1,63)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                gesture_label = LABELS[pred_idx]

            # Update history for smoothing
            pred_history.append(gesture_label)
            if len(pred_history) > PRED_HISTORY_LEN:
                pred_history.pop(0)

            # Majority vote over history
            stable_prediction = max(
                set(pred_history),
                key=pred_history.count
            )

            # Map to command
            command_text = COMMAND_MAP.get(stable_prediction, "UNKNOWN")

            # Only update last_command if changed
            if command_text != last_command:
                print(f"Gesture: {stable_prediction}  ->  COMMAND: {command_text}")
                last_command = command_text

            break  # only first hand

    # -------------------------
    # Overlay text on frame
    # -------------------------
    cv2.putText(
        frame,
        f"Gesture: {gesture_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Command: {command_text}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Gesture Drone - PyTorch", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
