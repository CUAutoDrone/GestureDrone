import cv2
import mediapipe as mp
import numpy as np
import joblib

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

# maps gesture labels -> drone commands
COMMAND_MAP = {
    # Pitch (move forward/back)
    "Thumb_Up":          "PITCH_FORWARD",
    "Thumb_Down":        "PITCH_BACKWARD",

    # Throttle (vertical motion)
    "Pointing_Up":       "THROTTLE_UP",
    "Pointing_Down":     "THROTTLE_DOWN",

    # Yaw (rotate left/right)
    "Thumb_Left":        "YAW_LEFT",
    "Thumb_Right":       "YAW_RIGHT",

    # Roll (bank left/right)
    "Palm_Left":         "ROLL_LEFT",
    "Palm_Right":        "ROLL_RIGHT",

    # Speed control
    "Victory":           "SPEED_UP",
    "Victory_Inverted":  "SPEED_DOWN",

    # Hover / hold position
    "Open_Palm":         "HOLD_POSITION",
    "Stop_Palm":         "HOLD_POSITION",  # legacy label you collected earlier

    # Emergency
    "Closed_Fist":       "KILL"
}

last_prediction = None        # most recent raw gesture label (frame-level)
stable_prediction = None      # gesture after stability filter
same_count = 0
STABILITY_FRAMES = 5          # how many consistent frames before we "lock" it

active_command = "NONE"       # what we're currently showing on screen

print("Press 'q' to quit.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # flip horizontally so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # extract same 63-dim feature vector we trained on
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            ).flatten().reshape(1, -1)

            # classifier -> index -> label string
            class_idx = clf.predict(landmarks)[0]
            class_label = le.inverse_transform([class_idx])[0]

            # temporal smoothing so we don't flicker
            if class_label == last_prediction:
                same_count += 1
            else:
                same_count = 0
                last_prediction = class_label

            if same_count >= STABILITY_FRAMES:
                stable_prediction = class_label

            # figure out the command based on the STABLE prediction
            if stable_prediction:
                mapped_command = COMMAND_MAP.get(stable_prediction, "UNKNOWN")
                active_command = mapped_command  # update what we overlay

            # overlay text: live gesture
            cv2.putText(
                frame,
                f"live: {class_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            # overlay text: stable gesture
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

    # overlay the active command (even if no hand in frame right now, show last known)
    cv2.putText(
        frame,
        f"COMMAND: {active_command}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    # show window
    cv2.imshow("Realtime Gesture Control", frame)

    # quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
