#!/usr/bin/env python3
"""
ROS 2 Node for Gesture Inference
Detects hand gestures from camera and publishes commands to /gesture_command topic
"""

import os
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import numpy as np
import torch
import torch.nn as nn

import mediapipe as mp


# -------------------------
# Paths
# -------------------------
WORKSPACE_ROOT = "/home/etienne/GestureDrone"
GESTURE_DATA_DIR = os.path.join(WORKSPACE_ROOT, "gesture_cv", "GestureData")

MODEL_PATH = os.path.join(GESTURE_DATA_DIR, "gesture_classifier.pt")
LABEL_MAP_PATH = os.path.join(GESTURE_DATA_DIR, "label_map.json")


# -------------------------
# Load labels
# -------------------------
try:
    with open(LABEL_MAP_PATH, "r") as f:
        LABELS = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find label map at {LABEL_MAP_PATH}")
    sys.exit(1)

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


# -------------------------
# Gesture -> drone command mapping
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
    "Victory_Inverted": "SPEED_DOWN",
    "Closed_Fist":     "KILL"
}


class GestureInferenceNode(Node):
    def __init__(self):
        super().__init__('gesture_inference')
        
        # Publisher for gesture commands
        self.cmd_pub = self.create_publisher(String, '/gesture_command', 10)
        
        # Load model
        try:
            self.model = GestureNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            self.model.eval()
            self.get_logger().info(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        # MediaPipe hand detector
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera")
            raise RuntimeError("Camera not available")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Prediction smoothing
        self.PRED_HISTORY_LEN = 5
        self.pred_history = []
        self.last_command = None
        
        # Timer for camera processing loop (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.process_frame)
        
        self.get_logger().info("Gesture Inference Node initialized and ready")

    def process_frame(self):
        """Process a frame from the camera and detect gestures"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        gesture_label = "NO_HAND"
        command_text = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                
                # Extract 21 * 3 = 63 normalized landmarks
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32
                ).flatten()  # (63,)
                
                # Torch inference
                with torch.no_grad():
                    x = torch.from_numpy(landmarks).unsqueeze(0)  # (1, 63)
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs, dim=1).item())
                    gesture_label = LABELS[pred_idx]
                
                # Update history for smoothing
                self.pred_history.append(gesture_label)
                if len(self.pred_history) > self.PRED_HISTORY_LEN:
                    self.pred_history.pop(0)
                
                # Majority vote over history
                stable_prediction = max(
                    set(self.pred_history),
                    key=self.pred_history.count
                )
                
                # Map to command
                command_text = COMMAND_MAP.get(stable_prediction, "UNKNOWN")
                
                # Only publish if command changed
                if command_text != self.last_command and command_text != "UNKNOWN":
                    self.get_logger().info(f"🚁 COMMAND: {command_text} (Gesture: {stable_prediction})")
                    msg = String()
                    msg.data = command_text
                    self.cmd_pub.publish(msg)
                    self.last_command = command_text
                
                break  # only first hand

        # Overlay text on frame
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
        
        cv2.imshow("Gesture Drone - ROS Inference", frame)
        
        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Quit request received, shutting down...")
            raise KeyboardInterrupt()

    def destroy(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
