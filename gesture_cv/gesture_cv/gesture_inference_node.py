#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from pathlib import Path


class GestureInferenceNode(Node):
    """ROS 2 node for gesture inference and command publishing."""

    def __init__(self):
        super().__init__('gesture_inference_node')
        
        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('show_video', True)
        self.declare_parameter('stability_frames', 5)
        self.declare_parameter('min_detection_confidence', 0.7)
        self.declare_parameter('min_tracking_confidence', 0.7)
        
        # Get parameters
        camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        self.show_video = self.get_parameter('show_video').get_parameter_value().bool_value
        self.stability_frames = self.get_parameter('stability_frames').get_parameter_value().integer_value
        min_detection = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        min_tracking = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        
        # Create publishers
        self.gesture_pub = self.create_publisher(String, 'gesture/raw', 10)
        self.command_pub = self.create_publisher(String, 'gesture/command', 10)
        self.stable_gesture_pub = self.create_publisher(String, 'gesture/stable', 10)
        
        # Get package path for model loading
        # Try installed location first, then fall back to source location
        try:
            package_dir = Path(get_package_share_directory('gesture_cv'))
            model_path = package_dir / 'GestureData' / 'gesture_classifier.joblib'
            encoder_path = package_dir / 'GestureData' / 'label_encoder.joblib'
        except:
            # Fallback to source location for development
            package_dir = Path(__file__).parent.parent.parent
            model_path = package_dir / 'GestureData' / 'gesture_classifier.joblib'
            encoder_path = package_dir / 'GestureData' / 'label_encoder.joblib'
        
        # Load trained model + label encoder
        try:
            self.clf = joblib.load(str(model_path))
            self.le = joblib.load(str(encoder_path))
            self.get_logger().info(f'Loaded model from {model_path}')
            self.get_logger().info(f'Loaded encoder from {encoder_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking
        )
        self.mp_drawing = mp_drawing
        self.mp_hands = mp_hands
        
        # Maps gesture labels -> drone commands
        self.COMMAND_MAP = {
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
            "Stop_Palm":         "HOLD_POSITION",
            
            # Emergency
            "Closed_Fist":       "KILL"
        }
        
        # State tracking
        self.last_prediction = None
        self.stable_prediction = None
        self.same_count = 0
        self.active_command = "NONE"
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            raise RuntimeError(f'Failed to open camera {camera_id}')
        
        # Create timer for processing frames
        timer_period = 0.033  # ~30 FPS
        self.timer = self.create_timer(timer_period, self.process_frame)
        
        self.get_logger().info('Gesture inference node started')
        self.get_logger().info(f'Publishing to topics: gesture/raw, gesture/command, gesture/stable')
    
    def process_frame(self):
        """Process a single frame from the camera."""
        if not rclpy.ok():
            return
            
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Failed to read frame from camera')
            return
        
        # Flip horizontally so it behaves like a mirror
        frame = cv2.flip(frame, 1)
        
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        gesture_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture_detected = True
                
                # Draw skeleton if showing video
                if self.show_video:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Extract same 63-dim feature vector we trained on
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                ).flatten().reshape(1, -1)
                
                # Classifier -> index -> label string
                class_idx = self.clf.predict(landmarks)[0]
                class_label = self.le.inverse_transform([class_idx])[0]
                
                # Publish raw gesture
                raw_msg = String()
                raw_msg.data = class_label
                self.gesture_pub.publish(raw_msg)
                
                # Temporal smoothing so we don't flicker
                if class_label == self.last_prediction:
                    self.same_count += 1
                else:
                    self.same_count = 0
                    self.last_prediction = class_label
                
                # Update stable prediction
                if self.same_count >= self.stability_frames:
                    if self.stable_prediction != class_label:
                        self.stable_prediction = class_label
                        
                        # Publish stable gesture
                        stable_msg = String()
                        stable_msg.data = self.stable_prediction
                        self.stable_gesture_pub.publish(stable_msg)
                        
                        # Figure out the command based on the STABLE prediction
                        mapped_command = self.COMMAND_MAP.get(self.stable_prediction, "UNKNOWN")
                        
                        # Only update command if it changed
                        if mapped_command != self.active_command:
                            self.active_command = mapped_command
                            
                            # Publish command
                            command_msg = String()
                            command_msg.data = self.active_command
                            self.command_pub.publish(command_msg)
                            
                            self.get_logger().info(f'Gesture: {self.stable_prediction} -> Command: {self.active_command}')
                
                # Overlay text on video if enabled
                if self.show_video:
                    cv2.putText(
                        frame,
                        f"live: {class_label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    if self.stable_prediction:
                        cv2.putText(
                            frame,
                            f"stable: {self.stable_prediction}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2
                        )
        
        # Overlay the active command (even if no hand in frame right now, show last known)
        if self.show_video:
            cv2.putText(
                frame,
                f"COMMAND: {self.active_command}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            cv2.imshow("Realtime Gesture Control", frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Quit key pressed, shutting down...')
                rclpy.shutdown()
        
        # Reset stable prediction if no hand detected for a while
        if not gesture_detected:
            self.same_count = 0
    
    def destroy_node(self):
        """Cleanup resources when node is destroyed."""
        self.get_logger().info('Shutting down gesture inference node...')
        if self.cap.isOpened():
            self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GestureInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
