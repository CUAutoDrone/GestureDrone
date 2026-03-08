# Gesture-Based Drone Control System - Setup & Running Instructions

## System Overview

This ROS 2 system enables gesture-based control of a drone using:
- **Hand Detection**: Simple skin-color based detection (fallback to MediaPipe if available)
- **Gesture Classification**: PyTorch neural network model trained on hand gestures
- **ROS 2 Architecture**: Two nodes communicating via `/gesture_command` topic

## Environment Setup

### 1. Virtual Environment

The system uses an isolated Python virtual environment at `/home/etienne/gesture_env` with pinned dependencies:

```bash
# Python venv already created with:
source /home/etienne/gesture_env/bin/activate
pip list  # Shows: numpy==1.26, mediapipe, torch, opencv-python, etc.
```

### 2. Installed Dependencies

- **numpy==1.26** (pinned to avoid 2.x incompatibility)
- **mediapipe** (hand detection - uses fallback if unavailable)
- **torch, torchvision** (gesture classification model)
- **opencv-python** (camera capture and visualization)
- **rclpy, std-msgs, geometry-msgs, tf-transformations** (ROS 2)

## Running the System

### Launch Everything at Once

```bash
cd ~/GestureDrone
source install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py
```

This starts:
1. **gesture_inference** node: Captures webcam → detects gestures → publishes `/gesture_command`
2. **flight_controller** node: Listens to `/gesture_command` → publishes drone movement commands

### What You'll See

- OpenCV window showing:
  - Detected hand contour (green outline)
  - Current gesture name (e.g., "Thumb_Left")
  - Detected command (e.g., "YAW_LEFT")

- Console output:
  ```
  [gesture_inference] 🚁 COMMAND: YAW_LEFT (Gesture: Thumb_Left)
  [flight_controller] Yaw LEFT by 90 degrees
  ```

### Exit

Press `q` in the OpenCV window, or `Ctrl+C` in terminal

## Gesture → Drone Command Mapping

| Gesture | Command | Effect |
|---------|---------|--------|
| Thumb_Up | PITCH_FORWARD | Move forward |
| Thumb_Down | PITCH_BACKWARD | Move backward |
| Pointing_Up | THROTTLE_UP | Increase altitude |
| Pointing_Down | THROTTLE_DOWN | Decrease altitude |
| Thumb_Left | YAW_LEFT | Turn left |
| Thumb_Right | YAW_RIGHT | Turn right |
| Palm_Left | ROLL_LEFT | Roll left |
| Palm_Right | ROLL_RIGHT | Roll right |
| Open_Palm | HOLD_POSITION | Hover |
| Victory | SPEED_UP | Increase speed |
| Victory_Inverted | SPEED_DOWN | Decrease speed |
| Closed_Fist | KILL | Emergency stop |

## File Structure

```
gesture_to_flight/
├── gesture_inference_ros_node.py    # Main gesture detection & ROS node
├── flightInstructions.py             # Flight controller ROS node
├── launch/
│   └── gesture_drone.launch.py       # Launch both nodes (uses venv Python)
├── setup.py                          # Package configuration
└── package.xml                       # ROS 2 package metadata

gesture_cv/
├── GestureData/
│   ├── gesture_classifier.pt         # PyTorch model
│   ├── gesture_classifier.joblib     # Sklearn fallback
│   ├── label_map.json                # Gesture labels
│   └── gesture_data_full.csv         # Training data
└── GestureTraining/
    ├── GestureInference.py           # Reference implementation
    └── GestureTraining.py            # Model training script
```

## Troubleshooting

### "Camera not available"
- Check if camera is connected: `ls /dev/video*`
- Try: `cheese` to test camera access

### "Model loading failed"
- Verify model files exist: 
  ```bash
  ls -la ~/GestureDrone/gesture_cv/GestureData/gesture_classifier.pt
  ls -la ~/GestureDrone/gesture_cv/GestureData/label_map.json
  ```

### "No hand detected"
- Improve lighting
- Use skin tones that work with HSV-based detection
- Check that hand fills ~5-10% of camera frame

### ROS 2 import errors
- Source setup: `source ~/GestureDrone/install/setup.bash`
- Verify venv: `/home/etienne/gesture_env/bin/python3 -c "import rclpy; print('OK')"`

## Development Notes

- **Prediction smoothing**: Uses 5-frame history with majority voting to reduce jitter
- **Command publishing**: Only publishes when gesture changes to reduce topic chatter
- **Hand detector**: Simple skin-color HSV-based (robust, doesn't require MediaPipe model)
- **Launch file**: Uses venv Python interpreter at `/home/etienne/gesture_env/bin/python3`

## Next Steps

- Train model on more diverse hand poses to improve accuracy
- Integrate with actual MAVROS drone interface (currently simulated)
- Add gesture recording/customization UI
- Implement multi-hand support for more complex gestures
