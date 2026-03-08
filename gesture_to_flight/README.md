# Gesture Drone Control System

This ROS 2 package enables gesture-based drone flight control using hand gesture recognition from a webcam.

## System Architecture

The system consists of two ROS 2 nodes that communicate via message passing:

1. **gesture_inference** - Gesture Detection Node
   - Captures video from webcam (camera 0)
   - Uses MediaPipe for hand landmark detection
   - Runs PyTorch neural network to classify hand gestures
   - Publishes recognized commands to `/gesture_command` topic
   - Displays live video feed with gesture overlay

2. **flight_controller** - Flight Control Node
   - Subscribes to `/gesture_command` topic
   - Converts gesture commands to drone flight instructions
   - Publishes to `/mavros/setpoint_position/local` topic
   - Maintains drone position state and speed factor

## Gesture to Command Mapping

| Gesture | Command | Effect |
|---------|---------|--------|
| Thumb Up | PITCH_FORWARD | Move forward |
| Thumb Down | PITCH_BACKWARD | Move backward |
| Pointing Up | THROTTLE_UP | Move up |
| Pointing Down | THROTTLE_DOWN | Move down |
| Thumb Left | YAW_LEFT | Rotate left |
| Thumb Right | YAW_RIGHT | Rotate right |
| Palm Left | ROLL_LEFT | Bank left |
| Palm Right | ROLL_RIGHT | Bank right |
| Open Palm | HOLD_POSITION | Hold current position |
| Victory | SPEED_UP | Increase movement speed (1.1x to 2.0x) |
| Victory Inverted | SPEED_DOWN | Decrease movement speed (0.1x to 1.0x) |
| Closed Fist | KILL | Emergency stop |

## Building the Package

```bash
cd ~/GestureDrone
colcon build --packages-select gesture_to_flight
```

## Running the System

### Option 1: Using the Launch File (Recommended)

```bash
# Source the setup script
source ~/GestureDrone/install/setup.bash

# Launch both nodes together
ros2 launch gesture_to_flight gesture_drone.launch.py
```

This will start:
- The gesture inference node (with live webcam feed display)
- The flight controller node (listening for gesture commands)

### Option 2: Running Nodes Individually

In separate terminals:

```bash
# Terminal 1 - Start gesture inference node
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight gesture_inference
```

```bash
# Terminal 2 - Start flight controller node
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight flight_controller
```

## Console Output

When a gesture command is detected and published, you'll see output like:

```
[gesture_inference-1] [INFO] [1646000000.123456]: 🚁 COMMAND: PITCH_FORWARD (Gesture: Thumb_Up)
[flight_controller-2] [INFO] [1646000000.234567]: Received command: PITCH_FORWARD
[flight_controller-2] [INFO] [1646000000.234890]: Moving drone by (10.0, 0.0, 0.0)
```

## Requirements

- ROS 2 (tested with Humble)
- Python 3.10+
- PyTorch
- OpenCV
- MediaPipe
- numpy
- tf_transformations
- MAVROS (for actual drone flight) - optional for testing

## Troubleshooting

### Camera not found
- Ensure camera is connected: `ls /dev/video*`
- Check camera permissions
- Modify `camera_id` in `gesture_inference_ros_node.py` if using non-default camera

### Model not loading
- Verify model files exist:
  - `gesture_cv/GestureData/gesture_classifier.pt`
  - `gesture_cv/GestureData/label_map.json`
- Rebuild after moving files: `colcon build --packages-select gesture_to_flight`

### No gestures detected
- Ensure adequate lighting
- Hand should be clearly visible in camera frame
- Try adjusting detection confidence in `gesture_inference_ros_node.py`:
  ```python
  self.hands = self.mp_hands.Hands(
      max_num_hands=1,
      min_detection_confidence=0.7,  # Lower for easier detection
      min_tracking_confidence=0.7
  )
  ```

## Files

```
gesture_to_flight/
├── gesture_to_flight/
│   ├── __init__.py
│   ├── flightInstructions.py          # Flight controller node
│   └── gesture_inference_ros_node.py  # Gesture detection node
├── launch/
│   └── gesture_drone.launch.py        # Main launch file
├── package.xml                         # ROS 2 package metadata
├── setup.py                            # Python package setup
└── README.md                           # This file
```

## Development Notes

### Adding New Gestures

1. Update the gesture classifier model in `gesture_cv/`
2. Add new mapping in `COMMAND_MAP` dictionary in `gesture_inference_ros_node.py`
3. Add corresponding handler in `FlightController.command_callback()` in `flightInstructions.py`

### Adjusting Movement Parameters

Edit `FlightController` class in `flightInstructions.py`:
- `max_speed`: Maximum speed multiplier (default: 2.0)
- `min_speed`: Minimum speed multiplier (default: 0.1)
- Movement increments in `command_callback()` (default: 10 units)

### Message Flow

```
[Webcam]
   ↓
[gesture_inference node]
   ├─ Detects hand landmarks with MediaPipe
   ├─ Classifies gesture with PyTorch
   └─ Publishes /gesture_command (String msg)
        ↓
[ROS 2 Topic: /gesture_command]
        ↓
[flight_controller node]
   ├─ Receives command
   ├─ Converts to position/orientation
   └─ Publishes /mavros/setpoint_position/local (PoseStamped msg)
        ↓
[MAVROS]
   └─ Sends to actual drone (if connected)
```

## License

TODO: Add license information
