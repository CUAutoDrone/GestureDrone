# Quick Start Guide - Gesture Drone System

## One-Command Startup

The complete gesture drone system is now ready to use with a single ROS 2 launch command!

### Start Everything

```bash
# In any terminal in the GestureDrone workspace:
source ~/GestureDrone/install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py
```

That's it! This will:
1. ✅ Start the webcam feed with live gesture detection
2. ✅ Detect your hand gestures in real-time
3. ✅ Display detected gesture and translated command on screen
4. ✅ Publish drone flight commands
5. ✅ Print command translations to console (e.g., "🚁 COMMAND: PITCH_FORWARD")

### What You'll See

**Console Output:**
```
[gesture_inference-1] [INFO] [...]: 🚁 COMMAND: PITCH_FORWARD (Gesture: Thumb_Up)
[flight_controller-2] [INFO] [...]: Received command: PITCH_FORWARD
[flight_controller-2] [INFO] [...]: Moving drone by (10.0, 0.0, 0.0)
```

**Webcam Window:**
- Live camera feed with hand landmarks drawn
- Text overlay showing detected gesture
- Text overlay showing translated command

## Command Cheat Sheet

Make these hand gestures to control the drone:

- 👍 **Thumb Up** → Move Forward
- 👎 **Thumb Down** → Move Backward  
- ☝️ **Pointing Up** → Move Up
- 🫵 **Pointing Down** → Move Down
- 👈 **Thumb Left** → Turn Left
- 👉 **Thumb Right** → Turn Right
- ✋ **Open Palm** → Hold Position
- ✌️ **Victory** → Speed Up
- ✌️ **Victory Upside Down** → Speed Down
- ✊ **Closed Fist** → Emergency Stop

## Exit

Press `q` in the webcam window to quit the system cleanly.

## Building from Scratch (if needed)

```bash
cd ~/GestureDrone
colcon build --packages-select gesture_to_flight
source ~/GestureDrone/install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py
```

## Running Individual Nodes (Advanced)

To run nodes separately:

```bash
# Terminal 1 - Gesture Detection
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight gesture_inference
```

```bash
# Terminal 2 - Flight Control
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight flight_controller
```

## System Architecture

```
Webcam → [gesture_inference node]
          ↓ (publishes /gesture_command)
          [flight_controller node]
          ↓ (publishes /mavros/setpoint_position/local)
          MAVROS → Drone
```

## Troubleshooting

**"Camera not found"**
- Check camera: `ls /dev/video*`
- Try different camera index in `gesture_inference_ros_node.py`: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

**"No gestures detected"**
- Ensure good lighting
- Keep hand clearly visible
- Gestures take ~5 frames to register (smoothing algorithm)

**"Model not found"**
- Verify files exist:
  - `gesture_cv/GestureData/gesture_classifier.pt`
  - `gesture_cv/GestureData/label_map.json`

For more details, see [README.md](README.md)
