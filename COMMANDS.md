# Available Commands

## Main Launch Command (Recommended)
Start everything with one command:
```bash
source ~/GestureDrone/install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py
```

## Individual Node Commands

### Start Gesture Detection Node Only
```bash
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight gesture_inference
```

### Start Flight Controller Node Only
```bash
source ~/GestureDrone/install/setup.bash
ros2 run gesture_to_flight flight_controller
```

## Monitoring Commands

### Watch Gesture Commands Being Published
```bash
source ~/GestureDrone/install/setup.bash
ros2 topic echo /gesture_command
```

### List All Active Topics
```bash
source ~/GestureDrone/install/setup.bash
ros2 topic list
```

### Show Topic Message Rate
```bash
source ~/GestureDrone/install/setup.bash
ros2 topic hz /gesture_command
```

### View Full Topic Information
```bash
source ~/GestureDrone/install/setup.bash
ros2 topic info /gesture_command
```

## Build Commands

### Build All Packages
```bash
cd ~/GestureDrone
colcon build
```

### Build Only gesture_to_flight
```bash
cd ~/GestureDrone
colcon build --packages-select gesture_to_flight
```

### Build with Verbose Output
```bash
cd ~/GestureDrone
colcon build --packages-select gesture_to_flight --event-handlers console_direct+
```

## Utility Commands

### Show Launch File Arguments
```bash
source ~/GestureDrone/install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py --show-args
```

### List Available Executables
```bash
source ~/GestureDrone/install/setup.bash
ros2 pkg executables gesture_to_flight
```

### Show Package Information
```bash
source ~/GestureDrone/install/setup.bash
ros2 pkg prefix gesture_to_flight
```

## Quick Alias Setup (Optional)

Add these to your `~/.bashrc` for faster commands:

```bash
# Add to ~/.bashrc
alias gesture-build="cd ~/GestureDrone && colcon build --packages-select gesture_to_flight"
alias gesture-source="source ~/GestureDrone/install/setup.bash"
alias gesture-start="gesture-source && ros2 launch gesture_to_flight gesture_drone.launch.py"
alias gesture-echo="gesture-source && ros2 topic echo /gesture_command"
```

Then use:
```bash
gesture-start      # Start the system
gesture-echo       # Watch commands in another terminal
gesture-build      # Rebuild package
```

## Complete Workflow Example

```bash
# Terminal 1 - Build and start the system
cd ~/GestureDrone
colcon build --packages-select gesture_to_flight
source ~/GestureDrone/install/setup.bash
ros2 launch gesture_to_flight gesture_drone.launch.py

# Terminal 2 (separate terminal) - Monitor commands
source ~/GestureDrone/install/setup.bash
ros2 topic echo /gesture_command

# Terminal 3 (separate terminal) - Monitor at Hz
source ~/GestureDrone/install/setup.bash
ros2 topic hz /gesture_command
```

## Keyboard Controls

In the gesture detection window:
- **q** - Quit and exit gracefully
- **Any other key** - Ignored (window just stays open)

## Expected Console Output

When system is running and gesture is detected:

```
[gesture_inference-1] [INFO] [1710000000.123456]: 🚁 COMMAND: PITCH_FORWARD (Gesture: Thumb_Up)
[flight_controller-2] [INFO] [1710000000.234567]: Received command: PITCH_FORWARD
[flight_controller-2] [INFO] [1710000000.345678]: Moving drone by (10.0, 0.0, 0.0)
```

The important part: **"🚁 COMMAND: PITCH_FORWARD"** - This shows the command translation is working!
