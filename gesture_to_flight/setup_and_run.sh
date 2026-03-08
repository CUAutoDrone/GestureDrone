#!/bin/bash
# Quick setup and test script for Gesture Drone system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Gesture Drone - Setup & Test"
echo "=================================="
echo ""

# Check if colcon is installed
if ! command -v colcon &> /dev/null; then
    echo "❌ colcon not found. Please install ROS 2."
    exit 1
fi

echo "📦 Building gesture_to_flight package..."
cd "$WORKSPACE_DIR"
colcon build --packages-select gesture_to_flight

echo ""
echo "✅ Build successful!"
echo ""
echo "=================================="
echo "To start the system, run:"
echo "=================================="
echo ""
echo "  source $WORKSPACE_DIR/install/setup.bash"
echo "  ros2 launch gesture_to_flight gesture_drone.launch.py"
echo ""
echo "Or run individual nodes:"
echo ""
echo "  # Terminal 1 - Gesture Detection"
echo "  source $WORKSPACE_DIR/install/setup.bash"
echo "  ros2 run gesture_to_flight gesture_inference"
echo ""
echo "  # Terminal 2 - Flight Control"
echo "  source $WORKSPACE_DIR/install/setup.bash"
echo "  ros2 run gesture_to_flight flight_controller"
echo ""
