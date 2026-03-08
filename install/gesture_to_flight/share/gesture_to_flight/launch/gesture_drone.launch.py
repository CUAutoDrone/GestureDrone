from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for Gesture-Based Drone Control System
    
    This launch file starts:
    1. gesture_inference - Captures webcam feed, detects hand gestures, publishes commands
    2. flight_controller - Listens to gesture commands and publishes drone flight commands
    
    Uses Python venv at /home/etienne/gesture_env for proper dependency isolation
    """
    
    # Use venv Python interpreter for proper dependencies
    python_venv = "/home/etienne/gesture_env/bin/python3"
    
    # Gesture Inference Node - reads webcam, detects gestures, publishes commands
    gesture_inference_node = Node(
        package='gesture_to_flight',
        executable='gesture_inference',
        name='gesture_inference',
        output='screen',
        emulate_tty=True,
        prefix=[python_venv],
    )
    
    # Flight Controller Node - receives gesture commands and publishes to drone
    flight_controller_node = Node(
        package='gesture_to_flight',
        executable='flight_controller',
        name='flight_controller',
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        gesture_inference_node,
        flight_controller_node,
    ])
