#x-axis → forward/backward

#y-axis → left/ right

#z-axis → up/down

#roll → rotation about x-axis

#pitch → rotation about y-axis

#yaw → rotation about z-axis


import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import tf_transformations
import math
import time


class FlightController(Node):
    def __init__(self):
        super().__init__('flight_controller')
        #whenever a message of type String is published to the topic /gesture_command, call the command_callback() function with the message
        self.sub = self.create_subscription(String, '/gesture_command', self.command_callback, 10)
        self.pos_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, 10)

        self.pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.current_pose = PoseStamped()
        self.get_logger().info("FlightController ready and listening to gestures")

    def command_callback(self, msg):
        cmd = msg.data
        self.get_logger().info(f"Received command: {cmd}")
        if cmd == "PITCH_FORWARD":
            self.pmove_increment(10, 0, 0)
        elif cmd == "PITCH_BACKWARD":
            self.move_increment(-10, 0, 0)
        elif cmd == "ROLL_LEFT":
            self.bank(90, "LEFT")
        elif cmd == "ROLL_RIGHT":
            self.bank(90, "RIGHT")
        elif cmd == "THROTTLE_UP":
            self.move_increment(0, 0, 10)
        elif cmd == "THROTTLE_DOWN":
            self.move_increment(0, 0, -10)
        elif cmd == "YAW_LEFT":
            self.yaw(90, "LEFT")
        elif cmd == "YAW_RIGHT":
            self.yaw(90, "RIGHT")

        #need to finish hovering
        elif cmd == "HOLD_POSITION":
            self.get_logger().info("Holding position.")
        #need to finish kill
        elif cmd == "KILL":
            self.get_logger().warn("Emergency stop triggered!")

        #need to finish speeding up and down
        elif cmd == "SPEED_UP":
            pass
        
        elif cmd == "SPEED_DOWN":
            pass

    def bank(self, angle_deg, direction):
        angle_rad = math.radians(angle_deg if direction == "RIGHT" else -angle_deg)
        self.get_logger().info(f"Banking {direction} by {angle_deg} degrees")

        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.pose.position = self.current_pose.pose.position

        # Convert roll angle into quaternion (roll, pitch, yaw)
        q = tf_transformations.quaternion_from_euler(angle_rad, 0, 0)
        new_pose.pose.orientation.x = q[0]
        new_pose.pose.orientation.y = q[1]
        new_pose.pose.orientation.z = q[2]
        new_pose.pose.orientation.w = q[3]


    def yaw(self, angle_deg, direction):
        angle_rad = math.radians(angle_deg if direction == "RIGHT" else -angle_deg)
        self.get_logger().info(f"Yaw {direction} by {angle_deg} degrees")

        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.pose.position = self.current_pose.pose.position

        # Convert yaw angle into quaternion (roll, pitch, yaw)
        q = tf_transformations.quaternion_from_euler(0, 0, angle_rad)
        new_pose.pose.orientation.x = q[0]
        new_pose.pose.orientation.y = q[1]
        new_pose.pose.orientation.z = q[2]
        new_pose.pose.orientation.w = q[3]

    def move_increment(self, x, y, z):
        new_loc = PoseStamped()
        new_loc.header.stamp = self.get_clock().now().to_msg()
        new_loc.pose.position.x = self.current_pose.pose.position.x + x
        new_loc.pose.position.y = self.current_pose.pose.position.y + y
        new_loc.pose.position.z = self.current_pose.pose.position.z + z

        new_loc.pose.orientation = self.current_pose.pose.orientation
        self.pos_pub.publish(new_loc)
        self.get_logger().info(f"Moving drone to ({x}, {y}, {z})")

    def pose_callback(self, msg):
        self.current_pose = msg

def main(args=None):
    rclpy.init(args=args)
    node = FlightController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()