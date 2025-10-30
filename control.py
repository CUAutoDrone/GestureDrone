import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
import time

class ArduPilotController(Node):
    def __init__(self):
        super().__init__('ardupilot_controller')

        # Publisher to send target positions
        self.pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        # Clients for arming and mode
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Give ROS2 time to initialize connections
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')

        # Define target altitude and square size
        self.target_altitude = 2.0
        self.square_size = 2.0
        self.rate_hz = 20.0  # Must be >10Hz for GUIDED

    def arm_drone(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().success:
            self.get_logger().info('Drone armed successfully!')
        else:
            self.get_logger().error('Failed to arm drone')

    def set_guided_mode(self):
        req = SetMode.Request()
        req.custom_mode = 'GUIDED'
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().mode_sent:
            self.get_logger().info('GUIDED mode set!')
        else:
            self.get_logger().error('Failed to set GUIDED mode')

    def send_position(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.pos_pub.publish(msg)

    def takeoff(self):
        self.get_logger().info('Taking off...')
        for _ in range(int(self.rate_hz * 5)):  # Send setpoints for 5 seconds
            self.send_position(0.0, 0.0, self.target_altitude)
            time.sleep(1.0 / self.rate_hz)
        self.get_logger().info(f'Reached altitude {self.target_altitude}m')

    def fly_square(self):
        self.get_logger().info('Flying square...')
        coords = [
            (self.square_size, 0.0, self.target_altitude),
            (self.square_size, self.square_size, self.target_altitude),
            (0.0, self.square_size, self.target_altitude),
            (0.0, 0.0, self.target_altitude)
        ]
        for x, y, z in coords:
            for _ in range(int(self.rate_hz * 3)):  # 3 seconds per leg
                self.send_position(x, y, z)
                time.sleep(1.0 / self.rate_hz)
        self.get_logger().info('Completed square trajectory!')

    def land(self):
        self.get_logger().info('Landing...')
        req = SetMode.Request()
        req.custom_mode = 'LAND'
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().mode_sent:
            self.get_logger().info('Landing mode activated!')
        else:
            self.get_logger().error('Failed to initiate landing')


def main(args=None):
    rclpy.init(args=args)
    controller = ArduPilotController()

    # Start streaming setpoints before arming or mode switch
    controller.get_logger().info('Priming GUIDED mode by sending setpoints...')
    for _ in range(int(controller.rate_hz * 3)):  # 3 seconds of pre-stream
        controller.send_position(0.0, 0.0, controller.target_altitude)
        time.sleep(1.0 / controller.rate_hz)
    # Now switch modes and arm
    controller.arm_drone()
    controller.set_guided_mode()
    controller.takeoff()
    controller.fly_square()
    controller.land()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
