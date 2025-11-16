// C++ port of gesture_to_flight/flight_instructions.py
// Listens on /gesture_command (std_msgs/String) and /mavros/local_position/pose (PoseStamped)
// Publishes setpoints to /mavros/setpoint_position/local (PoseStamped)

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
#include <algorithm>
#include <string>

class FlightController : public rclcpp::Node {
public:
  FlightController()
  : Node("flight_controller"), speed_factor_(1.0), min_speed_(0.1), max_speed_(3.0), pose_received_(false)
  {
    command_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/gesture_command", 10,
      std::bind(&FlightController::command_callback, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/mavros/local_position/pose", 10,
      std::bind(&FlightController::pose_callback, this, std::placeholders::_1));

    pos_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/mavros/setpoint_position/local", 10);

    RCLCPP_INFO(this->get_logger(), "FlightController ready and listening to gestures");
  }

private:
  void command_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    std::string cmd = msg->data;
    RCLCPP_INFO(this->get_logger(), "Received command: %s", cmd.c_str());

    if (cmd == "PITCH_FORWARD") {
      move_increment(10.0 * speed_factor_, 0.0, 0.0);
    } else if (cmd == "PITCH_BACKWARD") {
      move_increment(-10.0 * speed_factor_, 0.0, 0.0);
    } else if (cmd == "ROLL_LEFT") {
      bank(90.0, "LEFT");
    } else if (cmd == "ROLL_RIGHT") {
      bank(90.0, "RIGHT");
    } else if (cmd == "THROTTLE_UP") {
      move_increment(0.0, 0.0, 10.0 * speed_factor_);
    } else if (cmd == "THROTTLE_DOWN") {
      move_increment(0.0, 0.0, -10.0 * speed_factor_);
    } else if (cmd == "YAW_LEFT") {
      yaw(90.0, "LEFT");
    } else if (cmd == "YAW_RIGHT") {
      yaw(90.0, "RIGHT");
    } else if (cmd == "HOLD_POSITION") {
      RCLCPP_INFO(this->get_logger(), "Holding position.");
      if (pose_received_) {
        pos_pub_->publish(current_pose_);
      } else {
        RCLCPP_WARN(this->get_logger(), "No current pose available yet, cannot hold position.");
      }
    } else if (cmd == "KILL") {
      RCLCPP_WARN(this->get_logger(), "Emergency stop triggered!");
      // Note: for a real emergency stop, additional actions should be taken (land/disarm)
    } else if (cmd == "SPEED_UP") {
      speed_factor_ = std::min(speed_factor_ + 0.1, max_speed_);
      RCLCPP_INFO(this->get_logger(), "Speed increased → %.2fx", speed_factor_);
    } else if (cmd == "SPEED_DOWN") {
      speed_factor_ = std::max(speed_factor_ - 0.1, min_speed_);
      RCLCPP_INFO(this->get_logger(), "Speed decreased → %.2fx", speed_factor_);
    } else {
      RCLCPP_WARN(this->get_logger(), "Unknown command: %s", cmd.c_str());
    }
  }

  void bank(double angle_deg, const std::string & direction)
  {
    double angle_rad = angle_deg * M_PI / 180.0;
    if (direction != "RIGHT") angle_rad = -angle_rad;
    RCLCPP_INFO(this->get_logger(), "Banking %s by %.1f degrees", direction.c_str(), angle_deg);

    if (!pose_received_) {
      RCLCPP_WARN(this->get_logger(), "No current pose available yet; cannot bank.");
      return;
    }

    geometry_msgs::msg::PoseStamped new_pose;
    new_pose.header.stamp = this->now().to_msg();
    new_pose.header.frame_id = current_pose_.header.frame_id;
    new_pose.pose.position = current_pose_.pose.position;

    tf2::Quaternion q;
    // roll, pitch, yaw
    q.setRPY(angle_rad, 0.0, 0.0);
    new_pose.pose.orientation.x = q.x();
    new_pose.pose.orientation.y = q.y();
    new_pose.pose.orientation.z = q.z();
    new_pose.pose.orientation.w = q.w();

    pos_pub_->publish(new_pose);
  }

  void yaw(double angle_deg, const std::string & direction)
  {
    double angle_rad = angle_deg * M_PI / 180.0;
    if (direction != "RIGHT") angle_rad = -angle_rad;
    RCLCPP_INFO(this->get_logger(), "Yaw %s by %.1f degrees", direction.c_str(), angle_deg);

    if (!pose_received_) {
      RCLCPP_WARN(this->get_logger(), "No current pose available yet; cannot yaw.");
      return;
    }

    geometry_msgs::msg::PoseStamped new_pose;
    new_pose.header.stamp = this->now().to_msg();
    new_pose.header.frame_id = current_pose_.header.frame_id;
    new_pose.pose.position = current_pose_.pose.position;

    tf2::Quaternion q;
    // roll, pitch, yaw
    q.setRPY(0.0, 0.0, angle_rad);
    new_pose.pose.orientation.x = q.x();
    new_pose.pose.orientation.y = q.y();
    new_pose.pose.orientation.z = q.z();
    new_pose.pose.orientation.w = q.w();

    pos_pub_->publish(new_pose);
  }

  void move_increment(double x, double y, double z)
  {
    if (!pose_received_) {
      RCLCPP_WARN(this->get_logger(), "No current pose available yet; cannot move.");
      return;
    }

    geometry_msgs::msg::PoseStamped new_loc;
    new_loc.header.stamp = this->now().to_msg();
    new_loc.header.frame_id = current_pose_.header.frame_id;
    new_loc.pose.position.x = current_pose_.pose.position.x + x;
    new_loc.pose.position.y = current_pose_.pose.position.y + y;
    new_loc.pose.position.z = current_pose_.pose.position.z + z;
    new_loc.pose.orientation = current_pose_.pose.orientation;

    pos_pub_->publish(new_loc);
    RCLCPP_INFO(this->get_logger(), "Moving drone by (%.2f, %.2f, %.2f)", x, y, z);
  }

  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    current_pose_ = *msg;
    pose_received_ = true;
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pos_pub_;

  geometry_msgs::msg::PoseStamped current_pose_;
  bool pose_received_;

  double speed_factor_;
  double min_speed_;
  double max_speed_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FlightController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
