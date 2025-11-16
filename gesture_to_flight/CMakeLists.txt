cmake_minimum_required(VERSION 3.5)
project(gesture_to_flight)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)

add_executable(flight_instructions src/flight_instructions.cpp)

ament_target_dependencies(
  flight_instructions
  rclcpp
  std_msgs
  geometry_msgs
  tf2
)

install(TARGETS
  flight_instructions
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
