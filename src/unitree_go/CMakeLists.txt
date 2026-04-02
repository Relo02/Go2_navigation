cmake_minimum_required(VERSION 3.8)
project(unitree_go)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Core state messages used by go2_hw_bridge
  "msg/IMUState.msg"
  "msg/BmsState.msg"
  "msg/SportModeState.msg"
  "msg/TimeSpec.msg"
  # Low-level control (included for completeness; not used by nav stack)
  "msg/MotorCmd.msg"
  "msg/MotorState.msg"
  "msg/BmsCmd.msg"
  "msg/LowCmd.msg"
  "msg/LowState.msg"
  "msg/MotorCmds.msg"
  "msg/MotorStates.msg"
  "msg/PathPoint.msg"
  "msg/SportModeCmd.msg"
  "msg/WirelessController.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()
