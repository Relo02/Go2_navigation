cmake_minimum_required(VERSION 3.8)
project(graphnav_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/UUID.msg"
  "msg/KeyValue.msg"
  "msg/NodeTraversabilityProperties.msg"
  "msg/EdgeTraversability.msg"
  "msg/Node.msg"
  "msg/Edge.msg"
  "msg/NavigationGraph.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
