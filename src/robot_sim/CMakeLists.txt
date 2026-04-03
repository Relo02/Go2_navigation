cmake_minimum_required(VERSION 3.8)
project(robot_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config worlds rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/cloud_self_filter.py
  scripts/mock_nav_graph_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
