cmake_minimum_required(VERSION 3.8)
project(robot_safety)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python node as a runnable executable under lib/robot_safety/
install(PROGRAMS
  scripts/velocity_limiter_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME velocity_limiter
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
