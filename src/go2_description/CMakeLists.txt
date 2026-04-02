cmake_minimum_required(VERSION 3.8)
project(go2_description)

find_package(ament_cmake REQUIRED)

# Install URDF, mesh, and config directories for runtime use
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
