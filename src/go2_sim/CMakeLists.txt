cmake_minimum_required(VERSION 3.8)
project(go2_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config meshes launch worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
