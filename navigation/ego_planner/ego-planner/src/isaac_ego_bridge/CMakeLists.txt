cmake_minimum_required(VERSION 3.0.2)
project(isaac_ego_bridge)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/isaac_udp_ros_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
