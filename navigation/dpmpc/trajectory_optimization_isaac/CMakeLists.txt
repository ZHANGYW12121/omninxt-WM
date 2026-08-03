cmake_minimum_required(VERSION 3.0.2)
project(trajectory_optimization)

add_compile_options(-std=c++14 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  octomap_msgs
  octomap_ros
  roscpp
  std_msgs
  tf2
  tf2_geometry_msgs
)
find_package(ZLIB REQUIRED)

set(ACADO_ROOT "/opt/ACADOtoolkit" CACHE PATH "ACADO Toolkit source root")
set(QUADPROG_INCLUDE_DIR "/usr/local/include/QuadProg++" CACHE PATH "QuadProg++ headers")
set(QUADPROG_LIBRARY "/usr/local/lib/libquadprog.a" CACHE FILEPATH "QuadProg++ static library")
set(ACADO_LIBRARY "${ACADO_ROOT}/build/lib/libacado_toolkit_s.so" CACHE FILEPATH "ACADO shared library")

foreach(required_path
    "${ACADO_ROOT}/acado"
    "${QUADPROG_INCLUDE_DIR}"
    "${QUADPROG_LIBRARY}"
    "${ACADO_LIBRARY}")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "Required DPMPC dependency is missing: ${required_path}")
  endif()
endforeach()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs octomap_msgs octomap_ros roscpp std_msgs tf2 tf2_geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${QUADPROG_INCLUDE_DIR}
  SYSTEM ${ACADO_ROOT}
  ${ACADO_ROOT}/acado
  ${ACADO_ROOT}/acado/utils
)

add_library(polyTrajGen src/polyTrajGen.cpp)
target_link_libraries(polyTrajGen ${QUADPROG_LIBRARY})

add_library(mapModuleOctomap src/mapModuleOctomap.cpp)
target_link_libraries(mapModuleOctomap ${catkin_LIBRARIES})

add_library(mpcPlanner src/mpcPlanner.cpp)
target_link_libraries(mpcPlanner ${ACADO_LIBRARY})

add_executable(isaac_dpmpc_node src/isaac_dpmpc_node.cpp)
target_link_libraries(
  isaac_dpmpc_node
  ${catkin_LIBRARIES}
  ${ZLIB_LIBRARIES}
  mapModuleOctomap
  mpcPlanner
  polyTrajGen
)

install(TARGETS isaac_dpmpc_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
