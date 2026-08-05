cmake_minimum_required(VERSION 3.0.2)
project(quadcam_depth_est)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

if(NOT OpenCV_DIR)
  set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
endif()
SET("TENSORRT_UTILS_INC" "../tensorrt_utils/include/")
find_package(OpenCV 4.6 REQUIRED PATHS "${OpenCV_DIR}" NO_DEFAULT_PATH)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDA REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  std_msgs
  swarm_msgs
  cv_bridge
  message_generation
  camera_models
  message_filters
  image_transport
  pcl_ros
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES quadcam_depth_est
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_SOURCE_DIR}/../d2common/include
${TENSORRT_UTILS_INC}
${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIRS}
${YAML_CPP_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(../tensorrt_utils  tensorrt_utils)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/virtual_stereo.cpp
  src/hitnet.cpp
  src/camera_config_loader.cpp
  src/quadcam_depth_est_trt.cpp
)

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
  tensorrt_utils
  yaml-cpp
)

add_definitions("-Wall -g")

add_executable(${PROJECT_NAME}_node src/quadcam_depth_est_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
  tensorrt_utils
)

# add_executable(${PROJECT_NAME}_test src/quadcam_depth_est.cpp test/test_quadcam_depth_est.cpp)
# target_link_libraries(${PROJECT_NAME}_test
#   ${catkin_LIBRARIES}
#   ${YAML_CPP_LIBRARIES}
#   ${PROJECT_NAME}
#   ${d2frontend_LIBRARIES}
#   onnxruntime
# )
