FROM hkustswarm/d2slam:omni-depth-x86

ARG DEBIAN_FRONTEND=noninteractive
ARG OMNIDEPTH_BUILD_PROXY
ENV HTTP_PROXY=${OMNIDEPTH_BUILD_PROXY} \
    HTTPS_PROXY=${OMNIDEPTH_BUILD_PROXY} \
    http_proxy=${OMNIDEPTH_BUILD_PROXY} \
    https_proxy=${OMNIDEPTH_BUILD_PROXY}
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# Keep ROS Noetic on Ubuntu 20.04 while updating the CUDA/TensorRT userspace
# for the server's Ada GPUs (RTX 4090 D, SM 8.9).
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-compiler-12-8 \
        cuda-cudart-dev-12-8 \
        libcublas-dev-12-8 \
        libnpp-dev-12-8 \
        libnvinfer-headers-dev=10.9.0.34-1+cuda12.8 \
        libnvinfer-headers-plugin-dev=10.9.0.34-1+cuda12.8 \
        libnvinfer10=10.9.0.34-1+cuda12.8 \
        libnvinfer-dev=10.9.0.34-1+cuda12.8 \
        libnvinfer-plugin10=10.9.0.34-1+cuda12.8 \
        libnvinfer-plugin-dev=10.9.0.34-1+cuda12.8 \
        libnvonnxparsers10=10.9.0.34-1+cuda12.8 \
        libnvonnxparsers-dev=10.9.0.34-1+cuda12.8 && \
    rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb && \
    ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcufft-dev-12-8 \
        libcusolver-dev-12-8 && \
    rm -rf /var/lib/apt/lists/*

# OpenCV 4.6 uses texture-reference APIs removed by CUDA 12. Build the current
# API-compatible OpenCV release specifically for Ada SM 8.9.
RUN wget -q https://github.com/opencv/opencv/archive/4.12.0.tar.gz -O /tmp/opencv.tar.gz && \
    wget -q https://github.com/opencv/opencv_contrib/archive/4.12.0.tar.gz -O /tmp/opencv_contrib.tar.gz && \
    tar -xzf /tmp/opencv.tar.gz -C / && \
    tar -xzf /tmp/opencv_contrib.tar.gz -C / && \
    rm /tmp/opencv.tar.gz /tmp/opencv_contrib.tar.gz && \
    mkdir /opencv-4.12.0/build && \
    cd /opencv-4.12.0/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
      -DCUDA_ARCH_BIN=8.9 \
      -DCUDA_ARCH_PTX=8.9 \
      -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DWITH_CUBLAS=ON -DWITH_TBB=ON \
      -DOPENCV_DNN_CUDA=OFF -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-4.12.0/modules \
      -DBUILD_EXAMPLES=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_LIST=calib3d,features2d,highgui,dnn,imgproc,imgcodecs,cudev,cudaoptflow,cudaimgproc,cudalegacy,cudaarithm,cudacodec,cudastereo,cudafeatures2d,xfeatures2d,tracking,stereo,aruco,videoio,ccalib && \
    make -j$(nproc) && make install && ldconfig

RUN sed -i 's#developer.download.nvidia.com#developer.download.nvidia.cn#g' \
        /etc/apt/sources.list.d/cuda-ubuntu2004-x86_64.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcusparse-dev-12-8 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends libnvjitlink-dev-12-8 && \
    rm -rf /var/lib/apt/lists/*

# TensorRT 10.9 compatibility patches from the migrated source tree.
COPY D2SLAM/tensorrt_utils/include/tensorrt_utils/buffers.h \
     /root/swarm_ws/src/D2SLAM/tensorrt_utils/include/tensorrt_utils/buffers.h
COPY D2SLAM/tensorrt_utils/include/tensorrt_utils/common.h \
     /root/swarm_ws/src/D2SLAM/tensorrt_utils/include/tensorrt_utils/common.h
COPY D2SLAM/d2frontend/src/CNN/superpoint.cpp \
     /root/swarm_ws/src/D2SLAM/d2frontend/src/CNN/superpoint.cpp
COPY D2SLAM/d2frontend/CMakeLists.txt \
     /root/swarm_ws/src/D2SLAM/d2frontend/CMakeLists.txt
COPY D2SLAM/quadcam_depth_est/src/hitnet.cpp \
     /root/swarm_ws/src/D2SLAM/quadcam_depth_est/src/hitnet.cpp
COPY D2SLAM/quadcam_depth_est/src/quadcam_depth_est_trt.cpp \
     /root/swarm_ws/src/D2SLAM/quadcam_depth_est/src/quadcam_depth_est_trt.cpp
COPY D2SLAM/quadcam_depth_est/src/virtual_stereo.cpp \
     /root/swarm_ws/src/D2SLAM/quadcam_depth_est/src/virtual_stereo.cpp
COPY D2SLAM/quadcam_depth_est/include/quadcam_depth_est_trt.hpp \
     /root/swarm_ws/src/D2SLAM/quadcam_depth_est/include/quadcam_depth_est_trt.hpp
COPY D2SLAM/quadcam_depth_est/src/quadcam_depth_est_trt.cpp \
     /root/swarm_ws/src/D2SLAM/D2SLAM/quadcam_depth_est/src/quadcam_depth_est_trt.cpp
COPY D2SLAM/quadcam_depth_est/include/quadcam_depth_est_trt.hpp \
     /root/swarm_ws/src/D2SLAM/D2SLAM/quadcam_depth_est/include/quadcam_depth_est_trt.hpp
COPY D2SLAM/quadcam_depth_est/include/virtual_stereo.hpp \
     /root/swarm_ws/src/D2SLAM/quadcam_depth_est/include/virtual_stereo.hpp
COPY D2SLAM/d2common/include/d2common/fisheye_undistort.h \
     /root/swarm_ws/src/D2SLAM/d2common/include/d2common/fisheye_undistort.h

RUN cd /root/swarm_ws && \
    source /opt/ros/noetic/setup.bash && \
    catkin clean -y && \
    catkin config -DCMAKE_BUILD_TYPE=Release \
      --cmake-args -DONNXRUNTIME_LIB_DIR=/usr/local/lib \
      -DONNXRUNTIME_INC_DIR=/usr/local/include && \
    catkin build -j8 -p8

ENV HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy=
