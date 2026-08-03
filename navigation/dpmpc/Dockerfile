FROM ego-planner-isaac:noetic

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    liboctomap-dev \
    nlohmann-json3-dev \
    ros-noetic-octomap-msgs \
    ros-noetic-octomap-ros \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY dependencies/ACADOtoolkit /opt/ACADOtoolkit
RUN cmake -S /opt/ACADOtoolkit -B /opt/ACADOtoolkit/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DACADO_WITH_EXAMPLES=OFF \
    && cmake --build /opt/ACADOtoolkit/build \
      --target acado_toolkit_s --parallel 2

COPY dependencies/QuadProgpp /opt/QuadProgpp
RUN cmake -S /opt/QuadProgpp -B /opt/QuadProgpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /opt/QuadProgpp/build --parallel 2 \
    && cmake --install /opt/QuadProgpp/build

WORKDIR /dpmpc_ws
COPY trajectory_optimization_isaac /dpmpc_ws/src/trajectory_optimization
RUN . /opt/ros/noetic/setup.sh \
    && catkin_make -DCMAKE_BUILD_TYPE=Release -j2

COPY docker/dpmpc_entrypoint.sh /opt/dpmpc/dpmpc_entrypoint.sh
RUN chmod +x /opt/dpmpc/dpmpc_entrypoint.sh

ENTRYPOINT ["/opt/dpmpc/dpmpc_entrypoint.sh"]
CMD ["rosrun", "trajectory_optimization", "isaac_dpmpc_node"]
