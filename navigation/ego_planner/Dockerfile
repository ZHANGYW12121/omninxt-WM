FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libarmadillo-dev python3-numpy python3-rosdep \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY ego-planner/src /ws/src
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCMAKE_BUILD_TYPE=Release -j2

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "ego_planner", "isaac_single.launch"]
