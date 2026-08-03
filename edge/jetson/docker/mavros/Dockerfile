FROM omninxt/oak4p-noetic-arm64:jp512

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-noetic-mavros \
       ros-noetic-mavros-extras \
    && rm -rf /var/lib/apt/lists/*

