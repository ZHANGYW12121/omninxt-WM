#!/usr/bin/env bash
set -Eeuo pipefail

container_name="omninxt_oak4p"

echo "=== Container ==="
docker ps --filter "name=^/${container_name}$" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

echo "=== USB topology ==="
lsusb -t

echo "=== OAK USB IDs ==="
lsusb -d 03e7: || true

echo "=== ROS topics ==="
docker exec "${container_name}" bash -lc \
  'source /opt/ros/noetic/setup.bash
   source /root/oak_ffc_ws/devel/setup.bash
   rostopic list | sort'

for topic in \
  /oak_ffc_4p/CAM_A \
  /oak_ffc_4p/CAM_B \
  /oak_ffc_4p/CAM_C \
  /oak_ffc_4p/CAM_D \
  /oak_ffc_4p/assemble_image
do
  if docker exec "${container_name}" bash -lc \
    "source /opt/ros/noetic/setup.bash
     source /root/oak_ffc_ws/devel/setup.bash
     rostopic type '${topic}'" >/dev/null 2>&1
  then
    echo "=== ${topic} ==="
    docker exec "${container_name}" bash -lc \
      "source /opt/ros/noetic/setup.bash
       source /root/oak_ffc_ws/devel/setup.bash
       rostopic type '${topic}'
       timeout 8 rostopic hz '${topic}'" || true
  fi
done
