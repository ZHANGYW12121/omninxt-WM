#!/bin/bash
set -euo pipefail
docker run --rm --network host --name ego_planner_isaac \
  ego-planner-isaac:noetic \
  roslaunch ego_planner isaac_single.launch
