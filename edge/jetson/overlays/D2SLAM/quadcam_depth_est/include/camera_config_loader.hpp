#pragma once

#include <string>
#include <utility>

#include <camodocal/camera_models/Camera.h>
#include <swarm_msgs/Pose.h>
#include <yaml-cpp/yaml.h>

namespace D2QuadCamDepthEst {

// Standalone copy of the camera parser used by D2FrontendParams.  Keeping this
// small parser local avoids pulling PyTorch, ONNX Runtime and FAISS into the
// independent OmniDepth node.
std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(
    const std::string& camera_name, const YAML::Node& config,
    int32_t extrinsic_parameter_type = 1);

}  // namespace D2QuadCamDepthEst
