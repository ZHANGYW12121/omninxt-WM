#include "camera_config_loader.hpp"

#include <cstdlib>
#include <iostream>

#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <Eigen/Core>

namespace D2QuadCamDepthEst {

std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(
    const std::string& camera_name, const YAML::Node& config,
    int32_t extrinsic_parameter_type) {
  camodocal::CameraPtr camera;
  const std::string camera_model = config["camera_model"].as<std::string>();
  const std::string distortion_model =
      config["distortion_model"].as<std::string>();

  if (camera_model == "omni" && distortion_model == "radtan") {
    const int width = config["resolution"][0].as<int>();
    const int height = config["resolution"][1].as<int>();
    const double xi = config["intrinsics"][0].as<double>();
    const double gamma1 = config["intrinsics"][1].as<double>();
    const double gamma2 = config["intrinsics"][2].as<double>();
    const double u0 = config["intrinsics"][3].as<double>();
    const double v0 = config["intrinsics"][4].as<double>();
    const double k1 = config["distortion_coeffs"][0].as<double>();
    const double k2 = config["distortion_coeffs"][1].as<double>();
    const double p1 = config["distortion_coeffs"][2].as<double>();
    const double p2 = config["distortion_coeffs"][3].as<double>();
    camera = camodocal::CataCameraPtr(new camodocal::CataCamera(
        camera_name, width, height, xi, k1, k2, p1, p2, gamma1, gamma2, u0,
        v0));
  } else if (camera_model == "pinhole" && distortion_model == "radtan") {
    const int width = config["resolution"][0].as<int>();
    const int height = config["resolution"][1].as<int>();
    const double fx = config["intrinsics"][0].as<double>();
    const double fy = config["intrinsics"][1].as<double>();
    const double cx = config["intrinsics"][2].as<double>();
    const double cy = config["intrinsics"][3].as<double>();
    const double k1 = config["distortion_coeffs"][0].as<double>();
    const double k2 = config["distortion_coeffs"][1].as<double>();
    const double p1 = config["distortion_coeffs"][2].as<double>();
    const double p2 = config["distortion_coeffs"][3].as<double>();
    camera = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera(
        camera_name, width, height, k1, k2, p1, p2, fx, fy, cx, cy));
  } else {
    std::cerr << "Unsupported camera model for " << camera_name << ": "
              << camera_model << "-" << distortion_model << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Eigen::Matrix4d T;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      T(i, j) = config["T_cam_imu"][i][j].as<double>();
    }
  }

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  // Preserve the official D2FrontendParams convention exactly.  The current
  // OmniDepth call uses the default mode (1); geometry is not changed here.
  if (extrinsic_parameter_type == 0) {
    R = T.block<3, 3>(0, 0).transpose();
    t = -R * T.block<3, 1>(0, 3);
  } else {
    R = T.block<3, 3>(0, 0);
    t = T.block<3, 1>(0, 3);
  }
  Swarm::Pose pose(R, t);
  std::cout << "T_cam_imu (" << camera_name << "):\n" << T << std::endl;
  std::cout << "pose:\n" << pose.toStr() << std::endl;
  return std::make_pair(camera, pose);
}

}  // namespace D2QuadCamDepthEst
