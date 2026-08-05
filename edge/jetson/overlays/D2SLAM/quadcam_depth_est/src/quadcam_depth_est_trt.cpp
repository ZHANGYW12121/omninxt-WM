#include "quadcam_depth_est_trt.hpp"
#include <d2common/fisheye_undistort.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <NvInferRuntime.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <cmath>
#include <chrono>

#include "camera_config_loader.hpp"
#include "pcl_utils.hpp"


// namespace D2FrontEnd {
//     std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config, int32_t extrinsic_parameter_type = 1);
// };

namespace D2QuadCamDepthEst
{

cv::Mat quadReadVingette(const std::string & mask_file, double avg_brightness) {
    cv::Mat photometric_inv;
    cv::Mat photometric_calib = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
    std::cout << photometric_calib.type() << std::endl;
    if (photometric_calib.type() == CV_8U) {
        photometric_calib.convertTo(photometric_calib, CV_32FC1, 1.0/255.0);
    } else if (photometric_calib.type() == CV_16U) {
        photometric_calib.convertTo(photometric_calib, CV_32FC1, 1.0/65535.0);
    }
    cv::divide(avg_brightness, photometric_calib, photometric_inv);
    return photometric_inv;
}

QuadcamDepthEstTrt::QuadcamDepthEstTrt(ros::NodeHandle & nh):nh_(nh){
  std::string config_file_path;
  nh.getParam("depth_config",config_file_path);
  nh.getParam("show",show_);
  printf("[QuadCamDepthEstTrt]:read config from:%s show: %d\n",config_file_path.c_str(),show_);
  YAML::Node config = YAML::LoadFile(config_file_path);
  std::string config_dir = config_file_path.substr(0,config_file_path.find_last_of("/"));
  this->enable_texture_ = config["enable_texture"].as<bool>();
  this->pixel_step_ = config["pixel_step"].as<int>();
  this->image_step_ = config["image_step"].as<int>();
  this->min_z_ = config["min_z"].as<double>();
  this->max_z_ = config["max_z"].as<double>();
  this->width_ = config["width"].as<int>();
  this->height_ = config["height"].as<int>();
  this->fps_ = config["fps"].as<int>();
  this->raw_image_process_rate_ = std::make_unique<ros::Rate>(ros::Rate(this->fps_));
  this->inference_rate_ = std::make_unique<ros::Rate>(ros::Rate(this->fps_));
  // Publishing is event-gated by output_ready_. Poll faster than the target
  // output rate so a completed inference is not held for up to 100 ms or
  // overwritten by the next result.
  this->publish_rate_ = std::make_unique<ros::Rate>(ros::Rate(100.0));
  this->cnn_input_rgb_ = config["cnn_input_rgb"].as<bool>();
  if (config["use_cuda_rectification"].IsDefined()) {
    this->use_cuda_rectification_ =
        config["use_cuda_rectification"].as<bool>();
  }
  if (config["enable_photometric_calib"].IsDefined()) {
    this->enable_photometric_calib_ =
        config["enable_photometric_calib"].as<bool>();
  }
  if (config["enable_disparity_filter"].IsDefined()) {
    this->enable_disparity_filter_ =
        config["enable_disparity_filter"].as<bool>();
  }
  if (config["publish_debug_clouds"].IsDefined()) {
    this->publish_debug_clouds_ =
        config["publish_debug_clouds"].as<bool>();
  }
  if (config["enable_pointcloud_output"].IsDefined()) {
    this->enable_pointcloud_output_ =
        config["enable_pointcloud_output"].as<bool>();
  }
  nh.param("publish_pose_anchor_views", this->publish_pose_anchor_views_,
           this->publish_pose_anchor_views_);
  nh.param("pose_anchor_width", this->pose_anchor_width_,
           this->pose_anchor_width_);
  nh.param("pose_anchor_height", this->pose_anchor_height_,
           this->pose_anchor_height_);
  nh.param("pose_anchor_fov_deg", this->pose_anchor_fov_deg_,
           this->pose_anchor_fov_deg_);
  // The dedicated skeleton launch disables dense PCL construction while
  // preserving the normal OmniDepth launch default and all depth topics.
  nh.param("enable_pointcloud_output", this->enable_pointcloud_output_,
           this->enable_pointcloud_output_);
  if (!this->enable_pointcloud_output_) {
    this->publish_debug_clouds_ = false;
  }
  if (config["min_disparity"].IsDefined()) {
    this->min_disparity_ = config["min_disparity"].as<double>();
  }
  if (config["max_disparity"].IsDefined()) {
    this->max_disparity_ = config["max_disparity"].as<double>();
  }
  if (config["max_median_deviation"].IsDefined()) {
    this->max_median_deviation_ =
        config["max_median_deviation"].as<double>();
  }
  if (config["max_photometric_error"].IsDefined()) {
    this->max_photometric_error_ =
        config["max_photometric_error"].as<double>();
  }
  if (config["min_texture_gradient"].IsDefined()) {
    this->min_texture_gradient_ =
        config["min_texture_gradient"].as<double>();
  }
  if (config["valid_roi_margin"].IsDefined()) {
    this->valid_roi_margin_ = config["valid_roi_margin"].as<int>();
  }
  if (config["min_valid_component_size"].IsDefined()) {
    this->min_valid_component_size_ =
        config["min_valid_component_size"].as<int>();
  }

  this->loadVirtualCameras(config,config_dir);
  if(config["image_topic"].IsDefined()){
    this->image_topic_ = config["image_topic"].as<std::string>();
  }
  if(config["image_format"].IsDefined()){
    this->image_format_ = config["image_format"].as<std::string>();
  }
  // HITNet is the production disparity backend.
  this->onnx_path_ = config["onnx_path"].as<std::string>();
  this->trt_engine_path_ = config["trt_engine_path"].as<std::string>();
  this->hitnet_ = std::make_unique<TensorRTHitnet::HitnetTrt>(true);
  const int hitnet_status =
      this->hitnet_->init(onnx_path_, trt_engine_path_, 4);
  if (hitnet_status != 0) {
    throw std::runtime_error("HITNet initialization failed: " +
                             std::to_string(hitnet_status));
  }

  //subscribe
  image_transport::TransportHints hints(this->image_format_, ros::TransportHints().tcpNoDelay(true));
  image_transport_ = new image_transport::ImageTransport(nh_);
  image_sub_ = image_transport_->subscribe(this->image_topic_, 1, &QuadcamDepthEstTrt::quadcamImageCb, this, hints);
  if (enable_pointcloud_output_) {
    if (enable_texture_){
      pcl_color_ = new PointCloudRGB();
      pcl_color_->points.reserve(virtual_stereos_.size() * width_ * height_);
    } else {
      pcl_ = new PointCloud();
      pcl_->points.reserve(virtual_stereos_.size() * width_ * height_);
    }
    if (publish_debug_clouds_) {
      pcl_debug_color_ = new PointCloudRGB();
      pcl_debug_color_->points.reserve(
          virtual_stereos_.size() * width_ * height_);
    }
  }

  //publisher
  if (enable_pointcloud_output_) {
    this->pub_pcl_ =
        nh_.advertise<sensor_msgs::PointCloud2>(kPointCloudTopic_, 1);
  }
  for (int i = 0; i < kCamerasNum; ++i) {
    const std::string prefix =
        "/depth_estimation/stereo_" + std::to_string(i);
    const std::string input_prefix =
        "/depth_estimation/input_stereo_" + std::to_string(i);
    this->pub_input_rect_left_[i] =
        image_transport_->advertise(input_prefix + "/left", 1);
    this->pub_input_rect_right_[i] =
        image_transport_->advertise(input_prefix + "/right", 1);
    this->pub_pose_anchor_[i] = image_transport_->advertise(
        "/depth_estimation/pose_anchor_" + std::to_string(i), 1);
    this->pub_rect_left_[i] =
        image_transport_->advertise(prefix + "/left", 1);
    this->pub_rect_right_[i] =
        image_transport_->advertise(prefix + "/right", 1);
    this->pub_disparity_[i] =
        image_transport_->advertise(prefix + "/disparity", 1);
    this->pub_depth_[i] =
        image_transport_->advertise(prefix + "/depth", 1);
  }
  this->pub_pose_anchor_mosaic_ = image_transport_->advertise(
      "/depth_estimation/pose_anchor_mosaic", 1);
  this->pub_pose_stereo_mosaic_ = image_transport_->advertise(
      "/depth_estimation/pose_stereo_mosaic", 1);
  if (publish_debug_clouds_) {
    this->pub_pcl_colored_ =
        nh_.advertise<sensor_msgs::PointCloud2>(
            "/depth_estimation/pointcloud_sectors", 1);
    for (int i = 0; i < kCamerasNum; ++i) {
      this->pub_sector_pcl_[i] =
          nh_.advertise<sensor_msgs::PointCloud2>(
              "/depth_estimation/pointcloud_sector_" + std::to_string(i), 1);
    }
  }
  ROS_INFO(
      "Disparity filter: enabled=%d disp=[%.2f, %.2f] "
      "median_dev=%.2f photo_error=%.1f texture=%.1f roi_margin=%d "
      "min_component=%d photometric_calib=%d",
      enable_disparity_filter_, min_disparity_, max_disparity_,
      max_median_deviation_, max_photometric_error_,
      min_texture_gradient_, valid_roi_margin_,
      min_valid_component_size_, enable_photometric_calib_);
  ROS_INFO("Pointcloud output: enabled=%d", enable_pointcloud_output_);
  printf("QuadcamDepthEtsTrt constructed\n");
};

QuadcamDepthEstTrt::~QuadcamDepthEstTrt(){
  if(pcl_ != nullptr){
    delete pcl_;
    pcl_ = nullptr;
  }
  if(pcl_color_ != nullptr){
    delete pcl_color_;
    pcl_color_ = nullptr;
  }
  if(pcl_debug_color_ != nullptr){
    delete pcl_debug_color_;
    pcl_debug_color_ = nullptr;
  }
  if (this->hitnet_ != nullptr){
    this->hitnet_ = nullptr;
  }
};

void QuadcamDepthEstTrt::loadVirtualCameras(YAML::Node & config, std::string configPath){
  float avg_brightness = config["avg_brightness"].as<float>();
  const int32_t extrinsic_parameter_type =
      config["extrinsic_parameter_type"].IsDefined()
          ? config["extrinsic_parameter_type"].as<int32_t>()
          : 0;
  if (extrinsic_parameter_type != 0 && extrinsic_parameter_type != 1) {
    throw std::runtime_error(
        "extrinsic_parameter_type must be 0 (Kalibr/OmniNxt) or 1");
  }
  printf("[QuadcamDepthEstTrt]: extrinsic_parameter_type=%d\n",
         extrinsic_parameter_type);
  std::string photometric_calib_path = config["photometric_calib_path"].as<std::string>();
  //Read photometric calibration masks
  if (enable_photometric_calib_ &&
      access(photometric_calib_path.c_str(),F_OK) == 0){
    printf("[QuadcamDepthEstTrt]: loadVirtualCameras from %s\n",photometric_calib_path.c_str());
    for(int i=0 ; i < kCamerasNum ; i++){
      std::string mask_file = photometric_calib_path + "/" + std::string("cam_") + std::to_string(i) + std::string("_vig_mask.png");//search image "cam_i_vig_mask.png"
      if(access(mask_file.c_str(),F_OK) == 0){
        photometric_inv_vingette_[i] = quadReadVingette(mask_file, avg_brightness);
        printf("[QuadcamDepthEstTrt]: read vignette mask from %s\n",mask_file.c_str());
      } else {
        photometric_inv_vingette_[i] = cv::Mat();
      }
    }
  } else {
    if (!enable_photometric_calib_) {
      printf("[QuadcamDepthEstTrt]: photometric calibration disabled\n");
    }
    for(int i=0 ; i < kCamerasNum ; i++){
      photometric_inv_vingette_[i] = cv::Mat();
    }
  }
  //Read fisheye cameras intrinsic and extrinsic parameters
  std::string cam_calib_file_path = config["cam_calib_file_path"].as<std::string>();
  printf("[QuadcamDepthEstTrt]: load camera calibration from %s\n",cam_calib_file_path.c_str());
  if(access(cam_calib_file_path.c_str(),R_OK) == 0){
    YAML::Node fisheye_configs = YAML::LoadFile(cam_calib_file_path);
    int32_t photometric_inv_idx = 0;
    for (const auto & cam_para : fisheye_configs){
      std::string camera_name = cam_para.first.as<std::string>();
      printf("[QuadcamDepthEstTrt] Load camera %s\n", camera_name.c_str());
      //fisheye camera parameters
      const YAML::Node & camera_parameters = cam_para.second;
      auto cam_model = readCameraConfig(
          camera_name, camera_parameters, extrinsic_parameter_type);
      this->raw_cameras_.push_back(cam_model.first);
      //load distotors and photometric calibration
      double fov = config["fov"].as<double>();
      if(photometric_inv_idx >=4 || photometric_inv_idx < 0){
          photometric_inv_idx = 0;
      }
      printf("[Debug ]undistortor matrix init with size width:%d height:%d\n",this->width_,this->height_);
      this->undistortors_.push_back(new D2Common::FisheyeUndist(cam_model.first, 0, fov, true,
          D2Common::FisheyeUndist::UndistortPinhole2, this->width_, this->height_, photometric_inv_vingette_[photometric_inv_idx]));
      if (publish_pose_anchor_views_) {
        this->undistortors_.back()->prepare_center_map(
            pose_anchor_width_, pose_anchor_height_, pose_anchor_fov_deg_);
      }
      printf("[Debug] undistorter width and height:%d %d\n",this->width_,this->height_);
      photometric_inv_idx++;
      //set extrinsic paratmers
      this->raw_cam_extrinsics_.push_back(cam_model.second);
    }

    //Create virtual stereo
    for(const auto & vstereos:config["stereos"]){
      auto stereo_node =  vstereos.second;
      std::string stereo_name = vstereos.first.as<std::string>();
      int cam_idx_l = stereo_node["cam_idx_l"].as<int>();
      int cam_idx_r = stereo_node["cam_idx_r"].as<int>();
      int idx_l = stereo_node["idx_l"].as<int>();
      int idx_r = stereo_node["idx_r"].as<int>();
      std::string stereo_calib_file = stereo_node["stereo_config"].as<std::string>();
      Swarm::Pose baseline;
      YAML::Node stereo_calib = YAML::LoadFile(stereo_calib_file);
      Matrix4d T;
      for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
              T(i, j) = stereo_calib["cam1"]["T_cn_cnm1"][i][j].as<double>();
          }
      }
      baseline = Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
      auto KD0 = intrinsicsFromNode(stereo_calib["cam0"]);
      auto KD1 = intrinsicsFromNode(stereo_calib["cam1"]);

      printf("[QuadCamDepthEst] Load stereo %s, stereo %d(%d):%d(%d) baseline: %s\n",
          stereo_name.c_str(), cam_idx_l, idx_l, cam_idx_r, idx_r, baseline.toStr().c_str());
      auto stereo = new VirtualStereo(cam_idx_l, cam_idx_r, baseline,
          undistortors_[cam_idx_l], undistortors_[cam_idx_r], idx_l, idx_r);
      auto att = undistortors_[cam_idx_l]->t[idx_l];
      stereo->enable_texture = enable_texture_;
      stereo->initRecitfy(baseline, KD0.first, KD0.second, KD1.first, KD1.second);
      // reprojectImageTo3D returns points in the rectified-left frame.
      // R1 maps the original virtual-left frame into that rectified frame,
      // hence R1^T is required before virtual-left -> fisheye -> IMU.
      const cv::Mat & rectification = stereo->getLeftRectificationRotation();
      if (rectification.rows != 3 || rectification.cols != 3 ||
          rectification.type() != CV_64FC1) {
        delete stereo;
        throw std::runtime_error("Invalid left rectification rotation R1");
      }
      Matrix3d rectified_to_virtual;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          rectified_to_virtual(row, col) =
              rectification.at<double>(col, row);
        }
      }
      stereo->extrinsic =
          raw_cam_extrinsics_[cam_idx_l] *
          Swarm::Pose(att, Vector3d(0, 0, 0)) *
          Swarm::Pose(rectified_to_virtual, Vector3d(0, 0, 0));
      const Vector3d optical_axis =
          stereo->extrinsic.R() * Vector3d(0, 0, 1);
      printf(
          "[QuadCamDepthEst] Stereo %d optical axis in imu: "
          "[%+.6f, %+.6f, %+.6f]\n",
          stereo->stereo_id, optical_axis.x(), optical_axis.y(),
          optical_axis.z());
      virtual_stereos_.emplace_back(stereo);
    }
    printf("[QuadCamDepthEst] Init virtual cameras successfully\n");
  } else {
    printf("QuadcamDepthEstTrt][Failed]: read camera calibration from %s\n",cam_calib_file_path.c_str());
    return ;
  }
  return ;
}

std::pair<cv::Mat, cv::Mat> QuadcamDepthEstTrt::intrinsicsFromNode(const YAML::Node & node) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    printf("calibration parameters in size  height:%d width:%d\n",node["resolution"][1].as<int>(),node["resolution"][0].as<int>());

    K.at<double>(0, 0) = node["intrinsics"][0].as<double>();
    K.at<double>(1, 1) = node["intrinsics"][1].as<double>();
    K.at<double>(0, 2) = node["intrinsics"][2].as<double>();
    K.at<double>(1, 2) = node["intrinsics"][3].as<double>();

    cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
    D.at<double>(0, 0) = node["distortion_coeffs"][0].as<double>();
    D.at<double>(1, 0) = node["distortion_coeffs"][1].as<double>();
    D.at<double>(2, 0) = node["distortion_coeffs"][2].as<double>();
    D.at<double>(3, 0) = node["distortion_coeffs"][3].as<double>();
    return std::make_pair(K, D);
}

void QuadcamDepthEstTrt::startAllService(){
  this->raw_image_process_thread_ = std::thread(&QuadcamDepthEstTrt::rawImageProcessThread,this);
  this->inference_thread_ = std::thread(&QuadcamDepthEstTrt::inferrenceThread,this);
  this->publish_thread_ = std::thread(&QuadcamDepthEstTrt::publishThread,this);
  printf("[QuadcamDepthEstTrt]: start all service\n");
}

void QuadcamDepthEstTrt::stopAllService(){
  if(this->inference_thread_.joinable()){
    this->stopinfrenceThread();
    this->inference_thread_.join();
  }
  if(this->publish_thread_.joinable()){
    this->stoppublishThread();
    this->publish_thread_.join();
  }
  if(this->raw_image_process_thread_.joinable()){
    this->stoprawImageProcessThread();
    this->raw_image_process_thread_.join();
  }
}

void QuadcamDepthEstTrt::quadcamImageCb(const sensor_msgs::ImageConstPtr & images){
  if (!raw_image_mutex_.try_lock()){
    return;
  } else {
    raw_image_ = cv_bridge::toCvCopy(images, sensor_msgs::image_encodings::BGR8)->image;
    this->raw_image_header_ = images->header;
    raw_image_mutex_.unlock();
  }
  return;
}

//TODO:kCamearsNum = size of vitual_stereos_
void QuadcamDepthEstTrt::rawImageProcessThread(){
  while(raw_image_process_thread_running_){
    const auto preprocessing_started = std::chrono::steady_clock::now();
    static cv::Mat raw_image;
    static std_msgs::Header raw_header;
    /* Because raw_image_ always get new memory addr,
      so here we handle the memory and release raw_image_ for cb */
    if (raw_image_mutex_.try_lock()){
      if (raw_image_.empty()){
        this->raw_image_mutex_.unlock();
        this->raw_image_process_rate_->sleep();
        continue;
      } else {
        raw_image = raw_image_;
        raw_header = raw_image_header_;
        this->raw_image_mutex_.unlock();
      }
    } else {
      this->raw_image_process_rate_->sleep();
      continue;
    }

    if (raw_image.cols != 5120 || raw_image.rows != 720 ||
        raw_image.type() != CV_8UC3) {
      ROS_ERROR_THROTTLE(
          1.0, "Expected assembled bgr8 image 5120x720, got %dx%d type=%d",
          raw_image.cols, raw_image.rows, raw_image.type());
      this->raw_image_process_rate_->sleep();
      continue;
    }

    for(int32_t i = 0; i< kCamerasNum; i++){
      cv::Mat splited_image = raw_image(cv::Rect(i * raw_image.cols /kCamerasNum, 0,
        raw_image.cols /kCamerasNum, raw_image.rows));
      if(!this->cnn_input_rgb_){

        if (splited_image.empty()){
          printf("[QuadcamDepthEstTrt]: splited image is empty\n");
          this->raw_image_process_rate_->sleep();
          continue;
        }

        cv::cvtColor(splited_image,split_raw_images_[i],cv::COLOR_BGR2GRAY);//TODO: Bug openCV segement fault
        #ifdef DEBUG
        printf("[QuadcamDepthEstTrt]: split raw image to gray\n");
        char window_name[20];
        sprintf(window_name,"raw_image_%d",i);
        cv::imshow(window_name,split_raw_images_[i]);
        cv::waitKey(1);
        #endif
      } else {
        split_raw_images_[i] = splited_image;
      }
    }

    if (publish_pose_anchor_views_) {
      cv::Mat anchor_views[kCamerasNum];
      const bool publish_anchor_mosaic =
          pub_pose_anchor_mosaic_.getNumSubscribers() > 0;
      for (int32_t i = 0; i < kCamerasNum; ++i) {
        if (!publish_anchor_mosaic &&
            pub_pose_anchor_[i].getNumSubscribers() == 0) {
          continue;
        }
        anchor_views[i] = undistortors_[i]->undist_center_cpu(
            split_raw_images_[i], pose_anchor_width_, pose_anchor_height_,
            pose_anchor_fov_deg_);
        if (pub_pose_anchor_[i].getNumSubscribers() > 0) {
          std_msgs::Header anchor_header = raw_header;
          anchor_header.frame_id = "pose_anchor_cam_" + std::to_string(i);
          pub_pose_anchor_[i].publish(cv_bridge::CvImage(
              anchor_header, sensor_msgs::image_encodings::MONO8,
              anchor_views[i]).toImageMsg());
        }
      }
      if (publish_anchor_mosaic &&
          !anchor_views[0].empty() && !anchor_views[1].empty() &&
          !anchor_views[2].empty() && !anchor_views[3].empty()) {
        cv::Mat top, bottom, mosaic;
        cv::hconcat(anchor_views[0], anchor_views[1], top);
        cv::hconcat(anchor_views[2], anchor_views[3], bottom);
        cv::vconcat(top, bottom, mosaic);
        std_msgs::Header mosaic_header = raw_header;
        mosaic_header.frame_id = "pose_anchor_mosaic_A_B_C_D";
        pub_pose_anchor_mosaic_.publish(cv_bridge::CvImage(
            mosaic_header, sensor_msgs::image_encodings::MONO8,
            mosaic).toImageMsg());
      }
    }
    #ifdef DEBUG
    // printf("[QuadcamDepthEstTrt]: split raw image\n");
    cv::imshow("raw_image",raw_image_);
    cv::waitKey(1);
    #endif

    cv::Mat rectified_cpu[kCamerasNum][2];
    // On Orin Nano, four HITNet contexts saturate the GPU. CPU remapping can
    // overlap with inference instead of competing for the same CUDA cores.
    for(auto && stereo: this->virtual_stereos_){
      if (use_cuda_rectification_) {
        stereo->rectifyImage(
          split_raw_images_[stereo->cam_idx_a],
          split_raw_images_[stereo->cam_idx_b],
          rectified_images_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id],
          rectified_images_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      } else {
        stereo->rectifyImageCpu(
          split_raw_images_[stereo->cam_idx_a],
          split_raw_images_[stereo->cam_idx_b],
          rectified_cpu[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id],
          rectified_cpu[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      }
    }

    #ifdef DEBUG
    //show all pairs of rectified images
    for(auto && stereo: this->virtual_stereos_){
      char window_name[20];
      cv::Mat show_image;
      cv::Mat left;
      cv::Mat right;
      rectified_images_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id].download(left);
      rectified_images_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id].download(right);
      cv::hconcat(left,right,show_image);
      sprintf(window_name,"rectified_image_%d_%d",stereo->cam_idx_a,stereo->cam_idx_b);
      cv::imshow(window_name,show_image);
      cv::waitKey(1);
    }
    #endif

    //construct input images for hitnet inferrence and  TODO: can gpu mat be used directly?
    cv::Mat temp_left , temp_right, input_image[4];

    for (auto && stereo : this->virtual_stereos_){
      if (use_cuda_rectification_) {
        rectified_images_[stereo->cam_idx_a]
                         [stereo->cam_idx_a_right_half_id].download(temp_left);
        rectified_images_[stereo->cam_idx_b]
                         [stereo->cam_idx_b_left_half_id].download(temp_right);
      } else {
        temp_left = rectified_cpu[stereo->cam_idx_a]
                                 [stereo->cam_idx_a_right_half_id];
        temp_right = rectified_cpu[stereo->cam_idx_b]
                                  [stereo->cam_idx_b_left_half_id];
      }
      recity_images_for_show_and_texture_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id] = temp_left.clone();
      recity_images_for_show_and_texture_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id] = temp_right.clone();
      // redundant undistort image is already in size
      // cv::resize(temp_left,temp_left,cv::Size(this->width_,this->height_));
      // cv::resize(temp_right,temp_right,cv::Size(this->width_,this->height_));
      cv::vconcat(temp_left,temp_right,input_image[stereo->stereo_id]);
    }

    // Publish the fresh rectified stereo inputs before HITNet.  This stream
    // remains at the configured camera-processing rate even though four-pair
    // dense inference completes at roughly half that rate on Orin Nano.
    for (auto && stereo : this->virtual_stereos_) {
      const int id = stereo->stereo_id;
      std_msgs::Header image_header = raw_header;
      image_header.frame_id =
          "virtual_stereo_input_" + std::to_string(id);
      if (pub_input_rect_left_[id].getNumSubscribers() > 0) {
        pub_input_rect_left_[id].publish(
            cv_bridge::CvImage(image_header,
                               sensor_msgs::image_encodings::MONO8,
                               recity_images_for_show_and_texture_
                                   [stereo->cam_idx_a]
                                   [stereo->cam_idx_a_right_half_id])
                .toImageMsg());
      }
      if (pub_input_rect_right_[id].getNumSubscribers() > 0) {
        pub_input_rect_right_[id].publish(
            cv_bridge::CvImage(image_header,
                               sensor_msgs::image_encodings::MONO8,
                               recity_images_for_show_and_texture_
                                   [stereo->cam_idx_b]
                                   [stereo->cam_idx_b_left_half_id])
                .toImageMsg());
      }
    }

    // Bundle the same eight rectified images into one ROS message.  The pose
    // process then needs only one stereo callback instead of eight concurrent
    // rospy callbacks, while retaining all four pairs and exact timestamps.
    if (pub_pose_stereo_mosaic_.getNumSubscribers() > 0) {
      std::vector<cv::Mat> rows;
      rows.reserve(kCamerasNum);
      for (auto && stereo : this->virtual_stereos_) {
        cv::Mat row;
        cv::hconcat(
            recity_images_for_show_and_texture_[stereo->cam_idx_a]
                [stereo->cam_idx_a_right_half_id],
            recity_images_for_show_and_texture_[stereo->cam_idx_b]
                [stereo->cam_idx_b_left_half_id], row);
        rows.push_back(row);
      }
      cv::Mat mosaic;
      cv::vconcat(rows, mosaic);
      std_msgs::Header mosaic_header = raw_header;
      mosaic_header.frame_id = "pose_stereo_mosaic_AB_BC_CD_DA";
      pub_pose_stereo_mosaic_.publish(cv_bridge::CvImage(
          mosaic_header, sensor_msgs::image_encodings::MONO8,
          mosaic).toImageMsg());
    }

    //to reduce the time of mutex lock
    if (!input_tensors_mutex_.try_lock()){
      this->raw_image_process_rate_->sleep();
      continue;
    } else {
      for (auto && stereo : this->virtual_stereos_){
        input_image[stereo->stereo_id].convertTo(input_tensors_[stereo->stereo_id],CV_32FC1,1.0/255.0);
        input_rect_left_[stereo->stereo_id] =
            recity_images_for_show_and_texture_[stereo->cam_idx_a]
                                                [stereo->cam_idx_a_right_half_id].clone();
        input_rect_right_[stereo->stereo_id] =
            recity_images_for_show_and_texture_[stereo->cam_idx_b]
                                                [stereo->cam_idx_b_left_half_id].clone();
      }
      input_header_ = raw_header;
      input_ready_.store(true, std::memory_order_release);
      input_tensors_mutex_.unlock();
    }
    const double preprocessing_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocessing_started).count();
    ROS_INFO_THROTTLE(2.0, "Four-pair rectification/preprocessing: %.1f ms",
                      preprocessing_ms);
    this->raw_image_process_rate_->sleep();
  }
  return ;
}

void QuadcamDepthEstTrt::inferrenceThread(){
  static cv::Mat input_tensors[4];
  static cv::Mat input_left[4];
  static cv::Mat input_right[4];
  static std_msgs::Header input_header;
  while(inference_thread_running_){
    if (!input_ready_.load(std::memory_order_acquire)) {
      this->inference_rate_->sleep();
      continue;
    }
    if(input_tensors_mutex_.try_lock()){
      //if input_tensors_ is empty, wait for next loop
      if (this->input_tensors_[0].empty() ||
          !input_ready_.load(std::memory_order_acquire)){
        this->input_tensors_mutex_.unlock();
        this->inference_rate_->sleep();
        continue;
      }

      for (auto stereo : this->virtual_stereos_){
        input_tensors[stereo->stereo_id] = input_tensors_[stereo->stereo_id].clone();
        input_left[stereo->stereo_id] =
            input_rect_left_[stereo->stereo_id].clone();
        input_right[stereo->stereo_id] =
            input_rect_right_[stereo->stereo_id].clone();
      }
      input_header = input_header_;
      input_ready_.store(false, std::memory_order_release);
      input_tensors_mutex_.unlock();
    } else {
      this->inference_rate_->sleep();
      continue;
    }
    const auto inference_started = std::chrono::steady_clock::now();
    if (this->hitnet_->doInference(input_tensors) != 0) {
      ROS_ERROR_THROTTLE(1.0, "HITNet inference failed");
      this->inference_rate_->sleep();
      continue;
    }

    if (output_tensors_mutex_.try_lock()){
      if (this->hitnet_->getOutput(output_tensors_) != 0) {
        output_tensors_mutex_.unlock();
        ROS_ERROR_THROTTLE(1.0, "Failed to copy HITNet output");
        this->inference_rate_->sleep();
        continue;
      }
      for (auto stereo : this->virtual_stereos_) {
        output_rect_left_[stereo->stereo_id] =
            input_left[stereo->stereo_id].clone();
        output_rect_right_[stereo->stereo_id] =
            input_right[stereo->stereo_id].clone();
      }
      output_header_ = input_header;
      output_ready_.store(true, std::memory_order_release);
      output_tensors_mutex_.unlock();
    } else {
      this->inference_rate_->sleep();
      continue;
    }
    const double inference_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inference_started).count();
    ROS_INFO_THROTTLE(
        2.0,
        "Four-pair HITNet inference and output copy: %.1f ms (%.2f sets/s)",
        inference_ms, 1000.0 / std::max(1.0, inference_ms));
    this->inference_rate_->sleep();
  }
  return ;
}

cv::Mat QuadcamDepthEstTrt::buildDisparityValidityMask(
    const cv::Mat & disparity, const cv::Mat & left, const cv::Mat & right,
    const cv::Rect & stereo_roi) const {
  CV_Assert(disparity.type() == CV_32FC1);
  CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
  CV_Assert(disparity.size() == left.size() && left.size() == right.size());

  cv::Mat valid = cv::Mat::zeros(disparity.size(), CV_8UC1);
  const cv::Rect bounds(0, 0, disparity.cols, disparity.rows);
  cv::Rect roi = stereo_roi.empty() ? bounds : (stereo_roi & bounds);
  if (valid_roi_margin_ > 0 && roi.width > 2 * valid_roi_margin_ &&
      roi.height > 2 * valid_roi_margin_) {
    roi.x += valid_roi_margin_;
    roi.y += valid_roi_margin_;
    roi.width -= 2 * valid_roi_margin_;
    roi.height -= 2 * valid_roi_margin_;
  }

  cv::Mat median;
  cv::medianBlur(disparity, median, 5);

  // Separate histogram equalization makes this consistency test less
  // sensitive to the four cameras' remaining gain/exposure differences.
  cv::Mat left_equalized;
  cv::Mat right_equalized;
  cv::equalizeHist(left, left_equalized);
  cv::equalizeHist(right, right_equalized);

  cv::Mat grad_x;
  cv::Mat grad_y;
  cv::Sobel(left_equalized, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(left_equalized, grad_y, CV_32F, 0, 1, 3);

  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const float *disp_row = disparity.ptr<float>(y);
    const float *median_row = median.ptr<float>(y);
    const float *gx_row = grad_x.ptr<float>(y);
    const float *gy_row = grad_y.ptr<float>(y);
    const uchar *left_row = left_equalized.ptr<uchar>(y);
    uchar *valid_row = valid.ptr<uchar>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      const float d = disp_row[x];
      if (!std::isfinite(d) || d < min_disparity_ ||
          d > max_disparity_) {
        continue;
      }
      if (std::abs(d - median_row[x]) > max_median_deviation_) {
        continue;
      }
      const int right_x = cvRound(static_cast<float>(x) - d);
      if (right_x < roi.x || right_x >= roi.x + roi.width) {
        continue;
      }
      const float gradient =
          std::abs(gx_row[x]) + std::abs(gy_row[x]);
      if (gradient < min_texture_gradient_) {
        continue;
      }
      const int photo_error =
          std::abs(static_cast<int>(left_row[x]) -
                   static_cast<int>(right_equalized.at<uchar>(y, right_x)));
      if (photo_error > max_photometric_error_) {
        continue;
      }
      valid_row[x] = 255;
    }
  }

  if (min_valid_component_size_ > 1) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(
        valid, labels, stats, centroids, 8, CV_32S);
    cv::Mat filtered = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int label = 1; label < count; ++label) {
      if (stats.at<int>(label, cv::CC_STAT_AREA) >=
          min_valid_component_size_) {
        filtered.setTo(255, labels == label);
      }
    }
    valid = filtered;
  }
  return valid;
}

void QuadcamDepthEstTrt::publishThread(){
  //TODO: publish pointcloud and do visualization
  while(publish_thread_running_){
    //if output_tensors_ is empty, wait for next loop
    if (!output_ready_.load(std::memory_order_acquire)){
      this->publish_rate_->sleep();
      continue;
    }

    const auto publishing_started = std::chrono::steady_clock::now();
    // Consume every completed inference once. Keeping this flag set caused
    // repeated reprojection and publication of the same timestamp.
    if (output_tensors_mutex_.try_lock()){
      for (auto stereo : this->virtual_stereos_){
        publish_disparity_[stereo->stereo_id] = output_tensors_[stereo->stereo_id].clone();
        publish_rect_left_[stereo->stereo_id] =
            output_rect_left_[stereo->stereo_id].clone();
        publish_rect_right_[stereo->stereo_id] =
            output_rect_right_[stereo->stereo_id].clone();
      }
      publish_header_ = output_header_;
      output_ready_.store(false, std::memory_order_release);
      output_tensors_mutex_.unlock();
    } else {
      this->publish_rate_->sleep();
      continue;
    }
    //debug show disparity
    if(show_){
      if (recity_images_for_show_and_texture_[0][0].empty()){
        this->publish_rate_->sleep();
        continue;
      }
      for (auto stereo : this->virtual_stereos_){
        stereo->showDispartiy(publish_disparity_[stereo->stereo_id],
          recity_images_for_show_and_texture_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id],
          recity_images_for_show_and_texture_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      }
    }
    // Pointcloud construction is optional for the low-latency skeleton path.
    // Rectified images, disparity and metric depth remain available.
    if (enable_pointcloud_output_ && pcl_ != nullptr) {
      pcl_conversions::toPCL(publish_header_.stamp, pcl_->header.stamp);
      pcl_->header.frame_id = "imu";
      pcl_->points.clear();
    }
    const bool colored_subscriber = enable_pointcloud_output_ &&
        pub_pcl_colored_.getNumSubscribers() > 0;
    if (pcl_debug_color_ != nullptr && colored_subscriber) {
      pcl_conversions::toPCL(publish_header_.stamp,
                             pcl_debug_color_->header.stamp);
      pcl_debug_color_->header.frame_id = "imu";
      pcl_debug_color_->points.clear();
    }
    //TODO: if enable texture
    for (auto stereo : this->virtual_stereos_){
      const int id = stereo->stereo_id;
      const cv::Mat &left = publish_rect_left_[id];
      const cv::Mat &right = publish_rect_right_[id];
      if (enable_disparity_filter_) {
        publish_validity_mask_[id] = buildDisparityValidityMask(
            publish_disparity_[id], left, right, stereo->getValidRoi());
      } else {
        publish_validity_mask_[id] =
            cv::Mat(publish_disparity_[id].size(), CV_8UC1,
                    cv::Scalar(255));
      }
      cv::Mat points;
      const bool need_points = enable_pointcloud_output_ ||
          pub_depth_[id].getNumSubscribers() > 0;
      if (need_points) {
        cv::reprojectImageTo3D(publish_disparity_[id],
          points, stereo->getStereoPose(), false, CV_32F);
      }
      if (pub_rect_left_[id].getNumSubscribers() > 0 ||
          pub_rect_right_[id].getNumSubscribers() > 0 ||
          pub_disparity_[id].getNumSubscribers() > 0 ||
          pub_depth_[id].getNumSubscribers() > 0) {
        std_msgs::Header image_header = publish_header_;
        image_header.frame_id =
            "virtual_stereo_" + std::to_string(id);
        if (pub_rect_left_[id].getNumSubscribers() > 0) {
          pub_rect_left_[id].publish(
              cv_bridge::CvImage(
                  image_header, sensor_msgs::image_encodings::MONO8, left)
                  .toImageMsg());
        }
        if (pub_rect_right_[id].getNumSubscribers() > 0) {
          pub_rect_right_[id].publish(
              cv_bridge::CvImage(
                  image_header, sensor_msgs::image_encodings::MONO8, right)
                  .toImageMsg());
        }
        if (pub_disparity_[id].getNumSubscribers() > 0) {
          pub_disparity_[id].publish(
              cv_bridge::CvImage(
                  image_header, sensor_msgs::image_encodings::TYPE_32FC1,
                  publish_disparity_[id])
                  .toImageMsg());
        }
        if (pub_depth_[id].getNumSubscribers() > 0) {
          cv::Mat depth;
          cv::extractChannel(points, depth, 2);
          pub_depth_[id].publish(
              cv_bridge::CvImage(
                  image_header, sensor_msgs::image_encodings::TYPE_32FC1,
                  depth)
                  .toImageMsg());
        }
      }
      if (!enable_pointcloud_output_ || pcl_ == nullptr) {
        continue;
      }
      const std::size_t before = pcl_->points.size();
      addPointsToPCL(points, left, stereo->extrinsic, *this->pcl_,
          this->pixel_step_, this->min_z_, this->max_z_,
          publish_validity_mask_[id]);

      if (publish_debug_clouds_ &&
          pub_sector_pcl_[id].getNumSubscribers() > 0) {
        PointCloud sector;
        pcl_conversions::toPCL(publish_header_.stamp,
                               sector.header.stamp);
        sector.header.frame_id = "imu";
        addPointsToPCL(points, cv::Mat(), stereo->extrinsic, sector,
            this->pixel_step_, this->min_z_, this->max_z_,
            publish_validity_mask_[id]);
        pub_sector_pcl_[id].publish(sector);
      }

      if (publish_debug_clouds_ && colored_subscriber) {
        static const cv::Vec3b colors[kCamerasNum] = {
          cv::Vec3b(0, 0, 255), cv::Vec3b(0, 255, 0),
          cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 255)
        };
        cv::Mat sector_color(left.size(), CV_8UC3, colors[id]);
        addPointsToPCL(points, sector_color, stereo->extrinsic,
            *pcl_debug_color_, this->pixel_step_, this->min_z_,
            this->max_z_, publish_validity_mask_[id]);
      }
      ROS_INFO_THROTTLE(
          2.0, "Stereo %d retained %zu/%d sampled points", id,
          pcl_->points.size() - before,
          ((width_ + pixel_step_ - 1) / pixel_step_) *
          ((height_ + pixel_step_ - 1) / pixel_step_));
    }
    if (enable_pointcloud_output_ && pcl_ != nullptr) {
      pub_pcl_.publish(*pcl_);
      if (pcl_debug_color_ != nullptr && colored_subscriber) {
        pub_pcl_colored_.publish(*pcl_debug_color_);
      }
    }
    const double publishing_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - publishing_started).count();
    ROS_INFO_THROTTLE(2.0,
                      "Four-pair depth publish (pointcloud=%d): %.1f ms",
                      enable_pointcloud_output_, publishing_ms);
    this->publish_rate_->sleep();
  }
  return ;
}

} // namespace D2QuadCamDepthEst
