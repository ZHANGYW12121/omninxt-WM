#include "hitnet.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

double percentile(std::vector<float> values, double fraction) {
  values.erase(
      std::remove_if(values.begin(), values.end(),
                     [](float value) { return !std::isfinite(value); }),
      values.end());
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::round(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: diagnose_hitnet ENGINE OUTPUT_DIR "
                 "PAIR_0 PAIR_1 PAIR_2 PAIR_3\n";
    return 2;
  }
  const std::string engine = argv[1];
  const std::string output_dir = argv[2];
  cv::Mat inputs[4];
  for (int index = 0; index < 4; ++index) {
    const std::string prefix = argv[index + 3];
    cv::Mat left = cv::imread(prefix + "_left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(prefix + "_right.png", cv::IMREAD_GRAYSCALE);
    if (left.size() != cv::Size(320, 240) ||
        right.size() != cv::Size(320, 240)) {
      std::cerr << "Missing or invalid pair: " << prefix << "\n";
      return 3;
    }
    cv::Mat stacked;
    cv::vconcat(left, right, stacked);
    stacked.convertTo(inputs[index], CV_32FC1, 1.0 / 255.0);
  }

  TensorRTHitnet::HitnetTrt hitnet(true);
  if (hitnet.init("", engine, 4) != 0 || hitnet.doInference(inputs) != 0) {
    std::cerr << "HITNet initialization/inference failed\n";
    return 4;
  }
  cv::Mat disparities[4];
  if (hitnet.getOutput(disparities) != 0) {
    std::cerr << "HITNet output copy failed\n";
    return 5;
  }

  std::ofstream report(output_dir + "/hitnet_stats.txt");
  report << std::fixed << std::setprecision(6);
  for (int index = 0; index < 4; ++index) {
    std::vector<float> values(
        reinterpret_cast<float*>(disparities[index].data),
        reinterpret_cast<float*>(disparities[index].data) +
            disparities[index].total());
    size_t finite = 0;
    size_t positive = 0;
    for (float value : values) {
      finite += std::isfinite(value);
      positive += std::isfinite(value) && value > 0.0F;
    }
    report << "pair" << index
           << " finite=" << finite << "/" << values.size()
           << " positive=" << positive << "/" << values.size()
           << " p01=" << percentile(values, 0.01)
           << " p10=" << percentile(values, 0.10)
           << " p50=" << percentile(values, 0.50)
           << " p90=" << percentile(values, 0.90)
           << " p99=" << percentile(values, 0.99) << "\n";

    cv::Mat scaled;
    disparities[index].convertTo(scaled, CV_16U, 256.0);
    cv::imwrite(output_dir + "/pair" + std::to_string(index) +
                    "_disparity_x256.png",
                scaled);
    cv::Mat visible;
    disparities[index].convertTo(visible, CV_8U, 255.0 / 32.0);
    cv::applyColorMap(visible, visible, cv::COLORMAP_TURBO);
    cv::imwrite(output_dir + "/pair" + std::to_string(index) +
                    "_disparity_color.png",
                visible);
  }
  return 0;
}
