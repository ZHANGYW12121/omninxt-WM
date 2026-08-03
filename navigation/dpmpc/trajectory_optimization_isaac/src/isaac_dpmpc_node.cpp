#include <trajectory_optimization/mapModuleOctomap.h>
#include <trajectory_optimization/mpcPlanner.h>
#include <trajectory_optimization/staticPlanner.h>

#include <arpa/inet.h>
#include <endian.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zlib.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr char kCloudMagic[] = {'D', 'M', 'P', 'C'};
constexpr std::size_t kCloudHeaderBytes = 24;
constexpr std::size_t kMaximumDatagramBytes = 65535;
constexpr std::uint32_t kMaximumCloudPoints = 1000000;

struct CloudAssembly {
  std::uint32_t sequence = 0;
  std::uint32_t point_count = 0;
  std::uint16_t chunk_count = 0;
  std::vector<std::vector<unsigned char>> chunks;
  std::vector<bool> received;

  void reset(
      std::uint32_t new_sequence,
      std::uint32_t new_point_count,
      std::uint16_t new_chunk_count) {
    sequence = new_sequence;
    point_count = new_point_count;
    chunk_count = new_chunk_count;
    chunks.assign(chunk_count, {});
    received.assign(chunk_count, false);
  }

  bool complete() const {
    return chunk_count > 0
        && std::all_of(received.begin(), received.end(), [](bool value) {
             return value;
           });
  }
};

template <typename T>
T jsonNumber(const json& value, const char* key, T fallback = T{}) {
  if (!value.contains(key) || !value.at(key).is_number()) {
    return fallback;
  }
  return value.at(key).get<T>();
}

std::vector<double> jsonVector(
    const json& value, const char* key, std::size_t size) {
  if (!value.contains(key) || !value.at(key).is_array()
      || value.at(key).size() != size) {
    throw std::runtime_error(std::string("invalid vector: ") + key);
  }
  std::vector<double> result;
  result.reserve(size);
  for (const auto& item : value.at(key)) {
    const double number = item.get<double>();
    if (!std::isfinite(number)) {
      throw std::runtime_error(std::string("non-finite vector: ") + key);
    }
    result.push_back(number);
  }
  return result;
}

std::uint16_t readU16(const unsigned char* data) {
  std::uint16_t raw = 0;
  std::memcpy(&raw, data, sizeof(raw));
  return ntohs(raw);
}

std::uint32_t readU32(const unsigned char* data) {
  std::uint32_t raw = 0;
  std::memcpy(&raw, data, sizeof(raw));
  return ntohl(raw);
}

double readF64(const unsigned char* data) {
  std::uint64_t raw = 0;
  std::memcpy(&raw, data, sizeof(raw));
  raw = be64toh(raw);
  double result = 0.0;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}

void quaternionToRpy(
    const std::vector<double>& quaternion_xyzw,
    double& roll,
    double& pitch,
    double& yaw) {
  const double x = quaternion_xyzw[0];
  const double y = quaternion_xyzw[1];
  const double z = quaternion_xyzw[2];
  const double w = quaternion_xyzw[3];
  const double sin_roll = 2.0 * (w * x + y * z);
  const double cos_roll = 1.0 - 2.0 * (x * x + y * y);
  roll = std::atan2(sin_roll, cos_roll);
  const double sin_pitch = 2.0 * (w * y - z * x);
  pitch = std::abs(sin_pitch) >= 1.0
      ? std::copysign(M_PI / 2.0, sin_pitch)
      : std::asin(sin_pitch);
  const double sin_yaw = 2.0 * (w * z + x * y);
  const double cos_yaw = 1.0 - 2.0 * (y * y + z * z);
  yaw = std::atan2(sin_yaw, cos_yaw);
}

class IsaacDpmpcNode {
 public:
  explicit IsaacDpmpcNode(ros::NodeHandle& node)
      : node_(node) {
    node_.param("udp_bind_port", bind_port_, 15200);
    node_.param("udp_isaac_port", isaac_port_, 15201);
    node_.param("map_resolution", map_resolution_, 0.10);
    node_.param("robot_x_size", robot_x_size_, 0.64);
    node_.param("robot_y_size", robot_y_size_, 0.64);
    node_.param("robot_z_size", robot_z_size_, 0.30);
    node_.param("sampling_dt", sampling_dt_, 0.10);
    node_.param("horizon", horizon_, 20);
    node_.param("static_velocity", static_velocity_, 2.0);

    // Construct after parameters are read so the collision box exactly
    // matches the runtime vehicle parameters.
    map_ptr_.reset(new mapModule(
        node_, map_resolution_, robot_x_size_, robot_y_size_, robot_z_size_));
    openSocket();
    ROS_WARN_STREAM(
        "[DPMPC] Isaac sidecar listening on UDP " << bind_port_
        << ", sending commands to " << isaac_port_);
  }

  ~IsaacDpmpcNode() {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
    }
  }

  void run() {
    std::vector<unsigned char> buffer(kMaximumDatagramBytes);
    while (ros::ok()) {
      fd_set descriptors;
      FD_ZERO(&descriptors);
      FD_SET(socket_fd_, &descriptors);
      timeval timeout{};
      timeout.tv_usec = 100000;
      const int selected = ::select(
          socket_fd_ + 1, &descriptors, nullptr, nullptr, &timeout);
      if (selected < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("select failed");
      }
      if (selected == 0) {
        ros::spinOnce();
        continue;
      }
      sockaddr_in source{};
      socklen_t source_length = sizeof(source);
      const ssize_t size = ::recvfrom(
          socket_fd_,
          buffer.data(),
          buffer.size(),
          0,
          reinterpret_cast<sockaddr*>(&source),
          &source_length);
      if (size <= 0) {
        continue;
      }
      try {
        handleDatagram(buffer.data(), static_cast<std::size_t>(size));
      } catch (const std::exception& error) {
        ROS_ERROR_STREAM_THROTTLE(
            1.0, "[DPMPC] Dropped invalid Isaac datagram: " << error.what());
      }
      ros::spinOnce();
    }
  }

 private:
  void openSocket() {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("cannot create UDP socket");
    }
    int reuse = 1;
    ::setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<std::uint16_t>(bind_port_));
    if (::bind(
            socket_fd_,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) != 0) {
      throw std::runtime_error("cannot bind UDP planner port");
    }
    isaac_address_.sin_family = AF_INET;
    isaac_address_.sin_port = htons(static_cast<std::uint16_t>(isaac_port_));
    ::inet_pton(AF_INET, "127.0.0.1", &isaac_address_.sin_addr);
  }

  void handleDatagram(const unsigned char* data, std::size_t size) {
    if (size >= 4 && std::memcmp(data, kCloudMagic, 4) == 0) {
      handleCloudChunk(data, size);
      return;
    }
    const std::string text(reinterpret_cast<const char*>(data), size);
    const json message = json::parse(text);
    const std::string type = message.value("type", "");
    if (type == "goal") {
      const auto position = jsonVector(message, "position", 3);
      goal_ = pose(
          position[0], position[1], position[2],
          jsonNumber<double>(message, "yaw", 0.0));
      have_goal_ = true;
      have_static_trajectory_ = false;
      ROS_WARN_STREAM(
          "[DPMPC] Goal received: (" << goal_.x << ", " << goal_.y
          << ", " << goal_.z << ")");
    } else if (type == "observation") {
      handleObservation(message);
    } else if (type == "reset") {
      have_goal_ = false;
      have_static_trajectory_ = false;
      static_trajectory_.clear();
      ROS_WARN("[DPMPC] Episode state reset.");
    }
  }

  void handleCloudChunk(const unsigned char* data, std::size_t size) {
    if (size < kCloudHeaderBytes) {
      throw std::runtime_error("short cloud header");
    }
    const double stamp = readF64(data + 4);
    const std::uint32_t sequence = readU32(data + 12);
    const std::uint32_t point_count = readU32(data + 16);
    const std::uint16_t chunk_index = readU16(data + 20);
    const std::uint16_t chunk_count = readU16(data + 22);
    if (!std::isfinite(stamp) || point_count > kMaximumCloudPoints
        || chunk_count == 0 || chunk_index >= chunk_count) {
      throw std::runtime_error("invalid cloud metadata");
    }
    if (cloud_.sequence != sequence
        || cloud_.point_count != point_count
        || cloud_.chunk_count != chunk_count) {
      cloud_.reset(sequence, point_count, chunk_count);
    }
    cloud_.chunks[chunk_index].assign(data + kCloudHeaderBytes, data + size);
    cloud_.received[chunk_index] = true;
    if (!cloud_.complete()) {
      return;
    }

    std::vector<unsigned char> compressed;
    std::size_t compressed_size = 0;
    for (const auto& chunk : cloud_.chunks) {
      compressed_size += chunk.size();
    }
    compressed.reserve(compressed_size);
    for (const auto& chunk : cloud_.chunks) {
      compressed.insert(compressed.end(), chunk.begin(), chunk.end());
    }
    std::vector<float> xyz(static_cast<std::size_t>(point_count) * 3);
    if (point_count > 0) {
      uLongf output_size = static_cast<uLongf>(xyz.size() * sizeof(float));
      const int status = ::uncompress(
          reinterpret_cast<Bytef*>(xyz.data()),
          &output_size,
          reinterpret_cast<const Bytef*>(compressed.data()),
          static_cast<uLong>(compressed.size()));
      if (status != Z_OK || output_size != xyz.size() * sizeof(float)) {
        throw std::runtime_error("cannot decompress static cloud");
      }
    }
    std::vector<octomap::point3d> points;
    points.reserve(point_count);
    for (std::uint32_t index = 0; index < point_count; ++index) {
      const float x = xyz[3 * index];
      const float y = xyz[3 * index + 1];
      const float z = xyz[3 * index + 2];
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        points.emplace_back(x, y, z);
      }
    }
    map_ptr_->loadOccupiedPoints(points);
    have_map_ = true;
    have_static_trajectory_ = false;
    ROS_WARN_STREAM(
        "[DPMPC] Static Isaac cloud loaded: " << points.size()
        << " points, sequence=" << sequence);
  }

  std::vector<obstacle> parseObstacles(const json& message) const {
    std::vector<obstacle> obstacles;
    if (!message.contains("obstacles") || !message.at("obstacles").is_array()) {
      return obstacles;
    }
    for (const auto& item : message.at("obstacles")) {
      const auto position = jsonVector(item, "position", 3);
      const auto velocity = jsonVector(item, "velocity", 3);
      const auto size = jsonVector(item, "size", 3);
      const auto position_variance =
          jsonVector(item, "position_variance", 3);
      const auto velocity_variance =
          jsonVector(item, "velocity_variance", 3);
      obstacle value{};
      value.x = position[0];
      value.y = position[1];
      value.z = position[2];
      value.vx = velocity[0];
      value.vy = velocity[1];
      value.vz = velocity[2];
      value.xsize = std::max(0.05, size[0]);
      value.ysize = std::max(0.05, size[1]);
      value.zsize = std::max(0.05, size[2]);
      value.varX = std::max(1e-9, position_variance[0]);
      value.varY = std::max(1e-9, position_variance[1]);
      value.varZ = std::max(1e-9, position_variance[2]);
      value.varVx = std::max(0.0, velocity_variance[0]);
      value.varVy = std::max(0.0, velocity_variance[1]);
      value.varVz = std::max(0.0, velocity_variance[2]);
      obstacles.push_back(value);
    }
    return obstacles;
  }

  void buildStaticTrajectory(const DVector& current_states, double current_yaw) {
    std::vector<pose> path;
    path.emplace_back(
        current_states(0), current_states(1), current_states(2), current_yaw);
    pose target = goal_;
    target.yaw = std::atan2(
        goal_.y - current_states(1), goal_.x - current_states(0));
    path.push_back(target);
    std::vector<pose> loaded_path;
    static_trajectory_ = staticPlanner(
        map_ptr_.get(),
        7,
        static_velocity_,
        4,
        1.0,
        path,
        false,
        sampling_dt_,
        loaded_path);
    if (static_trajectory_.empty()) {
      throw std::runtime_error("official static planner returned no trajectory");
    }
    have_static_trajectory_ = true;
    ROS_WARN_STREAM(
        "[DPMPC] Static minimum-snap trajectory ready: "
        << static_trajectory_.size() << " samples.");
  }

  void handleObservation(const json& message) {
    if (!have_goal_ || !have_map_) {
      return;
    }
    const auto position = jsonVector(message, "position", 3);
    const auto velocity = jsonVector(message, "velocity", 3);
    const auto quaternion = jsonVector(message, "quaternion_xyzw", 4);
    const double stamp = jsonNumber<double>(message, "stamp", 0.0);
    const int observation_id = jsonNumber<int>(message, "observation_id", -1);
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    quaternionToRpy(quaternion, roll, pitch, yaw);
    DVector current_states(8);
    current_states(0) = position[0];
    current_states(1) = position[1];
    current_states(2) = position[2];
    current_states(3) = velocity[0];
    current_states(4) = velocity[1];
    current_states(5) = velocity[2];
    current_states(6) = roll;
    current_states(7) = pitch;

    if (!have_static_trajectory_) {
      buildStaticTrajectory(current_states, yaw);
    }
    const std::vector<obstacle> obstacles = parseObstacles(message);
    const auto solve_start = Clock::now();

    mpcPlanner planner(horizon_);
    planner.loadParameters(1.5, 10.0, 1.0, 10.0, 1.0);
    planner.loadControlLimits(3.0 * 9.8, PI_const / 6.0, PI_const / 6.0);
    planner.loadRefTrajectory(static_trajectory_, sampling_dt_);
    DVector next_states;
    VariablesGrid optimized_states;
    std::vector<pose> mpc_trajectory;
    const int solver_status = planner.optimize(
        current_states,
        yaw,
        obstacles,
        next_states,
        mpc_trajectory,
        optimized_states);

    std::vector<int> collision_indices;
    const bool valid = map_ptr_->checkCollisionTrajectory(
        mpc_trajectory, collision_indices, true);
    std::size_t available_count = mpc_trajectory.size();
    if (!valid && !collision_indices.empty()) {
      available_count = static_cast<std::size_t>(
          std::max(1, collision_indices.front()));
    }
    if (available_count == 0) {
      throw std::runtime_error("official dynamic planner returned no setpoint");
    }
    const std::size_t forward_index = std::min<std::size_t>(
        10, available_count - 1);
    const DVector setpoint_state =
        optimized_states.getVector(static_cast<int>(forward_index));
    const pose& setpoint_pose = mpc_trajectory[forward_index];
    const double solve_time_ms =
        std::chrono::duration<double, std::milli>(
            Clock::now() - solve_start).count();

    json response = {
        {"type", "dpmpc_command"},
        {"stamp", stamp},
        {"observation_id", observation_id},
        {"solver_status", solver_status},
        {"solve_time_ms", solve_time_ms},
        {"position", {
            setpoint_state(0), setpoint_state(1), setpoint_state(2)}},
        // The upstream MAVROS example ignores velocity. It is included only
        // for diagnostics and is not used by DpmpcController for actuation.
        {"velocity", {
            setpoint_state(3), setpoint_state(4), setpoint_state(5)}},
        {"acceleration", {0.0, 0.0, 0.0}},
        {"yaw", setpoint_pose.yaw},
        {"yaw_dot", 0.0},
    };
    const std::string payload = response.dump();
    ::sendto(
        socket_fd_,
        payload.data(),
        payload.size(),
        0,
        reinterpret_cast<sockaddr*>(&isaac_address_),
        sizeof(isaac_address_));
  }

  ros::NodeHandle& node_;
  std::unique_ptr<mapModule> map_ptr_;
  int socket_fd_ = -1;
  int bind_port_ = 15200;
  int isaac_port_ = 15201;
  sockaddr_in isaac_address_{};
  double map_resolution_ = 0.10;
  double robot_x_size_ = 0.64;
  double robot_y_size_ = 0.64;
  double robot_z_size_ = 0.30;
  double sampling_dt_ = 0.10;
  double static_velocity_ = 2.0;
  int horizon_ = 20;
  bool have_goal_ = false;
  bool have_map_ = false;
  bool have_static_trajectory_ = false;
  pose goal_;
  std::vector<pose> static_trajectory_;
  CloudAssembly cloud_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "isaac_dpmpc_node");
  ros::NodeHandle node("~");
  try {
    IsaacDpmpcNode planner(node);
    planner.run();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM("[DPMPC] Fatal sidecar error: " << error.what());
    return 1;
  }
  return 0;
}
