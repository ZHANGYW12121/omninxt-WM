#!/usr/bin/env python3
"""Translate the compact Isaac UDP protocol to native ROS1 messages."""

import json
import math
import socket
import struct
import threading
import time
import zlib

import numpy as np
import rospy
import rosnode
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField


CLOUD_MAGIC = b"EGPC"
CLOUD_HEADER = struct.Struct("!dIHH")


class IsaacUdpRosBridge:
    def __init__(self):
        listen_port = rospy.get_param("~listen_port", 15100)
        self.isaac_address = (
            rospy.get_param("~isaac_host", "127.0.0.1"),
            rospy.get_param("~isaac_port", 15101),
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", int(listen_port)))
        self.sock.setblocking(False)
        self.odom_pub = rospy.Publisher("/isaac/odom", Odometry, queue_size=20)
        self.cloud_pub = rospy.Publisher("/isaac/cloud", PointCloud2, queue_size=2)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=2)
        self.command_sub = rospy.Subscriber(
            "/planning/pos_cmd", PositionCommand, self.command_callback,
            queue_size=1, tcp_nodelay=True,
        )
        self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=2)
        self.cloud_chunks = {}
        self.mission_complete = False
        # rospy.Timer follows /clock when use_sim_time is enabled. It therefore
        # cannot be used to receive the first clock packet. Poll UDP on a wall-
        # time thread and publish Isaac's simulation clock from there.
        self.stop_event = threading.Event()
        self.receive_thread = threading.Thread(
            target=self.receive_loop, name="isaac_udp_rx", daemon=True
        )
        self.receive_thread.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Isaac UDP bridge listening on %d", listen_port)

    @staticmethod
    def ros_stamp(simulation_stamp):
        return rospy.Time.from_sec(max(0.0, float(simulation_stamp)))

    def receive_loop(self):
        last_cleanup = time.monotonic()
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            self.receive(None)
            now = time.monotonic()
            if now - last_cleanup >= 1.0:
                self.cleanup_chunks(None)
                last_cleanup = now
            time.sleep(0.002)

    def shutdown(self):
        self.stop_event.set()
        try:
            self.sock.close()
        except OSError:
            pass

    def receive(self, _event):
        for _ in range(64):
            try:
                payload, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError:
                return
            if payload.startswith(CLOUD_MAGIC):
                self.receive_cloud(payload)
            else:
                self.receive_json(payload)

    def receive_json(self, payload):
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Bad Isaac JSON: %s", exc)
            return
        kind = data.get("type")
        stamp = self.ros_stamp(data.get("stamp", 0.0))
        if kind == "clock":
            msg = Clock()
            msg.clock = stamp
            self.clock_pub.publish(msg)
        elif kind == "odom":
            p, v, q = data["position"], data["velocity"], data["quaternion_xyzw"]
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.child_frame_id = "base_link"
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = p
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = q[0], q[1]
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q[2], q[3]
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = v
            self.odom_pub.publish(msg)
        elif kind == "goal":
            p = data["position"]
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = p
            msg.pose.orientation.w = 1.0
            self.goal_pub.publish(msg)
        elif kind == "mission_complete" and not self.mission_complete:
            self.mission_complete = True
            p = data.get("position", [0.0, 0.0, 0.0])
            speed_xy = float(data.get("speed_xy", 0.0))
            rospy.logwarn(
                "[EGO][MISSION] Goal reached: position=(%.2f, %.2f, %.2f), "
                "speed_xy=%.2f m/s. Stopping planner and trajectory server.",
                p[0], p[1], p[2], speed_xy,
            )
            threading.Thread(
                target=self.stop_planner_nodes,
                name="ego_mission_stop",
                daemon=True,
            ).start()

    @staticmethod
    def stop_planner_nodes():
        # Run outside the UDP receive thread because XML-RPC node shutdown may
        # briefly block while the planner finishes its current callback.
        time.sleep(0.05)
        try:
            _, failed = rosnode.kill_nodes(["/ego_planner_node", "/traj_server"])
            if failed:
                rospy.logwarn("[EGO][MISSION] Failed to stop nodes: %s", failed)
            else:
                rospy.logwarn("[EGO][MISSION] Planner stopped at goal.")
        except Exception as exc:
            rospy.logerr("[EGO][MISSION] Could not stop planner nodes: %s", exc)

    def receive_cloud(self, payload):
        header_size = len(CLOUD_MAGIC) + CLOUD_HEADER.size
        if len(payload) < header_size:
            return
        stamp, sequence, index, count = CLOUD_HEADER.unpack(
            payload[len(CLOUD_MAGIC):header_size]
        )
        entry = self.cloud_chunks.setdefault(sequence, {
            "stamp": stamp, "count": count, "parts": {}, "wall": time.monotonic()
        })
        entry["parts"][index] = payload[header_size:]
        if len(entry["parts"]) != entry["count"]:
            return
        try:
            compressed = b"".join(entry["parts"][i] for i in range(entry["count"]))
            points = np.frombuffer(zlib.decompress(compressed), dtype="<f4").reshape(-1, 3)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Bad Isaac cloud: %s", exc)
            self.cloud_chunks.pop(sequence, None)
            return
        self.cloud_chunks.pop(sequence, None)
        msg = PointCloud2()
        msg.header.stamp = self.ros_stamp(stamp)
        msg.header.frame_id = "world"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = points.astype("<f4", copy=False).tobytes()
        self.cloud_pub.publish(msg)

    def cleanup_chunks(self, _event):
        now = time.monotonic()
        stale = [key for key, value in self.cloud_chunks.items() if now - value["wall"] > 2.0]
        for key in stale:
            self.cloud_chunks.pop(key, None)

    def command_callback(self, msg):
        yaw = float(msg.yaw)
        yaw_dot = float(msg.yaw_dot)
        if not math.isfinite(yaw):
            rospy.logwarn_throttle(2.0, "EGO PositionCommand yaw is non-finite; using 0")
            yaw = 0.0
        if not math.isfinite(yaw_dot):
            rospy.logwarn_throttle(2.0, "EGO PositionCommand yaw_dot is non-finite; using 0")
            yaw_dot = 0.0
        data = {
            "type": "position_command",
            "stamp": msg.header.stamp.to_sec(),
            "trajectory_id": int(msg.trajectory_id),
            "position": [msg.position.x, msg.position.y, msg.position.z],
            "velocity": [msg.velocity.x, msg.velocity.y, msg.velocity.z],
            "acceleration": [msg.acceleration.x, msg.acceleration.y, msg.acceleration.z],
            "yaw": yaw,
            "yaw_dot": yaw_dot,
        }
        self.sock.sendto(
            json.dumps(data, separators=(",", ":")).encode("utf-8"),
            self.isaac_address,
        )


if __name__ == "__main__":
    rospy.init_node("isaac_udp_ros_bridge")
    IsaacUdpRosBridge()
    rospy.spin()
