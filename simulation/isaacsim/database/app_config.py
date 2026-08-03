#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

import numpy as np

from warehouse_crowd_v2.crowd_templates import (
    VALID_DIRECTIONS,
    VALID_DRONE_DISTANCES,
    VALID_GROUP_SPACINGS,
    VALID_SPEEDS,
    build_crowd_scene,
    sample_warehouse_goal,
)

# ===== Paths and configs =====
HEADLESS = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ZYW_ROOT = os.path.expanduser(os.environ.get("ZYW_ROOT", "~/zyw"))
OMNINXT_ASSET_ROOT = os.path.expanduser(
    os.environ.get(
        "OMNINXT_ASSET_ROOT",
        os.path.join(APP_DIR, "..", "..", "..", ".local", "assets"),
    )
)

# Choose "warehouse" for the new warehouse scene, or "citytower" for the previous scene.
SCENE_PRESET = "warehouse"
SCENE_PRESETS = {
    "citytower": {
        "usd_path": os.path.expanduser(
            os.environ.get(
                "CITYTOWER_USD",
                os.path.join(OMNINXT_ASSET_ROOT, "warehouse", "World_CityTowerDemopack.usd"),
            )
        ),
        "goal_x_range": (-1.0, 5.0),
        "goal_y_min": 120.0,
        "walk_polygon": [
            (11.08, 95.48),
            (-7.83, 95.48),
            (-7.83, 122.46),
            (11.08, 122.46),
        ],
        "drone_x_range": (-5.0, 10.0),
        "drone_y_range": (97.0, 100.0),
        "direction_start_bounds": {
            "same_direction": (-2.0, 7.0, 100.0, 108.0),
            "opposite_direction": (-5.0, 10.0, 112.0, 120.0),
            "left_to_right": (-3.0, 7.0, 100.0, 120.0),
            "right_to_left": (-3.0, 7.0, 100.0, 120.0),
        },
        "num_people_choices": tuple(range(10, 21)),
        "default_num_people": 18,
    },
    "warehouse": {
        "usd_path": os.path.expanduser(
            os.environ.get(
                "WAREHOUSE_USD",
                os.path.join(OMNINXT_ASSET_ROOT, "warehouse", "warehouse.usd"),
            )
        ),
        "goal_x_range": (-3.0, 4.0),
        "goal_y_min": 27.0,
        "goal_y_range": (27.0, 29.0),
        "walk_polygon": [
            # Measured from the loaded Warehouse wall inner faces:
            # x=-10.13..9.21, y=-12.00..30.80. Keep pedestrian centers
            # roughly 0.5 m inside the walls; racks and props are removed
            # from this region separately by inflated obstacle AABBs.
            (8.7, -11.5),
            (-9.6, -11.5),
            (-9.6, 30.3),
            (8.7, 30.3),
        ],
        "drone_x_range": (-3.0, 4.0),
        "drone_y_range": (-10.0, -5.0),
        "direction_start_bounds": {
            "same_direction": (-9.6, 8.7, -11.5, 9.0),
            "opposite_direction": (-9.6, 8.7, 9.0, 30.3),
            "left_to_right": (-9.6, 8.7, -11.5, 30.3),
            "right_to_left": (-9.6, 8.7, -11.5, 30.3),
        },
        "num_people_choices": tuple(range(17, 24)),
        "default_num_people": 20,
    },
}
if SCENE_PRESET not in SCENE_PRESETS:
    raise ValueError(f"SCENE_PRESET must be one of {tuple(SCENE_PRESETS)}, got {SCENE_PRESET!r}")

ACTIVE_SCENE_PRESET = SCENE_PRESETS[SCENE_PRESET]
USD_PATH = ACTIVE_SCENE_PRESET["usd_path"]
DATASET_GOAL_X_RANGE = ACTIVE_SCENE_PRESET["goal_x_range"]
DATASET_GOAL_Y_MIN = ACTIVE_SCENE_PRESET["goal_y_min"]
TARGET_POINT = [
    0.5 * (DATASET_GOAL_X_RANGE[0] + DATASET_GOAL_X_RANGE[1]),
    DATASET_GOAL_Y_MIN,
    1.0,
]
DATA_RECORD_HZ = 10.0
# Flight-only mode: PX4 takeoff and autonomous navigation remain active, while
# dataset directories and frame files are never created.
DATA_RECORD_ENABLED = os.environ.get("OMNINXT_DATA_RECORD_ENABLED", "0") == "1"
# The autonomous policy is evaluated at the same rate as dataset sampling.
# Its velocity command is held constant between two control ticks.
DATA_CONTROL_HZ = DATA_RECORD_HZ
DATA_RECORD_QUEUE_SIZE = 16
DATA_RECORD_CHUNK_FRAMES = int(os.environ.get(
    "OMNINXT_DATA_RECORD_CHUNK_FRAMES", "256"))
DATA_RECORD_STORAGE_MAX_PEOPLE = int(os.environ.get(
    "OMNINXT_DATA_RECORD_STORAGE_MAX_PEOPLE", "32"))
DATA_RECORD_PRIVILEGED_MAX_PEOPLE = int(os.environ.get(
    "OMNINXT_DATA_RECORD_PRIVILEGED_MAX_PEOPLE", "32"))
DATA_RECORD_SYNC_TOLERANCE_SEC = float(os.environ.get(
    "OMNINXT_DATA_RECORD_SYNC_TOLERANCE_SEC", "0.075"))
DATA_RECORD_SKELETON_HOST = os.environ.get(
    "OMNINXT_SKELETON_BACKEND_HOST", "127.0.0.1")
DATA_RECORD_SKELETON_PORT = int(os.environ.get(
    "OMNINXT_SKELETON_BACKEND_PORT", "9765"))
DATASET_ROOT = os.path.expanduser(
    os.environ.get("OMNINXT_DATASET_ROOT", os.path.join(ZYW_ROOT, "database_quadcamera"))
)
DATA_RECORD_MAX_TRAJECTORIES = 600
DATA_RECORD_STUCK_TIMEOUT_SEC = 120.0
DATA_RECORD_MAX_RECORD_BYTES = 5 * 1024**3
DATA_RECORD_SIZE_CHECK_INTERVAL_SEC = 5.0
DATA_CAMERA_RESOLUTION = (1280, 720)
DATA_IMAGE_FORMAT = "jpg"
DATA_IMAGE_JPEG_QUALITY = 85
DATA_JSON_INDENT = None
# Keep every 10 Hz transition so one held action always matches one saved
# observation interval. The simulator may wait for disk I/O instead of dropping.
DATA_DROP_WHEN_WRITER_BUSY = False

# ===== MVD35 + OmniNxt sim2real vehicle model =====
# First engineering estimate from MVD35_OmniNxt_IsaacSim_Sim2Real_Config_v1.md.
# Pegasus uses ENU world and FLU body coordinates: +X forward, +Y left, +Z up.
#
# Disabled for now: use the original Pegasus/Iris vehicle dynamics and PX4
# actuator mapping while keeping the OmniNxt fisheye camera rig enabled below.
MVD35_SIM2REAL_ENABLED = False
# PX4 SITL lockstep is happiest when Pegasus publishes HIL data at the PX4
# backend rate. Keep dataset/control sampling at 10 Hz; only the physics clock
# needs to match the PX4 backend.
SIM_PHYSICS_HZ = 250.0
SIM_RENDER_HZ = 20.0
SIM_PHYSICS_DT = 1.0 / SIM_PHYSICS_HZ
SIM_RENDERING_DT = 1.0 / SIM_RENDER_HZ

MVD35_TOTAL_MASS_KG = 0.83
MVD35_ROTOR_MASS_KG = 0.005
MVD35_BODY_MASS_KG = MVD35_TOTAL_MASS_KG - 4.0 * MVD35_ROTOR_MASS_KG
MVD35_CENTER_OF_MASS_M = (0.0, 0.0, 0.008)
MVD35_DIAGONAL_INERTIA_KGM2 = (0.00150, 0.00165, 0.00260)
MVD35_ROTOR_DIAGONAL_INERTIA_KGM2 = (2.0e-6, 2.0e-6, 4.0e-6)

MVD35_MOTOR_DIAGONAL_DISTANCE_M = 0.152
MVD35_ROTOR_ARM_ABS_M = MVD35_MOTOR_DIAGONAL_DISTANCE_M / (2.0 * np.sqrt(2.0))
MVD35_ROTOR_Z_M = -0.008
# Preserve Pegasus/PX4 Iris motor order:
# rotor0 front-right CCW, rotor1 rear-left CCW, rotor2 front-left CW, rotor3 rear-right CW.
MVD35_ROTOR_POSITIONS_FLU_M = (
    (MVD35_ROTOR_ARM_ABS_M, -MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_Z_M),
    (-MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_Z_M),
    (MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_Z_M),
    (-MVD35_ROTOR_ARM_ABS_M, -MVD35_ROTOR_ARM_ABS_M, MVD35_ROTOR_Z_M),
)
MVD35_ROTOR_DIRECTIONS = (-1, -1, 1, 1)

MVD35_PROP_DIAMETER_M = 0.0889
MVD35_MAX_THRUST_PER_ROTOR_N = 6.5
MVD35_MAX_ROTOR_VELOCITY_RAD_S = 3800.0
MVD35_THRUST_COEFFICIENT = 4.50e-7
MVD35_MOMENT_COEFFICIENT = 5.40e-9
MVD35_MOTOR_TAU_UP_SEC = 0.035
MVD35_MOTOR_TAU_DOWN_SEC = 0.055
MVD35_THRUST_CURVE_CONFIG = {
    "num_rotors": 4,
    "rotor_constant": [MVD35_THRUST_COEFFICIENT] * 4,
    "rolling_moment_coefficient": [MVD35_MOMENT_COEFFICIENT] * 4,
    "rot_dir": list(MVD35_ROTOR_DIRECTIONS),
    "min_rotor_velocity": [0.0] * 4,
    "max_rotor_velocity": [MVD35_MAX_ROTOR_VELOCITY_RAD_S] * 4,
    "time_constant_up": [MVD35_MOTOR_TAU_UP_SEC] * 4,
    "time_constant_down": [MVD35_MOTOR_TAU_DOWN_SEC] * 4,
}
# Current Pegasus drag model is linear body-frame damping, not CdA quadratic drag.
MVD35_LINEAR_DRAG_COEFFICIENTS = (0.50, 0.30, 0.0)

# ===== OmniNxt visual model replacement =====
# This is a visual-only overlay. The Iris rigid bodies, colliders, rotor force
# application points, joints, sensors, and PX4 backend remain unchanged.
_OMNINXT_VISUAL_ASSET_DIR = os.path.expanduser(
    os.environ.get(
        "OMNINXT_VISUAL_ASSET_DIR",
        os.path.join(OMNINXT_ASSET_ROOT, "omninxt_usd"),
    )
)
OMNINXT_VISUAL_ENABLED = os.environ.get("OMNINXT_VISUAL_ENABLED", "1") == "1"
OMNINXT_VISUAL_BODY_USD = os.path.join(
    _OMNINXT_VISUAL_ASSET_DIR, "Omininxt_body.usdc"
)
OMNINXT_VISUAL_ROTOR_USDS = {
    "front_left": os.path.join(_OMNINXT_VISUAL_ASSET_DIR, "fl.usdc"),
    "front_right": os.path.join(_OMNINXT_VISUAL_ASSET_DIR, "fr.usdc"),
    "rear_left": os.path.join(_OMNINXT_VISUAL_ASSET_DIR, "rl.usdc"),
    "rear_right": os.path.join(_OMNINXT_VISUAL_ASSET_DIR, "rr.usdc"),
}
# The exported stages already use meters. Do not repeat Blender's 0.01 scale.
OMNINXT_VISUAL_SCALE_CORRECTION = 1.0
OMNINXT_VISUAL_BODY_TRANSLATION_M = (0.0, 0.0, 0.0)
# Blender/USD export leaves the model nose 90 degrees clockwise from Pegasus
# FLU +X. Rotate only the rendered body counter-clockwise when viewed from +Z.
OMNINXT_VISUAL_BODY_YAW_DEG = 90.0
OMNINXT_VISUAL_ROTOR_POSITIONS_FLU_M = {
    "front_left": (0.05374, 0.05374, MVD35_ROTOR_Z_M),
    "front_right": (0.05374, -0.05374, MVD35_ROTOR_Z_M),
    "rear_left": (-0.05374, 0.05374, MVD35_ROTOR_Z_M),
    "rear_right": (-0.05374, -0.05374, MVD35_ROTOR_Z_M),
}
# fr.usdc retained an exported world-space installation offset in its mesh
# vertices. The other three rotor assets are centered at their local origins.
# This measured correction makes all four assets share the same rotor-axis origin.
OMNINXT_VISUAL_ROTOR_ASSET_TRANSLATIONS_M = {
    "front_left": (0.0, 0.0, 0.0),
    "front_right": (0.05345174, 0.04601702, 0.01809603),
    "rear_left": (0.0, 0.0, 0.0),
    "rear_right": (0.0, 0.0, 0.0),
}
# Existing Pegasus/PX4 motor order: rotor0 FR, rotor1 RL, rotor2 FL, rotor3 RR.
OMNINXT_VISUAL_ROTOR_INDICES = {
    "front_left": 2,
    "front_right": 0,
    "rear_left": 1,
    "rear_right": 3,
}

MVD35_BATTERY_CELLS = 6
MVD35_BATTERY_CAPACITY_AH = 1.5
MVD35_BATTERY_VOLTAGE_FULL = 25.2
MVD35_BATTERY_VOLTAGE_NOMINAL = 22.2
MVD35_BATTERY_VOLTAGE_LOW = 19.8
MVD35_BATTERY_INTERNAL_RESISTANCE_OHM = 0.06

# PX4 HIL actuator controls are normalized before Pegasus converts them to rad/s.
# Keep the original 100 rad/s armed idle offset and map control=1.0 to 3800 rad/s.
MVD35_PX4_INPUT_OFFSET = (0.0, 0.0, 0.0, 0.0)
MVD35_PX4_ZERO_POSITION_ARMED = (100.0, 100.0, 100.0, 100.0)
MVD35_PX4_INPUT_SCALING = tuple(
    MVD35_MAX_ROTOR_VELOCITY_RAD_S - value
    for value in MVD35_PX4_ZERO_POSITION_ARMED
)

# ===== OmniNxt 2026-08-02 formal Sim2Real camera rig =====
OMNINXT_CAMERA_ENABLED = True
# raw_mei is the production four-fisheye chain. rectified_validation creates
# four 120-degree anchors plus four directly rectified stereo pairs solely for
# geometry acceptance tests.
OMNINXT_SENSOR_MODE = os.environ.get("OMNINXT_SENSOR_MODE", "raw_mei")
if OMNINXT_SENSOR_MODE not in ("raw_mei", "rectified_validation"):
    raise ValueError(
        "OMNINXT_SENSOR_MODE must be raw_mei or rectified_validation"
    )
OMNINXT_CAMERA_CONFIG_PATH = "omninxt_sync_20260802_camera_config.json"
# These names remain only as a fallback for legacy one-profile JSON files. The
# selected 20260802 profile supplies its own ordered camera list and names.
OMNINXT_CAMERA_NAMES = ("cam0", "cam1", "cam2", "cam3")
OMNINXT_CAMERA_SENSOR_NAMES = {
    "cam0": "front_cam",
    "cam1": "right_cam",
    "cam2": "rear_cam",
    "cam3": "left_cam",
}
OMNINXT_ACTIVE_CAMERA = "cam0"
OMNINXT_VIEW_WINDOWS_ENABLED = True
OMNINXT_VIEW_WINDOW_CAMERA_NAMES = ("cam0",)
OMNINXT_VIEW_WINDOW_LAYOUT = {
    "cam0": {
        "name": "OmniNxt front_cam (cam0)",
        "width": 640,
        "height": 360,
        "position_x": 60,
        "position_y": 60,
    },
}
OMNINXT_PREVIEW_SAVE_ON_START = True
OMNINXT_PREVIEW_PATH = "omninxt_front_cam_preview.jpg"
OMNINXT_PREVIEW_PATH_TEMPLATE = "omninxt_{camera_name}_{sensor_name}_preview.jpg"
OMNINXT_PREVIEW_SETTLE_FRAMES = 20
OMNINXT_GT_RANGE_EXPORT_ENABLED = os.environ.get("OMNINXT_GT_RANGE_EXPORT", "0") == "1"
OMNINXT_DEPTH_EXPORT_ENABLED = os.environ.get("OMNINXT_DEPTH_EXPORT", "0") == "1"
OMNINXT_DEPTH_EXPORT_ROOT = os.path.expanduser(
    os.environ.get(
        "OMNINXT_DEPTH_EXPORT_ROOT",
        os.path.join(ZYW_ROOT, "our_omni_depth", "shared"),
    )
)
OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES = int(
    os.environ.get("OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES", "30")
)
OMNINXT_DEPTH_EXPORT_COUNT = int(os.environ.get("OMNINXT_DEPTH_EXPORT_COUNT", "1"))
OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES = int(
    os.environ.get("OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES", "120")
)
OMNINXT_DEPTH_EXPORT_NAV_ONLY = (
    os.environ.get("OMNINXT_DEPTH_EXPORT_NAV_ONLY", "0") == "1"
)
OMNINXT_EXIT_AFTER_EXPORT = os.environ.get("OMNINXT_EXIT_AFTER_EXPORT", "0") == "1"
OMNINXT_LIVE_STREAM_ENABLED = os.environ.get("OMNINXT_LIVE_STREAM", "0") == "1"
OMNINXT_LIVE_STREAM_ROOT = os.environ.get(
    "OMNINXT_LIVE_STREAM_ROOT", "/dev/shm/omninxt_sync"
)
OMNINXT_LIVE_STREAM_HZ = float(os.environ.get("OMNINXT_LIVE_STREAM_HZ", "10"))

# ===== Dataset action and reward labels =====
# A fixed normalized action space is shared by offline data and the future
# Dreamer policy. Physical commands are still saved alongside normalized ones.
DATA_ACTION_NORMALIZATION = {
    "vx_body_mps": 2.15,
    "vy_body_mps": 2.15,
    "vz_world_mps": 1.0,
    "yaw_rate_rps": 0.8,
}

# Reward is attached to the transition ending at each saved frame. Continuous
# rates are multiplied by the measured simulation-time delta, while terminal
# event rewards are one-off values.
DATA_REWARD_CONFIG = {
    "progress_weight_per_m": 2.0,
    "max_progress_speed_mps": 3.0,
    "goal_height_tolerance_m": 0.20,
    "human_clearance_weight_per_sec": 4.0,
    "human_hard_clearance_m": 0.10,
    "human_safe_clearance_m": 0.70,
    "drone_collision_radius_m": 0.32,
    "human_ttc_horizon_sec": 2.5,
    "human_predictive_safe_clearance_m": 0.90,
    "flight_corridor_half_width_m": 0.75,
    "flight_corridor_lookahead_m": 4.0,
    "acceleration_weight_per_sec": 0.15,
    "acceleration_scale_mps2": 3.0,
    "jerk_weight_per_sec": 0.10,
    "jerk_scale_mps3": 10.0,
    "yaw_rate_weight_per_sec": 0.05,
    "yaw_rate_scale_rps": 1.5,
    "smooth_term_clip": 4.0,
    "acceleration_filter_alpha": 0.25,
    "height_weight_per_sec": 0.5,
    "cruise_height_m": 1.0,
    "height_tolerance_m": 0.15,
    "height_scale_m": 0.50,
    "time_cost_per_sec": 0.10,
    "event_rewards": {
        "reached_goal": 100.0,
        "collision": -120.0,
        "out_of_bounds": -100.0,
        "crash": -100.0,
        "stuck_timeout": -20.0,
        "record_size_limit": 0.0,
        "manual_stop": 0.0,
        "shutdown": 0.0,
    },
}

# ===== Control mode =====
# "gamepad" keeps the existing joystick workflow.
# "classic" uses keyboard-triggered A*/grid + local velocity-obstacle control
# through the local Pegasus Python backend.
# "px4_classic" keeps the same A*/grid + local velocity-obstacle planner, but
# replaces the local backend with PX4 SITL and sends offboard velocity commands
# through MAVSDK so the command/telemetry interface matches real flights.
CONTROL_MODE = "px4_classic"

# ===== PX4 + MAVSDK control =====
PX4_VEHICLE_ID = 0
PX4_AUTOLAUNCH = True
PX4_CLEAN_STALE_PROCESSES_ON_START = True
# Never fall back to Pegasus' machine-specific default. PX4_DIR is supplied by
# .local/machine.env and can still be overridden for an individual invocation.
PX4_DIR = os.path.expanduser(
    os.environ.get("PX4_DIR", os.path.join(ZYW_ROOT, "PX4-Autopilot"))
)
PX4_VEHICLE_MODEL = None  # None means use PegasusInterface().px4_default_airframe
PX4_MAVLINK_CONNECTION_TYPE = "tcpin"
PX4_MAVLINK_CONNECTION_IP = "localhost"
PX4_MAVLINK_CONNECTION_BASEPORT = 4560
PX4_ENABLE_LOCKSTEP = True
PX4_BACKEND_UPDATE_RATE_HZ = 250.0

MAVSDK_SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
MAVSDK_COMMAND_HZ = DATA_CONTROL_HZ
MAVSDK_CONNECT_TIMEOUT_SEC = 30.0
MAVSDK_HEALTH_TIMEOUT_SEC = 15.0
MAVSDK_OFFBOARD_RETRY_SEC = 2.0
MAVSDK_USE_TELEMETRY_STATE = True
MAVSDK_TELEMETRY_STALE_SEC = 0.75
MAVSDK_START_OFFBOARD_ON_TAKEOFF = True
# Current planner uses body FLU/z-up style commands:
# vx_body forward, vy_body left, vz_world up, yaw_rate positive z-up. MAVSDK
# body setpoints use FRD/NED style: forward, right, down, positive clockwise.
MAVSDK_BODY_RIGHT_SIGN = -1.0
MAVSDK_BODY_DOWN_SIGN = -1.0
MAVSDK_YAWSPEED_SIGN = -1.0

PX4_RESTART_BETWEEN_EPISODES = True
PX4_RESTART_DELAY_SEC = 1.0
PX4_LAND_BEFORE_EPISODE_RESET = False
PX4_EPISODE_LAND_TIMEOUT_SEC = 20.0
PX4_EPISODE_LAND_Z_THRESHOLD = 0.35
PX4_EPISODE_LAND_VZ_THRESHOLD = 0.35

# ===== Classic autonomous controller =====
# Keep automatic mission startup selectable per launcher.  "0" is the default
# used by the Omni-Depth capture launchers so the vehicle waits for T.
CLASSIC_AUTO_START = os.environ.get("CLASSIC_AUTO_START", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
CLASSIC_START_KEY = os.environ.get("CLASSIC_START_KEY", "T")
CLASSIC_NAV_USE_MAVSDK_TELEMETRY = True
CLASSIC_WAIT_FOR_GROUND_BEFORE_TAKEOFF = True
CLASSIC_GROUND_Z_THRESHOLD = 0.35
CLASSIC_GROUND_VZ_THRESHOLD = 0.35
CLASSIC_GROUND_SETTLE_SEC = 2.0
CLASSIC_GROUND_MAX_WAIT_SEC = 4.0
CLASSIC_TAKEOFF_HEIGHT = 1.0
CLASSIC_CRUISE_HEIGHT = 1.0
CLASSIC_TAKEOFF_SETTLE_SEC = 4.0
CLASSIC_TAKEOFF_RECORD_MIN_Z = 0.90
CLASSIC_TAKEOFF_MAX_WAIT_SEC = 45.0
CLASSIC_TAKEOFF_LOG_INTERVAL_SEC = 1.0
CLASSIC_NAV_LOG_INTERVAL_SEC = 1.0
CLASSIC_GRID_RESOLUTION = 0.75
CLASSIC_STATIC_OBSTACLE_CLEARANCE = 0.45
CLASSIC_PATH_REPLAN_INTERVAL_SEC = 1.5
CLASSIC_WAYPOINT_REACH_DISTANCE = 0.85
CLASSIC_PATH_LOOKAHEAD_DISTANCE = 3.5
CLASSIC_GOAL_REGION_X_GUARD = 0.25
CLASSIC_FINAL_APPROACH_DISTANCE = 5.0
CLASSIC_FINAL_MAX_SPEED = 0.65
CLASSIC_FINAL_LATERAL_SPEED = 0.20
CLASSIC_FINAL_YAW_RATE_SCALE = 0.35
CLASSIC_COMMAND_SMOOTHING_ALPHA = 0.35
CLASSIC_COMMAND_MAX_ACCEL_MPS2 = 1.20
CLASSIC_FINAL_COMMAND_MAX_ACCEL_MPS2 = 0.55
CLASSIC_MAX_SPEED = 2.15
CLASSIC_MIN_SPEED = 0.45
CLASSIC_SLOW_RADIUS = 1.4
CLASSIC_MAX_Z_SPEED = 0.65
CLASSIC_Z_KP = 1.0
CLASSIC_YAW_KP = 2.2
CLASSIC_ORCA_TIME_HORIZON = 1.4
CLASSIC_ORCA_NEIGHBOR_RADIUS = 4.5
CLASSIC_DRONE_RADIUS = 0.32
CLASSIC_PEDESTRIAN_RADIUS = 0.28
CLASSIC_AVOIDANCE_MARGIN = 0.10
CLASSIC_HARD_AVOIDANCE_MARGIN = 0.08
CLASSIC_SOFT_AVOIDANCE_MARGIN = 0.28
CLASSIC_DESIRED_VELOCITY_WEIGHT = 0.55
CLASSIC_GOAL_PROGRESS_WEIGHT = 5.0
CLASSIC_LATERAL_DETOUR_WEIGHT = 0.8
CLASSIC_BACKTRACK_PENALTY = 8.0
CLASSIC_VELOCITY_CHANGE_WEIGHT = 0.25
CLASSIC_FINAL_VELOCITY_CHANGE_WEIGHT = 3.0
CLASSIC_SOFT_CLEARANCE_WEIGHT = 1.8
CLASSIC_STATIC_LOOKAHEAD_SEC = 0.8
CLASSIC_PERSON_VEL_FILTER = 0.45
CLASSIC_MAX_PERSON_SPEED_ESTIMATE = 2.5

SCENE_DRONE_X_RANGE = ACTIVE_SCENE_PRESET["drone_x_range"]
SCENE_DRONE_Y_RANGE = ACTIVE_SCENE_PRESET["drone_y_range"]
# Both current scene presets place the random drone spawn strip against the
# lower edge of the pedestrian polygon.  Clip that entire lower band from the
# walkable polygon, with enough room for the drone footprint and a pedestrian
# body, so nobody can enter the takeoff area from a narrow side corridor.
PEDESTRIAN_DRONE_SPAWN_CLEARANCE = float(
    os.environ.get("PEDESTRIAN_DRONE_SPAWN_CLEARANCE", "1.2")
)
PEDESTRIAN_DRONE_SAFE_Y_MIN = (
    max(float(value) for value in SCENE_DRONE_Y_RANGE)
    + PEDESTRIAN_DRONE_SPAWN_CLEARANCE
)
PEDESTRIAN_GOAL_CLEARANCE = float(
    os.environ.get("PEDESTRIAN_GOAL_CLEARANCE", "1.2")
)
PEDESTRIAN_GOAL_SAFE_Y_MAX = DATASET_GOAL_Y_MIN - PEDESTRIAN_GOAL_CLEARANCE
PEDESTRIAN_WALK_POLYGON = [
    (
        float(x),
        min(
            max(float(y), PEDESTRIAN_DRONE_SAFE_Y_MIN),
            PEDESTRIAN_GOAL_SAFE_Y_MAX,
        ),
    )
    for x, y in ACTIVE_SCENE_PRESET["walk_polygon"]
]
SCENE_DIRECTION_START_BOUNDS = ACTIVE_SCENE_PRESET["direction_start_bounds"]
# Warehouse V2 uses static-geometry AABBs both for spawn placement and route
# planning. CityTower retains its original obstacle filter.
ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE = True
PEDESTRIAN_OBSTACLE_KEYWORDS = (
    [
        "Wall",
        "Rack",
        "Shelf",
        "Forklift",
        "Pallet",
        "CardBox",
        "Va_Box",
        "FuseBox",
        "Column",
        "Pillar",
    ]
    if SCENE_PRESET == "warehouse"
    else ["var_Bench", "Bench", "Bollard"]
)
PEDESTRIAN_OBSTACLE_ROOT_PATHS = (
    None
    if SCENE_PRESET == "warehouse"
    else ["/World/layout/assembly_CornerLot"]
)
# PillarPartA combines ground posts and overhead roof beams in one prim. Its
# projected AABB covers empty floor and is not a valid pedestrian obstacle.
PEDESTRIAN_OBSTACLE_EXCLUDE_KEYWORDS = (
    ["PillarPartA"] if SCENE_PRESET == "warehouse" else []
)
PEDESTRIAN_OBSTACLE_MAX_BOTTOM_Z = 0.35
PEDESTRIAN_OBSTACLE_MIN_TOP_Z = 0.15
PEDESTRIAN_OBSTACLE_MARGIN = 0.3
PEDESTRIAN_PERSONAL_SPACE = float(
    os.environ.get("PEDESTRIAN_PERSONAL_SPACE", "1.00")
)
PEDESTRIAN_SEPARATION_ACTIVE_DISTANCE = float(
    os.environ.get("PEDESTRIAN_SEPARATION_ACTIVE_DISTANCE", "1.60")
)
PEDESTRIAN_SEPARATION_STEP = 0.85
PEDESTRIAN_SEPARATION_STRENGTH = 1.55
PEDESTRIAN_PREDICTIVE_LOOKAHEAD_TIME = 2.2
PEDESTRIAN_PREDICTIVE_CONFLICT_DISTANCE = 0.9
PEDESTRIAN_PREDICTIVE_TIME_MARGIN = 0.8
PEDESTRIAN_PREDICTIVE_SIDE_STEP = 1.0
PEDESTRIAN_SAME_GROUP_MIN_DISTANCE = float(
    os.environ.get("PEDESTRIAN_SAME_GROUP_MIN_DISTANCE", "0.30")
)
PEDESTRIAN_INITIAL_MIN_DISTANCE = float(
    os.environ.get("PEDESTRIAN_INITIAL_MIN_DISTANCE", "1.15")
)
PEDESTRIAN_MAX_AVOIDANCE_TURN_DEG = float(
    os.environ.get("PEDESTRIAN_MAX_AVOIDANCE_TURN_DEG", "60")
)
PEDESTRIAN_CLEARANCE_CHECK_INTERVAL_SEC = 0.25
PEDESTRIAN_CLEARANCE_LOG_INTERVAL_SEC = 2.0
CROWD_MAP_STATE_PATH = os.environ.get(
    "CROWD_MAP_STATE_PATH", "/tmp/isaac_crowd_map_state.json"
).strip()
CROWD_MAP_EXPORT_INTERVAL_SEC = float(
    os.environ.get("CROWD_MAP_EXPORT_INTERVAL_SEC", "0.10")
)
# The Pegasus Person callback remains connected to the 250 Hz physics clock,
# but uses simulation-dt accumulation to perform all character state,
# controller, and AnimationGraph work at this bounded rate.
PEGASUS_PEOPLE_CONTROL_HZ = float(
    os.environ.get("PEGASUS_PEOPLE_CONTROL_HZ", "25")
)
PEGASUS_PEOPLE_STATE_CALLBACK = int(
    os.environ.get("PEGASUS_PEOPLE_STATE_CALLBACK", "0")
)
if PEGASUS_PEOPLE_CONTROL_HZ <= 0.0:
    raise ValueError("PEGASUS_PEOPLE_CONTROL_HZ must be greater than zero")
if PEGASUS_PEOPLE_STATE_CALLBACK != 0:
    raise ValueError(
        "PEGASUS_PEOPLE_STATE_CALLBACK must remain 0 because person state "
        "sampling is merged into the single people callback"
    )
# Keep the crowd controller rate aligned with the outer Person rate.  A
# separate override remains available for controlled experiments.
PEDESTRIAN_CONTROLLER_HZ = float(
    os.environ.get(
        "PEDESTRIAN_CONTROLLER_HZ", str(PEGASUS_PEOPLE_CONTROL_HZ)
    )
)
PEDESTRIAN_SKELETON_UPDATE_HZ = float(
    os.environ.get("PEDESTRIAN_SKELETON_UPDATE_HZ", "10")
)
if PEDESTRIAN_CONTROLLER_HZ <= 0.0:
    raise ValueError("PEDESTRIAN_CONTROLLER_HZ must be greater than zero")
if PEDESTRIAN_SKELETON_UPDATE_HZ <= 0.0:
    raise ValueError("PEDESTRIAN_SKELETON_UPDATE_HZ must be greater than zero")
PEDESTRIAN_REPLAN_INTERVAL_SEC = float(
    os.environ.get("PEDESTRIAN_REPLAN_INTERVAL_SEC", "1.0")
)
PEDESTRIAN_PATH_SAMPLE_ATTEMPTS = int(
    os.environ.get("PEDESTRIAN_PATH_SAMPLE_ATTEMPTS", "30")
)
PEDESTRIAN_FALLBACK_SAMPLE_ATTEMPTS = int(
    os.environ.get("PEDESTRIAN_FALLBACK_SAMPLE_ATTEMPTS", "30")
)
PEDESTRIAN_HOLD_LOG_INTERVAL_SEC = float(
    os.environ.get("PEDESTRIAN_HOLD_LOG_INTERVAL_SEC", "5.0")
)

# ===== Crowd template =====
# num_people: citytower uses 10-20; warehouse uses 17-23.
# group_spacing: "close" ~= 0.8-1.0 m, "far" ~= 1.8-2.0 m
# direction: "same_direction", "opposite_direction", "left_to_right",
#            "right_to_left", "random"
# left_to_right/right_to_left both mean horizontal crossing; each pedestrian
# starts inside the active scene's walk polygon and loops at the boundary.
# drone spawn uses the active scene's drone_x_range/drone_y_range.
# random picks one of the four directed walking modes per group.
# speed: "slow" ~= 0.65-0.95 m/s, "fast" ~= 1.15-1.55 m/s
# seed: one integer deterministically fixes population, grouping, routes,
# speeds, initial directions, pauses, and crossing reservations.
CROWD_RANDOMIZE_TEMPLATE = SCENE_PRESET != "warehouse"
CROWD_RANDOM_NUM_PEOPLE_CHOICES = tuple(ACTIVE_SCENE_PRESET["num_people_choices"])
CROWD_RANDOM_GROUP_SPACING_CHOICES = VALID_GROUP_SPACINGS
CROWD_RANDOM_DIRECTION_CHOICES = VALID_DIRECTIONS
CROWD_RANDOM_DRONE_DISTANCE_CHOICES = VALID_DRONE_DISTANCES
CROWD_RANDOM_SPEED_CHOICES = VALID_SPEEDS

_WAREHOUSE_CROWD_SEED_TEXT = os.environ.get("WAREHOUSE_CROWD_SEED", "1").strip()
WAREHOUSE_CROWD_SEED = (
    int(_WAREHOUSE_CROWD_SEED_TEXT) if _WAREHOUSE_CROWD_SEED_TEXT else None
)
if WAREHOUSE_CROWD_SEED is not None:
    random.seed(WAREHOUSE_CROWD_SEED)
    np.random.seed(WAREHOUSE_CROWD_SEED & 0xFFFFFFFF)
WAREHOUSE_CROWD_MODE = os.environ.get(
    "WAREHOUSE_CROWD_MODE", "server_v2"
).strip().lower()
if WAREHOUSE_CROWD_MODE not in ("server_v2", "legacy"):
    raise ValueError("WAREHOUSE_CROWD_MODE must be server_v2 or legacy")
WAREHOUSE_CROWD_LAYOUT = os.environ.get(
    "WAREHOUSE_CROWD_LAYOUT", "sparse"
).strip().lower()
if WAREHOUSE_CROWD_LAYOUT not in ("sparse", "dense"):
    raise ValueError("WAREHOUSE_CROWD_LAYOUT must be sparse or dense")
WAREHOUSE_DENSE_PROFILE = os.environ.get(
    "WAREHOUSE_DENSE_PROFILE", "transverse40"
).strip().lower()
if WAREHOUSE_DENSE_PROFILE not in ("transverse40", "legacy"):
    raise ValueError("WAREHOUSE_DENSE_PROFILE must be transverse40 or legacy")
if WAREHOUSE_CROWD_LAYOUT == "dense" and WAREHOUSE_CROWD_MODE != "server_v2":
    raise ValueError("dense layout requires WAREHOUSE_CROWD_MODE=server_v2")
_WAREHOUSE_COUNT_TEXT = os.environ.get("WAREHOUSE_CROWD_COUNT", "").strip()
if WAREHOUSE_CROWD_LAYOUT == "dense" and WAREHOUSE_DENSE_PROFILE == "transverse40":
    WAREHOUSE_CROWD_COUNT = int(_WAREHOUSE_COUNT_TEXT or "40")
    if WAREHOUSE_CROWD_COUNT != 40:
        raise ValueError("dense + transverse40 requires WAREHOUSE_CROWD_COUNT=40")
else:
    WAREHOUSE_CROWD_COUNT = (
        int(_WAREHOUSE_COUNT_TEXT)
        if _WAREHOUSE_COUNT_TEXT
        else random.Random(WAREHOUSE_CROWD_SEED).choice(
            tuple(ACTIVE_SCENE_PRESET["num_people_choices"])
        )
    )
WAREHOUSE_CROWD_TEMPLATE_VERSION = os.environ.get(
    "WAREHOUSE_CROWD_TEMPLATE_VERSION",
    "warehouse_v2" if WAREHOUSE_CROWD_MODE == "server_v2" else "legacy",
).strip().lower()
_WAREHOUSE_DEFAULT_DIRECTION = (
    "random_heading"
    if WAREHOUSE_CROWD_TEMPLATE_VERSION == "warehouse_v2"
    else "random"
)
WAREHOUSE_CROWD_DIRECTION = os.environ.get(
    "WAREHOUSE_CROWD_DIRECTION", _WAREHOUSE_DEFAULT_DIRECTION
).strip().lower()
PEDESTRIAN_ROUTE_X_MIN = (
    float(os.environ.get("PEDESTRIAN_ROUTE_X_MIN", "-7.0"))
    if SCENE_PRESET == "warehouse"
    else None
)

CROWD_TEMPLATE = {
    "num_people": (
        WAREHOUSE_CROWD_COUNT
        if SCENE_PRESET == "warehouse"
        else ACTIVE_SCENE_PRESET["default_num_people"]
    ),
    "group_spacing": "close" if SCENE_PRESET == "warehouse" else "far",
    "direction": (
        WAREHOUSE_CROWD_DIRECTION
        if SCENE_PRESET == "warehouse"
        else "left_to_right"
    ),
    "drone_distance": "near",
    "speed": "fast",
    "seed": WAREHOUSE_CROWD_SEED if SCENE_PRESET == "warehouse" else None,
    "walk_polygon": PEDESTRIAN_WALK_POLYGON,
    "valid_people_counts": CROWD_RANDOM_NUM_PEOPLE_CHOICES,
    "drone_x_range": SCENE_DRONE_X_RANGE,
    "drone_y_range": SCENE_DRONE_Y_RANGE,
    "direction_start_bounds": SCENE_DIRECTION_START_BOUNDS,
    # Warehouse V2 crowd route/motion boundary. Keep the global
    # PEDESTRIAN_WALK_POLYGON unchanged so the UI and obstacle scan retain the
    # full room; each Warehouse V2 controller receives this clipped polygon.
    "route_x_min": PEDESTRIAN_ROUTE_X_MIN,
    "crowd_layout": WAREHOUSE_CROWD_LAYOUT,
    "dense_profile": WAREHOUSE_DENSE_PROFILE,
    "template_version": (
        WAREHOUSE_CROWD_TEMPLATE_VERSION
        if SCENE_PRESET == "warehouse"
        else "legacy"
    ),
}
CROWD_POOL_PEOPLE_COUNT = (
    max(CROWD_RANDOM_NUM_PEOPLE_CHOICES)
    if CROWD_RANDOMIZE_TEMPLATE
    else CROWD_TEMPLATE["num_people"]
)


def sample_crowd_template(num_people=None):
    template = dict(CROWD_TEMPLATE)
    template.update(
        {
            "num_people": (
                int(num_people)
                if num_people is not None
                else random.choice(tuple(CROWD_RANDOM_NUM_PEOPLE_CHOICES))
            ),
            "group_spacing": random.choice(tuple(CROWD_RANDOM_GROUP_SPACING_CHOICES)),
            "direction": random.choice(tuple(CROWD_RANDOM_DIRECTION_CHOICES)),
            "drone_distance": random.choice(tuple(CROWD_RANDOM_DRONE_DISTANCE_CHOICES)),
            "speed": random.choice(tuple(CROWD_RANDOM_SPEED_CHOICES)),
            "seed": None,
            "walk_polygon": PEDESTRIAN_WALK_POLYGON,
            "valid_people_counts": CROWD_RANDOM_NUM_PEOPLE_CHOICES,
            "drone_x_range": SCENE_DRONE_X_RANGE,
            "drone_y_range": SCENE_DRONE_Y_RANGE,
            "direction_start_bounds": SCENE_DIRECTION_START_BOUNDS,
        }
    )
    return template


ACTIVE_CROWD_TEMPLATE = (
    sample_crowd_template() if CROWD_RANDOMIZE_TEMPLATE else dict(CROWD_TEMPLATE)
)
ACTIVE_CROWD_SCENE = build_crowd_scene(**ACTIVE_CROWD_TEMPLATE)
if SCENE_PRESET == "warehouse" and WAREHOUSE_CROWD_SEED is not None:
    TARGET_POINT = sample_warehouse_goal(
        WAREHOUSE_CROWD_SEED,
        ACTIVE_SCENE_PRESET["goal_x_range"],
        ACTIVE_SCENE_PRESET.get("goal_y_range", (DATASET_GOAL_Y_MIN,) * 2),
    )
DRONE_SPAWN_Z = 0.45
SPAWN_POS = list(ACTIVE_CROWD_SCENE.drone_spawn)
SPAWN_POS[2] = DRONE_SPAWN_Z
DRONE_INIT_YAW_DEG = 90.0

TARGET_JOINTS = [
    "Pelvis",
    "R_Hand",
    "L_Hand",
    "R_Foot",
    "L_Foot",
    "R_KneeShareBone",
    "L_KneeShareBone",
    "R_ElbowShareBone",
    "L_ElbowShareBone",
    "Head",
]

# ===== Gamepad velocity =====
VX = 2.0
VY = 2.0
VZ = 1.0
YAW_RATE = 0.8
