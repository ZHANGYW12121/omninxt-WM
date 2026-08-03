#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carb
import json
import math
import os
import numpy as np
import omni.timeline
import omni.usd
import signal
import time
from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from omni.isaac.core.world import World
from omni.isaac.core.objects import FixedCuboid
from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera

from skeleton_dataset_recorder import SkeletonStateDatasetRecorder
from skeleton_packet_receiver import SkeletonPacketReceiver
from keyboard_backend import SharedCommand, KeyboardVelocityController
from classic_controller import ClassicAlgorithmController
from mavsdk_bridge import MavsdkOffboardBridge

from app_config import (
    ACTIVE_CROWD_SCENE,
    CLASSIC_NAV_USE_MAVSDK_TELEMETRY,
    CLASSIC_TAKEOFF_HEIGHT,
    CONTROL_MODE,
    CROWD_MAP_EXPORT_INTERVAL_SEC,
    CROWD_MAP_STATE_PATH,
    CROWD_TEMPLATE,
    CROWD_POOL_PEOPLE_COUNT,
    CROWD_RANDOMIZE_TEMPLATE,
    DATA_ACTION_NORMALIZATION,
    DATA_CAMERA_RESOLUTION,
    DATA_CONTROL_HZ,
    DATA_IMAGE_FORMAT,
    DATA_IMAGE_JPEG_QUALITY,
    DATA_DROP_WHEN_WRITER_BUSY,
    DATA_JSON_INDENT,
    DATA_RECORD_ENABLED,
    DATA_RECORD_CHUNK_FRAMES,
    DATA_RECORD_HZ,
    DATA_RECORD_MAX_RECORD_BYTES,
    DATA_RECORD_MAX_TRAJECTORIES,
    DATA_RECORD_QUEUE_SIZE,
    DATA_RECORD_PRIVILEGED_MAX_PEOPLE,
    DATA_RECORD_SKELETON_HOST,
    DATA_RECORD_SKELETON_PORT,
    DATA_RECORD_STORAGE_MAX_PEOPLE,
    DATA_RECORD_SYNC_TOLERANCE_SEC,
    DATA_RECORD_SIZE_CHECK_INTERVAL_SEC,
    DATA_RECORD_STUCK_TIMEOUT_SEC,
    DATA_REWARD_CONFIG,
    DATASET_ROOT,
    DATASET_GOAL_X_RANGE,
    DATASET_GOAL_Y_MIN,
    DRONE_SPAWN_Z,
    DRONE_INIT_YAW_DEG,
    ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE,
    PEDESTRIAN_OBSTACLE_EXCLUDE_KEYWORDS,
    PEDESTRIAN_OBSTACLE_KEYWORDS,
    PEDESTRIAN_OBSTACLE_MARGIN,
    PEDESTRIAN_OBSTACLE_MAX_BOTTOM_Z,
    PEDESTRIAN_OBSTACLE_MIN_TOP_Z,
    PEDESTRIAN_OBSTACLE_ROOT_PATHS,
    PEDESTRIAN_CONTROLLER_HZ,
    PEDESTRIAN_CLEARANCE_CHECK_INTERVAL_SEC,
    PEDESTRIAN_CLEARANCE_LOG_INTERVAL_SEC,
    PEDESTRIAN_FALLBACK_SAMPLE_ATTEMPTS,
    PEDESTRIAN_HOLD_LOG_INTERVAL_SEC,
    PEDESTRIAN_INITIAL_MIN_DISTANCE,
    PEDESTRIAN_MAX_AVOIDANCE_TURN_DEG,
    PEDESTRIAN_PATH_SAMPLE_ATTEMPTS,
    PEDESTRIAN_PERSONAL_SPACE,
    PEDESTRIAN_PREDICTIVE_CONFLICT_DISTANCE,
    PEDESTRIAN_PREDICTIVE_LOOKAHEAD_TIME,
    PEDESTRIAN_PREDICTIVE_SIDE_STEP,
    PEDESTRIAN_PREDICTIVE_TIME_MARGIN,
    PEDESTRIAN_REPLAN_INTERVAL_SEC,
    PEDESTRIAN_SAME_GROUP_MIN_DISTANCE,
    PEDESTRIAN_SEPARATION_ACTIVE_DISTANCE,
    PEDESTRIAN_SEPARATION_STEP,
    PEDESTRIAN_SEPARATION_STRENGTH,
    PEDESTRIAN_SKELETON_UPDATE_HZ,
    PEDESTRIAN_WALK_POLYGON,
    PEGASUS_PEOPLE_CONTROL_HZ,
    PEGASUS_PEOPLE_STATE_CALLBACK,
    MAVSDK_COMMAND_HZ,
    MAVSDK_BODY_DOWN_SIGN,
    MAVSDK_BODY_RIGHT_SIGN,
    MAVSDK_CONNECT_TIMEOUT_SEC,
    MAVSDK_HEALTH_TIMEOUT_SEC,
    MAVSDK_SYSTEM_ADDRESS,
    MAVSDK_YAWSPEED_SIGN,
    MAVSDK_USE_TELEMETRY_STATE,
    MVD35_BODY_MASS_KG,
    MVD35_CENTER_OF_MASS_M,
    MVD35_DIAGONAL_INERTIA_KGM2,
    MVD35_LINEAR_DRAG_COEFFICIENTS,
    MVD35_PX4_INPUT_OFFSET,
    MVD35_PX4_INPUT_SCALING,
    MVD35_PX4_ZERO_POSITION_ARMED,
    MVD35_ROTOR_DIAGONAL_INERTIA_KGM2,
    MVD35_ROTOR_MASS_KG,
    MVD35_ROTOR_POSITIONS_FLU_M,
    MVD35_SIM2REAL_ENABLED,
    MVD35_THRUST_CURVE_CONFIG,
    MVD35_TOTAL_MASS_KG,
    OMNINXT_ACTIVE_CAMERA,
    OMNINXT_VISUAL_BODY_TRANSLATION_M,
    OMNINXT_VISUAL_BODY_USD,
    OMNINXT_VISUAL_BODY_YAW_DEG,
    OMNINXT_VISUAL_ENABLED,
    OMNINXT_VISUAL_ROTOR_ASSET_TRANSLATIONS_M,
    OMNINXT_VISUAL_ROTOR_INDICES,
    OMNINXT_VISUAL_ROTOR_POSITIONS_FLU_M,
    OMNINXT_VISUAL_ROTOR_USDS,
    OMNINXT_VISUAL_SCALE_CORRECTION,
    OMNINXT_CAMERA_NAMES,
    OMNINXT_CAMERA_CONFIG_PATH,
    OMNINXT_CAMERA_ENABLED,
    OMNINXT_CAMERA_SENSOR_NAMES,
    OMNINXT_SENSOR_MODE,
    OMNINXT_PREVIEW_PATH,
    OMNINXT_PREVIEW_PATH_TEMPLATE,
    OMNINXT_PREVIEW_SAVE_ON_START,
    OMNINXT_PREVIEW_SETTLE_FRAMES,
    OMNINXT_DEPTH_EXPORT_ENABLED,
    OMNINXT_GT_RANGE_EXPORT_ENABLED,
    OMNINXT_DEPTH_EXPORT_COUNT,
    OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES,
    OMNINXT_DEPTH_EXPORT_NAV_ONLY,
    OMNINXT_DEPTH_EXPORT_ROOT,
    OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES,
    OMNINXT_EXIT_AFTER_EXPORT,
    OMNINXT_LIVE_STREAM_ENABLED,
    OMNINXT_LIVE_STREAM_HZ,
    OMNINXT_LIVE_STREAM_ROOT,
    OMNINXT_VIEW_WINDOWS_ENABLED,
    OMNINXT_VIEW_WINDOW_CAMERA_NAMES,
    OMNINXT_VIEW_WINDOW_LAYOUT,
    PX4_EPISODE_LAND_TIMEOUT_SEC,
    PX4_EPISODE_LAND_VZ_THRESHOLD,
    PX4_EPISODE_LAND_Z_THRESHOLD,
    PX4_LAND_BEFORE_EPISODE_RESET,
    PX4_AUTOLAUNCH,
    PX4_BACKEND_UPDATE_RATE_HZ,
    PX4_CLEAN_STALE_PROCESSES_ON_START,
    PX4_DIR,
    PX4_ENABLE_LOCKSTEP,
    PX4_MAVLINK_CONNECTION_BASEPORT,
    PX4_MAVLINK_CONNECTION_IP,
    PX4_MAVLINK_CONNECTION_TYPE,
    PX4_RESTART_BETWEEN_EPISODES,
    PX4_RESTART_DELAY_SEC,
    PX4_VEHICLE_ID,
    PX4_VEHICLE_MODEL,
    SPAWN_POS,
    TARGET_JOINTS,
    TARGET_POINT,
    USD_PATH,
    SIM_PHYSICS_DT,
    SIM_RENDERING_DT,
    sample_crowd_template,
)
from warehouse_crowd_v2.crowd_templates import build_crowd_scene
from drone_camera_utils import (
    configure_omninxt_camera,
    create_camera_viewport_window,
    find_camera_named_under,
    find_first_camera_under,
    load_omninxt_camera_config,
    omninxt_camera_fps,
    omninxt_camera_resolution,
    save_camera_rgb_preview,
    set_active_viewport_perspective,
    set_camera_pose_and_view,
)
from omni_depth_exporter import IsaacQuadcamExporter
from omninxt_visual import apply_omninxt_visual
from gamepad_controller import GamepadController
from geometry_utils import (
    nearest_front_walkable_or_sample,
    nearest_walkable_or_sample,
    point_is_front_walkable,
    point_in_polygon_2d,
)
from mvd35_dynamics import FirstOrderQuadraticThrustCurve
from obstacle_utils import ObstacleScanResult, sample_paths_under_roots, scan_obstacles
from warehouse_crowd_v2.person_controllers import (
    NaturalFormationWaypointController,
    ObstacleAwarePolygonPersonController,
    WaypointPersonController,
)
from skeleton_tracker import MultiPersonSkeletonTracker


def _debug_log(*args, **kwargs):
    pass


class PegasusApp:
    def __init__(self, simulation_app, control_mode=None, headless=False):
        self.simulation_app = simulation_app
        self.timeline = omni.timeline.get_timeline_interface()
        self.headless = bool(headless)
        self.control_mode = (control_mode or CONTROL_MODE).strip().lower()
        if self.control_mode not in (
            "gamepad",
            "classic",
            "px4_classic",
        ):
            raise ValueError("control_mode must be 'gamepad', 'classic', or 'px4_classic'")

        self.pg = PegasusInterface()
        if MVD35_SIM2REAL_ENABLED:
            self.pg.set_world_settings(
                physics_dt=SIM_PHYSICS_DT,
                rendering_dt=SIM_RENDERING_DT,
            )
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        _debug_log(f"[APP] Loading environment: {USD_PATH}")
        self.pg.load_environment(USD_PATH)

        self.hidden_ground = FixedCuboid(
            prim_path="/World/hidden_ground",
            name="hidden_ground",
            position=[SPAWN_POS[0], SPAWN_POS[1], 0.0],
            scale=[20.0, 20.0, 0.2],
            visible=False,
        )

        self.shared_cmd = SharedCommand()

        drone_config = MultirotorConfig()
        if MVD35_SIM2REAL_ENABLED:
            drone_config.thrust_curve = FirstOrderQuadraticThrustCurve(
                MVD35_THRUST_CURVE_CONFIG
            )
            drone_config.drag = LinearDrag(MVD35_LINEAR_DRAG_COEFFICIENTS)
        self.keyboard_backend = None
        self.px4_backend = None
        self.mavsdk_bridge = None
        self._mavsdk_bridge_started = False
        self._mavsdk_bridge_wait_logged = False
        self._post_episode_landing_active = False
        self._post_episode_landing_reason = None
        self._post_episode_landing_start_wall = None
        self._post_episode_landing_ready_since = None
        self._px4_stack_restart_active = False

        self.omninxt_camera_enabled = bool(OMNINXT_CAMERA_ENABLED)
        self.omninxt_camera_config = None
        self.omninxt_camera_ids = tuple(OMNINXT_CAMERA_NAMES)
        self.omninxt_camera_sensor_names = dict(OMNINXT_CAMERA_SENSOR_NAMES)
        self.omninxt_active_camera = OMNINXT_ACTIVE_CAMERA
        self.omninxt_view_window_camera_names = tuple(
            OMNINXT_VIEW_WINDOW_CAMERA_NAMES
        )
        self.omninxt_camera_sensors = {}
        self.omninxt_camera_paths = {}
        self.omninxt_configured_cameras = set()
        self.omninxt_camera_applied = False
        self.omninxt_camera_view_windows = {}
        self._omninxt_preview_pending_frames = {}
        self._omninxt_preview_saved = set()
        self._omninxt_preview_error_logged = False
        self._omninxt_camera_error_logged = False
        self._omninxt_view_window_error_logged = False
        self._omni_depth_exporter = None
        self._omni_depth_export_done = False
        self._omni_depth_export_count = 0
        self._omni_depth_export_ready_frame = None
        self._omninxt_exit_requested = False
        self._omninxt_live_exporter = None
        self._omninxt_live_last_sim_time = None
        camera_frequency = DATA_RECORD_HZ
        camera_resolution = DATA_CAMERA_RESOLUTION
        if self.omninxt_camera_enabled:
            try:
                self.omninxt_camera_config = load_omninxt_camera_config(
                    OMNINXT_CAMERA_CONFIG_PATH,
                    profile=OMNINXT_SENSOR_MODE,
                )
                rig = self.omninxt_camera_config["camera_rig"]
                self.omninxt_camera_ids = tuple(
                    rig.get("camera_order", tuple(rig["cameras"]))
                )
                self.omninxt_camera_sensor_names = {
                    camera_id: str(
                        rig["cameras"][camera_id].get("sensor_name", camera_id)
                    )
                    for camera_id in self.omninxt_camera_ids
                }
                self.omninxt_active_camera = str(
                    rig.get("active_camera", self.omninxt_camera_ids[0])
                )
                self.omninxt_view_window_camera_names = (
                    self.omninxt_active_camera,
                )
                camera_frequency = omninxt_camera_fps(
                    self.omninxt_camera_config,
                    self.omninxt_active_camera,
                )
                camera_resolution = omninxt_camera_resolution(
                    self.omninxt_camera_config,
                    self.omninxt_active_camera,
                    render=True,
                )
                carb.log_warn(
                    f"[APP][CAM] OmniNxt 20260802 profile={rig.get('mode')} "
                    f"cameras={self.omninxt_camera_ids}, "
                    f"resolution={camera_resolution}, frequency={camera_frequency:.1f}Hz"
                )
            except Exception as exc:
                self.omninxt_camera_enabled = False
                carb.log_warn(f"[APP][CAM] Failed to load OmniNxt camera config: {exc}")

        if self.omninxt_camera_enabled:
            available_cameras = set(self.omninxt_camera_config["camera_rig"]["cameras"])
            if set(self.omninxt_camera_ids) != available_cameras:
                raise ValueError(
                    "OmniNxt camera_order does not exactly match camera specs"
                )
            if self.omninxt_active_camera not in self.omninxt_camera_ids:
                raise ValueError(
                    f"active_camera={self.omninxt_active_camera!r} is not in "
                    f"OMNINXT_CAMERA_NAMES={self.omninxt_camera_ids!r}"
                )
            for camera_id in self.omninxt_camera_ids:
                sensor_name = self.omninxt_camera_sensor_names.get(camera_id, camera_id)
                sensor_config = {
                    "frequency": omninxt_camera_fps(
                        self.omninxt_camera_config, camera_id
                    ),
                    "resolution": omninxt_camera_resolution(
                        self.omninxt_camera_config,
                        camera_id,
                        render=True,
                    ),
                    "depth": False,
                }
                self.omninxt_camera_sensors[camera_id] = MonocularCamera(
                    sensor_name,
                    config=sensor_config,
                )
            self.front_camera_sensor = self.omninxt_camera_sensors[
                self.omninxt_active_camera
            ]
        else:
            camera_config = {
                "frequency": camera_frequency,
                "resolution": camera_resolution,
                "depth": False,
            }
            self.front_camera_sensor = MonocularCamera(
                "front_cam",
                config=camera_config,
            )

        if self._uses_px4_backend():
            self.px4_backend = self._create_px4_backend()
            drone_config.backends = [self.px4_backend]
            self.mavsdk_bridge = MavsdkOffboardBridge(
                system_address=MAVSDK_SYSTEM_ADDRESS,
                command_hz=MAVSDK_COMMAND_HZ,
                connect_timeout_sec=MAVSDK_CONNECT_TIMEOUT_SEC,
                health_timeout_sec=MAVSDK_HEALTH_TIMEOUT_SEC,
            )
        else:
            self.keyboard_backend = KeyboardVelocityController(
                shared_cmd=self.shared_cmd,
                hover_height=CLASSIC_TAKEOFF_HEIGHT,
                mass_kg=MVD35_TOTAL_MASS_KG if MVD35_SIM2REAL_ENABLED else 1.50,
            )
            drone_config.backends = [self.keyboard_backend]
        if self.omninxt_camera_enabled:
            drone_config.graphical_sensors = [
                self.omninxt_camera_sensors[camera_id]
                for camera_id in self.omninxt_camera_ids
            ]
        else:
            drone_config.graphical_sensors = [self.front_camera_sensor]

        self.drone = Multirotor(
            "/World/quadrotor1",
            ROBOTS["Iris"],
            0,
            SPAWN_POS,
            Rotation.from_euler("XYZ", [0.0, 0.0, DRONE_INIT_YAW_DEG], degrees=True).as_quat(),
            config=drone_config,
        )
        self.drone_root_path = "/World/quadrotor1"
        self.current_spawn_pos = self._drone_spawn_pos(SPAWN_POS)
        if MVD35_SIM2REAL_ENABLED:
            self._apply_mvd35_vehicle_model()
        self.omninxt_visual = None
        if OMNINXT_VISUAL_ENABLED:
            try:
                self.omninxt_visual = apply_omninxt_visual(
                    omni.usd.get_context().get_stage(),
                    self.drone_root_path,
                    body_usd=OMNINXT_VISUAL_BODY_USD,
                    rotor_usds=OMNINXT_VISUAL_ROTOR_USDS,
                    rotor_positions=OMNINXT_VISUAL_ROTOR_POSITIONS_FLU_M,
                    rotor_asset_translations=OMNINXT_VISUAL_ROTOR_ASSET_TRANSLATIONS_M,
                    rotor_indices=OMNINXT_VISUAL_ROTOR_INDICES,
                    scale_correction=OMNINXT_VISUAL_SCALE_CORRECTION,
                    body_translation=OMNINXT_VISUAL_BODY_TRANSLATION_M,
                    body_yaw_deg=OMNINXT_VISUAL_BODY_YAW_DEG,
                    log=carb.log_warn,
                )
            except Exception as exc:
                carb.log_error(f"[APP][VISUAL] OmniNxt visual replacement failed: {exc}")

        self.walk_polygon = PEDESTRIAN_WALK_POLYGON
        self.obstacle_scan = self._scan_obstacles()
        self.obstacle_aabbs = self.obstacle_scan.aabbs
        if ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            self._try_load_initial_obstacles()
        self.crowd_scene = ACTIVE_CROWD_SCENE
        self.crowd_pool_scene = self._pool_scene_for_active_scene(self.crowd_scene)
        self.inactive_person_park_origin = [1000.0, 1000.0, 0.0]
        (
            self.people,
            self.person_controllers,
            self.person_initial_positions,
        ) = self._create_people_from_template(self.crowd_pool_scene)
        self._configure_crowd_separation()
        self._last_crowd_clearance_check_wall = 0.0
        self._last_crowd_clearance_log_wall = 0.0
        self._last_crowd_map_export_wall = 0.0
        self._crowd_map_export_error_logged = False

        self.world.reset()
        self._enable_drone_contact_reports()
        if self.crowd_pool_scene.num_people != self.crowd_scene.num_people:
            self._apply_crowd_scene_to_existing_people(self.crowd_scene)
        if ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            self._rescan_obstacles("post-reset")
            self._apply_obstacles_to_controllers()
            self._sanitize_people_initial_positions("post-reset")
        self._enforce_people_initial_positions()

        self._frame_count = 0
        self._timeline_was_playing = False
        self._timeline_stop_seen = False
        self.completed_trajectory_count = 0
        self.discarded_trajectory_count = 0
        self.trajectory_limit_reached = False
        self._last_record_size_check_wall = 0.0
        self._last_control_update_time = None

        if self.omninxt_camera_enabled:
            self._refresh_omninxt_camera_paths()
            self.front_camera_path = self.omninxt_camera_paths.get(
                self.omninxt_active_camera
            )
        else:
            self.front_camera_path = find_first_camera_under("/World/quadrotor1")
        if self.front_camera_path is not None:
            if self.omninxt_camera_enabled:
                self._ensure_omninxt_camera_configured()
            else:
                set_camera_pose_and_view(self.front_camera_path, set_viewport=False)
        else:
            _debug_log("[APP] No camera found under /World/quadrotor1")
        if not self.headless:
            set_active_viewport_perspective()

        self.skeleton_tracker = MultiPersonSkeletonTracker(
            self.people,
            update_hz=PEDESTRIAN_SKELETON_UPDATE_HZ,
        )
        self.skeleton_tracker.setup()
        carb.log_warn(
            "[APP][CROWD][RATE] "
            f"people={PEGASUS_PEOPLE_CONTROL_HZ:.1f}Hz, "
            f"controller={PEDESTRIAN_CONTROLLER_HZ:.1f}Hz, "
            f"separate_state_callback={bool(PEGASUS_PEOPLE_STATE_CALLBACK)}, "
            f"skeleton={PEDESTRIAN_SKELETON_UPDATE_HZ:.1f}Hz"
        )
        self.skeleton_packet_receiver = None
        if DATA_RECORD_ENABLED:
            self.skeleton_packet_receiver = SkeletonPacketReceiver(
                host=DATA_RECORD_SKELETON_HOST,
                port=DATA_RECORD_SKELETON_PORT,
            )
            self.skeleton_packet_receiver.start()
            carb.log_warn(
                "[REC][V2] Skeleton receiver listening on {}:{}".format(
                    DATA_RECORD_SKELETON_HOST, DATA_RECORD_SKELETON_PORT))
        self.data_recorder = SkeletonStateDatasetRecorder(
            drone=self.drone,
            skeleton_tracker=self.skeleton_tracker,
            skeleton_receiver=self.skeleton_packet_receiver,
            privileged_provider=self._dataset_privileged_snapshot,
            episode_metadata_provider=self._dataset_episode_metadata,
            target_point=TARGET_POINT,
            goal_region={
                "x_range": DATASET_GOAL_X_RANGE,
                "y_min": DATASET_GOAL_Y_MIN,
            },
            dataset_root=DATASET_ROOT,
            sample_rate_hz=DATA_RECORD_HZ,
            max_queue_size=DATA_RECORD_QUEUE_SIZE,
            image_format=DATA_IMAGE_FORMAT,
            jpeg_quality=DATA_IMAGE_JPEG_QUALITY,
            json_indent=DATA_JSON_INDENT,
            drop_when_writer_busy=DATA_DROP_WHEN_WRITER_BUSY,
            time_source=self._simulation_time,
            time_source_name="simulation_time",
            action_provider=self._dataset_action_snapshot,
            state_provider=self._vehicle_state_provider,
            altitude_agl_provider=self._altitude_agl_from_physx,
            reward_config=DATA_REWARD_CONFIG,
            control_rate_hz=DATA_CONTROL_HZ if self._uses_classic_controller() else None,
            storage_max_people=DATA_RECORD_STORAGE_MAX_PEOPLE,
            privileged_max_people=DATA_RECORD_PRIVILEGED_MAX_PEOPLE,
            chunk_frames=DATA_RECORD_CHUNK_FRAMES,
            sync_tolerance_sec=DATA_RECORD_SYNC_TOLERANCE_SEC,
        )

        self.gamepad = None
        self.classic_controller = None
        self.input_controller = None
        if self._uses_classic_controller():
            self.classic_controller = ClassicAlgorithmController(
                shared_cmd=self.shared_cmd,
                drone=self.drone,
                people=self.people,
                target_point=TARGET_POINT,
                walk_polygon=self.walk_polygon,
                obstacle_aabbs_getter=lambda: self.obstacle_aabbs,
                start_recording_callback=(
                    self._start_dataset_recording if DATA_RECORD_ENABLED else None
                ),
                abort_episode_callback=self._abort_classic_episode,
                time_source=self._simulation_time,
                control_rate_hz=DATA_CONTROL_HZ,
                command_sink=self.mavsdk_bridge,
                state_provider=(
                    self._vehicle_state_provider
                    if CLASSIC_NAV_USE_MAVSDK_TELEMETRY
                    else None
                ),
            )
            self.input_controller = self.classic_controller
        else:
            self.gamepad = GamepadController(
                self.shared_cmd,
                toggle_recording_callback=self._toggle_dataset_recording,
            )
            self.input_controller = self.gamepad

        for person in self.people:
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            _debug_log(
                f"[APP] {person_name} skelroot path = {person.character_skel_root_stage_path}"
            )
        _debug_log(
            f"[APP] Crowd template = {self.crowd_scene.key}, seed={self.crowd_scene.seed}"
        )
        _debug_log(
            f"[APP] Crowd details: people={self.crowd_scene.num_people}, "
            f"group_spacing={self.crowd_scene.group_spacing}, "
            f"direction={self.crowd_scene.direction}, "
            f"drone_distance={self.crowd_scene.drone_distance}, "
            f"speed={self.crowd_scene.speed}"
        )
        for spec in self.crowd_scene.person_specs:
            actual_init = self.person_initial_positions.get(spec.name, spec.init_pos)
            _debug_log(
                f"[APP][CROWD] {spec.name}: group={spec.group_id}, "
                f"member={spec.member_index + 1}/{spec.group_size}, "
                f"controller={spec.controller_kind}, speed={spec.speed:.2f}, "
                f"direction={spec.direction}, loop={spec.loop}, "
                f"spacing={spec.spacing:.2f}, template_init=({spec.init_pos[0]:.2f},"
                f"{spec.init_pos[1]:.2f}), actual_init=({actual_init[0]:.2f},"
                f"{actual_init[1]:.2f})"
            )
        _debug_log(f"[APP] Tracking pedestrian joints = {TARGET_JOINTS}")
        _debug_log(f"[APP] Pedestrian walk polygon = {PEDESTRIAN_WALK_POLYGON}")
        if ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            _debug_log(
                f"[APP][OBSTACLE] keywords={PEDESTRIAN_OBSTACLE_KEYWORDS}, "
                f"roots={PEDESTRIAN_OBSTACLE_ROOT_PATHS}, "
                f"keyword_hits={len(self.obstacle_scan.keyword_hits)}, "
                f"accepted={len(self.obstacle_scan.accepted_paths)}, "
                f"rejected={len(self.obstacle_scan.rejected_paths)}"
            )
            self._log_obstacle_scan_details()
            self._log_obstacle_root_samples_if_empty()
        else:
            _debug_log("[APP][OBSTACLE] Pedestrian obstacle avoidance disabled.")
        _debug_log(f"[APP] Dataset recording rate = {DATA_RECORD_HZ} Hz")
        carb.log_warn(
            f"[APP] Dataset recording enabled = {DATA_RECORD_ENABLED}. "
            f"Autonomous flight remains active at {DATA_CONTROL_HZ:.1f} Hz."
        )
        _debug_log(
            f"[APP] Dataset camera = {DATA_CAMERA_RESOLUTION}, image_format={DATA_IMAGE_FORMAT}"
        )
        _debug_log(f"[APP] Control mode = {self.control_mode}")
        _debug_log(f"[APP] Headless = {self.headless}")
        _debug_log(f"[APP] Drone spawned at: {SPAWN_POS}, yaw={DRONE_INIT_YAW_DEG:.1f} deg")
        _debug_log(f"[APP] {len(self.people)} pedestrians added.")

    @staticmethod
    def _altitude_agl_from_physx(position):
        """Measure center-to-ground height with a downward PhysX raycast."""
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            return None
        # Start below the vehicle collision body so the closest hit cannot be
        # the drone itself. Add the offset back to obtain center-to-surface AGL.
        body_clearance = 0.40
        origin = (float(position[0]), float(position[1]), float(position[2] - body_clearance))
        hit = get_physx_scene_query_interface().raycast_closest(
            origin, (0.0, 0.0, -1.0), 100.0
        )
        if not hit or not hit.get("hit", False):
            return None
        distance = float(hit.get("distance", float("nan")))
        return body_clearance + distance if np.isfinite(distance) and distance >= 0.0 else None

    def _create_px4_backend(self):
        px4_dir = PX4_DIR or self.pg.px4_path
        vehicle_model = PX4_VEHICLE_MODEL or self.pg.px4_default_airframe
        if PX4_AUTOLAUNCH and PX4_CLEAN_STALE_PROCESSES_ON_START:
            self._cleanup_stale_px4_stack(PX4_VEHICLE_ID)
        config = {
            "vehicle_id": PX4_VEHICLE_ID,
            "connection_type": PX4_MAVLINK_CONNECTION_TYPE,
            "connection_ip": PX4_MAVLINK_CONNECTION_IP,
            "connection_baseport": PX4_MAVLINK_CONNECTION_BASEPORT,
            "px4_autolaunch": PX4_AUTOLAUNCH,
            "px4_dir": px4_dir,
            "px4_vehicle_model": vehicle_model,
            "enable_lockstep": PX4_ENABLE_LOCKSTEP,
            "update_rate": PX4_BACKEND_UPDATE_RATE_HZ,
        }
        if MVD35_SIM2REAL_ENABLED:
            config.update(
                {
                    "input_offset": list(MVD35_PX4_INPUT_OFFSET),
                    "input_scaling": list(MVD35_PX4_INPUT_SCALING),
                    "zero_position_armed": list(MVD35_PX4_ZERO_POSITION_ARMED),
                }
            )
        mavlink_config = PX4MavlinkBackendConfig(config)
        carb.log_warn(
            f"[APP][PX4] Backend configured: autolaunch={PX4_AUTOLAUNCH}, "
            f"px4_dir={px4_dir}, vehicle_model={vehicle_model}, "
            f"mavlink={PX4_MAVLINK_CONNECTION_TYPE}:{PX4_MAVLINK_CONNECTION_IP}:"
            f"{PX4_MAVLINK_CONNECTION_BASEPORT + PX4_VEHICLE_ID}"
        )
        return PX4MavlinkBackend(mavlink_config)

    @staticmethod
    def _cleanup_stale_px4_stack(vehicle_id):
        """Stop stale single-vehicle SITL processes left by an unclean exit."""
        own_uid = os.getuid()
        targets = []
        instance_text = str(int(vehicle_id))

        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid():
                continue
            try:
                if entry.stat(follow_symlinks=False).st_uid != own_uid:
                    continue
                with open(f"/proc/{pid}/cmdline", "rb") as stream:
                    argv = [
                        item.decode(errors="replace")
                        for item in stream.read().split(b"\0")
                        if item
                    ]
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if not argv:
                continue

            executable = os.path.basename(argv[0])
            is_px4_instance = False
            if executable == "px4":
                try:
                    instance_index = argv.index("-i") + 1
                    is_px4_instance = argv[instance_index] == instance_text
                except (ValueError, IndexError):
                    is_px4_instance = int(vehicle_id) == 0

            is_mavsdk_endpoint = (
                executable == "mavsdk_server" and MAVSDK_SYSTEM_ADDRESS in argv
            )
            if is_px4_instance or is_mavsdk_endpoint:
                targets.append((pid, executable))

        if targets:
            carb.log_warn(
                "[APP][PX4] Cleaning stale processes before autolaunch: "
                + ", ".join(f"{name}(pid={pid})" for pid, name in targets)
            )
            for pid, _ in targets:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                alive = []
                for pid, name in targets:
                    try:
                        os.kill(pid, 0)
                        alive.append((pid, name))
                    except ProcessLookupError:
                        pass
                if not alive:
                    break
                time.sleep(0.05)
            else:
                for pid, _ in alive:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        for runtime_path in (
            f"/tmp/px4_lock-{int(vehicle_id)}",
            f"/tmp/px4-sock-{int(vehicle_id)}",
        ):
            try:
                os.unlink(runtime_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                carb.log_warn(
                    f"[APP][PX4] Could not remove stale runtime file "
                    f"{runtime_path}: {exc}"
                )

    def _uses_classic_controller(self):
        return self.control_mode in ("classic", "px4_classic")

    def _uses_px4_backend(self):
        return self.control_mode == "px4_classic"

    def _px4_backend_received_heartbeat(self):
        if self.px4_backend is None:
            return False
        return bool(
            getattr(self.px4_backend, "_received_first_hearbeat", False)
            or getattr(self.px4_backend, "_received_first_heartbeat", False)
        )

    def _ensure_mavsdk_bridge_started(self):
        if self.mavsdk_bridge is None or self._mavsdk_bridge_started:
            return

        if self._uses_px4_backend() and not self._px4_backend_received_heartbeat():
            if not self._mavsdk_bridge_wait_logged:
                carb.log_warn(
                    "[APP][PX4] Waiting for PX4 backend heartbeat before starting MAVSDK bridge."
                )
                self._mavsdk_bridge_wait_logged = True
            return

        try:
            self.mavsdk_bridge.start()
            self._mavsdk_bridge_started = True
            self._mavsdk_bridge_wait_logged = False
            carb.log_warn("[APP][PX4] MAVSDK bridge started after PX4 heartbeat.")
        except Exception as exc:
            carb.log_warn(f"[APP][PX4] MAVSDK bridge start failed: {exc}")

    def _vehicle_state_provider(self):
        if (
            self._uses_px4_backend()
            and MAVSDK_USE_TELEMETRY_STATE
            and self.mavsdk_bridge is not None
        ):
            return self.mavsdk_bridge.get_state()
        return None

    def run(self):
        self.timeline.play()

        while self.simulation_app.is_running() and not self._quit_requested():
            self._handle_timeline_resume()
            self.world.step(render=True)
            self._frame_count += 1
            self._update_omninxt_visual()
            self._ensure_mavsdk_bridge_started()
            self._ensure_omninxt_camera_configured()
            self._save_omninxt_preview_if_ready()
            self._stream_omninxt_frame_if_ready()
            self._export_omni_depth_frame_if_ready()
            self._refresh_obstacles_if_needed()
            self.skeleton_tracker.update_markers(self._simulation_time())
            self._monitor_crowd_clearance()
            self._export_crowd_map_state()
            recording_event = self._detect_recording_event()
            if recording_event is None:
                watchdog_event = self._detect_recording_watchdog_event()
                if watchdog_event is None:
                    self.data_recorder.update()
                else:
                    self._finish_recording_for_watchdog(watchdog_event)
            else:
                self._finish_recording_for_event(recording_event)
            self._update_post_episode_landing()
            # Controller cadence is independent of perception arrival and disk
            # writes. Replacing the flight policy therefore cannot change the
            # dataset timestamping contract, and recorder backpressure cannot
            # stall the policy update schedule.
            if not self.trajectory_limit_reached:
                if self._post_episode_landing_active:
                    pass
                elif self._control_update_due():
                    self._update_control_mode()

        _debug_log("Simulation closing.")
        self.data_recorder.stop(reason="shutdown")
        if self.skeleton_packet_receiver is not None:
            self.skeleton_packet_receiver.stop()
        if self.input_controller is not None:
            self.input_controller.shutdown()
        if self.mavsdk_bridge is not None:
            self.mavsdk_bridge.shutdown()
        if self.px4_backend is not None:
            try:
                self.px4_backend.stop()
            except Exception as exc:
                carb.log_warn(f"[APP][PX4] PX4 backend stop during shutdown failed: {exc}")
            self._force_clear_px4_backend()
        self.timeline.stop()
        self.simulation_app.close()

    def _update_omninxt_visual(self):
        if self.omninxt_visual is None:
            return
        thrusters = getattr(self.drone, "_thrusters", None)
        if thrusters is None:
            return
        velocities = getattr(thrusters, "velocity", ())
        directions = getattr(thrusters, "rot_dir", ())
        self.omninxt_visual.update(
            self._simulation_time(), velocities, directions
        )

    def _stream_omninxt_frame_if_ready(self):
        """Write a latest-only synchronized sensor bundle at simulation time."""
        if (not OMNINXT_LIVE_STREAM_ENABLED
                or not self.omninxt_camera_applied
                or OMNINXT_LIVE_STREAM_HZ <= 0.0):
            return
        sim_time = float(self._simulation_time())
        period = 1.0 / float(OMNINXT_LIVE_STREAM_HZ)
        if (self._omninxt_live_last_sim_time is not None
                and sim_time - self._omninxt_live_last_sim_time < period - 1e-9):
            return
        try:
            if self._omninxt_live_exporter is None:
                self._omninxt_live_exporter = IsaacQuadcamExporter(
                    self.omninxt_camera_sensors,
                    OMNINXT_LIVE_STREAM_ROOT,
                    camera_paths=self.omninxt_camera_paths,
                    calibration_set="sync_20260802_formal",
                    camera_config=self.omninxt_camera_config,
                    export_gt_range=OMNINXT_GT_RANGE_EXPORT_ENABLED,
                )
                carb.log_warn(
                    "[APP][OMNINXT][LIVE] Starting latest-only stream: "
                    f"mode={OMNINXT_SENSOR_MODE}, hz={OMNINXT_LIVE_STREAM_HZ:.1f}, "
                    f"root={OMNINXT_LIVE_STREAM_ROOT}"
                )
            self._omninxt_live_exporter.capture_live(
                self._frame_count, sim_time
            )
            self._omninxt_live_last_sim_time = sim_time
        except Exception as exc:
            carb.log_warn(
                f"[APP][OMNINXT][LIVE] Frame publish failed: {exc}"
            )

    def _export_omni_depth_frame_if_ready(self):
        """Atomically export one synchronized quad-camera frame for Omni-Depth."""
        if (not OMNINXT_DEPTH_EXPORT_ENABLED or self._omni_depth_export_done
                or not self.omninxt_camera_applied):
            return
        if (OMNINXT_DEPTH_EXPORT_NAV_ONLY and
                (self.classic_controller is None or self.classic_controller.state != "navigate")):
            return
        if self._omni_depth_export_ready_frame is None:
            self._omni_depth_export_ready_frame = (
                self._frame_count + max(0, OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES)
            )
            carb.log_warn("[APP][OMNI-DEPTH] Four cameras ready; waiting "
                          f"{OMNINXT_DEPTH_EXPORT_SETTLE_FRAMES} rendered frames.")
            return
        if self._frame_count < self._omni_depth_export_ready_frame:
            return
        try:
            if self._omni_depth_exporter is None:
                self._omni_depth_exporter = IsaacQuadcamExporter(
                    self.omninxt_camera_sensors,
                    OMNINXT_DEPTH_EXPORT_ROOT,
                    camera_paths=self.omninxt_camera_paths,
                    calibration_set="sync_20260802_formal",
                    camera_config=self.omninxt_camera_config,
                    export_gt_range=OMNINXT_GT_RANGE_EXPORT_ENABLED,
                )
            frame_dir = self._omni_depth_exporter.capture(
                self._frame_count, self._simulation_time()
            )
            self._omni_depth_export_count += 1
            self._omni_depth_export_done = (
                self._omni_depth_export_count >= max(1, OMNINXT_DEPTH_EXPORT_COUNT)
            )
            if not self._omni_depth_export_done:
                self._omni_depth_export_ready_frame = (
                    self._frame_count + max(1, OMNINXT_DEPTH_EXPORT_INTERVAL_FRAMES)
                )
            carb.log_warn(
                f"[APP][OMNI-DEPTH] Exported synchronized frame "
                f"{self._omni_depth_export_count}/"
                f"{max(1, OMNINXT_DEPTH_EXPORT_COUNT)}: {frame_dir}"
            )
            if self._omni_depth_export_done and OMNINXT_EXIT_AFTER_EXPORT:
                self._omninxt_exit_requested = True
                carb.log_warn(
                    "[APP][OMNI-DEPTH] Requested clean exit after final export."
                )
        except Exception as exc:
            carb.log_warn(f"[APP][OMNI-DEPTH] Frame export failed: {exc}")

    def _ensure_omninxt_camera_configured(self):
        if (
            not self.omninxt_camera_enabled
            or self.omninxt_camera_config is None
        ):
            return

        self._refresh_omninxt_camera_paths()
        if self.front_camera_path is None:
            self.front_camera_path = self.omninxt_camera_paths.get(
                self.omninxt_active_camera
            )

        try:
            for camera_id in self.omninxt_camera_ids:
                if camera_id in self.omninxt_configured_cameras:
                    continue

                camera_path = self.omninxt_camera_paths.get(camera_id)
                camera_sensor = self.omninxt_camera_sensors.get(camera_id)
                if camera_path is None or camera_sensor is None:
                    continue

                configured = configure_omninxt_camera(
                    camera_sensor,
                    camera_path,
                    self.omninxt_camera_config,
                    camera_name=camera_id,
                    # The main viewport remains Perspective. cam0 is shown in
                    # its own viewport window instead of taking over the UI.
                    set_viewport=False,
                    enable_gt_range=OMNINXT_GT_RANGE_EXPORT_ENABLED,
                )
                if configured:
                    self.omninxt_configured_cameras.add(camera_id)
                    if (
                        bool(OMNINXT_PREVIEW_SAVE_ON_START)
                        and camera_id not in self._omninxt_preview_saved
                    ):
                        self._omninxt_preview_pending_frames[camera_id] = int(
                            max(0, OMNINXT_PREVIEW_SETTLE_FRAMES)
                        )
                    carb.log_warn(
                        f"[APP][CAM] Applied OmniNxt sync_20260802 {camera_id} "
                        f"profile={self.omninxt_camera_config['camera_rig'].get('mode')} "
                        f"to {camera_path}"
                    )
            self.omninxt_camera_applied = (
                len(self.omninxt_configured_cameras) == len(self.omninxt_camera_ids)
            )
            self._ensure_omninxt_view_windows()
        except Exception as exc:
            if not self._omninxt_camera_error_logged:
                self._omninxt_camera_error_logged = True
                carb.log_warn(f"[APP][CAM] Failed to apply OmniNxt camera config: {exc}")

    def _ensure_omninxt_view_windows(self):
        if (
            not self.omninxt_camera_enabled
            or self.headless
            or not bool(OMNINXT_VIEW_WINDOWS_ENABLED)
        ):
            return

        for camera_id in self.omninxt_view_window_camera_names:
            if camera_id in self.omninxt_camera_view_windows:
                continue
            if camera_id not in self.omninxt_configured_cameras:
                continue

            camera_path = self.omninxt_camera_paths.get(camera_id)
            if camera_path is None:
                continue

            layout = dict(OMNINXT_VIEW_WINDOW_LAYOUT.get(camera_id, {}))
            sensor_name = self.omninxt_camera_sensor_names.get(camera_id, camera_id)
            spec = self.omninxt_camera_config["camera_rig"]["cameras"].get(
                camera_id, {}
            )
            try:
                window = create_camera_viewport_window(
                    camera_path,
                    name=layout.get(
                        "name", f"OmniNxt {spec.get('alias', sensor_name)}"
                    ),
                    width=layout.get("width", 640),
                    height=layout.get("height", 360),
                    position_x=layout.get("position_x", 60),
                    position_y=layout.get("position_y", 60),
                )
                self.omninxt_camera_view_windows[camera_id] = window
                carb.log_warn(
                    f"[APP][CAM] Opened OmniNxt {camera_id} viewport window: {camera_path}"
                )
            except Exception as exc:
                if not self._omninxt_view_window_error_logged:
                    self._omninxt_view_window_error_logged = True
                    carb.log_warn(
                        f"[APP][CAM] Failed to open OmniNxt viewport windows: {exc}"
                    )
                return

    def _save_omninxt_preview_if_ready(self):
        if (
            not self.omninxt_camera_enabled
            or not self.omninxt_configured_cameras
            or not self._omninxt_preview_pending_frames
        ):
            return

        for camera_id in tuple(self._omninxt_preview_pending_frames):
            if camera_id not in self.omninxt_configured_cameras:
                continue
            if camera_id in self._omninxt_preview_saved:
                self._omninxt_preview_pending_frames.pop(camera_id, None)
                continue

            pending_frames = self._omninxt_preview_pending_frames[camera_id]
            if pending_frames > 0:
                self._omninxt_preview_pending_frames[camera_id] = pending_frames - 1
                continue

            try:
                preview_path = self._omninxt_preview_path(camera_id)
                camera_sensor = self.omninxt_camera_sensors[camera_id]
                saved = save_camera_rgb_preview(camera_sensor, preview_path)
                if saved:
                    self._omninxt_preview_saved.add(camera_id)
                    self._omninxt_preview_pending_frames.pop(camera_id, None)
                    carb.log_warn(
                        f"[APP][CAM] Saved OmniNxt {camera_id} fisheye preview: {preview_path}"
                    )
            except Exception as exc:
                if not self._omninxt_preview_error_logged:
                    self._omninxt_preview_error_logged = True
                    carb.log_warn(f"[APP][CAM] Failed to save OmniNxt preview: {exc}")

    def _refresh_omninxt_camera_paths(self):
        if not self.omninxt_camera_enabled:
            return

        for camera_id in self.omninxt_camera_ids:
            if self.omninxt_camera_paths.get(camera_id):
                continue
            sensor_name = self.omninxt_camera_sensor_names.get(camera_id, camera_id)
            path = find_camera_named_under("/World/quadrotor1", sensor_name)
            if path is not None:
                self.omninxt_camera_paths[camera_id] = path

        self.front_camera_path = self.omninxt_camera_paths.get(
            self.omninxt_active_camera
        )

    def _omninxt_preview_path(self, camera_id):
        if camera_id == self.omninxt_active_camera:
            return OMNINXT_PREVIEW_PATH
        sensor_name = self.omninxt_camera_sensor_names.get(camera_id, camera_id)
        return OMNINXT_PREVIEW_PATH_TEMPLATE.format(
            camera_name=camera_id,
            sensor_name=sensor_name,
        )

    def _quit_requested(self):
        return (
            self.trajectory_limit_reached
            or self._omninxt_exit_requested
            or bool(getattr(self.input_controller, "quit", False))
        )

    def _simulation_time(self):
        for source, attribute_name in (
            (self.world, "current_time"),
            (self.timeline, "get_current_time"),
        ):
            try:
                value = getattr(source, attribute_name)
                if callable(value):
                    value = value()
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                return value
        return None

    def _dataset_action_snapshot(self):
        vx_body, vy_body, vz_world, yaw_rate = self.shared_cmd.get_motion()
        requested_values = np.array(
            [vx_body, vy_body, vz_world, yaw_rate],
            dtype=float,
        )
        limits = np.array(
            [
                DATA_ACTION_NORMALIZATION["vx_body_mps"],
                DATA_ACTION_NORMALIZATION["vy_body_mps"],
                DATA_ACTION_NORMALIZATION["vz_world_mps"],
                DATA_ACTION_NORMALIZATION["yaw_rate_rps"],
            ],
            dtype=float,
        )
        normalized = np.clip(
            requested_values / np.maximum(limits, 1e-6),
            -1.0,
            1.0,
        )

        control_update_time = None
        if self.classic_controller is not None:
            control_update_time = self.classic_controller.last_action_update_time
        now = self._simulation_time()
        action_age = None
        if now is not None and control_update_time is not None:
            action_age = max(0.0, float(now) - float(control_update_time))

        source = "classic_controller" if self.classic_controller is not None else "gamepad"
        applied = None
        applied_body_flu = None
        mavsdk_snapshot = None
        if self.mavsdk_bridge is not None:
            source = "px4_mavsdk_classic_controller"
            mavsdk_snapshot = self.mavsdk_bridge.snapshot()
            applied = mavsdk_snapshot.get("applied")
            if isinstance(applied, dict) and bool(
                    mavsdk_snapshot.get("offboard_started", False)):
                applied_body_flu = {
                    "vx_body_mps": float(applied.get("forward_m_s", 0.0)),
                    "vy_body_mps": float(applied.get("right_m_s", 0.0)) /
                    float(MAVSDK_BODY_RIGHT_SIGN),
                    "vz_world_mps": float(applied.get("down_m_s", 0.0)) /
                    float(MAVSDK_BODY_DOWN_SIGN),
                    "yaw_rate_rps": math.radians(
                        float(applied.get("yawspeed_deg_s", 0.0)) /
                        float(MAVSDK_YAWSPEED_SIGN)),
                }
        elif self.keyboard_backend is not None:
            applied = self.keyboard_backend.get_applied_motion()
            if isinstance(applied, dict):
                applied_body_flu = {
                    key: float(applied.get(key, 0.0))
                    for key in (
                        "vx_body_mps", "vy_body_mps",
                        "vz_world_mps", "yaw_rate_rps",
                    )
                }

        return {
            "source": source,
            "space": "velocity_setpoint",
            "normalized": normalized.astype(float).tolist(),
            "normalization_limits": dict(DATA_ACTION_NORMALIZATION),
            "requested": {
                "vx_body_mps": float(vx_body),
                "vy_body_mps": float(vy_body),
                "vz_world_mps": float(vz_world),
                "yaw_rate_rps": float(yaw_rate),
            },
            "applied": applied,
            # Stable controller-independent convention consumed by dataset v2.
            "applied_body_flu": applied_body_flu,
            "mavsdk": mavsdk_snapshot,
            "control_update_sim_time": (
                None if control_update_time is None else float(control_update_time)
            ),
            "control_action_age_sec": action_age,
        }

    def _dataset_privileged_snapshot(self):
        """Return simulator truth that must never enter the policy encoder."""
        drone_state = getattr(self.drone, "state", None)
        drone = {}
        if drone_state is not None:
            drone = {
                "position": np.asarray(drone_state.position, dtype=float).tolist(),
                "velocity": np.asarray(
                    drone_state.linear_velocity, dtype=float).tolist(),
                "acceleration": np.asarray(
                    drone_state.linear_acceleration, dtype=float).tolist(),
                "quaternion_xyzw": np.asarray(
                    drone_state.attitude, dtype=float).tolist(),
            }
        marker_positions = self.skeleton_tracker.marker_positions or {}
        people = []
        active_people = self.people[:int(self.crowd_scene.num_people)]
        for person_index, person in enumerate(active_people):
            name = person._stage_prefix.rstrip("/").split("/")[-1]
            state = getattr(person, "state", None)
            if state is None:
                continue
            position = np.asarray(state.position, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                continue
            velocity = np.asarray(
                getattr(state, "linear_velocity", np.zeros(3)), dtype=float)
            controller = getattr(person, "_controller", None)
            joints_by_name = marker_positions.get(name, {}) or {}
            collision_joints = []
            collision_valid = []
            for joint_name in self.skeleton_tracker.joint_names:
                value = joints_by_name.get(joint_name)
                valid = value is not None
                array = np.zeros(3) if not valid else np.asarray(value, dtype=float)
                valid = bool(valid and array.shape == (3,) and np.isfinite(array).all())
                collision_joints.append(
                    array.astype(float).tolist() if valid else [0.0, 0.0, 0.0])
                collision_valid.append(valid)
            pelvis = joints_by_name.get("Pelvis")
            people.append({
                "id": int(person_index),
                "name": name,
                "group_id": getattr(controller, "crowd_group_id", None),
                "position": position.astype(float).tolist(),
                "velocity": velocity.astype(float).tolist(),
                "pelvis": None if pelvis is None else
                np.asarray(pelvis, dtype=float).tolist(),
                "collision_joints": collision_joints,
                "collision_joint_valid": collision_valid,
            })
        return {"drone": drone, "people": people}

    def _dataset_episode_metadata(self):
        people = []
        for person_index, spec in enumerate(self.crowd_scene.person_specs):
            group_id = getattr(spec, "group_id", None)
            people.append({
                "id": int(person_index),
                "name": str(spec.name),
                "group_id": None if group_id is None else int(group_id),
                "group_size": int(getattr(spec, "group_size", 1)),
                "member_index": int(getattr(spec, "member_index", 0)),
                "initial_position": [float(value) for value in spec.init_pos],
                "waypoints": [
                    [float(value) for value in point]
                    for point in getattr(spec, "waypoints", ())
                ],
            })
        return {
            "scene_key": str(self.crowd_scene.key),
            "crowd_seed": int(self.crowd_scene.seed),
            "crowd_num_people": int(self.crowd_scene.num_people),
            "drone_spawn_world": [float(value) for value in SPAWN_POS],
            "goal_world": [float(value) for value in TARGET_POINT],
            "people": people,
        }

    def _update_control_mode(self):
        update = getattr(self.input_controller, "update", None)
        if callable(update):
            update()

    def _control_update_due(self):
        now = self._simulation_time()
        if now is None:
            return True

        period = 1.0 / max(float(DATA_CONTROL_HZ), 1e-6)
        if (
            self._last_control_update_time is None
            or now < self._last_control_update_time
            or now - self._last_control_update_time >= period - 1e-9
        ):
            self._last_control_update_time = now
            return True
        return False

    def _toggle_dataset_recording(self):
        if not DATA_RECORD_ENABLED:
            carb.log_warn(
                "[REC] Dataset recording is disabled by DATA_RECORD_ENABLED=0."
            )
            return
        if self.data_recorder.is_recording:
            self.data_recorder.stop(reason="manual_stop")
            self._register_completed_trajectory("manual_stop")
            return

        self._start_dataset_recording()

    def _start_dataset_recording(self):
        if not DATA_RECORD_ENABLED:
            return False
        if self.trajectory_limit_reached:
            return False
        if self.data_recorder.is_recording:
            return True

        self._last_record_size_check_wall = 0.0
        self.data_recorder.start()
        return bool(self.data_recorder.is_recording)

    def _abort_classic_episode(self):
        if self.data_recorder.is_recording:
            self.data_recorder.stop(reason="manual_stop")
            self._register_completed_trajectory("manual_stop")
            if self.trajectory_limit_reached:
                return
        self._reset_classic_episode("manual_stop")

    def _enable_drone_contact_reports(self):
        if PhysxSchema is None:
            return

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(self.drone_root_path)
        if not root_prim.IsValid():
            return

        for prim in self._iter_prim_tree(root_prim):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            contact_report_api = self._apply_api(PhysxSchema.PhysxContactReportAPI, prim)
            contact_report_api.CreateThresholdAttr().Set(0.0)

    def _apply_mvd35_vehicle_model(self):
        stage = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(f"{self.drone_root_path}/body")
        if not body_prim.IsValid():
            carb.log_warn("[APP][MVD35] Cannot find drone body prim; physics override skipped.")
            return

        self._set_mass_properties(
            body_prim,
            mass_kg=MVD35_BODY_MASS_KG,
            center_of_mass_m=MVD35_CENTER_OF_MASS_M,
            diagonal_inertia_kgm2=MVD35_DIAGONAL_INERTIA_KGM2,
        )

        for rotor_index, rotor_position in enumerate(MVD35_ROTOR_POSITIONS_FLU_M):
            rotor_prim = stage.GetPrimAtPath(f"{self.drone_root_path}/rotor{rotor_index}")
            if not rotor_prim.IsValid():
                carb.log_warn(
                    f"[APP][MVD35] Cannot find rotor{rotor_index}; rotor override skipped."
                )
                continue
            self._set_local_translation(rotor_prim, rotor_position)
            self._set_mass_properties(
                rotor_prim,
                mass_kg=MVD35_ROTOR_MASS_KG,
                center_of_mass_m=(0.0, 0.0, 0.0),
                diagonal_inertia_kgm2=MVD35_ROTOR_DIAGONAL_INERTIA_KGM2,
            )

        hover_omega = np.sqrt(
            (MVD35_TOTAL_MASS_KG * 9.81 / 4.0)
            / MVD35_THRUST_CURVE_CONFIG["rotor_constant"][0]
        )
        carb.log_warn(
            "[APP][MVD35] Applied MVD35 sim2real vehicle model: "
            f"total_mass={MVD35_TOTAL_MASS_KG:.3f}kg, "
            f"body_mass={MVD35_BODY_MASS_KG:.3f}kg, "
            f"rotor_mass={MVD35_ROTOR_MASS_KG:.3f}kg x4, "
            f"hover_omega={hover_omega:.1f}rad/s."
        )

    def _set_mass_properties(
        self,
        prim,
        mass_kg,
        center_of_mass_m,
        diagonal_inertia_kgm2,
    ):
        mass_api = self._apply_api(UsdPhysics.MassAPI, prim)
        self._get_or_create_attr(mass_api, "GetMassAttr", "CreateMassAttr").Set(
            float(mass_kg)
        )
        self._get_or_create_attr(
            mass_api,
            "GetCenterOfMassAttr",
            "CreateCenterOfMassAttr",
        ).Set(Gf.Vec3f(*[float(v) for v in center_of_mass_m]))
        self._get_or_create_attr(
            mass_api,
            "GetDiagonalInertiaAttr",
            "CreateDiagonalInertiaAttr",
        ).Set(Gf.Vec3f(*[float(v) for v in diagonal_inertia_kgm2]))
        self._get_or_create_attr(
            mass_api,
            "GetPrincipalAxesAttr",
            "CreatePrincipalAxesAttr",
        ).Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    @staticmethod
    def _get_or_create_attr(api, getter_name, creator_name):
        attr = getattr(api, getter_name)()
        if not attr:
            attr = getattr(api, creator_name)()
        return attr

    @staticmethod
    def _set_local_translation(prim, position):
        xform = UsdGeom.Xformable(prim)
        value = Gf.Vec3d(*[float(v) for v in position])
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(value)
                return
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(value)

    @staticmethod
    def _iter_prim_tree(root_prim):
        stack = [root_prim]
        while stack:
            prim = stack.pop()
            yield prim
            children = list(prim.GetChildren())
            stack.extend(reversed(children))

    @staticmethod
    def _apply_api(api_cls, prim):
        api = api_cls(prim)
        if not api:
            api = api_cls.Apply(prim)
        return api

    def _detect_recording_event(self):
        if not self.data_recorder.is_recording:
            return None

        drone_position = np.array(self.drone.state.position, dtype=float)
        reached_goal = self._drone_reached_goal(drone_position)
        collision = self._detect_drone_pedestrian_collision(drone_position)
        if collision is not None:
            collision["reached_goal_at_collision"] = bool(reached_goal)
            return {
                "reason": "collision",
                "collision": True,
                "reached_goal": reached_goal,
                "details": collision,
            }

        if reached_goal:
            return {
                "reason": "reached_goal",
                "collision": False,
                "reached_goal": True,
                "details": self._goal_details(drone_position),
            }

        return None

    def _detect_recording_watchdog_event(self):
        if not self.data_recorder.is_recording:
            return None

        elapsed_sim = self.data_recorder.elapsed_time()
        elapsed_wall = self.data_recorder.elapsed_wall_time()
        if (
            DATA_RECORD_STUCK_TIMEOUT_SEC is not None
            and DATA_RECORD_STUCK_TIMEOUT_SEC > 0.0
            and elapsed_sim >= DATA_RECORD_STUCK_TIMEOUT_SEC
        ):
            return {
                "reason": "stuck_timeout",
                "details": {
                    "elapsed_sim_sec": float(elapsed_sim),
                    "elapsed_wall_sec": float(elapsed_wall),
                    "limit_sim_sec": float(DATA_RECORD_STUCK_TIMEOUT_SEC),
                },
            }

        if DATA_RECORD_MAX_RECORD_BYTES is None or DATA_RECORD_MAX_RECORD_BYTES <= 0:
            return None

        check_interval = max(float(DATA_RECORD_SIZE_CHECK_INTERVAL_SEC), 0.0)
        if elapsed_wall - self._last_record_size_check_wall < check_interval:
            return None
        self._last_record_size_check_wall = elapsed_wall

        record_size = self.data_recorder.current_record_size_bytes()
        if record_size < DATA_RECORD_MAX_RECORD_BYTES:
            return None

        return {
            "reason": "record_size_limit",
            "details": {
                "elapsed_sim_sec": float(elapsed_sim),
                "elapsed_wall_sec": float(elapsed_wall),
                "record_size_bytes": int(record_size),
                "record_size_gib": float(record_size) / float(1024**3),
                "limit_size_bytes": int(DATA_RECORD_MAX_RECORD_BYTES),
                "limit_size_gib": float(DATA_RECORD_MAX_RECORD_BYTES) / float(1024**3),
            },
        }

    @staticmethod
    def _drone_reached_goal(drone_position):
        x = float(drone_position[0])
        y = float(drone_position[1])
        goal_x_min, goal_x_max = DATASET_GOAL_X_RANGE
        return goal_x_min <= x <= goal_x_max and y >= DATASET_GOAL_Y_MIN

    @staticmethod
    def _goal_details(drone_position):
        return {
            "goal_x_range": [float(DATASET_GOAL_X_RANGE[0]), float(DATASET_GOAL_X_RANGE[1])],
            "goal_y_min": float(DATASET_GOAL_Y_MIN),
            "drone_position": np.array(drone_position, dtype=float).tolist(),
        }

    def _detect_drone_pedestrian_collision(self, drone_position):
        contact = self._detect_drone_pedestrian_contact()
        if contact is None:
            return None

        contact["drone_position"] = np.array(drone_position, dtype=float).tolist()
        return contact

    def _detect_drone_pedestrian_contact(self):
        try:
            contact_headers, _ = get_physx_simulation_interface().get_contact_report()
        except Exception:
            return None

        for contact_header in contact_headers:
            if not self._is_collision_contact_event(contact_header.type):
                continue

            collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
            contact = self._classify_drone_pedestrian_contact(collider0, collider1)
            if contact is not None:
                contact["source"] = "isaacsim_physx_contact_report"
                contact["contact_event_type"] = self._contact_event_name(contact_header.type)
                return contact

        return None

    def _classify_drone_pedestrian_contact(self, collider0, collider1):
        marker0 = self.skeleton_tracker.parse_marker_collider_path(collider0)
        marker1 = self.skeleton_tracker.parse_marker_collider_path(collider1)
        collider0_is_drone = self._is_drone_collider_path(collider0)
        collider1_is_drone = self._is_drone_collider_path(collider1)

        if collider0_is_drone and marker1 is not None:
            marker1["drone_collider"] = collider0
            marker1["pedestrian_collider"] = collider1
            return marker1
        if collider1_is_drone and marker0 is not None:
            marker0["drone_collider"] = collider1
            marker0["pedestrian_collider"] = collider0
            return marker0

        return None

    def _is_drone_collider_path(self, collider_path):
        collider_path = str(collider_path)
        return collider_path == self.drone_root_path or collider_path.startswith(
            f"{self.drone_root_path}/"
        )

    @staticmethod
    def _is_collision_contact_event(event_type):
        if event_type == ContactEventType.CONTACT_FOUND:
            return True
        if hasattr(ContactEventType, "CONTACT_PERSIST"):
            return event_type == ContactEventType.CONTACT_PERSIST
        return False

    @staticmethod
    def _contact_event_name(event_type):
        for name in ("CONTACT_FOUND", "CONTACT_PERSIST", "CONTACT_LOST"):
            if hasattr(ContactEventType, name) and event_type == getattr(ContactEventType, name):
                return name
        return str(event_type)

    def _finish_recording_for_event(self, event):
        self.data_recorder.set_event_status(
            collision=event["collision"],
            reached_goal=event["reached_goal"],
            termination_reason=event["reason"],
            event_details=event["details"],
        )
        self.data_recorder.update(force=True)
        if event["reason"] == "collision":
            details = event["details"]
            carb.log_warn(
                f"[REC] Collision detected: pedestrian={details['pedestrian_id']}, "
                f"joint={details['joint_name']}, drone_collider={details['drone_collider']}, "
                f"pedestrian_collider={details['pedestrian_collider']}"
            )
        elif event["reason"] == "reached_goal":
            carb.log_warn(
                f"[REC] Goal reached: {DATASET_GOAL_X_RANGE[0]:.2f} <= x <= "
                f"{DATASET_GOAL_X_RANGE[1]:.2f}, y >= {DATASET_GOAL_Y_MIN:.2f}"
            )
        self.data_recorder.stop(
            reason=event["reason"],
            collision=event["collision"],
            reached_goal=event["reached_goal"],
            event_details=event["details"],
        )
        self._register_completed_trajectory(event["reason"])
        if self.trajectory_limit_reached:
            return
        if self._uses_classic_controller():
            self._handle_completed_classic_episode(event["reason"])

    def _finish_recording_for_watchdog(self, event):
        details = dict(event.get("details") or {})
        self.data_recorder.set_event_status(
            collision=False,
            reached_goal=False,
            termination_reason=event["reason"],
            event_details=details,
        )
        self.data_recorder.update(force=True)
        self.data_recorder.stop(
            reason=event["reason"],
            collision=False,
            reached_goal=False,
            event_details=details,
        )
        carb.log_warn(
            f"[REC] Preserved failed/truncated trajectory: "
            f"reason={event['reason']}, details={details}"
        )
        self._register_completed_trajectory(event["reason"])
        if self.trajectory_limit_reached:
            return
        if self._uses_classic_controller():
            self._handle_completed_classic_episode(event["reason"])

    def _handle_completed_classic_episode(self, reason):
        if self._should_restart_px4_between_episodes():
            self._reset_classic_episode(reason)
            return
        if (
            self.control_mode == "px4_classic"
            and PX4_LAND_BEFORE_EPISODE_RESET
            and self.mavsdk_bridge is not None
        ):
            self._begin_post_episode_landing(reason)
            return
        self._reset_classic_episode(reason)

    def _begin_post_episode_landing(self, reason):
        if self._post_episode_landing_active:
            return

        self._post_episode_landing_active = True
        self._post_episode_landing_reason = reason
        self._post_episode_landing_start_wall = time.perf_counter()
        self._post_episode_landing_ready_since = None
        self.shared_cmd.reset()
        if self.classic_controller is not None:
            self.classic_controller.hold_after_episode(reason)
        if self.mavsdk_bridge is not None:
            self.mavsdk_bridge.set_motion(0.0, 0.0, 0.0, 0.0)
            self.mavsdk_bridge.trigger_land()
        carb.log_warn(
            f"[REC] Episode ended with {reason}; landing PX4 before next reset."
        )

    def _update_post_episode_landing(self):
        if not self._post_episode_landing_active:
            return

        landed = self._post_episode_landing_is_settled()
        timed_out = False
        if self._post_episode_landing_start_wall is not None:
            timed_out = (
                time.perf_counter() - self._post_episode_landing_start_wall
                >= float(PX4_EPISODE_LAND_TIMEOUT_SEC)
            )

        if not landed and not timed_out:
            return

        reason = self._post_episode_landing_reason or "post_episode"
        if timed_out and not landed:
            carb.log_warn(
                f"[REC] PX4 landing wait timed out after "
                f"{PX4_EPISODE_LAND_TIMEOUT_SEC:.1f}s; forcing episode reset."
            )
        else:
            carb.log_warn("[REC] PX4 landing settled; resetting episode.")

        self._post_episode_landing_active = False
        self._post_episode_landing_reason = None
        self._post_episode_landing_start_wall = None
        self._post_episode_landing_ready_since = None
        self._reset_classic_episode(reason)

    def _post_episode_landing_is_settled(self):
        try:
            position = np.array(self.drone.state.position, dtype=float)
            velocity = np.array(self.drone.state.linear_velocity, dtype=float)
        except Exception:
            return False

        z = float(position[2])
        vz = float(velocity[2]) if velocity.size >= 3 else 0.0
        low_enough = z <= float(PX4_EPISODE_LAND_Z_THRESHOLD)
        px4_landed = self._post_episode_px4_reports_landed()
        settled = (
            (low_enough or px4_landed)
            and abs(vz) <= float(PX4_EPISODE_LAND_VZ_THRESHOLD)
        )
        now = time.perf_counter()

        if settled:
            if self._post_episode_landing_ready_since is None:
                self._post_episode_landing_ready_since = now
            return now - self._post_episode_landing_ready_since >= 1.0

        self._post_episode_landing_ready_since = None
        return False

    def _post_episode_px4_reports_landed(self):
        if self.mavsdk_bridge is None:
            return False

        try:
            snapshot = self.mavsdk_bridge.snapshot()
        except Exception:
            return False

        if bool(snapshot.get("in_air_known")) and not bool(snapshot.get("in_air")):
            return True
        if bool(snapshot.get("armed_known")) and not bool(snapshot.get("armed")):
            return True
        return False

    def _should_restart_px4_between_episodes(self):
        return (
            self.control_mode == "px4_classic"
            and bool(PX4_RESTART_BETWEEN_EPISODES)
            and self.px4_backend is not None
        )

    def _register_completed_trajectory(self, reason):
        if DATA_RECORD_MAX_TRAJECTORIES is None or DATA_RECORD_MAX_TRAJECTORIES <= 0:
            return

        self.completed_trajectory_count += 1
        carb.log_warn(
            f"[REC] Completed trajectory {self.completed_trajectory_count}/"
            f"{DATA_RECORD_MAX_TRAJECTORIES}, reason={reason}"
        )
        if self.completed_trajectory_count >= DATA_RECORD_MAX_TRAJECTORIES:
            self.trajectory_limit_reached = True
            if self.input_controller is not None:
                self.input_controller.quit = True
            carb.log_warn(
                f"[REC] Trajectory limit reached ({DATA_RECORD_MAX_TRAJECTORIES}). "
                "Stopping simulation."
            )

    def _reset_classic_episode(self, reason):
        self.shared_cmd.reset()
        restart_px4 = self._should_restart_px4_between_episodes()
        classic_reset_done = False
        if restart_px4:
            self._stop_px4_stack_for_episode_reset(reason)
            self._px4_stack_restart_active = True
        else:
            self._reset_control_backend()

        try:
            self._resample_scene_after_timeline_stop()
            self._resume_people_after_timeline_resume()
        finally:
            if restart_px4:
                if self.classic_controller is not None:
                    self.classic_controller.reset_after_episode(reason)
                    classic_reset_done = True
                self._px4_stack_restart_active = False
                self._start_px4_stack_for_episode_reset(reason)

        if self.classic_controller is not None and not classic_reset_done:
            self.classic_controller.reset_after_episode(reason)

    def _reset_control_backend(self):
        if self.keyboard_backend is not None:
            self.keyboard_backend.reset()
        if self.mavsdk_bridge is not None:
            if self._px4_stack_restart_active:
                return
            self.mavsdk_bridge.reset_after_episode()

    def _stop_px4_stack_for_episode_reset(self, reason):
        carb.log_warn(f"[APP][PX4] Stopping PX4 stack after episode: reason={reason}")
        if self.mavsdk_bridge is not None:
            try:
                self.mavsdk_bridge.shutdown()
                self._mavsdk_bridge_started = False
                self._mavsdk_bridge_wait_logged = False
            except Exception as exc:
                carb.log_warn(f"[APP][PX4] MAVSDK bridge shutdown failed: {exc}")

        if self.px4_backend is not None:
            px4_process = self._current_px4_process()
            try:
                self.px4_backend.stop()
            except Exception as exc:
                carb.log_warn(f"[APP][PX4] PX4 backend stop failed: {exc}")
            self._wait_for_px4_process_exit(px4_process)
            self._force_clear_px4_backend()

        delay = max(0.0, float(PX4_RESTART_DELAY_SEC))
        if delay > 0.0:
            time.sleep(delay)

    def _start_px4_stack_for_episode_reset(self, reason):
        if self.px4_backend is None:
            return

        carb.log_warn(f"[APP][PX4] Starting fresh PX4 stack for next episode: reason={reason}")
        try:
            self.px4_backend.start()
            self._mavsdk_bridge_started = False
            self._mavsdk_bridge_wait_logged = False
        except Exception as exc:
            carb.log_warn(f"[APP][PX4] PX4 backend start failed: {exc}")
            return
        carb.log_warn("[APP][PX4] MAVSDK bridge will restart after the next PX4 heartbeat.")

    def _force_clear_px4_backend(self):
        if self.px4_backend is None:
            return

        px4_tool = getattr(self.px4_backend, "px4_tool", None)
        if px4_tool is not None:
            try:
                px4_tool.kill_px4()
            except Exception as exc:
                carb.log_warn(f"[APP][PX4] PX4 process kill cleanup failed: {exc}")
            try:
                self.px4_backend.px4_tool = None
            except Exception:
                pass

        connection = getattr(self.px4_backend, "_connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
            try:
                self.px4_backend._connection = None
            except Exception:
                pass

        for name, value in (
            ("_is_running", False),
            ("_received_first_actuator", False),
            ("_received_actuator", False),
            ("_received_first_hearbeat", False),
        ):
            try:
                setattr(self.px4_backend, name, value)
            except Exception:
                pass

        rotor_data = getattr(self.px4_backend, "_rotor_data", None)
        zero_input = getattr(rotor_data, "zero_input_reference", None)
        if callable(zero_input):
            try:
                zero_input()
            except Exception:
                pass

    def _current_px4_process(self):
        if self.px4_backend is None:
            return None
        px4_tool = getattr(self.px4_backend, "px4_tool", None)
        if px4_tool is None:
            return None
        return getattr(px4_tool, "px4_process", None)

    def _wait_for_px4_process_exit(self, process):
        if process is None:
            return

        try:
            process.wait(timeout=2.0)
            return
        except Exception:
            pass

        try:
            process.kill()
        except Exception:
            pass

        try:
            process.wait(timeout=1.0)
        except Exception:
            carb.log_warn("[APP][PX4] PX4 process did not exit cleanly after kill.")

    def _handle_timeline_resume(self):
        is_playing = self.world.is_playing()
        is_stopped = self.world.is_stopped()

        if is_stopped:
            if not self._timeline_stop_seen:
                self.data_recorder.stop(reason="manual_stop")
            self._timeline_stop_seen = True
            self._timeline_was_playing = False
            return

        if is_playing and not self._timeline_was_playing:
            if self._timeline_stop_seen:
                self._resample_scene_after_timeline_stop()
                if self.classic_controller is not None:
                    self.classic_controller.reset_after_episode("timeline_resume")
                self._timeline_stop_seen = False
            self._resume_people_after_timeline_resume()
        self._timeline_was_playing = is_playing

    def _pool_scene_for_active_scene(self, scene):
        if not CROWD_RANDOMIZE_TEMPLATE:
            return scene
        if CROWD_POOL_PEOPLE_COUNT <= scene.num_people:
            return scene

        template = {
            "num_people": CROWD_POOL_PEOPLE_COUNT,
            "group_spacing": scene.group_spacing,
            "direction": scene.direction,
            "drone_distance": scene.drone_distance,
            "speed": scene.speed,
            "seed": None,
            "walk_polygon": self.walk_polygon,
            "valid_people_counts": CROWD_TEMPLATE["valid_people_counts"],
            "drone_x_range": CROWD_TEMPLATE["drone_x_range"],
            "drone_y_range": CROWD_TEMPLATE["drone_y_range"],
            "direction_start_bounds": CROWD_TEMPLATE["direction_start_bounds"],
        }
        return build_crowd_scene(**template)

    def _resample_scene_after_timeline_stop(self):
        if CROWD_RANDOMIZE_TEMPLATE:
            template = sample_crowd_template()
        else:
            template = dict(CROWD_TEMPLATE)
        scene = build_crowd_scene(**template)
        if len(scene.person_specs) > len(self.people):
            _debug_log(
                f"[APP][CROWD] Cannot hot-resample active person count to "
                f"{len(scene.person_specs)}; pool has {len(self.people)}."
            )
            return

        self.crowd_scene = scene
        self.current_spawn_pos = self._drone_spawn_pos(scene.drone_spawn)
        self._move_drone_to_spawn(self.current_spawn_pos)
        self._move_hidden_ground_to_spawn(self.current_spawn_pos)
        self._apply_crowd_scene_to_existing_people(scene)
        self._configure_crowd_separation()
        self._sanitize_people_initial_positions("timeline-resample")
        self._enforce_people_initial_positions()
        self.skeleton_tracker.update_markers(
            self._simulation_time(), force=True
        )

    def _move_drone_to_spawn(self, spawn_pos):
        spawn_pos = np.array(spawn_pos, dtype=float)
        quat_xyzw = Rotation.from_euler(
            "XYZ",
            [0.0, 0.0, DRONE_INIT_YAW_DEG],
            degrees=True,
        ).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=float,
        )

        self.shared_cmd.reset()
        if not self._px4_stack_restart_active:
            self._reset_control_backend()
        try:
            self.drone.set_world_pose(position=spawn_pos, orientation=quat_wxyz)
        except Exception:
            self._set_prim_world_pose(self.drone_root_path, spawn_pos, quat_xyzw)

        for setter_name in ("set_linear_velocity", "set_angular_velocity"):
            setter = getattr(self.drone, setter_name, None)
            if callable(setter):
                try:
                    setter(np.zeros(3))
                except Exception:
                    pass

        if self.mavsdk_bridge is not None and not self._px4_stack_restart_active:
            self.mavsdk_bridge.reset_after_episode()

        self.drone._state.position = spawn_pos.copy()
        self.drone._state.attitude = quat_xyzw.copy()
        self.drone._state.linear_velocity = np.zeros(3)
        self.drone._state.linear_body_velocity = np.zeros(3)
        self.drone._state.angular_velocity = np.zeros(3)
        self.drone._state.linear_acceleration = np.zeros(3)
        self.drone._vehicle_dc_interface = None

    @staticmethod
    def _drone_spawn_pos(spawn_pos):
        spawn_pos = list(spawn_pos)
        spawn_pos[2] = DRONE_SPAWN_Z
        return spawn_pos

    def _move_hidden_ground_to_spawn(self, spawn_pos):
        try:
            self.hidden_ground.set_world_pose(
                position=np.array([float(spawn_pos[0]), float(spawn_pos[1]), 0.0])
            )
        except Exception:
            self._set_prim_world_pose(
                "/World/hidden_ground",
                [float(spawn_pos[0]), float(spawn_pos[1]), 0.0],
                None,
            )

    def _set_prim_world_pose(self, prim_path, position, quat_xyzw=None):
        prim = self.world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(tuple(float(value) for value in position))
        if quat_xyzw is not None:
            euler_deg = Rotation.from_quat(quat_xyzw).as_euler("XYZ", degrees=True)
            xform.SetRotate(
                tuple(float(value) for value in euler_deg),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )

    def _apply_crowd_scene_to_existing_people(self, scene):
        self.person_initial_positions = {}
        self.person_specs_by_name = {}
        safe_initial_positions = self._formation_safe_initial_positions(scene)

        for index, (person, controller) in enumerate(zip(self.people, self.person_controllers)):
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            if index >= len(scene.person_specs):
                self._park_inactive_person(person, controller, index)
                continue

            spec = scene.person_specs[index]
            init_pos = list(safe_initial_positions.get(spec.name, spec.init_pos))

            self.person_initial_positions[person_name] = init_pos
            self.person_specs_by_name[person_name] = spec
            self.person_specs_by_name[spec.name] = spec
            self._configure_existing_person_controller(controller, spec)
            self._reset_person_to_initial_pose(person, init_pos, spec.init_yaw)
            self._set_person_visibility(person, True)

    def _park_inactive_person(self, person, controller, index):
        person_name = person._stage_prefix.rstrip("/").split("/")[-1]
        park_pos = [
            self.inactive_person_park_origin[0] + float(index) * 2.0,
            self.inactive_person_park_origin[1],
            self.inactive_person_park_origin[2],
        ]
        self.person_initial_positions[person_name] = park_pos
        if hasattr(controller, "set_waypoints"):
            controller.set_waypoints([park_pos])
        if hasattr(controller, "speed"):
            controller.speed = 0.0
        if hasattr(controller, "loop"):
            controller.loop = False
        self._reset_person_to_initial_pose(person, park_pos, 0.0)
        person.update_target_position(park_pos, 0.0)
        self._set_person_visibility(person, False)

    def _set_person_visibility(self, person, visible):
        prim = self.world.stage.GetPrimAtPath(person._stage_prefix)
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _configure_existing_person_controller(self, controller, spec):
        controller.crowd_group_id = spec.group_id
        if hasattr(controller, "speed"):
            controller.speed = spec.speed
        if hasattr(controller, "change_interval"):
            controller.change_interval = spec.change_interval
        if hasattr(controller, "loop"):
            controller.loop = spec.loop

        configure_natural = getattr(controller, "configure_natural_motion", None)
        if callable(configure_natural):
            configure_natural(
                speed=spec.speed,
                start_waypoint_index=spec.start_waypoint_index,
                route_direction=spec.route_direction,
                speed_seed=spec.speed_seed,
                speed_change_interval=spec.change_interval,
                formation_lateral=spec.formation_lateral,
                formation_longitudinal=spec.formation_longitudinal,
                group_size=spec.group_size,
                member_index=spec.member_index,
                traffic_axis=getattr(spec, "traffic_axis", ""),
                traffic_gates=getattr(spec, "traffic_gates", ()),
                traffic_cycle_sec=getattr(spec, "traffic_cycle_sec", 18.0),
                traffic_green_start_sec=getattr(
                    spec, "traffic_green_start_sec", 0.0
                ),
                traffic_green_duration_sec=getattr(
                    spec, "traffic_green_duration_sec", 7.5
                ),
                traffic_clearance_sec=getattr(
                    spec, "traffic_clearance_sec", 1.5
                ),
                planned_walk_sec=getattr(spec, "planned_walk_sec", 0.0),
                planned_pause_sec=getattr(spec, "planned_pause_sec", 0.0),
                planned_pause_phase_sec=getattr(
                    spec, "planned_pause_phase_sec", 0.0
                ),
            )

        if hasattr(controller, "set_waypoints"):
            waypoints = [
                nearest_walkable_or_sample(
                    waypoint,
                    self.walk_polygon,
                    self.obstacle_aabbs,
                )
                for waypoint in spec.waypoints
            ]
            controller.set_waypoints(waypoints)
        elif hasattr(controller, "resume_after_pause"):
            controller.resume_after_pause()

    def _reset_person_to_initial_pose(self, person, init_pos, init_yaw):
        self._set_person_root_yaw(person, init_yaw)
        reference_path = self._person_reference_path(person)
        reference_pos = self._get_prim_world_position(reference_path)
        self._move_person_reference_to_world_position(person, init_pos, reference_pos)
        person._state.attitude = Rotation.from_euler("z", init_yaw, degrees=False).as_quat()
        person._state.linear_velocity = np.zeros(3)
        person._state.linear_acceleration = np.zeros(3)
        person.update_target_position(init_pos, 0.0)
        if person.character_graph is not None:
            try:
                person.character_graph.set_variable("Walk", 0.0)
                person.character_graph.set_variable("Action", "Idle")
            except Exception:
                pass

    def _resume_people_after_timeline_resume(self):
        resumed_count = 0
        for controller in self.person_controllers:
            resume = getattr(controller, "resume_after_pause", None)
            if callable(resume):
                resume()
                resumed_count += 1
        if resumed_count:
            _debug_log(
                f"[APP][CROWD] Resynced {resumed_count} pedestrian controllers after timeline resume."
            )

    def _create_people_from_template(self, scene):
        people = []
        controllers = []
        initial_positions = {}
        self.person_specs_by_name = {}
        safe_initial_positions = self._formation_safe_initial_positions(scene)

        for spec in scene.person_specs:
            init_pos = list(safe_initial_positions.get(spec.name, spec.init_pos))
            initial_positions[spec.name] = init_pos
            self.person_specs_by_name[spec.name] = spec

            if spec.controller_kind == "random":
                controller = ObstacleAwarePolygonPersonController(
                    polygon_points=spec.motion_polygon or self.walk_polygon,
                    obstacle_aabbs=self.obstacle_aabbs,
                    change_interval=spec.change_interval,
                    speed=spec.speed,
                )
            elif spec.controller_kind == "natural_waypoint":
                waypoints = [
                    nearest_walkable_or_sample(
                        waypoint,
                        self.walk_polygon,
                        self.obstacle_aabbs,
                    )
                    for waypoint in spec.waypoints
                ]
                controller = NaturalFormationWaypointController(
                    waypoints=waypoints,
                    obstacle_aabbs=self.obstacle_aabbs,
                    polygon_points=self.walk_polygon,
                    speed=spec.speed,
                    start_waypoint_index=spec.start_waypoint_index,
                    route_direction=spec.route_direction,
                    speed_seed=spec.speed_seed,
                    speed_change_interval=spec.change_interval,
                    formation_lateral=spec.formation_lateral,
                    formation_longitudinal=spec.formation_longitudinal,
                    group_size=spec.group_size,
                    member_index=spec.member_index,
                    traffic_axis=getattr(spec, "traffic_axis", ""),
                    traffic_gates=getattr(spec, "traffic_gates", ()),
                    traffic_cycle_sec=getattr(spec, "traffic_cycle_sec", 18.0),
                    traffic_green_start_sec=getattr(
                        spec, "traffic_green_start_sec", 0.0
                    ),
                    traffic_green_duration_sec=getattr(
                        spec, "traffic_green_duration_sec", 7.5
                    ),
                    traffic_clearance_sec=getattr(
                        spec, "traffic_clearance_sec", 1.5
                    ),
                    planned_walk_sec=getattr(spec, "planned_walk_sec", 0.0),
                    planned_pause_sec=getattr(spec, "planned_pause_sec", 0.0),
                    planned_pause_phase_sec=getattr(
                        spec, "planned_pause_phase_sec", 0.0
                    ),
                    control_hz=PEDESTRIAN_CONTROLLER_HZ,
                    replan_interval=PEDESTRIAN_REPLAN_INTERVAL_SEC,
                    sample_attempts=PEDESTRIAN_PATH_SAMPLE_ATTEMPTS,
                    fallback_sample_attempts=PEDESTRIAN_FALLBACK_SAMPLE_ATTEMPTS,
                    log_interval=PEDESTRIAN_HOLD_LOG_INTERVAL_SEC,
                )
            else:
                waypoints = [
                    nearest_walkable_or_sample(
                        waypoint,
                        self.walk_polygon,
                        self.obstacle_aabbs,
                    )
                    for waypoint in spec.waypoints
                ]
                controller = WaypointPersonController(
                    waypoints=waypoints,
                    obstacle_aabbs=self.obstacle_aabbs,
                    polygon_points=self.walk_polygon,
                    speed=spec.speed,
                    loop=spec.loop,
                )
            controller.crowd_group_id = spec.group_id

            person = Person(
                spec.name,
                spec.character_name,
                init_pos=init_pos,
                init_yaw=spec.init_yaw,
                controller=controller,
            )
            actual_person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            initial_positions[actual_person_name] = init_pos
            self.person_specs_by_name[actual_person_name] = spec
            people.append(person)
            controllers.append(controller)

        return people, controllers, initial_positions

    def _formation_safe_initial_positions(self, scene):
        """Translate whole formations to safety without splitting members."""
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return {spec.name: list(spec.init_pos) for spec in scene.person_specs}
        grouped = {}
        for spec in scene.person_specs:
            grouped.setdefault(spec.group_id, []).append(spec)
        occupied = []
        result = {}
        front_min_x, front_max_x = self._front_x_limits()

        def formation_is_clear(positions):
            if not all(
                point_is_front_walkable(
                    position[0], position[1], self.walk_polygon,
                    self.obstacle_aabbs, front_min_x, max_x=front_max_x,
                )
                for position in positions
            ):
                return False
            return all(
                math.hypot(position[0] - other[0], position[1] - other[1])
                >= PEDESTRIAN_INITIAL_MIN_DISTANCE
                for position in positions for other in occupied
            )

        for group_id in sorted(grouped):
            specs = grouped[group_id]
            raw = [list(spec.init_pos) for spec in specs]
            candidates = [(0.0, 0.0)]
            for radius in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0):
                candidates.extend(
                    (
                        radius * math.cos(2.0 * math.pi * index / 32.0),
                        radius * math.sin(2.0 * math.pi * index / 32.0),
                    )
                    for index in range(32)
                )
            chosen = None
            for shift_x, shift_y in candidates:
                translated = [
                    [position[0] + shift_x, position[1] + shift_y, position[2]]
                    for position in raw
                ]
                if formation_is_clear(translated):
                    chosen = translated
                    break
            if chosen is None:
                chosen = [
                    nearest_walkable_or_sample(
                        position, self.walk_polygon, self.obstacle_aabbs
                    )
                    for position in raw
                ]
            for spec, position in zip(specs, chosen):
                result[spec.name] = position
                occupied.append(position)
        return result

    def _configure_crowd_separation(self):
        for controller in self.person_controllers:
            if not hasattr(controller, "configure_crowd_separation"):
                continue
            controller.configure_crowd_separation(
                self.people,
                min_distance=PEDESTRIAN_PERSONAL_SPACE,
                active_distance=PEDESTRIAN_SEPARATION_ACTIVE_DISTANCE,
                step_distance=PEDESTRIAN_SEPARATION_STEP,
                strength=PEDESTRIAN_SEPARATION_STRENGTH,
                prediction_time=PEDESTRIAN_PREDICTIVE_LOOKAHEAD_TIME,
                path_conflict_distance=PEDESTRIAN_PREDICTIVE_CONFLICT_DISTANCE,
                path_time_margin=PEDESTRIAN_PREDICTIVE_TIME_MARGIN,
                path_side_step=PEDESTRIAN_PREDICTIVE_SIDE_STEP,
                same_group_min_distance=PEDESTRIAN_SAME_GROUP_MIN_DISTANCE,
                max_avoidance_turn_deg=PEDESTRIAN_MAX_AVOIDANCE_TURN_DEG,
            )
        _debug_log(
            f"[APP][CROWD] Pedestrian separation enabled: "
            f"min={PEDESTRIAN_PERSONAL_SPACE:.2f}m, "
            f"active={PEDESTRIAN_SEPARATION_ACTIVE_DISTANCE:.2f}m, "
            f"predictive_conflict={PEDESTRIAN_PREDICTIVE_CONFLICT_DISTANCE:.2f}m"
        )

    def _monitor_crowd_clearance(self):
        """Report the closest violating pair and whether it is one group."""
        now = time.monotonic()
        if (
            now - self._last_crowd_clearance_check_wall
            < PEDESTRIAN_CLEARANCE_CHECK_INTERVAL_SEC
        ):
            return
        self._last_crowd_clearance_check_wall = now

        closest = None
        for index, person in enumerate(self.people):
            position = getattr(getattr(person, "state", None), "position", None)
            if position is None:
                continue
            for other in self.people[index + 1:]:
                other_position = getattr(getattr(other, "state", None), "position", None)
                if other_position is None:
                    continue
                distance = math.hypot(
                    float(position[0]) - float(other_position[0]),
                    float(position[1]) - float(other_position[1]),
                )
                first_controller = getattr(person, "_controller", None)
                second_controller = getattr(other, "_controller", None)
                first_group = getattr(first_controller, "crowd_group_id", None)
                second_group = getattr(second_controller, "crowd_group_id", None)
                same_group = first_group is not None and first_group == second_group
                required = (
                    PEDESTRIAN_SAME_GROUP_MIN_DISTANCE
                    if same_group else PEDESTRIAN_PERSONAL_SPACE
                )
                if distance >= required:
                    continue
                violation = distance / max(required, 1e-3)
                if closest is None or violation < closest[0]:
                    closest = (violation, distance, required, person, other)

        if closest is None:
            return
        if now - self._last_crowd_clearance_log_wall < PEDESTRIAN_CLEARANCE_LOG_INTERVAL_SEC:
            return
        self._last_crowd_clearance_log_wall = now

        _, distance, required, first, second = closest
        first_controller = getattr(first, "_controller", None)
        second_controller = getattr(second, "_controller", None)
        first_group = getattr(first_controller, "crowd_group_id", None)
        second_group = getattr(second_controller, "crowd_group_id", None)
        first_name = first._stage_prefix.rstrip("/").split("/")[-1]
        second_name = second._stage_prefix.rstrip("/").split("/")[-1]
        carb.log_warn(
            f"[CROWD][CLEARANCE] {first_name}(group={first_group}) <-> "
            f"{second_name}(group={second_group}), distance={distance:.2f}m, "
            f"required={required:.2f}m, "
            f"same_group={first_group is not None and first_group == second_group}"
        )

    def _export_crowd_map_state(self):
        """Publish a small atomic JSON snapshot for the external map window."""
        if not CROWD_MAP_STATE_PATH:
            return
        now = time.monotonic()
        if now - self._last_crowd_map_export_wall < CROWD_MAP_EXPORT_INTERVAL_SEC:
            return
        self._last_crowd_map_export_wall = now

        people = []
        for person in self.people:
            position = getattr(getattr(person, "state", None), "position", None)
            if position is None or abs(float(position[0])) > 100.0 or abs(float(position[1])) > 100.0:
                continue
            name = person._stage_prefix.rstrip("/").split("/")[-1]
            controller = getattr(person, "_controller", None)
            spec = self.person_specs_by_name.get(name)
            commanded_target = getattr(controller, "_last_commanded_target", None)
            people.append({
                "name": name,
                "group_id": getattr(controller, "crowd_group_id", None),
                "group_size": int(getattr(spec, "group_size", 1)),
                "member_index": int(getattr(spec, "member_index", 0)),
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "commanded_speed": float(
                    getattr(controller, "_last_commanded_speed", 0.0) or 0.0
                ),
                "commanded_target": (
                    None if commanded_target is None else [
                        float(commanded_target[0]),
                        float(commanded_target[1]),
                        float(commanded_target[2]),
                    ]
                ),
                "route_target_index": int(getattr(controller, "target_index", -1)),
                "group_passing": bool(
                    getattr(controller, "group_passing_active", False)
                ),
                "stuck_elapsed": float(
                    getattr(controller, "deadlock_elapsed", 0.0) or 0.0
                ),
                "deadlock_escape": bool(
                    getattr(controller, "deadlock_escape_target", None) is not None
                ),
                "traffic_waiting": bool(
                    getattr(controller, "traffic_waiting", False)
                ),
                "traffic_waiting_gate": getattr(
                    controller, "traffic_waiting_gate", None
                ),
                "traffic_waiting_opponent": getattr(
                    controller, "traffic_waiting_opponent", None
                ),
                "traffic_cycle_override": bool(
                    getattr(controller, "traffic_cycle_override", False)
                ),
                "traffic_axis": str(
                    getattr(controller, "traffic_axis", "") or ""
                ),
                "planned_pause": bool(
                    getattr(controller, "planned_pause_active", False)
                ),
                "initial_position": [
                    float(value)
                    for value in self.person_initial_positions.get(name, position)
                ],
            })

        drone_position = getattr(getattr(self.drone, "state", None), "position", None)
        planned_routes = []
        specs_by_group = {}
        for spec in self.crowd_scene.person_specs:
            specs_by_group.setdefault(spec.group_id, []).append(spec)
        for group_id, specs in sorted(specs_by_group.items()):
            leader = min(
                specs,
                key=lambda item: (
                    abs(float(getattr(item, "formation_lateral", 0.0))),
                    int(getattr(item, "member_index", 0)),
                ),
            )
            route_point_count = min(len(spec.waypoints) for spec in specs)
            group_center_route = [
                [
                    sum(float(spec.waypoints[index][0]) for spec in specs)
                    / len(specs),
                    sum(float(spec.waypoints[index][1]) for spec in specs)
                    / len(specs),
                ]
                for index in range(route_point_count)
            ]
            planned_routes.append({
                "group_id": group_id,
                "direction": str(getattr(leader, "traffic_axis", "") or ""),
                "points": group_center_route,
                "gates": [
                    [
                        float(gate[0]),
                        float(gate[1]),
                        float(gate[3]) if len(gate) >= 5 else None,
                        float(gate[4]) if len(gate) >= 5 else None,
                    ]
                    for gate in getattr(leader, "traffic_gates", ())
                ],
                "cycle_sec": float(
                    getattr(leader, "traffic_cycle_sec", 18.0)
                ),
                "green_start_sec": float(
                    getattr(leader, "traffic_green_start_sec", 0.0)
                ),
                "green_duration_sec": float(
                    getattr(leader, "traffic_green_duration_sec", 7.5)
                ),
                "walk_sec": float(
                    getattr(leader, "planned_walk_sec", 0.0)
                ),
                "pause_sec": float(
                    getattr(leader, "planned_pause_sec", 0.0)
                ),
            })

        payload = {
            "ready": True,
            "seed": self.crowd_scene.seed,
            "scene_key": self.crowd_scene.key,
            "wall_time": time.time(),
            "simulation_time": float(self._simulation_time()),
            "walk_polygon": [[float(p[0]), float(p[1])] for p in self.walk_polygon],
            "obstacles": [[float(value) for value in aabb] for aabb in self.obstacle_aabbs],
            "obstacle_paths": list(self.obstacle_scan.accepted_paths),
            "planned_routes": planned_routes,
            "people": people,
            "drone": None if drone_position is None else {
                "name": "OmniNxt",
                "position": [
                    float(drone_position[0]),
                    float(drone_position[1]),
                    float(drone_position[2]),
                ],
            },
        }
        temporary_path = f"{CROWD_MAP_STATE_PATH}.tmp.{os.getpid()}"
        try:
            parent = os.path.dirname(os.path.abspath(CROWD_MAP_STATE_PATH))
            os.makedirs(parent, exist_ok=True)
            with open(temporary_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary_path, CROWD_MAP_STATE_PATH)
        except Exception as exc:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            if not self._crowd_map_export_error_logged:
                self._crowd_map_export_error_logged = True
                carb.log_warn(f"[APP][CROWD][MAP] Failed to export map state: {exc}")

    def _front_x_limits(self):
        xs = [point[0] for point in self.walk_polygon]
        return min(xs), max(xs)

    def _scan_obstacles(self):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return ObstacleScanResult([], [], [], [])
        return scan_obstacles(
            self.world.stage,
            PEDESTRIAN_OBSTACLE_KEYWORDS,
            walk_polygon=self.walk_polygon,
            margin=PEDESTRIAN_OBSTACLE_MARGIN,
            root_paths=PEDESTRIAN_OBSTACLE_ROOT_PATHS,
            exclude_keywords=PEDESTRIAN_OBSTACLE_EXCLUDE_KEYWORDS,
            max_bottom_z=PEDESTRIAN_OBSTACLE_MAX_BOTTOM_Z,
            min_top_z=PEDESTRIAN_OBSTACLE_MIN_TOP_Z,
        )

    def _try_load_initial_obstacles(self):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return
        if self.obstacle_aabbs:
            return

        for attempt in range(3):
            try:
                self.simulation_app.update()
            except Exception:
                break

            self._rescan_obstacles(f"initial-load attempt {attempt + 1}")
            if self.obstacle_aabbs:
                return

    def _rescan_obstacles(self, reason):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            self.obstacle_scan = ObstacleScanResult([], [], [], [])
            self.obstacle_aabbs = []
            return
        self.obstacle_scan = self._scan_obstacles()
        self.obstacle_aabbs = self.obstacle_scan.aabbs
        _debug_log(
            f"[APP][OBSTACLE][{reason}] keyword_hits={len(self.obstacle_scan.keyword_hits)}, "
            f"accepted={len(self.obstacle_scan.accepted_paths)}, "
            f"rejected={len(self.obstacle_scan.rejected_paths)}"
        )

    def _apply_obstacles_to_controllers(self):
        for controller in self.person_controllers:
            if hasattr(controller, "set_obstacles"):
                controller.set_obstacles(self.obstacle_aabbs)

        for person, controller in zip(self.people, self.person_controllers):
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            spec = self.person_specs_by_name.get(person_name)
            if spec is None or not hasattr(controller, "set_waypoints"):
                continue

            waypoints = [
                nearest_walkable_or_sample(
                    waypoint,
                    self.walk_polygon,
                    self.obstacle_aabbs,
                )
                for waypoint in spec.waypoints
            ]
            controller.set_waypoints(waypoints)
            for idx, waypoint in enumerate(waypoints, start=1):
                _debug_log(
                    f"[APP][CROWD][WAYPOINT] {person_name} #{idx}: "
                    f"({waypoint[0]:.2f},{waypoint[1]:.2f},{waypoint[2]:.2f})"
                )

    def _sanitize_people_initial_positions(self, reason):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return
        front_min_x, front_max_x = self._front_x_limits()
        occupied = []

        for index, person in enumerate(self.people):
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            old_pos = self.person_initial_positions.get(person_name)
            if old_pos is None:
                continue

            controller = getattr(person, "_controller", None)
            group_id = getattr(controller, "crowd_group_id", None)
            safe_pos = nearest_front_walkable_or_sample(
                old_pos,
                self.walk_polygon,
                self.obstacle_aabbs,
                min_x=front_min_x,
                max_x=front_max_x,
            )
            safe_pos = self._separated_initial_position(
                safe_pos,
                occupied,
                front_min_x,
                front_max_x,
                phase=index,
                group_id=group_id,
            )
            self.person_initial_positions[person_name] = safe_pos
            occupied.append((safe_pos, group_id))

            if not point_is_front_walkable(
                safe_pos[0],
                safe_pos[1],
                self.walk_polygon,
                self.obstacle_aabbs,
                front_min_x,
                max_x=front_max_x,
            ):
                _debug_log(
                    f"[APP][CROWD][INIT_WARN] {person_name}: failed to find strict "
                    f"front walkable init, candidate=({safe_pos[0]:.2f},"
                    f"{safe_pos[1]:.2f}), reason={reason}"
                )
            elif abs(safe_pos[0] - old_pos[0]) > 1e-3 or abs(safe_pos[1] - old_pos[1]) > 1e-3:
                _debug_log(
                    f"[APP][CROWD][INIT_SAFE] {person_name}: "
                    f"({old_pos[0]:.2f},{old_pos[1]:.2f}) -> "
                    f"({safe_pos[0]:.2f},{safe_pos[1]:.2f}), reason={reason}"
                )

    def _separated_initial_position(
        self,
        preferred,
        occupied,
        front_min_x,
        front_max_x,
        phase=0,
        group_id=None,
    ):
        """Find a deterministic walkable spawn that does not overlap people."""
        minimum = max(0.5, float(PEDESTRIAN_INITIAL_MIN_DISTANCE))

        def has_clearance(candidate):
            return all(
                math.hypot(
                    float(candidate[0]) - float(other[0]),
                    float(candidate[1]) - float(other[1]),
                ) >= (
                    PEDESTRIAN_SAME_GROUP_MIN_DISTANCE
                    if group_id is not None and group_id == other_group else minimum
                )
                for other, other_group in occupied
            )

        if has_clearance(preferred):
            return list(preferred)

        # Search outwards in small rings. The per-person phase prevents a
        # group remapped away from a shelf from collapsing onto one ray.
        angle_count = 32
        phase_angle = (int(phase) * 11 % angle_count) * 2.0 * math.pi / angle_count
        for radius_scale in (1.0, 1.25, 1.55, 1.9, 2.4, 3.0, 3.8):
            radius = minimum * radius_scale
            for angle_index in range(angle_count):
                angle = phase_angle + angle_index * 2.0 * math.pi / angle_count
                candidate = [
                    float(preferred[0]) + radius * math.cos(angle),
                    float(preferred[1]) + radius * math.sin(angle),
                    float(preferred[2]) if len(preferred) >= 3 else 0.0,
                ]
                if not point_is_front_walkable(
                    candidate[0],
                    candidate[1],
                    self.walk_polygon,
                    self.obstacle_aabbs,
                    front_min_x,
                    max_x=front_max_x,
                ):
                    continue
                if has_clearance(candidate):
                    return candidate

        _debug_log(
            f"[APP][CROWD][INIT_SPACING_WARN] Could not provide "
            f"{minimum:.2f}m initial clearance near "
            f"({preferred[0]:.2f},{preferred[1]:.2f})."
        )
        return list(preferred)

    def _enforce_people_initial_positions(self):
        for person in self.people:
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            target_pos = self.person_initial_positions.get(person_name)
            if target_pos is None:
                continue

            reference_path = self._person_reference_path(person)
            before_pos = self._get_prim_world_position(reference_path)
            self._move_person_reference_to_world_position(person, target_pos, before_pos)
            before_label = (
                "unknown"
                if before_pos is None
                else f"({before_pos[0]:.2f},{before_pos[1]:.2f},{before_pos[2]:.2f})"
            )
            _debug_log(
                f"[APP][CROWD][INIT_FIX] {person_name}: reference={reference_path}, "
                f"from {before_label} to ({target_pos[0]:.2f},{target_pos[1]:.2f},"
                f"{target_pos[2]:.2f})"
            )

            after_pos = self._get_prim_world_position(reference_path)
            if after_pos is not None:
                in_walk_area = point_in_polygon_2d(
                    after_pos[0],
                    after_pos[1],
                    self.walk_polygon,
                )
                _debug_log(
                    f"[APP][CROWD][STAGE_INIT] {person_name}: "
                    f"world=({after_pos[0]:.2f},{after_pos[1]:.2f},{after_pos[2]:.2f}), "
                    f"in_walk_area={in_walk_area}"
                )

    def _relocate_people_if_needed(self, reason):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return
        front_min_x, front_max_x = self._front_x_limits()
        moved_count = 0

        for person in self.people:
            person_name = person._stage_prefix.rstrip("/").split("/")[-1]
            reference_path = self._person_reference_path(person)
            reference_pos = self._get_prim_world_position(reference_path)
            if reference_pos is not None and point_is_front_walkable(
                reference_pos[0],
                reference_pos[1],
                self.walk_polygon,
                self.obstacle_aabbs,
                front_min_x,
                max_x=front_max_x,
            ):
                continue

            preferred = reference_pos or self.person_initial_positions.get(
                person_name,
                [front_min_x + 1.0, self.current_spawn_pos[1], 0.0],
            )
            safe_pos = nearest_front_walkable_or_sample(
                preferred,
                self.walk_polygon,
                self.obstacle_aabbs,
                min_x=front_min_x,
                max_x=front_max_x,
            )
            self.person_initial_positions[person_name] = safe_pos
            self._move_person_reference_to_world_position(person, safe_pos, reference_pos)
            moved_count += 1

            before_label = (
                "unknown"
                if reference_pos is None
                else f"({reference_pos[0]:.2f},{reference_pos[1]:.2f},{reference_pos[2]:.2f})"
            )
            _debug_log(
                f"[APP][CROWD][RELOCATE] {person_name}: {before_label} -> "
                f"({safe_pos[0]:.2f},{safe_pos[1]:.2f},{safe_pos[2]:.2f}), "
                f"reason={reason}"
            )

        if moved_count:
            _debug_log(f"[APP][CROWD] Relocated {moved_count} pedestrians after {reason}.")

    def _get_prim_world_position(self, prim_path):
        prim = self.world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        translate = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
        return [float(translate[0]), float(translate[1]), float(translate[2])]

    def _person_reference_path(self, person):
        skel_root_path = getattr(person, "character_skel_root_stage_path", None)
        if skel_root_path:
            prim = self.world.stage.GetPrimAtPath(skel_root_path)
            if prim.IsValid():
                return skel_root_path
        return person._stage_prefix

    def _move_person_reference_to_world_position(self, person, target_pos, reference_pos=None):
        prim = self.world.stage.GetPrimAtPath(person._stage_prefix)
        if not prim.IsValid():
            _debug_log(f"[APP][CROWD][INIT_FIX] Invalid person prim: {person._stage_prefix}")
            return

        root_pos = self._get_prim_world_position(person._stage_prefix)
        if root_pos is None:
            root_pos = [0.0, 0.0, 0.0]
        if reference_pos is None:
            reference_pos = root_pos

        target_reference_world = Gf.Vec3d(
            float(target_pos[0]),
            float(target_pos[1]),
            float(target_pos[2]),
        )
        root_world = Gf.Vec3d(float(root_pos[0]), float(root_pos[1]), float(root_pos[2]))
        reference_world = Gf.Vec3d(
            float(reference_pos[0]),
            float(reference_pos[1]),
            float(reference_pos[2]),
        )
        target_root_world = root_world + (target_reference_world - reference_world)

        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_world = omni.usd.get_world_transform_matrix(parent)
            target_local = parent_world.GetInverse().Transform(target_root_world)
        else:
            target_local = target_root_world

        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(target_local)

        person._state.position = np.array(target_pos, dtype=float)
        person._previous_position = np.array(target_pos, dtype=float)
        person._target_position = np.array(target_pos, dtype=float)

    def _set_person_root_yaw(self, person, yaw_rad):
        prim = self.world.stage.GetPrimAtPath(person._stage_prefix)
        if not prim.IsValid():
            return

        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetRotate(
            (0.0, 0.0, float(np.degrees(yaw_rad))),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    def _refresh_obstacles_if_needed(self):
        if not ENABLE_PEDESTRIAN_OBSTACLE_AVOIDANCE:
            return
        if self.obstacle_aabbs or self._frame_count > 120 or self._frame_count % 30 != 0:
            return

        self.obstacle_scan = scan_obstacles(
            self.world.stage,
            PEDESTRIAN_OBSTACLE_KEYWORDS,
            walk_polygon=self.walk_polygon,
            margin=PEDESTRIAN_OBSTACLE_MARGIN,
            root_paths=PEDESTRIAN_OBSTACLE_ROOT_PATHS,
        )
        self.obstacle_aabbs = self.obstacle_scan.aabbs
        _debug_log(
            f"[APP][OBSTACLE][refresh frame {self._frame_count}] "
            f"keyword_hits={len(self.obstacle_scan.keyword_hits)}, "
            f"accepted={len(self.obstacle_scan.accepted_paths)}, "
            f"rejected={len(self.obstacle_scan.rejected_paths)}"
        )
        self._log_obstacle_scan_details()
        self._log_obstacle_root_samples_if_empty()

        if self.obstacle_aabbs:
            self._apply_obstacles_to_controllers()
            self._relocate_people_if_needed(f"obstacle refresh frame {self._frame_count}")
            _debug_log(
                f"[APP][OBSTACLE] Updated pedestrian controllers with {len(self.obstacle_aabbs)} obstacles."
            )

    def _log_obstacle_scan_details(self):
        for idx, path in enumerate(self.obstacle_scan.accepted_paths[:80], start=1):
            aabb = self.obstacle_aabbs[idx - 1]
            _debug_log(
                f"[APP][OBSTACLE] accepted #{idx}: {path}, "
                f"inflated_xy=({aabb[0]:.2f},{aabb[1]:.2f})-({aabb[2]:.2f},{aabb[3]:.2f})"
            )
        for idx, (path, reason) in enumerate(self.obstacle_scan.rejected_paths[:80], start=1):
            _debug_log(f"[APP][OBSTACLE] rejected #{idx}: {reason}: {path}")

    def _log_obstacle_root_samples_if_empty(self):
        if self.obstacle_scan.keyword_hits:
            return

        sample_paths = sample_paths_under_roots(
            self.world.stage,
            PEDESTRIAN_OBSTACLE_ROOT_PATHS,
            limit=80,
        )
        _debug_log(
            f"[APP][OBSTACLE] keyword_hits=0, sample paths under roots "
            f"({len(sample_paths)} shown):"
        )
        for path in sample_paths:
            _debug_log(f"[APP][OBSTACLE] sample: {path}")
