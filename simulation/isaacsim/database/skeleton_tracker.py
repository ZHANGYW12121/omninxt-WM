#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carb
import numpy as np
import omni.usd
import omni.anim.graph.core as ag
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

from app_config import TARGET_JOINTS


def _debug_log(*args, **kwargs):
    pass


class MultiPersonSkeletonTracker:
    DEFAULT_COLLISION_RADII = {
        "Pelvis": 0.14,
        "Head": 0.12,
        "R_Hand": 0.08,
        "L_Hand": 0.08,
        "R_Foot": 0.09,
        "L_Foot": 0.09,
        "R_KneeShareBone": 0.09,
        "L_KneeShareBone": 0.09,
        "R_ElbowShareBone": 0.08,
        "L_ElbowShareBone": 0.08,
    }

    def __init__(
        self,
        people,
        joint_names=None,
        enable_collision=True,
        collision_radius=0.08,
        contact_offset=0.02,
        rest_offset=0.0,
        update_hz=10.0,
    ):
        self.people = list(people)
        self.joint_names = list(joint_names or TARGET_JOINTS)
        self.enable_collision = enable_collision
        self.collision_radius = collision_radius
        self.contact_offset = contact_offset
        self.rest_offset = rest_offset
        self.update_interval = 1.0 / max(float(update_hz), 1e-6)
        self._last_update_simulation_time = None
        self.marker_root = "/World/PersonJointMarkers"
        self.marker_paths = {}
        self.marker_xforms = {}
        self.marker_positions = {}
        self._characters = {}
        self._warned_missing_characters = set()

    def setup(self):
        self._create_marker_root()
        stage = omni.usd.get_context().get_stage()
        for person in self.people:
            person_name = self._person_name(person)
            self._create_person_marker_root(person_name)
            self.marker_paths[person_name] = {}
            self.marker_xforms[person_name] = {}
            self.marker_positions[person_name] = {}
            for joint_name in self.joint_names:
                marker_path = self._create_joint_marker(
                    person_name, joint_name
                )
                self.marker_paths[person_name][joint_name] = marker_path
                marker_prim = stage.GetPrimAtPath(marker_path)
                if marker_prim.IsValid():
                    self.marker_xforms[person_name][joint_name] = UsdGeom.XformCommonAPI(
                        marker_prim
                    )
        if self.enable_collision:
            _debug_log(
                f"[APP] Skeleton joint collision enabled for {len(self.people)} pedestrians."
            )

    def _create_marker_root(self):
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.marker_root)

    def _create_person_marker_root(self, person_name: str):
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, f"{self.marker_root}/{person_name}")

    def _create_joint_marker(self, person_name: str, joint_name: str):
        stage = omni.usd.get_context().get_stage()

        marker_xform_path = f"{self.marker_root}/{person_name}/{joint_name}"
        sphere_path = f"{marker_xform_path}/sphere"

        xform = UsdGeom.Xform.Define(stage, marker_xform_path)
        xform_api = UsdGeom.XformCommonAPI(xform)
        xform_api.SetTranslate((0.0, 0.0, 0.0))

        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(self._collision_radius(joint_name))
        sphere.MakeInvisible()
        self._configure_collision(xform.GetPrim(), sphere.GetPrim())

        color_map = {
            "Pelvis": (1.0, 0.1, 0.1),
            "R_Hand": (1.0, 0.6, 0.1),
            "L_Hand": (1.0, 1.0, 0.1),
            "R_Foot": (0.1, 0.6, 1.0),
            "L_Foot": (0.1, 0.3, 1.0),
            "R_KneeShareBone": (0.2, 0.8, 1.0),
            "L_KneeShareBone": (0.2, 0.5, 1.0),
            "R_ElbowShareBone": (1.0, 0.5, 0.3),
            "L_ElbowShareBone": (1.0, 0.8, 0.3),
            "Head": (0.9, 0.2, 0.9),
        }
        color = color_map.get(joint_name, (1.0, 0.0, 0.0))

        material_path = f"{marker_xform_path}/Looks/marker_mat"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)

        return marker_xform_path

    def _collision_radius(self, joint_name: str):
        return self.DEFAULT_COLLISION_RADII.get(joint_name, self.collision_radius)

    @staticmethod
    def _apply_api(api_cls, prim):
        api = api_cls(prim)
        if not api:
            api = api_cls.Apply(prim)
        return api

    def _configure_collision(self, rigid_body_prim, collider_prim):
        if not self.enable_collision:
            return

        rigid_body_api = self._apply_api(UsdPhysics.RigidBodyAPI, rigid_body_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)

        collision_api = self._apply_api(UsdPhysics.CollisionAPI, collider_prim)
        collision_api.CreateCollisionEnabledAttr(True)

        if PhysxSchema is None:
            return

        contact_report_api = self._apply_api(PhysxSchema.PhysxContactReportAPI, rigid_body_prim)
        contact_report_api.CreateThresholdAttr().Set(0.0)

        physx_collision_api = self._apply_api(PhysxSchema.PhysxCollisionAPI, collider_prim)
        if self.rest_offset is not None:
            physx_collision_api.CreateRestOffsetAttr(float(self.rest_offset))
        if self.contact_offset is not None:
            physx_collision_api.CreateContactOffsetAttr(float(self.contact_offset))

    def parse_marker_collider_path(self, collider_path):
        collider_path = str(collider_path)
        marker_root = f"{self.marker_root}/"
        if not collider_path.startswith(marker_root):
            return None

        relative_path = collider_path[len(marker_root):].strip("/")
        parts = relative_path.split("/")
        if len(parts) < 2:
            return None

        return {
            "pedestrian_id": parts[0],
            "joint_name": parts[1],
            "marker_path": f"{self.marker_root}/{parts[0]}/{parts[1]}",
            "collider_path": collider_path,
        }

    @staticmethod
    def _person_name(person):
        stage_prefix = getattr(person, "_stage_prefix", None)
        if stage_prefix:
            return stage_prefix.rstrip("/").split("/")[-1]
        return getattr(person, "name", "person")

    @staticmethod
    def _skel_root_path(person):
        return getattr(person, "character_skel_root_stage_path", None)

    def get_character(self, person):
        person_name = self._person_name(person)
        if person_name not in self._characters or self._characters[person_name] is None:
            skel_root_path = self._skel_root_path(person)
            if skel_root_path is None:
                if person_name not in self._warned_missing_characters:
                    _debug_log(f"[APP] {person_name} has no skeleton root path.")
                    self._warned_missing_characters.add(person_name)
                self._characters[person_name] = None
                return None

            self._characters[person_name] = ag.get_character(skel_root_path)
            if self._characters[person_name] is not None:
                _debug_log(
                    f"[APP] Runtime animation character acquired: {person_name} ({skel_root_path})"
                )
            else:
                warning_key = (person_name, skel_root_path)
                if warning_key not in self._warned_missing_characters:
                    _debug_log(
                        f"[APP] Failed to acquire runtime animation character: {person_name} ({skel_root_path})"
                    )
                    self._warned_missing_characters.add(warning_key)
        return self._characters[person_name]

    def update_markers(self, simulation_time=None, force=False):
        """Update joint markers at a bounded simulation-time rate."""
        if simulation_time is not None:
            simulation_time = float(simulation_time)
            last_time = self._last_update_simulation_time
            if (
                not force
                and last_time is not None
                and simulation_time >= last_time
                and simulation_time - last_time + 1e-9 < self.update_interval
            ):
                return False

            if (
                last_time is None
                or simulation_time < last_time
                or force
            ):
                self._last_update_simulation_time = simulation_time
            else:
                elapsed = simulation_time - last_time
                completed_periods = max(
                    1, int(elapsed / self.update_interval)
                )
                self._last_update_simulation_time = (
                    last_time + completed_periods * self.update_interval
                )

        for person in self.people:
            person_name = self._person_name(person)
            character = self.get_character(person)
            if character is None:
                continue

            for joint_name in self.joint_names:
                pos = carb.Float3(0.0, 0.0, 0.0)
                rot = carb.Float4(0.0, 0.0, 0.0, 1.0)

                try:
                    character.get_joint_transform(joint_name, pos, rot)
                except Exception:
                    continue

                xform = self.marker_xforms.get(person_name, {}).get(joint_name)
                if xform is not None:
                    xform.SetTranslate((pos.x, pos.y, pos.z))
                    self.marker_positions[person_name][joint_name] = np.array(
                        [pos.x, pos.y, pos.z], dtype=float
                    )
        return True

    def get_joint_distances(self, drone_position):
        drone_position = np.array(drone_position, dtype=float)
        distances = []
        for person_name, joints in self.marker_positions.items():
            for joint_name, joint_position in joints.items():
                distances.append(
                    {
                        "pedestrian_id": person_name,
                        "joint_name": joint_name,
                        "distance": float(np.linalg.norm(joint_position - drone_position)),
                    }
                )
        return distances
