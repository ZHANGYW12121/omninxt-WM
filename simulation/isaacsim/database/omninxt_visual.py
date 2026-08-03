#!/usr/bin/env python3
"""Visual-only OmniNxt overlay for a Pegasus multirotor stage."""

import math
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics


class OmniNxtVisual:
    """Own the referenced visual prims and their render-only rotor animation."""

    def __init__(self, spin_ops, rotor_indices, log=None):
        self._spin_ops = spin_ops
        self._rotor_indices = rotor_indices
        self._angles_deg = {name: 0.0 for name in spin_ops}
        self._last_sim_time = None
        self._log = log or (lambda _message: None)

    def update(self, sim_time, rotor_velocities, rotor_directions):
        sim_time = float(sim_time)
        if self._last_sim_time is None or sim_time < self._last_sim_time:
            self._last_sim_time = sim_time
            return

        dt = sim_time - self._last_sim_time
        self._last_sim_time = sim_time
        if dt <= 0.0:
            return

        for name, spin_op in self._spin_ops.items():
            rotor_index = self._rotor_indices[name]
            if rotor_index >= len(rotor_velocities):
                continue
            direction = (
                float(rotor_directions[rotor_index])
                if rotor_index < len(rotor_directions)
                else 1.0
            )
            angle = self._angles_deg[name]
            angle += math.degrees(float(rotor_velocities[rotor_index]) * dt * direction)
            angle %= 360.0
            spin_op.Set(angle)
            self._angles_deg[name] = angle

    def reset_clock(self):
        self._last_sim_time = None


def apply_omninxt_visual(
    stage,
    drone_root_path,
    body_usd,
    rotor_usds,
    rotor_positions,
    rotor_asset_translations,
    rotor_indices,
    scale_correction=1.0,
    body_translation=(0.0, 0.0, 0.0),
    body_yaw_deg=0.0,
    log=None,
):
    """Replace only the rendered Iris meshes with referenced OmniNxt assets."""
    log = log or (lambda _message: None)
    _validate_inputs(
        body_usd,
        rotor_usds,
        rotor_positions,
        rotor_asset_translations,
        rotor_indices,
        scale_correction,
    )

    body_path = f"{drone_root_path}/body"
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Pegasus body prim does not exist: {body_path}")

    visual_root_path = f"{body_path}/visuals/OmniNxt"
    if stage.GetPrimAtPath(visual_root_path).IsValid():
        stage.RemovePrim(visual_root_path)

    hidden_paths = []
    hidden_paths.extend(_hide_meshes_under(stage, body_path, exclude_prefix=visual_root_path))
    for rotor_index in range(4):
        hidden_paths.extend(
            _hide_meshes_under(stage, f"{drone_root_path}/rotor{rotor_index}")
        )

    visual_root = UsdGeom.Xform.Define(stage, visual_root_path).GetPrim()
    body_mount = UsdGeom.Xform.Define(stage, f"{visual_root_path}/Body")
    body_mount.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, body_translation))
    )
    body_mount.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(
        float(body_yaw_deg)
    )
    body_mount.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*([float(scale_correction)] * 3))
    )
    _add_visual_reference(stage, f"{body_mount.GetPath()}/Asset", body_usd)

    spin_ops = {}
    for name in ("front_left", "front_right", "rear_left", "rear_right"):
        mount = UsdGeom.Xform.Define(stage, f"{visual_root_path}/Rotors/{name}")
        mount.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*map(float, rotor_positions[name]))
        )
        spin = UsdGeom.Xform.Define(stage, f"{mount.GetPath()}/Spin")
        spin_ops[name] = spin.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
        spin_ops[name].Set(0.0)
        scale = UsdGeom.Xform.Define(stage, f"{spin.GetPath()}/Scale")
        scale.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*([float(scale_correction)] * 3))
        )
        correction = UsdGeom.Xform.Define(stage, f"{scale.GetPath()}/AssetCorrection")
        correction.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*map(float, rotor_asset_translations[name]))
        )
        _add_visual_reference(stage, f"{correction.GetPath()}/Asset", rotor_usds[name])

    _assert_visual_has_no_physics(stage, visual_root)
    log(
        "[APP][VISUAL] OmniNxt visual overlay applied at "
        f"{visual_root_path}; scale={float(scale_correction):.6f}, "
        f"body_yaw={float(body_yaw_deg):.1f}deg, "
        f"hidden_iris_meshes={len(hidden_paths)}"
    )
    for path in hidden_paths:
        log(f"[APP][VISUAL] Hidden original Iris mesh: {path}")

    return OmniNxtVisual(spin_ops, dict(rotor_indices), log=log)


def _validate_inputs(
    body_usd,
    rotor_usds,
    rotor_positions,
    rotor_asset_translations,
    rotor_indices,
    scale_correction,
):
    if not os.path.isfile(body_usd):
        raise FileNotFoundError(f"OmniNxt body asset does not exist: {body_usd}")
    if float(scale_correction) <= 0.0:
        raise ValueError("OmniNxt visual scale correction must be positive")

    expected = {"front_left", "front_right", "rear_left", "rear_right"}
    for name, mapping in (
        ("rotor_usds", rotor_usds),
        ("rotor_positions", rotor_positions),
        ("rotor_asset_translations", rotor_asset_translations),
        ("rotor_indices", rotor_indices),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    for path in rotor_usds.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OmniNxt rotor asset does not exist: {path}")
    if sorted(int(value) for value in rotor_indices.values()) != [0, 1, 2, 3]:
        raise ValueError("OmniNxt visual rotor indices must map one-to-one to 0..3")


def _hide_meshes_under(stage, root_path, exclude_prefix=None):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []

    hidden = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if exclude_prefix and path.startswith(exclude_prefix):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden.append(path)
    return hidden


def _add_visual_reference(stage, prim_path, asset_path):
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(os.path.abspath(asset_path))
    # Blender exports an environment light into each file. Keep its material
    # dependencies, but deactivate the light in the composed vehicle stage.
    embedded_light = stage.OverridePrim(f"{prim_path}/env_light")
    if embedded_light.IsValid():
        embedded_light.SetActive(False)
    return prim


def _assert_visual_has_no_physics(stage, visual_root):
    forbidden = []
    for prim in Usd.PrimRange(visual_root):
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            forbidden.append(str(prim.GetPath()))
    if forbidden:
        raise RuntimeError(
            "OmniNxt visual assets contain forbidden physics APIs: "
            + ", ".join(forbidden)
        )
