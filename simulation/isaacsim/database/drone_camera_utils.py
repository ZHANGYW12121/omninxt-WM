#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
from pathlib import Path

import numpy as np
import omni.usd
from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
from pxr import Gf, Sdf, UsdGeom

from omninxt_projection import ideal_ftheta_coefficients


DEFAULT_OMNINXT_CONFIG_PATH = Path(__file__).with_name(
    "omninxt_sync_20260802_camera_config.json"
)
_OPTICAL_FROM_USD_CAMERA = np.diag([1.0, -1.0, -1.0])


def find_first_camera_under(root_path: str):
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_path) and prim.IsA(UsdGeom.Camera):
            return prim_path
    return None


def find_camera_named_under(root_path: str, camera_name: str):
    stage = omni.usd.get_context().get_stage()
    expected_name = str(camera_name)
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if (
            prim_path.startswith(root_path)
            and prim.IsA(UsdGeom.Camera)
            and prim.GetName() == expected_name
        ):
            return prim_path
    return None


def set_camera_pose_and_view(camera_path: str, set_viewport: bool = True):
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        return

    xform = UsdGeom.XformCommonAPI(camera_prim)
    xform.SetTranslate(Gf.Vec3d(0.18, 0.0, 0.08))
    xform.SetRotate((0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    if set_viewport:
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path


def set_active_viewport_perspective():
    """Restore the main viewport to Kit's free Perspective camera."""
    viewport = get_active_viewport()
    if viewport is None:
        return False
    viewport.camera_path = "/OmniverseKit_Persp"
    return True


def create_camera_viewport_window(
    camera_path: str,
    name: str,
    width: int = 640,
    height: int = 360,
    position_x: int = 60,
    position_y: int = 60,
):
    return create_viewport_window(
        name=name,
        width=int(width),
        height=int(height),
        position_x=int(position_x),
        position_y=int(position_y),
        camera_path=Sdf.Path(camera_path),
    )


def load_omninxt_camera_config(config_path=None, profile=None):
    path = _resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "profiles" not in config:
        return config
    profile_name = str(
        profile
        or os.environ.get("OMNINXT_SENSOR_MODE")
        or config.get("default_profile", "raw_mei")
    )
    if profile_name not in config["profiles"]:
        raise ValueError(
            f"Unknown OmniNxt profile {profile_name!r}; "
            f"available={tuple(config['profiles'])}"
        )
    rig = dict(config["profiles"][profile_name])
    rig["render"] = dict(config.get("render", {}))
    return {
        "schema": config.get("schema"),
        "source_geometry": config.get("source_geometry"),
        "matrix_convention": config.get("matrix_convention"),
        "selected_profile": profile_name,
        "camera_rig": rig,
    }


def omninxt_camera_resolution(config, camera_name=None, render=True):
    rig = config["camera_rig"]
    if camera_name is None:
        camera_name = rig.get("active_camera", next(iter(rig["cameras"])))
    spec = _camera_spec(config, camera_name)
    key = "render_resolution" if render else "target_resolution"
    if key in spec:
        return tuple(int(value) for value in spec[key])
    return int(rig["width"]), int(rig["height"])


def omninxt_camera_fps(config, camera_name=None):
    return float(config["camera_rig"].get("fps", 20.0))


def omninxt_intrinsics_matrix(config, camera_name="cam0"):
    spec = _camera_spec(config, camera_name)
    values = [float(v) for v in spec["intrinsics"]]
    if len(values) == 5:
        _, fx, fy, cx, cy = values
    elif len(values) == 4:
        fx, fy, cx, cy = values
    else:
        raise ValueError(
            f"Unexpected intrinsics for {camera_name}: {spec['intrinsics']}"
        )
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def configure_omninxt_camera(
    camera_sensor,
    camera_path,
    config,
    camera_name="cam0",
    set_viewport=False,
    enable_gt_range=False,
):
    """Apply the OmniNxt cam config to the existing Pegasus MonocularCamera.

    Returns True only after the underlying Isaac Camera has been initialized and
    the renderer-facing distortion properties have been overwritten.
    """
    if not camera_path:
        return False

    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        return False

    rig = config["camera_rig"]
    spec = _camera_spec(config, camera_name)
    if spec.get("kind") == "raw_mei" or len(spec.get("intrinsics", ())) == 5:
        fit = {
            "coefficients": ideal_ftheta_coefficients(
                spec.get("render_resolution", spec["target_resolution"]),
                float(spec.get("render_fov_deg", 250.0)),
            ),
            "theta_rms_error_deg": 0.0,
            "theta_max_error_deg": 0.0,
        }
    else:
        fit = {
            "coefficients": [0.0] * 5,
            "theta_rms_error_deg": 0.0,
            "theta_max_error_deg": 0.0,
        }

    _set_pose_from_t_cam_imu(camera_prim, spec)
    _write_omninxt_metadata(camera_prim, rig, camera_name, spec, fit)
    if set_viewport:
        _set_viewport_camera(camera_path)
    _update_pegasus_camera_metadata(camera_sensor, config, camera_name)

    camera = getattr(camera_sensor, "_camera", None)
    if camera is None or not bool(getattr(camera_sensor, "_camera_full_set", False)):
        return False

    _configure_isaac_camera_object(camera, rig, spec, fit)
    if enable_gt_range and hasattr(camera, "add_distance_to_camera_to_frame"):
        camera.add_distance_to_camera_to_frame()
    return True


def save_camera_rgb_preview(camera_sensor, output_path):
    camera = getattr(camera_sensor, "_camera", None)
    if camera is None:
        return False

    for method_name in ("get_rgb_image", "get_rgb", "get_rgba"):
        method = getattr(camera, method_name, None)
        if method is None:
            continue
        try:
            image = method()
            if image is None:
                continue
            image = np.asarray(image)
            if image.size == 0:
                continue
            _write_preview_image(image, output_path)
            return True
        except Exception:
            continue
    return False


def _resolve_config_path(config_path):
    if config_path is None:
        return DEFAULT_OMNINXT_CONFIG_PATH

    path = Path(config_path)
    if path.is_absolute():
        return path

    local_path = Path(__file__).resolve().parent / path
    if local_path.exists():
        return local_path

    return Path.cwd() / path


def _write_preview_image(image, output_path):
    path = _resolve_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] >= 3:
        image = image[:, :, :3]
    image = np.clip(image, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except Exception:
        pass

    ppm_path = path.with_suffix(".ppm")
    with ppm_path.open("wb") as f:
        height, width = image.shape[:2]
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        f.write(np.ascontiguousarray(image[:, :, :3]).tobytes())


def _resolve_output_path(output_path):
    path = Path(output_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _camera_spec(config, camera_name):
    cameras = config["camera_rig"]["cameras"]
    if camera_name not in cameras:
        raise KeyError(f"OmniNxt camera {camera_name!r} is not present in config")
    return cameras[camera_name]


def _set_viewport_camera(camera_path):
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = camera_path


def _update_pegasus_camera_metadata(camera_sensor, config, camera_name):
    if camera_sensor is None:
        return
    width, height = omninxt_camera_resolution(config, camera_name, render=True)
    intrinsics = omninxt_intrinsics_matrix(config, camera_name)
    camera_sensor._resolution = (width, height)
    camera_sensor._frequency = omninxt_camera_fps(config, camera_name)
    camera_sensor._intrinsics = intrinsics
    camera_sensor.fx = float(intrinsics[0, 0])
    camera_sensor.fy = float(intrinsics[1, 1])
    camera_sensor.cx = float(intrinsics[0, 2])
    camera_sensor.cy = float(intrinsics[1, 2])


def _configure_isaac_camera_object(camera, rig, spec, fit):
    width, height = (
        int(value) for value in spec.get(
            "render_resolution",
            spec.get("target_resolution", (rig.get("width"), rig.get("height"))),
        )
    )
    fps = float(rig.get("fps", 20.0))
    fov = float(spec.get("render_fov_deg", rig.get("nominal_fov_deg", 190.0)))
    render = rig.get("render", {})
    values = [float(v) for v in spec["intrinsics"]]
    raw_mei = len(values) == 5
    if raw_mei:
        _, target_fx, target_fy, target_cx, target_cy = values
        # The renderer creates a square ideal equidistant bridge.  The exact
        # calibrated optical centre and focal values are applied by the static
        # Mei remap, not by the intermediate source camera.
        source_focal_px = min(width, height) / (
            2.0 * math.radians(fov * 0.5)
        )
        fx = fy = source_focal_px
        cx = (width - 1.0) * 0.5
        cy = (height - 1.0) * 0.5
    else:
        fx, fy, cx, cy = values

    if hasattr(camera, "set_resolution"):
        try:
            camera.set_resolution((width, height), maintain_square_pixels=True)
        except TypeError:
            camera.set_resolution((width, height))

    if hasattr(camera, "set_frequency"):
        camera.set_frequency(fps)

    clipping = render.get("clipping_range_m", [0.03, 100.0])
    if hasattr(camera, "set_clipping_range"):
        camera.set_clipping_range(float(clipping[0]), float(clipping[1]))

    pixel_size_m = float(render.get("pixel_size_um", 3.0)) * 1.0e-6
    focal_length_m = pixel_size_m * 0.5 * (fx + fy)
    horizontal_aperture_m = pixel_size_m * width
    vertical_aperture_m = pixel_size_m * height

    if hasattr(camera, "set_focal_length"):
        camera.set_focal_length(focal_length_m)
    if hasattr(camera, "set_horizontal_aperture"):
        camera.set_horizontal_aperture(horizontal_aperture_m)
    if hasattr(camera, "set_vertical_aperture"):
        camera.set_vertical_aperture(vertical_aperture_m)
    if hasattr(camera, "set_focus_distance"):
        camera.set_focus_distance(float(render.get("focus_distance_m", 3.0)))
    if hasattr(camera, "set_lens_aperture"):
        camera.set_lens_aperture(float(render.get("f_stop", 1.8)))

    # Raw mode uses an ideal equidistant bridge which is remapped exactly to
    # Mei+radtan after rendering.  Validation cameras remain pinhole.
    if raw_mei and hasattr(camera, "set_ftheta_properties"):
        camera.set_ftheta_properties(
            nominal_height=float(height),
            nominal_width=float(width),
            optical_center=(float(cx), float(cy)),
            max_fov=fov,
            distortion_coefficients=fit["coefficients"],
        )
    elif raw_mei and hasattr(camera, "set_projection_type") and hasattr(
        camera, "set_fisheye_polynomial_properties"
    ):
        camera.set_projection_type("fisheyePolynomial")
        camera.set_fisheye_polynomial_properties(
            nominal_width=float(width),
            nominal_height=float(height),
            optical_centre_x=float(cx),
            optical_centre_y=float(cy),
            max_fov=fov,
            polynomial=fit["coefficients"],
        )

    elif hasattr(camera, "set_projection_type"):
        camera.set_projection_type("pinhole")

    prim = getattr(camera, "prim", None)
    if prim is not None and raw_mei:
        _set_legacy_ftheta_attrs(prim, width, height, cx, cy, fov, fit["coefficients"])
        _set_attr(prim, "omninxt:activeXi", Sdf.ValueTypeNames.Float, values[0])


def _set_pose_from_t_cam_imu(camera_prim, spec):
    t_cam_imu = np.array(spec["T_cam_imu"], dtype=float)
    t_imu_cam = np.linalg.inv(t_cam_imu)
    body_from_optical = t_imu_cam[:3, :3]
    translation = t_imu_cam[:3, 3]
    body_from_usd_camera = body_from_optical @ _OPTICAL_FROM_USD_CAMERA

    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*[float(v) for v in translation]))

    qw, qx, qy, qz = _quaternion_from_matrix(body_from_usd_camera)
    orient_op.Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))


def _quaternion_from_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m[2, 1] - m[1, 2]) / scale
        qy = (m[0, 2] - m[2, 0]) / scale
        qz = (m[1, 0] - m[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / scale
            qx = 0.25 * scale
            qy = (m[0, 1] + m[1, 0]) / scale
            qz = (m[0, 2] + m[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / scale
            qx = (m[0, 1] + m[1, 0]) / scale
            qy = 0.25 * scale
            qz = (m[1, 2] + m[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / scale
            qx = (m[0, 2] + m[2, 0]) / scale
            qy = (m[1, 2] + m[2, 1]) / scale
            qz = 0.25 * scale

    quat = np.array([qw, qx, qy, qz], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm > 0.0:
        quat /= norm
    return quat.tolist()


def _fit_mei_to_ftheta(spec, fov_deg):
    xi, fx, fy, cx, cy = [float(v) for v in spec["intrinsics"]]
    k1, k2, p1, p2 = [float(v) for v in spec["distortion"]]
    theta_max = math.radians(float(fov_deg) * 0.5)
    theta_values = np.linspace(0.0, theta_max, 96)
    phi_values = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)

    radii = []
    thetas = []
    for theta in theta_values:
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for phi in phi_values:
            x_3d = sin_t * math.cos(phi)
            y_3d = sin_t * math.sin(phi)
            z_3d = cos_t
            denom = z_3d + xi
            if abs(denom) < 1.0e-9:
                continue
            x = x_3d / denom
            y = y_3d / denom
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            u = fx * x_d + cx
            v = fy * y_d + cy
            radius = math.hypot(u - cx, v - cy)
            if math.isfinite(radius):
                radii.append(radius)
                thetas.append(theta)

    radii = np.asarray(radii, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    coeff_desc = np.polyfit(radii, thetas, deg=4)
    theta_fit = np.polyval(coeff_desc, radii)
    residual = theta_fit - thetas
    return {
        "coefficients": [float(v) for v in coeff_desc[::-1]],
        "theta_rms_error_deg": float(math.degrees(math.sqrt(float(np.mean(residual * residual))))),
        "theta_max_error_deg": float(math.degrees(float(np.max(np.abs(residual))))),
    }


def _write_omninxt_metadata(camera_prim, rig, camera_name, spec, fit):
    values = [float(v) for v in spec["intrinsics"]]
    if len(values) == 5:
        xi, fx, fy, cx, cy = values
    else:
        fx, fy, cx, cy = values
        xi = 0.0
    k1, k2, p1, p2 = [float(v) for v in spec.get("distortion", [0.0] * 4)]
    _set_attr(camera_prim, "omninxt:rigName", Sdf.ValueTypeNames.String, rig["name"])
    _set_attr(camera_prim, "omninxt:cameraName", Sdf.ValueTypeNames.String, camera_name)
    _set_attr(camera_prim, "omninxt:cameraAlias", Sdf.ValueTypeNames.String, spec.get("alias", camera_name))
    _set_attr(camera_prim, "omninxt:projectionModel", Sdf.ValueTypeNames.String, spec.get("projection_model", rig["projection_model"]))
    _set_attr(camera_prim, "omninxt:distortionModel", Sdf.ValueTypeNames.String, rig["distortion_model"])
    _set_attr(camera_prim, "omninxt:nativeIsaacBridge", Sdf.ValueTypeNames.String, rig.get("native_isaac_bridge", "direct_pinhole"))
    _set_attr(camera_prim, "omninxt:kind", Sdf.ValueTypeNames.String, spec.get("kind", "unknown"))
    _set_attr(camera_prim, "omninxt:xi", Sdf.ValueTypeNames.Float, xi)
    _set_attr(camera_prim, "omninxt:fx", Sdf.ValueTypeNames.Float, fx)
    _set_attr(camera_prim, "omninxt:fy", Sdf.ValueTypeNames.Float, fy)
    _set_attr(camera_prim, "omninxt:cx", Sdf.ValueTypeNames.Float, cx)
    _set_attr(camera_prim, "omninxt:cy", Sdf.ValueTypeNames.Float, cy)
    _set_attr(camera_prim, "omninxt:k1", Sdf.ValueTypeNames.Float, k1)
    _set_attr(camera_prim, "omninxt:k2", Sdf.ValueTypeNames.Float, k2)
    _set_attr(camera_prim, "omninxt:p1", Sdf.ValueTypeNames.Float, p1)
    _set_attr(camera_prim, "omninxt:p2", Sdf.ValueTypeNames.Float, p2)
    _set_attr(camera_prim, "omninxt:T_cam_imu", Sdf.ValueTypeNames.String, json.dumps(spec["T_cam_imu"]))
    _set_attr(camera_prim, "omninxt:targetResolution", Sdf.ValueTypeNames.String, json.dumps(spec.get("target_resolution")))
    _set_attr(camera_prim, "omninxt:renderResolution", Sdf.ValueTypeNames.String, json.dumps(spec.get("render_resolution")))
    _set_attr(camera_prim, "omninxt:fthetaFitRmsDeg", Sdf.ValueTypeNames.Float, fit["theta_rms_error_deg"])
    _set_attr(camera_prim, "omninxt:fthetaFitMaxDeg", Sdf.ValueTypeNames.Float, fit["theta_max_error_deg"])


def _set_legacy_ftheta_attrs(camera_prim, width, height, cx, cy, fov, coefficients):
    _set_attr(camera_prim, "cameraProjectionType", Sdf.ValueTypeNames.String, "fisheyePolynomial")
    _set_attr(camera_prim, "fthetaWidth", Sdf.ValueTypeNames.Float, float(width))
    _set_attr(camera_prim, "fthetaHeight", Sdf.ValueTypeNames.Float, float(height))
    _set_attr(camera_prim, "fthetaCx", Sdf.ValueTypeNames.Float, float(cx))
    _set_attr(camera_prim, "fthetaCy", Sdf.ValueTypeNames.Float, float(cy))
    _set_attr(camera_prim, "fthetaMaxFov", Sdf.ValueTypeNames.Float, float(fov))
    for suffix, value in zip("ABCDE", coefficients):
        _set_attr(camera_prim, f"fthetaPoly{suffix}", Sdf.ValueTypeNames.Float, float(value))


def _set_attr(prim, name, value_type, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, value_type, False)
    attr.Set(value)
    return attr
