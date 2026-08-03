#!/usr/bin/env python3
"""Atomic OmniNxt four-camera exporter for use inside Isaac Sim.

Instantiate once with Pegasus' ``omninxt_camera_sensors`` mapping and call
``capture(frame_index, sim_time_sec)`` immediately after one rendered world step.
"""
import json
import time
import os
import shutil
from pathlib import Path

import numpy as np

from omninxt_projection import MeiRadtanRemapper


LEGACY_CAMERA_ORDER = ("cam0", "cam1", "cam2", "cam3")


class IsaacQuadcamExporter:
    def __init__(self, camera_sensors, output_root, camera_paths=None,
                 calibration_set="quadcam_drone_nxt_tmp", camera_config=None,
                 export_gt_range=False):
        self.sensors = camera_sensors
        self.root = Path(output_root).expanduser()
        self.camera_paths = camera_paths or {}
        self.calibration_set = calibration_set
        self.camera_config = camera_config
        self.export_gt_range = bool(export_gt_range)
        rig = (camera_config or {}).get("camera_rig", {})
        self.mode = str(rig.get("mode", "legacy_raw"))
        self.camera_order = tuple(
            rig.get("camera_order", LEGACY_CAMERA_ORDER)
        )
        self.camera_specs = dict(rig.get("cameras", {}))
        self.remappers = {}
        if self.mode == "raw_mei":
            for camera_id in self.camera_order:
                spec = self.camera_specs[camera_id]
                self.remappers[camera_id] = MeiRadtanRemapper(
                    spec["intrinsics"],
                    spec["distortion"],
                    spec["target_resolution"],
                    spec["render_resolution"],
                    spec["render_fov_deg"],
                )

    def capture(self, frame_index, sim_time_sec):
        final = self.root / "input" / f"frame_{int(frame_index):06d}"
        if final.exists():
            final = self.root / "input" / (
                f"frame_{int(frame_index):06d}_{time.strftime('%Y%m%d_%H%M%S')}")
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        if self.mode == "rectified_validation":
            return self._capture_rectified_validation(
                tmp, final, frame_index, sim_time_sec
            )
        return self._capture_raw_mei(tmp, final, frame_index, sim_time_sec)

    def capture_live(self, frame_index, sim_time_sec, keep_frames=4):
        """Publish one latest-only frame bundle without PNG compression.

        The live root is normally a tmpfs (``/dev/shm``).  A complete frame is
        committed by directory rename and only then exposed through ``LATEST``.
        """
        live_root = self.root / "live"
        live_root.mkdir(parents=True, exist_ok=True)
        # The Pegasus episode can reset its frame index without recreating this
        # exporter.  A monotonic wall-clock token keeps directory names unique
        # across resets while metadata still carries the simulation frame/time.
        final = live_root / f"frame_{time.monotonic_ns():020d}"
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        if self.mode == "rectified_validation":
            metadata = self._write_live_rectified(tmp, frame_index, sim_time_sec)
        else:
            metadata = self._write_live_raw(tmp, frame_index, sim_time_sec)
        (tmp / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, final)
        latest_tmp = live_root / "LATEST.tmp"
        latest_tmp.write_text(final.name + "\n", encoding="utf-8")
        os.replace(latest_tmp, live_root / "LATEST")
        # Sort by commit time, not by name.  This also upgrades safely from the
        # former frame-index naming scheme and can never classify a fresh frame
        # as older merely because the simulation index reset.
        committed = sorted(
            (path for path in live_root.glob("frame_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
        for old in committed[:-max(2, int(keep_frames))]:
            shutil.rmtree(old, ignore_errors=True)
        return final

    def _write_live_raw(self, tmp, frame_index, sim_time_sec):
        images = []
        ranges = []
        for camera_id in self.camera_order:
            source_rgb = self._rgb(self.sensors[camera_id])
            remapper = self.remappers.get(camera_id)
            images.append(
                remapper.remap(source_rgb) if remapper is not None else source_rgb
            )
            if self.export_gt_range:
                source_distance = self._range_to_camera(self.sensors[camera_id])
                distance = (
                    remapper.remap(source_distance)
                    if remapper is not None else source_distance
                )
                distance = np.asarray(distance, dtype=np.float32)
                if remapper is not None:
                    distance[~remapper.valid_mask] = np.nan
                if distance.shape != images[-1].shape[:2]:
                    raise ValueError(
                        f"{camera_id} live GT range shape {distance.shape} "
                        f"!= RGB {images[-1].shape[:2]}"
                    )
                ranges.append(np.ascontiguousarray(distance))
        quad = np.ascontiguousarray(np.concatenate(images, axis=1))
        np.save(tmp / "quad_rgb.npy", quad, allow_pickle=False)
        metadata = {
            "frame_index": int(frame_index),
            "sim_time_sec": float(sim_time_sec),
            "sensor_mode": self.mode,
            "camera_order": list(self.camera_order),
            "payload": "quad_rgb.npy",
            "shape": list(quad.shape),
            "encoding": "rgb8",
            "synchronization": "same rendered simulation step",
        }
        if self.export_gt_range:
            quad_range = np.ascontiguousarray(
                np.concatenate(ranges, axis=1), dtype=np.float32
            )
            np.save(tmp / "quad_range.npy", quad_range, allow_pickle=False)
            metadata.update({
                "gt_range_enabled": True,
                "gt_range_payload": "quad_range.npy",
                "gt_range_shape": list(quad_range.shape),
                "gt_range_encoding": "32FC1",
                "gt_range_definition": (
                    "Isaac distance_to_camera: Euclidean distance from each "
                    "physical fisheye optical center, remapped to raw Mei pixels"
                ),
            })
        else:
            metadata["gt_range_enabled"] = False
        return metadata

    def _write_live_rectified(self, tmp, frame_index, sim_time_sec):
        images = {}
        for camera_id in self.camera_order:
            gray = self._gray(self._rgb(self.sensors[camera_id]))
            images[camera_id] = self._correct_pinhole_principal_point(
                gray, self.camera_specs[camera_id]
            )
        anchors = [images[f"anchor{index}"] for index in range(4)]
        anchor_mosaic = np.ascontiguousarray(np.vstack((
            np.hstack((anchors[0], anchors[1])),
            np.hstack((anchors[2], anchors[3])),
        )))
        stereo_mosaic = np.ascontiguousarray(np.vstack([
            np.hstack((images[f"rect{pair}_left"],
                       images[f"rect{pair}_right"]))
            for pair in range(4)
        ]))
        np.save(tmp / "pose_anchor_mosaic.npy", anchor_mosaic, allow_pickle=False)
        np.save(tmp / "pose_stereo_mosaic.npy", stereo_mosaic, allow_pickle=False)
        return {
            "frame_index": int(frame_index),
            "sim_time_sec": float(sim_time_sec),
            "sensor_mode": self.mode,
            "anchor_payload": "pose_anchor_mosaic.npy",
            "stereo_payload": "pose_stereo_mosaic.npy",
            "anchor_shape": list(anchor_mosaic.shape),
            "stereo_shape": list(stereo_mosaic.shape),
            "encoding": "mono8",
            "timestamp_contract": "both mosaics share sim_time_sec exactly",
        }

    def _capture_raw_mei(self, tmp, final, frame_index, sim_time_sec):
        rgb_images = []
        range_stats = {}
        remap_diagnostics = {}
        for camera_id in self.camera_order:
            source_rgb = self._rgb(self.sensors[camera_id])
            remapper = self.remappers.get(camera_id)
            rgb = remapper.remap(source_rgb) if remapper is not None else source_rgb
            rgb_images.append(rgb)
            self._save_png(tmp / f"{camera_id}.png", rgb)
            if remapper is not None:
                diagnostics = remapper.diagnostics
                remap_diagnostics[camera_id] = {
                    "valid_fraction": diagnostics.valid_fraction,
                    "median_forward_error_px": diagnostics.median_forward_error_px,
                    "max_forward_error_px": diagnostics.max_forward_error_px,
                    "max_ray_angle_deg": diagnostics.max_ray_angle_deg,
                }
                self._save_mask(tmp / f"{camera_id}_valid_mask.png", remapper.valid_mask)
            if self.export_gt_range:
                source_distance = self._range_to_camera(self.sensors[camera_id])
                distance = (
                    remapper.remap(source_distance)
                    if remapper is not None else source_distance
                )
                if remapper is not None:
                    distance = np.asarray(distance, dtype=np.float32)
                    distance[~remapper.valid_mask] = np.nan
                if distance.shape != rgb.shape[:2]:
                    raise ValueError(f"{camera_id} GT range shape {distance.shape} != RGB {rgb.shape[:2]}")
                np.save(tmp / f"{camera_id}_range.npy", distance.astype(np.float32))
                finite = distance[np.isfinite(distance)]
                range_stats[camera_id] = {
                    "finite_pixels": int(finite.size),
                    "min_m": float(finite.min()) if finite.size else None,
                    "max_m": float(finite.max()) if finite.size else None,
                }
        shapes = {image.shape for image in rgb_images}
        if len(shapes) != 1:
            raise ValueError(f"camera image shapes differ: {sorted(shapes)}")
        quad = np.concatenate(rgb_images, axis=1)
        self._save_png(tmp / "quad_input.png", quad)
        metadata = {
            "frame_index": int(frame_index),
            "sim_time_sec": float(sim_time_sec),
            "source_resolution": [int(rgb_images[0].shape[1]), int(rgb_images[0].shape[0])],
            "quad_resolution": [int(quad.shape[1]), int(quad.shape[0])],
            "sensor_mode": self.mode,
            "camera_order": list(self.camera_order),
            "physical_camera_order": [
                self.camera_specs.get(key, {}).get("physical_name", key)
                for key in self.camera_order
            ],
            "layout": "horizontal_left_to_right",
            "encoding_on_disk": "RGB PNG",
            "ros_encoding": "bgr8 (feeder converts decoded BGR pixels)",
            "isaac_camera_prim_paths": {k: self.camera_paths.get(k) for k in self.camera_order},
            "calibration_set": self.calibration_set,
            "synchronization": "all images fetched after the same rendered simulation step",
            "rotation_or_flip": "none",
            "gt_range_enabled": self.export_gt_range,
            "gt_range_definition": "distance_to_camera (Euclidean optical-center range)",
            "gt_range_files": {k: f"{k}_range.npy" for k in self.camera_order} if self.export_gt_range else {},
            "gt_range_stats": range_stats,
            "projection_pipeline": "ideal_equidistant_Isaac_render_then_exact_Mei_radtan_remap",
            "mei_remap_diagnostics": remap_diagnostics,
        }
        return self._commit(tmp, final, metadata)

    def _capture_rectified_validation(
        self, tmp, final, frame_index, sim_time_sec
    ):
        images = {}
        ranges = {}
        for camera_id in self.camera_order:
            rgb = self._rgb(self.sensors[camera_id])
            gray = self._gray(rgb)
            spec = self.camera_specs[camera_id]
            gray = self._correct_pinhole_principal_point(gray, spec)
            images[camera_id] = gray
            self._save_gray_png(tmp / f"{camera_id}.png", gray)
            if self.export_gt_range:
                distance = self._range_to_camera(self.sensors[camera_id])
                distance = self._correct_pinhole_principal_point(distance, spec)
                ranges[camera_id] = distance.astype(np.float32)
                np.save(tmp / f"{camera_id}_range.npy", ranges[camera_id])
                if str(spec.get("kind", "")).startswith("rectified"):
                    optical_z = self._pinhole_range_to_z(distance, spec)
                    np.save(tmp / f"{camera_id}_optical_z.npy", optical_z)

        anchors = [images[f"anchor{index}"] for index in range(4)]
        anchor_mosaic = np.vstack(
            (np.hstack((anchors[0], anchors[1])),
             np.hstack((anchors[2], anchors[3])))
        )
        stereo_rows = []
        for pair_id in range(4):
            stereo_rows.append(np.hstack((
                images[f"rect{pair_id}_left"],
                images[f"rect{pair_id}_right"],
            )))
        stereo_mosaic = np.vstack(stereo_rows)
        self._save_gray_png(tmp / "pose_anchor_mosaic.png", anchor_mosaic)
        self._save_gray_png(tmp / "pose_stereo_mosaic.png", stereo_mosaic)
        metadata = {
            "frame_index": int(frame_index),
            "sim_time_sec": float(sim_time_sec),
            "sensor_mode": self.mode,
            "camera_order": list(self.camera_order),
            "camera_kinds": {
                key: self.camera_specs[key].get("kind")
                for key in self.camera_order
            },
            "isaac_camera_prim_paths": {
                key: self.camera_paths.get(key) for key in self.camera_order
            },
            "calibration_set": self.calibration_set,
            "synchronization": "all 12 images fetched after the same rendered simulation step",
            "timestamp_contract": "anchor and stereo mosaics share sim_time_sec exactly",
            "rotation_or_flip": "none",
            "anchor_mosaic": {
                "file": "pose_anchor_mosaic.png",
                "resolution": [832, 640],
                "layout": [["CAM_A", "CAM_B"], ["CAM_C", "CAM_D"]],
                "encoding": "mono8",
            },
            "stereo_mosaic": {
                "file": "pose_stereo_mosaic.png",
                "resolution": [640, 960],
                "rows": ["A_B_RIGHT", "B_C_REAR", "C_D_LEFT", "D_A_FRONT"],
                "layout_per_row": "left|right",
                "encoding": "mono8",
            },
            "gt_range_enabled": self.export_gt_range,
            "gt_depth_definition": "rectified-left optical Z for *_optical_z.npy",
        }
        return self._commit(tmp, final, metadata)

    def _commit(self, tmp, final, metadata):
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(final)
        os.replace(tmp, final)
        (self.root / "input" / "LATEST").write_text(
            final.name + "\n", encoding="utf-8")
        return final

    @staticmethod
    def _gray(rgb):
        import cv2
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    @staticmethod
    def _correct_pinhole_principal_point(image, spec):
        import cv2

        array = np.asarray(image)
        height, width = array.shape[:2]
        fx, fy, cx, cy = (float(value) for value in spec["intrinsics"])
        pixel_y, pixel_x = np.indices((height, width), dtype=np.float32)
        map_x = pixel_x + (width * 0.5 - cx)
        map_y = pixel_y + (height * 0.5 - cy)
        return np.ascontiguousarray(cv2.remap(
            array, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ))

    @staticmethod
    def _pinhole_range_to_z(distance, spec):
        fx, fy, cx, cy = (float(value) for value in spec["intrinsics"])
        height, width = distance.shape
        pixel_y, pixel_x = np.indices((height, width), dtype=np.float32)
        ray_norm = np.sqrt(
            ((pixel_x - cx) / fx) ** 2
            + ((pixel_y - cy) / fy) ** 2
            + 1.0
        )
        depth = np.asarray(distance, dtype=np.float32) / ray_norm
        depth[~np.isfinite(distance)] = np.nan
        return np.ascontiguousarray(depth.astype(np.float32))


    @staticmethod
    def _rgb(sensor):
        camera = getattr(sensor, "_camera", sensor)
        for name in ("get_rgba", "get_rgb", "get_rgb_image"):
            method = getattr(camera, name, None)
            if method is None:
                continue
            image = method()
            if image is not None and np.asarray(image).size:
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] < 3:
                    raise ValueError(f"{name} returned invalid shape {array.shape}")
                return np.ascontiguousarray(np.clip(array[:, :, :3], 0, 255).astype(np.uint8))
        raise RuntimeError("Isaac camera has no readable RGB/RGBA frame")



    @staticmethod
    def _range_to_camera(sensor):
        camera = getattr(sensor, "_camera", sensor)
        annotators = getattr(camera, "_custom_annotators", {})
        annotator = annotators.get("distance_to_camera")
        if annotator is None:
            raise RuntimeError("distance_to_camera annotator is not attached")
        data = annotator.get_data(device="cpu")
        array = np.asarray(data, dtype=np.float32).squeeze()
        if array.ndim != 2 or array.size == 0:
            raise RuntimeError(f"invalid distance_to_camera shape {array.shape}")
        return np.ascontiguousarray(array)


    @staticmethod
    def _save_png(path, rgb):
        from PIL import Image
        Image.fromarray(rgb, "RGB").save(path, format="PNG")

    @staticmethod
    def _save_gray_png(path, gray):
        from PIL import Image
        Image.fromarray(np.asarray(gray, dtype=np.uint8), "L").save(path, format="PNG")

    @staticmethod
    def _save_mask(path, mask):
        from PIL import Image
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, "L").save(path, format="PNG")
