#!/usr/bin/env python3
"""Inspect OmniNxt USD assets with the USD runtime bundled with Isaac Sim."""

import argparse
import json
import os
from pathlib import Path

from isaacsim import SimulationApp


def _value(value):
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except TypeError:
        return str(value)


def _inspect_asset(path):
    from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")

    default_prim = stage.GetDefaultPrim()
    roots = list(stage.GetPseudoRoot().GetChildren())
    target = default_prim if default_prim.IsValid() else (roots[0] if roots else None)
    if target is None:
        raise RuntimeError(f"USD stage has no root prim: {path}")

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), purposes, useExtentsHint=True
    ).ComputeWorldBound(target).ComputeAlignedBox()
    bbox_min = bbox.GetMin()
    bbox_max = bbox.GetMax()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

    mesh_paths = []
    face_count = 0
    triangle_count = 0
    bound_material_mesh_count = 0
    applied_schemas = set()
    asset_references = []
    missing_assets = []
    resolver = Ar.GetResolver()

    for prim in stage.Traverse():
        applied_schemas.update(str(schema) for schema in prim.GetAppliedSchemas())
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(str(prim.GetPath()))
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []
            face_count += len(counts)
            triangle_count += sum(max(0, int(count) - 2) for count in counts)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                bound_material_mesh_count += 1

        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            asset = attribute.Get()
            if not asset or not asset.path:
                continue
            resolved = asset.resolvedPath or str(resolver.Resolve(asset.path))
            item = {
                "prim": str(prim.GetPath()),
                "attribute": attribute.GetName(),
                "asset": asset.path,
                "resolved": resolved,
            }
            asset_references.append(item)
            if not resolved or not os.path.exists(resolved):
                missing_assets.append(item)

    xformable = UsdGeom.Xformable(target)
    root_ops = []
    if xformable:
        for op in xformable.GetOrderedXformOps():
            root_ops.append(
                {
                    "name": str(op.GetOpName()),
                    "type": str(op.GetOpType()),
                    "value": _value(op.Get()),
                }
            )

    size_stage = [float(bbox_max[i] - bbox_min[i]) for i in range(3)]
    center_stage = [float((bbox_max[i] + bbox_min[i]) * 0.5) for i in range(3)]
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "meters_per_unit": meters_per_unit,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim.IsValid() else None,
        "root_prims": [str(prim.GetPath()) for prim in roots],
        "root_xform_ops": root_ops,
        "bbox_min_stage": [float(value) for value in bbox_min],
        "bbox_max_stage": [float(value) for value in bbox_max],
        "bbox_size_m": [value * meters_per_unit for value in size_stage],
        "bbox_center_m": [value * meters_per_unit for value in center_stage],
        "mesh_count": len(mesh_paths),
        "face_count": face_count,
        "triangle_count": triangle_count,
        "bound_material_mesh_count": bound_material_mesh_count,
        "applied_schemas": sorted(applied_schemas),
        "mesh_paths": mesh_paths,
        "asset_references": asset_references,
        "missing_assets": missing_assets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default=Path(__file__).resolve().parent / "usd")
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": True})
    try:
        asset_dir = Path(args.asset_dir).resolve()
        names = ("Omininxt_body.usdc", "fl.usdc", "fr.usdc", "rl.usdc", "rr.usdc")
        report = [_inspect_asset(asset_dir / name) for name in names]
        print("OMNINXT_ASSET_REPORT_BEGIN")
        print(json.dumps(report, indent=2, ensure_ascii=True))
        print("OMNINXT_ASSET_REPORT_END")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
