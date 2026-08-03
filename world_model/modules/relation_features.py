"""Observation-only explicit physical relation features."""

from __future__ import annotations

import torch


def ego_explicit_relation(ego_state: torch.Tensor, extent=(0.8, 0.8, 0.35)) -> torch.Tensor:
    out = ego_state.new_zeros((*ego_state.shape[:-1], 1, 10))
    out[..., 0, 3:6] = ego_state[..., 3:6]
    out[..., 0, 6:9] = ego_state.new_tensor(extent)
    out[..., 0, 9] = 1.0
    return out


def env_explicit_relation(centers: torch.Tensor, extents: torch.Tensor,
                          occupancy: torch.Tensor) -> torch.Tensor:
    out = centers.new_zeros((*centers.shape[:-1], 10))
    out[..., :3] = centers
    out[..., 6:9] = extents
    out[..., 9] = occupancy.to(out)
    return out


def human_explicit_relation(skeleton: torch.Tensor, human_mask: torch.Tensor,
                            joint_mask: torch.Tensor) -> torch.Tensor:
    """Build pelvis/root motion, body extent, and confidence for each person."""
    valid = joint_mask.bool() & human_mask.bool()[..., None]
    xyz = skeleton[..., :3]
    vel = skeleton[..., 3:6]
    conf = skeleton[..., 6] if skeleton.shape[-1] > 6 else valid.to(skeleton.dtype)
    # Prefer COCO hips; fall back to a masked joint mean if either hip is absent.
    hips_valid = valid[..., 11] & valid[..., 12] if skeleton.shape[-2] > 12 else torch.zeros_like(human_mask)
    hip_pos = 0.5 * (xyz[..., 11, :] + xyz[..., 12, :]) if skeleton.shape[-2] > 12 else xyz[..., 0, :]
    hip_vel = 0.5 * (vel[..., 11, :] + vel[..., 12, :]) if skeleton.shape[-2] > 12 else vel[..., 0, :]
    weight = valid.to(xyz.dtype)[..., None]
    denom = weight.sum(dim=-2).clamp_min(1.0)
    mean_pos = (xyz * weight).sum(dim=-2) / denom
    mean_vel = (vel * weight).sum(dim=-2) / denom
    root = torch.where(hips_valid[..., None], hip_pos, mean_pos)
    root_vel = torch.where(hips_valid[..., None], hip_vel, mean_vel)
    inf = torch.full_like(xyz, torch.inf)
    low = torch.where(valid[..., None], xyz, inf).amin(dim=-2)
    high = torch.where(valid[..., None], xyz, -inf).amax(dim=-2)
    extent = torch.where(valid.any(-1)[..., None], high - low, torch.zeros_like(low))
    confidence = (conf * valid.to(conf.dtype)).sum(-1) / valid.sum(-1).clamp_min(1).to(conf.dtype)
    out = skeleton.new_zeros((*human_mask.shape, 10))
    out[..., :3], out[..., 3:6], out[..., 6:9], out[..., 9] = root, root_vel, extent, confidence
    return out.masked_fill(~human_mask.bool()[..., None], 0.0)
