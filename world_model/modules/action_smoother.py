"""Stateful action filtering shared by collection and latent imagination."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ActionSmootherConfig:
    """Deterministic explicit-state policy-to-command transformation."""

    time_step_s: float = 0.1
    time_constant_s: float = 0.35
    slew_rate_per_s: tuple[float, ...] = (1.2, 1.0, 0.45)
    parameterization: str = "legacy_absolute_target"


LEGACY_ABSOLUTE_TARGET = "legacy_absolute_target"
REACHABLE_INTERVAL_FRACTION = "reachable_interval_fraction"
AFFINE_REACHABLE_INTERVAL = "affine_reachable_interval"
SOFT_ABSOLUTE_TARGET = "soft_absolute_target"


class StatefulActionSmoother(nn.Module):
    """Map an Actor output and previous applied target to the next target.

    The module has no trainable parameters.  Its sole dynamic state is the
    previous smoothed policy action, which callers must carry explicitly.  It
    can therefore be used identically by the online policy and Dreamer
    imagination without hiding a non-Markov actuator state.

    ``soft_absolute_target`` is the pure-Dreamer production parameterization.
    The Actor output keeps the ordinary continuous-control meaning of an
    absolute normalized velocity/yaw-rate target.  A smooth rational limiter
    applies the configured low-pass response and slew bound while retaining a
    strictly positive Actor-to-command Jacobian at exact command bounds.  In
    particular, a zero-mean stochastic Actor mean-reverts an existing command
    toward zero instead of integrating its samples into a random walk.

    ``affine_reachable_interval`` is retained for historical checkpoints.  It
    maps raw ``[-1,1]`` onto the complete command interval reachable from the
    previous action.  Raw zero therefore holds the previous command and random
    Actor samples accumulate as command increments; it must not be used by the
    current velocity-target policy.

    ``reachable_interval_fraction`` is retained for historical checkpoints.
    Its raw zero holds the previous command, but its selected-side Jacobian is
    zero at an exact command bound and must not be used by the current pure
    Dreamer objective.
    """

    def __init__(self, config: ActionSmootherConfig) -> None:
        super().__init__()
        dt_s = float(config.time_step_s)
        tau_s = float(config.time_constant_s)
        slew = tuple(float(value) for value in config.slew_rate_per_s)
        parameterization = str(config.parameterization)
        if dt_s <= 0.0 or tau_s <= 0.0:
            raise ValueError("action smoother time constants must be positive")
        if not slew or any(value <= 0.0 for value in slew):
            raise ValueError("action smoother slew rates must be positive")
        if parameterization not in {
            LEGACY_ABSOLUTE_TARGET,
            REACHABLE_INTERVAL_FRACTION,
            AFFINE_REACHABLE_INTERVAL,
            SOFT_ABSOLUTE_TARGET,
        }:
            raise ValueError(
                "unsupported action smoother parameterization: "
                f"{parameterization!r}")
        self.parameterization = parameterization
        self.register_buffer(
            "filter_alpha",
            torch.tensor(1.0 - math.exp(-dt_s / tau_s), dtype=torch.float32),
        )
        self.register_buffer(
            "maximum_step_delta",
            torch.tensor(slew, dtype=torch.float32) * dt_s,
        )

    @property
    def action_dim(self) -> int:
        return int(self.maximum_step_delta.numel())

    def forward(
        self,
        raw_target: torch.Tensor,
        previous_smoothed: torch.Tensor,
    ) -> torch.Tensor:
        if raw_target.shape != previous_smoothed.shape:
            raise ValueError(
                "raw and previous smoothed actions must have identical shapes")
        if raw_target.shape[-1] != self.action_dim:
            raise ValueError(
                "action smoother input does not match configured dimension")
        raw = raw_target.float().clamp(-1.0, 1.0)
        previous = previous_smoothed.float().clamp(-1.0, 1.0)
        maximum_delta = self.maximum_step_delta.to(
            device=raw.device, dtype=raw.dtype)
        if self.parameterization == SOFT_ABSOLUTE_TARGET:
            filtered_delta = self.filter_alpha.to(raw) * (raw - previous)
            bounded_delta = filtered_delta / (
                1.0 + filtered_delta.abs() / maximum_delta)
            # The rational limiter has the sign of (raw - previous) and a
            # magnitude no greater than that error, so the result is already
            # inside both [-1,1] and the segment from previous to raw.  Clamp
            # only protects against floating-point roundoff at exact bounds.
            return (previous + bounded_delta).clamp(-1.0, 1.0)
        if self.parameterization == AFFINE_REACHABLE_INTERVAL:
            lower = torch.maximum(-torch.ones_like(previous),
                                  previous - maximum_delta)
            upper = torch.minimum(torch.ones_like(previous),
                                  previous + maximum_delta)
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            return midpoint + half_width * raw
        if self.parameterization == REACHABLE_INTERVAL_FRACTION:
            positive_cap = torch.minimum(maximum_delta, 1.0 - previous)
            negative_cap = torch.minimum(maximum_delta, 1.0 + previous)
            bounded_delta = torch.where(
                raw >= 0.0, raw * positive_cap, raw * negative_cap)
            return previous + bounded_delta
        filtered_delta = self.filter_alpha * (raw - previous)
        bounded_delta = torch.maximum(
            torch.minimum(filtered_delta, maximum_delta), -maximum_delta)
        return (previous + bounded_delta).clamp(-1.0, 1.0)

    def local_action_scale(
        self,
        raw_target: torch.Tensor,
        previous_smoothed: torch.Tensor,
    ) -> torch.Tensor:
        """Return the diagonal local command Jacobian for diagnostics."""
        if raw_target.shape != previous_smoothed.shape:
            raise ValueError(
                "raw and previous smoothed actions must have identical shapes")
        raw = raw_target.float().clamp(-1.0, 1.0)
        previous = previous_smoothed.float().clamp(-1.0, 1.0)
        maximum_delta = self.maximum_step_delta.to(
            device=raw.device, dtype=raw.dtype)
        if self.parameterization == SOFT_ABSOLUTE_TARGET:
            filtered_delta = self.filter_alpha.to(raw) * (raw - previous)
            ratio = filtered_delta.abs() / maximum_delta
            return self.filter_alpha.to(raw) / (1.0 + ratio).square()
        if self.parameterization == AFFINE_REACHABLE_INTERVAL:
            lower = torch.maximum(-torch.ones_like(previous),
                                  previous - maximum_delta)
            upper = torch.minimum(torch.ones_like(previous),
                                  previous + maximum_delta)
            return 0.5 * (upper - lower)
        if self.parameterization == REACHABLE_INTERVAL_FRACTION:
            positive_cap = torch.minimum(maximum_delta, 1.0 - previous)
            negative_cap = torch.minimum(maximum_delta, 1.0 + previous)
            return torch.where(raw >= 0.0, positive_cap, negative_cap)
        filtered_delta = self.filter_alpha.to(raw) * (raw - previous)
        active = filtered_delta.abs() < maximum_delta
        unclamped = previous + filtered_delta
        active &= unclamped.abs() < 1.0
        return active.to(raw.dtype) * self.filter_alpha.to(raw)

    def raw_target_for_smoothed(
        self,
        desired_smoothed: torch.Tensor,
        previous_smoothed: torch.Tensor,
        *,
        atol: float = 2.0e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert one filter step and report which rows are exactly reachable.

        Replay stores the action that reached the environment, whereas an
        imagined policy override is a *raw* Actor target.  Treating the former
        as the latter applies this filter twice.  The slew limit and tanh
        bounds also mean that not every historical transition is reachable by
        the current policy dynamics, so callers must consume the returned
        validity mask instead of silently projecting such transitions.
        """
        if desired_smoothed.shape != previous_smoothed.shape:
            raise ValueError(
                "desired and previous smoothed actions must have identical "
                "shapes")
        if desired_smoothed.shape[-1] != self.action_dim:
            raise ValueError(
                "action smoother inverse does not match configured dimension")
        if not math.isfinite(float(atol)) or float(atol) < 0.0:
            raise ValueError("action smoother inverse tolerance must be finite")
        desired = desired_smoothed.float()
        previous = previous_smoothed.float()
        finite = torch.isfinite(desired).all(-1, keepdim=True)
        finite &= torch.isfinite(previous).all(-1, keepdim=True)
        in_range = desired.abs().le(1.0 + float(atol)).all(-1, keepdim=True)
        in_range &= previous.abs().le(1.0 + float(atol)).all(
            -1, keepdim=True)
        desired = desired.clamp(-1.0, 1.0)
        previous = previous.clamp(-1.0, 1.0)
        if self.parameterization == SOFT_ABSOLUTE_TARGET:
            maximum_delta = self.maximum_step_delta.to(
                device=desired.device, dtype=desired.dtype)
            delta = desired - previous
            # y = z / (1 + |z| / m), z = alpha * (raw - previous).
            # Its exact inverse exists only for |y| < m; reconstruction below
            # owns the final reachability decision after raw is clamped.
            denominator = (1.0 - delta.abs() / maximum_delta).clamp_min(
                1.0e-12)
            filtered_delta = delta / denominator
            raw = (
                previous
                + filtered_delta / self.filter_alpha.to(
                    device=desired.device, dtype=desired.dtype)
            ).clamp(-1.0, 1.0)
        elif self.parameterization == AFFINE_REACHABLE_INTERVAL:
            maximum_delta = self.maximum_step_delta.to(
                device=desired.device, dtype=desired.dtype)
            lower = torch.maximum(-torch.ones_like(previous),
                                  previous - maximum_delta)
            upper = torch.minimum(torch.ones_like(previous),
                                  previous + maximum_delta)
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            raw = ((desired - midpoint) / half_width.clamp_min(
                1.0e-12)).clamp(-1.0, 1.0)
        elif self.parameterization == REACHABLE_INTERVAL_FRACTION:
            maximum_delta = self.maximum_step_delta.to(
                device=desired.device, dtype=desired.dtype)
            delta = desired - previous
            positive_cap = torch.minimum(maximum_delta, 1.0 - previous)
            negative_cap = torch.minimum(maximum_delta, 1.0 + previous)
            selected_cap = torch.where(
                delta >= 0.0, positive_cap, negative_cap)
            raw = torch.where(
                selected_cap > 0.0,
                delta / selected_cap.clamp_min(1.0e-12),
                torch.zeros_like(delta),
            ).clamp(-1.0, 1.0)
        else:
            raw = (
                previous + (desired - previous) / self.filter_alpha.to(
                    device=desired.device, dtype=desired.dtype)
            ).clamp(-1.0, 1.0)
        reconstructed = self(raw, previous)
        reachable = reconstructed.sub(desired).abs().le(
            float(atol)).all(-1, keepdim=True)
        return raw, finite & in_range & reachable
