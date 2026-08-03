"""Dependency-light inference wrapper for the official NavRL checkpoint.

The upstream quick demo depends on TorchRL/TensorDict.  Isaac Sim already ships
PyTorch, so reproducing the same feed-forward modules directly keeps this path
isolated from Isaac's Python environment while loading the unmodified official
state_dict.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _NavRLInferenceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lidar = nn.Sequential(
            nn.Conv2d(1, 4, (5, 3), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(4, 16, (5, 3), stride=(2, 1), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(16, 16, (5, 3), stride=(2, 2), padding=(2, 1)), nn.ELU(),
            nn.Flatten(), nn.Linear(288, 128), nn.LayerNorm(128),
        )
        self.dynamic = nn.Sequential(
            nn.Flatten(), nn.Linear(50, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(200, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
        )
        self.alpha = nn.Linear(256, 3)
        self.beta = nn.Linear(256, 3)

    def distribution_parameters(self, state, lidar, dynamic):
        # TorchRL CatTensors sorts the official keys by default, producing:
        # _cnn_feature, _dynamic_obstacle_feature, observation/state.
        # The checkpoint's first fusion layer was trained with exactly this
        # ordering; changing it silently yields valid-shaped but wrong actions.
        feature = torch.cat((self.lidar(lidar), self.dynamic(dynamic), state), -1)
        feature = self.fusion(feature)
        alpha = 1.0 + nn.functional.softplus(self.alpha(feature)) + 1e-6
        beta = 1.0 + nn.functional.softplus(self.beta(feature)) + 1e-6
        return alpha, beta

    def forward(self, state, lidar, dynamic):
        alpha, beta = self.distribution_parameters(state, lidar, dynamic)
        # ExplorationType.MEAN for a Beta distribution, as in the official demo.
        return alpha / (alpha + beta)


class NavRLPolicy:
    LIDAR_RANGE_M = 4.0
    HORIZONTAL_BEAMS = 36
    VERTICAL_ANGLES_DEG = (-10.0, 0.0, 10.0, 20.0)
    MAX_DYNAMIC_OBSTACLES = 5
    # The checkpoint was trained with 2 m/s, while the upstream ROS flight
    # runner deliberately deploys it with vel_limit=1.0 m/s. Match the flight
    # configuration here; 2 m/s left too little stopping distance indoors.
    ACTION_LIMIT_MPS = 1.0

    _KEY_MAP = {
        "feature_extractor.module.0.module.0": "lidar.0",
        "feature_extractor.module.0.module.2": "lidar.2",
        "feature_extractor.module.0.module.4": "lidar.4",
        "feature_extractor.module.0.module.7": "lidar.7",
        "feature_extractor.module.0.module.8": "lidar.8",
        "feature_extractor.module.1.module.1.0": "dynamic.1",
        "feature_extractor.module.1.module.1.2": "dynamic.3",
        "feature_extractor.module.1.module.1.3": "dynamic.4",
        "feature_extractor.module.1.module.1.5": "dynamic.6",
        "feature_extractor.module.3.module.0": "fusion.0",
        "feature_extractor.module.3.module.2": "fusion.2",
        "feature_extractor.module.3.module.3": "fusion.3",
        "feature_extractor.module.3.module.5": "fusion.5",
        "actor.module.0.module.alpha_layer": "alpha",
        "actor.module.0.module.beta_layer": "beta",
    }

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = _NavRLInferenceNet().to(self.device)
        upstream = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        converted = {}
        for old_prefix, new_prefix in self._KEY_MAP.items():
            for suffix in ("weight", "bias"):
                old_key = f"{old_prefix}.{suffix}"
                if old_key in upstream:
                    converted[f"{new_prefix}.{suffix}"] = upstream[old_key]
        missing, unexpected = self.net.load_state_dict(converted, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"NavRL checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.net.eval()
        self.last_local_velocity = np.zeros(3, dtype=float)
        self.last_normalized_action = np.full(3, 0.5, dtype=float)
        self.last_alpha = np.zeros(3, dtype=float)
        self.last_beta = np.zeros(3, dtype=float)

    @torch.inference_mode()
    def infer(self, state8: np.ndarray, lidar36x4: np.ndarray,
              dynamic5x10: np.ndarray, goal_direction_world: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state8, dtype=torch.float32, device=self.device)[None]
        lidar = torch.as_tensor(lidar36x4, dtype=torch.float32, device=self.device)[None, None]
        dynamic = torch.as_tensor(dynamic5x10, dtype=torch.float32, device=self.device)[None, None]
        alpha, beta = self.net.distribution_parameters(state, lidar, dynamic)
        normalized = (alpha / (alpha + beta))[0]
        local_velocity = (2.0 * normalized - 1.0) * self.ACTION_LIMIT_MPS
        local = local_velocity.detach().cpu().numpy()
        self.last_local_velocity = local.astype(float, copy=True)
        self.last_normalized_action = normalized.detach().cpu().numpy().astype(float, copy=True)
        self.last_alpha = alpha[0].detach().cpu().numpy().astype(float, copy=True)
        self.last_beta = beta[0].detach().cpu().numpy().astype(float, copy=True)
        direction = np.asarray(goal_direction_world, dtype=float)
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-8:
            return np.zeros(3, dtype=float)
        x_axis = np.array([direction[0] / norm, direction[1] / norm, 0.0])
        y_axis = np.array([-x_axis[1], x_axis[0], 0.0])
        return x_axis * float(local[0]) + y_axis * float(local[1]) + np.array([0.0, 0.0, float(local[2])])
