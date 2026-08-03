from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PointPillarsConfig:
    """Minimal PointPillars front-end config for xyz-only point clouds."""

    voxel_size: Tuple[float, float] = (1.0, 1.0)
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        -200.0,
        -200.0,
        -10.0,
        200.0,
        200.0,
        40.0,
    )
    max_points_per_pillar: int = 32
    max_pillars: int = 12000
    pfn_out_channels: int = 64

    # 1  = batch size
    # 64 = 每个 BEV cell 的特征维度
    # 400 = y 方向格子数
    # 400 = x 方向格子数

    @property
    def grid_size(self) -> Tuple[int, int]:
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        voxel_x, voxel_y = self.voxel_size
        width = int(round((x_max - x_min) / voxel_x))
        height = int(round((y_max - y_min) / voxel_y))
        return height, width


class PillarFeatureNet(nn.Module):
    """PFN layer used by the original PointPillars encoder, simplified to one layer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode pillars.

        Args:
            features: Tensor shaped [num_pillars, max_points, in_channels].
            mask: Bool tensor shaped [num_pillars, max_points], true for real points.

        Returns:
            Tensor shaped [num_pillars, out_channels].
        """

        num_pillars, max_points, _ = features.shape
        x = self.linear(features)
        x = self.norm(x.reshape(num_pillars * max_points, -1))#BatchNorm1d 要吃二维输入，所以先把前两维合并：-1是自我推断的意思
        x = self.relu(x).reshape(num_pillars, max_points, -1)
        x = x.masked_fill(~mask[..., None], float('-inf'))  # 用 -inf 填充 padding，确保 max pooling 只取真实的点
        return x.max(dim=1).values  #点内最大池化


class SimplePointPillarsBEV(nn.Module):
    """Pure PyTorch .npy/points -> pillar features -> BEV feature map."""

    def __init__(self, config: Optional[PointPillarsConfig] = None, point_dim: int = 3) -> None:
        super().__init__()
        self.config = config or PointPillarsConfig()
        if point_dim < 3:
            raise ValueError("point_dim must be at least 3 for xyz coordinates.")

        # xyz/intensity/etc + cluster xyz offset + pillar-center xy offset
        pfn_in_channels = point_dim + 3 + 2
        self.pfn = PillarFeatureNet(pfn_in_channels, self.config.pfn_out_channels)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        pillars, mask, coords = pillarize(points, self.config)
        if pillars.shape[0] == 0:
            height, width = self.config.grid_size
            return points.new_zeros((1, self.config.pfn_out_channels, height, width))#[batch, channel, height, width]

        pillar_features = self.pfn(pillars, mask)
        return scatter_to_bev(pillar_features, coords, self.config)


        #核心是这句：

        # bev[0, :, coords[:, 0], coords[:, 1]] = pillar_features.t()

        # 意思是：把每个 pillar 的 64 维特征放到它对应的 BEV 格子里。

        # 等价于：

        # for i in range(P):
        #     y = coords[i, 0]
        #     x = coords[i, 1]
        #     bev[0, :, y, x] = pillar_features[i]

        # 为什么要 .t()？

        # 因为左边：

        # bev[0, :, coords[:, 0], coords[:, 1]]

        # 取出来的形状是：

        # [C, P]

        # 而 pillar_features 原本是：

        # [P, C]

        # 所以要转置成：

        # pillar_features.t()  # [C, P]

        # 最后返回：

        # [1, C, H, W]

        # 也就是可以接 CNN / tokenization / Dreamer 的 BEV 特征图。


def make_config_from_points(
    points: np.ndarray,
    voxel_size: Tuple[float, float] = (1.0, 1.0),
    margin: float = 1.0,
    max_points_per_pillar: int = 32,
    max_pillars: int = 12000,
    pfn_out_channels: int = 64,
) -> PointPillarsConfig:
    """Create a frame-local config that covers all points in one npy file."""

    xyz_min = points[:, :3].min(axis=0)
    xyz_max = points[:, :3].max(axis=0)
    point_cloud_range = (
        float(np.floor(xyz_min[0] - margin)),
        float(np.floor(xyz_min[1] - margin)),
        float(np.floor(xyz_min[2] - margin)),
        float(np.ceil(xyz_max[0] + margin)),
        float(np.ceil(xyz_max[1] + margin)),
        float(np.ceil(xyz_max[2] + margin)),
    )
    return PointPillarsConfig(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_points_per_pillar=max_points_per_pillar,
        max_pillars=max_pillars,
        pfn_out_channels=pfn_out_channels,
    )


def pillarize(
    points: torch.Tensor,
    config: PointPillarsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert points to dense per-pillar tensors without custom CUDA ops."""

    device = points.device
    dtype = points.dtype
    x_min, y_min, z_min, x_max, y_max, z_max = config.point_cloud_range
    voxel_x, voxel_y = config.voxel_size
    height, width = config.grid_size

    in_range = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    points = points[in_range]#留下在range内的点
    if points.numel() == 0:
        feature_dim = points.shape[1] + 5
        empty_pillars = points.new_zeros((0, config.max_points_per_pillar, feature_dim))
        empty_mask = torch.zeros((0, config.max_points_per_pillar), dtype=torch.bool, device=device)

        #         mask[0]
        # =
        # [True, True, True, False, False, ...]

        empty_coords = torch.zeros((0, 2), dtype=torch.long, device=device)

        #[有多少个 pillar, 每个 pillar 的坐标维度]

        return empty_pillars, empty_mask, empty_coords

    x_idx = torch.floor((points[:, 0] - x_min) / voxel_x).long().clamp(0, width - 1)
    #计算每个点所在的pillar的x索引，clamp确保索引在[0, width-1]范围内，.long()将索引转换为整数类型
    y_idx = torch.floor((points[:, 1] - y_min) / voxel_y).long().clamp(0, height - 1)
    point_coords = torch.stack((y_idx, x_idx), dim=1)
    # point_coords的shape是[N, 2]，每行包含一个点所在pillar的(y_idx, x_idx)坐标
    unique_coords, inverse = torch.unique(point_coords, dim=0, return_inverse=True)

    #     假设：

    # point_coords = tensor([
    #     [3, 10],
    #     [3, 10],
    #     [5, 12],
    #     [5, 12],
    #     [8, 20],
    # ])

    # 那么：

    # unique_coords = tensor([
    #     [3, 10],
    #     [5, 12],
    #     [8, 20],
    # ])

    # 意思是：当前点云里一共有 3 个非空 pillar。

    # 而：

    # inverse = tensor([0, 0, 1, 1, 2])

    # 意思是：

    # 第 0 个点属于 unique_coords[0]，也就是 [3,10]
    # 第 1 个点属于 unique_coords[0]，也就是 [3,10]
    # 第 2 个点属于 unique_coords[1]，也就是 [5,12]
    # 第 3 个点属于 unique_coords[1]，也就是 [5,12]
    # 第 4 个点属于 unique_coords[2]，也就是 [8,20]

    # 所以这句的作用就是：

    # point_coords: 每个点在哪个 pillar
    # unique_coords: 实际有哪些 pillar 被点占用了
    # inverse: 每个点对应 unique_coords 里的哪一个 pillar 编号


    num_pillars = min(unique_coords.shape[0], config.max_pillars)#简易化截断 先后顺序
    keep_pillar = inverse < num_pillars
    #inverse 表示每个点属于第几个 pillar。这里是在判断：这个点所属的 pillar 编号是否小于 num_pillars。如果小于，就保留；否则丢掉。
    points = points[keep_pillar]
    inverse = inverse[keep_pillar]
    coords = unique_coords[:num_pillars]

    counts = torch.bincount(inverse, minlength=num_pillars)
    #     统计每个 pillar 里有多少个点。比如：

    # inverse = [0, 0, 1, 1, 1, 2]
    # counts = [2, 3, 1]
    order = torch.argsort(inverse, stable=True)
    sorted_inverse = inverse[order]
    sorted_points = points[order]

    #inverse = tensor([2, 0, 1, 0, 2, 1])

    # 先给每个元素标上它原来的位置：

    # 位置:      0  1  2  3  4  5
    # inverse:  2  0  1  0  2  1

    # argsort 做的事情是：按照 inverse 的值从小到大排序，但返回的是原来的位置编号。

    # 从小到大排：

    # 值为 0 的位置: 1, 3
    # 值为 1 的位置: 2, 5
    # 值为 2 的位置: 0, 4

    # 所以拼起来：

    # order = tensor([1, 3, 2, 5, 0, 4])

    # 然后用这个 order 去取原数组：

    # inverse[order]

    # 就等于：

    # tensor([
    #     inverse[1],
    #     inverse[3],
    #     inverse[2],
    #     inverse[5],
    #     inverse[0],
    #     inverse[4],
    # ])

    # 也就是：

    # tensor([0, 0, 1, 1, 2, 2])

    # 所以 order 本质上是：“为了把 inverse 排序，应该按什么下标顺序去取原数组”。

    start_offsets = torch.cumsum(counts, dim=0) - counts
    point_offsets = torch.arange(sorted_points.shape[0], device=device) - start_offsets[sorted_inverse]
    keep_point = point_offsets < config.max_points_per_pillar

    sorted_points = sorted_points[keep_point]
    sorted_inverse = sorted_inverse[keep_point]
    point_offsets = point_offsets[keep_point]

    max_points = config.max_points_per_pillar
    pillars_xyz = points.new_zeros((num_pillars, max_points, points.shape[1]))
    mask = torch.zeros((num_pillars, max_points), dtype=torch.bool, device=device)
    pillars_xyz[sorted_inverse, point_offsets] = sorted_points
    mask[sorted_inverse, point_offsets] = True


    #     核心是这句：

    # pillars_xyz[sorted_inverse, point_offsets] = sorted_points

    # 意思是把每个点放到：

    # pillars_xyz[pillar编号, 该pillar内部第几个点]

    # 里面。

    # 比如某个点属于 pillar 5，并且是这个 pillar 里的第 2 个点，那它就被放到：

    # pillars_xyz[5, 2] = 这个点的xyz

    # 然后：

    # mask[sorted_inverse, point_offsets] = True

    # 把这些真实点的位置标成 True。剩下没被填的位置还是 False，表示 padding.



    real_counts = mask.sum(dim=1).clamp(min=1).to(dtype)
    xyz_sum = (pillars_xyz[:, :, :3] * mask[..., None]).sum(dim=1)

    #     先看：

    # pillars_xyz[:, :, :3]

    # 意思是：

    # 所有 pillar
    # 所有点
    # 取前3维 xyz

    # 因为有些点可能还有 intensity 等额外特征，所以这里只取 xyz。

    # 结果形状：

    # [P, M, 3]

    # 再看：

    # mask[..., None]

    # 这里：

    # mask.shape = [P, M]

    # ... 表示前面所有维度保持不变。

    # None

    # 是在最后新增一维。

    # 所以：

    # mask[..., None]

    # 等价于：

    # mask[:, :, None]

    # 形状变成：

    # [P, M, 1]

    # 比如：

    # mask =
    # [
    #  [True, True, False],
    #  [True, False, False]
    # ]

    # 变成：

    # [
    #  [[1],[1],[0]],
    #  [[1],[0],[0]]
    # ]
    # ]

    # 然后：

    # pillars_xyz * mask[..., None]

    # PyTorch 会广播：

    # [P,M,3] * [P,M,1]

    # 自动扩成：

    # [P,M,3]

    # 于是：

    # 真实点 × 1
    # padding点 × 0

    # padding 的 xyz 被清零。

    # 最后：

    # .sum(dim=1)

    # 在第 1 维求和，也就是：

    # 沿着 pillar 内点的维度求和

    # 所以：

    # [P, M, 3]
    # → sum(dim=1)
    # → [P, 3]

    # 得到每个 pillar 所有真实点 xyz 的总和：

    # xyz_sum[p]
    # =
    # pillar p 内所有点 xyz 相加



    xyz_mean = xyz_sum / real_counts[:, None]
    cluster_offset = pillars_xyz[:, :, :3] - xyz_mean[:, None, :]

    x_center = x_min + (coords[:, 1].to(dtype) + 0.5) * voxel_x
    y_center = y_min + (coords[:, 0].to(dtype) + 0.5) * voxel_y
    center_offset = torch.stack(
        (pillars_xyz[:, :, 0] - x_center[:, None], pillars_xyz[:, :, 1] - y_center[:, None]),
        dim=-1,
    )

    features = torch.cat((pillars_xyz, cluster_offset, center_offset), dim=-1)#每个点的特征由原始xyz + cluster offset + center offset 组成，特征维度是 point_dim + 3 + 2
    features = features * mask[..., None]
    return features, mask, coords


def scatter_to_bev(
    pillar_features: torch.Tensor,
    coords: torch.Tensor,
    config: PointPillarsConfig,
) -> torch.Tensor:
    """Scatter [P, C] pillar features to [1, C, H, W] BEV pseudo-image."""

    height, width = config.grid_size
    channels = pillar_features.shape[1]
    bev = pillar_features.new_zeros((1, channels, height, width))
    bev[0, :, coords[:, 0], coords[:, 1]] = pillar_features.t()
    return bev


def load_npy_points(path: str | Path) -> np.ndarray:
    points = np.load(path).astype(np.float32, copy=False)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point array with shape [N, >=3], got {points.shape}.")
    return points


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure PyTorch PointPillars front-end demo.")
    parser.add_argument("npy_path", type=Path, help="Input .npy point cloud shaped [N, 3] or [N, C].")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save BEV feature .npy.")
    parser.add_argument("--voxel-size", type=float, nargs=2, default=(1.0, 1.0), metavar=("VX", "VY"))
    parser.add_argument("--range", type=float, nargs=6, default=None, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--auto-range", action="store_true", help="Infer point cloud range from this frame.")
    parser.add_argument("--max-points-per-pillar", type=int, default=32)
    parser.add_argument("--max-pillars", type=int, default=12000)
    parser.add_argument("--pfn-out-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)#随机初始化种子 seed for reproducibility, affects random weight initialization in the model

    np_points = load_npy_points(args.npy_path)
    if args.auto_range:
        config = make_config_from_points(
            np_points,
            voxel_size=tuple(args.voxel_size),
            max_points_per_pillar=args.max_points_per_pillar,
            max_pillars=args.max_pillars,
            pfn_out_channels=args.pfn_out_channels,
        )
    else:
        config = PointPillarsConfig(
            voxel_size=tuple(args.voxel_size),
            point_cloud_range=tuple(args.range) if args.range else PointPillarsConfig().point_cloud_range,
            max_points_per_pillar=args.max_points_per_pillar,
            max_pillars=args.max_pillars,
            pfn_out_channels=args.pfn_out_channels,
        )

    model = SimplePointPillarsBEV(config, point_dim=np_points.shape[1]).eval() #.eval() to disable BatchNorm running stats updates
    points = torch.from_numpy(np_points)
    with torch.no_grad():
        bev = model(points)

    print(f"input points: {tuple(points.shape)}")
    print(f"grid HxW: {config.grid_size}")
    print(f"bev feature map: {tuple(bev.shape)}")
    print(f"nonzero bev cells: {(bev.abs().sum(dim=1) > 0).sum().item()}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, bev.squeeze(0).cpu().numpy())
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
