import torch
import torch.nn as nn

from modules.skeleton_topology import COCO12_BODY_JOINT_COUNT, skeleton_edges


def build_skeleton_adjacency(num_joints=COCO12_BODY_JOINT_COUNT):
    A = torch.zeros(num_joints, num_joints)

    for i, j in skeleton_edges(num_joints):
        A[i, j] = 1.0
        A[j, i] = 1.0

    # self connection
    A += torch.eye(num_joints)

    # symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class GraphTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0,
                 num_joints=COCO12_BODY_JOINT_COUNT):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.temporal_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            bias=False,
        )

        # GroupNorm avoids batch/time-statistic leakage between independently
        # flattened windows from different sequence timesteps.
        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups:
            groups -= 1
        self.bn = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

        A = build_skeleton_adjacency(num_joints)
        self.register_buffer("A", A)

    def forward(self, x):
        """
        x: [B, C, T, V]
        """
        res = self.residual(x)

        x = self.temporal_conv(x)

        # graph propagation over joints
        # x: [B, C, T, V], A: [V, V]
        x = torch.einsum("bctv,vw->bctw", x, self.A)

        x = self.bn(x)
        x = x + res
        x = self.relu(x)
        x = self.dropout(x)

        return x


class STGCNLiteEncoder(nn.Module):
    def __init__(
        self,
        num_joints=COCO12_BODY_JOINT_COUNT,
        in_channels=3,
        hidden_channels=64,
        out_dim=256,
        image_size=None,
        dropout=0.0,
    ):
        """
        image_size: None or (H, W)
        如果输入是像素坐标，建议传 image_size=(H, W)，把 x/y 归一化到 0~1。
        """
        super().__init__()

        self.num_joints = num_joints
        self.in_channels = in_channels
        self.image_size = image_size

        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.blocks = nn.Sequential(
            GraphTemporalBlock(hidden_channels, hidden_channels, kernel_size=3, dropout=dropout, num_joints=num_joints),
            GraphTemporalBlock(hidden_channels, hidden_channels, kernel_size=3, dropout=dropout, num_joints=num_joints),
            GraphTemporalBlock(hidden_channels, hidden_channels * 2, kernel_size=3, dropout=dropout, num_joints=num_joints),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, skeleton):
        """
        skeleton: [B, T, V, C] 或 [T, V, C]
                  C=3, 分别是 x, y, confidence

        return: [B, T, out_dim]
        """
        if skeleton.dim() == 3:
            skeleton = skeleton.unsqueeze(0)

        if skeleton.dim() != 4:
            raise ValueError(f"Expected skeleton shape [B,T,V,C] or [T,V,C], got {skeleton.shape}")

        B, T, V, C = skeleton.shape

        if V != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {V}")

        if C == 2:
            conf = torch.ones_like(skeleton[..., :1])
            skeleton = torch.cat([skeleton, conf], dim=-1)
            C = 3

        if C != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {C}")

        x = skeleton.float()

        # 像素坐标归一化，但不做中心化。
        # 对 UAV 任务来说，人体在画面中的绝对位置很重要，所以不要轻易减去人体中心。
        if self.image_size is not None:
            H, W = self.image_size
            x = x.clone()
            x[..., 0] = x[..., 0] / float(W)
            x[..., 1] = x[..., 1] / float(H)

        # [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.input_proj(x)
        x = self.blocks(x)

        # joint pooling: [B, C, T, V] -> [B, C, T]
        x = x.mean(dim=-1)

        # [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1).contiguous()

        x = self.out_proj(x)

        return x
