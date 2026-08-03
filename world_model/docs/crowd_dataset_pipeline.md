# Isaac crowd offline dataset pipeline

这套代码把原始 `record_xxx/frames` 作为只读数据源，不会修改已经录好的数据集。

## 1. 原始数据保持不动

原始目录形如：

```text
/media/b403/张靖凯/zyw/
  record_20260625_150258/
    metadata.json
    summary.json
    frames/
      frame_000000.json
      frame_000000_camera.jpg
      frame_000000_lidar.npy
      ...
```

Dataset adapter 只读取这些文件。过滤 episode、动作归一化、序列组织都发生在运行时。

## 2. 构建可用 episode index

```bash
python scripts/build_crowd_dataset_index.py \
  --data_root "/media/b403/张靖凯/zyw" \
  --out crowd_dataset_index.json \
  --min_frames 30 \
  --exclude_termination stuck_timeout
```

这个命令会保存一个派生索引 `crowd_dataset_index.json`，不会写入原始 `record_xxx`。

默认过滤：

- 少于 30 帧的 episode；
- `termination_reason == stuck_timeout`；
- 缺少 json/camera/lidar 三件套的 episode；
- 帧号不连续的 episode。

碰撞 episode 默认保留，因为世界模型和奖励/continue 模型需要看到失败状态。

## 3. Dreamer/RSSM 序列格式

`datasets.isaac_crowd.IsaacCrowdSequenceDataset` 会把散落的 frame 文件整理成固定长度序列：

```python
{
    "ego_state":    [T, 17],
    "action":       [T, 4],
    "reward":       [T, 1],
    "is_first":     [T, 1],
    "is_last":      [T, 1],
    "is_terminal":  [T, 1],
    "image_path":   list[str],
    "lidar_path":   list[str],
    "pose_path":    list[str],  # 如果提供 pose_cache_root
}
```

其中 `ego_state` 默认是 17 维 episode-local / goal-relative 状态：

```text
x_local, y_local, z_rel,
vx_local, vy_local, vz,
ax_local, ay_local, az,
roll, pitch, yaw_rel,
goal_dx_body, goal_dy_body, goal_dz,
goal_distance,
heading_error
```

这里的 `x_local/y_local/z_rel` 不是 Isaac Sim 世界坐标，而是相对 episode 第一帧无人机位置的局部坐标；`vx_local/vy_local` 和 `ax_local/ay_local` 也被旋转到 episode 初始 yaw 坐标系。`goal_dx_body/goal_dy_body` 是当前无人机机体系下的目标相对向量，`heading_error` 是当前 yaw 到目标方向的角度误差。

## 4. 动作字段选择

训练世界模型时使用：

```text
normalize(action.applied)
```

而不是直接用原始 JSON 里的：

```text
action.normalized
```

原因是 `action.normalized` 对应自动控制器的 `requested` 指令；真正经过底层限斜率后作用到无人机的是 `action.applied`。

归一化方式：

```python
[
    applied["vx_body_mps"] / limits["vx_body_mps"],
    applied["vy_body_mps"] / limits["vy_body_mps"],
    applied["vz_world_mps"] / limits["vz_world_mps"],
    applied["yaw_rate_rps"] / limits["yaw_rate_rps"],
]
```

## 5. 多人 RTMPose 缓存

RTMPose 不建议训练时实时跑，应该提前生成派生缓存：

```bash
python scripts/cache_multiperson_pose.py \
  --data_root "/media/b403/张靖凯/zyw" \
  --index_path crowd_dataset_index.json \
  --out_root "/media/b403/张靖凯/zyw_pose_cache" \
  --device cpu \
  --mode balanced \
  --skip_existing
```

输出形如：

```text
/media/b403/张靖凯/zyw_pose_cache/
  record_20260625_150258/
    frame_000000_pose.npz
    frame_000001_pose.npz
    pose_cache_metadata.json
```

每个 `.npz` 包含：

```text
keypoints_xyc: [num_people, 17, 3]
scores:        [num_people, 17]
person_scores: [num_people]
valid_mask:    [num_people]
track_ids:     [num_people]
bboxes_xyxy:   [num_people, 4]
centers_xy:    [num_people, 2]
```

`track_ids` 是一个轻量的逐帧中心点最近邻 tracker，适合给后续 ST-GCN 构造人群时间窗口。它不是复杂多目标跟踪器；如果后续遮挡很严重，可以替换成更强的 tracker，但缓存格式不用变。

## 6. 快速检查

```bash
python scripts/smoke_test_crowd_dataset.py \
  --data_root "/media/b403/张靖凯/zyw" \
  --index_path crowd_dataset_index.json \
  --pose_cache_root "/media/b403/张靖凯/zyw_pose_cache" \
  --sequence_length 64 \
  --stride 64
```

如果要实际加载小尺寸图像和定长点云采样：

```bash
python scripts/smoke_test_crowd_dataset.py \
  --data_root "/media/b403/张靖凯/zyw" \
  --index_path crowd_dataset_index.json \
  --sequence_length 16 \
  --load_images \
  --image_width 160 \
  --image_height 90 \
  --load_lidar \
  --max_lidar_points 4096
```

## 7. 后续接世界模型

推荐数据流：

```text
pose_cache keypoints -> Dataset causal pose windows -> trainable ST-GCN pose tokens
lidar.npy            -> fixed-range ego-centric PointPillars / BEV tokens
ego_state            -> MLP ego token
action               -> RSSM transition input
reward/is_terminal   -> reward/continue heads
```

Dataset 可以直接构造因果骨架窗口：

```python
dataset = IsaacCrowdSequenceDataset(
    index_path="crowd_dataset_index.json",
    pose_cache_root="/media/b403/LENOVO_USB_HDD/zyw_pose_cache",
    sequence_length=64,
    load_pose_windows=True,
    pose_window_size=8,
    max_pose_people=16,
)
```

输出字段：

```text
pose_windows:       [T, max_people, pose_window_size, 17, 3]
pose_window_mask:   [T, max_people, pose_window_size]
pose_token_mask:    [T, max_people]
pose_track_ids:     [T, max_people]
pose_person_scores: [T, max_people]
pose_bboxes_xyxy:   [T, max_people, 4]
pose_image_size_hw: [T, 2]
```

`pose_windows[t, i]` 只包含当前帧第 `i` 个人在 `t-window+1 ... t` 的历史骨架，不包含未来帧；缺失历史用 0 填充，并由 `pose_window_mask` 标记。

真正的可训练 ST-GCN 位于：

```python
from modules import CausalMultiPersonSTGCNEncoder

pose_encoder = CausalMultiPersonSTGCNEncoder(out_dim=256)
out = pose_encoder(
    batch["pose_windows"],
    pose_window_mask=batch["pose_window_mask"],
    pose_token_mask=batch["pose_token_mask"],
    image_size_hw=batch["pose_image_size_hw"],
)
pose_tokens = out["pose_tokens"]          # [B, T, max_people, 256]
pose_mask = out["pose_token_mask"]        # [B, T, max_people]
```

注意：这一步不会离线保存 ST-GCN feature。ST-GCN 在 world-model forward 里动态计算 token，并和世界模型一起反向传播训练。

## 8. 可训练 PointPillars / BEV 编码器

可训练 PointPillars 位于：

```python
from modules import EgoPointPillarsBEVEncoder, EgoPointPillarsConfig
```

默认 BEV 范围是以无人机 / LiDAR 为坐标原点的固定范围：

```text
x: [-16, 16] m   # 横向，覆盖 warehouse 的 +/-7.5m 人群宽度和 citytower 横向偏移
y: [ -8, 40] m   # 后方少量 + 前方主要任务区域
z: [ -2,  8] m   # 低空飞行、行人和主要近地环境结构
```

这个范围不会随每帧点云变化，因此同一个 BEV cell 在所有 episode、两个场景中都表示相同的 ego-centric 空间位置。

示例：

```python
dataset = IsaacCrowdSequenceDataset(
    index_path="crowd_dataset_index_lenovo.json",
    sequence_length=64,
    load_lidar=True,
    max_lidar_points=8192,
)

pointpillars = EgoPointPillarsBEVEncoder(
    EgoPointPillarsConfig(
        voxel_size=(0.5, 0.5),
        point_cloud_range=(-16.0, -8.0, -2.0, 16.0, 40.0, 8.0),
    )
)

out = pointpillars(batch["lidar"], point_mask=batch["lidar_mask"])
bev_feature = out["bev_feature"]        # [B, T, C, Hf, Wf]
bev_tokens = out["bev_tokens"]          # [B, T, Hf*Wf, C]
bev_token_mask = out["bev_token_mask"]  # [B, T, Hf*Wf]
```

默认参数下：

```text
raw BEV grid:     H=96, W=64
backbone output:  H=24, W=16
BEV tokens:       384 tokens/frame
token dim:        128
```

快速检查：

```bash
python scripts/smoke_test_pointpillars.py \
  --data_root "/media/b403/LENOVO_USB_HDD/zyw" \
  --index_path crowd_dataset_index_lenovo.json \
  --sequence_length 4 \
  --batch_size 1 \
  --max_lidar_points 8192
```

## 9. 三路世界模型编码器

三路编码器位于：

```python
from modules import ThreeStreamWorldModelEncoder, ThreeStreamEncoderConfig
```

它不会把所有东西直接揉成一个大 embedding，而是保留三路结构化输出：

```text
human stream:
  pose_windows -> Causal ST-GCN -> pose tokens
  pose tokens + BEV tokens -> cross-attention -> human_tokens
  human_tokens masked pooling -> human_embed

environment stream:
  lidar -> PointPillars -> bev_map + env_tokens
  env_tokens masked pooling -> env_embed

ego stream:
  ego_state -> MLP -> ego_embed
```

概念区分：

```text
human_tokens: [B, T, max_people, D]
  每个行人一个 token，保留“有哪些人、每个人在哪里/姿态如何”的结构。
  后续做人群预测、人与 BEV cross-attention 时应优先用它。

human_embed: [B, T, D]
  对 human_tokens 做 masked pooling 后的一帧人群摘要。
  适合给 RSSM posterior adapter / reward head 这种只需要全局摘要的模块。

bev_map: [B, T, C, H, W]
  保留二维空间布局的 BEV 特征图。
  适合 BEV 重建、occupancy/flow 预测、卷积式环境预测 head。

env_tokens: [B, T, H*W, D]
  把 BEV map flatten 后得到的空间 tokens。
  适合和 human_tokens 做 cross-attention。

env_embed: [B, T, D]
  对 env_tokens 做 masked pooling 后的一帧环境摘要。
  适合给 RSSM adapter / value / reward 等低维模块。
```

所以 `bev_map` 和 `env_embed` 不是重复的：

```text
bev_map   = 保留空间结构，用于预测/重建“哪里有什么”
env_embed = 压缩摘要，用于告诉 RSSM“当前环境整体是什么状态”
```

快速检查：

```bash
python scripts/smoke_test_three_stream_encoder.py \
  --data_root "/media/b403/LENOVO_USB_HDD/zyw" \
  --index_path crowd_dataset_index_lenovo.json \
  --pose_cache_root "/media/b403/LENOVO_USB_HDD/zyw_pose_cache" \
  --sequence_length 4 \
  --batch_size 1 \
  --max_lidar_points 4096 \
  --pose_window_size 4 \
  --max_pose_people 8
```

## 10. 共享 RSSM 的结构化后验

第五步修改的是 RSSM 的 posterior，而不是改成三个独立 RSSM。

原始 Dreamer/RSSM 后验大致是：

```text
deter_t + obs_embed_t -> obs_net -> posterior stochastic logits
```

现在 crowd world model 可以使用结构化后验：

```text
human_embed_t ┐
env_embed_t   ├─ StructuredPosteriorAdapter(deter_t, streams) -> obs_net -> posterior logits
ego_embed_t   ┘
```

也就是说：

```text
prior / imagine:
  仍然只由共享 RSSM latent + action 推进

posterior / observe:
  用 human/env/ego 三路观测证据修正同一个共享 latent
```

对应代码：

```python
from modules import ThreeStreamWorldModelEncoder
import rssm

encoder = ThreeStreamWorldModelEncoder(...)
model = rssm.RSSM(config.rssm, encoder.rssm_embed_size, act_dim=4)

encoded = encoder(batch)
post_stoch, post_deter, post_logit = model.observe(
    encoded,
    batch["action"],
    initial,
    batch["is_first"],
)
```

这里 `encoder.rssm_embed_size` 是：

```python
{
    "human_embed": D,
    "env_embed": D,
    "ego_embed": D,
}
```

RSSM 在初始化时如果收到的是这个 dict，就会自动启用 `StructuredPosteriorAdapter`。如果收到的仍然是普通整数 embed size，就保持原来的 `torch.cat([deter, embed])` 后验路径，所以旧任务和旧训练流程不会被这个改动破坏。

结构化后验内部做了三件事：

```text
1. 每一路 stream 单独投影到 hidden 维度；
2. 每一路都有一个由 deter_t 和该 stream 共同决定的 gate；
3. gated stream evidence 与 deter projection 相加，再经过 residual mixer。
```

这样做的目的不是把三路完全独立，而是让它们以“不同证据来源”的形式共同修正同一个共享 RSSM latent。

快速检查：

```bash
python scripts/smoke_test_structured_rssm.py \
  --data_root "/media/b403/LENOVO_USB_HDD/zyw" \
  --index_path crowd_dataset_index_lenovo.json \
  --pose_cache_root "/media/b403/LENOVO_USB_HDD/zyw_pose_cache" \
  --sequence_length 4 \
  --batch_size 1 \
  --max_lidar_points 1024 \
  --pose_window_size 4 \
  --max_pose_people 8
```

## 11. 三个预测头

第六步增加的是从共享 RSSM latent feature 解码三类任务观测的预测头：

```text
RSSM feat_t = concat(stoch_t, deter_t)
  ├─ HumanPredictionHead      -> 多人骨架 / 人是否存在
  ├─ EnvironmentPredictionHead-> ego-centric BEV occupancy
  └─ EgoStatePredictionHead   -> 无人机自身 17 维局部/目标相对状态
```

对应代码：

```python
from modules import CrowdWorldModelPredictionHeads, PredictionHeadConfig

heads = CrowdWorldModelPredictionHeads(
    feat_dim=rssm.feat_size,
    config=PredictionHeadConfig(
        max_pose_people=16,
        bev_hw=(24, 16),
    ),
)

feat = rssm.get_feat(post_stoch, post_deter)
pred, pred_losses, pred_metrics = heads.forward_loss(feat, batch, encoded)
```

三个 head 的输出：

```text
human.xy:                  [B, T, M, 17, 2]   # 归一化图像坐标
human.keypoint_conf_logits:[B, T, M, 17]
human.person_logits:       [B, T, M]

env.occupancy_logits:      [B, T, 1, Hbev, Wbev]

ego.state_norm:            [B, T, 17]
ego.state:                 [B, T, 17]         # 反归一化后的物理量
```

训练目标：

```text
human:
  target 来自 batch["pose_windows"] 当前帧，即 pose_windows[:, :, :, -1]
  xy 使用 SmoothL1，只在可见关键点上计算
  keypoint_conf 使用 BCE
  person presence 使用 BCE

environment:
  target 来自 encoded["pillar_occupancy"]
  下采样到 BEV feature grid，比如 [24, 16]
  occupancy 使用带正样本权重的 BCE

ego:
  target 来自 batch["ego_state"]
  先按固定物理尺度归一化，再 SmoothL1
```

当前 loss key：

```text
pred_human_xy
pred_human_keypoint
pred_human_presence
pred_env_occupancy
pred_ego_state
```

这些 loss scale 已经写进 `configs/model/_base_.yaml`，后续把三路编码器/RSSM/预测头正式接入 Dreamer 训练主流程时，可以直接把这些 loss 加到 world model loss 里。

正式训练前建议统计全量数据集的 ego state 归一化参数：

```bash
python scripts/compute_ego_state_stats.py \
  --data_root "/media/b403/LENOVO_USB_HDD/zyw" \
  --index_path crowd_dataset_index_lenovo.json \
  --out ego_state_stats_lenovo.json
```

脚本会输出可以复制进 `configs/model/_base_.yaml` 的：

```yaml
prediction_heads:
  ego_state_mean: [...]
  ego_state_std:  [...]
```

其中 `ego_state_mean/std` 会被 `EgoStatePredictionHead` 用于：

```text
target_norm = (ego_state - ego_state_mean) / ego_state_std
```

快速检查：

```bash
python scripts/smoke_test_prediction_heads.py \
  --data_root "/media/b403/LENOVO_USB_HDD/zyw" \
  --index_path crowd_dataset_index_lenovo.json \
  --pose_cache_root "/media/b403/LENOVO_USB_HDD/zyw_pose_cache" \
  --sequence_length 4 \
  --batch_size 1 \
  --max_lidar_points 1024 \
  --pose_window_size 4 \
  --max_pose_people 8
```
