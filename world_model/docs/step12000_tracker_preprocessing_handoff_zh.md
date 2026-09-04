# Step-12000 tracker 与预处理交付说明

本文冻结 12000-step 模型所依赖的人体 tracker、数据字段和 Human 输入预处理契约，供实机端逐字段对拍。模型文件为 `world_model/deployment_checkpoints/step_00012000_v67_human_exogeneity_klt_audit_compat.pt`，SHA256 为 `7c3cba29d55c22c5bc671a35d68d0b1fe87a53caf5be86ae498c9b92e969ad52`。该 v67 文件的迁移记录明确 `changed_agent_tensors=[]`，即迁移没有改变 step-12000 的网络参数。

## 版本与可追溯性

训练 run `pure_dreamer_v114_preemptive_boundary_latest_seed_20260901` 中共检查到 1401 份 replay `metadata.json`，它们记录的 `git_commit` 全部是：

```text
ecb28555292fb997597f5f0760bc59958445fb1c
```

但这个值只表示采集时仓库的 `HEAD`。当时 tracker 和预处理改动仍在工作树中，metadata 没有记录 dirty 状态和文件哈希，因此仅 checkout 上述 commit **不能**还原实际输入链路。`source_manifest.json` 给出了本次冻结源码的逐文件 SHA256；本交付 commit 才是外星人端应 checkout 的可复现版本。历史事实应表述为“recorded HEAD = ecb2855 + 未提交的 tracker/preprocessing overlay”，不能把 ecb2855 单独称为实际完整源码版本。

原始 `step_00012000.pt` 生成于 2026-09-03 12:56:37（Asia/Shanghai）。tracker、packet 和 recorder 源文件的文件时间均早于它；模型侧 compact loader 后续还承担了 v62–v67 部署兼容和审计，当前冻结版本与最终 v67 checkpoint 的输入契约一致。由于训练进程没有保存当时工作树快照，无法对原始 learner 进程中的每一个 Python 字节作超出上述证据的声明。

## 源码入口

- `simulation/omnidepth/runtime/102_live_stereo_pose.py`：同步输入、3D 骨架融合、tracker 实例化。
- `simulation/omnidepth/runtime/motion_skeleton_tracker.py`：ID、root velocity、`velocity_valid`、`velocity_sigma_mps`、identity confidence 和轨迹生命周期。
- `simulation/omnidepth/runtime/skeleton_stream.py`：线上 packet 的固定字段及数值化。
- `simulation/omnidepth/runtime/episode_sync.py`：图像、深度和 Ego pose 的 exact-stamp bundle，以及时钟回绕后的 episode generation。
- `simulation/omnidepth/runtime/person_detection_policy.py`：进入 3D tracker 前的 detector ROI 和 articulated-body 有效性判定。
- `simulation/omnidepth/runtime/trt_rtmpose.py` 及其 `rtmlib/.../yolox.py`：产生 tracker measurement 的 TensorRT pose/detector 包装和后处理。
- `simulation/isaacsim/database/skeleton_packet_receiver.py`：packet 校验与 session 变化。
- `simulation/isaacsim/database/skeleton_dataset_recorder.py`：按原始仿真时间将 skeleton packet 与 Ego snapshot 配对。
- `simulation/isaacsim/database/dataset_v3_schema.py`：COCO17 到模型所用 COCO12 字段。
- `world_model/datasets/compact_skeleton_v3.py`：稳定 slot、短缺帧保持、速度和 quality 输入。
- `world_model/datasets/factorized_schema.py`、`world_model/modules/skeleton_topology.py`：root/joint 分解、拓扑补全与几何清洗。

## 时间对齐

前端要求 anchor 图像、stereo 图像、Ego pose，以及 `isaac_gt` 模式下四路 depth 使用**完全相同的 `timestamp_ns`**才组成一帧。该时间戳同时传入 tracker 并写入 skeleton packet。episode generation 改变或仿真时钟回绕超过 0.5 s 时，未完成 bundle 被清空，tracker session 重置。

recorder 以仿真时钟 10 Hz 采集 Ego/action snapshot，再把 skeleton packet 匹配给绝对时间差最小的空 snapshot；允许误差为 0.075 s，等待感知的 wall-time 上限为 0.75 s。超时行写显式空 skeleton，并将 `skeleton_fresh=false`，不会把较新的骨架冒充同一时刻观测。

实机端应复现第一层 exact-stamp 或以 Ego ring buffer 插值到 skeleton 时间戳；不能直接把“最新骨架”和“当前 Ego”拼接。

## Root 定义

tracker 用于关联和速度历史的 center：

1. 若上游给出有限的 `range_gate_center_m`，直接使用它；
2. 否则取 COCO17 的 left/right shoulder 与 left/right hip（索引 5、6、11、12）有效点的逐坐标中位数；
3. core 少于 2 个时，退化为所有有效关节的逐坐标中位数。

模型记录 COCO17 的 5..16 共 12 个 body joints。模型 `human_root` 位置优先采用 COCO12 索引 6、7（原 COCO17 hip 11、12）的中点，但仅在 hip 距离和 hip-to-body-median 偏移通过几何检查时使用；否则采用有效 body joints 的稳健逐坐标中位中心。模型 root velocity 最终由 tracker 的 `root_velocity_base_link_mps` 覆盖，不由髋关节差分替代。

## Tracker 的准确配置

训练仿真使用 `isaac_gt` joint depth，构造参数如下：

| 参数 | 值 | 含义 |
|---|---:|---|
| `confirmation_hits` | 2 | 连续两次命中后分配正 `person_id` |
| `prediction_timeout` | 1.2 s | 最长对外发布预测轨迹的时间 |
| `deletion_timeout` | 4.0 s | 已确认轨迹的内部删除时间 |
| `base_gate` | 0.70 m | 关联基础门限 |
| `max_speed` | 2.0 m/s | root velocity 向量模长上限 |
| `max_person_range` | 6.0 m | 输出人体的 Euclidean `base_link` 距离门限 |
| `enable_ray_recovery` | false | `isaac_gt` 深度关闭同射线宽松恢复；实机 hybrid 默认开启 |
| root 历史 deque | 15 samples | 容量上限；另按时间只保留最近 1.0 s |

速度历史只有在至少 3 个样本、首尾跨度至少 0.20 s 时才有效。斜率集合包含窗口中所有 `dt >= 0.08 s` 的样本对，每对计算 `(p_j-p_i)/(t_j-t_i)`；至少要得到 3 条斜率。三轴分别取中位数，再按 2.0 m/s 的向量模长限幅。

更新时的瞬时中心速度先限到 `1.4 * max_speed = 2.8 m/s`，再以 `0.70 * old + 0.30 * measured` 平滑并限到 2.0 m/s；有效稳健历史出现后，使用 `0.25 * filtered + 0.75 * median_slope` 并再次限到 2.0 m/s。旧速度大于 0.15 m/s、0.45 s 内出现反向瞬时分量时，反向分量会被去掉。中心 innovation 上限为：

```text
0.40 + 0.18 * min(1.5, dt_s) + 0.10 * ||old_root_velocity||
```

## 异常值和人体几何处理

- 输入至少要有 4 个 body joints，且 5、6、11、12 中至少 2 个有效；否则整个人体 measurement 被拒绝。
- 完整人体按固定骨长范围计分；可评估骨段不少于 6 时，score < 0.45 的 measurement 被拒绝。score < 0.70 的新轨迹至少需 4 次确认。
- 每个关节是 6 维常速度 Kalman state。measurement sigma 在 `isaac_gt_depth` 下最低 0.035 m，普通深度最低 0.05 m，detection-only 最低 0.20 m，统一裁剪到不超过 1.0 m。
- 关节 measurement 同时要求 3D innovation 不超过 0.55 m 且 NIS 不超过 25。关节 Kalman velocity 限到 2.5 m/s。
- 若多个 core joints 给出一致的大位移（core delta 离散度不超过 0.22 m）且人体几何可信，可执行 coherent relocalization；此时 root velocity 历史清空，`velocity_valid=false`。
- 左右对称关节只有在交叉匹配比直接匹配至少好 0.08 m 时才交换。
- 模型侧只用同帧两个实测端点补缺失的中间肢体关节，不外推端点。超出人体拓扑上限的坐标被投影回可行几何，且改动点标成 predicted、不得作为实测监督。
- 模型侧逐关节速度是过去两帧差分，向量模长限到 3.0 m/s；root velocity 仍使用 tracker 值。

## `velocity_sigma_mps` 的真实含义

代码没有把位置残差再除以时间。对稳健速度 `v` 和最后样本 `(t_L,p_L)`，每个历史样本的残差为：

```text
r_i = ||p_i - (p_L - v * (t_L - t_i))||
velocity_sigma_mps = median(r_i)
```

因此该字段虽然历史命名为 `velocity_sigma_mps`，实际数值量纲是**米**，是“该速度假设对最近轨迹的中位位置残差”，不是速度标准差，也不是协方差。外星人端为了与 step-12000 模型严格兼容，必须先保持此计算和数值尺度；若以后改成真正的 m/s 不确定度，需要改 schema 并重新训练/校准，不能静默替换。

## 有效性、置信度与 ID 生命周期

`velocity_valid=true` 当且仅当历史窗口、跨度、斜率数量满足上述要求，且 median-slope velocity 和残差都是有限数。无效时线上 `velocity_sigma_mps=null`；数据 schema 存 0，并保持 `human_velocity_valid=false`，模型 quality 中 sigma 和 identity quality 都置 0。

`identity_confidence` 不是分类器输出，而是距最后一次观测时间的确定性衰减：

```text
exp(-time_since_observation_s / max(0.1, prediction_timeout_s))
```

实测帧为 1.0；默认 1.2 s hold 下，丢失 0.1 s 为 0.920044，丢失 0.2 s 为 0.846482。

新轨迹先为 provisional。默认连续 2 次命中且相邻观测 `dt <= 0.35 s` 后分配从 1 递增的 `person_id`。未确认轨迹 0.65 s 未观测即删除；已确认轨迹 1.2 s 后停止输出，但内部保留到 4.0 s 以便重关联。`track_uid = session_id + ':' + person_id`。

episode generation 变化、检测到时钟回绕或显式调用 `reset_session()` 时，全部轨迹删除、随机生成新 `session_id`、`person_id` 重新从 1 开始。也就是说 `person_id` 只保证一个 session 内的短期稳定，跨 PX4/Isaac episode reset 的全局身份必须由 `track_uid` 命名空间或外部 ReID 处理。

## 模型实际接收的字段

数据 schema 保存 `human_root_velocity`、`human_velocity_valid`、`human_velocity_sigma_mps`、measurement age、track age、prediction run 和 identity confidence。每帧按 ID 分配稳定 Human slot；内部 slot ID 使用记录 ID 加 1，以保留 0 作为空/无身份哨兵。缺失 1–2 个 10-Hz replay row 时，只用过去状态匀速保持，不查看未来帧。

每个有效 slot 的 7 维 quality 顺序为：

```text
[measured_joint_ratio,
 predicted_joint_ratio,
 track_age_frames * 0.1,
 measurement_age_s,
 prediction_run_frames,
 velocity_sigma_mps if velocity_valid else 0,
 float(velocity_valid) * identity_confidence]
```

模型**不接收完整速度协方差矩阵**。关节 Kalman 内部 covariance 只用于门控和滤波，跨 packet 输出的只有上述历史残差标量。

## Golden trace 对拍

仓库提供一段 11 帧 COCO17 输入：包含 0.8 m/s 平移、单关节 2 m 异常、短时遮挡、1.2 s 输出超时、4.0 s 删除和 episode reset。运行：

```bash
cd /path/to/omninxt-WM
python tools/verify_step12000_tracker_golden.py
```

预期输出：

```text
PASS: 11 golden frames match .../golden_expected.json
```

如需查看本机实际输出而不比较：

```bash
python tools/verify_step12000_tracker_golden.py --print-current
```

实机实现应先通过该 golden trace，再用同一批带时间戳的真实 3D 骨架和 Ego pose 做端到端 packet/模型输入对拍。
