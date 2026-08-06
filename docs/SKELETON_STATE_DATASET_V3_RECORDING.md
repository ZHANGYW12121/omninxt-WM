# 三维骨架与无人机状态数据集 v3（精简版）

## 当前状态

录制代码已接入，但默认关闭：`OMNINXT_DATA_RECORD_ENABLED=0`。不开启时不会创建
episode，也不会监听骨架录制端口。Isaac 物理 250 Hz、人群控制 25 Hz、骨架更新
10 Hz 和事件触发路径规划均未改变。

正式接口是 `interfaces/dataset/CROWD_SKELETON_STATE_V3.yaml`。v2 保留用于读取旧数据，
新录制统一写 v3。

## 为什么这样精简

每帧只保存以下不可安全重建的数据：

- `base_link`（ROS FLU）下 COCO12_BODY 的 XYZ、置信度、有效掩码和跟踪 ID；
- Ego14 无人机状态；
- 已实际执行并归一化的四维动作、奖励、奖励分量和终止标志；
- 行人世界位置/速度以及最小间距、TTC、碰撞、目标距离等特权标签。

不再重复保存 RGB、深度、点云、二维骨架、四元数副本、请求动作副本、逐关节质量
诊断、Pelvis 和 10 个碰撞关节世界坐标。行人速度、模型 20 个稳定槽位、
`human_root`、相对关节、Goal8、`dt`、`discount` 在训练加载时生成。骨架上游仍是
COCO17，但录制时永久丢弃鼻、双眼、双耳（源索引 0–4），只保存源索引 5–16：

```text
left/right shoulder, elbow, wrist, hip, knee, ankle
```

落盘、加载、ST-GCN、root 计算、人体预测头和 MPJPE 均使用这同一套 12 关节定义。

录制维度 `N` 是该 episode 的实际场景人数（默认场景 17–23，配置安全上限 60），
不会在录制时截成模型的 20 人。训练加载器根据距离、TTC 和前向走廊风险选择 20 人，
并保持同一序列内 track ID 的槽位稳定。

## 坐标与速度

- 骨架 XYZ：当前无人机 `base_link`，`+X` 前、`+Y` 左、`+Z` 上，单位米。
- Ego14 位置：episode 起点且按起始 yaw 对齐的局部坐标。
- Ego14 水平速度/加速度：当前机头坐标；垂直分量为世界 Z。
- 行人关节速度：加载器把上一帧关节通过两帧 Ego14 位姿转换到当前
  `base_link` 后差分。因此无人机自身平移或转向不会被误认为行人在运动。
- 行人世界位置/速度：仿真世界坐标，仅供奖励、评估和排错，禁止输入策略。

## 控制算法接口

控制器仍只需提供统一的 `applied_body_flu`：

```text
[vx_body_mps, vy_body_mps, vz_world_mps, yaw_rate_rps]
```

录制器用 episode metadata 中的四个正数上限归一化到 `[-1,1]`。第一帧没有前序
转移，写零动作且 `action_valid=false`。以后更换离线采集控制算法不需要改数据字段。

## 结果与终止语义

成功判定与 benchmark 一致：无人机世界坐标到当前目标点的三维距离不超过 `1.0 m`。
无人机与行人接触记为 `human_collision`，与墙、货架、地面等环境碰撞记为
`static_collision`。每帧 NPZ 都直接保存 `success`、`termination_code`、`is_last` 和
`is_terminal`，具体文字原因及 `outcome_class` 同时写入 `summary.json`。

| outcome_class | termination_reason | is_last | is_terminal |
|---|---|---:|---:|
| success | reached_goal | true | true |
| task_failure | human/static collision、out_of_bounds、crash、stuck_timeout | true | true |
| truncated | time_limit、manual_stop、shutdown、record_size_limit | true | false |
| invalid | controller_error 或未知错误 | true | false |

这些标签可供以后分层采样、统计和监督，但不能输入 Ego/Human 观察编码器。录制阶段保留
所有成功、任务失败和截断轨迹；正负 episode 比例与终端窗口加权按当前决定留到训练
采样阶段处理，本次不在录制时删除或复制样本。

第一阶段离线训练不是“只训练预测器”。现有 `train_factorized_offline.py` 会从已录数据
完成 Ego/Human RSSM、重建/预测头、Reward/Continue、Actor、Value 和 Replay Value 的
联合优化，只是不与 Isaac 实时交互；第二阶段再由 Dreamer 在线交互继续训练。

## 以后开始录制

确认控制算法后再执行：

```bash
cd /path/to/omninxt-WM
OMNINXT_DATA_RECORD_ENABLED=1 \
OMNINXT_DATA_RECORD_SEED_START=1 \
OMNINXT_DATA_RECORD_SEED_END=100 \
OMNINXT_DATA_RECORD_RUN_ID=navrl_no_shield_stage1 \
OMNINXT_DATA_RECORD_RESUME=1 \
OMNINXT_POSE_DEPTH_SOURCE=isaac_gt \
NAVRL_SAFETY_SHIELD_ENABLED=0 \
./simulation/omnidepth/run_isaac_sync_live.sh --control-mode px4_navrl
```

seed 区间首尾都包含在内。仓库模板每完成一个 episode 后使用下一 seed 重建预规划
地图；该 seed 同时确定人数、行人路线/停留安排、出生点与目标点。metadata 会记录
实际 seed、区间、run ID、人数、导航算法和 safety-shield 状态。

`PegasusApp.current_target_point` 是每个 episode 的唯一任务目标：NavRL、录制 metadata、
goal 特征、奖励和终止判定都使用该点。录制模式还会把 NavRL 的悬停半径限制在录制
成功半径以内（当前统一为 `1.0 m`），避免策略已经悬停但 episode 被错误标记为
`time_limit`。同组行人的
硬碰撞/最小间距阈值默认为 `0.15 m`；不同组仍使用完整 personal-space 阈值。

进度原子写入：

```text
${OMNINXT_DATASET_ROOT}/recording_progress/
  navrl_no_shield_stage1_seed_1_100.json
```

重启并使用完全相同的 `RUN_ID/SEED_START/SEED_END` 时，会扫描进度文件和已经完整落盘
的 episode summary，跳过连续完成的 seed，从第一个未完成 seed 重新录制。强制关闭或
掉电时，`shutdown`/缺少 summary 的当前 seed 不算完成，因此会重录。`manual_stop`、
`controller_error` 和 `record_size_limit` 同样不推进 seed；成功、任务碰撞/失败和正常
`time_limit` 才算该 seed 已完成。残留目录保留用于审计，但训练加载器会跳过没有
summary 的不完整目录。若要对同一 seed 区间重新进行一组独立算法实验，应换一个
`OMNINXT_DATA_RECORD_RUN_ID`。

输出仍是：

```text
${OMNINXT_DATASET_ROOT}/
├── dataset_manifest.json
└── episodes/episode_YYYYMMDD_HHMMSS/
    ├── metadata.json
    ├── chunks/chunk_000000.npz
    ├── events.jsonl
    └── summary.json
```

若同一根目录已有历史 v2 manifest，新录制会另写 `dataset_manifest_v3.json`，不会覆盖
旧 manifest；每个 episode 仍由自己的 `metadata.json` 明确标识 schema。

## 校验与训练

录完一个 episode 后：

```bash
python3 tools/validate_skeleton_dataset_v3.py \
  "${OMNINXT_DATASET_ROOT}/episodes/episode_YYYYMMDD_HHMMSS" \
  --require-isaac-gt
```

离线世界模型训练直接读取 chunk，不再需要 `--pose-root`：

```bash
python world_model/scripts/train_factorized_offline.py \
  --data-root "${OMNINXT_DATASET_ROOT}" \
  --output /path/out/factorized_world_model.pt
```

训练侧入口是 `world_model.datasets.CompactSkeletonV3Dataset`。它只把模型可见字段交给
Ego/Human 编码器；`priv_*` 字段随 batch 保留供奖励复核，但不进入观察编码器。
