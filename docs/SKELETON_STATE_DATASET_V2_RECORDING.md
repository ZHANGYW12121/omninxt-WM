# 三维骨架与无人机状态数据集 v2

## 当前状态

录制代码已经接入，但默认关闭：

```text
OMNINXT_DATA_RECORD_ENABLED=0
```

默认启动仿真不会创建 episode 或数据文件，也不会监听骨架录制端口。Isaac 物理仍为
250 Hz，人群控制和静态路径规划配置不受本录制器影响。

正式接口见：

```text
interfaces/dataset/CROWD_SKELETON_STATE_V2.yaml
```

## 观测与特权信息

模型可见观测只有：

- `base_link` ROS FLU 坐标系下的 COCO17 三维骨架；
- 14维无人机状态；
- 任务目标的机体系相对向量。

仿真骨架保留 YOLOX、RTMPose、跟踪、漏检和置信度，关节深度统一来自与图像
完全同时间戳的 Isaac `distance_to_camera` 真值。严格模式不会回退到 HITNet、稀疏
双目深度或单目人体尺度。

动作、奖励、终止信号是训练转移所需标签，不属于额外传感器观测。行人世界坐标、
编组、Pelvis、10个碰撞关节、无人机世界真值、最小间距和 TTC 均标记为
`privileged`，不能输入策略或世界模型观测编码器。

## 与控制算法解耦

录制器只读取 `_dataset_action_snapshot()` 提供的统一动作：

```text
[vx_body_mps, vy_body_mps, vz_world_mps, yaw_rate_rps]
```

当前经典控制器、MAVSDK和键盘后端都已转换到该 ROS FLU 语义。以后替换控制算法时：

1. 新控制器继续写入 `SharedCommand`，录制器无需修改；或
2. 只增加一个动作适配器，返回 `requested`、`normalized` 和
   `applied_body_flu`，不要修改数据集字段。

控制器固定按 `DATA_CONTROL_HZ` 更新，不再依赖骨架是否到达、样本是否写盘或 writer
队列是否繁忙。

## 以后开始录制

当前不要执行以下命令。确认控制算法后再运行：

```bash
cd /path/to/omninxt-WM
OMNINXT_DATA_RECORD_ENABLED=1 \
OMNINXT_POSE_DEPTH_SOURCE=isaac_gt \
./simulation/omnidepth/run_isaac_sync_live.sh
```

路径由 `.local/machine.env` 的 `OMNINXT_DATASET_ROOT` 决定。输出结构为：

```text
${OMNINXT_DATASET_ROOT}/
├── dataset_manifest.json
└── episodes/
    └── episode_YYYYMMDD_HHMMSS/
        ├── metadata.json
        ├── chunks/chunk_000000.npz
        ├── events.jsonl
        └── summary.json
```

经典自动任务在起飞稳定并进入导航时开始录制；手动模式可使用原来的录制按键。每个
NPZ 默认保存256帧，使用临时文件写完后原子重命名。数据中不保存 RGB、深度图、
点云或二维骨架。

## 录制后的校验

```bash
python3 tools/validate_skeleton_dataset_v2.py \
  "${OMNINXT_DATASET_ROOT}/episodes/episode_YYYYMMDD_HHMMSS"
```

校验器会检查字段、形状、时间连续性、NaN/Inf、禁止的图像/点云文件，并拒绝仿真
主观测中出现 HITNet 来源的三维关节。
