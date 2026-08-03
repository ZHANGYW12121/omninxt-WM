# OmniNxt Sim2Real 同步配置（2026-08-02）

本配置已把实机包 `omnidepth_server_sim_sync_20260802` 的四鱼眼内外参、
OmniDepth/HITNet 深度链路和 COCO-17 三维骨架链路接入当前 Isaac OmniNxt。
正式默认输入是 **4 个 1280×720 原始 Mei+radtan 鱼眼**；12 个针孔相机只用于
校验中间结果，不是正式运行入口。

## 1. 正式数据链路

```text
Isaac 同一渲染步
  4 × 理想等距桥接图（内部 1280×1280）
  -> 精确 Mei+radtan 重映射
  -> CAM_A|CAM_B|CAM_C|CAM_D（4 × 1280×720）
  -> /oak_ffc_4p/assemble_image（5120×720, rgb8）
  -> quadcam_depth_est（实机同一套 fisheye + 4 组 stereo 标定）
  -> 4 组 320×240 rectified stereo
  -> HITNet TensorRT 深度（rectified-left optical Z）
  -> 4 个 416×320 中心 anchor + 4 组左右 stereo mosaic
  -> YOLOX + RTMPose + 双目三角化/HITNet fallback
  -> base_link 下的 COCO-17 三维骨架
```

相机顺序固定为 `CAM_A | CAM_B | CAM_C | CAM_D`，分别朝右、后、左、前。
所有同帧图像使用同一个 Isaac 仿真时间戳。原始图与实机一样不额外旋转 180°。

## 2. 一键启动

在项目目录执行：

```bash
cd /home/neu/zyw/our_omni_depth
./run_isaac_sync_live.sh
```

脚本会在 GPU1 启动 Docker 深度/骨架后端，在 GPU0 打开 Isaac UI。默认模式为
`raw_mei`，实时桥接默认 10 Hz，并自动打开
`http://127.0.0.1:8766` 深度/骨架窗口。窗口上方显示四向相机二维骨架与
`base_link` 三维骨架，下方显示右、后、左、前四幅 HITNet optical-Z 深度。
如不希望自动打开浏览器，可设置 `OMNINXT_OPEN_VIEWER=0`。仿真关闭后可执行：

```bash
./stop_sync_backend.sh
```

若只想先启动后端：

```bash
./start_sync_backend.sh
```

日志位于：

```text
shared/logs/roscore.log
shared/logs/sync_live_feeder.log
shared/logs/sync_pose.log
```

## 3. 12 针孔对照模式

该模式由 4 个 416×320 物理相机中心视图和 8 个 320×240 校正左右目组成，
绕过“原始鱼眼到虚拟针孔”的 C++ 步骤，用于判断误差是在 Isaac 原始成像、
鱼眼重映射还是下游网络。启动命令：

```bash
OMNINXT_SENSOR_MODE=rectified_validation ./run_isaac_sync_live.sh
```

校验模式渲染 12 个相机，速度明显低于正式四鱼眼模式，这是预期现象。

## 4. ROS 输出

主要话题：

```text
/oak_ffc_4p/assemble_image                 四鱼眼拼图
/depth_estimation/stereo_0/depth           右侧 Z 深度
/depth_estimation/stereo_1/depth           后侧 Z 深度
/depth_estimation/stereo_2/depth           左侧 Z 深度
/depth_estimation/stereo_3/depth           前侧 Z 深度
/depth_estimation/pose_anchor_mosaic       832×640 anchor 拼图
/depth_estimation/pose_stereo_mosaic       640×960 stereo 拼图
/omninxt_pose/joints_3d                    三维关节点
/omninxt_pose/skeleton_frame               固定 COCO-17 骨架包
/omninxt_pose/skeleton_markers             RViz MarkerArray
/omninxt_pose/status                       处理频率、人数和关节统计
```

`/omninxt_pose/status` 中 `output_frame=base_link`。骨架顺序和协议以
`sync_20260802/evidence/SKELETON_STGCN_STREAM.md` 为准。

## 4.1 低实时倍率下的 PX4 心跳

Isaac 视口 FPS 只表示每秒画面数，不等于仿真实时倍率。四个内部 1280×1280
相机、250 Hz 物理和深度处理同时运行时，1 秒仿真时间可能需要约4秒墙上时间。
PX4 默认 1 Hz 仿真时间心跳会因此超过 MAVSDK 的墙上时间超时门限。当前配置已在
Offboard 链路 `14580 -> 14540` 将 HEARTBEAT 提高为 5 Hz；源码和当前 build
副本均已修改，不需要降低物理频率或关闭相机。

## 4.2 五米行人检测与防闪烁

骨架输出默认只保留 `base_link` 原点欧氏距离 5 m 内的行人。距离门控不依赖当前
质量较差的 HITNet 深度图，而是综合检测框和人体肩宽、髋宽、躯干长度估算稳定的
单目距离；双目可靠时仍用于生成三维关节。新目标需进入 4.75 m 才加入，已加入的
目标在平滑距离超过 5.00 m 后移除，避免目标在 5 m 边界反复出现和消失。

检测端每个输入帧轮询一个 416×320 物理相机中心视图，每个方向约 2.5 Hz 做一次
YOLOX 全图刷新；检测框最多跨 3 次本相机漏检保留，三维轨迹最多短时预测 0.6 s。
RTMPose 会在中间帧继续更新。YOLOX 后处理已修正为真正使用 0.20 置信度阈值，
不会再错误地把 0.45 NMS IoU 阈值当作第二个置信度阈值。

查看实时效果：

```text
http://127.0.0.1:8766
```

顶部状态中的“5m内人体”是最终输出人数；“短时保持”是当前由轨迹预测补齐的人数；
“距离过滤”是当前检测到但位于 5 m 外的人数。二维相机画面会保留 5 m 外人员的
绿色骨架作为诊断信息，但他们不会出现在右侧三维骨架和 ROS 最终输出中。

## 5. 构建与恢复

已构建镜像：

```text
omnidepth:ada-sync-20260802
```

需重新构建时：

```bash
docker build -f Dockerfile.sync_20260802 -t omnidepth:ada-sync-20260802 .
./build_sync_engines.sh
./start_sync_backend.sh
```

YOLOX/RTMPose engine 由 ONNX 在目标 GPU 上构建。HITNet engine 由 TensorRT 10
C++ 节点首次启动时生成，因此首次启动可能等待数分钟；以后直接复用。不要把这些
engine 当作跨 GPU/TensorRT 版本的通用文件。

权威标定副本：

```text
D2SLAM/config/quadcam_drone_nxt_sync_20260802/
```

原始交付包完整保存在：

```text
sync_20260802/
```

## 6. 安装自检

```bash
./verify_sync_installation.sh
```

成功结束时输出 `SYNC_INSTALLATION_OK`。该自检验证交付包哈希、矩阵方向、Isaac
相机几何、三套 engine、Docker 镜像，并在后端已运行时检查 C++ ROS 订阅端。

## 7. 已实测结果

- 四鱼眼正式模式：Isaac 原始拼图、四组 C++ 校正、TensorRT 10 HITNet 深度均通过。
- 原始图与 `stereo_0/depth` 时间戳逐纳秒一致。
- HITNet 实测输出为 `320×240 32FC1`，有效深度约 `1.20–2.84 m`（该测试帧）。
- 姿态链路稳定处理约 10 Hz；仿真人物进入视野时检测到 1 人，产生 7 个双目三角化
  关节、9 个 HITNet fallback 关节，共 17 个有效 COCO 关节。
- 12 针孔校验模式能连续发布两张严格同步拼图并被同一姿态节点处理。

CAM_D 的实机标定 `xi=1.871921`、畸变较强，正式 1280×720 图中的精确可映射区域
约 66%；无效区填黑是标定模型的自然结果，不能擅自扩大视场或替换内参。
