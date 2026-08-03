# OmniDepth 三维人体骨架使用说明

## 一键采集

让人体完整出现在任一相机扇区，执行：

```bash
cd /home/neu/OmniNxt
./scripts/100_capture_3d_skeleton.sh
```

脚本会启动现有 OAK/OmniDepth 链路，取得四组完全同时间戳的虚拟双目左图和深度图，运行人体检测与 COCO-17 关键点检测，并生成二维和三维结果。它不会保存或显示完整点云。

默认使用与 Alienware 程序同档的 balanced 模型：YOLOX-M HumanArt 人体检测器和 RTMPose-M Body17。四扇区 CPU 推理通常需要约 10～15 秒，因此当前入口是高质量单帧抓取，不是实时视频。

## 输出文件

每次结果保存在：

```text
/home/neu/OmniNxt/runtime/pose_3d/capture_YYYYMMDD_HHMMSS/
```

主要文件：

```text
pose_depth_overview.png       四组左图与深度图上的骨架、关节深度
skeleton_3d_viewer.html       可旋转、缩放的三维骨架
skeletons_3d.json             全部二维/深度/三维数值
capture_metadata.json         ROS 同步时间戳和输入元数据

A_B_RIGHT/pose_on_image.png   右侧灰度图骨架
A_B_RIGHT/pose_on_depth.png   右侧深度图骨架
B_C_REAR/...
C_D_LEFT/...
D_A_FRONT/...
```

打开最近一次结果：

```bash
latest="$(find /home/neu/OmniNxt/runtime/pose_3d -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
xdg-open "${latest}/skeleton_3d_viewer.html"
xdg-open "${latest}/pose_depth_overview.png"
```

## 坐标与计算

姿态推理直接作用于四个 320×240 的校正左图，因而关键点像素与对应深度图严格对齐。对每个有效关节：

1. 在关键点周围 5×5 像素内取有限深度中位数；
2. 使用该双目校正后的 `P1` 将 `(u,v,Z)` 反投影到校正左相机；
3. 使用 OmniDepth 当前的 `R1^T`、虚拟相机旋转和正式 `T_cam_imu` 转入 `imu` 坐标。

当前 `imu` 即无人机机体系原点：

```text
+X：机头/前方
+Y：机体左方
+Z：机体上方
```

JSON 中每个关节包含：

```text
pixel          二维关键点坐标
score          RTMPose 置信度
depth_m        局部深度中位数
depth_mad_m    局部深度中位绝对偏差
xyz_rect_m     校正左相机坐标
xyz_imu_m      无人机/IMU坐标
```

相邻扇区可能看见同一个人。脚本保留 `raw_people`，并根据三维躯干中心距离生成 `fused_people`；HTML 默认显示融合骨架，可切换为原始扇区骨架。

## 可调参数

环境变量示例：

```bash
PERSON_THRESHOLD=0.45 \
KEYPOINT_THRESHOLD=0.35 \
POSE_MAX_DEPTH_M=5.0 \
POSE_MERGE_DISTANCE_M=0.60 \
./scripts/100_capture_3d_skeleton.sh
```

深度来自当前 HITNet，因此三维关节精度不可能高于深度本身。二维姿态正确而三维位置异常时，应优先查看同一关节在 `pose_on_depth.png` 的深度和 JSON 中的 `depth_mad_m`，而不是把问题归因于 RTMPose。

## 实时动态测试

```bash
cd /home/neu/OmniNxt
./scripts/101_live_pose_3d_test.sh
```

浏览器会自动打开 `http://127.0.0.1:8765`。页面同步显示四组校正左图、深度图和无人机坐标系中的三维骨架；按 `Ctrl+C` 结束测试。页面只处理最新数据，不会积压旧帧。

实时入口使用 YOLOX-Tiny 检测器、TensorRT FP16 RTMPose-S，并在检测器两次运行之间使用上一帧关键点更新人体框。可调参数：

```bash
POSE_DET_INTERVAL=20 \
PERSON_THRESHOLD=0.35 \
KEYPOINT_THRESHOLD=0.30 \
./scripts/101_live_pose_3d_test.sh
```

页面中的“真实新深度”按不同 ROS 时间戳计数。不能用 `/depth_estimation/stereo_*/depth` 的发布消息频率代替，因为当前发布线程可能以 10 Hz 重复发布同一份 HITNet 输出。

2026-08-01 初始实机测量：RTMPose-S TensorRT 单人推理约 12.5 ms（单模型约 80 FPS）；包含跟踪、绘图和文件服务时，非检测帧骨架部分约 44 ms。初始整条链路约3.0 FPS、真实新深度约3.4 Hz。完成重复发布修复、CUDA Graph、CPU校正流水和新旧输入握手后，真实四向深度约4.8 Hz，动态页面通常约4～5 FPS。瓶颈仍是四组HITNet而不是RTMPose；详细结果见 `reports/HITNET_RUNTIME_OPTIMIZATION.md`。
