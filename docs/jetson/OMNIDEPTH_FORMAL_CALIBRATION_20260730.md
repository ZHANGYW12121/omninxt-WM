# OmniDepth 正式标定与验收报告

日期：2026-07-30  
设备：OAK-FFC-4P，MXID `19443010E16C6A2E00`

## 结论

四鱼眼内参、四路相机—PX4 IMU 外参、四组相邻虚拟双目参数均已完成。
正式参数已安装到 D2SLAM/OmniDepth 配置目录，并通过离线与在线端到端验收。

相机物理顺序：

| 逻辑相机 | 物理方向 |
|---|---|
| CAM_A | 右前 |
| CAM_B | 右后 |
| CAM_C | 左后 |
| CAM_D | 左前 |

环形相邻关系为 `A-B-C-D-A`。

## 标定输入

- AprilGrid：6×6
- `tagSize`：0.088 m（用户确认打印成品与名义值接近）
- `tagSpacing`：0.3
- 鱼眼模型：Omni + Radtan
- 虚拟相机：Pinhole + Radtan
- 虚拟相机尺寸：320×240
- 虚拟视场：190°
- 原始相机数据：20 Hz
- 内参求解子集：约 5 Hz；原始 20 Hz 数据未被修改
- 相机—IMU求解：原始相机帧与 PX4 `/mavros/imu/data_raw`

标定板尺寸直接决定平移和深度的绝对尺度。若以后精确测量发现 Tag 边长不是
88.0 mm，所有平移和深度尺度应按 `实测值 / 88.0 mm` 修正或重新标定。

## 虚拟双目结果

| 相机对 | 物理侧 | 有效标定帧 | 最终使用帧 | 基线 | 双目重投影标准差 |
|---|---|---:|---:|---:|---|
| CAM_A–CAM_B | 右侧 | 741 | 92 | 144.880 mm | 0.163–0.205 px |
| CAM_B–CAM_C | 后侧 | 743 | 92 | 144.732 mm | 0.158–0.225 px |
| CAM_C–CAM_D | 左侧 | 738 | 88 | 144.919 mm | 0.169–0.203 px |
| CAM_D–CAM_A | 前侧 | 741 | 93 | 145.195 mm | 0.165–0.192 px |

四组基线极差为 0.462 mm。生成的虚拟双目 bag 含 8 个压缩图像话题，
每个话题 2964 帧；每组左右图像时间戳逐帧完全一致。

通用 ORB 在 AprilGrid 重复纹理上会产生跨 Tag 错配，因此不作为极线验收
判据。正式判据是 Kalibr 使用 Tag ID 建立的一一对应角点及其双目重投影残差。

## 正式安装文件

- `source/D2SLAM/config/quadcam_drone_nxt_tmp/quadcam_depth.yaml`
- `source/D2SLAM/config/quadcam_drone_nxt_tmp/fisheye_cams.yaml`
- `source/D2SLAM/config/quadcam_drone_nxt_tmp/stereo_calib_0_1_240_320.yaml`
- `source/D2SLAM/config/quadcam_drone_nxt_tmp/stereo_calib_1_2_240_320.yaml`
- `source/D2SLAM/config/quadcam_drone_nxt_tmp/stereo_calib_2_3_240_320.yaml`
- `source/D2SLAM/config/quadcam_drone_nxt_tmp/stereo_calib_3_0_240_320.yaml`

SHA256：

```text
quadcam_depth.yaml                 d4ef078e9f8e13d7910b9555c99c901c98b43402206ff42b164d2833654ae1bd
fisheye_cams.yaml                  908ec37172aa9912c0a94db286ef7cfdb1f36eb4eb0c2df08f12f962e0546e47
stereo_calib_0_1_240_320.yaml      cca8fae1710f61f561251aa35f5c8505b5485a8bd41934bf71e80c02deab4fb8
stereo_calib_1_2_240_320.yaml      cb066634eecb760bb40bd4b3d65f9ae667dbc9509e8d6c90f45aeee173682e16
stereo_calib_2_3_240_320.yaml      e02173639c5ffef525c3927e9260ba52019fe877caf149912c305c9c9104b606
stereo_calib_3_0_240_320.yaml      683bdde46493f3477ac3be5af464b1f3632b97fff6916022f0b8f525f5ec7282
```

旧示例参数保存在：

`source/D2SLAM/config/quadcam_drone_nxt_tmp/example_before_formal_calibration_20260730/`

## 端到端验收

离线回放使用本次正式标定采集的真实 OAK 拼接图：

- 正式配置与四组虚拟双目参数全部加载成功
- TensorRT engine 反序列化成功
- 生成 76,799 个有限 XYZ 点
- PCD SHA256：
  `eede8c77f6b5b876a762af663d22bc417a0305a61bddb5cdea192716e4b1540b`
- 无 CUDA illegal memory、反序列化失败、OOM 或节点崩溃

60 秒在线验收：

- OAK 运行态：`USB speed: SUPER`
- 输入：`bgr8`，5120×720，约 20.00 Hz
- 点云：约 10.00 Hz，`frame_id=imu`
- 结束时点云序号：932
- CPU 峰值：61.25°C
- GPU 峰值：60.656°C
- RAM 峰值：5221 MiB / 7472 MiB
- GR3D 平均：约 90.0%
- 无 CUDA、TensorRT、OOM、USB 断连或节点崩溃

OAK 日志中的第二个 `roscore` 报告已有 master，是启动脚本检测到共享 ROS
master 后的预期信息；相机节点随后正常连接并持续发布，不属于相机错误。

## 产物和证据

- 虚拟双目目录：
  `runtime/calibration/runs/quarterkalibr_20260729_204843/prepared/virtual_stereo_calibration_190/`
- 自动验收：
  `virtual_stereo_calibration_190/validation/virtual_stereo_validation.txt`
- 四份 8 页 Kalibr PDF：`stereo_calib_*_240_320-report-cam.pdf`
- 离线点云：`runtime/offline_20260730_024546/pointcloud.pcd`
- 离线日志：`logs/40_offline_smoke_20260730_024546.log`
- 在线日志：`logs/50_live_pipeline_20260730_024656.log`
- 在线资源日志：`logs/50_live_tegrastats_20260730_024656.log`

## 使用边界

- 调整任一镜头、排线映射或相机支架后，必须重新标定。
- 改变飞控相对相机的安装位置或方向后，必须重做相机—IMU外参。
- 相机—IMU结果使用 MAVLink/USB 传输的 PX4 IMU；高精度 VIO 上线前仍应
  长时间检查时间戳、传输抖动，并用正式 Allan 方差参数替换临时 BMI088
  噪声参数。
- 当前结果已足够用于单帧 OmniDepth；该推理链本身不需要实时 IMU话题。

## 点云坐标变换修正

后续几何检查发现，最初在线验收只验证了有限点数量、频率和运行稳定性，
没有验证四扇区在 `imu` 坐标系中的物理方向。独立检查确认 OmniDepth 节点
曾把 Kalibr 的 `T_cam_imu`（IMU到相机）直接用于相机点到IMU的合并，
并遗漏了双目校正左目的 `R1^T`。

2026-07-30 已完成以下修正：

- 配置显式使用 `extrinsic_parameter_type: 0`，对 `T_cam_imu` 求逆；
- 点云从校正左目变换回虚拟左目时纳入 `R1^T`；
- 节点启动时输出四组虚拟双目光轴；
- 重新编译并在线采集同帧图像和彩色扇区点云。

在线实测扇区点分布中位方位角：

```text
A-B 右侧： -89.6°
B-C 后侧：-176.9°
C-D 左侧： +93.1°
D-A 前侧：  +1.3°
```

这与ROS FLU坐标系的 `+X前、+Y左、+Z上` 一致。修正后的在线点云约
10.0 Hz，未发现CUDA、TensorRT、OOM或节点异常。

修正后证据：

- `runtime/live_omnidepth_20260730_130009.log`
- `runtime/html_pointclouds/capture_20260730_130610/pointcloud_viewer.html`
- `runtime/html_pointclouds/capture_20260730_130610/xy_sector_validation.png`
