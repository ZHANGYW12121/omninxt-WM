# OmniDepth 标定前部署报告（V2）

生成日期：2026-07-25  
平台：Jetson Orin Nano 8GB / ARM64  
工作目录：`/home/neu/OmniNxt`

## 结论

已按 V2 在本机从官方源码完成 OAK‑4P ROS1 驱动、JetPack 5.1.2 原生 OmniDepth、Orin 专属 TensorRT engine、真实相机离线链路及 ARM64 quarterKalibr/TartanCalib/Kalibr 环境。未复制其他设备的 engine 或动态库，未运行 amd64 镜像，未改变 JetPack/L4T，未在宿主 Python 安装标定依赖。

正式标定尚未开始。`CAM_A/B/C/D` 物理方位、标定板实测参数和最终 IMU 话题仍待用户确认；现有占位 YAML 不能用于几何精度判断。

## 1. 系统与容器

- 设备：NVIDIA Orin Nano Developer Kit（F301 载板）
- 架构：`aarch64`
- Ubuntu：20.04.6 LTS
- 内核：`5.10.120-tegra`
- L4T：R35.4.1
- JetPack：5.1.2
- CUDA：11.4.315
- cuDNN：8.6.0
- TensorRT：8.5.2
- Docker：26.1.3，NVIDIA runtime 已通过真实 CUDA/TensorRT/cuDNN 容器检查
- 根文件系统：NVMe，约 233 GiB
- 内存：约 7.3 GiB；zram swap 约 3.6 GiB
- nvpmodel：15W

JetPack 版本保持不变。Docker 代理配置位于
`/etc/systemd/system/docker.service.d/proxy.conf`。基础容器
`nvcr.io/nvidia/l4t-jetpack:r35.4.1` 已确认 ARM64。

OmniDepth 最终镜像：

```text
omninxt/omnidepth-orin:jp512
sha256:e6381f7769ee9d2dafdc9854ff6b34f7348c41e8395510b1bbe2741d8acfa90e
Architecture=arm64
```

标定最终镜像：

```text
omninxt/tartancalib-noetic-arm64:jp512
sha256:077eb9a4948172a5225290cc48b864f799191b0abc63ffa1e25b0c170c0453bd
Architecture=arm64
```

## 2. 官方源码基线

| 仓库 | 分支/基线 | commit |
|---|---|---|
| HKUST OmniNxt | `main` | `79ce1d4b69a5a8c861417716adaac00c2bc6f594` |
| HKUST D2SLAM | 官方 `pr_fix_main`；本地 `jetson-orin-nano-jp512` | `3a4b8071f6f9c40a151c7685aa738b717ce8916a` |
| OAK‑FFC‑4P ROS driver | `main` | `3babba51c6842492dfef1bbf646fa11b9d3416cd` |
| tools-quarterKalibr | 本地 `jetson-orin-nano-jp512` | `5a2be86094d7f513c343d89fdf0d39ca009a1263` |
| LCM | v1.4.0 | `abc18a8f48a2ce23e41898f4a928c0b1ffb66fdb` |
| tartancalib | `main` | `78d3f26c9f458bc60c26012399c06614d8d45763` |
| Kalibr | `master` | `1f60227442d25e36365ef5f72cd80b9666d73467` |
| OpenCV / contrib | 4.6.0 | `b0dc474…` / `db16caf…` |

没有 push 任何远程仓库。工作树差异保存在
`reports/d2slam_jetpack512_port.patch` 和
`reports/quarterkalibr_safety.patch`。

## 3. OAK‑4P 验收

- 型号：OAK‑FFC‑4P / OAK‑4P‑New‑B033501
- MXID：`19443010E16C6A2E00`
- 发现 4 个相机
- 运行态 DepthAI USB：`SUPER_PLUS`
- `lsusb -t`：OAK 位于 Bus 02，`10000M`（达到并超过 USB 3.x 门槛）
- 驱动容器：`omninxt/oak4p-noetic-arm64:jp512`
- 四路均为真实彩色 `bgr8 1280×720`，约 20 Hz
- 同一组 A/B/C/D 消息时间戳一致
- 独立话题：
  - `/oak_ffc_4p/CAM_A`
  - `/oak_ffc_4p/CAM_B`
  - `/oak_ffc_4p/CAM_C`
  - `/oak_ffc_4p/CAM_D`
- 拼接话题：`/oak_ffc_4p/assemble_image`
- 拼接契约：`bgr8 5120×720`，约 20 Hz
- 源码和保存图像均确认拼接顺序为 `CAM_A | CAM_B | CAM_C | CAM_D`

证据：

- `reports/oak4p_frame_metadata.json`
- `reports/oak4p_assembled_metadata.json`
- `runtime/oak4p_frames/assemble_CAM_A_B_C_D.png`
- `runtime/oak4p_smoke_20260725_115633.bag`

`runtime/camera_mapping.yaml` 的物理方位保持 `UNASSIGNED`，等待最终刚性安装后由用户确认。

## 4. CUDA OpenCV 与 OmniDepth 构建

OpenCV 4.6 从官方源码为 SM 8.7 构建，启用：

```text
calib3d ccalib core cudaarithm cudaimgproc cudastereo cudawarping
cudev features2d flann highgui imgcodecs imgproc photo video videoio
```

构建信息见 `reports/opencv_build_information.txt`。最小 Catkin 构建只包含：

```text
camera_models
swarmtal_msgs
swarm_msgs
quadcam_depth_est
```

结果为 4/4 成功。`roslaunch --files quadcam_depth_est depth-node.launch` 可正确解析。
可执行文件动态链接已确认：

- `libnvinfer.so.8` / `libnvinfer_plugin.so.8`
- `/usr/local/lib/libopencv_*.so.406`
- CUDA 11.x runtime
- 无 amd64 库
- 无 TensorRT 10 库

JetPack 5.1.2 移植内容和所有权/内存安全修复详见
`reports/JETPACK512_PORT_NOTES.md`。

## 5. ONNX 与 Orin TensorRT engine

ONNX：

```text
models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt.onnx
SHA256=b589c3ff5e751603874de7d7ca0e88d06db4f9db0d87378d894da9e226ce369f
```

本机 TensorRT 解析得到的实际张量契约：

```text
input  input                         float32 [1,2,240,320]
output reference_output_disparity    float32 [1,240,320,1]
```

仓库历史 `.trt` 已可恢复地移动到
`backups/prebuilt_engines/`，没有加载。新 engine 由目标节点在本机 TensorRT 8.5.2
下以 FP16、1 GiB workspace 构建，约耗时 18 分钟：

```text
models/hitnet_series/hitnet_1x240x320_model_float16_quant_opt.trt
bytes=6958779
SHA256=4197ba2c3ac0696ad9e9e3483e2eac04ee5f7c6e8b39f5883ece1da660fe7142
```

engine 与本报告所列设备、JetPack、CUDA、TensorRT、ONNX 和 D2SLAM 工作树绑定；
上述任一关键项变化后必须删除运行目录中的 engine 并在目标机重建。

## 6. 离线软件链路

输入为本机真实 OAK 录制的 `/oak_ffc_4p/assemble_image` bag。显式使用
`quadcam_depth_PLACEHOLDER_NOT_CALIBRATED.yaml`，只验证接口。

结果：

- 节点正常加载本机 engine
- 接收 `bgr8 5120×720`
- HITNet 推理完成
- `/depth_estimation/pointcloud` 有输出
- PCD：`runtime/offline_20260725_142628/pointcloud.pcd`
- `POINTS=76799`
- 76,799 个 XYZ 均通过有限值检查
- 无 CUDA illegal memory access
- 无 TensorRT deserialize error
- 无 OOM

这不代表真实深度、尺度或方位正确。

## 7. 实时链路稳定性

使用真实 OAK 正常拼接模式和占位配置，在无 RViz 条件下连续运行 1800 秒。
测试从 15:47:06 至 16:17:34，开始和结束均重新测量话题：

| 指标 | 开始 | 结束 |
|---|---:|---:|
| `/oak_ffc_4p/assemble_image` | 约 20.0 Hz | 20.002 Hz |
| `/depth_estimation/pointcloud` | 约 10.0 Hz | 10.001 Hz |
| 点云 seq | 454 | 18303 |

898 个 tegrastats 样本汇总：

```text
RAM          3315..5955 MiB / 7472 MiB
zram swap    2063..2276 MiB / 3736 MiB
CPU temp     54.7..60.2 °C
GPU temp     53.7..59.6 °C
GR3D         94.4% average, 98% maximum
nvpmodel     15W
```

- 测试前后输入编码和尺寸保持 `bgr8 5120×720`
- 相机容器和深度容器未退出
- 点云持续增长，无“进程存活但话题停止”
- 内核日志 USB reset/disconnect/xHCI error 匹配数：0
- OAK 日志 DepthAI/X_LINK/device lost 匹配数：0
- ROS 日志无 CUDA illegal memory、TensorRT deserialize、OOM 匹配
- 未观察到温度导致的降频

OAK 日志中的一次 `roscore cannot run as another roscore/master is already running`
来自启动脚本发现 host network 上已有健康 ROS master；驱动随后正常连接，不是相机掉线。

证据：

- `logs/50_live_pipeline_20260725_154648.log`
- `logs/50_live_tegrastats_20260725_154648.log`
- `reports/live_30min_resource_summary.txt`

## 8. 标定工具准备

quarterKalibr 官方 Notebook 引用的 Docker Hub 镜像经 manifest 检查为 amd64，
未拉取、未运行。已从官方 tartancalib/Kalibr 源码在 ARM64 Noetic 容器中编译：

```text
37/37 Catkin packages succeeded
rosrun kalibr tartan_calibrate --help                 PASS
rosrun kalibr kalibr_calibrate_imu_camera --help     PASS
```

首次构建发现并补齐官方 Dockerfile中已有的 `libglew-dev` 依赖；未污染宿主 Python。

已创建：

- `runtime/calibration/{bags,results,templates}`
- `calibration_ws`
- `70_record_camera_calib_bag.sh`
- `71_record_cam_imu_calib_bag.sh`
- `72_check_calib_bag.sh`
- 必须由用户填写的标定板与输入模板
- `reports/CALIBRATION_READY.md`

当前没有已验证的 `sensor_msgs/Imu` 话题。相机‑IMU录包脚本默认拒绝执行，只有显式提供且通过类型检查的 `IMU_TOPIC` 才会运行。

## 9. 可重复运行入口

```text
00_preflight.sh
10_prepare_docker.sh
20_start_oak4p_driver.sh
21_check_oak4p_driver.sh
22_record_oak4p_smoke_bag.sh
29_stop_oak4p_driver.sh
30_build_omnidepth_image.sh
31_start_omnidepth_container.sh
32_build_omnidepth_workspace.sh
40_offline_smoke_test.sh
50_live_pipeline_smoke_test.sh
60_collect_runtime_report.sh
70_record_camera_calib_bag.sh
71_record_cam_imu_calib_bag.sh
72_check_calib_bag.sh
90_stop_all.sh
```

另有 `ros1_offline_feeder.py` 与 `pointcloud_saver.py`。脚本不删除用户数据，
日志写入 `logs/`，运行数据写入 `runtime/`。

## 10. 未完成且不能猜测的项目

1. CAM_A/B/C/D 的真实 front/right/rear/left 映射。
2. 标定板类型、行列、tag/方格尺寸、间距、实际打印尺寸和缩放。
3. 最终 IMU 来源、安装方向、话题、频率、噪声参数和时间基准。
4. 四相机真实内参、相邻外参、相机‑IMU时空外参和虚拟双目 Q 矩阵。
5. 使用真实标定后的尺度、方向、扇区接缝和几何精度验收。

到此只完成标定前部署。除非用户明确确认“现在开始正式标定”且物理信息齐全，
不得把占位 YAML 改名为正式配置或自动计算最终参数。
