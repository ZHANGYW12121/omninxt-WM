# OmniNxt 正式标定就绪说明

状态：工具、目录、模板和录包入口已准备；**尚未采集正式标定包，也未计算或写入任何最终内外参。**

## 已就绪

- OAK-FFC-4P 设备：MXID `19443010E16C6A2E00`
- 图像输入：`/oak_ffc_4p/assemble_image`
- 消息契约：`sensor_msgs/Image`，`bgr8`，`5120×720`
- 拼接顺序：`CAM_A | CAM_B | CAM_C | CAM_D`
- ARM64 标定镜像：`omninxt/tartancalib-noetic-arm64:jp512`
- TartanCalib 和 Kalibr camera-IMU 命令已通过 `--help` 启动检查
- 录包目录：`/home/neu/OmniNxt/runtime/calibration/bags`
- 结果目录：`/home/neu/OmniNxt/runtime/calibration/results`
- 模板目录：`/home/neu/OmniNxt/runtime/calibration/templates`

quarterKalibr 的读取器会按 A、B、C、D 顺序从拼接消息拆分四帧。已移除工具中写死的 AprilGrid 尺寸；正式运行必须通过
`QUARTERKALIBR_TARGET_YAML` 指向用户确认的板参数文件。

## 正式标定前必须由用户确认

1. 四相机和 IMU 已完成最终刚性安装，此后不再移动。
2. `CAM_A/B/C/D` 的真实物理方位已于 2026-07-28 通过 ROS 四路独立话题逐镜头遮挡确认：
   `CAM_A=右前`、`CAM_B=右后`、`CAM_C=左后`、`CAM_D=左前`。映射保存在
   `/home/neu/OmniNxt/runtime/camera_mapping.yaml`；正式标定前仍需确认此后未交换排线、
   旋转镜头或调整机械结构。
3. 标定板类型（AprilGrid 或 checkerboard）。
4. `tagRows`、`tagCols`、`tagSize`、`tagSpacing`，或棋盘格行列和方格尺寸。
5. 实际打印尺寸，以及打印缩放确为 100%。
6. 最终 IMU 来源、安装方向和实际 `sensor_msgs/Imu` 话题。
7. IMU 频率、时间戳稳定性，以及相机与 IMU 是否处于同一 ROS time 基准。

当前 OAK 驱动没有已验证的 IMU 输出，所以相机‑IMU标定脚本默认拒绝运行。确认最终 IMU 后，显式传入
`IMU_TOPIC=/真实话题`。

## 建议的正式操作顺序

1. 固定最终机械结构，确认镜头洁净、曝光适合且 USB 3.x 链路稳定。
2. 启动正常拼接模式：

   ```bash
   cd /home/neu/OmniNxt
   sg docker -c './scripts/20_start_oak4p_driver.sh normal'
   sg docker -c './scripts/21_check_oak4p_driver.sh'
   ```

3. 填写并复核 `aprilgrid_USER_MUST_FILL.yaml`（或创建已测量的 checkerboard 文件）。
4. 完成 CAM_A、B、C、D 物理方位表，并冻结图像顺序。
5. 录制覆盖整幅视场、距离和姿态变化充分且无运动模糊的相机标定 bag：

   ```bash
   sg docker -c './scripts/70_record_camera_calib_bag.sh'
   ```

6. 用 `72_check_calib_bag.sh` 核对话题、消息数、时长和类型。
7. 按 quarterKalibr 工作流分别完成四个鱼眼内参，并求相邻相机对
   A‑B、B‑C、C‑D、D‑A 的外参。保留每一阶段中间结果和重投影报告。
8. 有可靠 IMU 后记录相机/IMU bag：

   ```bash
   IMU_TOPIC=/已验证的imu话题 \
     sg docker -c './scripts/71_record_cam_imu_calib_bag.sh'
   ```

9. 用 Kalibr 求相机‑IMU空间外参与时间偏移，并检查残差、时间偏移合理性和可重复性。
10. 根据真实内外参生成四组虚拟双目参数和 OmniDepth YAML。
11. 将最终文件放入新的用户配置目录；不要覆盖
    `quadcam_depth_PLACEHOLDER_NOT_CALIBRATED.yaml`。
12. 使用真实标定重新做离线和实时点云几何验收，包括尺度、方向、相邻扇区接缝和静态场景一致性。

## 边界

当前生成的点云只完成软件接口和稳定性验证。占位 YAML 引用了官方示例几何，**不描述这台 OAK 的真实结构，不能用于距离、尺度、方位或建图质量判断**。只有用户明确确认“现在开始正式标定”且上述物理信息齐全后，才进入最终标定。
