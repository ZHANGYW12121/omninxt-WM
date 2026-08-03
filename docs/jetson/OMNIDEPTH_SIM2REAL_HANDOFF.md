# OmniNxt / OmniDepth 真机几何与三维骨架仿真交付说明

日期：2026-08-02  
真机：OAK-FFC-4P，MXID `19443010E16C6A2E00`  
用途：在外星人电脑的仿真中复现真机的相机几何、深度定义和三维骨架算法，减少 Sim2Real 差异。

当前运行时明确不使用飞控数据：不连接PX4、不启动MAVROS、不订阅
`/mavros/*`。稠密深度、左右骨架、三角化和Kalman均只依赖相机数据。
标定文件中的 `imu` 字样是Kalibr遗留的静态参考系名称；本文运行时将
这个固定参考系作为无人机机体系 `base_link`，并不代表需要实时IMU。
`evidence/OMNIDEPTH_FORMAL_CALIBRATION_20260730.md`中出现的PX4/MAVROS仅是
生成历史相机—机体静态外参时的离线取证，不能照搬到当前运行链路。

## 1. 权威文件与使用边界

本交付包中的权威机器可读几何文件是：

```text
calibration/fisheye_cams.yaml
calibration/stereo_calib_0_1_240_320.yaml
calibration/stereo_calib_1_2_240_320.yaml
calibration/stereo_calib_2_3_240_320.yaml
calibration/stereo_calib_3_0_240_320.yaml
derived/sim_geometry.yaml
```

`sim_geometry.yaml`由上述五份正式YAML通过OpenCV `stereoRectify`生成，已经把容易混淆的矩阵方向、`R1^T`和四组虚拟相机位姿展开。所有矩阵统一命名为：

```text
T_target_source：把source坐标中的齐次点变换到target坐标。
```

不要使用包外旧示例、占位YAML或TensorRT engine代替这些文件。TensorRT engine与GPU架构、TensorRT版本绑定，外星人电脑应从包中的ONNX重新构建。

当前绝对尺度仍有一个已知来源：AprilGrid使用名义 `tagSize=0.088 m`，用户确认成品尺寸接近名义值但未进行高精度多点测量。若以后测得真实Tag边长为 `s_real`，平移、基线、深度和三维坐标的尺度应乘以：

```text
s_real / 0.088
```

## 2. 物理相机和坐标系

从无人机顶部向下看，机头朝前：

```text
CAM_A = 右前 FRONT_RIGHT
CAM_B = 右后 REAR_RIGHT
CAM_C = 左后 REAR_LEFT
CAM_D = 左前 FRONT_LEFT

顺时针：A -> B -> C -> D -> A
```

ROS机体坐标系 `base_link` 使用FLU：

```text
+X：机头/前
+Y：左
+Z：上
```

OpenCV光学坐标系为：

```text
+X：图像右
+Y：图像下
+Z：镜头前方
```

图像拼接及逻辑顺序固定为：

```text
CAM_A | CAM_B | CAM_C | CAM_D
```

真机没有执行180°旋转：`enable_upside_down=false`。仿真图像也不要额外上下或左右翻转。

## 3. 原始四鱼眼内参与IMU外参

每个原始相机分辨率均为1280×720，模型为Mei统一全向模型（Kalibr `omni`）加Radtan畸变。

| 相机 | xi | fx | fy | cx | cy | k1 | k2 | p1 | p2 | 时间偏移s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAM_A | 0.581376 | 398.4169 | 398.9234 | 619.4287 | 327.2006 | -0.211945 | 0.027224 | 0.000826 | -0.000602 | -0.071919 |
| CAM_B | 0.558480 | 391.3027 | 391.6835 | 640.0229 | 329.4148 | -0.209447 | 0.027054 | -0.000071 | 0.000318 | -0.071278 |
| CAM_C | 0.589170 | 399.8916 | 400.4918 | 604.2184 | 334.7839 | -0.210457 | 0.026676 | 0.001403 | 0.000880 | -0.070867 |
| CAM_D | 1.871921 | 718.7847 | 719.8156 | 638.8267 | 320.0514 | 0.731360 | -1.538646 | -0.000818 | 0.005654 | -0.069538 |

Mei模型的归一化投影可写为：

```text
d = sqrt(X² + Y² + Z²)
x = X / (Z + xi*d)
y = Y / (Z + xi*d)
r² = x² + y²

xd = x*(1 + k1*r² + k2*r⁴) + 2*p1*x*y + p2*(r² + 2*x²)
yd = y*(1 + k1*r² + k2*r⁴) + p1*(r² + 2*y²) + 2*p2*x*y

u = fx*xd + cx
v = fy*yd + cy
```

完整 `T_camera_imu` 和它的逆 `T_imu_camera` 在 `derived/sim_geometry.yaml` 中。Kalibr的 `T_cam_imu` 表示静态标定参考原点到相机；把相机点变到 `base_link` 时必须求逆，不能直接左乘原矩阵。矩阵键名为兼容历史文件而保留 `imu`，但运行时不读取任何IMU数据。

四个相机光心在 `base_link` 静态参考系中的位置约为：

| 相机 | X m | Y m | Z m |
|---|---:|---:|---:|
| CAM_A | +0.08421 | -0.06813 | +0.07497 |
| CAM_B | -0.06227 | -0.07179 | +0.07190 |
| CAM_C | -0.07024 | +0.08481 | +0.07495 |
| CAM_D | +0.08097 | +0.08159 | +0.07548 |

## 4. 真机的两级校正流程

真机不是把两张1280×720鱼眼图直接输入HITNet。实际流程为两级变换。

### 4.1 鱼眼到虚拟针孔

`FisheyeUndist::UndistortPinhole2`使用：

```text
配置FOV：190°
虚拟针孔水平FOV：190° - 90° = 100°
输出：320×240，灰度
虚拟理想焦距：320 / (2*tan(50°))
```

每个鱼眼生成两个绕相机光学Y轴旋转的虚拟视图：

```text
idx 0：-45°
idx 1：+45°
```

运行时四组输入为：

| 双目ID | 物理侧 | 左图 | 右图 |
|---:|---|---|---|
| 0 | 右侧 | CAM_A idx1 | CAM_B idx0 |
| 1 | 后侧 | CAM_B idx1 | CAM_C idx0 |
| 2 | 左侧 | CAM_C idx1 | CAM_D idx0 |
| 3 | 前侧 | CAM_D idx1 | CAM_A idx0 |

### 4.2 虚拟双目极线校正

第一级生成的虚拟图仍使用四份单独标定出的Pinhole+Radtan参数进行双目校正：

```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K0, D0, K1, D1, (320, 240), R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1,
)
```

四份校准文件不是通用模板，每组 `K/D/R/T` 都不同。最终校正相机摘要为：

| 双目 | 基线mm | rectified fx=fy | cx | cy | IMU平面方位角 |
|---|---:|---:|---:|---:|---:|
| A-B右 | 144.880 | 132.0927 | 161.8177 | 119.0899 | -89.87° |
| B-C后 | 144.732 | 133.1813 | 160.4294 | 120.3311 | -179.94° |
| C-D左 | 144.919 | 133.1024 | 159.3174 | 119.5942 | +90.70° |
| D-A前 | 145.195 | 134.1242 | 161.1440 | 118.7960 | -0.43° |

精确的 `R1/R2/P1/P2/Q/ROI/T_imu_rectleft/T_imu_rectright` 全部位于 `derived/sim_geometry.yaml`。

### 4.3 三维骨架专用的物理相机中心视图

稠密HITNet仍使用上面的四组320×240虚拟双目，不做改变。为避免一个人恰好位于
两个虚拟扇区接缝时，两边都只看到半个人，实时骨架前端另外从每个原始鱼眼生成：

```text
输出：416×320，mono8
水平FOV：120°
光轴：物理CAM_A/B/C/D各自中心光轴（不做±45°偏转）
单路调试话题：/depth_estimation/pose_anchor_0 ... pose_anchor_3
正式骨架输入：/depth_estimation/pose_anchor_mosaic（A/B在上、C/D在下，832×640）
时间戳：与同一帧四组rectified stereo严格相同
```

相邻物理相机光轴相差约90°，所以中心视图约有30°重叠。该视图只负责完整人体
检测与单目二维姿态；三维坐标仍严格来自四组正式双目标定的P1/P2三角化。C++只有
在骨架节点订阅anchor话题时才执行这四次remap，普通OmniDepth运行不增加这部分开销。
四组左右rectified图同时按`AB、BC、CD、DA`四行、每行`left|right`打包到
`/depth_estimation/pose_stereo_mosaic`（640×960）。Python正式运行只同步这两个
mosaic，而不是同时接收12个图像回调；数据内容和分辨率没有减少。

## 5. HITNet视差和深度定义

真机使用包中的：

```text
models/hitnet_1x240x320_model_float16_quant_opt.onnx
```

张量契约：

```text
input name: input
input dtype/shape: float32 [1,2,240,320]
channel 0: left rectified grayscale / 255
channel 1: right rectified grayscale / 255

output name: reference_output_disparity
output dtype/shape: float32 [1,240,320,1]
output unit: pixel disparity, d = u_left - u_right
```

C++中先把左、右320×240图垂直拼成一个480×320连续 `CV_32F` 内存区；其内存布局恰好对应NCHW的两个通道。仿真Python实现应显式使用：

```python
tensor = np.stack([left, right], axis=0)[None].astype(np.float32) / 255.0
```

稠密三维使用：

```python
xyz_rectleft = cv2.reprojectImageTo3D(disparity, Q)
depth = xyz_rectleft[..., 2]
```

这里的深度是校正左目光学坐标的Z，不是相机到点的欧氏距离。仿真真值若是range，必须先变换成rectified optical Z再比较。

对于理想校正双目，近似关系为：

```text
Z = fx * baseline / disparity
```

HITNet目前不做正式稠密有效性过滤：`enable_disparity_filter=false`。为了公平比较网络，不要只在仿真侧先删除坏视差再计算指标；应同时报告原始结果和统一mask后的结果。

## 6. 从校正坐标变到无人机机体坐标

`cv2.reprojectImageTo3D`和稀疏三角化都首先产生rectified-left坐标。真机采用：

```text
p_base_link = T_imu_rectleft * p_rectleft

T_imu_rectleft
  = inverse(T_leftcamera_imu)
  * Ry(+45°)
  * R1^T
```

这里三个操作都不能省略：

1. `T_cam_imu`必须取逆；
2. 当前每组左图均为该鱼眼的idx1，即+45°虚拟视图；
3. `R1`把virtual-left旋转到rectified-left，因此反向恢复必须使用 `R1^T`。

历史上遗漏前两项或 `R1^T` 会让四组点云看起来都朝错方向或近似落在同一平面。包中 `T_imu_rectleft` 已经完成组合，仿真侧优先直接使用，避免再次手工解释矩阵。

## 7. 当前三维骨架方法

权威实现为 `python/102_live_stereo_pose.py`。

### 7.1 检测与二维姿态

- 四个416×320物理相机中心视图组成2×2 mosaic；所有方向始终进入同一个
  YOLOX-Nano输入。旧的四组校正左图仍供双目匹配和HITNet使用，不再作为主要人体
  检测画面。
- YOLOX-Nano输入416×416，TensorRT FP16；检测阈值0.28，NMS 0.45。
- 为保证Orin约10 Hz，YOLOX每10个骨架帧做一次全方向刷新；中间帧用二维骨架
  更新框。每3帧轮转检查一个没有活动框的物理相机，并允许一次416×320单相机救援
  检测。这不是只开启有人扇区，四个方向和四路HITNet始终参与。
- 相邻anchor中的框先将中心像素射线变换到`base_link`，按方位、俯仰和尺度去重，
  保留同一个人最靠近画面中心的一份。因此同一个人跨两个相机时通常只执行一次
  RTMPose。
- RTMPose-S输入192×256，只在保留的完整物理相机中心框上运行一次；不再为右图
  额外运行一次姿态网络。
- COCO Body-17顺序：nose、左右eye、左右ear、左右shoulder、左右elbow、左右wrist、左右hip、左右knee、左右ankle。
- 关键点内部可见阈值0.22；至少3个可见点，并要求最高8个置信度均值满足人体阈值。
  较宽松的截断准入由后续双目和三维一致性约束兜底，用来提高远处/边缘人体召回。

### 7.2 从anchor关节投影到相邻双目

- 每个物理相机恰好属于两组相邻双目。例如CAM_C同时属于B-C后侧和C-D左侧。
- anchor像素先通过120°针孔射线、对应相机`T_cam_imu`、该组±45°虚拟旋转和
  `R1/R2`投影到两组正式320×240rectified图。
- 对每个投影关节，以11×11模板在另一张图的同一极线附近搜索，视差范围
  0.75–96 px，纵向搜索约±3 px。
- 使用归一化相关系数NCC，并用一维抛物线拟合得到亚像素x。
- 两组相邻双目都得到候选时，按NCC、重投影误差、传播深度不确定度和HITNet一致性
  选择质量较高者。全过程不需要第二张图再次做人检/姿态推理。

### 7.3 三角化

```python
X_h = cv2.triangulatePoints(P1, P2, left_uv, right_uv)
X_rect = X_h[:3] / X_h[3]
X_imu = T_imu_rectleft @ [X_rect, 1]
```

当前边界：

```text
视差：0.75–96 px
Z深度：0.25–8.0 m
最大重投影误差：3 px
```

关节深度误差近似传播为：

```text
sigma_Z ≈ Z² / (fx*baseline) * sigma_disparity
```

同一人体至少5个可靠关节时，再用中位视差保护单点：允许区间为 `0.35×median` 到 `3×median`。这是为了删除“近零视差把一个肩点放到十米外”的明显错配，并非稠密点云滤波。

anchor新路径还对肩/髋中位中心使用非常宽松的米制人体范围，只拒绝明显不可能的
单关节匹配（例如身体约0.8 m、某个眼睛却落在4 m）。被拒绝的原值保存在状态JSON
的`rejected_xyz_imu_m`中，便于调试。这仍是稀疏关节对应有效性检查，不会改变或过滤
HITNet稠密深度输出。

### 7.4 HITNet校验和回退

- HITNet四组始终运行，实机并发时约4–5 Hz。
- 在左关节周围5×5窗口取有效深度中位数和MAD。
- 稀疏深度与HITNet之差不超过 `max(0.35 m, 0.25*Z)` 时标为一致。
- 稀疏三角化缺失时，允许使用不超过350 ms的HITNet深度回退。
- HITNet回退测量的最小标准差设为0.35 m，权重显著低于稀疏三角化。

### 7.5 跨相机同人融合

- 在昂贵的RTMPose前，重叠anchor框用机体系视线、俯仰和人体框尺度做第一层去重。
- 若遮挡/检测波动仍让同一个人保留多份3D骨架，后端用肩髋中心距离和对应有效关节
  的中位3D距离聚类。
- 同一关节按姿态分数和三角化不确定度加权融合，输出一个全局`person_id`。
- 当前方法不使用ReID外观编码。多人发生完全同方位遮挡时，应在后端再加入轻量外观
  embedding；普通相邻相机重叠不需要为此增加前端GPU负载。

### 7.6 纯机体系Kalman（无飞控输入）

- 使用图像原始时间戳同步八张校正图；只接受四组左右时间戳完全相同的集合。
- 每个关节的常速度Kalman直接在 `base_link` 中预测和更新。
- 不读取姿态、位置、IMU、串口或任何飞控消息，不做世界系旋转/平移补偿。
- 仿真基线也应只使用视觉三维测量和图像时间戳；若另加世界系补偿，必须标为增强实验，不能与当前真机结果混为一组。

注意：YAML中的约-70 ms `timeshift_cam_imu` 是历史相机—IMU标定字段，当前纯视觉运行时不使用它。

## 8. 仿真侧两种推荐复现方案

### 方案A：完整传感器链，最接近真机

1. 按 `raw_cameras`中的 `T_imu_camera`布置四个1280×720 Mei鱼眼相机。
2. 使用完整 `xi/fx/fy/cx/cy/k1/k2/p1/p2` 渲染。
3. 输出顺序固定A|B|C|D，图像方向不翻转，同一时刻曝光。
4. 直接编译包中C++参考源，执行与真机相同的±45°虚拟展开和第二级双目校正。
5. 运行相同HITNet/YOLOX/RTMPose ONNX。

该方案能模拟鱼眼采样、插值和边缘信息损失，是评估Sim2Real最可信的方式。

### 方案B：直接渲染最终校正双目，最容易对齐几何

每组直接创建两个无畸变320×240针孔相机：

- 左相机位姿：`T_imu_rectleft`
- 右相机位姿：`T_imu_rectright`
- 左内参：`P1[:3,:3]`
- 右内参：`P2[:3,:3]`
- 图像畸变：0

这样渲染的结果就是HITNet和骨架程序看到的左右校正图，可绕过仿真器是否支持Mei模型的问题。它不会复现原始鱼眼到虚拟视图的重采样损失，因此适合先验证深度/骨架数学链，再与方案A比较。

## 9. 仿真真值建议

每个校正双目同时导出以下真值：

```text
left/right grayscale image
rectified-left optical Z
pixel disparity = fx*baseline/Z（含左右可见性）
17个关节在left/right图像中的像素
17个关节在rectified-left和IMU坐标中的XYZ
遮挡/截断标记
```

建议至少计算四层指标：

1. 图像层：灰度均值、对比度、梯度、噪声、运动模糊和曝光分布；
2. 视差层：EPE、bad-1、bad-3，分距离和纹理区域统计；
3. 稀疏骨架层：左右像素误差、三角化XYZ误差、PCK/MPJPE；
4. 系统层：每方向召回率、四扇区重叠区一致性、端到端频率和延迟。

真机当前采集特征也应进入domain randomization：

```text
原始彩色：1280×720 @ 20 Hz
运行处理：320×240灰度 @ 10 Hz
当前曝光：手动5000 us
当前ISO：200
当前白平衡：自动
无180°旋转
无光度/暗角补偿（enable_photometric_calib=false）
```

不要只随机RGB颜色；HITNet实际看到灰度，应优先随机化左右曝光差、gamma、暗角、传感器噪声、局部反光、运动模糊和鱼眼重映射模糊。

## 10. 常见错误检查表

- [ ] 把A/B/C/D当成前/右/后/左，而不是右前/右后/左后/左前；
- [ ] 交换了左右图，导致正视差变负；
- [ ] 忽略idx1=+45°、idx0=-45°；
- [ ] 只做鱼眼展开，没有使用每组虚拟双目标定再做 `stereoRectify`；
- [ ] 用 `T_cam_imu`直接把相机点变到IMU，而没有求逆；
- [ ] 遗漏 `R1^T`；
- [ ] 把rectified Z当成欧氏range；
- [ ] 给HITNet输入0–255或RGB，而不是真机的0–1双通道灰度；
- [ ] 复用Orin生成的TensorRT engine；
- [ ] 仿真只保留完美无遮挡像素，导致比真机任务简单很多；
- [ ] 分别采集左右图而没有严格同步；
- [ ] 在仿真中提前应用过滤，却与真机原始HITNet输出直接比较。

## 11. 代码和版本

官方源码基线：

```text
D2SLAM commit: 3a4b8071f6f9c40a151c7685aa738b717ce8916a
OmniNxt commit: 79ce1d4b69a5a8c861417716adaac00c2bc6f594
rtmlib remote: https://github.com/Tau-J/rtmlib.git
```

真机运行环境：JetPack 5.1.2、CUDA 11.4、TensorRT 8.5.2；OmniDepth C++链接自编译OpenCV 4.6 CUDA/ccalib。`sim_geometry.yaml`导出器使用宿主OpenCV 4.5.4的相同 `stereoRectify` API。

模型SHA256：

```text
HITNet ONNX     b589c3ff5e751603874de7d7ca0e88d06db4f9db0d87378d894da9e226ce369f
YOLOX-Nano ONNX c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
RTMPose-S ONNX  9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d
```

Orin TensorRT engine仅作真机追溯，不应复制到外星人电脑运行：

```text
HITNet TRT      4197ba2c3ac0696ad9e9e3483e2eac04ee5f7c6e8b39f5883ece1da660fe7142
YOLOX-Nano TRT  6550c03eb90fbdc4518a04714f915339c13da4028c9b63ec7f8cb331f287c17c
RTMPose-S TRT   dda74bdf3b21c8aac4ed078a453710345e4db2f6024f71a73abbb26b17d67e6a
```

## 12. 在外星人电脑上首先执行的验证

```bash
sha256sum -c SHA256SUMS
python3 python/104_export_sim_geometry.py \
  --config-dir calibration \
  --output /tmp/sim_geometry_regenerated.yaml
```

重新生成文件应与包内 `derived/sim_geometry.yaml` 在数值精度内一致。然后用 `python/105_sim_reference_geometry.py`或直接导入其中的函数，对仿真左右17点做三角化：

```bash
python3 python/105_sim_reference_geometry.py \
  --geometry derived/sim_geometry.yaml \
  --pair A_B_RIGHT \
  --left-json left_coco17.json \
  --right-json right_coco17.json
```

第一阶段不要急着训练网络。先用仿真真值左右关节和真值视差通过本参考几何，确认四组方向、尺度、Z定义和IMU变换完全正确；然后固定几何，只替换为HITNet/RTMPose预测，才能清楚区分“几何错误”和“网络域差异”。
