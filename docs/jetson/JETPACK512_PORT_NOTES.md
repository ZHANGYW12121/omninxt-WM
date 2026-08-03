# D2SLAM OmniDepth — JetPack 5.1.2 移植说明

目标平台：Jetson Orin Nano 8GB、L4T R35.4.1、CUDA 11.4、cuDNN 8.6、TensorRT 8.5.2、SM 8.7、ROS1 Noetic、ARM64。

官方基线为 D2SLAM `pr_fix_main` 的 `3a4b8071f6f9c40a151c7685aa738b717ce8916a`；本机工作分支为
`jetson-orin-nano-jp512`。本文件描述工作树中的未提交兼容改动，完整差异见
`d2slam_jetpack512_port.patch`。

## 构建边界

- 仅构建独立 `quadcam_depth_est` 所需的 4 个包，不引入完整 VIO/PGO 图。
- 显式使用 `/usr/local/lib/cmake/opencv4`，确保 C++ 节点链接本机 CUDA OpenCV 4.6。
- OpenCV 启用 `calib3d`、`ccalib`、CUDA image processing/warping 等必要模块，目标架构 `8.7`。
- 使用官方 LCM 1.4.0；Ubuntu Focal 的 LCM 1.3.1 生成头文件与该源码不兼容。
- 不安装 PyTorch，不复制外部 CUDA、TensorRT、OpenCV 动态库或预生成 engine。

## 源码兼容与正确性修复

- 将相机 YAML 解析器从完整 `d2frontend` 依赖中最小独立出来，保留官方
  `D2FrontendParams` 的相机模型和外参解释约定。
- 为 TensorRT 对象、执行上下文、CUDA stream 和 buffer 建立明确所有权与析构顺序，消除重复析构和 engine 数据泄漏。
- ONNX 解析后验证真实张量契约：输入 `float32 [1,2,240,320]`，输出
  `float32 [1,240,320,1]`；静态模型不强行改为动态。
- 从官方 ONNX 使用 TensorRT 8.5.2 FP16 构建序列化 plan；只有动态输入时才添加 MIN/OPT/MAX profile。
- 校验 binding 名称、类型、维度和 buffer 字节数，推理同步后才读取输出。
- 对所有 CUDA `GpuMat` 到 CPU `Mat` 的路径显式 `download()`。
- 生产/消费图像及输出使用独立数据，避免浅拷贝跨线程竞争；首次输出就绪前不发布未初始化点云。
- 实时输入必须为 `bgr8 5120×720`，否则明确拒绝。

## 几何边界

未修改物理相机方向、标定内外参或虚拟双目几何。烟测配置
`quadcam_depth_PLACEHOLDER_NOT_CALIBRATED.yaml` 显式引用官方示例，仅验证接口与稳定性。
