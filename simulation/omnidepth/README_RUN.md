# Isaac Sim -> Omni-Depth 使用说明

> 2026-08-02 的实机四鱼眼、深度和三维骨架同步配置请优先阅读
> `SIM2REAL_OMNIDEPTH_SYNC_20260802.md`，正式一键入口为
> `./run_isaac_sync_live.sh`。下文保留的是单帧离线处理流程。

## 1. 启动深度容器

执行：`/home/neu/zyw/our_omni_depth/start_omnidepth.sh`。

脚本固定使用 GPU1，启动 ROS master 和 OmniDepth 节点，并等待图像订阅端就绪。
停止时执行：`/home/neu/zyw/our_omni_depth/stop_omnidepth.sh`。

## 2. 启动处理端

先在终端 A 执行：

`/home/neu/zyw/our_omni_depth/process_latest_frame.sh`

不带参数时，脚本会忽略已有的 `LATEST`，只等待启动后由 Isaac
写入的新同步帧，然后发布 100 次图像并保存结果。

如需明确地重新处理磁盘上已有的最新帧，使用：

`/home/neu/zyw/our_omni_depth/process_latest_frame.sh --latest`

## 3. 启动 Isaac

在终端 B 执行：

`/home/neu/zyw/our_omni_depth/run_isaac_capture.sh`

脚本固定 Isaac 使用 GPU0 并关闭 multi-GPU。无人机完成起飞、控制器进入
`navigate` 状态后，再稳定 30 个渲染帧并自动采集一次。看到
`[APP][OMNI-DEPTH] Exported synchronized frame` 即表示采集完成。

## 4. 输出位置

- 四张鱼眼和 5120x720 拼图：`shared/input/frame_XXXXXX/`
- 官方点云：`shared/output/frame_XXXXXX.pcd`
- PLY 和预览图：`shared/output/frame_XXXXXX.ply`、`shared/output/frame_XXXXXX_densemap.png`

## 5. 离线验证

可直接运行一帧端到端验证：

`docker run --rm --gpus 'device=1' --network host --entrypoint /bin/bash -v /home/neu/zyw/our_omni_depth/D2SLAM:/root/swarm_ws/src/D2SLAM/D2SLAM -v /home/neu/zyw/our_omni_depth/D2SLAM/config:/root/swarm_ws/src/D2SLAM/config -v /home/neu/zyw/our_omni_depth/D2SLAM/models:/root/swarm_ws/src/D2SLAM/models -v /home/neu/zyw/our_omni_depth/shared:/root/omninxt_shared -v /home/neu/zyw/our_omni_depth:/root/omninxt_offline:ro omnidepth:ada /root/omninxt_offline/run_offline_test.sh /root/omninxt_shared/input/frame_001025_20260717_173359`

首次构建如需代理，可向两个 Dockerfile 传入
`--build-arg OMNIDEPTH_BUILD_PROXY=http://172.17.0.1:7893`；最终镜像不会保留该代理。
