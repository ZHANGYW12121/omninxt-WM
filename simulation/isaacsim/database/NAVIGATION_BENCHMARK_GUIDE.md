# EGO-Planner / NavRL / NavRL 无安全层 / DPMPC Warehouse 对比实验

入口程序：

`simulation/isaacsim/database/run_navigation_benchmark.sh`（从仓库根目录运行）。

## 正式运行

同时运行四种对比算法，种子 1～30，每个种子一次：

```bash
cd omninxt-WM

./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm all \
  --seeds 1-30 \
  --repeats 1
```

四种算法分别是：

- `ego`：EGO-Planner；
- `navrl`：NavRL 网络/直飞分支 + 动态 VO Safety Shield + 静态 LiDAR Guard；
- `navrl_no_shield`：完全相同的 NavRL 权重、输入、网络封装和直飞分支，但绕过
  动态 VO Safety Shield 与静态 LiDAR Guard。
- `dpmpc`：官方 DPMPC 的 minimum-snap 静态轨迹与 ACADO
  chance-constrained MPC；动态行人使用 Isaac GT 位置、速度和官方尺寸/方差模型，
  静态地图使用 Isaac GT 点云初始化 OctoMap。

这里的静态 LiDAR Guard 是当前 Isaac 接入中对官方 `safe_action` 静态激光点
约束的轻量实现：它在墙面或货架进入近距离时，限制最终速度朝障碍物法向的接近
分量，但不负责路径规划。`navrl_no_shield` 会同时跳过它和动态 VO Shield。

只对比带安全层和不带安全层的 NavRL：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm navrl_pair \
  --seeds 1-30 \
  --repeats 1
```

原来的 `--algorithm both` 仍表示只运行 EGO-Planner 与带安全层的 NavRL。

只运行 EGO-Planner：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm ego \
  --seeds 1-30 \
  --repeats 1
```

只运行 NavRL：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm navrl \
  --seeds 1-30 \
  --repeats 1
```

只运行不带安全层的纯 NavRL：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm navrl_no_shield \
  --seeds 1-30 \
  --repeats 1
```

只运行 DPMPC：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm dpmpc \
  --seeds 1-30 \
  --repeats 1
```

先用一个种子调试并显示 Isaac 窗口：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm navrl \
  --seeds 1 \
  --repeats 1 \
  --no-headless
```

默认使用 PX4/MAVSDK。仅调试本地 Pegasus 控制链路时可增加：

```text
--no-px4
```

中断后继续同一个实验目录：

```bash
./simulation/isaacsim/database/run_navigation_benchmark.sh \
  --algorithm all \
  --seeds 1-30 \
  --repeats 1 \
  --output-root /指定/结果目录 \
  --resume
```

## 调度和隔离

一个命令会自动完成指定的全部任务。每个 `算法 × seed × repeat` 都会启动
全新的 Isaac、PX4 和算法进程：

- 不继承上一轮无人机速度；
- 不继承上一轮 EGO 占据地图或轨迹；
- 不继承上一轮 NavRL 策略状态或 DPMPC 静态/MPC 轨迹；
- seed 相同时，所有被选算法得到相同的出生点、目标点、人数和行人模板。

EGO 与 DPMPC 实验各自的 ROS1 Docker sidecar 都由启动器自动启动和关闭，
不需要另开终端。DPMPC 使用镜像 `dpmpc-planner-isaac:noetic`；构建与上游版本
记录位于仓库的 `navigation/dpmpc/`。
算法先后顺序按 seed 循环轮换，避免总由同一个算法占用机器冷启动阶段。
每轮还有独立墙钟看门狗；Isaac/ROS/PX4 卡死后会清理整棵进程并继续下一轮。
`process_error`、`wall_timeout` 或 `gpu_failure` 默认自动重试一次；基础设施
失败会单独报告，不混入算法有效任务的成功率分母。

每个真正要启动的任务前，启动器会先等待默认 10 秒冷却，再通过
`nvidia-smi` 检查 GPU 0。检查内容包括驱动响应、GPU 标识、温度、显存和利用
率，而且温度、显存、利用率必须能解析为有效数值；`GPU requires reset`、
`fallen off the bus`、`unknown error` 等状态会明确判为不健康。查询失败、超时
或返回不完整时，启动器默认每 5 秒继续监听；GPU 恢复正常后自动启动尚未开始
的当前任务：

- 不会把 GPU 故障记作算法碰撞或算法失败；
- GPU 不健康时不会继续重试启动 Isaac；
- GPU 不健康期间不会启动任何新 Isaac/PX4 进程；
- 已完成的 JSON 和汇总文件都会保留；
- 检查记录保存在实验目录的 `gpu_health.jsonl`；
- 可以按 `Ctrl+C` 安全停止，之后用原命令、原输出目录和 `--resume` 继续。

相关运行参数：

```text
--gpu-index 0
--gpu-health-attempts 0
--gpu-health-timeout-sec 10
--gpu-health-retry-sec 5
--inter-trial-cooldown-sec 10
--gpu-monitor-interval-sec 5
--gpu-monitor-failures 1
```

`--gpu-health-attempts 0` 表示无限等待。若希望连续失败指定次数后退出，可将它
改成正整数，例如 `--gpu-health-attempts 12` 表示最多检查 12 次。

这些参数属于运行基础设施保护，不改变 seed、地图或算法，因此也不影响已有
实验目录的 `--resume` 配置匹配。

Isaac运行期间也会每5秒执行一次有超时限制的GPU健康查询。任意一次运行期查询
失败时：

1. 当前任务会被标记为 `gpu_failure` 基础设施故障；
2. 当前Isaac、PX4和算法进程树会被终止；
3. 这条记录不会进入算法成功率、碰撞率等有效实验统计；
4. 启动器回到任务前GPU等待状态；
5. GPU恢复后自动重新运行同一条任务。

### 通用活动实验自动恢复服务

任何非 `--dry-run` 的批量实验都会将完整参数登记到：

`simulation/isaacsim/database/navigation_benchmark_autoresume_state.json`。

登记内容包括算法、任意seed列表或区间、重复次数、结果目录、PX4/headless设置
以及GPU保护参数。保存的恢复命令总是显式包含实际 `output-root` 和 `--resume`。
因此它不局限于31～60，例如 `1-30`、`31-60`、`1,5,8,20-40` 都使用同一套
恢复机制。

模板位于
`simulation/isaacsim/database/systemd/navigation-benchmark-autoresume.service`。
通过 `tools/install_navigation_benchmark_service.sh` 安装，脚本会把当前仓库
绝对路径写入用户级 systemd 配置；仓库模板本身不绑定机器路径。

```bash
cd omninxt-WM
./tools/install_navigation_benchmark_service.sh
```

桌面会话崩溃导致启动器退出时，systemd会在30秒后重新启动；系统重启且用户
linger已启用时，会读取唯一的“活动实验”，等待Docker和GPU恢复，然后跳过已有
有效结果，从第一个缺失或基础设施失败任务继续。全部任务完成后状态变为
`completed`，以后重启服务只保持空闲，不会重跑。

启动器还持有全局文件锁
`navigation_benchmark_launcher.lock`。如果systemd已经恢复了一份实验，再误从
终端启动第二份，后者会立即退出，不会并发创建两套Isaac/PX4进程。

普通启动命令会自动登记。如果只想登记任务、交给systemd启动而不在当前终端
直接运行，可给原命令增加：

```text
--register-autoresume-only
```

常用管理命令：

```bash
systemctl --user start navigation-benchmark-autoresume.service
systemctl --user status navigation-benchmark-autoresume.service
systemctl --user stop navigation-benchmark-autoresume.service
journalctl --user -u navigation-benchmark-autoresume.service -f
```

启动器标准输出同时保存在：

`simulation/isaacsim/database/navigation_benchmark_autoresume_service.log`。

批量实验默认 headless，并关闭与这些 GT 感知基线无关的 OmniNxt 相机和深度
导出；最新无人机可视模型仍会加载。EGO 使用原来的 Isaac GT 点云，两个
NavRL 变体使用完全相同的 Isaac GT raycast LiDAR，DPMPC 使用静态 GT 点云与
动态行人 GT 状态。两个 NavRL 变体的唯一差异是最终速度是否经过动态 VO
Safety Shield 和静态 LiDAR Guard。

## 统一终止条件

成功：

```text
无人机中心到随机三维目标点的欧氏距离 <= 1.00 m
```

失败：

- 进入导航阶段后首次接触任何行人；
- 进入导航阶段后首次接触墙、货架等环境碰撞体；
- 从发出起飞请求开始累计 120 s 仿真时间仍未成功；
- 控制器、PX4或进程异常。

起飞阶段的地面接触不计碰撞。同一更新区间既碰撞又到达时，碰撞优先。

## 指标

评估器默认以 10 Hz 仿真频率运行，可通过 `--eval-hz` 修改，但程序强制要求：

```text
0 < eval_hz <= 25
```

持续计算：

- 三维路径长度；
- 三维速度的时间平均；
- 三维目标距离；
- 人体表面最小间距；
- 人体侵犯时间比例，进入阈值为 `clearance < 0.50 m`、退出阈值为
  `clearance >= 0.60 m`；
- 每个任务的人体侵犯事件次数；
- 发出起飞请求时的实际三维位置到目标点的三维直线参考长度；
- `reference_length / actual_length` 效率；
- 观测准备完成到最终速度输出的墙钟延迟。

人体表面由骨架关节球和相邻骨架胶囊近似；无人机外边缘使用当前物理安全半径
0.32 m。评估器在相邻采样间进行扫掠插值，`clearance <= 0` 直接补充判定为
人体碰撞，避免只靠关节球 PhysX 接触时漏掉躯干、上臂和大腿中段。该几何仅
用于评估，不反馈给 EGO-Planner 或 NavRL。静态碰撞仍由 PhysX 接触事件判定。

NavRL 延迟同时输出三种口径：

- `decision_latency_p95_ms`：实际整套控制链路，包含网络和无障碍直飞分支；
- `decision_latency_policy_p95_ms`：只统计真正调用 NavRL 网络的周期；
- `decision_latency_direct_goal_p95_ms`：只统计无障碍直接朝目标飞行的周期。

EGO 只统计每条新轨迹的第一条最终速度命令，避免周期发布同一轨迹造成重复样本。
由于官方异步地图/优化链路没有传递点云序号，EGO 结果会标注
`decision_latency_causal_pairing=false`，表示点云与轨迹使用最近可配对观测，
不是严格逐帧因果追踪。

DPMPC 在同一个 UDP 观测中原子发送无人机状态和全部行人状态，并把
`observation_id` 传过 ACADO 求解再返回，因此延迟是严格因果配对；它包含官方
MPC 求解和将官方 PX4 位置 setpoint 接到当前 MAVSDK velocity-offboard 所必需的
位置跟踪转换，不包含 Isaac 渲染时间。

任务用时、平均速度、路径长度、效率和 P95 延迟只汇总成功任务。人体侵犯比例
同时输出全部任务平均值和成功任务平均值。

## 输出

默认结果目录：

```text
navigation_benchmark_results/benchmark_YYYYMMDD_HHMMSS/
```

内容：

```text
experiment_config.json   实验参数
runs/*.json              每次任务的完整原始指标
logs/*.log               Isaac/PX4/算法终端日志
logs/*_ego_sidecar.log   EGO ROS1 sidecar日志
logs/*_dpmpc_sidecar.log DPMPC ROS1/ACADO sidecar日志
stale_results/*.json     被新实验或 --resume 重试隔离的旧/异常结果
runs.csv                 全部任务表格
summary.json             按算法汇总的机器可读结果
comparison.md            所选算法汇总简表
```

每个 run 还保存完整场景指纹；`summary.json` 会检查相同 seed/repeat 下所有
被选算法是否真正使用同一场景。人体距离不可用时不会静默当作零侵犯，而会额外
报告有效覆盖时长、覆盖比例和仅在有效覆盖上的侵犯比例。

关键可选参数：

```text
--seeds 1-30
--repeats 3
--eval-hz 10
--timeout-sec 120
--wall-timeout-sec 2100
--infrastructure-retries 1
--goal-radius-m 1.00
--crowd-count 20
--output-root /path/to/results
--resume
--no-headless
--no-px4
```
