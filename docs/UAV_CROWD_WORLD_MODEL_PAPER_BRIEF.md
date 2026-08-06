# 无人机基于世界模型的人群穿梭方案说明

> 用途：把本文件交给 GPT，辅助整理论文结构、技术路线、实验设计和写作思路。
>
> 版本依据：2026-08-06 当前代码库。本文严格区分“已经实现”“正在采集”和“尚待闭环验证”，不能把设计目标写成实验结论。

## 1. 一句话概括

本项目面向动态人群中的目标导向无人机导航：在 Isaac Sim 中用现有 NavRL 控制器采集无人机状态、目标、人体三维骨架、执行动作和任务结果，训练一个 **Ego–Human 双分支因子化 Dreamer 世界模型**；模型分别建模无人机自身动力学和每个行人的时空运动，通过结构化稀疏注意力交换信息，再在潜空间中进行多步想象，由 Goal-conditioned Actor 输出无人机速度与偏航动作。

建议论文暂定主题：

> **Factorized Ego–Human World Models for Goal-Conditioned UAV Navigation in Dynamic Crowds**

中文可表述为：

> **面向动态人群穿梭的自机—行人因子化世界模型无人机导航方法**

## 2. 任务定义与研究边界

### 2.1 任务

给定无人机当前状态、目标点以及视野内多人的三维骨架历史，策略连续输出四维控制量，使无人机：

1. 到达 episode 随机生成的目标点；
2. 避免与行人、墙体和货架碰撞；
3. 尽量缩短路径和到达时间；
4. 保持动作平滑、飞行高度合理，并与行人保持安全间距。

当前成功判定为无人机三维位置与目标点距离不超过 **1.0 m**。动作定义为：

```text
[vx_body, vy_body, vz_world, yaw_rate]
```

其中水平速度在无人机机体系 ROS FLU（前、左、上）中表达，垂直速度沿世界 Z，所有动作按照 metadata 中的控制上限归一化到 `[-1, 1]`。

### 2.2 当前必须明确的边界

当前最终模型是 **Ego + Human 两分支**，不输入 RGB、深度、LiDAR 点云、BEV，也没有 Environment RSSM 或 PointPillars 分支。因此：

- 它适合研究“已知或固定静态场景中的动态人群交互与避碰”；
- Ego14 中的位置可能使模型对训练仓库形成隐式空间先验，但这不等价于显式静态障碍感知；
- 目前不能声称模型能在未知地图中泛化避开墙体和货架；
- 若论文目标包含未知场景静态避障，需要新增地图、LiDAR/深度或环境分支，并重新采集/训练；
- 旧文档中的 Ego–Environment–Human 三分支和 PointPillars 是历史方案，不是当前最终代码路径。

论文现阶段更稳妥的任务表述是：**固定仓库布局中，重点研究动态人群运动建模和目标导向穿梭。**

## 3. 系统全链路

```text
Isaac Sim 动态仓库
  ├─ 无人机物理与 PX4/MAVSDK
  ├─ 随机出生点、目标点、人数和人群路线
  └─ 人体 COCO17 姿态（录制时保留身体 12 关节）
                 │ 10 Hz
                 ▼
Compact Skeleton Dataset V3
  Ego14 + Goal + Human skeleton + applied action
  + reward/terminal + privileged evaluation labels
                 │
                 ▼
训练加载器
  ├─ 补偿无人机自运动后计算人体关节速度
  ├─ Track ID 稳定槽位
  ├─ 超过 20 人时按风险选择
  └─ 构造 HumanRoot10、Joint7、Goal8 和 mask
                 │
                 ▼
Ego–Human Factorized Dreamer
  ├─ Ego MLP encoder
  ├─ Human root MLP + causal ST-GCN
  ├─ 后验稀疏非对称注意力
  ├─ Ego RSSM + 共享参数的多 Human RSSM
  ├─ 想象阶段的潜状态耦合
  └─ Action/Goal latent policy attention
                 │
                 ▼
Actor: normalized [vx, vy, vz, yaw_rate]
                 │
                 ▼
待完成实时适配器 → PX4/MAVSDK → Isaac/真机闭环
```

## 4. 仿真、任务随机化与数据采集

### 4.1 仿真频率

- Isaac 物理：250 Hz；
- 人群控制：25 Hz；
- 三维骨架更新与数据录制：10 Hz；
- 控制执行：经 PX4/MAVSDK 进入飞行器闭环。

每个 seed 同时确定人数、人群路线和停留安排、无人机出生点及目标点。不同算法使用相同 seed，可以做配对公平比较。每个 PX4 模式 episode 都重启 PX4/MAVSDK，降低上一局状态残留造成的污染。

### 4.2 当前行为策略

正在录制的数据由 `px4_navrl` 控制，无 Safety Shield：

```text
control mode = px4_navrl
NAVRL_SAFETY_SHIELD_ENABLED = 0
run id = navrl_no_shield_1000_v1
seed = 1 ... 1000
```

这批轨迹是世界模型的 **离线行为数据**，不是世界模型自身的闭环测试结果。NavRL 采集策略可使用它自己的环境观测，而训练数据只向最终世界模型暴露 V3 接口规定的 Ego、Human 和 Goal 字段。

截至 2026-08-06 本文生成时的录制快照：244 个 seed 已完整结束，其中到达目标 185、行人碰撞 57、静态碰撞 1、超时 1；原始行为策略成功率约 75.8%。该数字只描述当前采集分布，不能作为世界模型的性能结果。

### 4.3 人群设置

- 默认 episode 实际人数约 17–23 人，录制维度保留实际人数，不提前截断；
- 模型最多接收 20 个行人槽位；
- 同组行人之间的硬碰撞/最小距离阈值设为 0.15 m；
- 行人数量超过模型容量时，加载器依据距离、TTC、前向走廊风险和骨架置信度选择更重要的 20 人。

## 5. 数据集 V3

正式 schema 为 `omninxt.crowd_skeleton_state.v3`。每个 episode 的目录为：

```text
episodes/episode_YYYYMMDD_HHMMSS/
├── metadata.json
├── chunks/chunk_000000.npz
├── chunks/chunk_000001.npz
├── events.jsonl
└── summary.json
```

一个 chunk 最多保存 256 帧。由于录制为 10 Hz，单个 chunk 最多约 25.6 秒。一个 episode 出现两个或更多 chunk 只是时间分块，不代表两个 episode；加载器会按连续 `frame_index` 拼成一条轨迹。

### 5.1 模型可见输入

| 输入 | 形状 | 含义 |
|---|---:|---|
| Ego14 | `[T,14]` | 局部位置、速度、加速度、离地高度、姿态及相对偏航 |
| Goal position | `[T,3]` | episode 起点坐标系中的固定目标位置 |
| Goal8 | `[T,8]` | 当前机体系目标差、距离、单位方向和航向误差 |
| Human root | `[T,20,10]` | 根节点位置/速度、人体尺度、置信度 |
| Human joints | `[T,20,12,7]` | 相对根节点的位置/速度及置信度 |
| Human masks/IDs | 变长 | 有效人、有效关节、track ID、槽位重置标记 |
| Action | `[T,4]` | 实际执行且归一化后的控制动作 |

Ego14 顺序为：

```text
local_x, local_y, local_z,
body_vx, body_vy, world_vz,
body_ax, body_ay, world_az,
altitude_agl, roll, pitch,
sin(relative_yaw), cos(relative_yaw)
```

人体骨架来自 COCO17，但永久只保留源索引 5–16 的身体 12 关节：双肩、双肘、双腕、双髋、双膝和双踝。位置坐标采用当前无人机 `base_link` 的 ROS FLU，单位米。

### 5.2 特权标签

数据还保存行人世界位置/速度、最小间距、TTC、碰撞、目标距离、奖励分量和终止结果。这些字段用于奖励复核、采样、评估和排错，**禁止进入观察编码器和策略输入**，避免仿真特权信息泄漏。

### 5.3 终止语义

- `reached_goal`、人与环境碰撞等真实任务终止：`is_terminal=true`；
- `time_limit`、人工停止、系统关闭等截断：`is_terminal=false`；
- Continue 头的监督目标为 `1 - is_terminal`，时间截断不能被错误当作物理终止。

## 6. 数据预处理的关键设计

### 6.1 自运动补偿的人体速度

骨架以无人机当前机体系表达。如果直接对相邻帧位置做差，无人机自身平移和旋转会被误认为行人运动。加载器先利用两帧 Ego14 位姿，把上一帧关节抬升到 episode 坐标系，再变换到当前机体系，最后差分得到人体速度。这一处理使 Human RSSM 学到的是人的相对动态，而不是传感器坐标系运动伪影。

### 6.2 稳定槽位与状态清零

加载器利用 track ID 在序列内保持行人与槽位稳定。新行人进入、ID 变化、槽位失效或 episode 重置时产生 `human_is_first`；模型据此清空对应 Human RSSM 和因果姿态历史，防止把前一个人的隐状态泄漏给新的人。

### 6.3 风险选择而非简单最近邻

当场景人数超过 20 时，选择综合考虑：

- 当前距离；
- 碰撞时间 TTC；
- 是否位于无人机前向通道；
- 骨架置信度。

这样可避免只保留最近的人，却漏掉稍远但高速迎面接近的高风险行人。

## 7. Ego–Human 双分支世界模型

### 7.1 观察编码

**Ego 编码器**：Ego14 先按训练集统计量标准化，再经 LayerNorm/MLP/SiLU 映射为一个 Ego token。

**Human 编码器**：每个行人的 Root10 经 MLP 编码；身体 12 关节的过去 8 帧由只看历史的 causal ST-GCN 编码；两者融合为一个 Human token。所有行人共享编码器参数，但保持各自槽位与时间历史。

### 7.2 后验观察注意力

观察 token 使用内容驱动的稀疏非对称注意力：

```text
Ego query    ← Ego + 所有有效 Human
Human_n query ← Ego + Human_n 自身
```

该结构表达了无人机需要同时综合所有人，而每个行人的局部动态主要受无人机与自身状态条件化。它也避免 Human–Human 全连接注意力随人数平方增长。当前实现没有人工构造的 relation-only Q/K，也没有所谓 private/relation 特征拆分。

### 7.3 因子化 RSSM

模型将潜在动力学拆为：

- 一个 Ego RSSM：建模无人机自身受控动力学；
- 最多 20 个 Human RSSM 状态：每个槽位状态独立，但共享同一套 Human RSSM 参数。

Ego RSSM 与 Human RSSM 参数不共享。在每次 prior transition 前，另一个与后验注意力参数独立的稀疏注意力模块耦合当前潜状态，并将耦合上下文与无人机动作连接后送入各分支转移：

```text
z^ego_{t+1}   ~ p(z^ego_{t+1} | z^ego_t, z^humans_t, a_t)
z^human_n_{t+1} ~ p(z^human_n_{t+1} | z^human_n_t, z^ego_t, a_t)
```

这里的人体转移使用无人机动作，是因为人体观测处于运动的无人机机体系中，同时需要刻画自机—行人交互条件。

### 7.4 Goal-conditioned 策略读出

策略注意力的 token 序列为：

```text
[Action token, Goal token, Ego latent, Human_1 latent, ..., Human_N latent]
```

仅 Action token 的输出送入 Actor、Value、Reward 和 Continue 头。Goal 用于策略和价值/奖励读出，但不直接进入物理 RSSM transition，避免把“任务意图”混同为“环境动力学原因”。

### 7.5 预测头

- Ego 头重建当前 Ego14，并预测下一帧 Ego 状态增量；
- Human 头预测下一时刻 root 增量、相对骨架和存在概率；
- yaw 的正余弦额外施加单位圆约束；
- 人体损失只在有效 person/joint mask 上计算。

## 8. Dreamer 潜空间想象与决策

离线阶段不仅训练预测器，也训练完整 Dreamer Actor–Critic。想象 rollout 不再调用观察编码器：

1. 从真实序列的后验状态出发；
2. 解码想象中的 Ego14；
3. 用固定目标位置和想象 Ego 位姿重新计算 Goal8；
4. 策略注意力产生动作；
5. Ego/Human 耦合 prior 前进一步；
6. Reward、Continue 和 Value 估计多步回报；
7. 用 imagined lambda-return 更新 Actor 和 Value。

当前默认想象长度为 15 步；数据频率 10 Hz 时对应约 1.5 秒。配置中的 `horizon=333` 是回报折扣时间尺度，不是一次想象 rollout 的步数。

## 9. 训练目标

世界模型损失可概括为：

```text
L_WM = L_dyn^ego + 0.1 L_rep^ego
     + L_dyn^human + 0.1 L_rep^human
     + L_ego_recon + L_ego_pred + 0.1 L_yaw_unit
     + L_human_root + L_human_mpjpe + 0.2 L_human_presence
     + L_reward + 0.2 L_reward_components + L_continue
```

策略学习部分为：

```text
L_AC = L_policy + L_value + 0.3 L_replay_value
L_total = L_WM + L_AC
```

训练使用动态 KL 与表示 KL、free nats、慢速 Value target、lambda-return、动作熵正则、AMP、AGC/梯度裁剪和 LaProp 优化器。当前离线脚本从 V3 chunk 构造长度 64 的序列，联合更新世界模型、预测头、Actor、Value 和 Replay Value。

## 10. 预期训练与部署阶段

### 阶段 A：离线数据采集（正在进行）

用 NavRL no-shield 在随机人群和目标下采集 1000 seeds，保留成功、碰撞和超时轨迹。失败轨迹对学习风险和终止尤其重要，不应在录制阶段删除。

### 阶段 B：离线预训练（代码主体已具备，尚待正式运行验证）

使用不重叠 seed 划分 train/validation/test。只用训练集计算 Ego 标准化统计量，完成世界模型与 Actor–Critic 联合训练。当前离线脚本默认对发现的全部序列 shuffle，尚未内置严格数据划分和 outcome-balanced sampler，因此正式论文训练前必须补齐实验协议。

### 阶段 C：Isaac 在线微调（接口设计已有，环境闭环未完成）

`FactorizedDreamerAgent` 已提供 `act/update/state/checkpoint` 形式的接口，但通用 `world_model/envs` 当前只支持 DMC、Atari、MemoryMaze、Crafter 和 MetaWorld，没有 Isaac UAV wrapper。需要实现：

1. 将 Isaac 实时 Ego/骨架/目标转换为与 V3 完全一致的张量；
2. 保持 10 Hz 时间同步、track slot 和 reset 语义；
3. 将模型归一化动作反变换为 PX4/MAVSDK 控制量；
4. 把碰撞、成功、超时映射为 reward/continue；
5. 在线 replay、评估和 checkpoint 恢复；
6. 设置动作限幅、通信超时和失控保护。

### 阶段 D：闭环评估与真机迁移（尚未完成）

先在 Isaac 对未见 seed 做确定性闭环评估，再考虑真机。真机必须用感知得到的三维骨架替换 Isaac GT，并检查坐标、时延、丢帧和置信度分布差异。当前代码库中没有可作为论文结果引用的因子化世界模型正式 checkpoint、训练曲线或 Isaac 闭环 benchmark。

## 11. 可提炼的论文创新点

以下是合理的候选贡献，但最终需由对比实验和消融支持：

1. **面向动态人群的自机—行人因子化潜在动力学。** 将受控无人机动力学与多个共享参数但状态独立的人体动力学分开，降低单一整体潜状态的纠缠。
2. **跨后验与想象阶段一致的结构化交互。** 后验编码与 prior imagination 都采用 Ego 读全体、Human 读 Ego 与自身的稀疏非对称注意力。
3. **因果骨架运动表达。** 利用身体拓扑、过去 8 帧和自运动补偿速度描述人体运动，避免依赖 RGB/深度大张量。
4. **显式解决动态人数与身份切换。** 通过 track ID 稳定槽位、风险截断和 slot reset，处理变长人群并阻断跨身份隐状态泄漏。
5. **Goal-conditioned Action-token 决策。** 在共享潜空间中聚合任务目标、自机状态和多人状态，直接用于潜空间想象下的 Actor–Critic。
6. **紧凑但可审计的数据接口。** 策略可见量与仿真特权标签严格分离，便于验证奖励、碰撞和 sim-to-real 可用性。

不要在没有实验前写“首次”“显著优于”“实时性更高”“保证安全”或“SOTA”。

## 12. 论文实验设计建议

### 12.1 数据划分

按 seed 划分，禁止把同一 episode 的不同 chunk 分到不同集合。建议在完成 1000 seeds 后固定：

- Train：1–700；
- Validation：701–850；
- Test：851–1000。

如果采集过程曾修复目标或记录逻辑，应优先按代码版本/run ID 分层，确保修复前异常数据不混入。正式划分可改变，但必须在所有模型之间保持一致。

### 12.2 对比方法

闭环导航基线可包括：

- NavRL without Safety Shield：当前数据行为策略；
- NavRL with Safety Shield；
- EGO-Planner；
- DPMPC；
- 标准/非因子化 Dreamer；
- 只做行为克隆的相同编码器策略。

不同方法必须使用相同测试 seed、目标、出生点和人群脚本。传统规划器与学习方法的观测条件不完全相同时，要在论文中明确区分“系统级比较”和“同观测公平比较”。

### 12.3 核心指标

- success rate；
- human collision rate、static collision rate；
- 最小人体表面间距、低于安全距离的持续时间和事件次数；
- 路径长度、路径效率、到达时间；
- 终点距离；
- 动作平滑度/加加速度；
- 决策延迟、10 Hz 实时达标率和显存占用；
- 世界模型 open-loop Ego 误差、Human root ADE/FDE、MPJPE、Reward/Continue 误差。

### 12.4 关键消融

1. 双分支 RSSM vs 单一联合 RSSM；
2. 去掉 imagination 中的 Ego–Human prior coupling；
3. 稀疏非对称 attention vs 全连接 attention vs 无 attention；
4. causal ST-GCN vs 只用 root vs 普通时序 MLP/GRU；
5. 去掉自运动补偿的人体速度；
6. 不做 slot reset 或不保持稳定 track slot；
7. 风险选人 vs 最近 20 人；
8. Action-token policy attention vs 直接拼接 latent；
9. Goal 只进 Actor vs 当前 Goal-conditioned Actor/Value/Reward/Continue；
10. 不同想象长度与不同人群密度。

### 12.5 泛化实验

- 未见 seed；
- 人数分布外测试；
- 人群速度、路线和停留模式变化；
- 骨架噪声、漏检、ID 切换与延迟注入；
- 若保持双分支无环境感知，应只在同一静态仓库布局内报告主要结论；
- 若要跨地图，必须先补充环境观测，再单独评估布局泛化。

## 13. 目前实现状态与证据边界

| 模块 | 当前状态 | 可以写什么 |
|---|---|---|
| Isaac/PX4/NavRL 数据录制 | 正在运行 | 可写采集系统和数据分布 |
| V3 schema、chunk、原子写入和恢复 | 已实现 | 可写数据协议与完整性机制 |
| 骨架预处理、风险槽位、自运动补偿 | 已实现 | 可写方法设计，需再给定量消融 |
| Ego/Human 编码器与稀疏注意力 | 已实现并有单元测试 | 可写网络方法 |
| 双分支 RSSM、想象和预测头 | 已实现并有单元测试 | 可写算法方法 |
| 离线 Dreamer 联合训练脚本 | 已实现，配置合成与 Agent 构建 smoke test 已通过 | 正式长训练前仍需固定数据划分并进行小数据过拟合验证 |
| 正式训练 checkpoint/收敛曲线 | 未发现 | 不能写最终性能 |
| Isaac 在线环境 wrapper | 未实现 | 只能写后续步骤 |
| 世界模型到 PX4 的实时动作闭环 | 未实现 | 不能声称已完成世界模型自主飞行 |
| 闭环基线比较与统计显著性 | 未完成 | 不能写优越性结论 |
| 真机部署 | 未完成 | 只能作为未来工作或后续阶段 |

## 14. 当前方案的主要风险

1. **静态障碍观测缺失。** 这是最重要的范围限制。固定仓库可作为研究设定，但未知地图需要扩展模型。
2. **离线分布偏移。** 数据主要由单一 NavRL no-shield 策略产生，Dreamer 学到的 Actor 在偏离数据支持区域时可能产生模型利用误差。
3. **失败/成功不平衡。** 当前约四分之一轨迹为失败，既有价值也可能需要 outcome-balanced 或 terminal-window 采样；需要通过验证集选择，不能凭直觉复制数据。
4. **GT 骨架到真实感知差距。** Isaac GT 骨架比真机姿态估计更稳定，必须做噪声、延迟、漏检和 ID 切换增强。
5. **最多 20 人的截断。** 风险选择缓解但不能完全消除遗漏；需报告被截断人数和高密度场景性能。
6. **1.5 秒想象窗口可能偏短。** 适合局部动态避让，但长时协作或绕行能力需用 horizon 消融验证。
7. **当前离线训练协议未固定。** 数据切分、标准化统计、超参数、随机种子和最好模型选择规则必须在正式实验前冻结。

## 15. 推荐的论文叙事

### 15.1 核心研究问题

在人群密集、人数变化且身份持续切换的场景中，如何让无人机世界模型同时：

- 保留自身受控动力学；
- 建模每个行人的独立时空运动；
- 表达自机与多人之间的交互；
- 在不依赖高带宽图像输入的情况下进行目标导向潜空间规划？

### 15.2 方法逻辑

传统整体 latent 容易把自机、多人和交互混在一起；本方法先用结构先验分解动力学，再用稀疏注意力只在必要位置耦合，最后用 Goal/Action token 聚合决策。因子化不是让各分支完全独立，而是让“状态归属清晰、交互路径受控”。

### 15.3 推荐章节结构

1. Introduction：问题、困难、核心思想、贡献；
2. Related Work：无人机人群导航、行人运动预测、世界模型/Dreamer、图骨架建模；
3. Problem Formulation：POMDP、观测/动作/奖励、固定场景假设；
4. Method：数据表示、因果人体编码、双分支 RSSM、两类注意力、潜空间想象；
5. Experimental Setup：Isaac/PX4、数据划分、基线、指标和实现细节；
6. Results：导航结果、模型预测、消融、鲁棒性和实时性；
7. Limitations：静态环境输入、离线分布偏移、sim-to-real；
8. Conclusion。

## 16. 代码依据索引

- 数据接口：`interfaces/dataset/CROWD_SKELETON_STATE_V3.yaml`
- 录制说明：`docs/SKELETON_STATE_DATASET_V3_RECORDING.md`
- 数据加载：`world_model/datasets/compact_skeleton_v3.py`
- 模型配置：`world_model/configs/model/factorized_dreamer.yaml`
- 环境张量契约：`world_model/configs/env/isaac_uav_human.yaml`
- Ego/Human 编码：`world_model/modules/factorized_encoders.py`
- 因子化 RSSM：`world_model/factorized_rssm.py`
- Dreamer 与 imagination：`world_model/factorized_dreamer.py`
- 预测头：`world_model/modules/factorized_prediction_heads.py`
- 联合训练步骤：`world_model/factorized_trainer.py`
- Agent 接口：`world_model/factorized_agent.py`
- 离线训练入口：`world_model/scripts/train_factorized_offline.py`
- 导航 benchmark：`simulation/isaacsim/database/NAVIGATION_BENCHMARK_GUIDE.md`
- 当前最终架构说明：`world_model/docs/factorized_dreamer_final.md`

## 17. 可直接交给 GPT 的提示词

```text
请阅读我提供的《无人机基于世界模型的人群穿梭方案说明》，以机器人/强化学习论文导师的视角帮我整理论文。

要求：
1. 先给出论文核心问题、中心假设和 3–4 个可被实验验证的贡献点；
2. 给出完整论文大纲，并说明每节要回答的问题、需要的图表和证据；
3. 用 POMDP 和 Dreamer 形式化方法，解释 Ego–Human 双分支 RSSM、后验稀疏注意力、想象阶段 prior coupling、causal ST-GCN 和 Action/Goal policy attention；
4. 设计公平的训练/验证/测试 seed 划分、对比基线、指标、消融和鲁棒性实验；
5. 严格区分已实现模块、正在采集的数据、待完成的在线闭环和尚未取得的实验结果；
6. 不得虚构 checkpoint、训练曲线、性能提升、统计显著性或真机实验；
7. 明确当前模型无 RGB/深度/LiDAR/BEV/环境分支，只能把主要结论限定在固定静态仓库布局；如果建议跨地图研究，请单列为扩展方案；
8. 分别给出“保守可投稿版本”和“增加环境感知后的增强版本”的论文路线；
9. 最后列出开始写论文前必须补齐的实验清单，按优先级排序。
```
