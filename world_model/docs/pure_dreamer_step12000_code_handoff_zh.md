# Pure Dreamer 12000-step 代码交接

本分支提供与当前 12000-step checkpoint 匹配的世界模型推理源码、Hydra 配置、
策略 TCP 协议和回归测试。实机 PX4/感知桥接由外星人电脑上的 Codex 继续完成。

## 模型文件

checkpoint 已通过 Git LFS 随本分支发布，克隆或拉取分支后位于：

```text
world_model/deployment_checkpoints/step_00012000_v67_human_exogeneity_klt_audit_compat.pt
size:   262157125 bytes
sha256: 7c3cba29d55c22c5bc671a35d68d0b1fe87a53caf5be86ae498c9b92e969ad52
```

若工作区中只有很小的 LFS 指针文件，先安装 Git LFS，再执行：

```bash
git lfs install
git lfs pull
```

加载后应满足：

```text
step=12000
training_objective_version=factorized_pure_dreamer_v67_action_exogenous_human_trajectory
architecture_version=factorized_dreamer_v8.7
actor_human_physical_slots=23
actor_input_dim=7229
critic_input_dim=7229
```

## 推理入口

在仓库根目录执行：

```bash
cd world_model
python scripts/serve_pure_dreamer_policy.py \
  --checkpoint deployment_checkpoints/step_00012000_v67_human_exogeneity_klt_audit_compat.pt \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 9775
```

正常启动标志：

```text
PURE_DREAMER_ACTOR_READY ... step=12000 ... planner=0 ... stochastic_actor=0
```

该入口固定执行确定性 Actor，不启用 MPC、候选动作搜索或 shield。请求与响应格式见
`interfaces/factorized_policy/PROTOCOL_V1.yaml`。

当前验证环境为 Python 3.11.15、PyTorch 2.8.0+cu128、NumPy 1.26.0；其余依赖沿用
`world_model/requirements.txt`。外星人已有 Dreamer 环境时，优先核对版本，不要覆盖
现有 CUDA PyTorch。

## 交接检查

```bash
sha256sum world_model/deployment_checkpoints/step_00012000_v67_human_exogeneity_klt_audit_compat.pt
python -m py_compile \
  world_model/scripts/serve_pure_dreamer_policy.py \
  world_model/scripts/serve_factorized_policy.py
```

策略服务要求 10 Hz 连续物理控制步、`base_link` ROS-FLU 人体骨架、有效 AGL、
14 维 Ego 状态和显式任务边界/静态几何。具体字段、shape 和动作顺序以版本化协议为准。
