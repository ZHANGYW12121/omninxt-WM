# OmniNxt 多电脑 Git 与 Codex 协作手册

本文用于把 Jetson Nano、服务器和外星人电脑上的 OmniNxt 代码纳入同一个 GitHub 私有仓库，同时避免把 Isaac Sim 安装目录、Conda 环境、数据集、模型权重、运行结果和机器缓存错误上传。

## 1. 唯一远程仓库

- GitHub：`git@github.com:ZHANGYW12121/omninxt-WM.git`
- 网页：<https://github.com/ZHANGYW12121/omninxt-WM>
- 默认分支：`main`
- Nano 当前本地仓库：`/home/neu/omnix-stack`

所有电脑都以这个仓库为代码真值源。原有工程目录先保留，不要直接改造成仓库，也不要在初次迁移时删除任何原文件。

## 2. 仓库的职责划分

建议最终目录如下：

```text
omninxt-WM/
├── AGENTS.md
├── edge/
│   └── jetson/                 # Nano上的相机、深度、骨架与传输代码
├── simulation/
│   └── isaacsim/               # Isaac Sim场景脚本、扩展、配置和测试
├── world_model/                # ST-GCN、时序编码、训练和推理代码
├── backend/                    # 骨架接收、服务、存储、API和部署代码
├── interfaces/                 # 三端共享的数据结构、坐标系与协议
├── calibration/                # 实机相机内外参和仿真使用的对应配置
├── environments/               # 环境清单、锁定文件和版本快照
├── third_party/                # 上游版本清单与本项目补丁
├── tools/                      # 跨机器共用的检查、转换和发布工具
└── docs/                       # 设计、操作、实验和交接文档
```

推荐所有权：

| 机器 | 主要负责目录 | 不应随意修改 |
|---|---|---|
| Jetson Nano | `edge/jetson/`、`calibration/` | `world_model/`、`simulation/` |
| 外星人电脑 | `simulation/isaacsim/` | 实机标定真值、Nano运行代码 |
| 服务器 | `world_model/`、`backend/` | 实机驱动和Isaac Sim安装环境 |
| 三端共同 | `interfaces/`、`environments/`、`docs/` | 修改接口时必须同步通知另外两端 |

“所有权”不是权限封锁，而是减少无意冲突。跨目录修改必须在提交说明和 PR 中明确原因。

## 3. 哪些内容进入 Git

### 3.1 应当提交

- 自己编写或明确修改过的 Python、C++、Shell、ROS、Isaac Sim 扩展代码。
- 小型 YAML、JSON、TOML、XML、URDF、USD 配置和启动文件。
- 相机内外参、坐标系定义、关键点顺序和消息协议。
- `Dockerfile`、`requirements.txt`、Conda YAML、版本锁定文件和环境检测脚本。
- 自动化测试、小型合成测试输入、README 和实验说明。
- 上游仓库 URL、分支、commit 和本项目补丁。

### 3.2 不应提交

- Isaac Sim、Omniverse、CUDA、ROS、Conda 或 Python 虚拟环境的完整安装目录。
- `~/.cache`、Kit cache、shader cache、pip cache、编译目录和 IDE 索引。
- ROS bag、相机原始图像、点云、数据集、训练样本和运行日志。
- TensorRT engine、ONNX、PyTorch checkpoint、训练 checkpoint 和下载的模型权重。
- `wandb/`、TensorBoard、训练输出、渲染输出和自动生成报告。
- `.env`、Token、密码、SSH 私钥、云凭据和数据库文件。
- Docker image、容器可写层和数据库 volume。

大模型和数据集应放在对象存储、NAS或专用数据盘中。仓库只保存下载位置、版本、SHA256、许可说明和生成命令，例如放在：

```text
environments/model-manifests/
environments/dataset-manifests/
```

不要把 Nano 的 SSH 私钥复制到服务器或外星人电脑。每台电脑都生成自己的密钥。

## 4. 三端共享接口必须先统一

三块代码对接时，最先稳定的不是内部实现，而是 `interfaces/`。至少需要版本化以下内容：

- 17个人体关键点的名称、编号和缺失点表示。
- 单位：位置统一使用米，时间统一说明秒或纳秒。
- 坐标系：明确原点、右手系、轴方向；当前可视化约定应保持 `X/Y` 为水平面、`Z` 向上。
- 时间戳语义：采集时间、推理完成时间和发送时间不能混用。
- 关键点字段：`x/y/z`、置信度、三维有效性、跟踪 ID、相机扇区和帧序号。
- 无飞控情况下的行为：Nano骨架和深度链路不得强依赖PX4数据。
- 序列化方式：JSON、MessagePack、Protobuf或ROS消息只能有一个正式版本。
- 版本号：不兼容修改必须提升接口版本，不能静默改变字段含义。

接口修改推荐使用独立分支，例如：

```text
interface/skeleton-v1
interface/timestamp-contract
```

## 5. 每台新电脑第一次配置 Git

以下步骤在服务器和外星人电脑上分别执行一次。

### 5.1 安装与身份配置

```bash
git --version

git config --global user.name "你的姓名或GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

生成该电脑自己的 SSH 密钥：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

一路按回车可使用默认路径。然后显示公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

把整行公钥添加到 GitHub 的 `Settings → SSH and GPG keys`。私钥 `~/.ssh/id_ed25519` 不能上传、发送或复制给其他电脑。

测试：

```bash
ssh -T git@github.com
```

### 5.2 克隆干净仓库

不要克隆到已有 Isaac Sim 或世界模型目录之上。选择新的目录：

```bash
cd ~
git clone git@github.com:ZHANGYW12121/omninxt-WM.git
cd omninxt-WM

git remote -v
git status
```

正常情况下应位于 `main`，且工作区干净。

## 6. 在新电脑上启动 Codex

从仓库根目录启动 Codex CLI，或用 Codex IDE 扩展打开整个仓库：

```bash
cd ~/omninxt-WM
codex
```

不要从 Isaac Sim 安装目录、数据集目录或用户主目录启动 Codex。仓库根目录能让 Codex正确读取 Git 状态和仓库级说明。

第一次对 Codex 输入：

```text
请先完整阅读 docs/GIT_MULTI_MACHINE_CODEX_WORKFLOW.md，并检查当前仓库的
git status、当前分支、远程地址和.gitignore。当前只进行只读审计，不复制、
移动、删除或提交任何文件。列出本机旧工程中应进入Git的第一方代码、应排除的
环境/缓存/数据/权重，以及建议的导入分支和验证命令。未经我确认不要开始导入。
```

这一步的目标是让 Codex 先识别旧工程结构，不能让它直接把整个目录复制进 Git。

## 7. 外星人电脑：迁移 Isaac Sim 仿真代码

先告诉 Codex 原仿真工程的绝对路径，用下面模板替换 `<旧仿真目录>`：

```text
本机是外星人电脑，负责 simulation/isaacsim。旧工程位于：
<旧仿真目录>

请先只读审计旧工程：
1. 找出我自己编写的Isaac Sim扩展、Python脚本、场景配置、URDF/USD、启动脚本和测试；
2. 找出Isaac Sim安装目录、Omniverse缓存、shader缓存、日志、录屏、渲染结果和生成数据；
3. 识别嵌套Git仓库并记录remote、branch、commit和本地修改；
4. 给出将第一方代码导入simulation/isaacsim的文件清单；
5. 给出需要写入environments/isaacsim的版本和依赖清单；
6. 不修改旧工程，不复制大文件，不执行git add/commit/push。
```

确认审计清单后，再输入：

```text
按刚才确认的清单执行导入。先从最新main创建分支
import/alienware-isaacsim；只复制第一方源码和可复现配置，保留旧工程不变。
先补充.gitignore，再检查大文件和秘密信息。完成后运行最小仿真启动或语法测试，
展示git diff和测试证据，但不要直接合并main，也不要使用force push。
```

Codex应使用以下分支步骤：

```bash
git switch main
git pull --ff-only
git switch -c import/alienware-isaacsim
```

Isaac Sim的完整安装仍留在原位置。Git仓库中的脚本通过配置或环境变量引用它，不复制安装目录。

## 8. 服务器：迁移世界模型

把 `<旧世界模型目录>` 替换成服务器上的真实绝对路径：

```text
本机是服务器，先只处理world_model。旧工程位于：
<旧世界模型目录>

请先只读审计：
1. 识别ST-GCN、时序编码、训练、评估和推理的第一方源码；
2. 识别数据集、checkpoint、预训练权重、wandb、TensorBoard和实验输出；
3. 记录Python/CUDA/PyTorch依赖及启动命令；
4. 对照interfaces中的17关键点顺序、坐标系、单位、时间戳和有效性字段；
5. 给出导入world_model的清单和最小回归测试；
6. 不修改旧工程，不复制数据或模型，不提交任何秘密。
```

确认清单后输入：

```text
按确认清单从最新main创建import/server-world-model分支并导入world_model。
只提交源码、配置、环境锁定文件、小型测试和模型/数据manifest；不要提交数据、
checkpoint或训练输出。运行最小单元测试和一次小型前向推理，展示git diff和测试证据，
不要直接合并main，不要使用force push。
```

分支步骤：

```bash
git switch main
git pull --ff-only
git switch -c import/server-world-model
```

## 9. 服务器：迁移后端

世界模型和后端必须使用两个独立分支和两个独立 PR。世界模型导入结束并回到最新 `main` 后，再处理后端。

```text
本机是服务器，现在只处理backend。旧工程位于：
<旧后端目录>

请先只读审计：
1. 识别骨架接收、队列、跟踪、API、数据库访问和部署代码；
2. 识别.env、Token、证书、数据库、volume、日志和用户数据；
3. 对照interfaces中的骨架协议和版本；
4. 生成.env.example，但绝不复制真实秘密；
5. 给出导入backend的清单、启动方式和最小接口测试；
6. 不修改旧工程，不执行提交或推送。
```

确认后输入：

```text
按确认清单从最新main创建import/server-backend分支并导入backend。
只提交第一方源码、无秘密配置模板、容器/部署声明和测试；排除数据库、volume、
日志和真实.env。运行最小服务启动和接口测试，展示git diff和测试证据，
不要直接合并main，不要使用force push。
```

## 10. 日常开发分支规则

不要在多台电脑上直接修改并推送 `main`。每个任务创建独立分支：

```text
sim/<任务名>
wm/<任务名>
backend/<任务名>
edge/<任务名>
interface/<任务名>
fix/<问题名>
docs/<文档名>
```

开始任务：

```bash
git switch main
git pull --ff-only
git switch -c wm/example-task
```

提交前：

```bash
git status
git diff
git diff --check
find . -type f -size +50M -not -path './.git/*' -print
```

只暂存本任务文件，不建议无检查地执行 `git add .`：

```bash
git add world_model/ interfaces/
git diff --cached --stat
git diff --cached
git commit -m "feat(world-model): describe the change"
git push -u origin HEAD
```

推荐提交前缀：

```text
feat(sim):
feat(world-model):
feat(backend):
feat(edge):
fix(...):
test(...):
docs(...):
chore(...):
```

在 GitHub 创建 PR 合并到 `main`。三块代码的导入不得压在一个巨大提交中。

## 11. 每次让 Codex 修改代码时的标准指令

可把下面模板交给任意机器上的 Codex：

```text
先读取根目录AGENTS.md、本目录最近的AGENTS.md和相关接口文档。检查git status，
保留已有用户修改。只修改本任务涉及的目录；如果需要修改interfaces，先说明兼容性影响。
不要提交模型、数据、缓存、运行结果、秘密或本机绝对路径。实现后运行与风险匹配的测试，
展示变更文件、测试结果和仍未解决的问题。未经明确要求不要force push、删除分支、
重写历史或使用git reset --hard。

本次任务：<在这里填写任务>
```

## 12. AGENTS.md 的建议

Codex官方将 `AGENTS.md` 用作随仓库保存的持久项目指导。根目录文件只放全局规则；各模块可放更具体的文件：

```text
AGENTS.md
simulation/isaacsim/AGENTS.md
world_model/AGENTS.md
backend/AGENTS.md
edge/jetson/AGENTS.md
```

第一次导入每个模块后，让对应电脑上的 Codex 根据真实构建和测试命令生成该模块的 `AGENTS.md`。内容应简短，至少包含：

- 模块用途和边界。
- 可信入口文件。
- 环境创建、构建、测试和启动命令。
- 不允许提交的产物。
- 接口和坐标系约束。
- 代码审查重点。

不要把临时聊天记录、机器密码或大段背景材料写入 `AGENTS.md`。

官方说明：<https://learn.chatgpt.com/docs/agent-configuration/agents-md>

## 13. 安全回滚方式

### 13.1 只查看历史，不修改

```bash
git status
git log --oneline --decorate -20
git show <commit>
git diff <旧commit>..<新commit>
```

### 13.2 已推送的错误提交

共享分支优先使用 `revert` 生成反向提交：

```bash
git switch <出错分支>
git pull --ff-only
git revert <错误commit>
git push
```

不要对共享分支使用 `git reset --hard` 加 `git push --force`。

### 13.3 在旧版本旁边安全测试

使用独立 worktree，不覆盖当前工作目录：

```bash
git worktree add ../omninxt-test-<短commit> <commit>
```

测试结束并确认其中没有需要保留的修改后，再移除该 worktree。Codex也支持把不同任务放在隔离的 Git worktree 中。

官方说明：<https://learn.chatgpt.com/docs/environments/git-worktrees>

## 14. 秘密和大文件检查

提交前至少执行：

```bash
find . -type f -size +50M -not -path './.git/*' -print

rg -n 'github_pat_|ghp_|PRIVATE KEY|password[[:space:]]*[:=]|token[[:space:]]*[:=]' \
  . --glob '!.git/**'
```

命中并不一定都是秘密，但必须人工检查。不要把任何实际密码粘贴进 Codex 提示、脚本、Git commit 或 Issue。

如果某个秘密曾经提交过，仅把文件加入 `.gitignore` 并不能消除历史记录；应立即撤销或轮换该秘密，再单独处理 Git 历史。

## 15. Codex Cloud 与 GitHub（可选）

本地 Codex 只要在 Git 仓库根目录运行，就能使用本地 Git 做差异检查和回滚，不需要额外连接器。

如果希望使用 Codex Cloud 或在 GitHub PR 中请求代码审查：

1. 在 Codex/ChatGPT 设置中授权 GitHub，并选择 `ZHANGYW12121/omninxt-WM`。
2. 为仓库配置 Codex Cloud 环境。
3. 在 PR 评论中使用 `@codex review` 请求审查；也可以在设置中启用自动审查。
4. 把重要审查规则写在根目录或模块目录的 `AGENTS.md`。

官方说明：<https://learn.chatgpt.com/docs/third-party/github>

GitHub连接不是本地提交的替代品。仍然要使用分支、清晰提交、测试和PR。

## 16. 初次迁移完成的验收标准

每一块代码只有同时满足以下条件才算完成迁移：

- 原工程目录仍完整保留。
- Git中只有第一方源码和可复现配置。
- 没有数据、模型、缓存、日志和秘密。
- 环境版本和外部依赖有清单。
- 关键上游依赖记录了URL和commit。
- 有最小构建、启动或前向推理测试。
- 与 `interfaces/` 中的关键点、坐标系、单位和时间戳一致。
- 独立分支已经推送并创建PR。
- 另一台机器能从干净clone中复现最小测试。

## 17. 推荐实施顺序

1. 先完善 `interfaces/` 中的骨架消息、坐标系和版本。
2. 外星人电脑导入 `simulation/isaacsim/`。
3. 服务器导入 `world_model/`。
4. 服务器用另一个分支导入 `backend/`。
5. 分别增加模块级 `AGENTS.md` 和环境清单。
6. 用一小段固定测试数据做 Nano、仿真、世界模型和后端的端到端一致性测试。
7. 通过PR逐块合并，避免一次性迁移全部目录。

这样可以让仿真、实机、世界模型和后端共享同一套接口与标定定义，同时各自保留适合本机的安装环境和大数据。
