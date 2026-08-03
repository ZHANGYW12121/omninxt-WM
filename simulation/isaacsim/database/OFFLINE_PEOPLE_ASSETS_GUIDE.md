# 外星人联网下载 Isaac Sim 行人资源并转为离线加载

本文记录服务器上实际采用的配置方式：直接在目标电脑上联网运行下载脚本，从 NVIDIA Isaac Sim 5.1 资源服务器下载当前工程需要的人物、纹理、骨骼和动画，验证完成后让 Pegasus 固定从本地目录加载。全程不需要 U 盘。

## 1. 前提条件

外星人上应已经安装并配置：

- Isaac Sim 5.1 standalone；
- 当前 `isaacsim/database` 自定义工程；
- PegasusSimulator；
- 能访问 NVIDIA 的 `omniverse-content-production.s3-us-west-2.amazonaws.com`。

确认以下文件存在：

```bash
cd "$HOME/zyw/isaacsim"

test -x ./python.sh
test -f database/download_people_assets.py
test -f database/patches/pegasus_local_people_assets.patch
test -f PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/people/person.py
```

后续命令必须使用 Isaac Sim 自带的 `./python.sh`，不要使用系统 `python3` 或 Conda Python。

## 2. 给 Pegasus 加入本地资源路径支持

原版 Pegasus 的 `Person` 类将人物地址写成 NVIDIA 网络 URL。先应用工程中准备好的补丁，使其优先读取 `PEGASUS_PEOPLE_ASSET_ROOT`，并在没有显式设置变量时自动寻找 `database/assets/people/Characters`。

```bash
cd "$HOME/zyw/isaacsim"

PERSON_FILE="PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/logic/people/person.py"

if rg -q 'PEGASUS_PEOPLE_ASSET_ROOT' "$PERSON_FILE"; then
  echo "Pegasus local people asset support already installed."
else
  git -C PegasusSimulator apply \
    ../database/patches/pegasus_local_people_assets.patch
fi
```

检查结果：

```bash
rg -n 'PEGASUS_PEOPLE_ASSET_ROOT|_default_local_people_asset_root' "$PERSON_FILE"
```

应能看到 `_configured_asset_root`、`_default_local_people_asset_root` 等代码。如果 `git apply` 报错，先确认该文件是否已经被其他版本修改；不要重复应用补丁。

## 3. 在外星人上直接联网下载

执行与服务器当时相同的下载命令：

```bash
cd "$HOME/zyw/isaacsim"
./python.sh database/download_people_assets.py
```

脚本会自动完成以下工作：

1. 从 Isaac Sim 5.1 官方资源地址下载两个人物目录；
2. 下载人物纹理；
3. 下载 `Biped_Setup.usd` 和 `biped_demo` 基础骨骼资源；
4. 下载当前工程引用的 16 个走路、镜像走路、待机和动作动画；
5. 将内容写入 `database/assets/people`；
6. 下载结束后自动打开 USD，检查所有依赖是否完整；
7. 确认人物 USD 不再包含远程依赖。

默认下载目录为：

```text
$HOME/zyw/isaacsim/database/assets/people/
├── Animations/
└── Characters/
```

正常完成时最后会输出类似：

```text
Offline people assets ready: .../database/assets/people/Characters
```

下载约 198 MB。网络较慢时不要中途关闭终端；每个文件下载完成后才会从 `.part` 临时文件原子替换成正式文件。

## 4. 单独执行完整性验证

下载完成后再运行一次只读验证：

```bash
cd "$HOME/zyw/isaacsim"
./python.sh database/download_people_assets.py --verify-only
```

最后必须出现：

```text
Offline people assets verified: .../database/assets/people/Characters
```

还可以检查文件数量和大小：

```bash
find database/assets/people -type f | wc -l
du -sh database/assets/people
```

当前工程预期为 32 个文件，约 198 MB。不同文件系统显示的大小可能略有差异。

## 5. 启动时使用本地资源

当前启动脚本会根据自身位置自动设置本地目录，因此直接运行即可：

```bash
cd "$HOME/zyw/isaacsim/database"
./run_people_warehouse_with_map.sh
```

它内部等价于：

```bash
export PEGASUS_PEOPLE_ASSET_ROOT="$HOME/zyw/isaacsim/database/assets/people/Characters"
```

如果不使用上述启动脚本，而是直接运行 `main.py`，需要在同一个终端先设置变量：

```bash
export PEGASUS_PEOPLE_ASSET_ROOT="$HOME/zyw/isaacsim/database/assets/people/Characters"
cd "$HOME/zyw/isaacsim"
./python.sh database/main.py
```

变量必须指向 `Characters`，不能指向上一级 `people`。

## 6. 断网验收

完成第 4 节验证后关闭 Wi-Fi 或临时断开网络，再启动人群仿真。

验收标准：

1. 两种人物都能正常出现；
2. 人物贴图正常，不是灰色或紫色材质；
3. 走路和待机动画正常；
4. 日志中没有 `Isaac/People/Characters` 的网络访问失败；
5. 启动过程中不需要等待在线人物资源。

如果人物未出现，执行：

```bash
echo "$PEGASUS_PEOPLE_ASSET_ROOT"
test -d "$PEGASUS_PEOPLE_ASSET_ROOT"
./python.sh database/download_people_assets.py --verify-only
```

如果 USD 可以打开但人物没有贴图，通常是 `textures` 或 `biped_demo/Textures` 下载不完整，重新联网运行下载脚本即可覆盖缺失文件。

## 7. 下载脚本当前覆盖的资源

人物：

```text
original_male_adult_construction_05
original_female_adult_business_02
```

动画包括 `stand_walk_1` 到当前使用的多套行走动画、对应 mirror 动画、`stand_idle_loop`、`stand_idle_wave_loop`、`LookAround` 和 `Sit`。

以后增加人物时，需要同时修改：

- `database/download_people_assets.py` 中的 `CHARACTERS`；
- `database/warehouse_crowd_v2/crowd_templates.py` 中的 `DEFAULT_CHARACTER_NAMES`。

修改后重新联网运行下载脚本，再执行 `--verify-only`。
