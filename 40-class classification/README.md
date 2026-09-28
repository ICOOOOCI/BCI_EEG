# 固定环境、启动与故障诊断

启动不会安装、升级或修复依赖。先显式安装本目录的 `.venv`，之后所有 CMD / Mac 启动入口只使用该虚拟环境，不回退到全局 Python。运行时发现缺包或版本漂移会报告 `ENVIRONMENT` 并退出。离线算法仍可通过 import 调用。

## 环境与验证范围

| 项目 | Windows 基线 | macOS 候选环境 |
| --- | --- | --- |
| 系统 / 架构 | Windows x64，build 26200 | macOS 15，arm64 / x86_64 |
| Python | **CPython 3.10.11，64 位** | **CPython 3.10.11，64 位** |
| NumPy / SciPy | 2.2.6 / 1.15.3 | 2.2.6 / 1.15.3 |
| PsychoPy / pylsl / liblsl | 2026.2.4 / 1.18.2 / 1.17.7 | 相同，待真机验证 |
| pyglet | **1.4.11** | **1.5.27** |
| 平台锁文件 | `requirements/windows-py310.lock` | `requirements/macos-py310.lock` |
| 2026-09-28 验证状态 | 全新隔离环境安装、pip check、全屏静态界面及 26 项自动测试通过；启动器故障退出码 40 / 50 实测通过 | 共享依赖约束与 wheel 附带的 liblsl v1.17.7 已核对，Shell 语法检查通过；**未在 Mac 真机安装或运行，不能称为已验证** |

两个锁文件均包含精确 `==` 版本的完整依赖列表及安装工具。Windows 列表从当前可用环境的依赖闭包提取；macOS 列表按平台元数据调整，增加 PyObjC / zeroconf，移除 pywin32、pypiwin32、pywinhook、pyparallel。Mac 的 PyObjC 集合覆盖 Darwin 24（macOS 15，包括 15.4 新增框架）；其他 macOS 大版本需另行锁定和验证。不要把 Windows 的虚拟环境复制到 Mac，也不要混用 Intel 与 ARM 动态库。

本次没有进行新的真实 EEG 实验；静态界面通过不代表设备采样、识别准确率或物理显示延迟通过。历史实验记录不作为新环境的完整端到端验收。

## Windows

先安装上述版本的 Python，再双击 `setup_windows.cmd`。安装器只写项目 `.venv`，固定安装工具和依赖，禁止构建隔离环境隐式下载新依赖，最后执行 `pip check`。安装失败会保留输出及部分环境，修复原因后可重跑安装器。

在本目录的 PowerShell 中：

```powershell
.\setup_windows.cmd
.\start_keyboard_dual_mode.cmd --diagnose
.\start_ui_self_test.cmd
.\start_keyboard_dual_mode.cmd
```

`start_ui_self_test.cmd` 可双击；默认全屏、展示 5 秒后自动结束，另需数秒初始化和刷新率检查。其他参数会传给 Python，例如：

```powershell
.\start_keyboard_dual_mode.cmd --self-test-ui --windowed --seconds 10
.\start_keyboard_dual_mode.cmd --self-test-ui --screen 1
.\start_keyboard_dual_mode.cmd --diagnose --record-root "D:\EEG Records"
```

脚本默认结束后暂停，便于看到故障。自动化调用时设置 `$env:FBCCA_NO_PAUSE='1'`；退出码保持不变。默认保留已有快速入口行为；若需要完整启动核验对话框，设置 `$env:FBCCA_QUICK_ENTRY='0'`。

## macOS（待真机验收）

使用 macOS 15 和上述 Python 版本。Apple Silicon 使用 arm64 Python 和 arm64/universal2 库，Intel 使用 x86_64 库。进入程序目录，第一次赋予双击入口执行权限：

```bash
chmod +x setup_macos.command start_keyboard_dual_mode.command start_ui_self_test.command
bash setup_macos.command
bash start_keyboard_dual_mode.sh --diagnose
bash start_keyboard_dual_mode.sh --self-test-ui
bash start_keyboard_dual_mode.sh
```

也可双击 `.command` 文件；`.sh` 适合终端调用，会直接返回退出码。路径含空格时必须加引号。所有入口都以程序目录定位文件，与调用者当前目录无关。

`pylsl` 的 Python 包与 `liblsl` 动态库是两层依赖。优先使用锁定 wheel 附带的库；如果未附带或无法加载，显式提供 **v1.17.7 且架构匹配**的库路径，然后重跑诊断：

```bash
export PYLSL_LIB="/absolute/path/to/liblsl.dylib"
bash start_keyboard_dual_mode.sh --diagnose
```

启动入口不会安装 Homebrew、下载动态库或改系统路径。Mac 验收时记录 OS、CPU 架构、Python 和 `--diagnose` / `--self-test-ui` 生成的 JSON；还应实测 Retina、多显示器编号、ESC 退出、实际 EEG 连续采样和 NPZ 保存。完成这些步骤后才能将对应平台状态改为“已验证”。

## 三种无需开始 EEG 实验的检查

| 参数 | 检查内容 | 不检查的内容 |
| --- | --- | --- |
| `--diagnose` | 固定包版本、数值/界面模块导入、liblsl 构建版本、保存目录的写入/同步/重命名/读回/删除 | 不查找 EEG 流、不创建窗口 |
| `--self-test-ui` | 生产界面的 40 个键、字体、输出框、固定提示、布局、实际窗口与静态绘制时序 | **不导入 pylsl、不连接 EEG、不创建 Marker、不分类、不闪烁、不生成 EEG 记录** |
| `--preview-layout` | 交互调整布局及四角主观检查，参见 `布局设置说明.md` | 不连接 EEG、不做自动刷新率验收 |

静态自检允许完全没有 pylsl/liblsl；其他固定依赖仍须满足。自检只重复提交固定亮度画面，禁用闪烁帧和实验入口。ESC 中途取消返回 130，不记作通过。`--seconds` 必须是有限正数。诊断 JSON 会在自检结束后保存，这不是 EEG 会话文件。

正常实验启动前检查保存目录；程序首次取得试次 EEG 后才建立会话暂存目录。程序中出现可恢复的无效试次仍沿用重试行为，并记录分类原因；不可恢复的采样/显示异常会结束实验。已有算法、质量门控及布局确认流程保留。

## 故障分类与恢复

| 退出码 / 标识 | 含义 | 排查方向 |
| --- | --- | --- |
| 10 / `ENVIRONMENT` | Python 不匹配、缺依赖、包版本漂移 | 使用固定版本 Python，显式重跑对应安装器；不要在启动时临时补包 |
| 20 / `LSL_LIBRARY` | pylsl 无法加载 liblsl，或动态库版本不符 | 检查 DLL/dylib、v1.17.7 构建、位数和 CPU 架构；Mac 可设置 `PYLSL_LIB` |
| 21 / `LSL_NOT_FOUND` | 没有匹配的 EEG 流 | 设备连接不等于 LSL 已发布；开启输出，核对 type/name、网络和防火墙；终端会列出其他可见流 |
| 22 / `LSL_CONNECTION` | 多个匹配流、建连/通道配置/Marker 异常 | 指定唯一流名，检查流元数据及发布端 |
| 30 / `ACQUISITION` | 已发现流但采样未就绪、中断、时间轴或 EEG 数据异常 | 检查采集软件持续发送、设备连接、采样率、时戳及信号质量；不使用伪造样本补齐 |
| 40 / `DISPLAY` | GUI/OpenGL、窗口、显示器、布局或刷新时序失败 | 先跑静态自检；核对显示器编号、刷新率、驱动和负载；窗口模式仅作辅助排查 |
| 50 / `SAVE` | 目录不可写、逐试次保存/最终导出/临时清理失败 | 检查权限、空间；**保留暂存目录**，不要覆盖原记录 |
| 70 / `INTERNAL` | 未分类的程序错误 | 查看诊断报告的异常堆栈 |
| 130 | 自检/预览或命令行中断 | 不代表自检通过 |

每次运行在 `diagnostics/run_*.json` 写入模式、状态、分类错误、完整堆栈、Python 路径、系统/架构、实际包版本、锁文件 SHA256；自检还含显示信息。项目不可写时尝试系统临时目录 `bci-eeg-diagnostics`，并在终端打印路径。帮助和参数语法错误由 argparse 直接处理。

多个故障会同时保留，例如采样中断后导出失败不会覆盖最初的采集错误；保存失败优先返回 50。有效 NPZ 已写入但临时文件清理失败时，会明确报告 NPZ 路径。会话元数据同时保存运行环境版本；有真实数据的逐试次文件在导出失败时保留。

## 自动检查与环境更新

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -p "test_*.py" -v
```

Mac 将解释器替换为 `.venv/bin/python`。这些测试不用 EEG、真实窗口或网络，覆盖环境检查、LSL 缺失/中断、两种模式的显示异常、保存失败保留文件，以及静态模式不触发采集/闪烁。真实窗口仍需另外执行 `--self-test-ui`。

升级时请在单独的环境中验证，重新生成相应平台的完整版本锁，执行 `pip check`、上述测试、环境诊断和静态自检，再用真实 EEG 验证采集与保存；更新本表的验证日期和范围。不要删除 `==` 或把一次平台测试推广为两个平台都已通过。

平台依赖依据包括已安装 PsychoPy 2026.2.4 的发行元数据，以及 [PsychoPy 官方安装说明](https://psychopy.org/download.html)、[pylsl 官方动态库加载说明](https://github.com/labstreaminglayer/pylsl#liblsl-loading) 和 [PyObjC 11.1 的发行元数据](https://pypi.org/pypi/pyobjc/11.1/json)。
