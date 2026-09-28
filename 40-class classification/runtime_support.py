"""Standard-library-only environment checks and structured runtime diagnostics."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
PYTHON_VERSION = (3, 10, 11)
FAULTS = {
    "ENVIRONMENT": (10, "运行环境不匹配", "按 README 显式安装固定环境；启动过程不会安装或升级软件包。"),
    "LSL_LIBRARY": (20, "无法加载 LSL 动态库", "检查 pylsl/liblsl 和 Python 的架构是否一致；macOS 可用 PYLSL_LIB 指向 liblsl.dylib。"),
    "LSL_NOT_FOUND": (21, "找不到 EEG LSL 流", "开启发布端的 LSL 输出，核对 type=EEG 或配置的流名，并检查防火墙和网络。"),
    "LSL_CONNECTION": (22, "LSL 连接或事件流异常", "检查重复流、通道配置、发布端和网络；重启发布端后重新连接。"),
    "ACQUISITION": (30, "采样中断或 EEG 数据异常", "检查设备连接、持续发送、采样率和时间戳；无效试次不会写入字符。"),
    "DISPLAY": (40, "显示异常", "运行 --self-test-ui；检查显示器编号、分辨率、刷新率、显卡驱动和系统负载。"),
    "SAVE": (50, "保存失败", "检查目标目录权限、磁盘空间；保留临时目录和逐试次文件以便恢复。"),
    "INTERNAL": (70, "程序异常", "保留诊断 JSON 中的异常堆栈以便排查。"),
}


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


class RuntimeFault(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def fault_record(exc: BaseException, default: str = "INTERNAL") -> dict:
    code = getattr(exc, "code", default)
    exit_code, title, action = FAULTS[code]
    return {"code": code, "exit_code": exit_code, "title": title, "detail": str(exc),
            "action": action, "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def print_fault(fault: dict) -> None:
    print(f"[{fault['code']}] {fault['title']}：{fault['detail']}\n处理建议：{fault['action']}",
          file=sys.stderr, flush=True)


def lock_path() -> Path:
    name = {"win32": "windows-py310.lock", "darwin": "macos-py310.lock"}.get(sys.platform)
    if name is None:
        raise RuntimeFault("ENVIRONMENT", f"未提供 {sys.platform} 的固定环境；当前支持 Windows/macOS。")
    return ROOT / "requirements" / name


def pinned_packages(path: Path) -> dict[str, str]:
    """Lock files are flat exact pins; platform selection happens by filename."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            result[name] = version
    return result


def environment_snapshot() -> dict:
    lock = lock_path() if sys.platform in ("win32", "darwin") else None
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "platform": platform.platform(), "machine": platform.machine(),
            "bits": struct.calcsize("P") * 8,
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock and lock.is_file() else None,
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                         if d.metadata.get("Name")},
            "pylsl_lib_override": os.environ.get("PYLSL_LIB")}


def check_environment(*, static_ui: bool = False) -> dict:
    """Read-only check. Never invoke pip, ensurepip, or a package manager."""
    issues = []
    if (sys.version_info[:3] != PYTHON_VERSION or struct.calcsize("P") != 8
            or platform.python_implementation() != "CPython"):
        issues.append(f"需要 64 位 CPython {'.'.join(map(str, PYTHON_VERSION))}；当前 {platform.python_version()}")
    if sys.platform == "darwin" and platform.mac_ver()[0].split(".")[0] != "15":
        issues.append("macOS 候选锁文件仅覆盖 macOS 15 的 PyObjC 框架集合")
    path = lock_path()
    # Static UI is intentionally usable without either pylsl or native liblsl.
    try:
        pins = pinned_packages(path)
    except (OSError, ValueError) as exc:
        raise RuntimeFault("ENVIRONMENT", f"无法读取固定版本锁文件 {path}：{exc}") from exc
    for name, expected in pins.items():
        if static_ui and name.lower() == "pylsl":
            continue
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "未安装"
        if actual != expected:
            issues.append(f"{name}: 需要 {expected}，实际 {actual}")
    if issues:
        raise RuntimeFault("ENVIRONMENT", "\n".join(issues) + f"\n锁文件：{path}\nPython：{sys.executable}")
    try:
        for name in ("numpy", "scipy", "threadpoolctl"):
            importlib.import_module(name)
    except Exception as exc:
        raise RuntimeFault("ENVIRONMENT", f"数值依赖无法导入：{exc}") from exc
    try:
        for name in ("psychopy.visual", "psychopy.event", "psychopy.gui"):
            importlib.import_module(name)
    except Exception as exc:
        raise RuntimeFault("DISPLAY", f"界面依赖无法加载：{exc}") from exc
    result = {"lock_file": str(path), "python_version": platform.python_version(),
              "lsl_checked": not static_ui}
    if not static_ui:
        try:
            pylsl = importlib.import_module("pylsl")
            result["liblsl_version"] = pylsl.library_version()
            result["liblsl_build"] = pylsl.library_info()
            # library_version only gives major/minor; build info includes the patch tag.
            if "v1.17.7" not in result["liblsl_build"]:
                raise RuntimeError("固定环境需要 liblsl v1.17.7；实际：" + result["liblsl_build"])
        except Exception as exc:
            raise RuntimeFault("LSL_LIBRARY", str(exc)) from exc
    return result


def probe_save_directory(directory: str | Path) -> str:
    """Exercise create/write/fsync/rename/read/delete without creating EEG records."""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".write-check-", dir=directory) as tmp:
            source, target = Path(tmp) / "probe.tmp", Path(tmp) / "probe.ok"
            with source.open("wb") as handle:
                handle.write(b"BCI write check\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(source, target)
            if target.read_bytes() != b"BCI write check\n":
                raise OSError("保存后内容校验失败")
    except Exception as exc:
        raise RuntimeFault("SAVE", f"目录 {directory} 不可可靠写入：{exc}") from exc
    return str(directory)


def write_diagnostic(report: dict, directory: Path | None = None) -> str | None:
    """Fall back to OS temp if the project itself is read-only; retain all faults."""
    name = time.strftime("run_%Y%m%dT%H%M%SZ_", time.gmtime()) + uuid.uuid4().hex[:8] + ".json"
    try:
        report["environment"] = environment_snapshot()
    except Exception as exc:
        report["environment_snapshot_error"] = str(exc)
    for folder in (directory or ROOT / "diagnostics", Path(tempfile.gettempdir()) / "bci-eeg-diagnostics"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"诊断报告：{path}", flush=True)
            return str(path)
        except OSError as exc:
            print(f"[SAVE] 诊断报告写入失败：{folder}：{exc}", file=sys.stderr, flush=True)
    return None
