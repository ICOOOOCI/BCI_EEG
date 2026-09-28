"""Explicit, repeatable installation into this project's .venv (never at app startup)."""
from __future__ import annotations

import platform
import struct
import subprocess
import sys
import venv

from runtime_support import ROOT, PYTHON_VERSION, RuntimeFault, configure_console, fault_record, lock_path, print_fault


def setup() -> int:
    configure_console()
    try:
        if sys.version_info[:3] != PYTHON_VERSION or struct.calcsize("P") != 8:
            raise RuntimeFault("ENVIRONMENT", "请使用64位 CPython 3.10.11 执行安装脚本。")
        requirements = lock_path()
        if sys.platform == "darwin" and platform.mac_ver()[0].split(".")[0] != "15":
            raise RuntimeFault("ENVIRONMENT", "macOS 候选锁文件针对 macOS 15；其他系统版本需重新锁定并验证 PyObjC 框架集合。")
        destination = ROOT / ".venv"
        python = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not destination.exists():
            print(f"创建独立环境：{destination}", flush=True)
            venv.EnvBuilder(with_pip=True).create(destination)
        elif not python.is_file() or not (destination / "pyvenv.cfg").is_file():
            raise RuntimeFault("ENVIRONMENT", f"{destination} 已存在但不是完整虚拟环境；请先另行保留该目录再重试。")
        subprocess.run([str(python), "-c", "import sys,struct; assert sys.version_info[:3] == (3,10,11) and struct.calcsize('P') == 8"], check=True)
        # Build tools themselves are fixed, and isolated builds cannot fetch new tools.
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                        "pip==23.0.1", "setuptools==78.1.0", "wheel==0.48.0", "packaging==26.3"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-deps", "--no-build-isolation", "-r", str(requirements)], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
        print("固定依赖安装完成。下一步运行启动器 --diagnose，再运行 --self-test-ui。", flush=True)
        return 0
    except Exception as exc:
        print_fault(fault_record(exc, "ENVIRONMENT"))
        return 10


if __name__ == "__main__":
    raise SystemExit(setup())
