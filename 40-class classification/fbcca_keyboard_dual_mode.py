"""40-target SSVEP keyboard: FREE SPELLING + CUED TEST.

基于 fbcca_keyboard_quick_start_with_output.py 增加模式选择。
启动界面：1=自由输入（默认），2=原有提示测试；SPACE/ENTER开始，ESC退出。
自由输入：自行注视字符，每轮真实LSL EEG经FBCCA识别后写入顶部OUTPUT框。
电脑键盘SPACE暂停/继续；屏幕内SPACE目标插入空格，BACK目标删除上一字符。
采用4×10 QWERTY布局；按新行列重排40类及频率（8–17.6 Hz）。
保持M3七子带、五谐波和现有LSL快速联调处理。
本版不是个人校准程序，也没有自动空闲检测；不要将快速模式当最终实验验证。
不包含迷宫，不注入操作系统按键，不保存EEG/CSV/NPZ/XDF/实验日志文件。
算法函数仍可import调用；import不会安装依赖或启动闪烁。

UI API references checked 2026-09-17:
https://psychopy.org/api/event.html
https://psychopy.org/api/visual/textstim.html
https://psychopy.org/api/visual/window.html
"""
from __future__ import annotations

# ======================== 启动时自动检查/安装运行依赖 ========================
# 只在“直接运行本文件”时执行；如果本文件被其他代码 import，不会擅自安装软件包。
# 使用 sys.executable -m pip，确保依赖安装到“当前正在运行本程序的 Python/虚拟环境”。
import importlib
import importlib.util
import subprocess
import sys


def _configure_console_encoding() -> None:
    """让旧版 Windows 控制台也能显示启动诊断，不改变文件输出。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                # 某些 IDE 将 stdout 替换成不支持 reconfigure 的对象；
                # 诊断输出仍应继续执行。
                pass


_configure_console_encoding()

_REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pylsl": "pylsl",
    "psychopy": "psychopy",
    "threadpoolctl": "threadpoolctl",
}

# PsychoPy 2026.x still declares ``pywinhook`` as a Windows dependency.  That
# package has no Python 3.12 wheel and its source build requires SWIG, so a
# normal ``pip install psychopy`` aborts before *any* package is installed.
# The keyboard handling used by this program is PsychoPy's pyglet backend and
# does not need pywinhook.  Install PsychoPy's published dependencies in a
# separate step and install PsychoPy with ``--no-deps``; this keeps the
# optional legacy hook from blocking the actual display application.
_PSYCHOPY_DEPENDENCIES = (
    "matplotlib", "pyglet==1.4.11", "pillow>=9.4.0", "pyqt6", "pandas>=1.5.3",
    "questplus>=2023.1", "openpyxl", "xmlschema", "soundfile", "sounddevice",
    "psychtoolbox", "imageio", "imageio-ffmpeg", "zope.event==5.0",
    "zope.interface==7.2", "gevent==25.5.1", "MeshPy", "psutil", "pyzmq>=22.2.1",
    "ujson", "msgpack", "msgpack-numpy", "pyyaml", "freetype-py", "python-bidi",
    "arabic-reshaper", "websockets", "requests", "i18next", "astunparse", "esprima",
    "jedi>=0.16", "pyserial", "pyparallel", "ffpyplayer", "opencv-python",
    "python-vlc>=3.0.21203", "pypiwin32", "tables", "packaging>=24.0", "moviepy",
    "pyarrow", "beautifulsoup4",
)


def _ensure_runtime_dependencies() -> None:
    """缺什么装什么；已安装的包不会重复安装。"""
    missing = [
        pip_name
        for import_name, pip_name in _REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    print("\n=== Python 运行依赖检查 ===", flush=True)
    print("当前 Python：" + sys.executable, flush=True)
    if not missing:
        print("[OK] 主包均已找到；继续验证界面模块是否能实际加载。", flush=True)
        return

    print("检测到缺失依赖：" + ", ".join(missing), flush=True)
    print("将自动安装到当前 Python：" + sys.executable, flush=True)

    # venv 通常自带 pip；若某个环境没有 pip，则先用标准库 ensurepip 恢复。
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if pip_check.returncode != 0:
        print("当前环境未检测到可用 pip，正在尝试启用 pip...", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        except Exception as exc:
            raise RuntimeError(
                "当前 Python 环境没有可用 pip，且 ensurepip 启用失败。"
                f"\nPython: {sys.executable}\n原始错误: {exc}"
            ) from exc

    try:
        # Do not let PsychoPy's pywinhook dependency force a native build.
        # Install the scientific/LSL packages and PsychoPy's other runtime
        # dependencies first, then install PsychoPy itself without resolving
        # dependencies a second time.
        base_missing = [name for name in missing if name != "psychopy"]
        psychopy_missing = "psychopy" in missing
        if base_missing or psychopy_missing:
            deps = list(dict.fromkeys(base_missing + list(_PSYCHOPY_DEPENDENCIES)))
            if deps:
                deps_cmd = [sys.executable, "-m", "pip", "install",
                            "--disable-pip-version-check", *deps]
                print("执行核心依赖安装（跳过不兼容的 pywinhook）：" +
                      " ".join(deps_cmd), flush=True)
                subprocess.check_call(deps_cmd)
        if psychopy_missing:
            psychopy_cmd = [sys.executable, "-m", "pip", "install",
                            "--disable-pip-version-check", "--no-deps", "psychopy"]
            print("执行 PsychoPy 安装（不拉取 pywinhook）：" +
                  " ".join(psychopy_cmd), flush=True)
            subprocess.check_call(psychopy_cmd)
    except subprocess.CalledProcessError as exc:
        failed_cmd = " ".join(str(x) for x in (getattr(exc, "cmd", None) or []))
        raise RuntimeError(
            "自动安装依赖失败。请检查网络、pip 源或该 Python 版本是否受依赖支持。"
            f"\n失败依赖: {', '.join(missing)}"
            f"\nPython: {sys.executable}"
            f"\n命令: {failed_cmd}"
            f"\npip 退出码: {exc.returncode}"
        ) from exc

    importlib.invalidate_caches()
    still_missing = [
        import_name
        for import_name in _REQUIRED_PACKAGES
        if importlib.util.find_spec(import_name) is None
    ]
    if still_missing:
        raise RuntimeError(
            "pip 已执行完成，但仍找不到以下模块：" + ", ".join(still_missing)
            + f"\n当前 Python: {sys.executable}"
        )

    print("[完成] 缺失依赖已自动安装，继续启动 FBCCA 程序。\n", flush=True)


if __name__ == "__main__":
    _ensure_runtime_dependencies()

import os
# 避免小型 CCA 矩阵运算动用大量 BLAS 线程；已有环境设置不覆盖。
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
import math
import queue
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ======================== 配置：通常只修改这一处 ========================
@dataclass
class Config:
    # 窗长预设与输入模式相互独立；均不冒充论文严格1.8秒在线时序。
    # 新增两种模式：free=自由输入；cued=原有40目标提示测试。
    # 启动界面按1/2选择，SPACE或ENTER开始；默认free，不要求先做校准。
    session_mode: str = "free"
    free_prepare_s: float = 1.0  # 每轮闪烁前留出时间，自行选择下一个字符。
    # debug_2s: 刺激2 s, 分析2 s; paper_window_1p25s: 两者1.25 s;
    # paper_offline_5s: 两者5 s。最后一种做240次时同时设置 blocks=6。
    protocol: str = "debug_2s"
    # 快速调试模式：优先进入程序。临时放宽LSL时间戳/显示时序的严格实验级检查。
    # 仅用于当前联调，不应拿本模式下的数据作为最终论文实验结果。
    quick_entry_mode: bool = True
    blocks: int = 1
    random_seed: int = 20260916
    cue_s: float = 0.5
    response_delay_s: float = 0.14
    blank_min_s: float = 0.5
    feedback_s: float = 0.5
    block_rest_s: float = 30.0
    max_consecutive_invalid: int = 3

    eeg_stream_name: Optional[str] = None
    eeg_stream_type: str = "EEG"
    # 假定 LSL 前8列按设备 CH1..CH8 输出；启动时必须核对这一假定。
    # 用户2026-09-13接线：P1->8,P2->7,PO3->6,POz->5,PO4->4,O1->2,Oz->3,O2->1。
    channel_indices: tuple[int, ...] = (7, 6, 5, 4, 3, 1, 2, 0)
    channel_positions: tuple[str, ...] = ("P1", "P2", "PO3", "POz", "PO4", "O1", "Oz", "O2")
    buffer_s: float = 60.0
    connect_timeout_s: float = 10.0
    startup_timeout_s: float = 15.0
    warmup_s: float = 2.0
    data_wait_timeout_s: float = 8.0
    clock_refresh_s: float = 1.0
    max_clock_step_s: float = 0.002
    max_receive_age_s: float = 2.0
    max_gap_factor: float = 1.5  # 发现一个明确缺样(2个采样间隔)也拒绝，不跨缺口补齐。
    rate_tolerance: float = 0.02
    min_valid_channels: int = 8
    # AUTO 从流元数据读取；缺失时不会猜测单位。对话框允许人工明确覆盖。
    input_unit: str = "AUTO"
    upstream_filter_description: str = "UNKNOWN"
    upstream_lowpass_hz: Optional[float] = None
    # 已独立测得的固定设备滞后：正值表示发布端时间戳晚于实际采样，需要减去。
    # 0仅表示未应用修正，不表示已测得延迟为零；不得随意填写。
    device_timestamp_lag_s: float = 0.0

    target_fs: float = 250.0
    n_harmonics: int = 5
    weight_a: float = 1.25
    weight_b: float = 0.25
    notch_hz: Optional[float] = 50.0  # 保留原文件50 Hz; 按采集地供电/已有滤波核对。
    cca_regularization: float = 1e-10

    full_screen: bool = True
    screen_index: int = 0
    window_size: tuple[int, int] = (1920, 1080)
    key_size_px: float = 140.0
    key_gap_px: float = 50.0
    allow_layout_scaling: bool = True  # 小屏等比例缩小，并明确打印非原尺寸。
    viewing_distance_cm: float = 70.0  # 目标观看距离，不代表自动测量结果。
    screen_diagonal_inches: Optional[float] = None
    frame_long_factor: float = 1.5
    frame_short_factor: float = 0.5
    marker_stream_name: str = "FBCCA_40_Keyboard_Events"
    output_box_height_px: float = 72.0
    output_box_margin_px: float = 24.0

    @property
    def window_s(self) -> float:
        presets = {"debug_2s": 2.0, "paper_window_1p25s": 1.25, "paper_offline_5s": 5.0}
        if self.protocol not in presets:
            raise ValueError(f"未知 protocol: {self.protocol}; 可选 {tuple(presets)}")
        return presets[self.protocol]

    @property
    def stimulus_s(self) -> float:
        return self.window_s

    def validate(self) -> None:
        _ = self.window_s
        if self.session_mode not in ("free", "cued"):
            raise ValueError("session_mode 必须为 free 或 cued")
        if not math.isfinite(self.free_prepare_s) or self.free_prepare_s <= 0:
            raise ValueError("free_prepare_s 必须是有限正数")
        for name in ("blocks", "n_harmonics", "min_valid_channels", "max_consecutive_invalid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("cue_s", "response_delay_s", "blank_min_s", "feedback_s", "block_rest_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        for name in ("connect_timeout_s", "startup_timeout_s", "warmup_s", "data_wait_timeout_s",
                     "buffer_s", "clock_refresh_s", "max_clock_step_s", "max_receive_age_s",
                     "key_size_px", "viewing_distance_cm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not math.isfinite(self.key_gap_px) or self.key_gap_px < 0:
            raise ValueError("key_gap_px 必须非负")
        if not 1 < self.max_gap_factor < 2:
            raise ValueError("max_gap_factor 必须在1和2之间，不得跨明确缺样插值")
        if not (1 < self.frame_long_factor < 2 and 0 < self.frame_short_factor < 1):
            raise ValueError("帧间隔阈值不合法")
        if not 0 < self.rate_tolerance < 0.1:
            raise ValueError("rate_tolerance 必须在0和0.1之间")
        if not math.isfinite(self.device_timestamp_lag_s):
            raise ValueError("device_timestamp_lag_s 必须有限")
        _validate_fs(self.target_fs)
        _normalise_channel_indices(self.channel_indices)
        if len(self.channel_indices) != len(self.channel_positions):
            raise ValueError("channel_indices 和 channel_positions 数量必须一致")
        if not 1 <= self.min_valid_channels <= len(self.channel_indices):
            raise ValueError("min_valid_channels 超出所选通道数")
        if self.buffer_s <= self.window_s + self.response_delay_s + self.data_wait_timeout_s + 2:
            raise ValueError("内存缓冲区过短，可能覆盖仍待分类的 EEG")
        _validate_harmonics(self.n_harmonics, BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        if self.upstream_lowpass_hz is not None:
            if not math.isfinite(self.upstream_lowpass_hz) or self.upstream_lowpass_hz < 90:
                raise ValueError("已知发布端低通小于90 Hz，不能使用当前完整M3频带")


CONFIG = Config()


@dataclass(frozen=True)
class Target:
    class_id: int             # 对外统一使用1-based类别编号
    symbol: str
    row: int                  # 内部0-based行、列
    col: int
    frequency_hz: float
    phase_rad: float = 0.0    # 普通正弦频率编码；没有冒充JFPM或引入相位分类。

    @property
    def display_label(self) -> str:
        # 视觉标签与输入语义分开；空白键仍是可识别的SPACE目标。
        return {"SPACE": "", "BACK": "<-"}.get(self.symbol, self.symbol)


# 唯一目标表：界面、参考信号、事件、预测字符和评估均从这里读取。
KEY_ROWS = (
    tuple("1234567890"),
    tuple("QWERTYUIOP"),
    (*tuple("ASDFGHJKL"), "BACK"),
    ("SPACE", *tuple("ZXCVBNM,.")),
)
# 频率继续使用原行列公式；新布局的范围为8–17.6 Hz，不能沿用旧字符映射。
TARGETS = tuple(Target(i + 1, symbol, r, c, round(8 + c + .2 * r, 1))
                for i, (r, c, symbol) in enumerate(
                    (r, c, symbol) for r, row in enumerate(KEY_ROWS) for c, symbol in enumerate(row)))
BENCHMARK_FREQUENCIES_HZ = np.asarray([t.frequency_hz for t in TARGETS])
M3_BANDS = tuple((float(8 * n - 2), 90.0) for n in range(1, 8))
DEFAULT_WINDOW_S = 2.0
TARGET_FS = 250.0
N_HARMONICS = 5
EEG_CHANNEL_INDICES = np.asarray(CONFIG.channel_indices, dtype=int)
LAST_SESSION: Optional[dict[str, Any]] = None


class InvalidTrial(RuntimeError):
    """数据/时序未通过质量检查；必须计入无效试次，而不是静默跳过。"""


class DegenerateSignalError(ValueError):
    """没有足够的变化信号，不能产生有意义的 CCA 结果。"""


class AbortSession(RuntimeError):
    """用户取消或急停。"""


class PauseSelection(RuntimeError):
    """自由输入暂停：立即停止闪烁，取消尚未提交的选择；不关闭EEG采集。"""


# ======================== 原算法核心 + 数值修正 ========================
def _normalise_channel_indices(channel_indices: Optional[Sequence[int]]) -> np.ndarray:
    indices = EEG_CHANNEL_INDICES if channel_indices is None else np.asarray(channel_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("channel_indices 必须是一维非空数组")
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.round(indices)):
            raise ValueError("channel_indices 必须是整数")
    indices = indices.astype(int)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("通道索引必须非负且不重复")
    return indices


def sample_count(duration_s: float, fs: float) -> int:
    """[start, end)内的均匀采样点数；1.25s×250Hz需要313点而非静默截掉半点。

    最后一采样点必须严格小于请求终点；浮点容差只处理整数乘积舍入误差。
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("时长和采样率必须为有限正数")
    return int(math.ceil(duration_s * fs - 1e-9))


def _validate_fs(fs: float) -> None:
    # 在重采样之前检查原始带宽；绝不通过上采样掩盖不足的Nyquist频率。
    if not np.isfinite(fs) or fs <= 2 * max(high for _, high in M3_BANDS):
        raise ValueError("原始/处理采样率必须大于180 Hz，以支持90 Hz子带上限")


def _validate_harmonics(n: int, frequencies: np.ndarray, fs: float) -> None:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("谐波数必须为正整数")
    if not np.isfinite(fs) or fs <= 0 or np.max(frequencies) * n >= fs / 2:
        raise ValueError("最高参考谐波必须低于Nyquist频率")


def generate_reference_signals(frequencies_hz: Sequence[float], fs: float,
                               n_samples: int, n_harmonics: int = 5) -> np.ndarray:
    """原参考信号公式；返回(targets, 2*harmonics, samples)。"""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if (frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or len(np.unique(frequencies)) != len(frequencies)):
        raise ValueError("候选频率须为一维、不重复的有限正数")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples 必须是大于等于2的整数")
    _validate_harmonics(n_harmonics, frequencies, fs)
    t = np.arange(n_samples, dtype=float) / fs
    harmonic = np.arange(1, n_harmonics + 1, dtype=float)
    phase = 2 * np.pi * frequencies[:, None, None] * harmonic[None, :, None] * t[None, None, :]
    return np.stack((np.sin(phase), np.cos(phase)), axis=2).reshape(
        len(frequencies), 2 * n_harmonics, n_samples)


def _normalised_rows(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """先按每行幅值缩放，再中心化/标准化，避免微伏/伏尺度及平方下溢。

    阈值仅用于“相对于该行原始幅值”的数值可分辨性，不是固定伏值下限。
    返回非恒值行及它们的索引；全零/全恒值输入明确报错。
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("输入须为有限二维数组(channels, samples)，至少2个样本")
    amplitude = np.max(np.abs(x), axis=1, keepdims=True)
    scaled = np.divide(x, amplitude, out=np.zeros_like(x), where=amplitude > 0)
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    active = np.flatnonzero(rms > 64 * np.finfo(float).eps)
    if not len(active):
        raise DegenerateSignalError("全零或全部恒值通道，拒绝输出任意最高分类")
    return centered[active] / rms[active, None], active


def _inverse_sqrt_covariance(covariance: np.ndarray, regularization: float) -> np.ndarray:
    """相对特征值下限；不再用固定绝对eps截断微小协方差。"""
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    scale = float(np.max(values))
    if not np.isfinite(scale) or scale <= 0:
        raise DegenerateSignalError("协方差不含有效正特征值")
    floor = scale * max(float(regularization), np.finfo(float).eps * len(values))
    return (vectors * (1 / np.sqrt(np.maximum(values, floor)))) @ vectors.T


def _whiten_rows(data: np.ndarray, regularization: float) -> np.ndarray:
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization 必须非负且有限")
    z, _ = _normalised_rows(data)
    covariance = z @ z.T / (z.shape[1] - 1)
    # 标准化后按相对协方差尺度加正则，而不是max(trace/dim, 1)。
    scale = float(np.trace(covariance)) / len(covariance)
    covariance += regularization * scale * np.eye(len(covariance))
    return (_inverse_sqrt_covariance(covariance, regularization) @ z
            / math.sqrt(z.shape[1] - 1))


def cca_correlation(eeg: np.ndarray, reference: np.ndarray, regularization: float = 1e-10) -> float:
    """修正后的最大典型相关系数；单位缩放不改变结果。"""
    x, y = np.asarray(eeg, float), np.asarray(reference, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("CCA输入必须二维且样本数相等")
    cross = _whiten_rows(x, regularization) @ _whiten_rows(y, regularization).T
    return float(np.clip(np.linalg.svd(cross, compute_uv=False)[0], 0, 1))


def fbcca_fusion(eeg_subbands: np.ndarray, reference_signals: np.ndarray,
                 a: float = 1.25, b: float = 0.25, regularization: float = 1e-10
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保留 score=sum((k**(-a)+b)*rho**2)，复用白化参考以减少重复计算。"""
    bands, refs = np.asarray(eeg_subbands, float), np.asarray(reference_signals, float)
    if (bands.ndim != 3 or refs.ndim != 3 or bands.shape[-1] != refs.shape[-1]
            or min(bands.shape) < 1 or min(refs.shape) < 1):
        raise ValueError("FBCCA形状必须为(bands, channels, samples)/(targets, components, samples)")
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError("权重参数a、b须非负且有限")
    wy = [_whiten_rows(ref, regularization) for ref in refs]
    correlations = np.empty((len(bands), len(refs)))
    for k, band in enumerate(bands):
        wx = _whiten_rows(band, regularization)
        for j, ref_w in enumerate(wy):
            correlations[k, j] = np.clip(np.linalg.svd(wx @ ref_w.T, compute_uv=False)[0], 0, 1)
    weights = np.arange(1, len(bands) + 1, dtype=float) ** (-a) + b
    return weights @ (correlations ** 2), correlations, weights


def preprocess_lsl_window(data_ch_samples: np.ndarray, fs: float,
                          window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                          target_fs: float = TARGET_FS, notch_hz: Optional[float] = 50.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """保留原 Chebyshev-I 阶数4/纹波0.5dB和filtfilt；不是连续因果滤波。

    输入为均匀时间网格。onset_s仅为数组裁剪偏移；事件截窗后必须传0。
    零相位滤波只使用已到齐的本窗口及SciPy默认反射边界，未引入后续试次。
    """
    from scipy.signal import cheby1, filtfilt, iirnotch, resample_poly, sosfiltfilt
    _validate_fs(fs)
    _validate_fs(target_fs)
    data = np.asarray(data_ch_samples, float)
    if data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError("LSL数据须为有限(channels, samples)数组")
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("window_s须为正、onset_s须非负")
    start, length = round(onset_s * fs), sample_count(window_s, fs)
    if length < 32 or start + length > data.shape[1]:
        raise ValueError("数据不足或窗口过短；不得补零凑齐窗口")
    window = data[:, start:start + length].copy()
    _normalised_rows(window)  # 对原始退化输入显式报错。
    if not math.isclose(fs, target_fs, rel_tol=0, abs_tol=1e-9):
        ratio = Fraction(target_fs / fs).limit_denominator(100000)
        actual_fs = fs * ratio.numerator / ratio.denominator
        if not math.isclose(actual_fs, target_fs, rel_tol=1e-8):
            raise ValueError("无法得到准确的重采样比")
        window = resample_poly(window, ratio.numerator, ratio.denominator, axis=-1)
    wanted = sample_count(window_s, target_fs)
    if window.shape[1] < wanted:
        raise ValueError("重采样后样本不足，不补零")
    window = window[:, :wanted]
    if notch_hz is not None:
        if not np.isfinite(notch_hz) or not 0 < notch_hz < target_fs / 2:
            raise ValueError("陷波频率须位于0与Nyquist之间")
        bn, an = iirnotch(notch_hz, 30.0, fs=target_fs)
        window = filtfilt(bn, an, window, axis=-1)
    bank = [sosfiltfilt(cheby1(4, .5, [low, high], btype="bandpass", fs=target_fs,
                             output="sos"), window, axis=-1) for low, high in M3_BANDS]
    return window, np.stack(bank)


def classify_eeg_window(data_ch_samples: np.ndarray, fs: float,
                        window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                        target_frequencies_hz: Optional[Sequence[float]] = None,
                        n_harmonics: int = N_HARMONICS, *, target_fs: float = TARGET_FS,
                        notch_hz: Optional[float] = 50.0, a: float = 1.25, b: float = .25,
                        regularization: float = 1e-10, min_valid_channels: int = 1) -> dict:
    """只接收EEG/算法参数，不接受真实目标，防止标签泄漏。

    FBCCA和标准CCA共享完全相同的EEG窗、通道、参考和第一子带。
    分数不是概率；本方法仍强制选择最高分，没有空闲检测或自由输入可靠性保证。
    """
    frequencies = (BENCHMARK_FREQUENCIES_HZ if target_frequencies_hz is None
                   else np.asarray(target_frequencies_hz, float))
    _validate_fs(fs)
    _validate_fs(target_fs)
    data = np.asarray(data_ch_samples, float)
    if data.ndim != 2 or not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("EEG须二维，窗长为正，偏移非负")
    start, length = round(onset_s * fs), sample_count(window_s, fs)
    if length < 32 or start + length > data.shape[1]:
        raise ValueError("EEG时间窗覆盖不足")
    normalized, active = _normalised_rows(data[:, start:start + length])
    if isinstance(min_valid_channels, bool) or not isinstance(min_valid_channels, int) or min_valid_channels < 1:
        raise ValueError("min_valid_channels须为正整数")
    if len(active) < min_valid_channels:
        raise DegenerateSignalError(f"有效变化通道仅{len(active)}个，至少需要{min_valid_channels}个")
    # 输入窗口已裁好，后续不再重复裁剪或加140ms。单位标准化先于滤波。
    window, bank = preprocess_lsl_window(normalized, fs, window_s, 0.0, target_fs, notch_hz)
    refs = generate_reference_signals(frequencies, target_fs, window.shape[-1], n_harmonics)
    scores, rho, weights = fbcca_fusion(bank, refs, a, b, regularization)
    best = int(np.argmax(scores))
    cca_scores = rho[0] ** 2
    cca_best = int(np.argmax(cca_scores))
    return {
        "prediction": best + 1, "frequency_hz": float(frequencies[best]),
        "scores": scores, "correlations": rho, "weights": weights,
        "cca_prediction": cca_best + 1, "cca_frequency_hz": float(frequencies[cca_best]),
        "cca_scores": cca_scores, "active_channels": active,
        "dropped_channels": np.setdiff1d(np.arange(data.shape[0]), active),
        "window_shape": window.shape, "bank_shape": bank.shape, "reference_shape": refs.shape,
        "processing_fs": target_fs,
    }


# ======================== 连续采集与事件截窗 ========================
@dataclass
class Epoch:
    data: np.ndarray
    fs: float
    requested_start: float
    requested_end: float
    raw_timestamps: np.ndarray
    local_timestamps: np.ndarray
    clock_corrections: np.ndarray
    diagnostics: dict[str, Any]


class TimestampBuffer:
    """有界环形缓冲：样本、原始时间、校正时间、校正量和有效性均保留。

    重复/逆序时间戳不排序、不删除：明确失败，防止伪造连续数据。
    NaN样本保留并打标，仅令覆盖它的试次无效。
    """
    def __init__(self, capacity: int, n_channels: int):
        if capacity < 2 or n_channels < 1:
            raise ValueError("缓冲容量/通道数不合法")
        self.capacity, self.n_channels = capacity, n_channels
        self.samples = np.empty((capacity, n_channels))
        self.raw_ts = np.empty(capacity)
        self.local_ts = np.empty(capacity)
        self.corrections = np.empty(capacity)
        self.valid = np.zeros(capacity, dtype=bool)
        self.cursor = self.size = self.total_received = 0
        self.last_raw = self.last_local = -math.inf
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray, raw_ts: np.ndarray, correction: float,
               device_lag_s: float = 0.0) -> None:
        x, raw = np.asarray(samples, float), np.asarray(raw_ts, float)
        if raw.ndim != 1 or x.ndim != 2 or x.shape != (len(raw), self.n_channels):
            raise ValueError("EEG样本/时间戳维度不一致")
        if not len(raw):
            return
        if not np.isfinite(raw).all() or not np.isfinite(correction) or not np.isfinite(device_lag_s):
            raise InvalidTrial("非有限时间戳/校正量")
        local = raw + correction - device_lag_s
        with self.lock:
            # 将时间轴异常拆开诊断，便于区分“发布端原始时间戳问题”与
            # “接收端时间校正问题”。仍然保持严格原则：不排序、不静默丢弃。
            raw_diff = np.diff(raw)
            if np.any(raw_diff <= 0):
                bad = int(np.flatnonzero(raw_diff <= 0)[0])
                raise InvalidTrial(
                    "LSL单个chunk内部出现重复/逆序原始时间戳："
                    f"index={bad}->{bad + 1}, previous={raw[bad]:.9f}, "
                    f"current={raw[bad + 1]:.9f}, delta={raw_diff[bad]:.9f}s"
                )
            if raw[0] <= self.last_raw:
                raise InvalidTrial(
                    "LSL跨chunk原始时间戳回退/重复："
                    f"previous_last={self.last_raw:.9f}, current_first={raw[0]:.9f}, "
                    f"delta={raw[0] - self.last_raw:.9f}s"
                )

            local_diff = np.diff(local)
            if np.any(local_diff <= 0):
                bad = int(np.flatnonzero(local_diff <= 0)[0])
                raise InvalidTrial(
                    "校正后单个chunk内部时间戳非递增："
                    f"index={bad}->{bad + 1}, previous={local[bad]:.9f}, "
                    f"current={local[bad + 1]:.9f}, delta={local_diff[bad]:.9f}s, "
                    f"fixed_correction={correction:.9f}s"
                )
            if local[0] <= self.last_local:
                raise InvalidTrial(
                    "校正后跨chunk时间轴回退/重复："
                    f"previous_last={self.last_local:.9f}, current_first={local[0]:.9f}, "
                    f"delta={local[0] - self.last_local:.9f}s, "
                    f"fixed_correction={correction:.9f}s"
                )

            self.last_raw, self.last_local = float(raw[-1]), float(local[-1])
            self.total_received += len(raw)
            if len(raw) >= self.capacity:
                x, raw, local = x[-self.capacity:], raw[-self.capacity:], local[-self.capacity:]
            n = len(raw)
            indices = (self.cursor + np.arange(n)) % self.capacity
            self.samples[indices], self.raw_ts[indices], self.local_ts[indices] = x, raw, local
            self.corrections[indices] = correction
            self.valid[indices] = np.isfinite(x).all(axis=1)
            self.cursor = (self.cursor + n) % self.capacity
            self.size = min(self.capacity, self.size + n)

    @property
    def latest(self) -> float:
        with self.lock:
            return self.last_local

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            indices = (self.cursor - self.size + np.arange(self.size)) % self.capacity
            return (self.samples[indices].copy(), self.raw_ts[indices].copy(),
                    self.local_ts[indices].copy(), self.corrections[indices].copy(), self.valid[indices].copy())

    def epoch(self, start: float, duration_s: float, nominal_fs: float,
              max_gap_factor: float = 1.5, rate_tolerance: float = .02) -> Epoch:
        _validate_fs(nominal_fs)
        if not np.isfinite(start) or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("事件时间/窗口长度不合法")
        end = start + duration_s
        x, raw, local, corrections, valid = self.snapshot()
        # 需要两侧插值支撑且至少到完整窗口终点；绝不使用np.interp的端点外推。
        if len(local) < 2 or local[0] > start or local[-1] < end:
            raise InvalidTrial("事件窗口数据未到齐或已被缓冲覆盖")
        left = int(np.searchsorted(local, start, side="right") - 1)
        right = int(np.searchsorted(local, end, side="left"))
        x, raw, local, corrections, valid = (v[left:right + 1] for v in (x, raw, local, corrections, valid))
        if len(local) < 3 or not np.all(valid):
            raise InvalidTrial("窗口含NaN/Inf或有效样本不足")
        dt_raw, dt_local = np.diff(raw), np.diff(local)
        nominal_dt = 1 / nominal_fs
        if np.any(dt_raw <= 0) or np.any(dt_local <= 0):
            raise InvalidTrial("窗口时间戳不严格递增")
        if max(float(dt_raw.max()), float(dt_local.max())) > nominal_dt * max_gap_factor:
            raise InvalidTrial("窗口存在缺样/过大时间戳抖动；拒绝跨缺口插值")
        measured_fs = (len(local) - 1) / (local[-1] - local[0])
        raw_measured_fs = (len(raw) - 1) / (raw[-1] - raw[0])
        if abs(measured_fs / nominal_fs - 1) > rate_tolerance:
            raise InvalidTrial(f"时间戳估计采样率{measured_fs:.3f}与声明{nominal_fs:g} Hz不符")
        _validate_fs(min(measured_fs, raw_measured_fs))
        grid = start + np.arange(sample_count(duration_s, nominal_fs)) / nominal_fs
        if grid.size < 32 or grid[-1] > local[-1] or grid[0] < local[0]:
            raise InvalidTrial("窗口不满足重采样时间覆盖")
        # 仅在已验证无明确缺样后，把轻微时间抖动对齐到统一时间网格。
        # 保留高原始采样率后交给resample_poly抗混叠降采样，不能直接插值到250 Hz。
        uniform = np.vstack([np.interp(grid, local, channel) for channel in x.T])
        return Epoch(uniform, nominal_fs, start, end, raw.copy(), local.copy(), corrections.copy(), {
            "source_support_samples": len(local), "uniform_samples": len(grid),
            "timestamp_estimated_fs": float(measured_fs), "source_clock_estimated_fs": float(raw_measured_fs),
            "max_source_interval_s": float(dt_raw.max()), "max_local_interval_s": float(dt_local.max()),
            "source_support_start": float(local[0]), "source_support_end": float(local[-1]),
            "uniform_last_sample": float(grid[-1]), "end_is_exclusive": True,
            "clock_correction_min_s": float(corrections.min()), "clock_correction_max_s": float(corrections.max()),
            "interpolation": "timestamp-grid alignment only; gaps rejected; no extrapolation",
        })


def _parse_stream_metadata(info: Any, indices: Sequence[int]) -> dict:
    xml = info.as_xml()
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise RuntimeError("无法解析LSL流元数据") from exc
    channels = root.findall("./desc/channels/channel")
    selected = []
    for i in indices:
        node = channels[i] if i < len(channels) else None
        selected.append({"lsl_index": i, "label": node.findtext("label", "UNKNOWN") if node is not None else "UNKNOWN",
                         "unit": node.findtext("unit", "UNKNOWN") if node is not None else "UNKNOWN"})
    # 仅显示发布端原样声明；不从任意标签推断真实硬件滤波设置。
    filter_nodes = []
    for element in root.findall("./desc//*"):
        if any(s in element.tag.lower() for s in ("filter", "lowpass", "highpass", "notch")):
            text = " ".join(element.itertext()).strip()
            if text:
                filter_nodes.append(f"{element.tag}: {text}")
    return {"name": info.name(), "type": info.type(), "source_id": info.source_id(),
            "channel_count": info.channel_count(), "nominal_fs": float(info.nominal_srate()),
            "selected_channels": selected, "publisher_filter_metadata": filter_nodes or ["UNKNOWN"],
            "raw_xml": xml}


class ContinuousLSL:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inlet: Any = None
        self.buffer: Optional[TimestampBuffer] = None
        self.metadata: dict = {}
        self.fs = math.nan
        self.error: Optional[BaseException] = None
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.clock: Callable[[], float] = time.monotonic
        self.clock_updates: list[tuple[float, float]] = []
        self.estimated_fs = math.nan

    def start(self) -> None:
        try:
            import pylsl
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("不能加载pylsl/liblsl。先安装pylsl；Linux还需可加载的liblsl。见README。") from exc
        self.clock = pylsl.local_clock
        prop = ("name", self.cfg.eeg_stream_name) if self.cfg.eeg_stream_name else ("type", self.cfg.eeg_stream_type)
        # OpenBCI 通常只发布一条 EEG 流；minimum=2 会让 pylsl 在单流时
        # 等到超时并返回空列表，导致程序误报“未找到 LSL 流”。先允许单流
        # 正常连接，再对实际返回的结果做唯一性检查。
        streams = pylsl.resolve_byprop(*prop, minimum=1, timeout=self.cfg.connect_timeout_s)
        if not streams:
            # 失败时顺便列出当前可见流，避免用户只能看到“设备已连接”却不知道
            # 实际没有启动 OpenBCI 的 LSL 发布端，或发布端的 type/name 不匹配。
            visible = pylsl.resolve_streams(wait_time=min(1.0, self.cfg.connect_timeout_s))
            details = "; ".join(
                f"{stream.name()!r}(type={stream.type()!r}, channels={stream.channel_count()}, "
                f"fs={stream.nominal_srate():g}Hz)"
                for stream in visible
            ) or "无"
            raise RuntimeError(
                f"未找到LSL流 {prop}。当前可见流：{details}\n"
                "设备连接成功不代表已开启LSL输出。本程序通过LSL接收EEG，不直接连接USB设备。\n"
                "若使用OpenBCI GUI：先Start Data Stream，再在Networking选择LSL，"
                "数据选择TimeSeriesRaw、Type填EEG，点击Start LSL Stream，保持GUI运行后重试。"
            )
        if len(streams) != 1:
            names = ", ".join(repr(stream.name()) for stream in streams)
            raise RuntimeError(
                f"找到多个匹配的EEG流（{names}），请在CONFIG.eeg_stream_name填写唯一名称"
            )
        # 保留原始时间戳并手动+time_correction恰好一次；不启用clocksync/monotonize/dejitter。
        self.inlet = pylsl.StreamInlet(streams[0], max_buflen=math.ceil(self.cfg.buffer_s),
                                      recover=False, processing_flags=0)
        try:
            info = self.inlet.info(timeout=self.cfg.connect_timeout_s)
            self.fs = float(info.nominal_srate())
            _validate_fs(self.fs)
            if max(self.cfg.channel_indices) >= info.channel_count():
                raise RuntimeError("所选LSL通道索引超出实际流通道数")
            self.metadata = _parse_stream_metadata(info, self.cfg.channel_indices)
            self.inlet.open_stream(timeout=self.cfg.connect_timeout_s)
            correction = float(self.inlet.time_correction(timeout=self.cfg.connect_timeout_s))
            self.clock_updates.append((self.clock(), correction))
            self.buffer = TimestampBuffer(math.ceil(self.cfg.buffer_s * self.fs * 1.05), len(self.cfg.channel_indices))
            self.thread = threading.Thread(target=self._run, args=(correction,), name="EEG-LSL-collector", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            self.check_health()
            _, _, local, _, _ = self.buffer.snapshot()
            if len(local) > 2 and local[-1] - local[0] >= self.cfg.warmup_s:
                self.estimated_fs = (len(local) - 1) / (local[-1] - local[0])
                if abs(self.estimated_fs / self.fs - 1) > self.cfg.rate_tolerance:
                    raise RuntimeError(f"启动实测时间戳速率{self.estimated_fs:.3f}与声明{self.fs:g} Hz不符")
                _validate_fs(self.estimated_fs)
                return
            self.stop_event.wait(.02)
        raise RuntimeError("启动阶段未收到足够的连续EEG数据")

    def _run(self, correction: float) -> None:
        assert self.inlet is not None and self.buffer is not None

        # 快速调试模式：优先保证 OpenBCI -> LSL -> 程序链路可以进入界面。
        # OpenBCI GUI 的 LSL 原始 timestamp 偶尔会在单个 chunk 内轻微回退。
        # 在 quick_entry_mode 下不再因此终止，而是保持 EEG 样本原有顺序，按
        # LSL 声明的 nominal sampling rate 构造连续时间轴。
        # 这不是最终实验级时间同步方案，仅用于当前联调/进入程序。
        timestamp_correction = float(correction)
        observed_correction = float(correction)
        synthetic_next_raw: Optional[float] = None
        warned_timestamp_rebuild = False
        warned_clock = False
        warned_age = False

        next_clock_update = time.monotonic() + self.cfg.clock_refresh_s
        last_arrival = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if time.monotonic() >= next_clock_update:
                    if self.cfg.quick_entry_mode:
                        # 仅观测，不让时钟微调阻止进入程序。
                        try:
                            if hasattr(self.inlet, "was_clock_reset") and self.inlet.was_clock_reset():
                                if not warned_clock:
                                    print("[快速模式警告] LSL报告源时钟重置；当前联调继续运行。", flush=True)
                                    warned_clock = True
                            new_correction = float(self.inlet.time_correction(timeout=.1))
                            self.clock_updates.append((self.clock(), new_correction))
                            observed_correction = new_correction
                        except Exception as exc:
                            if not warned_clock:
                                print(f"[快速模式警告] time_correction监控失败，当前联调继续：{exc}", flush=True)
                                warned_clock = True
                    else:
                        if hasattr(self.inlet, "was_clock_reset") and self.inlet.was_clock_reset():
                            raise InvalidTrial("LSL源时钟已重置，必须重新建立实验同步")
                        new_correction = float(self.inlet.time_correction(timeout=.1))
                        correction_step = new_correction - observed_correction
                        if abs(correction_step) > self.cfg.max_clock_step_s:
                            raise InvalidTrial(
                                "LSL时钟校正突变超过阈值，停止以避免错配试次："
                                f"previous={observed_correction:.9f}s, current={new_correction:.9f}s, "
                                f"step={correction_step:.9f}s, limit={self.cfg.max_clock_step_s:.9f}s"
                            )
                        observed_correction = new_correction
                        self.clock_updates.append((self.clock(), observed_correction))
                    next_clock_update = time.monotonic() + self.cfg.clock_refresh_s

                samples, ts = self.inlet.pull_chunk(
                    timeout=.05, max_samples=max(32, math.ceil(self.fs / 10))
                )
                if not ts:
                    if time.monotonic() - last_arrival > self.cfg.max_receive_age_s:
                        raise InvalidTrial("EEG流停止发送，已停止实验")
                    continue

                last_arrival = time.monotonic()
                x = np.asarray(samples, float)
                if x.ndim != 2 or x.shape[1] <= max(self.cfg.channel_indices):
                    raise InvalidTrial("EEG流通道数或形状发生变化")

                incoming_ts = np.asarray(ts, dtype=float)
                if self.cfg.quick_entry_mode:
                    if not np.isfinite(incoming_ts).all():
                        # 第一个锚点都不可用时，以接收端LSL时钟反推源时钟域。
                        anchor = self.clock() - timestamp_correction
                    elif synthetic_next_raw is None:
                        anchor = float(incoming_ts[0])
                    else:
                        anchor = float(synthetic_next_raw)

                    # 始终按样本顺序构造连续时间轴；不排序、不删除EEG样本。
                    repaired_ts = anchor + np.arange(len(incoming_ts), dtype=float) / self.fs
                    synthetic_next_raw = float(repaired_ts[-1] + 1.0 / self.fs)

                    # 仅第一次发现发布端原始timestamp异常时打印提示。
                    if len(incoming_ts) > 1:
                        bad = np.any(np.diff(incoming_ts) <= 0)
                    else:
                        bad = False
                    if bad and not warned_timestamp_rebuild:
                        print(
                            "[快速模式] 检测到OpenBCI/LSL原始timestamp逆序；"
                            "当前按nominal sampling rate重建连续时间轴以优先进入程序。",
                            flush=True,
                        )
                        warned_timestamp_rebuild = True
                    used_ts = repaired_ts
                else:
                    used_ts = incoming_ts

                self.buffer.append(
                    x[:, self.cfg.channel_indices], used_ts, timestamp_correction,
                    self.cfg.device_timestamp_lag_s
                )

                age = self.clock() - self.buffer.latest
                if age > self.cfg.max_receive_age_s or age < -.1:
                    if self.cfg.quick_entry_mode:
                        if not warned_age:
                            print(
                                f"[快速模式警告] EEG时间轴与本地LSL时钟差{age:.3f}s；"
                                "当前联调不终止。", flush=True
                            )
                            warned_age = True
                    else:
                        raise InvalidTrial(
                            f"EEG时间轴不在合理本地LSL范围，样本年龄{age:.3f}s；检查发布端时间戳"
                        )
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def check_health(self) -> None:
        if self.error is not None:
            raise InvalidTrial(f"采集线程异常：{self.error}") from self.error
        if self.stop_event.is_set():
            raise InvalidTrial("采集已停止")

    def wait_epoch(self, onset: float, cancel: threading.Event) -> Epoch:
        assert self.buffer is not None
        start = onset + self.cfg.response_delay_s
        end = start + self.cfg.window_s
        deadline = time.monotonic() + self.cfg.data_wait_timeout_s
        while self.buffer.latest < end:
            if cancel.is_set():
                raise AbortSession("已取消等待EEG")
            self.check_health()
            if time.monotonic() >= deadline:
                raise InvalidTrial(f"EEG未覆盖窗口终点{end:.6f}，该试次不分类")
            cancel.wait(.01)
        self.check_health()
        return self.buffer.epoch(start, self.cfg.window_s, self.fs, self.cfg.max_gap_factor, self.cfg.rate_tolerance)

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        # 不在pull_chunk尚在执行时从另一线程关闭inlet。
        if self.inlet is not None and (self.thread is None or not self.thread.is_alive()):
            self.inlet.close_stream()
            self.inlet = None


def decode_trial(trial_id: int, onset: float, collector: ContinuousLSL, cfg: Config,
                 cancel: threading.Event) -> dict:
    """分类工作线程；函数签名有意不包含真实类别。"""
    epoch = collector.wait_epoch(onset, cancel)
    if cancel.is_set():
        raise AbortSession("已取消分类")
    started = time.monotonic()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = classify_eeg_window(epoch.data, epoch.fs, window_s=cfg.window_s, onset_s=0.0,
            target_frequencies_hz=BENCHMARK_FREQUENCIES_HZ, n_harmonics=cfg.n_harmonics,
            target_fs=cfg.target_fs, notch_hz=cfg.notch_hz, a=cfg.weight_a, b=cfg.weight_b,
            regularization=cfg.cca_regularization, min_valid_channels=cfg.min_valid_channels)
    result.update(trial_id=trial_id, epoch=epoch, computation_s=time.monotonic() - started)
    return result


# ======================== 翻转事件与内存评估 ========================
@dataclass
class EventStamp:
    kind: str
    trial_id: int
    target_id: Optional[int] = None
    details: dict = field(default_factory=dict)
    timestamp: Optional[float] = None


class EventMarkers:
    """callOnFlip回调只记本地LSL时间并入队；网络发送不占用呈现线程。

    EEG截窗直接使用这个事件对象的时间，不经由Marker流绕一圈，也不重复校时。
    Marker线程随后仍以原始事件时间作为LSL timestamp，而非发送时刻。
    """
    def __init__(self, cfg: Config, clock: Callable[[], float], outlet: Any = None):
        self.clock, self.error = clock, None
        self.events: list[EventStamp] = []
        self.pending: queue.SimpleQueue = queue.SimpleQueue()
        self.closed = False
        if outlet is None:
            import pylsl
            info = pylsl.StreamInfo(cfg.marker_stream_name, "Markers", 1, 0,
                                   "string", f"fbcca-keyboard-{uuid.uuid4()}")
            desc = info.desc()
            desc.append_child_value("clock", "local_clock in PsychoPy callOnFlip")
            desc.append_child_value("physical_display_latency", "UNMEASURED")
            desc.append_child_value("event_delivery", "queued; original timestamp preserved")
            target_node = desc.append_child("targets")
            for target in TARGETS:
                node = target_node.append_child("target")
                for key, value in vars(target).items():
                    node.append_child_value(key, str(value))
            outlet = pylsl.StreamOutlet(info, chunk_size=1)
        self.outlet = outlet
        self.thread = threading.Thread(target=self._send, name="LSL-markers", daemon=True)
        self.thread.start()

    def mark(self, event: EventStamp) -> None:
        if self.closed or event.timestamp is not None:
            raise RuntimeError("事件重复标记或事件模块已经关闭")
        event.timestamp = self.clock()
        self.events.append(event)
        self.pending.put(event)

    def _send(self) -> None:
        while True:
            event = self.pending.get()
            if event is None:
                return
            try:
                payload = json.dumps({"event": event.kind, "trial_id": event.trial_id,
                    "target_id": event.target_id, "details": event.details}, ensure_ascii=True)
                self.outlet.push_sample([payload], timestamp=event.timestamp)
            except Exception as exc:
                self.error = exc

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = RuntimeError("Marker发送线程未及时退出")


@dataclass
class TrialRecord:
    trial_id: int
    block_id: int
    true_class: Optional[int]  # 自由输入必须为None，绝不把预测充当真实标签。
    status: str = "started"
    reason: str = ""
    start: float = 0.0
    end: float = 0.0
    cue_onset: Optional[float] = None
    stimulus_onset: Optional[float] = None
    stimulus_offset: Optional[float] = None
    phase_times: list[tuple[str, float]] = field(default_factory=list)
    frame_flip_times: list[float] = field(default_factory=list)
    frame_diagnostics: dict = field(default_factory=dict)
    result: Optional[dict] = None
    text_applied: bool = False
    mode: str = "cued"
    requested_window_start: Optional[float] = None
    requested_window_end: Optional[float] = None
    actual_stimulus_s: Optional[float] = None


class TrialLedger:
    """每个trial_id最多提交一次；有效和无效试次都计数。"""
    def __init__(self):
        self.records: list[TrialRecord] = []
        self.ids: set[int] = set()
        self.typed_text = ""

    def apply_prediction(self, pred: int) -> None:
        if not 1 <= pred <= len(TARGETS):
            raise ValueError("预测类别超出范围")
        symbol = TARGETS[pred - 1].symbol
        if symbol == "BACK":
            self.typed_text = self.typed_text[:-1]
        elif symbol == "SPACE":
            self.typed_text += " "
        else:
            self.typed_text += symbol

    def commit(self, record: TrialRecord) -> None:
        if record.trial_id in self.ids:
            raise RuntimeError(f"试次{record.trial_id}重复提交")
        if record.status not in ("valid", "invalid", "aborted"):
            raise RuntimeError("未结束的试次不能提交")
        if record.mode == "free":
            if record.true_class is not None:
                raise ValueError("自由输入没有真实目标，true_class必须为None")
        elif record.mode == "cued":
            if record.true_class is None or not 1 <= record.true_class <= len(TARGETS):
                raise ValueError("提示测试的真实类别超出范围")
        else:
            raise ValueError("未知输入模式")
        if record.status == "valid":
            if record.result is None or record.result.get("trial_id") != record.trial_id:
                raise RuntimeError("分类结果缺失或属于其他试次")
            pred = record.result["prediction"]
            if not 1 <= pred <= len(TARGETS) or not 1 <= record.result["cca_prediction"] <= len(TARGETS):
                raise ValueError("分类结果超出范围")
            if not record.text_applied:
                self.apply_prediction(pred)
                record.text_applied = True
        self.ids.add(record.trial_id)
        self.records.append(record)


def frame_diagnostics(flip_times: Sequence[float], refresh_hz: float,
                      long_factor: float = 1.5, short_factor: float = .5) -> dict:
    ts = np.asarray(flip_times, float)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or refresh_hz <= 0:
        raise InvalidTrial("刺激帧时间记录不足或非法")
    intervals = np.diff(ts)
    period = 1 / refresh_hz
    long = intervals > long_factor * period
    short = intervals < short_factor * period
    missed = np.maximum(1, np.rint(intervals[long] / period).astype(int) - 1).sum() if np.any(long) else 0
    return {"intervals_s": intervals, "long_intervals": int(long.sum()), "short_intervals": int(short.sum()),
            "estimated_missed_frames": int(missed), "min_interval_ms": float(intervals.min() * 1000),
            "median_interval_ms": float(np.median(intervals) * 1000),
            "max_interval_ms": float(intervals.max() * 1000),
            "valid": not bool(np.any(long | short)), "refresh_hz": refresh_hz}


def make_schedule(blocks: int, seed: int) -> list[tuple[int, int]]:
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError("blocks必须是正整数")
    rng = np.random.default_rng(seed)
    return [(b + 1, int(class_id)) for b in range(blocks)
            for class_id in rng.permutation(np.arange(1, len(TARGETS) + 1))]


def make_luminance_table(refresh_hz: float, duration_s: float) -> np.ndarray:
    if not np.isfinite(refresh_hz) or refresh_hz <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
        raise ValueError("刷新率无法支持当前最高刺激频率")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("刺激时长必须为正")
    frames = max(1, round(duration_s * refresh_hz))
    t = np.arange(frames)[:, None] / refresh_hz
    phases = np.array([target.phase_rad for target in TARGETS])[None, :]
    return .5 * (1 + np.sin(2 * np.pi * t * BENCHMARK_FREQUENCIES_HZ[None, :] + phases))


def itr_bits_per_minute(p: float, seconds_per_selection: float, n: int = 40) -> float:
    """均匀先验/对称错误模型的ITR估计；低于机会水平记0，不能当成实测互信息。"""
    if not 0 <= p <= 1 or n < 2 or seconds_per_selection <= 0:
        raise ValueError("ITR参数不合法")
    if p <= 1 / n:
        return 0.0
    bits = math.log2(n)
    if p < 1:
        bits += p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return max(0.0, bits * 60 / seconds_per_selection)


def summarize_trials(records: Sequence[TrialRecord], planned: int) -> dict:
    if any(r.mode != "cued" or r.true_class is None for r in records):
        raise ValueError("自由输入没有真实标签，不可计算提示测试准确率")
    attempted = len(records)
    valid = [r for r in records if r.status == "valid"]
    n = len(TARGETS)
    counts, valid_counts, invalid_counts = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    fb_cm, cca_cm = np.zeros((n, n), int), np.zeros((n, n), int)
    for record in records:
        t = record.true_class - 1
        counts[t] += 1
        if record.status == "valid":
            if record.result is None:
                raise ValueError("有效试次缺失结果")
            valid_counts[t] += 1
            fb_cm[t, record.result["prediction"] - 1] += 1
            cca_cm[t, record.result["cca_prediction"] - 1] += 1
        else:
            invalid_counts[t] += 1
    active_s = sum(max(0.0, r.end - r.start) for r in records)
    wall_s = max((r.end for r in records), default=0) - min((r.start for r in records), default=0)
    report = {"planned": planned, "attempted": attempted, "not_started": max(0, planned - attempted),
        "valid": len(valid), "invalid_or_aborted": attempted - len(valid),
        "active_selection_s": active_s, "wall_s_including_breaks": max(0, wall_s),
        "mean_actual_selection_s": active_s / attempted if attempted else None,
        "target_attempts": counts, "target_valid": valid_counts, "target_invalid": invalid_counts,
        "fbcca_confusion": fb_cm, "cca_confusion": cca_cm}
    for name, cm in (("fbcca", fb_cm), ("cca", cca_cm)):
        correct = int(np.trace(cm))
        success_all = correct / attempted if attempted else None
        report[name] = {"correct": correct, "accuracy_valid": correct / len(valid) if valid else None,
            "success_all_attempted": success_all,
            "itr_conservative_actual_bpm": itr_bits_per_minute(success_all, active_s / attempted, n)
                if attempted and active_s > 0 else None,
            "itr_conservative_wall_bpm": itr_bits_per_minute(success_all, wall_s / attempted, n)
                if attempted and wall_s > 0 else None}
    return report


def _pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def print_summary(report: dict, ledger: TrialLedger) -> None:
    if not ledger.records:
        print("尚未开始试次，请先处理上方的结束原因。", flush=True)
        return
    print("\n" + "=" * 76)
    print("提示式40分类结果（非自由输入；不保存文件）")
    print(f"计划{report['planned']}次 | 已开始{report['attempted']} | 有效{report['valid']} | "
          f"无效/中止{report['invalid_or_aborted']} | 未开始{report['not_started']}")
    mean_time = report['mean_actual_selection_s']
    if mean_time is not None:
        print(f"实际每次选择平均{mean_time:.3f}s，包含提示、刺激、等待、计算、空白和反馈。")
        print(f"试次内总时间{report['active_selection_s']:.3f}s；含区块休息/试次间开销墙钟{report['wall_s_including_breaks']:.3f}s。")
    for name in ("fbcca", "cca"):
        stats = report[name]
        print(f"{name.upper()}: 正确{stats['correct']} | 有效试次准确率 {_pct(stats['accuracy_valid'])} | "
              f"全部已开始试次成功率 {_pct(stats['success_all_attempted'])}")
        if stats['itr_conservative_actual_bpm'] is not None:
            print(f"  ITR模型估计（无效视为不成功、实际试次耗时）{stats['itr_conservative_actual_bpm']:.3f} bit/min；"
                  f"含休息{stats['itr_conservative_wall_bpm']:.3f} bit/min")
    print("\n逐目标：ID  字符    Hz  尝试 有效 无效 FBCCA正确 CCA正确 FBCCA有效准确率")
    for target in TARGETS:
        j = target.class_id - 1
        nv = report['target_valid'][j]
        correct = report['fbcca_confusion'][j, j]
        accuracy = correct / nv if nv else None
        print(f"{target.class_id:02d} {target.symbol:>5} {target.frequency_hz:5.1f} "
              f"{report['target_attempts'][j]:4d} {nv:4d} {report['target_invalid'][j]:4d} "
              f"{correct:9d} {report['cca_confusion'][j,j]:7d} {_pct(accuracy):>12}")
    for name in ("fbcca", "cca"):
        cm = report[f'{name}_confusion']
        print(f"\n{name.upper()} 错分明细（真实ID -> 预测ID: 次数）：")
        errors = [(i, j, cm[i, j]) for i in range(len(TARGETS)) for j in range(len(TARGETS)) if i != j and cm[i, j]]
        print("; ".join(f"{i+1:02d}->{j+1:02d}: {count}" for i, j, count in errors) or "没有已记录的有效试次错分。")
    invalid = [r for r in ledger.records if r.status != "valid"]
    for r in invalid:
        print(f"无效/中止 trial={r.trial_id} true={r.true_class:02d}: {r.reason}")
    print(f"\n键盘输出（仅本程序内存）：{ledger.typed_text!r}")
    print("完整40x40混淆矩阵、逐帧时间、事件、每窗EEG/时间戳和分数均在 LAST_SESSION 内存对象中。")
    print("这些结果不能证明屏幕物理同步、设备延迟、眼动或伪迹已验证；进程退出后数据不保留。")


def summarize_free_trials(records: Sequence[TrialRecord], ledger: TrialLedger) -> dict:
    """自由输入只报告操作记录；不虚构正确率、混淆矩阵或ITR。"""
    if any(r.mode != "free" or r.true_class is not None for r in records):
        raise ValueError("自由输入记录不应含提示目标")
    return {
        "mode": "free", "attempted": len(records),
        "valid": sum(r.status == "valid" for r in records),
        "input_actions": sum(r.text_applied for r in records),
        "invalid_or_aborted": sum(r.status != "valid" for r in records),
        "typed_text": ledger.typed_text,
        "ground_truth_available": False, "accuracy": None, "itr_bits_per_minute": None,
    }


def print_free_summary(report: dict) -> None:
    print("\n" + "=" * 76)
    print("自由输入结束（真实LSL EEG；无预设目标；文本仅在本程序中）")
    print(f"已开始{report['attempted']}轮 | 已执行{report['input_actions']}次输入操作"
          f"（包括SPACE/BACK） | 无效或未完成{report['invalid_or_aborted']}轮")
    print("最终文本：", repr(report['typed_text']))
    print("自由输入没有真实目标标签，不计算准确率、混淆矩阵或ITR。")
    print("当前快速模式未校准个人模型，也不具备自动空闲检测。离开注视前请先暂停。")


def output_display_text(text: str, max_chars: int) -> str:
    """仅裁剪显示尾部，不删减真正的输入字符串；静态|标记插入位置。"""
    if max_chars < 12:
        raise ValueError("输出显示区域过短")
    tail_capacity = max_chars - len("OUTPUT: ") - 1
    visible = text if len(text) <= tail_capacity else "..." + text[-(tail_capacity - 3):]
    return "OUTPUT: " + visible + "|"


def _canonical_unit(value: str) -> str:
    unit = str(value).strip().replace("μ", "u").replace("µ", "u").lower()
    return {"uv": "uV", "microvolt": "uV", "microvolts": "uV", "v": "V", "volt": "V", "volts": "V",
            "mv": "mV", "millivolt": "mV", "millivolts": "mV"}.get(unit, "UNKNOWN")


def preflight_review(cfg: Config, collector: ContinuousLSL) -> dict:
    """启动核验。quick_entry_mode下跳过人工勾选，优先进入程序。"""
    meta = collector.metadata
    if cfg.quick_entry_mode:
        print("\n=== 快速调试模式：跳过严格EEG启动核验对话框 ===", flush=True)
        print(f"流: {meta.get('name')} | 总通道: {meta.get('channel_count')} | "
              f"声明采样率: {collector.fs:g} Hz | 当前时间轴估计: {collector.estimated_fs:.3f} Hz", flush=True)
        print("当前目标是先进入40目标键盘并验证完整链路；最终实验前再恢复严格核验。", flush=True)
        return {
            "quick_entry_mode": True,
            "wiring_user_reviewed": False,
            "input_units": ["UNVERIFIED"] * len(cfg.channel_indices),
            "unit_override": cfg.input_unit,
            "upstream_filter_description": cfg.upstream_filter_description,
            "upstream_lowpass_hz": cfg.upstream_lowpass_hz,
            "receiver_notch_hz": cfg.notch_hz,
            "unverified": ["快速调试模式：接线/单位/滤波/物理时延暂未做最终实验级核验"],
            "device_timestamp_lag_applied_s": cfg.device_timestamp_lag_s,
        }
    from psychopy import gui
    print("\n=== EEG启动核验 ===")
    print(f"流: {meta['name']} | 来源ID: {meta['source_id']} | 总通道: {meta['channel_count']}")
    print(f"声明原始采样率: {collector.fs:g} Hz | 启动时间戳估计: {collector.estimated_fs:.4f} Hz")
    print("以下LSL列到设备CH的映射需要核对；软件标签不是实际帽位：")
    for j, position in enumerate(cfg.channel_positions):
        channel = meta['selected_channels'][j]
        print(f"分析通道{j+1}: {position:3s} <- LSL[{channel['lsl_index']}] "
              f"(假定设备CH{channel['lsl_index']+1}); 流标签={channel['label']!r}, 单位={channel['unit']!r}")
    print("发布端过滤元数据：", "; ".join(meta['publisher_filter_metadata']))
    print("软件校时：raw EEG timestamp + inlet.time_correction() - 已校准设备滞后(默认未应用)。")
    print("未知单位/滤波不会被猜测为已确认。已知上游低通<90Hz时拒绝完整M3。")
    choices = list(dict.fromkeys([cfg.input_unit, "AUTO", "uV", "V", "mV", "UNKNOWN"]))
    fields = {
        "Stream / nominal / observed Hz": f"{meta['name']} / {collector.fs:g} / {collector.estimated_fs:.3f}",
        "Channel map (see terminal)": "  ".join(f"{p}<-LSL[{i}]" for p, i in zip(cfg.channel_positions, cfg.channel_indices)),
        "Input unit override": choices,
        "Publisher filtering (write UNKNOWN if not known)": cfg.upstream_filter_description,
        "Known publisher lowpass Hz (blank if unknown)": "" if cfg.upstream_lowpass_hz is None else str(cfg.upstream_lowpass_hz),
        "Receiver notch Hz": ["50", "60", "OFF"] if cfg.notch_hz == 50 else list(dict.fromkeys([
            "OFF" if cfg.notch_hz is None else str(cfg.notch_hz), "50", "60", "OFF"])),
        "I checked selected LSL columns against actual wiring": False,
        "I understand UNKNOWN metadata means unverified conditions": False,
        "I read the flicker warning printed in the terminal": False,
    }
    dialog = gui.DlgFromDict(fields, title="FBCCA acquisition review / No flicker yet", sortKeys=False,
        fixed=["Stream / nominal / observed Hz", "Channel map (see terminal)"])
    if not dialog.OK:
        raise AbortSession("取消启动核验")
    if not all(fields[key] for key in ("I checked selected LSL columns against actual wiring",
        "I understand UNKNOWN metadata means unverified conditions",
        "I read the flicker warning printed in the terminal")):
        raise AbortSession("尚未完成接线/未知信息/闪烁风险核验；本次不启动刺激")
    cfg.input_unit = str(fields['Input unit override'])
    cfg.upstream_filter_description = str(fields['Publisher filtering (write UNKNOWN if not known)']).strip() or "UNKNOWN"
    lowpass = str(fields['Known publisher lowpass Hz (blank if unknown)']).strip()
    cfg.upstream_lowpass_hz = float(lowpass) if lowpass else None
    notch = fields['Receiver notch Hz']
    cfg.notch_hz = None if notch == "OFF" else float(notch)
    cfg.validate()
    units = ([_canonical_unit(c['unit']) for c in meta['selected_channels']]
             if cfg.input_unit == "AUTO" else [_canonical_unit(cfg.input_unit)] * len(cfg.channel_indices))
    unknowns = []
    if "UNKNOWN" in units:
        unknowns.append("输入单位未全部确认（CCA按通道标准化；不报告物理幅度）")
    if cfg.upstream_filter_description.upper() == "UNKNOWN" or cfg.upstream_lowpass_hz is None:
        unknowns.append("发布端滤波/90Hz有效带宽尚未完整核验")
    unknowns.extend(["屏幕真实发光与软件事件延迟未由光电二极管测量",
                     "OpenBCI设备采样到LSL时间戳的固定延迟未由本程序自动测量"])
    review = {"input_units": units, "unit_override": cfg.input_unit,
        "upstream_filter_description": cfg.upstream_filter_description, "upstream_lowpass_hz": cfg.upstream_lowpass_hz,
        "receiver_notch_hz": cfg.notch_hz, "wiring_user_reviewed": True, "unverified": unknowns,
        "device_timestamp_lag_applied_s": cfg.device_timestamp_lag_s}
    print("最终输入单位（分析通道顺序）：", units)
    print(f"人工声明发布端滤波: {cfg.upstream_filter_description}; 低通: {cfg.upstream_lowpass_hz}; 接收端陷波: {cfg.notch_hz}")
    print("未核验边界：\n  " + "\n  ".join(unknowns))
    return review


# ======================== 主线程：预创建刺激、逐帧更新、试次管理 ========================
def calculate_keyboard_layout(cfg: Config, window_size: Sequence[float]
                              ) -> tuple[np.ndarray, float, float, float]:
    """按目标表计算布局；为顶部输出/状态和底部提示预留空间，无需创建窗口。"""
    width, height = map(float, window_size)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("窗口宽高必须为有限正数")
    rows, cols = len(KEY_ROWS), max(map(len, KEY_ROWS))
    top_reserved = cfg.output_box_height_px + cfg.output_box_margin_px + 70.0
    bottom_reserved = 70.0
    scale = min(1., (width - 80) / (cols * cfg.key_size_px + (cols - 1) * cfg.key_gap_px),
                (height - top_reserved - bottom_reserved) /
                (rows * cfg.key_size_px + (rows - 1) * cfg.key_gap_px))
    if scale <= 0 or (scale < 1 and not cfg.allow_layout_scaling):
        raise RuntimeError("屏幕不足以容纳固定按键布局；请更换分辨率或允许等比例缩放")
    key, gap = cfg.key_size_px * scale, cfg.key_gap_px * scale
    center_y = (bottom_reserved - top_reserved) / 2
    positions = np.asarray([
        ((target.col - (cols - 1) / 2) * (key + gap),
         center_y + ((rows - 1) / 2 - target.row) * (key + gap))
        for target in TARGETS
    ])
    return positions, key, gap, scale


class PsychoPyKeyboard:
    def __init__(self, cfg: Config, collector: ContinuousLSL, markers: EventMarkers, ledger: TrialLedger):
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.visual, self.event = visual, event
        self.clock = collector.clock
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FBCCA-decoder")
        self.win: Any = None
        self.display_info: dict = {}
        self.warning_reason = ""
        self._pause_enabled = False
        self._active_decode_cancel: Optional[threading.Event] = None
        self._decode_future: Any = None
        self._output_cache: Optional[str] = None
        # 禁止刺激属性自动日志及重复掉帧警告占用关键时段；自己在刺激后汇报。
        logging.console.setLevel(logging.ERROR)
        try:
            self._build_window()
        except BaseException:
            self.close()
            raise

    def _build_window(self) -> None:
        visual, cfg = self.visual, self.cfg
        self.win = visual.Window(size=cfg.window_size, fullscr=cfg.full_screen, screen=cfg.screen_index,
            units="pix", color=(-.78, -.77, -.73), colorSpace="rgb", waitBlanking=True,
            allowGUI=not cfg.full_screen, checkTiming=False, autoLog=False)
        self.win.mouseVisible = False
        width, height = map(float, self.win.size)
        self.positions, key, gap, scale = calculate_keyboard_layout(cfg, (width, height))
        self.key_size, self.gap_size, self.layout_scale = key, gap, scale
        self._static_colors = np.tile((.72, .58, .82), (len(TARGETS), 1))
        # 批量绘制边框，避免逐帧增加40个Rect对象的绘制开销。
        self.key_borders = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key + 4 * scale, key + 4 * scale), elementTex=None, elementMask=None,
            colors=np.tile((-.88, -.89, -.84), (len(TARGETS), 1)), colorSpace="rgb", autoLog=False)
        self.squares = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key, key), elementTex=None, elementMask=None,
            colors=self._static_colors, colorSpace="rgb", autoLog=False)
        self.labels = []
        for target, pos in zip(TARGETS, self.positions):
            label_height = key * (.32 if len(target.display_label) > 1 else .42)
            self.labels.append(visual.TextStim(self.win, text=target.display_label, pos=pos,
                height=label_height, units="pix", color=(-.64, -.67, -.58),
                colorSpace="rgb", bold=False, autoLog=False))
        self.outline = visual.Rect(self.win, width=key + 8 * scale, height=key + 8 * scale,
            fillColor=None, lineColor="red", lineWidth=3, units="pix", autoLog=False)
        self.triangle = visual.ShapeStim(self.win, vertices=((-8*scale, -6*scale), (8*scale, -6*scale), (0, 6*scale)),
            fillColor="red", lineColor="red", units="pix", autoLog=False)
        output_y = height/2 - cfg.output_box_margin_px - cfg.output_box_height_px/2
        self.output_box = visual.Rect(self.win, width=width*.86, height=cfg.output_box_height_px,
            pos=(0, output_y), fillColor=(-0.82, -0.82, -0.82), lineColor="white", lineWidth=2,
            units="pix", autoLog=False)
        output_font_height = min(28, max(16, cfg.output_box_height_px*0.42))
        self.output_text = visual.TextStim(self.win, text="", pos=(0, output_y),
            height=output_font_height, wrapWidth=width*.80,
            font="Courier New", color="white", alignText="left", anchorHoriz="center", autoLog=False)
        # 自动绘制使顶部框在提示、闪烁、空白、反馈、暂停阶段都可见。
        # 文字只在字符串变化后重建，不在每个闪烁帧反复设置TextStim.text。
        self.output_box.autoDraw = True
        self.output_text.autoDraw = True
        self._output_max_chars = max(12, int(width*.78 / output_font_height))
        self._sync_output_text(force=True)
        self.header = visual.TextStim(self.win, text="Preparing display",
            pos=(0, output_y - cfg.output_box_height_px/2 - 22),
            height=min(24, max(12, height/40)), wrapWidth=width-40, color="white", autoLog=False)
        self.footer = visual.TextStim(self.win, text="ESC: stop | Cued evaluation",
            pos=(0, -height/2+28), height=min(18, max(10, height/60)), wrapWidth=width-40,
            color="white", autoLog=False)
        self.message = visual.TextStim(self.win, text="", pos=(0, -15),
            height=min(26, height/32), wrapWidth=width*.88, color="white", autoLog=False)
        # 预热字体/绘制对象，避免第一次刺激时才上传纹理。
        for _ in range(30):
            self._check_abort()
            self._draw_keys()
            self.win.flip()
        refresh = self.win.getActualFrameRate(nIdentical=30, nMaxFrames=180, nWarmUpFrames=30, threshold=.5)
        if refresh is None or not np.isfinite(refresh) or refresh <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
            if cfg.quick_entry_mode:
                refresh = 60.0
                print("[快速模式警告] 未测得稳定刷新率，暂按60Hz创建界面；最终实验前必须恢复实测。", flush=True)
            else:
                raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
        self.luminance = make_luminance_table(self.refresh_hz, cfg.stimulus_s)
        # 论文亮度0..1 -> PsychoPy rgb -1..1；全部40目标同步逐帧更新。
        self.rgb_frames = np.repeat((2 * self.luminance - 1)[:, :, None], 3, axis=2)
        self.header.text = "Static render timing check - no flicker"
        flips = []
        for _ in range(121):
            self._check_abort()
            self._draw_keys()
            flips.append(self.win.flip())
        check = frame_diagnostics(flips, self.refresh_hz, cfg.frame_long_factor, cfg.frame_short_factor)
        if not check['valid']:
            if cfg.quick_entry_mode:
                print("[快速模式警告] 静态绘制检测到异常帧间隔；当前联调继续进入键盘。", flush=True)
            else:
                raise RuntimeError("完整键盘静态绘制已出现异常帧间隔，请先解决显示/负载问题再启动刺激")
        framebuffer = getattr(self.win, "frameBufferSize", self.win.size)
        content_scale = self.win.getContentScaleFactor() if hasattr(self.win, 'getContentScaleFactor') else None
        self.display_info = {"window_size_reported": tuple(map(int, self.win.size)),
            "framebuffer_size_reported": tuple(map(int, framebuffer)), "content_scale_factor": content_scale,
            "refresh_hz_measured": self.refresh_hz, "key_size_pix_units": key, "gap_pix_units": gap,
            "layout_scale": scale, "layout_rows": len(KEY_ROWS), "layout_cols": max(map(len, KEY_ROWS)),
            "target_viewing_distance_cm": cfg.viewing_distance_cm,
            "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
            "stimulus_frames": len(self.rgb_frames), "scheduled_stimulus_s": len(self.rgb_frames)/self.refresh_hz,
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}
        print(f"\n屏幕报告: {self.display_info['window_size_reported']}; framebuffer={self.display_info['framebuffer_size_reported']}; "
              f"scaleFactor={content_scale}; 实测刷新率={self.refresh_hz:.5f}Hz")
        print(f"按键{key:.2f} pix、间隙{gap:.2f} pix；目标观看距离{cfg.viewing_distance_cm:g}cm（人工摆放）。")
        if scale < 1:
            print(f"注意：布局缩放为{scale:.4f}，不再是140/50像素基线！")
        print("High-DPI/物理尺寸和显示gamma未校准，pix相同不保证视角/物理亮度相同。")
        print(f"刺激{len(self.rgb_frames)}帧≈{len(self.rgb_frames)/self.refresh_hz:.6f}s；"
              f"分析窗onset+{cfg.response_delay_s:.3f}到onset+{cfg.response_delay_s+cfg.window_s:.3f}s。")

    def _cancel_decode(self) -> None:
        if self._active_decode_cancel is not None:
            self._active_decode_cancel.set()
        if self._decode_future is not None:
            self._decode_future.cancel()

    def _check_abort(self) -> None:
        key_list = ["escape", "space"] if self._pause_enabled else ["escape"]
        keys = self.event.getKeys(keyList=key_list)
        if "escape" in keys:
            self.cancel.set()
            self._cancel_decode()
            raise AbortSession("用户按ESC停止")
        if self.cancel.is_set():
            self._cancel_decode()
            raise AbortSession("已取消实验")
        self.collector.check_health()
        if self._pause_enabled and "space" in keys:
            self._cancel_decode()
            raise PauseSelection("用户按空格暂停；尚未提交的本轮不会写入字符")

    def _sync_output_text(self, force: bool = False) -> None:
        text = self.ledger.typed_text
        if force or text != self._output_cache:
            self.output_text.text = output_display_text(text, self._output_max_chars)
            self._output_cache = text

    def _draw_status(self) -> None:
        # 顶部输出框由autoDraw绘制；这里不画刺激方块，闪烁确实已经停止。
        self.header.draw()
        self.footer.draw()

    def _draw_keys(self, target_id: Optional[int] = None, flicker_rgb: Optional[np.ndarray] = None,
                   show_outline: bool = False) -> None:
        self.squares.colors = self._static_colors if flicker_rgb is None else flicker_rgb
        self.key_borders.draw()
        self.squares.draw()
        for label in self.labels:
            label.draw()
        if target_id is not None:
            pos = self.positions[target_id - 1]
            self.triangle.pos = (pos[0], pos[1] - self.key_size / 2 - self.gap_size * .28)
            self.triangle.draw()
            if show_outline:
                self.outline.pos = pos
                self.outline.draw()
        self.header.draw()
        self.footer.draw()

    def show_message(self, text: str, min_duration_s: float = 0.0, wait_space: bool = False) -> None:
        self.message.text = text
        self._sync_output_text()
        # 自由输入时SPACE是暂停键，不在反馈开始时吞掉暂停请求。
        if wait_space:
            self.event.getKeys(keyList=["space"])
        started = self.clock()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            self._draw_status()
            self.message.draw()
            self.win.flip()
            if self.clock() - started >= min_duration_s:
                if not wait_space or "space" in self.event.getKeys(keyList=["space"]):
                    return

    def _phase(self, record: TrialRecord, name: str) -> None:
        record.phase_times.append((name, self.clock()))

    def _blank_until(self, when: float) -> None:
        while self.clock() < when:
            self._check_abort()
            self._draw_status()
            self.win.flip()

    def _run_trial(self, record: TrialRecord) -> None:
        cfg = self.cfg
        free = record.mode == "free"
        if free and record.true_class is not None:
            raise ValueError("自由输入不应有预设目标")
        target = None if free else TARGETS[record.true_class - 1]
        target_id = None if target is None else target.class_id
        if self._decode_future is not None and not self._decode_future.done():
            raise RuntimeError("前一次分类尚未退出；请暂停后重试")
        self._decode_future = None
        self._active_decode_cancel = threading.Event()
        self._phase(record, "PREPARE")
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Choose your next character"
            self.footer.text = "Keyboard SPACE: pause | ESC: exit | Blank key: space | <-: delete"
        else:
            self.header.text = f"Block {record.block_id}/{cfg.blocks} | Trial {record.trial_id}/{cfg.blocks*40} | Look at {target.symbol}"
            self.footer.text = "ESC: stop | Follow the red triangle"
        self._sync_output_text()
        details = {"mode": record.mode, "window_s": cfg.window_s}
        if target is not None:
            details["frequency_hz"] = target.frequency_hz
        cue_event = EventStamp("ready_onset" if free else "cue_onset", record.trial_id, target_id)
        onset_event = EventStamp("stimulus_onset", record.trial_id, target_id, details)
        offset_event = EventStamp("stimulus_offset", record.trial_id, target_id)
        self._phase(record, "READY" if free else "CUE")
        prepare_s = cfg.free_prepare_s if free else cfg.cue_s
        for frame in range(max(1, round(prepare_s * self.refresh_hz))):
            self._check_abort()
            self._draw_keys(target_id, show_outline=not free)
            if frame == 0:
                self.win.callOnFlip(self.markers.mark, cue_event)
            self.win.flip()
        record.cue_onset = cue_event.timestamp
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Keep looking at your chosen character"
        self._phase(record, "STIMULATE")
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            for frame, rgb in enumerate(self.rgb_frames):
                self._check_abort()
                # target_id=None时不显示红三角/红框；所有目标按同一灰度正弦公式闪烁。
                self._draw_keys(target_id, rgb, show_outline=False)
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, onset_event)
                flip_time = self.win.flip()
                record.frame_flip_times.append(float(flip_time))
                if frame == 0:
                    record.stimulus_onset = onset_event.timestamp
            # 这一帧不再画任何闪烁目标；输出框仍保持可见。
            self.win.callOnFlip(self.markers.mark, offset_event)
            offset_flip = self.win.flip()
            record.frame_flip_times.append(float(offset_flip))
        finally:
            self.win.recordFrameIntervals = False
        record.stimulus_offset = offset_event.timestamp
        if onset_event.timestamp is None or offset_event.timestamp is None:
            raise InvalidTrial("刺激事件标记缺失")
        record.actual_stimulus_s = offset_event.timestamp - onset_event.timestamp
        record.requested_window_start = onset_event.timestamp + cfg.response_delay_s
        record.requested_window_end = record.requested_window_start + cfg.window_s
        record.frame_diagnostics = frame_diagnostics(record.frame_flip_times, self.refresh_hz,
                                                     cfg.frame_long_factor, cfg.frame_short_factor)
        record.frame_diagnostics['psychopy_frame_intervals_s'] = np.asarray(self.win.frameIntervals).copy()
        self._phase(record, "BLANK")
        if not record.frame_diagnostics['valid']:
            if cfg.quick_entry_mode:
                print(f"[快速模式警告] trial {record.trial_id} 刺激帧时序异常："
                      f"长间隔{record.frame_diagnostics['long_intervals']}，"
                      f"短间隔{record.frame_diagnostics['short_intervals']}；当前联调继续分类。", flush=True)
            else:
                self._blank_until(offset_event.timestamp + cfg.blank_min_s)
                raise InvalidTrial(f"刺激时序异常：长间隔{record.frame_diagnostics['long_intervals']}，"
                                   f"短间隔{record.frame_diagnostics['short_intervals']}")
        if self.markers.error is not None:
            raise InvalidTrial(f"LSL Marker发送失败：{self.markers.error}")
        self._phase(record, "WAIT_DATA_AND_CLASSIFY")
        self.header.text = (f"FREE | Selection {record.trial_id} | Processing EEG..." if free
                            else f"Trial {record.trial_id} | Processing EEG...")
        self._decode_future = self.executor.submit(decode_trial, record.trial_id, onset_event.timestamp,
                                                   self.collector, cfg, self._active_decode_cancel)
        while not self._decode_future.done() or self.clock() < offset_event.timestamp + cfg.blank_min_s:
            self._check_abort()
            self._draw_status()
            self.win.flip()
        # 捕获结果提交前的暂停/退出。暂停后绝不将后台迟到的结果写入文本。
        self._check_abort()
        result = self._decode_future.result()
        if result['trial_id'] != record.trial_id:
            raise InvalidTrial("收到属于其他试次的分类结果，拒绝提交")
        if self.markers.error is not None:
            raise InvalidTrial(f"LSL Marker发送失败：{self.markers.error}")
        if not 1 <= result['prediction'] <= len(TARGETS) or not 1 <= result['cca_prediction'] <= len(TARGETS):
            raise InvalidTrial("预测类别超出40目标范围")
        record.result = result
        self.ledger.apply_prediction(result['prediction'])
        record.text_applied = True
        record.status = "valid"
        self._sync_output_text()
        self._phase(record, "FEEDBACK")
        pred = TARGETS[result['prediction'] - 1]
        cca_pred = TARGETS[result['cca_prediction'] - 1]
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Recognized: {pred.symbol}"
            feedback = (f"Recognized: {pred.symbol} ({pred.frequency_hz:.1f} Hz)\n\n"
                        "Choose your next character in the next round.\n"
                        "Keyboard SPACE: pause   ESC: exit")
        else:
            correct = result['prediction'] == target.class_id
            feedback = (f"Target: {target.symbol} ({target.frequency_hz:.1f} Hz)\n"
                        f"FBCCA: {pred.symbol} ({pred.frequency_hz:.1f} Hz)  {'CORRECT' if correct else 'WRONG'}\n"
                        f"CCA: {cca_pred.symbol} ({cca_pred.frequency_hz:.1f} Hz)\n\nESC: stop")
        self.show_message(feedback, cfg.feedback_s)

    def _select_mode(self) -> str:
        self._pause_enabled = False
        selected = self.cfg.session_mode
        self.event.getKeys(keyList=["1", "2", "num_1", "num_2", "space", "return", "num_enter"])
        previous = None
        self.header.text = "40-target SSVEP keyboard | Select a mode"
        self.footer.text = "QUICK ENTRY | ESC: exit | No flicker until you press SPACE or ENTER"
        while True:
            self._check_abort()
            keys = self.event.getKeys(keyList=["1", "2", "num_1", "num_2", "space", "return", "num_enter"])
            if "1" in keys or "num_1" in keys:
                selected = "free"
            if "2" in keys or "num_2" in keys:
                selected = "cued"
            if selected != previous:
                label = "FREE SPELLING" if selected == "free" else "CUED TEST"
                self.message.text = (
                    "1   FREE SPELLING - choose your own characters\n"
                    "2   CUED TEST - follow 40 prompted targets\n\n"
                    f"Selected: {label}\n\n"
                    "SPACE or ENTER: start   ESC: exit\n\n"
                    "Free mode: keyboard SPACE pauses/resumes.\n"
                    "Blank bottom-left key: insert space.  <-: delete.\n"
                    "No automatic calibration or idle detection.")
                previous = selected
            self._draw_status()
            self.message.draw()
            self.win.flip()
            if any(k in keys for k in ("space", "return", "num_enter")):
                self.cfg.session_mode = selected
                print("\n选择模式：" + ("自由输入" if selected == "free" else "40目标提示测试"), flush=True)
                return selected

    def run(self) -> str:
        mode = self._select_mode()
        return self._run_free() if mode == "free" else self._run_cued()

    def _wait_free_pause(self, message: str) -> None:
        # 暂停界面不产生刺激；EEG仍连续采集。后台运算结束前不启动新一轮。
        self._pause_enabled = False
        self._cancel_decode()
        self.event.getKeys(keyList=["space", "return", "num_enter"])
        started = self.clock()
        self.header.text = "FREE | PAUSED | No character is being selected"
        self.footer.text = "SPACE or ENTER: resume | ESC: exit | Output is kept"
        self.message.text = message + "\n\nSPACE or ENTER: resume\nESC: exit"
        self._sync_output_text()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            keys = self.event.getKeys(keyList=["space", "return", "num_enter"])
            ready = self._decode_future is None or self._decode_future.done()
            self._draw_status()
            self.message.draw()
            self.win.flip()
            if ready and self.clock() - started >= .35 and keys:
                # 被取消的后台结果直接丢弃，不会被下一轮领取。
                self._decode_future = None
                self._active_decode_cancel = None
                self._pause_enabled = True
                return

    @staticmethod
    def _release_free_eeg(record: TrialRecord) -> None:
        # 自由输入可持续很久：保留轻量预测/时序摘要，不累积每轮EEG与逐帧数组。
        if record.result is not None:
            record.result.pop("epoch", None)
        record.frame_flip_times.clear()
        record.frame_diagnostics.pop("intervals_s", None)
        record.frame_diagnostics.pop("psychopy_frame_intervals_s", None)

    def _run_free(self) -> str:
        self._pause_enabled = True
        selection_id = 0
        consecutive_invalid = 0
        try:
            while True:
                selection_id += 1
                record = TrialRecord(selection_id, 1, None, mode="free", start=self.clock())
                pause_requested = False
                exit_requested = False
                fatal = False
                try:
                    self._run_trial(record)
                except PauseSelection as exc:
                    pause_requested = True
                    record.status = "valid" if record.text_applied else "aborted"
                    record.reason = str(exc)
                except (AbortSession, KeyboardInterrupt) as exc:
                    exit_requested = True
                    record.status = "valid" if record.text_applied else "aborted"
                    record.reason = str(exc) or "KeyboardInterrupt"
                    self.cancel.set()
                except Exception as exc:
                    record.status = "valid" if record.text_applied else "invalid"
                    record.reason = f"{type(exc).__name__}: {exc}"
                    fatal = self.collector.error is not None or self.markers.error is not None
                finally:
                    self._cancel_decode()
                    try:
                        self.win.recordFrameIntervals = False
                        self._draw_status()
                        self.win.flip()  # 清除所有闪烁目标；autoDraw保留输出框。
                    except Exception as exc:
                        fatal = True
                        record.reason += f"; display-close error: {exc}"
                        if not record.text_applied:
                            record.status = "invalid"
                    record.end = self.clock()
                    self._phase(record, "COMPLETE")
                    try:
                        self.markers.mark(EventStamp("selection_end", record.trial_id, None,
                            {"mode": "free", "status": record.status, "reason": record.reason,
                             "prediction": record.result['prediction'] if record.result else None,
                             "text_applied": record.text_applied}))
                    except Exception as exc:
                        record.reason += f"; marker error: {exc}"
                        fatal = True
                    self.ledger.commit(record)
                    self._release_free_eeg(record)
                if record.text_applied:
                    consecutive_invalid = 0
                    pred = TARGETS[record.result['prediction'] - 1]
                    print(f"FREE {selection_id:04d}: FBCCA={pred.symbol} ({pred.frequency_hz:.1f}Hz)"
                          f" | text={self.ledger.typed_text!r}", flush=True)
                else:
                    print(f"FREE {selection_id:04d}: no character added | {record.reason}", flush=True)
                    if not pause_requested and not exit_requested:
                        consecutive_invalid += 1
                if exit_requested or fatal:
                    return record.reason or "Free spelling stopped"
                if pause_requested:
                    self._wait_free_pause("Paused. Output already shown is kept.\n"
                                          "An unfinished selection is discarded.")
                    consecutive_invalid = 0
                elif consecutive_invalid >= self.cfg.max_consecutive_invalid:
                    self._wait_free_pause("Several selections could not be decoded.\n"
                                          "No fallback characters were inserted.\n"
                                          "Check the terminal and EEG stream before resuming.")
                    consecutive_invalid = 0
        finally:
            self._pause_enabled = False
            self._cancel_decode()

    def _run_cued(self) -> str:
        cfg = self.cfg
        self._pause_enabled = False
        schedule = make_schedule(cfg.blocks, cfg.random_seed)
        consecutive_invalid = 0
        previous_block = 1
        stop_reason = "Completed planned trials"
        for trial_id, (block_id, target_id) in enumerate(schedule, 1):
            if block_id != previous_block:
                self.show_message(f"Block {previous_block} complete. Rest your eyes.\n"
                    f"Minimum break: {cfg.block_rest_s:g} seconds.\n\nThen SPACE to continue; ESC to finish.",
                    cfg.block_rest_s, wait_space=True)
                previous_block = block_id
            record = TrialRecord(trial_id, block_id, target_id, start=self.clock())
            fatal = False
            try:
                self._run_trial(record)
            except (AbortSession, KeyboardInterrupt) as exc:
                record.status = "valid" if record.text_applied else "aborted"
                record.reason = str(exc) or "KeyboardInterrupt"
                # 已经显示/提交的字符保留；尚未完成的选择不写入。
                self.cancel.set()
                fatal = True
                stop_reason = record.reason
            except Exception as exc:
                record.status, record.reason = "invalid", f"{type(exc).__name__}: {exc}"
                fatal = self.collector.error is not None or self.markers.error is not None
                stop_reason = record.reason
            finally:
                try:
                    self.win.recordFrameIntervals = False
                    self.win.flip()  # 所有异常路径立即去掉闪烁画面
                except Exception as clear_error:
                    # 例如用户关闭了窗口：即使清屏失败，也必须留下已开始试次的记录。
                    if record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; display-close error: {clear_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                record.end = self.clock()
                self._phase(record, "COMPLETE")
                try:
                    self.markers.mark(EventStamp("trial_end", record.trial_id, record.true_class,
                        {"status": record.status, "reason": record.reason,
                         "prediction": record.result['prediction'] if record.result else None}))
                except Exception as marker_error:
                    if record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; marker error: {marker_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                self.ledger.commit(record)
            if record.status == 'valid':
                consecutive_invalid = 0
                r = record.result
                print(f"Trial {trial_id:03d}: true={target_id:02d}, FBCCA={r['prediction']:02d}, CCA={r['cca_prediction']:02d}, "
                    f"window=[{record.requested_window_start:.6f}, {record.requested_window_end:.6f}), "
                    f"maxFrame={record.frame_diagnostics['max_interval_ms']:.2f}ms, "
                    f"compute={r['computation_s']:.3f}s, total={record.end-record.start:.3f}s", flush=True)
            else:
                consecutive_invalid += 1
                print(f"Trial {trial_id:03d}: {record.status.upper()} - {record.reason}", flush=True)
            if fatal or consecutive_invalid >= cfg.max_consecutive_invalid:
                if not fatal:
                    stop_reason = f"连续{consecutive_invalid}次无效，停止以便排查：{record.reason}"
                break
        else:
            stop_reason = "Completed planned trials"
        return stop_reason

    def close(self) -> None:
        self.cancel.set()
        self._cancel_decode()
        if self.win is not None:
            try:
                self.win.recordFrameIntervals = False
                self.win.flip()
            except Exception:
                pass
            self.win.close()
            self.win = None
        self.executor.shutdown(wait=True, cancel_futures=True)


def print_target_mapping() -> None:
    print("\n=== 唯一目标映射（类别1-based，显示行列也转为1-based）===")
    for target in TARGETS:
        print(f"ID{target.class_id:02d}: {target.symbol:>5}  行{target.row+1}列{target.col+1}  {target.frequency_hz:4.1f}Hz")


def main() -> dict:
    """唯一运行入口；不要求CLI参数；返回全部内存记录。"""
    global LAST_SESSION
    cfg = CONFIG
    cfg.validate()
    # 直接运行时已在文件顶部自动补齐缺失依赖；这里再做一次实际导入验证。
    print("[1/4] 检查界面和采集依赖…", flush=True)
    try:
        import scipy
        import pylsl
        # 只import psychopy无法发现i18next等缺失的界面依赖。
        from psychopy import visual, event, gui
        import threadpoolctl
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            f"界面/采集依赖导入失败，当前Python：{sys.executable}\n"
            "这台电脑请先用同目录的 start_keyboard_dual_mode.cmd 启动（优先使用Python 3.10）。\n"
            "原始错误：" + str(exc)) from exc
    print(f"安全提示：此程序呈现{BENCHMARK_FREQUENCIES_HZ.min():g}–{BENCHMARK_FREQUENCIES_HZ.max():g}Hz视觉闪烁。"
          "对闪光敏感、有光敏性癫痫史者不要自行测试；"
          "出现眼部不适、头痛、眩晕等应立即按ESC停止。先阅读风险并核对接线，程序不会直接开始闪烁。")
    print("模式：启动界面1=自由输入（默认），2=提示测试；没有迷宫或操作系统按键注入。")
    print("自由输入不需要先做40轮测试；本版未添加个人校准或空闲检测，休息时按SPACE暂停。")
    if cfg.quick_entry_mode:
        print("*** 当前为 QUICK ENTRY 快速联调模式：优先进入程序，严格时间戳/显示核验暂时放宽。 ***", flush=True)
    print(f"参数：M3七子带、{cfg.n_harmonics}谐波、a={cfg.weight_a}, b={cfg.weight_b}, target_fs={cfg.target_fs}Hz")
    print_target_mapping()
    collector = ContinuousLSL(cfg)
    ledger = TrialLedger()
    markers: Optional[EventMarkers] = None
    app: Optional[PsychoPyKeyboard] = None
    review: dict = {}
    reason = "Not started"
    try:
        print("\n[2/4] 连接EEG LSL并检查采样时间轴（请保持采集软件的LSL输出开启）…", flush=True)
        collector.start()
        print("[3/4] EEG已接收，快速模式直接继续…" if cfg.quick_entry_mode
              else "[3/4] EEG已接收，打开启动核验对话框…", flush=True)
        review = preflight_review(cfg, collector)
        markers = EventMarkers(cfg, collector.clock)
        # 预热SciPy/BLAS与分类代码，但不使用真实目标，不计入准确率。
        t = np.arange(sample_count(cfg.window_s, cfg.target_fs)) / cfg.target_fs
        warm = np.tile(np.sin(2*np.pi*10*t), (len(cfg.channel_indices), 1))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            classify_eeg_window(warm, cfg.target_fs, cfg.window_s, target_fs=cfg.target_fs,
                notch_hz=cfg.notch_hz, n_harmonics=cfg.n_harmonics,
                a=cfg.weight_a, b=cfg.weight_b, regularization=cfg.cca_regularization)
        print("[4/4] 创建键盘窗口；在菜单按1自由输入、2提示测试，再按SPACE开始…", flush=True)
        app = PsychoPyKeyboard(cfg, collector, markers, ledger)
        reason = app.run()
    except (AbortSession, KeyboardInterrupt) as exc:
        reason = str(exc) or "用户中止"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"\n实验停止：{reason}", flush=True)
    finally:
        if app is not None:
            app.close()
        collector.close()
        if markers is not None:
            markers.close()
        report = (summarize_free_trials(ledger.records, ledger) if cfg.session_mode == "free"
                  else summarize_trials(ledger.records, cfg.blocks * len(TARGETS)))
        LAST_SESSION = {"mode": cfg.session_mode, "config": cfg, "targets": TARGETS, "records": ledger.records,
            "events": markers.events if markers else [], "acquisition_metadata": collector.metadata,
            "acquisition_review": review, "clock_updates": collector.clock_updates,
            "display_info": app.display_info if app else {}, "summary": report,
            "typed_text": ledger.typed_text, "stop_reason": reason,
            "marker_error": str(markers.error) if markers and markers.error else None,
            "validation_scope": "software events only; physical screen/device delays unmeasured"}
        print(f"\n结束原因：{reason}")
        if markers is not None and markers.error is not None:
            print(f"外部Marker发送存在错误：{markers.error}；请勿把本次外部Marker流视为完整。")
        if cfg.session_mode == "free":
            print_free_summary(report)
        else:
            print_summary(report, ledger)
    return LAST_SESSION


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"启动失败：{exc}") from exc
