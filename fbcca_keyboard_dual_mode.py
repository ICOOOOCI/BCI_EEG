"""40-target SSVEP keyboard: FREE SPELLING + CUED TEST.

启动界面：1=自由输入，2=提示测试（默认）；SPACE/ENTER开始，ESC退出。
启动前静态预览：调整尺寸/间距/距离，逐个检查四角；--preview-layout 可单独预览。
--check-lsl-timing 可在无闪烁时预热并持续检查EEG流；--seconds指定预热后的观察时长。
提示测试：SPACE启动整组；每完成10个目标静态休息30秒，最后3秒逐秒提示音后自动继续。
自由输入：自行注视字符，每轮真实LSL EEG经FBCCA识别后写入顶部OUTPUT框。
电脑键盘SPACE暂停/继续；屏幕内SPACE目标插入空格，BACK目标删除上一字符。
采用4×10 QWERTY布局；按新行列重排40类及频率（8–15.8 Hz）。
默认使用抗工频的48Hz上限、三谐波配置；原M3配置仍可在Config中选择。
可通过 --calibration 和 --participant 加载个人模板；没有自动空闲检测。
不包含迷宫，不注入操作系统按键；首个已开始试次结束后暂存逐试次数据，包括无EEG的失败试次。
模式2将同名NPZ、质控JSON、个人模板PKL和事件CSV封装为一个会话ZIP；导出失败时保留逐试次数据。
模式1仍只导出会话NPZ。
算法函数仍可import调用；import不会安装依赖或启动闪烁。
实时EEG由MNE-LSL StreamLSL采集与缓冲，使用LSL内建时钟同步、去抖和单调化。

UI API references checked 2026-09-17:
https://psychopy.org/api/event.html
https://psychopy.org/api/visual/textstim.html
https://psychopy.org/api/visual/window.html
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata as metadata
import os
# 避免小型 CCA 矩阵运算动用大量 BLAS 线程；已有环境设置不覆盖。
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import hashlib
import io
import json
import math
from pathlib import Path
import platform
import pickle
import queue
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ======================== 运行支持（合并自原辅助模块） ========================
ROOT = Path(__file__).resolve().parent
PYTHON_VERSION = (3, 10, 11)
FAULTS = {
    "ENVIRONMENT": (10, "运行环境不匹配", "请安装 64 位 CPython 3.10.11 及程序依赖。"),
    "LSL_LIBRARY": (20, "无法加载 MNE-LSL", "在项目虚拟环境安装 mne-lsl==1.12.0 并检查 liblsl。"),
    "LSL_NOT_FOUND": (21, "找不到 EEG LSL 流", "开启发布端的 LSL 输出并检查网络。"),
    "LSL_CONNECTION": (22, "LSL 连接或事件流异常", "检查重复流、通道配置、发布端和网络。"),
    "ACQUISITION": (30, "采样中断或 EEG 数据异常", "检查设备连接、采样率和时间戳。"),
    "DISPLAY": (40, "显示异常", "检查显示器、刷新率、显卡驱动和系统负载。"),
    "AUDIO": (41, "提示音异常", "检查扬声器、系统音量和 PsychoPy 音频设备。"),
    "SAVE": (50, "保存失败", "检查目标目录权限和磁盘空间。"),
    "INTERNAL": (70, "程序异常", "查看终端错误信息和系统临时目录中的诊断报告。"),
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
            "action": action,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def print_fault(fault: dict) -> None:
    print(f"[{fault['code']}] {fault['title']}：{fault['detail']}\n处理建议：{fault['action']}",
          file=sys.stderr, flush=True)


def environment_snapshot() -> dict:
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "platform": platform.platform(), "machine": platform.machine(),
            "bits": struct.calcsize("P") * 8,
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                         if d.metadata.get("Name")},
            "mne_lsl_lib_override": os.environ.get("MNE_LSL_LIB") or os.environ.get("PYLSL_LIB")}


def check_environment(*, static_ui: bool = False) -> dict:
    issues = []
    if (sys.version_info[:3] != PYTHON_VERSION or struct.calcsize("P") != 8
            or platform.python_implementation() != "CPython"):
        issues.append(f"需要 64 位 CPython {'.'.join(map(str, PYTHON_VERSION))}；当前 {platform.python_version()}")
    for name in ("numpy", "scipy", "threadpoolctl", "psychopy.visual", "psychopy.event", "psychopy.gui"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            issues.append(f"{name} 无法导入：{exc}")
    result = {"python_version": platform.python_version(), "lsl_checked": not static_ui}
    if not static_ui:
        try:
            mne_lsl = importlib.import_module("mne_lsl")
            lsl = importlib.import_module("mne_lsl.lsl")
            result["mne_lsl_version"] = mne_lsl.__version__
            result["liblsl_version"] = lsl.library_version()
        except Exception as exc:
            raise RuntimeFault("LSL_LIBRARY", str(exc)) from exc
    if issues:
        raise RuntimeFault("ENVIRONMENT", "\n".join(issues) + f"\nPython：{sys.executable}")
    return result


def probe_save_directory(directory: str | Path) -> str:
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


def write_diagnostic(report: dict) -> str | None:
    """诊断记录写入系统临时目录，避免在主程序目录留下辅助文件。"""
    name = time.strftime("run_%Y%m%dT%H%M%SZ_", time.gmtime()) + uuid.uuid4().hex[:8] + ".json"
    try:
        report["environment"] = environment_snapshot()
    except Exception as exc:
        report["environment_snapshot_error"] = str(exc)
    folder = Path(tempfile.gettempdir()) / "bci-eeg-diagnostics"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"诊断报告：{path}", flush=True)
        return str(path)
    except OSError as exc:
        print(f"[SAVE] 诊断报告写入失败：{folder}：{exc}", file=sys.stderr, flush=True)
        return None


# ======================== 配置：通常只修改这一处 ========================
@dataclass
class Config:
    # 窗长预设与输入模式相互独立；均不冒充论文严格1.8秒在线时序。
    # 新增两种模式：free=自由输入；cued=原有40目标提示测试。
    # 启动界面按1/2选择，SPACE或ENTER开始；默认进行40目标提示测试。
    session_mode: str = "cued"
    free_prepare_s: float = 1.0  # 每轮闪烁前留出时间，自行选择下一个字符。
    # 两种模式默认分析窗2秒；闪烁额外覆盖0.14秒起始延迟及帧余量。
    protocol: str = "standard_2s"
    # 快速模式只跳过启动核验弹窗；保留模式菜单及时间戳、质量、显示时序门控。
    quick_entry_mode: bool = False
    blocks: int = 1
    random_seed: int = 20260928
    cue_s: float = 2.0  # 模式2每个目标的固定定位提示时长，无需逐试次按键。
    cued_settle_s: float = 0.5  # 提示后保持静态高亮，留出稳定注视时间。
    response_delay_s: float = 0.14
    blank_min_s: float = 0.5
    feedback_s: float = 0.5
    cued_rest_every: int = 10  # 模式2每完成多少个有效目标后休息；不在整组结束后重复休息。
    cued_rest_s: float = 30.0  # 模式2组内休息时长；到时自动继续。
    block_rest_s: float = 30.0
    max_consecutive_invalid: int = 3

    eeg_stream_name: Optional[str] = None
    eeg_stream_type: str = "EEG"
    # EEG与刺激事件统一使用本机LSL时钟。LSL执行同步、去抖与单调化；
    # 不再按接收样本计数另造时间轴，也不重复应用time_correction。
    timestamp_mode: str = "lsl"
    # 以下旧参数只为读取历史会话配置保留；新接收路径不使用它们。
    continue_on_timestamp_anomaly: bool = True
    timestamp_max_jitter_s: float = 0.05
    timestamp_jitter_grace_s: float = 0.5
    timestamp_max_discontinuity_s: float = 0.25
    timestamp_slew_rate_s_per_s: float = 0.01
    # 假定 LSL 前8列按设备 CH1..CH8 输出；启动时必须核对这一假定。
    # 用户2026-09-13接线：P1->8,P2->7,PO3->6,POz->5,PO4->4,O1->2,Oz->3,O2->1。
    channel_indices: tuple[int, ...] = (7, 6, 5, 4, 3, 1, 2, 0)
    channel_positions: tuple[str, ...] = ("P1", "P2", "PO3", "POz", "PO4", "O1", "Oz", "O2")
    buffer_s: float = 60.0
    connect_timeout_s: float = 10.0
    startup_timeout_s: float = 45.0
    warmup_s: float = 30.0
    data_wait_timeout_s: float = 8.0
    clock_refresh_s: float = 1.0
    max_clock_step_s: float = 0.002
    clock_slew_rate_s_per_s: float = 0.001
    max_receive_age_s: float = 2.0
    max_gap_factor: float = 1.5  # 发现一个明确缺样(2个采样间隔)也拒绝，不跨缺口补齐。
    hard_gap_factor: float = 2.0
    rate_tolerance: float = 0.02
    hard_rate_tolerance: float = 0.05
    max_warning_gaps: int = 1
    max_warning_frame_anomalies: int = 1
    min_valid_channels: int = 6
    max_warning_bad_channels: int = 2
    quality_max_abs: Optional[float] = None
    quality_rail_min: Optional[float] = None
    quality_rail_max: Optional[float] = None
    quality_rail_margin: float = 0.0  # 与设备轨值相同的EEG数值单位。
    quality_max_clip_fraction: float = 0.0  # 每通道允许的削顶样本比例，[0, 1)。
    quality_jump_z: float = 12.0
    # 默认直接采用有效试次的分类结果；启用拒识前须用真实EEG标定并验证两个阈值。
    rejection_enabled: bool = False
    rejection_min_score: Optional[float] = None
    rejection_min_margin: Optional[float] = None
    record_root: str = "session_records"
    # AUTO 从流元数据读取；缺失时不会猜测单位。对话框允许人工明确覆盖。
    input_unit: str = "AUTO"
    upstream_filter_description: str = "UNKNOWN"
    upstream_lowpass_hz: Optional[float] = None
    # 已独立测得的固定设备滞后：正值表示发布端时间戳晚于实际采样，需要减去。
    # 0仅表示未应用修正，不表示已测得延迟为零；不得随意填写。
    device_timestamp_lag_s: float = 0.0

    target_fs: float = 250.0
    n_harmonics: int = 3
    # m3保持历史算法；默认line_robust为待独立验证的三谐波/48Hz三子带方案。
    filter_bank_profile: str = "line_robust"
    weight_a: float = 1.25
    weight_b: float = 0.25
    notch_hz: Optional[float] = 50.0  # 保留原文件50 Hz; 按采集地供电/已有滤波核对。
    # 默认关闭；2026-09-28离线探索的待复测方案，不替代采集端工频排查。
    line_regression_hz: Optional[float] = None
    cca_regularization: float = 1e-10
    # 个人模板必须显式绑定受试者；未指定时使用原 FBCCA。
    participant_id: Optional[str] = None
    calibration_file: Optional[str] = None
    # 拒识分数尺度随算法变化；启用个人模板时须绑定重新标定的模型ID。
    rejection_calibration_id: Optional[str] = None

    full_screen: bool = True  # Windows/macOS 默认铺满目标显示器；仅调试时关闭。
    screen_index: int = 0
    window_size: tuple[int, int] = (1920, 1080)  # 仅用于非全屏调试窗口。
    layout_units: str = "pixels"  # pixels=PsychoPy 的 pix 坐标；Retina 上是窗口逻辑像素。
    key_size_px: float = 140.0
    key_gap_px: float = 30.0  # 行列使用相同边缘间距，剩余空间留在网格外侧。
    key_size_deg: float = 2.0
    key_gap_deg: float = 0.5
    allow_layout_scaling: bool = True  # 放不下时整体缩小；预览显示实际尺寸和缩放比例。
    viewing_distance_cm: float = 70.0  # 目标观看距离，不代表自动测量结果。
    screen_diagonal_inches: Optional[float] = None
    frame_long_factor: float = 1.5
    frame_short_factor: float = 0.5
    # 40目标最小频差0.2Hz；0.5%速率误差在最高15.8Hz处约为0.079Hz。
    frame_rate_tolerance: float = 0.005
    marker_stream_name: str = "FBCCA_40_Keyboard_Events"
    output_box_height_px: float = 72.0
    output_box_margin_px: float = 24.0

    @property
    def window_s(self) -> float:
        presets = {"standard_5s": 5.0, "standard_2s": 2.0, "debug_2s": 2.0,
                   "paper_window_1p25s": 1.25, "paper_offline_5s": 5.0}
        if self.protocol not in presets:
            raise ValueError(f"未知 protocol: {self.protocol}; 可选 {tuple(presets)}")
        return presets[self.protocol]

    @property
    def stimulus_s(self) -> float:
        # 分类窗从刺激起点延后 response_delay_s，必须在持续闪烁时结束。
        return self.response_delay_s + self.window_s

    @property
    def filter_bands(self) -> tuple[tuple[float, float], ...]:
        profiles = {"m3": M3_BANDS, "line_robust": LINE_ROBUST_BANDS}
        if self.filter_bank_profile not in profiles:
            raise ValueError("filter_bank_profile 必须为 m3 或 line_robust")
        return profiles[self.filter_bank_profile]

    @property
    def rejection_thresholds_configured(self) -> bool:
        """Whether free mode has an explicitly enabled, two-threshold gate."""
        return bool(
            self.rejection_enabled
            and self.rejection_min_score is not None
            and self.rejection_min_margin is not None
            and math.isfinite(float(self.rejection_min_score))
            and math.isfinite(float(self.rejection_min_margin))
        )

    def validate(self) -> None:
        _ = self.window_s
        if self.calibration_file and not (self.participant_id and self.participant_id.strip()):
            raise ValueError("加载个人模型必须设置 participant_id / --participant")
        if self.session_mode not in ("free", "cued"):
            raise ValueError("session_mode 必须为 free 或 cued")
        if self.timestamp_mode not in ("lsl", "auto", "strict", "openbci"):
            raise ValueError("timestamp_mode 必须为 lsl；旧值仅供历史会话离线读取")
        if not isinstance(self.continue_on_timestamp_anomaly, bool):
            raise ValueError("continue_on_timestamp_anomaly 必须为布尔值")
        if not math.isfinite(self.timestamp_max_jitter_s) or self.timestamp_max_jitter_s <= 0:
            raise ValueError("timestamp_max_jitter_s 必须为有限正数")
        if not math.isfinite(self.timestamp_jitter_grace_s) or self.timestamp_jitter_grace_s <= 0:
            raise ValueError("timestamp_jitter_grace_s 必须为有限正数")
        if (not math.isfinite(self.timestamp_max_discontinuity_s)
                or self.timestamp_max_discontinuity_s <= self.timestamp_max_jitter_s):
            raise ValueError("timestamp_max_discontinuity_s 必须大于持续偏移门限")
        if not 0 < self.timestamp_slew_rate_s_per_s < .1:
            raise ValueError("时间轴微调速率必须在0和0.1之间")
        if not math.isfinite(self.free_prepare_s) or self.free_prepare_s <= 0:
            raise ValueError("free_prepare_s 必须是有限正数")
        for name in ("blocks", "n_harmonics", "min_valid_channels", "max_consecutive_invalid",
                     "cued_rest_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("cue_s", "cued_settle_s", "response_delay_s", "blank_min_s", "feedback_s",
                     "cued_rest_s", "block_rest_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        for name in ("connect_timeout_s", "startup_timeout_s", "warmup_s", "data_wait_timeout_s",
                     "buffer_s", "clock_refresh_s", "max_clock_step_s", "max_receive_age_s",
                     "key_size_px", "viewing_distance_cm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.timestamp_mode == "lsl" and self.startup_timeout_s <= self.warmup_s:
            raise ValueError("LSL启动超时必须长于去抖预热时长")
        if not math.isfinite(self.key_gap_px) or self.key_gap_px < 0:
            raise ValueError("key_gap_px 必须非负")
        if self.layout_units not in ("pixels", "degrees"):
            raise ValueError("layout_units 必须为 pixels 或 degrees")
        if not math.isfinite(self.key_size_deg) or not 0 < self.key_size_deg < 90:
            raise ValueError("key_size_deg 必须在0和90度之间")
        if not math.isfinite(self.key_gap_deg) or not 0 <= self.key_gap_deg < 90:
            raise ValueError("key_gap_deg 必须在0和90度之间（可为0）")
        if self.screen_diagonal_inches is not None and (
                not math.isfinite(self.screen_diagonal_inches) or self.screen_diagonal_inches <= 0):
            raise ValueError("screen_diagonal_inches 必须为空或有限正数")
        if not 1 < self.max_gap_factor < 2:
            raise ValueError("max_gap_factor 必须在1和2之间，不得跨明确缺样插值")
        if not (1 < self.frame_long_factor < 2 and 0 < self.frame_short_factor < 1):
            raise ValueError("帧间隔阈值不合法")
        if not math.isfinite(self.frame_rate_tolerance) or not 0 < self.frame_rate_tolerance < 1:
            raise ValueError("frame_rate_tolerance 必须是(0, 1)内的有限比例")
        if not 0 < self.rate_tolerance < 0.1:
            raise ValueError("rate_tolerance 必须在0和0.1之间")
        if not math.isfinite(self.device_timestamp_lag_s):
            raise ValueError("device_timestamp_lag_s 必须有限")
        for name in ("clock_slew_rate_s_per_s", "quality_rail_margin", "quality_jump_z"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        if not math.isfinite(self.quality_max_clip_fraction) or not 0 <= self.quality_max_clip_fraction < 1:
            raise ValueError("quality_max_clip_fraction 必须是[0, 1)内的有限样本比例")
        if not self.hard_gap_factor > self.max_gap_factor:
            raise ValueError("hard_gap_factor 必须大于 max_gap_factor")
        for name in ("max_warning_gaps", "max_warning_frame_anomalies", "max_warning_bad_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须为非负整数")
        if not self.hard_rate_tolerance > self.rate_tolerance:
            raise ValueError("hard_rate_tolerance 必须大于 rate_tolerance")
        for name in ("quality_max_abs", "quality_rail_min", "quality_rail_max"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value)):
                raise ValueError(f"{name} 必须有限")
        if self.quality_max_abs is not None and self.quality_max_abs <= 0:
            raise ValueError("quality_max_abs 必须为正")
        if ((self.quality_rail_min is None) != (self.quality_rail_max is None)
                or (self.quality_rail_min is not None and self.quality_rail_min >= self.quality_rail_max)):
            raise ValueError("quality_rail_min/max 必须同时设置且 min < max")
        if (self.quality_rail_min is not None
                and 2 * self.quality_rail_margin >= self.quality_rail_max - self.quality_rail_min):
            raise ValueError("quality_rail_margin 必须小于设备轨值跨度的一半")
        for name in ("rejection_min_score", "rejection_min_margin"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} 必须非负且有限")
        band_high = max(high for _, high in self.filter_bands)
        _validate_fs(self.target_fs, band_high)
        _validate_line_regression_hz(self.line_regression_hz, self.target_fs)
        if self.filter_bank_profile == "line_robust" and self.n_harmonics != 3:
            raise ValueError("line_robust 复测方案必须使用 n_harmonics=3")
        _normalise_channel_indices(self.channel_indices)
        if len(self.channel_indices) != len(self.channel_positions):
            raise ValueError("channel_indices 和 channel_positions 数量必须一致")
        if not 1 <= self.min_valid_channels <= len(self.channel_indices):
            raise ValueError("min_valid_channels 超出所选通道数")
        if self.buffer_s <= self.window_s + self.response_delay_s + self.data_wait_timeout_s + 2:
            raise ValueError("内存缓冲区过短，可能覆盖仍待分类的 EEG")
        _validate_harmonics(self.n_harmonics, BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        if self.upstream_lowpass_hz is not None:
            if not math.isfinite(self.upstream_lowpass_hz) or self.upstream_lowpass_hz < band_high:
                raise ValueError(f"已知发布端低通小于{band_high:g} Hz，不能使用当前频带")


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
# 频率按图片中的4×10交错表分配；布局和字符顺序保持不变。
FREQUENCIES_HZ_BY_ROW = (
    (8.0, 12.2, 8.4, 12.6, 8.8, 13.0, 9.2, 13.4, 9.6, 13.8),
    (10.0, 14.2, 10.4, 14.6, 10.8, 15.0, 11.2, 15.4, 11.6, 15.8),
    (12.0, 8.2, 12.4, 8.6, 12.8, 9.0, 13.2, 9.4, 13.6, 9.8),
    (14.0, 10.2, 14.4, 10.6, 14.8, 11.0, 15.2, 11.4, 15.6, 11.8),
)
TARGETS = tuple(Target(i + 1, symbol, r, c, FREQUENCIES_HZ_BY_ROW[r][c])
                for i, (r, c, symbol) in enumerate(
                    (r, c, symbol) for r, row in enumerate(KEY_ROWS) for c, symbol in enumerate(row)))
BENCHMARK_FREQUENCIES_HZ = np.asarray([t.frequency_hz for t in TARGETS])
M3_BANDS = tuple((float(8 * n - 2), 90.0) for n in range(1, 8))
# 最低目标8Hz的第三谐波是24Hz；最后一带从22Hz开始，所有目标都有通带内参考。
LINE_ROBUST_BANDS = tuple((float(8 * n - 2), 48.0) for n in range(1, 4))
DEFAULT_WINDOW_S = 2.0
TARGET_FS = 250.0
N_HARMONICS = 5
LAST_SESSION: Optional[dict[str, Any]] = None


class InvalidTrial(RuntimeError):
    """数据/时序未通过质量检查；必须计入无效试次，而不是静默跳过。"""
    def __init__(self, message: str, *, epoch: Optional["Epoch"] = None,
                 code: str = "ACQUISITION"):
        super().__init__(message)
        self.epoch = epoch
        self.code = code


class DegenerateSignalError(ValueError):
    """没有足够的变化信号，不能产生有意义的 CCA 结果。"""


class AbortSession(RuntimeError):
    """用户取消或急停。"""


class PauseSelection(RuntimeError):
    """自由输入暂停：立即停止闪烁，取消尚未提交的选择；不关闭EEG采集。"""


# ======================== 原算法核心 + 数值修正 ========================
def _normalise_channel_indices(channel_indices: Sequence[int]) -> np.ndarray:
    indices = np.asarray(channel_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("channel_indices 必须是一维非空数组")
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.isfinite(indices)) or not np.all(indices == np.round(indices)):
            raise ValueError("channel_indices 必须是整数")
    indices = indices.astype(int)
    if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
        raise ValueError("通道索引必须非负且不重复")
    return indices


def _validate_eeg_ct(data: np.ndarray, *, name: str = "EEG",
                     allow_empty_samples: bool = False) -> np.ndarray:
    """Return an EEG matrix with the single internal layout: channels × samples."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or (
            not allow_empty_samples and x.shape[1] < 1):
        raise ValueError(f"{name}必须是二维C×T数组（通道数×采样点数）")
    return x


def sample_count(duration_s: float, fs: float) -> int:
    """[start, end)内的均匀采样点数；1.25s×250Hz需要313点而非静默截掉半点。

    最后一采样点必须严格小于请求终点；浮点容差只处理整数乘积舍入误差。
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("时长和采样率必须为有限正数")
    return int(math.ceil(duration_s * fs - 1e-9))


def _validate_fs(fs: float, upper_hz: Optional[float] = 90.0) -> None:
    # 与滤波无关的时间轴工具只验证采样率；采集/解码入口验证实际频带。
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数")
    if upper_hz is not None and fs <= 2 * upper_hz:
        raise ValueError(f"原始/处理采样率必须大于{2 * upper_hz:g} Hz，以支持{upper_hz:g} Hz子带上限")


def _validate_filter_bands(filter_bands: Optional[Sequence[Sequence[float]]],
                           fs: float, target_fs: float) -> tuple[tuple[float, float], ...]:
    bands = M3_BANDS if filter_bands is None else tuple(tuple(band) for band in filter_bands)
    if not bands or any(len(band) != 2 for band in bands):
        raise ValueError("滤波组必须包含(low, high)频带")
    for low, high in bands:
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("滤波频带必须满足 0 < low < high")
    high = max(high for _, high in bands)
    _validate_fs(fs, high)
    _validate_fs(target_fs, high)
    return bands


def _validate_harmonics(n: int, frequencies: np.ndarray, fs: float) -> None:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("谐波数必须为正整数")
    if not np.isfinite(fs) or fs <= 0 or np.max(frequencies) * n >= fs / 2:
        raise ValueError("最高参考谐波必须低于Nyquist频率")


def _validate_line_regression_hz(value: Optional[float], fs: float) -> None:
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value < fs / 2):
        raise ValueError("line_regression_hz须为None或低于Nyquist频率的有限正数")


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
    x = _validate_eeg_ct(data, name="输入")
    if x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("输入须为有限二维数组(channels, samples)，至少2个样本")
    amplitude = np.max(np.abs(x), axis=1, keepdims=True)
    scaled = np.divide(x, amplitude, out=np.zeros_like(x), where=amplitude > 0)
    centered = scaled - scaled.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    active = np.flatnonzero(rms > 64 * np.finfo(float).eps)
    if not len(active):
        raise DegenerateSignalError("全零或全部恒值通道，拒绝输出任意最高分类")
    return centered[active] / rms[active, None], active


def assess_signal_quality(data: np.ndarray, *, unit: str = "UNKNOWN",
                           max_abs: Optional[float] = None,
                           rail_min: Optional[float] = None,
                           rail_max: Optional[float] = None,
                           rail_margin: float = 0.0,
                           max_clip_fraction: float = 0.0,
                           jump_z: float = 12.0,
                          min_valid_channels: int = 6,
                          max_warning_bad_channels: int = 2) -> dict[str, Any]:
    """检查标准化前的原始 EEG，不改变 FBCCA 的输入和公式。

    绝对幅值和削顶只有在设备单位/轨值明确时才有物理意义；未知单位只产生
    警告，避免把任意 LSL 数值误当成微伏或伏特。返回的摘要可直接写入试次日志。
    rail_margin 使用与输入/轨值相同的数值单位；max_clip_fraction 是[0, 1)样本比例。
    """
    if not math.isfinite(rail_margin) or rail_margin < 0:
        raise ValueError("rail_margin 必须是非负有限的EEG数值余量")
    if not math.isfinite(max_clip_fraction) or not 0 <= max_clip_fraction < 1:
        raise ValueError("max_clip_fraction 必须是[0, 1)内的有限样本比例")
    try:
        x = _validate_eeg_ct(data, name="原始EEG")
    except ValueError as exc:
        raise InvalidTrial("原始EEG形状不足，无法进行质量检查") from exc
    if min(x.shape, default=0) < 2:
        raise InvalidTrial("原始EEG形状不足，无法进行质量检查")
    report: dict[str, Any] = {
        "unit": str(unit), "hard_fail": False, "hard_reasons": [], "warnings": [],
        "valid_channels": [], "bad_channels": [], "max_abs": [], "peak_to_peak": [],
        "clip_fraction": [], "jump_z": [],
    }
    if not np.isfinite(x).all():
        report["hard_fail"] = True
        report["hard_reasons"].append("EEG含NaN/Inf")
        return report

    centered = x - np.mean(x, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1))
    scale = np.maximum(np.max(np.abs(x), axis=1), np.finfo(float).eps)
    active = rms > 64 * np.finfo(float).eps * scale
    report["valid_channels"] = np.flatnonzero(active).astype(int).tolist()
    report["bad_channels"] = np.flatnonzero(~active).astype(int).tolist()
    if len(report["bad_channels"]):
        report["warnings"].append(
            "恒值/无变化通道: " + ",".join(str(i + 1) for i in report["bad_channels"]))
    if len(report["bad_channels"]) > max_warning_bad_channels or len(report["valid_channels"]) < min_valid_channels:
        report["hard_fail"] = True
        report["hard_reasons"].append(
            f"有效通道仅{len(report['valid_channels'])}个，至少需要{min_valid_channels}个")

    max_abs_values = np.max(np.abs(x), axis=1)
    p2p_values = np.ptp(x, axis=1)
    report["max_abs"] = max_abs_values.tolist()
    report["peak_to_peak"] = p2p_values.tolist()
    unit_known = str(unit).upper() not in ("AUTO", "UNKNOWN", "UNVERIFIED", "")
    if max_abs is not None and unit_known:
        if not np.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("max_abs 必须为有限正数")
        too_large = np.flatnonzero(max_abs_values > max_abs)
        if len(too_large):
            report["hard_fail"] = True
            report["hard_reasons"].append(
                "幅值超过阈值的通道: " + ",".join(str(i + 1) for i in too_large))

    clip_fraction = np.zeros(x.shape[0], dtype=float)
    if not unit_known:
        report["warnings"].append("单位未确认，绝对幅值/削顶仅记录未门控")
    elif rail_min is not None or rail_max is not None:
        if (rail_min is None or rail_max is None or not rail_min < rail_max
                or not np.isfinite(rail_min) or not np.isfinite(rail_max)):
            raise ValueError("rail_min/rail_max 必须同时为有限值且 min < max")
        if 2 * rail_margin >= rail_max - rail_min:
            raise ValueError("rail_margin 必须小于设备轨值跨度的一半")
        clipped = (x <= rail_min + rail_margin) | (x >= rail_max - rail_margin)
        clip_fraction = clipped.mean(axis=1)
        if np.any(clip_fraction > max_clip_fraction):
            report["hard_fail"] = True
            report["hard_reasons"].append(
                "检测到设备轨值削顶通道: " + ",".join(
                    str(i + 1) for i in np.flatnonzero(clip_fraction > max_clip_fraction)))
    elif max_abs is None:
        report["warnings"].append("单位或设备满量程阈值未配置，绝对幅值/削顶仅记录未门控")
    report["clip_fraction"] = clip_fraction.tolist()

    jump_values = np.zeros(x.shape[0], dtype=float)
    for i, row in enumerate(x):
        diff = np.diff(row)
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        denom = max(1.4826 * mad, np.finfo(float).eps)
        jump_values[i] = float(np.max(np.abs(diff - med)) / denom) if len(diff) else 0.0
    report["jump_z"] = jump_values.tolist()
    jump_channels = np.flatnonzero(jump_values > max(0.0, float(jump_z)))
    if len(jump_channels):
        report["warnings"].append(
            "存在异常相邻跳变通道: " + ",".join(str(i + 1) for i in jump_channels))
    return report


def mains_noise_diagnostics(data: np.ndarray, fs: float,
                            notch_hz: Optional[float] = 50.0) -> dict[str, Any]:
    """Describe narrow-band power before receiver filtering, without rejecting EEG.

    The Hann-periodogram ratio is independent of channel units. A median ratio
    of at least 50% describes line-dominated input; it is not a validated SSVEP
    quality/rejection threshold and does not establish a physical noise source.
    Disabling the receiver notch still observes the default 49--51 Hz band.
    The denominator ends at min(90 Hz, Nyquist), excluding the Nyquist bin.
    Ratios using different denominator bands should not be directly compared.
    """
    from scipy.signal import periodogram

    x = _validate_eeg_ct(data, name="工频诊断EEG")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("工频诊断采样率必须为有限正数")
    center = 50.0 if notch_hz is None else float(notch_hz)
    if not math.isfinite(center) or center <= 0:
        raise ValueError("工频诊断中心频率必须为有限正数")
    reference_high = min(90.0, fs / 2)
    report: dict[str, Any] = {
        "available": False,
        "method": "Hann periodogram; per-channel line-band / reference-band power",
        "input_scope": "uniform EEG before receiver notch and bandpass filtering",
        "sampling_rate_hz": float(fs),
        "receiver_notch_hz": notch_hz,
        "line_center_hz": center,
        "line_center_source": "default_50_hz_notch_off" if notch_hz is None else "receiver_notch_setting",
        "line_band_hz": [center - 1.0, center + 1.0],
        "reference_band_hz": [6.0, float(reference_high)],
        "nyquist_bin_excluded": True,
        "frequency_resolution_hz": float(fs / x.shape[1]),
        "line_fraction_per_channel": [None] * x.shape[0],
        "median_line_fraction": None,
        "valid_channel_indices": [],
        "line_dominated": False,
        "description_threshold": 0.5,
        "rejection_threshold_validated": False,
        "used_for_rejection": False,
        "warnings": [],
    }
    if x.shape[1] < 32 or not np.isfinite(x).all():
        report["unavailable_reason"] = "insufficient or non-finite samples"
        return report
    if (center - 1.0 < 6.0 or center + 1.0 > reference_high
            or center + 1.0 >= fs / 2):
        report["unavailable_reason"] = "full line band not covered below Nyquist / within the reference band"
        return report
    try:
        normalized, active = _normalised_rows(x)
    except DegenerateSignalError:
        report["unavailable_reason"] = "no varying channels"
        return report
    frequencies, psd = periodogram(normalized, fs=fs, window="hann",
                                  detrend="constant", axis=-1)
    line = (frequencies >= center - 1.0) & (frequencies <= center + 1.0)
    reference = ((frequencies >= 6.0) & (frequencies <= reference_high)
                 & (frequencies < fs / 2))
    if not np.any(line) or not np.any(reference):
        report["unavailable_reason"] = "no frequency bins in the diagnostic bands"
        return report
    denominator = psd[:, reference].sum(axis=-1)
    usable = denominator > np.finfo(float).tiny
    if not np.any(usable):
        report["unavailable_reason"] = "no measurable power in the reference band"
        return report
    fractions = np.clip(psd[usable][:, line].sum(axis=-1) / denominator[usable], 0.0, 1.0)
    indices = active[usable]
    for index, fraction in zip(indices, fractions):
        report["line_fraction_per_channel"][int(index)] = float(fraction)
    median = float(np.median(fractions))
    report.update(available=True, valid_channel_indices=indices.tolist(),
                  median_line_fraction=median, line_dominated=median >= 0.5)
    if report["line_dominated"]:
        report["warnings"].append(
            f"工频附近能量占比高：{center - 1:g}–{center + 1:g} Hz / 6–{reference_high:g} Hz"
            f"通道中位数={median:.1%}；仅作相对频谱提示，不用于拒识或物理幅值判断")
    return report


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
    return _fusion_whitened(bands, wy, a, b, regularization)


def _fusion_whitened(bands: np.ndarray, wy: Sequence[np.ndarray],
                     a: float, b: float, regularization: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量计算各候选的小矩阵SVD；保持原最大典型相关和融合公式。"""
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError("权重参数a、b须非负且有限")
    # 公共API允许不同参考含有不同数量的恒值行，不能盲目stack。
    same_shape = len({ref.shape for ref in wy}) == 1
    transposed = np.stack(wy).swapaxes(-1, -2) if same_shape else None
    correlations = np.empty((len(bands), len(wy)))
    for k, band in enumerate(bands):
        wx = _whiten_rows(band, regularization)
        if transposed is not None:
            correlations[k] = np.clip(
                np.linalg.svd(wx @ transposed, compute_uv=False)[:, 0], 0, 1)
        else:
            for j, ref_w in enumerate(wy):
                correlations[k, j] = np.clip(
                    np.linalg.svd(wx @ ref_w.T, compute_uv=False)[0], 0, 1)
    weights = np.arange(1, len(bands) + 1, dtype=float) ** (-a) + b
    return weights @ (correlations ** 2), correlations, weights


@lru_cache(maxsize=32)
def _cached_reference_whitening(frequencies: tuple[float, ...], fs: float,
                                n_samples: int, n_harmonics: int,
                                regularization: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """只缓存确定性的数学参考，不缓存EEG、标签或试次结果。"""
    refs = generate_reference_signals(frequencies, fs, n_samples, n_harmonics)
    white = tuple(_whiten_rows(ref, regularization) for ref in refs)
    refs.setflags(write=False)
    for ref in white:
        ref.setflags(write=False)
    return refs, white


@lru_cache(maxsize=32)
def _cached_filter_design(fs: float, notch_hz: Optional[float],
                          bands: tuple[tuple[float, float], ...]) -> tuple:
    from scipy.signal import cheby1, iirnotch
    notch = None if notch_hz is None else iirnotch(notch_hz, 30.0, fs=fs)
    bank = tuple(cheby1(4, .5, [low, high], btype="bandpass", fs=fs, output="sos")
                 for low, high in bands)
    for coefficient in (*(() if notch is None else notch), *bank):
        coefficient.setflags(write=False)
    return notch, bank


def _slice_eeg_window(data: np.ndarray, fs: float, window_s: float,
                      onset_s: float, *, short_message: str) -> np.ndarray:
    """Select one complete EEG window after the caller has checked layout and sampling rate."""
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("window_s须为正、onset_s须非负")
    start, length = round(onset_s * fs), sample_count(window_s, fs)
    if length < 32 or start + length > data.shape[1]:
        raise ValueError(short_message)
    return data[:, start:start + length].copy()


def _regress_line_noise(window: np.ndarray, fs: float, line_hz: float) -> np.ndarray:
    """仅使用当前窗拟合常数/正弦/余弦，减去正弦和余弦；与离线冻结方案一致。"""
    x = window.copy()
    t = np.arange(x.shape[-1]) / fs
    design = np.column_stack((np.ones(t.size), np.sin(2*np.pi*line_hz*t),
                              np.cos(2*np.pi*line_hz*t)))
    coefficients = np.linalg.lstsq(design, x.T, rcond=None)[0]
    x -= (design[:, 1:] @ coefficients[1:]).T
    return x


def _preprocess_validated_window(window: np.ndarray, fs: float, window_s: float,
                                 target_fs: float, notch_hz: Optional[float],
                                 bands: tuple[tuple[float, float], ...]
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Filter a sliced, checked window; the public entry points validate it once."""
    from scipy.signal import filtfilt, resample_poly, sosfiltfilt

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
        if not np.isfinite(notch_hz) or not 0 < notch_hz < min(fs, target_fs) / 2:
            raise ValueError("陷波频率须位于0与Nyquist之间")
    notch, filters = _cached_filter_design(target_fs, notch_hz, bands)
    if notch is not None:
        bn, an = notch
        window = filtfilt(bn, an, window, axis=-1)
    # scipy部分版本要求SOS系数为可写buffer；仅复制小系数矩阵。
    bank = [sosfiltfilt(sos.copy(), window, axis=-1) for sos in filters]
    return window, np.stack(bank)


def preprocess_lsl_window(data_ch_samples: np.ndarray, fs: float,
                          window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                          target_fs: float = TARGET_FS, notch_hz: Optional[float] = 50.0,
                          *, filter_bands: Optional[Sequence[Sequence[float]]] = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """保留原 Chebyshev-I 阶数4/纹波0.5dB和filtfilt；不是连续因果滤波。

    输入为均匀时间网格。onset_s仅为数组裁剪偏移；事件截窗后必须传0。
    零相位滤波只使用已到齐的本窗口及SciPy默认反射边界，未引入后续试次。
    """
    bands = _validate_filter_bands(filter_bands, fs, target_fs)
    data = _validate_eeg_ct(data_ch_samples, name="LSL数据")
    if not np.isfinite(data).all():
        raise ValueError("LSL数据须为有限(channels, samples)数组")
    window = _slice_eeg_window(data, fs, window_s, onset_s,
                               short_message="数据不足或窗口过短；不得补零凑齐窗口")
    _normalised_rows(window)  # 对原始退化输入显式报错。
    return _preprocess_validated_window(window, fs, window_s, target_fs, notch_hz, bands)


def classify_eeg_window(data_ch_samples: np.ndarray, fs: float,
                        window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                        target_frequencies_hz: Optional[Sequence[float]] = None,
                        n_harmonics: int = N_HARMONICS, *, target_fs: float = TARGET_FS,
                        notch_hz: Optional[float] = 50.0, a: float = 1.25, b: float = .25,
                        regularization: float = 1e-10, min_valid_channels: int = 1,
                        filter_bands: Optional[Sequence[Sequence[float]]] = None,
                        line_regression_hz: Optional[float] = None,
                        calibration: Optional[PersonalCalibration] = None) -> dict:
    """只接收EEG/算法参数，不接受真实目标，防止标签泄漏。

    FBCCA和标准CCA共享完全相同的EEG窗、通道、参考和第一子带。
    分数不是概率；本方法仍强制选择最高分，没有空闲检测或自由输入可靠性保证。
    """
    frequencies = (BENCHMARK_FREQUENCIES_HZ if target_frequencies_hz is None
                   else np.asarray(target_frequencies_hz, float))
    bands = _validate_filter_bands(filter_bands, fs, target_fs)
    _validate_harmonics(n_harmonics, frequencies, fs)
    _validate_line_regression_hz(line_regression_hz, min(fs, target_fs))
    data = _validate_eeg_ct(data_ch_samples, name="EEG")
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("EEG须二维，窗长为正，偏移非负")
    window = _slice_eeg_window(data, fs, window_s, onset_s,
                               short_message="EEG时间窗覆盖不足")
    if line_regression_hz is not None:
        window = _regress_line_noise(window, fs, line_regression_hz)
    normalized, active = _normalised_rows(window)
    if isinstance(min_valid_channels, bool) or not isinstance(min_valid_channels, int) or min_valid_channels < 1:
        raise ValueError("min_valid_channels须为正整数")
    if len(active) < min_valid_channels:
        raise DegenerateSignalError(f"有效变化通道仅{len(active)}个，至少需要{min_valid_channels}个")
    # 输入窗口已裁好，后续不再重复裁剪或加140ms。单位标准化先于滤波。
    window, bank = _preprocess_validated_window(normalized, fs, window_s, target_fs, notch_hz, bands)
    refs, white_refs = _cached_reference_whitening(
        tuple(float(f) for f in frequencies), float(target_fs), window.shape[-1],
        n_harmonics, float(regularization))
    scores, rho, weights = _fusion_whitened(bank, white_refs, a, b, regularization)
    baseline_scores = scores.copy()
    features = None
    fallback = None
    if calibration is not None:
        calibration.validate_decoder(frequencies, target_fs, window_s, n_harmonics,
                                     notch_hz, bands, regularization, line_regression_hz,
                                     a, b, data.shape[0])
        if len(active) != data.shape[0]:
            fallback = "通道缺失，个人空间模板不可用，本试次回退FBCCA"
        else:
            try:
                scores, features = calibration.score(bank, weights)
            except DegenerateSignalError:
                fallback = "滤波后通道退化，本试次回退FBCCA"
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
        "filter_bands_hz": bands,
        "n_harmonics": n_harmonics,
        "line_regression_hz": line_regression_hz,
        "decoder": "FB-eCCA" if features is not None else "FBCCA",
        "calibration_id": calibration.model_id if calibration is not None else None,
        "calibration_fallback": fallback,
        "fbcca_scores": baseline_scores,
        "fbcca_prediction": int(np.argmax(baseline_scores)) + 1,
        "ecca_features": features,
    }


def calibration_settings(cfg: Config) -> dict:
    """模板的采样、通道、相位和预处理契约；不包含测试标签或目标顺序。"""
    return {
        "frequencies": BENCHMARK_FREQUENCIES_HZ.tolist(),
        "phases": [t.phase_rad for t in TARGETS],
        "channel_indices": list(cfg.channel_indices),
        "channel_positions": list(cfg.channel_positions),
        "target_fs": cfg.target_fs, "window_s": cfg.window_s,
        "response_delay_s": cfg.response_delay_s,
        "timestamp_mode": cfg.timestamp_mode,
        "timestamp_processing": ("mne_lsl_streamlsl_clocksync_dejitter_monotize_v1"
                                 if cfg.timestamp_mode == "lsl" else "legacy_sample_clock"),
        "device_timestamp_lag_s": cfg.device_timestamp_lag_s,
        "n_harmonics": cfg.n_harmonics, "notch_hz": cfg.notch_hz,
        "filter_bands": [list(b) for b in cfg.filter_bands],
        "regularization": cfg.cca_regularization,
        "line_regression_hz": cfg.line_regression_hz,
        "a": cfg.weight_a, "b": cfg.weight_b,
    }


def _calibration_rows(x: np.ndarray) -> np.ndarray:
    z, active = _normalised_rows(x)
    if len(active) != len(x):
        raise DegenerateSignalError("个人模板要求所有校准通道有效")
    return z


def _calibration_whitener(x: np.ndarray, reg: float) -> np.ndarray:
    cov = x @ x.T / (x.shape[-1] - 1)
    cov += reg * np.trace(cov) / len(cov) * np.eye(len(cov))
    return _inverse_sqrt_covariance(cov, reg)


def _projected_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.clip(np.divide(np.sum(x * y, axis=-1), denominator,
                            out=np.zeros_like(denominator), where=denominator > 0), -1, 1)


class PersonalCalibration:
    """固定四特征 eCCA + 原滤波器组权重，所有40目标始终参与分类。

    参照 Nakanishi et al. 2015, doi:10.1371/journal.pone.0140703，式14–15。
    每个训练窗先逐通道标准化；各子带再次标准化后逐类平均。
    三种CCA空间滤波保留投影相关的正负号，sum(sign(r)*r**2)融合。
    不训练阈值、不自学习、不使用当前真实标签；不会把分数解释为概率。
    """
    schema_version = 1

    def __init__(self, templates: np.ndarray, counts: np.ndarray, metadata: dict):
        self._metadata = json.loads(json.dumps(metadata, allow_nan=False))
        m = self._metadata
        if m.get("schema_version") != self.schema_version or m.get("algorithm") != "FB-eCCA-4":
            raise ValueError("不支持的个人模型版本")
        if not isinstance(m.get("participant_id"), str) or not m["participant_id"].strip():
            raise ValueError("个人模型缺少受试者编号")
        s = m["settings"]
        self._settings = s
        # 显式验证磁盘设置，不允许损坏模型进入矩阵运算。
        _validate_filter_bands(s["filter_bands"], s["target_fs"], s["target_fs"])
        _validate_harmonics(s["n_harmonics"], np.asarray(s["frequencies"]), s["target_fs"])
        _validate_line_regression_hz(s["line_regression_hz"], s["target_fs"])
        for value in (s["regularization"], s["a"], s["b"], s["response_delay_s"]):
            if not np.isfinite(value) or value < 0:
                raise ValueError("个人模型正则化、权重或时延参数异常")
        if not np.isfinite(s["device_timestamp_lag_s"]):
            raise ValueError("个人模型设备时延参数异常")
        if s["notch_hz"] is not None and not 0 < s["notch_hz"] < s["target_fs"] / 2:
            raise ValueError("个人模型陷波参数异常")
        expected = (len(s["filter_bands"]), len(TARGETS), len(s["channel_positions"]),
                    sample_count(s["window_s"], s["target_fs"]))
        templates = np.array(templates, dtype=float, copy=True)
        counts = np.asarray(counts)
        if (templates.shape != expected or not np.isfinite(templates).all()
                or counts.shape != (len(TARGETS),) or counts.dtype.kind not in "iu"
                or np.any(counts < 1) or expected[-1] < 32
                or m.get("training_trials") != int(counts.sum())
                or len(s["channel_indices"]) != expected[2]
                or s["frequencies"] != BENCHMARK_FREQUENCIES_HZ.tolist()
                or s["phases"] != [t.phase_rad for t in TARGETS]):
            raise ValueError("个人模型模板、类别覆盖或目标表异常")
        _normalise_channel_indices(s["channel_indices"])
        for band in templates:
            for template in band:
                _calibration_rows(template)
        if (not np.allclose(templates.mean(-1), 0, atol=1e-8)
                or not np.allclose(templates.std(-1), 1, atol=1e-8)):
            raise ValueError("个人模板须逐通道中心化/标准化")
        self.templates = templates
        self.counts = np.array(counts, dtype=np.int64, copy=True)
        reg = s["regularization"]
        refs = np.array([_calibration_rows(r) for r in generate_reference_signals(
            s["frequencies"], s["target_fs"], expected[-1], s["n_harmonics"])])
        self._white_refs = np.array([_calibration_whitener(r, reg) @ r for r in refs])
        self._wt = np.array([[_calibration_whitener(t, reg) for t in band] for band in templates])
        self._white_templates = self._wt @ templates
        u, _, _ = np.linalg.svd(self._white_templates @ self._white_refs.swapaxes(-1, -2)
                                / (expected[-1] - 1), full_matrices=False)
        self._template_filters = np.einsum("bkij,bkj->bki", self._wt, u[:, :, :, 0])
        self._projected_templates = np.einsum("bkc,bkct->bkt", self._template_filters, templates)
        for value in (self.templates, self.counts, self._white_refs, self._wt,
                      self._white_templates, self._template_filters, self._projected_templates):
            value.setflags(write=False)
        digest = hashlib.sha256(json.dumps(m, sort_keys=True, allow_nan=False).encode())
        digest.update(self.templates.astype("<f8").tobytes())
        digest.update(self.counts.astype("<i8").tobytes())
        self.model_id = digest.hexdigest()

    @property
    def metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    @classmethod
    def fit(cls, windows: Sequence[np.ndarray], labels: Sequence[int],
            fs: float | Sequence[float],
            cfg: Config, *, sources: Sequence[dict] = ()) -> PersonalCalibration:
        cfg.validate()
        if not cfg.participant_id or not cfg.participant_id.strip():
            raise ValueError("训练个人模型必须声明受试者编号")
        y = np.asarray(labels)
        if (y.ndim != 1 or y.dtype.kind not in "iu" or len(y) != len(windows)
                or np.any((y < 1) | (y > len(TARGETS)))):
            raise ValueError("校准标签必须为与窗口一一对应的1–40整数")
        counts = np.bincount(y, minlength=len(TARGETS) + 1)[1:]
        if np.any(counts == 0):
            raise ValueError(f"个人模板缺少目标：{(np.flatnonzero(counts == 0) + 1).tolist()}")
        try:
            sampling_rates = np.asarray(fs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("校准窗采样率须为数值") from exc
        if sampling_rates.ndim == 0:
            sampling_rates = np.full(len(windows), float(sampling_rates))
        if (sampling_rates.shape != (len(windows),)
                or not np.isfinite(sampling_rates).all()
                or np.any(sampling_rates <= 0)):
            raise ValueError("校准窗采样率须与窗口一一对应且为有限正数")
        banks = []
        for x, window_fs in zip(windows, sampling_rates):
            x = _validate_eeg_ct(x)
            if x.shape != (len(cfg.channel_positions), sample_count(cfg.window_s, window_fs)):
                raise ValueError("校准窗的通道数/长度与配置不一致；须提供已对齐的完整窗")
            bands = _validate_filter_bands(cfg.filter_bands, window_fs, cfg.target_fs)
            _validate_harmonics(cfg.n_harmonics, BENCHMARK_FREQUENCIES_HZ, window_fs)
            _validate_line_regression_hz(cfg.line_regression_hz, min(window_fs, cfg.target_fs))
            if cfg.line_regression_hz is not None:
                x = _regress_line_noise(x, window_fs, cfg.line_regression_hz)
            _, bank = _preprocess_validated_window(_calibration_rows(x), window_fs, cfg.window_s,
                cfg.target_fs, cfg.notch_hz, bands)
            banks.append([_calibration_rows(b) for b in bank])
        banks = np.asarray(banks)
        templates = np.array([[_calibration_rows(b) for b in banks[y == k].mean(0)]
                              for k in range(1, len(TARGETS) + 1)]).swapaxes(0, 1)
        return cls(templates, counts, {
            "schema_version": cls.schema_version, "algorithm": "FB-eCCA-4",
            "participant_id": cfg.participant_id, "settings": calibration_settings(cfg),
            "sources": list(sources), "training_trials": len(y),
        })

    def validate_config(self, cfg: Config) -> None:
        if cfg.participant_id != self._metadata["participant_id"]:
            raise ValueError("个人模型受试者与 --participant 不一致")
        if (cfg.timestamp_mode == "lsl" and
                self._settings.get("timestamp_processing") != "mne_lsl_streamlsl_clocksync_dejitter_monotize_v1"):
            raise ValueError("个人模型使用旧接收路径；请用MNE-LSL重新完成40目标校准")
        if calibration_settings(cfg) != self._settings:
            raise ValueError("个人模型的通道、采样、时延、相位或滤波配置不匹配")
        if cfg.rejection_enabled and cfg.rejection_calibration_id != self.model_id:
            raise ValueError("个人模型改变了分数尺度；拒识阈值必须重新标定并绑定此模型ID")

    def validate_decoder(self, frequencies, fs, window_s, harmonics, notch, bands,
                         reg, line, a, b, channels) -> None:
        s = self._settings
        actual = [list(frequencies), fs, window_s, harmonics, notch, [list(v) for v in bands],
                  reg, line, a, b, channels]
        expected = [s["frequencies"], s["target_fs"], s["window_s"], s["n_harmonics"],
                    s["notch_hz"], s["filter_bands"], s["regularization"],
                    s["line_regression_hz"], s["a"], s["b"], len(s["channel_positions"])]
        if actual != expected:
            raise ValueError("分类参数与个人模型不匹配，拒绝套用模板")

    def score(self, bank: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bank, weights = np.asarray(bank, dtype=float), np.asarray(weights, dtype=float)
        expected = (self.templates.shape[0], *self.templates.shape[2:])
        if (bank.shape != expected or not np.isfinite(bank).all()
                or weights.shape != (expected[0],) or not np.isfinite(weights).all()
                or np.any(weights < 0)):
            raise ValueError("个人模板评分的子带形状或权重异常")
        features = []
        reg = self._settings["regularization"]
        for k, band in enumerate(bank):
            x = _calibration_rows(band)
            wx = _calibration_whitener(x, reg)
            xx = wx @ x
            u, singular, _ = np.linalg.svd(xx @ self._white_refs.swapaxes(-1, -2)
                                         / (x.shape[-1] - 1), full_matrices=False)
            ax = u[:, :, 0] @ wx
            ut, _, _ = np.linalg.svd(xx @ self._white_templates[k].swapaxes(-1, -2)
                                    / (x.shape[-1] - 1), full_matrices=False)
            at = ut[:, :, 0] @ wx
            template = self.templates[k]
            features.append(np.array([
                np.clip(singular[:, 0], 0, 1),
                _projected_correlation(at @ x, np.einsum("kc,kct->kt", at, template)),
                _projected_correlation(ax @ x, np.einsum("kc,kct->kt", ax, template)),
                _projected_correlation(self._template_filters[k] @ x, self._projected_templates[k]),
            ]))
        features = np.asarray(features)
        return weights @ np.sum(np.sign(features) * features ** 2, axis=1), features

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        SessionRecorder._atomic_npz(path, {
            "templates": self.templates, "counts": self.counts,
            "metadata_json": np.array(json.dumps(self._metadata, ensure_ascii=False, allow_nan=False)),
            "model_id": np.array(self.model_id),
        })

    @classmethod
    def load(cls, path: str | Path, cfg: Config) -> PersonalCalibration:
        path = Path(path)
        if path.suffix.lower() in (".pkl", ".zip"):
            # 会话PKL只允许内建容器及字节串；不执行外部类的反序列化构造器。
            class DataOnlyUnpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    raise pickle.UnpicklingError(f"不允许加载PKL对象：{module}.{name}")

            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    member = path.stem + "_model.pkl"
                    with archive.open(member) as handle:
                        bundle = DataOnlyUnpickler(handle).load()
            else:
                with path.open("rb") as handle:
                    bundle = DataOnlyUnpickler(handle).load()
            if not isinstance(bundle, dict) or bundle.get("format") != "fbcca_keyboard_model_v1":
                raise ValueError("不支持的会话PKL模型格式")
            if bundle.get("training_status") != "trained":
                raise ValueError("该会话PKL没有完整的40类个人模型：" +
                                 str(bundle.get("training_reason", "样本不足")))
            calibration = bundle.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError("会话PKL缺少个人模板")
            shape = calibration.get("templates_shape")
            raw = calibration.get("templates_float64_le")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 4
                    or not all(isinstance(v, int) and v > 0 for v in shape)
                    or not isinstance(raw, bytes)
                    or len(raw) != math.prod(shape) * 8):
                raise ValueError("会话PKL模板形状或字节长度异常")
            templates = np.frombuffer(raw, dtype="<f8").reshape(shape)
            model = cls(templates, np.asarray(calibration["counts"]), calibration["metadata"])
            if model.model_id != calibration.get("model_id"):
                raise ValueError("个人模型完整性校验失败")
        else:
            with np.load(path, allow_pickle=False) as saved:
                model = cls(saved["templates"], saved["counts"], json.loads(saved["metadata_json"].item()))
                if model.model_id != saved["model_id"].item():
                    raise ValueError("个人模型完整性校验失败")
        model.validate_config(cfg)
        return model


# ======================== 连续采集与事件截窗 ========================
@dataclass
class Epoch:
    data: np.ndarray
    fs: float
    raw_timestamps: np.ndarray
    local_timestamps: np.ndarray
    clock_corrections: np.ndarray
    diagnostics: dict[str, Any]
    source_data: np.ndarray
    uniform_timestamps: np.ndarray
    source_timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.data = _validate_eeg_ct(self.data, name="Epoch.data",
                                     allow_empty_samples=True)
        self.source_data = _validate_eeg_ct(
            self.source_data, name="Epoch.source_data", allow_empty_samples=True)
        self.raw_timestamps = np.asarray(self.raw_timestamps, dtype=float)
        self.local_timestamps = np.asarray(self.local_timestamps, dtype=float)
        self.clock_corrections = np.asarray(self.clock_corrections, dtype=float)
        self.uniform_timestamps = np.asarray(self.uniform_timestamps, dtype=float)
        self.source_timestamps = np.asarray(
            self.raw_timestamps.copy() if self.source_timestamps is None else self.source_timestamps,
            dtype=float)
        self.validate_dimensions()

    def validate_dimensions(self) -> None:
        """Validate C×T arrays and their one-dimensional time axes."""
        if self.data.shape[0] != self.source_data.shape[0]:
            raise ValueError("Epoch.data 与 Epoch.source_data 通道数不一致")
        for name, timestamps in (
                ("raw_timestamps", self.raw_timestamps),
                ("source_timestamps", self.source_timestamps),
                ("local_timestamps", self.local_timestamps),
                ("clock_corrections", self.clock_corrections)):
            if timestamps.ndim != 1:
                raise ValueError(f"Epoch.{name}必须是一维时间轴")
            if len(timestamps) != self.source_data.shape[1]:
                raise ValueError(
                    f"Epoch.{name}长度必须等于source_data的采样点数")
        if self.uniform_timestamps.ndim != 1:
            raise ValueError("Epoch.uniform_timestamps必须是一维时间轴")
        if len(self.uniform_timestamps) != self.data.shape[1]:
            raise ValueError(
                "Epoch.uniform_timestamps长度必须等于data的采样点数")


def _parse_stream_metadata(info: Any, indices: Sequence[int]) -> dict:
    xml = info.as_xml
    if callable(xml):
        xml = xml()
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
    def value(name: str, legacy_name: str) -> Any:
        item = getattr(info, name, None)
        return item if item is not None else getattr(info, legacy_name)()
    return {"name": value("name", "name"), "type": value("stype", "type"),
            "source_id": value("source_id", "source_id"),
            "channel_count": value("n_channels", "channel_count"),
            "nominal_fs": float(value("sfreq", "nominal_srate")),
            "selected_channels": selected, "publisher_filter_metadata": filter_nodes or ["UNKNOWN"],
            "raw_xml": xml}


class MNEStreamWindow:
    """从 MNE-LSL 的环形缓冲读取事件窗口，不另建采集缓冲。"""

    timestamp_mode = "lsl"

    def __init__(self, stream: Any, indices: Sequence[int], device_lag_s: float):
        self.stream = stream
        self.indices = list(indices)
        self.n_channels = len(indices)
        self.device_lag_s = device_lag_s
        self.total_received = 0

    def snapshot(self) -> tuple[np.ndarray, ...]:
        # get_data()返回缓冲视图，且数据/时间戳分别更新；两次完整读取
        # 必须一致，避免恰好读到一半旧样本、一半新时间戳。
        for _ in range(5):
            data1, ts1 = self.stream.get_data(picks=self.indices)
            x = np.asarray(data1, dtype=float).copy()
            received = np.asarray(ts1, dtype=float).copy()
            data2, ts2 = self.stream.get_data(picks=self.indices)
            other_x = np.asarray(data2, dtype=float).copy()
            other_ts = np.asarray(ts2, dtype=float).copy()
            if (x.shape == other_x.shape == (self.n_channels, len(received))
                    and np.array_equal(received, other_ts)
                    and np.array_equal(x, other_x, equal_nan=True)):
                break
            time.sleep(.002)
        else:
            raise InvalidTrial("MNE-LSL缓冲正在更新，无法获得一致的试次快照")
        nonempty = np.flatnonzero(received != 0)  # 未填满的环形缓冲以时间戳0表示空位。
        x, received = x[:, nonempty], received[nonempty]
        local = received - self.device_lag_s
        corrections = np.zeros(len(received), dtype=float)
        valid = np.isfinite(x).all(axis=0)
        return x, received, local, corrections, valid

    @property
    def latest(self) -> float:
        _, received = self.stream.get_data(
            winsize=1 / float(self.stream.info["sfreq"]), picks=[self.indices[0]])
        return float(received[-1] - self.device_lag_s) if len(received) and received[-1] else -math.inf

    def window_snapshot(self, start: float, end: float) -> tuple[np.ndarray, ...]:
        x, received, local, corrections, valid = self.snapshot()
        if len(local) < 2 or local[0] > start or local[-1] < end:
            raise InvalidTrial("事件窗口数据未到齐或已被MNE-LSL缓冲覆盖")
        left = int(np.searchsorted(local, start, side="right") - 1)
        right = int(np.searchsorted(local, end, side="left"))
        support = slice(left, right + 1)
        return (x[:, support], received[support], local[support],
                corrections[support], valid[support], received[support].copy())


    def epoch(self, start: float, duration_s: float, nominal_fs: float,
              max_gap_factor: float = 1.5, rate_tolerance: float = .02,
              hard_gap_factor: float = 2.0, hard_rate_tolerance: float = .05,
              max_warning_gaps: int = 1) -> Epoch:
        _validate_fs(nominal_fs, None)
        if not np.isfinite(start) or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("事件时间/窗口长度不合法")
        end = start + duration_s
        # 需要两侧插值支撑且至少到完整窗口终点；绝不使用np.interp的端点外推。
        x, raw, local, corrections, valid, source = self.window_snapshot(start, end)
        def failure_epoch(message: str) -> Epoch:
            return Epoch(
                data=np.empty((self.n_channels, 0), dtype=float), fs=nominal_fs,
                raw_timestamps=raw.copy(), local_timestamps=local.copy(),
                clock_corrections=corrections.copy(),
                diagnostics={"hard_reason": message, "timestamp_mode": self.timestamp_mode},
                source_data=x.copy(), uniform_timestamps=np.empty(0, dtype=float),
                source_timestamps=source.copy())
        if len(local) < 3 or not np.all(valid):
            message = "窗口含NaN/Inf或有效样本不足"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        dt_raw, dt_source, dt_local = np.diff(raw), np.diff(source), np.diff(local)
        nominal_dt = 1 / nominal_fs
        if np.any(dt_source <= 0) or np.any(dt_local <= 0):
            message = "窗口时间戳不严格递增"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        interval = np.maximum(dt_source, dt_local)
        gap_mask = interval > nominal_dt * max_gap_factor
        hard_gap_mask = interval >= nominal_dt * hard_gap_factor
        warnings: list[str] = ["LSL时间戳已同步/去抖；无设备样本计数器，不能逐点确认丢样或重复样本"]
        if np.any(hard_gap_mask) or int(np.count_nonzero(gap_mask)) > max_warning_gaps:
            message = ("LSL处理后窗口存在过大时间戳间隔："
                       f"warning_gaps={int(np.count_nonzero(gap_mask))}, "
                       f"hard_gaps={int(np.count_nonzero(hard_gap_mask))}")
            raise InvalidTrial(message, epoch=failure_epoch(message))
        if np.any(gap_mask):
            warnings.append(f"存在{int(np.count_nonzero(gap_mask))}个轻微异常采样间隔")
        measured_fs = (len(local) - 1) / (local[-1] - local[0])
        source_measured_fs = (len(source) - 1) / (source[-1] - source[0])
        local_rate_error = abs(measured_fs / nominal_fs - 1)
        source_rate_error = abs(source_measured_fs / nominal_fs - 1)
        if max(local_rate_error, source_rate_error) > hard_rate_tolerance:
            message = (f"源/本地时间戳采样率偏差过大：source={source_measured_fs:.3f}Hz, "
                       f"local={measured_fs:.3f}Hz, declared={nominal_fs:g}Hz")
            raise InvalidTrial(message, epoch=failure_epoch(message))
        if max(local_rate_error, source_rate_error) > rate_tolerance:
            warnings.append(
                f"时间戳采样率轻微偏差：source={source_measured_fs:.3f}Hz, local={measured_fs:.3f}Hz")
        _validate_fs(min(measured_fs, source_measured_fs), None)
        grid = start + np.arange(sample_count(duration_s, nominal_fs)) / nominal_fs
        if grid.size < 32 or grid[-1] > local[-1] or grid[0] < local[0]:
            message = "窗口不满足重采样时间覆盖"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        # 将LSL处理后的时间戳对齐到统一网格；逐点丢样仍需设备计数器核验。
        # 保留高原始采样率后交给resample_poly抗混叠降采样，不能直接插值到250 Hz。
        uniform = np.asarray([np.interp(grid, local, channel) for channel in x], dtype=float)
        return Epoch(data=uniform, fs=nominal_fs, raw_timestamps=raw.copy(),
                     local_timestamps=local.copy(), clock_corrections=corrections.copy(), diagnostics={
            "source_support_samples": len(local), "uniform_samples": len(grid),
            "timestamp_estimated_fs": float(measured_fs), "source_clock_estimated_fs": float(source_measured_fs),
            "max_source_interval_s": float(dt_source.max()), "max_local_interval_s": float(dt_local.max()),
            "timestamp_mode": self.timestamp_mode,
            "timestamp_semantics": "LSL postprocessed local_clock timestamps; original outlet stamps unavailable",
            "raw_nonmonotonic_count": int(np.count_nonzero(dt_raw <= 0)),
            "raw_min_interval_s": float(dt_raw.min()), "raw_max_interval_s": float(dt_raw.max()),
            "timestamp_adjustment_max_abs_s": float(np.max(np.abs(source - raw))),
            "source_support_start": float(local[0]), "source_support_end": float(local[-1]),
            "uniform_last_sample": float(grid[-1]), "end_is_exclusive": True,
            "clock_correction_min_s": float(corrections.min()), "clock_correction_max_s": float(corrections.max()),
            "clock_correction_scope": "already applied inside LSL; saved values are zero placeholders",
            "interpolation": "LSL-postprocessed timestamp grid; no extrapolation; device sample loss unverified",
            "warnings": warnings,
            "warning_gap_count": int(np.count_nonzero(gap_mask)),
        }, source_data=x.copy(), uniform_timestamps=grid.copy(), source_timestamps=source.copy())

class ContinuousLSL:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stream: Any = None
        self.buffer: Optional[MNEStreamWindow] = None
        self.metadata: dict = {}
        self.fs = math.nan
        self.error: Optional[BaseException] = None
        self.stop_event = threading.Event()
        self.clock: Callable[[], float] = time.monotonic
        self.clock_updates: list[dict[str, float]] = []
        self.estimated_fs = math.nan
        self.timestamp_mode = "lsl"
        self.last_arrival = time.monotonic()
        self.first_timestamp: Optional[float] = None
        self.last_timestamp = -math.inf

    def _timestamp_summary(self) -> dict[str, Any]:
        return {
            "mode": "mne_lsl_streamlsl",
            "flags": ["clocksync", "dejitter", "monotonize"],
            "clock_domain": "mne_lsl.lsl.local_clock",
            "manual_time_correction_applied": False,
            "sample_index_reconstruction": False,
            "acquisition_buffer": "MNE-LSL StreamLSL",
            "original_outlet_timestamps_available": False,
            "saved_raw_timestamp_field": "legacy name; contains LSL-postprocessed timestamps",
            "device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
            "accepted_samples_total": self.buffer.total_received if self.buffer is not None else 0,
            "sample_integrity": "device sample counter unavailable in selected EEG stream",
            "physical_latency": "unmeasured; GUI and display delays require independent validation",
        }

    def start(self) -> None:
        if self.cfg.timestamp_mode != "lsl":
            raise RuntimeFault("ACQUISITION", "实时采集仅支持LSL内建时间处理；旧重建模式已停用")
        try:
            from mne_lsl.lsl import local_clock, resolve_streams
            from mne_lsl.stream import StreamLSL
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeFault("LSL_LIBRARY", "不能加载MNE-LSL/liblsl：" + str(exc)) from exc
        self.clock = local_clock
        name = self.cfg.eeg_stream_name or None
        stype = None if name else self.cfg.eeg_stream_type
        streams = resolve_streams(timeout=self.cfg.connect_timeout_s, name=name, stype=stype)
        if not streams:
            visible = resolve_streams(timeout=min(1.0, self.cfg.connect_timeout_s))
            details = "; ".join(
                f"{sinfo.name!r}(type={sinfo.stype!r}, channels={sinfo.n_channels}, "
                f"fs={sinfo.sfreq:g}Hz)"
                for sinfo in visible
            ) or "无"
            raise RuntimeFault("LSL_NOT_FOUND",
                f"未找到LSL流 {('name', name) if name else ('type', stype)}。当前可见流：{details}\n"
                "设备连接成功不代表已开启LSL输出。本程序通过LSL接收EEG，不直接连接USB设备。\n"
                "若使用OpenBCI GUI：先Start Data Stream，再在Networking选择LSL，"
                "数据选择TimeSeriesRaw、Type填EEG，点击Start LSL Stream，保持GUI运行后重试。"
            )
        if len(streams) != 1:
            names = ", ".join(repr(sinfo.name) for sinfo in streams)
            raise RuntimeFault("LSL_CONNECTION",
                f"找到多个匹配的EEG流（{names}），请在CONFIG.eeg_stream_name填写唯一名称"
            )
        # 由MNE-LSL负责后台采集和环形缓冲；事件窗口通过get_data()读取。
        self.stream = StreamLSL(self.cfg.buffer_s, name=streams[0].name,
                                stype=streams[0].stype,
                                source_id=streams[0].source_id or None)
        try:
            self.stream.connect(acquisition_delay=.01,
                                processing_flags=("clocksync", "dejitter", "monotize"),
                                timeout=self.cfg.connect_timeout_s)
            info = self.stream.sinfo
            self.fs = float(info.sfreq)
            _validate_fs(self.fs, max(high for _, high in self.cfg.filter_bands))
            if max(self.cfg.channel_indices) >= info.n_channels:
                raise RuntimeError("所选LSL通道索引超出实际流通道数")
            self.metadata = _parse_stream_metadata(info, self.cfg.channel_indices)
            self.metadata["timestamp_processing"] = self._timestamp_summary()
            print("[时间轴] MNE-LSL StreamLSL后台采集与环形缓冲；"
                  "LSL内建时钟同步、去抖和单调化。\n"
                  f"启动预热{self.cfg.warmup_s:g}秒；原始发布端时间戳在此模式下不可恢复。"
                  "设备和屏幕的物理延迟仍需独立测量。", flush=True)
            self.buffer = MNEStreamWindow(self.stream, self.cfg.channel_indices,
                                          self.cfg.device_timestamp_lag_s)
            self.last_arrival = time.monotonic()
            self.stream.add_callback(self._observe_chunk)
        except BaseException:
            self.close()
            raise
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            self.check_health()
            if (self.first_timestamp is not None and self.buffer.total_received > 2
                    and self.last_timestamp - self.first_timestamp >= self.cfg.warmup_s):
                self.estimated_fs = ((self.buffer.total_received - 1) /
                                     (self.last_timestamp - self.first_timestamp))
                startup_error = abs(self.estimated_fs / self.fs - 1)
                if startup_error > self.cfg.hard_rate_tolerance:
                    raise RuntimeFault("ACQUISITION",
                        f"启动LSL时间戳速率不符：local={self.estimated_fs:.3f}Hz, "
                        f"declared={self.fs:g}Hz")
                if startup_error > self.cfg.rate_tolerance:
                    print(f"[时间轴警告] 启动LSL采样率{self.estimated_fs:.3f}Hz，"
                          f"与声明值{self.fs:g}Hz轻微偏差。", flush=True)
                _validate_fs(self.estimated_fs, max(high for _, high in self.cfg.filter_bands))
                return
            self.stop_event.wait(.02)
        raise RuntimeFault("ACQUISITION", "已找到LSL流，但启动阶段未收到足够的连续EEG数据")

    def _observe_chunk(self, data: np.ndarray, ts: np.ndarray, info: Any) -> tuple[np.ndarray, np.ndarray]:
        """MNE-LSL采集回调仅记录健康状态，不复制或重建EEG缓冲。"""
        try:
            stamps = np.asarray(ts, dtype=float)
            if (data.ndim != 2 or len(stamps) != len(data)
                    or data.shape[1] <= max(self.cfg.channel_indices)
                    or not np.isfinite(stamps).all()):
                raise InvalidTrial("MNE-LSL采集块形状或时间戳无效")
            if len(stamps) and (np.any(np.diff(stamps) <= 0) or stamps[0] <= self.last_timestamp):
                raise InvalidTrial("MNE-LSL时间戳回退或重复，停止以避免刺激与EEG错配")
            if len(stamps):
                if self.first_timestamp is None:
                    self.first_timestamp = float(stamps[0])
                self.last_timestamp = float(stamps[-1])
                self.last_arrival = time.monotonic()
                if self.buffer is not None:
                    self.buffer.total_received += len(stamps)
        except BaseException as exc:
            self.error = exc
        return data, ts

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeFault("ACQUISITION", f"采集线程异常：{self.error}") from self.error
        if self.stop_event.is_set():
            raise RuntimeFault("ACQUISITION", "采集已停止")
        if self.stream is not None:
            if not self.stream.connected:
                raise RuntimeFault("ACQUISITION", "MNE-LSL采集线程已断开")
            if time.monotonic() - self.last_arrival > self.cfg.max_receive_age_s:
                raise RuntimeFault("ACQUISITION", "EEG流停止发送，已停止实验")

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
        return self.buffer.epoch(
            start, self.cfg.window_s, self.fs, self.cfg.max_gap_factor,
            self.cfg.rate_tolerance, self.cfg.hard_gap_factor,
            self.cfg.hard_rate_tolerance, self.cfg.max_warning_gaps)

    def close(self) -> None:
        self.stop_event.set()
        if self.stream is not None:
            try:
                if self.stream.connected:
                    self.stream.disconnect()
            finally:
                self.stream = None
        self.metadata["timestamp_processing"] = self._timestamp_summary()


def decode_trial(trial_id: int, onset: float, collector: ContinuousLSL, cfg: Config,
                 cancel: threading.Event) -> dict:
    """分类工作线程；函数签名有意不包含真实类别。"""
    epoch = collector.wait_epoch(onset, cancel)
    if cancel.is_set():
        raise AbortSession("已取消分类")
    if cfg.input_unit == "AUTO":
        units = [str(item.get("unit", "UNKNOWN")) for item in
                 collector.metadata.get("selected_channels", [])]
        quality_unit = units[0] if units and len(set(units)) == 1 else "UNKNOWN"
    else:
        quality_unit = cfg.input_unit
    quality = assess_signal_quality(
        epoch.source_data, unit=quality_unit, max_abs=cfg.quality_max_abs,
        rail_min=cfg.quality_rail_min, rail_max=cfg.quality_rail_max,
        rail_margin=cfg.quality_rail_margin,
        max_clip_fraction=cfg.quality_max_clip_fraction, jump_z=cfg.quality_jump_z,
        min_valid_channels=cfg.min_valid_channels,
        max_warning_bad_channels=cfg.max_warning_bad_channels)
    quality["mains_noise"] = mains_noise_diagnostics(epoch.data, epoch.fs, cfg.notch_hz)
    quality["warnings"].extend(quality["mains_noise"]["warnings"])
    epoch.diagnostics["quality"] = quality
    epoch.diagnostics.setdefault("warnings", []).extend(quality["warnings"])
    if quality["hard_fail"]:
        raise InvalidTrial("EEG信号质量不合格：" + "; ".join(quality["hard_reasons"]), epoch=epoch)
    started = time.monotonic()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = classify_eeg_window(epoch.data, epoch.fs, window_s=cfg.window_s, onset_s=0.0,
            target_frequencies_hz=BENCHMARK_FREQUENCIES_HZ, n_harmonics=cfg.n_harmonics,
            target_fs=cfg.target_fs, notch_hz=cfg.notch_hz, a=cfg.weight_a, b=cfg.weight_b,
            regularization=cfg.cca_regularization, min_valid_channels=cfg.min_valid_channels,
            filter_bands=cfg.filter_bands, line_regression_hz=cfg.line_regression_hz,
            calibration=getattr(collector, "calibration", None))
    result.update(trial_id=trial_id, epoch=epoch, quality=quality,
                  computation_s=time.monotonic() - started)
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
            from mne_lsl.lsl import StreamInfo, StreamOutlet
            info = StreamInfo(cfg.marker_stream_name, "Markers", 1, 0,
                              "string", f"fbcca-keyboard-{uuid.uuid4()}")
            desc = info.desc
            desc.append_child_value("clock", "local_clock in PsychoPy callOnFlip")
            desc.append_child_value("physical_display_latency", "UNMEASURED")
            desc.append_child_value("event_delivery", "queued; original timestamp preserved")
            target_node = desc.append_child("targets")
            for target in TARGETS:
                node = target_node.append_child("target")
                for key, value in vars(target).items():
                    node.append_child_value(key, str(value))
            outlet = StreamOutlet(info, chunk_size=1)
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
    artifact_refs: dict[str, str] = field(default_factory=dict)
    cue_confirmed_at: Optional[float] = None
    fixation_onset: Optional[float] = None


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
        if record.status not in ("valid", "rejected", "invalid", "aborted"):
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
            if (not 1 <= pred <= len(TARGETS)
                    or not 1 <= record.result["fbcca_prediction"] <= len(TARGETS)
                    or not 1 <= record.result["cca_prediction"] <= len(TARGETS)):
                raise ValueError("分类结果超出范围")
            if not record.text_applied:
                raise RuntimeError("有效试次必须先显式写入字符，再提交")
        elif record.text_applied:
            raise RuntimeError("拒识/无效/中止试次不得写入字符")
        self.ids.add(record.trial_id)
        self.records.append(record)


def _json_safe(value: Any) -> Any:
    """将试次元数据转换为不依赖 pickle 的 JSON 值。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def spectral_quality(data: np.ndarray, fs: float, target_id: Optional[int],
                     n_harmonics: int) -> dict[str, Any]:
    """退出时计算频谱诊断，不参与在线门控；频率分辨率由真实窗长决定。"""
    x = np.asarray(data, dtype=float)
    if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 8
            or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0):
        return {"available": False, "reason": "No finite uniform EEG window"}
    from scipy.signal import periodogram
    frequencies, psd = periodogram(x, fs=fs, window="hann", detrend="constant",
                                  scaling="density", axis=-1)
    harmonics = []
    if target_id is not None:
        fundamental = TARGETS[target_id - 1].frequency_hz
        for order in range(1, n_harmonics + 1):
            frequency = fundamental * order
            if frequency >= fs / 2:
                break
            index = int(np.argmin(np.abs(frequencies - frequency)))
            # 跳过相邻1个频点，取两侧各第2至第4个频点作为局部背景。
            neighbors = np.asarray([index + offset for offset in (-4, -3, -2, 2, 3, 4)])
            neighbors = neighbors[(neighbors > 0) & (neighbors < len(frequencies) - 1)]
            noise = (psd[:, neighbors].mean(axis=1) if neighbors.size
                     else np.full(x.shape[0], np.nan))
            signal = psd[:, index]
            snr_db = np.full(x.shape[0], np.nan)
            valid = (signal > 0) & (noise > 0) & np.isfinite(noise)
            snr_db[valid] = 10 * (np.log10(signal[valid]) - np.log10(noise[valid]))
            harmonics.append({"order": order, "frequency_hz": frequency,
                              "bin_frequency_hz": float(frequencies[index]),
                              "psd_per_channel": signal, "snr_db_per_channel": snr_db})
    return _json_safe({
        "available": True, "sampling_rate_hz": fs,
        "method": "Hann periodogram; constant detrend; no zero padding or receiver filtering",
        "psd_units": "input_unit_squared_per_hz; physical unit requires acquisition metadata",
        "snr_method": "10*log10(nearest-bin PSD / mean PSD at bin offsets -4,-3,-2,2,3,4)",
        "interpretation": "Descriptive local spectral contrast, not calibrated detection confidence",
        "frequency_resolution_hz": float(frequencies[1] - frequencies[0]),
        "frequencies_hz": frequencies, "psd_per_channel": psd,
        "reference_class": target_id, "harmonics": harmonics,
    })


def _finite_median(values: Sequence[Any]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def _cued_model_bundle(archive: Any, session: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """从本场有效且有真实标签的EEG窗训练个人模板；不虚构交叉验证结果。"""
    trials = session["trials"]
    valid = [t for t in trials if t["status"] == "valid" and t["true_class"] is not None
             and f"trial_{t['trial_id']:04d}_uniform_data" in archive]
    counts = np.bincount([t["true_class"] for t in valid], minlength=len(TARGETS) + 1)[1:]
    missing = (np.flatnonzero(counts == 0) + 1).tolist()
    bundle: dict[str, Any] = {
        "format": "fbcca_keyboard_model_v1",
        "session_id": session["session_id"],
        "source_npz": session["session_id"] + ".npz",
        "online_decoder": session["algorithm"],
        "target_count": len(TARGETS),
        "valid_trial_ids": [t["trial_id"] for t in valid],
        "counts_by_class": counts.tolist(),
        "missing_class_ids": missing,
        "decoder_settings": calibration_settings(cfg),
        "training_status": "incomplete" if missing else "trained",
        "training_reason": (f"缺少{len(missing)}个类别的有效EEG窗" if missing else None),
        "used_for_online_decoding": False,
        "validation": "未对本场新训练的模板进行独立测试或留一验证",
        "calibration": None,
    }
    if missing:
        return bundle
    # 缺省受试者编号只标识这场会话，不暗示已识别人的身份。
    training_cfg = replace(cfg, participant_id=cfg.participant_id or session["session_id"],
                           calibration_file=None)
    windows = [archive[f"trial_{t['trial_id']:04d}_uniform_data"] for t in valid]
    try:
        model = PersonalCalibration.fit(
            windows, [t["true_class"] for t in valid],
            [t.get("sampling_rate_hz") for t in valid], training_cfg,
            sources=[{"session_id": session["session_id"],
                      "trial_ids": bundle["valid_trial_ids"], "scope": "valid_cued_trials"}])
    except (ValueError, DegenerateSignalError, np.linalg.LinAlgError) as exc:
        bundle["training_status"] = "failed"
        bundle["training_reason"] = f"{type(exc).__name__}: {exc}"
        return bundle
    bundle["calibration"] = {
        "metadata": model.metadata,
        "model_id": model.model_id,
        "counts": model.counts.tolist(),
        "templates_shape": list(model.templates.shape),
        "templates_float64_le": model.templates.astype("<f8", copy=False).tobytes(),
    }
    bundle["model_id"] = model.model_id
    bundle["participant_id"] = training_cfg.participant_id
    return bundle


def _cued_quality_report(session: dict[str, Any], bundle: dict[str, Any],
                         npz_sha256: str) -> dict[str, Any]:
    trials = session["trials"]
    summaries = []
    for target in TARGETS:
        attempts = [t for t in trials if t["true_class"] == target.class_id]
        valid = [t for t in attempts if t["status"] == "valid"]
        harmonics = []
        for order in range(1, session["decoder_settings"]["n_harmonics"] + 1):
            trial_medians = []
            for trial in valid:
                spectral = trial.get("spectral_quality") or {}
                for harmonic in spectral.get("harmonics", []):
                    if harmonic["order"] == order:
                        trial_medians.append(_finite_median(harmonic["snr_db_per_channel"]))
                        break
            harmonics.append({"order": order,
                              "frequency_hz": target.frequency_hz * order,
                              "median_snr_db": _finite_median(trial_medians)})
        summaries.append({"class_id": target.class_id, "symbol": target.symbol,
                          "frequency_hz": target.frequency_hz,
                          "attempts": len(attempts), "valid": len(valid),
                          "correct_online": sum(t["prediction"] == target.class_id for t in valid),
                          "harmonics": harmonics})
    frame = [t.get("frame_diagnostics") or {} for t in trials]
    trial_qc = []
    for trial in trials:
        quality = trial.get("quality") or {}
        timing = trial.get("frame_diagnostics") or {}
        trial_qc.append({
            "trial_id": trial["trial_id"], "block_id": trial["block_id"],
            "class_id": trial["true_class"], "status": trial["status"],
            "reason": trial.get("reason"), "prediction": trial.get("prediction"),
            "signal_unit": quality.get("unit"),
            "signal_hard_fail": quality.get("hard_fail"),
            "signal_hard_reasons": quality.get("hard_reasons", []),
            "signal_warnings": quality.get("warnings", []),
            "valid_channels": quality.get("valid_channels"),
            "frame_hard_invalid": timing.get("hard_invalid"),
            "frame_anomalies": timing.get("anomaly_count"),
            "estimated_missed_frames": timing.get("estimated_missed_frames"),
        })
    return {
        "schema_version": 1,
        "session_id": session["session_id"],
        "source_npz": session["session_id"] + ".npz",
        "source_npz_sha256": npz_sha256,
        "mode": "cued",
        "target_count": len(TARGETS),
        "labels": [target.symbol for target in TARGETS],
        "freqs_hz": [target.frequency_hz for target in TARGETS],
        "channel_indices": session["config"]["channel_indices"],
        "channel_positions": session["config"]["channel_positions"],
        "online_decoder": session["algorithm"],
        "decoder_settings": session["decoder_settings"],
        "online_summary": session["summary"],
        "model": {key: value for key, value in bundle.items()
                  if key in ("training_status", "training_reason", "model_id",
                             "participant_id", "counts_by_class", "missing_class_ids",
                             "used_for_online_decoding", "validation")},
        "presentation_qc": {
            "trial_count_with_frame_diagnostics": sum(bool(f) for f in frame),
            "median_refresh_hz": _finite_median([f.get("refresh_hz") for f in frame]),
            "total_anomalous_intervals": sum(int(f.get("anomaly_count") or 0) for f in frame),
            "total_severe_intervals": sum(int(f.get("severe_interval_count") or 0) for f in frame),
        },
        "signal_qc": {
            "method": "各有效试次Hann周期图的局部PSD比；先跨通道、再跨试次取中位数",
            "scope": "描述性频谱对比，不是识别概率或独立验证准确率",
            "frequency_resolution_hz": (next((t["spectral_quality"].get("frequency_resolution_hz")
                                             for t in trials if t.get("spectral_quality", {}).get("available")), None)),
            "targets": summaries,
        },
        "trial_qc": trial_qc,
    }


class SessionRecorder:
    """先逐试次落盘；模式2结束时将NPZ、JSON、PKL、CSV封装为一个ZIP。"""
    schema_version = 3

    def __init__(self, cfg: Config, *, root: Optional[str | Path] = None,
                 context: Optional[dict[str, Any]] = None):
        base = Path(root if root is not None else cfg.record_root)
        if not base.is_absolute():
            base = Path(__file__).resolve().parent / base
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.session_dir = base / f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        self.manifest_path = self.session_dir / "manifest.json"
        self.progress_path = self.session_dir / "session.json"
        self.session_path = base / (self.session_dir.name + ".npz")
        self.archive_path = base / (self.session_dir.name + ".zip")
        self._started = False
        self.cfg = cfg
        self.export_artifacts: dict[str, str] = {}
        self._records: list[TrialRecord] = []
        self.program = {
            "program_version": "dual-mode-auto-cued-5", "export_schema_version": 4,
            "record_schema_version": self.schema_version,
            "source_file": Path(__file__).name,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": sys.version, "numpy_version": np.__version__,
            "runtime_environment": environment_snapshot(),
        }
        self.manifest: dict[str, Any] = {
            "schema_version": self.schema_version,
            "session_dir": str(self.session_dir),
            "created_utc": timestamp,
            "context": _json_safe(context or {}),
            "trials": [],
        }

    def _start_on_first_trial(self) -> None:
        if self._started:
            return
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_json(self.manifest_path, self.manifest)
        self._started = True

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def write_trial(self, record: TrialRecord) -> dict[str, str]:
        self._start_on_first_trial()
        trial_stem = f"trial_{record.trial_id:04d}"
        metadata_path = self.session_dir / f"{trial_stem}.json"
        arrays_path = self.session_dir / f"{trial_stem}.npz"
        record.artifact_refs = {"metadata": str(metadata_path), "arrays": str(arrays_path)}
        result = record.result or {}
        epoch = result.get("epoch")
        arrays: dict[str, np.ndarray] = {
            "frame_flip_times": np.asarray(record.frame_flip_times, dtype=float),
        }
        frame_intervals = record.frame_diagnostics.get("psychopy_frame_intervals_s")
        if frame_intervals is not None:
            arrays["psychopy_frame_intervals_s"] = np.asarray(frame_intervals, dtype=float)
        if epoch is not None:
            epoch.validate_dimensions()
            arrays.update({
                "source_data": np.asarray(epoch.source_data, dtype=float),
                "source_lsl_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                # 旧字段仅为既有读取程序保留；新会话中并非发布端原始时间戳。
                "source_raw_timestamps": np.asarray(epoch.raw_timestamps, dtype=float),
                "source_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                "timestamp_adjustments": epoch.source_timestamps - epoch.raw_timestamps,
                "source_local_timestamps": np.asarray(epoch.local_timestamps, dtype=float),
                "clock_corrections": np.asarray(epoch.clock_corrections, dtype=float),
                "uniform_data": np.asarray(epoch.data, dtype=float),
                "uniform_timestamps": np.asarray(epoch.uniform_timestamps, dtype=float),
            })
        for key in ("scores", "correlations", "weights", "cca_scores", "fbcca_scores", "ecca_features"):
            if result.get(key) is not None:
                arrays[key] = np.asarray(result[key], dtype=float)
        self._atomic_npz(arrays_path, arrays)
        frame_diag = dict(record.frame_diagnostics)
        frame_diag.pop("intervals_s", None)
        trial_metadata = {
            "schema_version": self.schema_version,
            "trial_id": record.trial_id, "block_id": record.block_id,
            "mode": record.mode, "true_class": record.true_class,
            "true_symbol": (TARGETS[record.true_class - 1].symbol
                             if record.true_class is not None else None),
            "true_frequency_hz": (TARGETS[record.true_class - 1].frequency_hz
                                   if record.true_class is not None else None),
            "status": record.status, "reason": record.reason,
            "text_applied": record.text_applied,
            "start": record.start, "end": record.end,
            "duration_s": max(0.0, record.end - record.start),
            "cue_onset": record.cue_onset,
            "cue_progression": "automatic_timed" if record.mode == "cued" else None,
            "cue_confirmed_at": record.cue_confirmed_at, "fixation_onset": record.fixation_onset,
            "stimulus_onset": record.stimulus_onset,
            "stimulus_offset": record.stimulus_offset,
            "requested_window_start": record.requested_window_start,
            "requested_window_end": record.requested_window_end,
            "actual_stimulus_s": record.actual_stimulus_s,
            "phase_times": record.phase_times,
            "frame_diagnostics": frame_diag,
            "prediction": result.get("prediction"),
            "decoder": result.get("decoder", "FBCCA"),
            "calibration_id": result.get("calibration_id"),
            "calibration_fallback": result.get("calibration_fallback"),
            "fbcca_prediction": result.get("fbcca_prediction"),
            "frequency_hz": result.get("frequency_hz"),
            "filter_bands_hz": result.get("filter_bands_hz"),
            "n_harmonics": result.get("n_harmonics"),
            "line_regression_hz": result.get("line_regression_hz", self.cfg.line_regression_hz),
            "rejection_diagnostics": result.get("rejection_diagnostics"),
            "cca_prediction": result.get("cca_prediction"),
            "cca_frequency_hz": result.get("cca_frequency_hz"),
            "computation_s": result.get("computation_s"),
            "quality": result.get("quality"),
            "sampling_rate_hz": epoch.fs if epoch is not None else None,
            "epoch_diagnostics": epoch.diagnostics if epoch is not None else None,
            "artifact_refs": record.artifact_refs,
            "arrays_file": arrays_path.name,
            "array_keys": sorted(arrays),
        }
        self._atomic_json(metadata_path, trial_metadata)
        entry = {"trial_id": record.trial_id, "status": record.status,
                 "metadata_file": metadata_path.name, "arrays_file": arrays_path.name}
        self.manifest["trials"].append(entry)
        self._atomic_json(self.manifest_path, self.manifest)
        self._records.append(record)
        return dict(record.artifact_refs)

    @staticmethod
    def _npz_entry(archive: zipfile.ZipFile, name: str, value: Any) -> None:
        with archive.open(name + ".npy", "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)

    def _export_session(self, payload: dict[str, Any]) -> str:
        """逐试次合并原始EEG与元数据，生成可直接读取的NPZ。"""
        stem = self.session_dir.name
        if self.cfg.session_mode == "cued":
            self.session_path = self.session_dir / f"{stem}.npz"
        trials = []
        for entry in self.manifest["trials"]:
            with (self.session_dir / entry["metadata_file"]).open(encoding="utf-8") as handle:
                trials.append(json.load(handle))
        # 同一磁盘上先完成并核对归档，再原子替换正式文件。
        with tempfile.TemporaryDirectory(prefix=".export-", dir=self.session_dir) as folder:
            staging = Path(folder)
            raw_path = staging / f"{stem}.npz"
            with zipfile.ZipFile(raw_path, "w", compression=zipfile.ZIP_DEFLATED) as raw:
                has_eeg = []
                for trial in trials:
                    prefix = f"trial_{trial['trial_id']:04d}"
                    with np.load(self.session_dir / trial["arrays_file"], allow_pickle=False) as arrays:
                        has_eeg.append("source_data" in arrays and arrays["source_data"].shape[1] > 0)
                        for key in arrays.files:
                            self._npz_entry(raw, f"{prefix}_{key}", arrays[key])
                        true_class = trial["true_class"]
                        reference = true_class if true_class is not None else trial["prediction"]
                        spectral = spectral_quality(
                            arrays["uniform_data"] if "uniform_data" in arrays else np.empty((0, 0)),
                            trial["sampling_rate_hz"] or math.nan, reference, self.cfg.n_harmonics)
                        spectral["reference_kind"] = ("true_label" if true_class is not None else
                                                      "prediction" if reference is not None else "none")
                    trial["spectral_quality"] = spectral
                    trial["artifact_refs"] = {"session_file": self.session_path.name,
                                              "array_prefix": prefix + "_"}
                    trial.pop("arrays_file", None)
                metadata = {
                    **payload, "program": self.program, "session_id": stem,
                    "targets": _json_safe(TARGETS), "trials": trials,
                    "algorithm": "FB-eCCA" if self.cfg.calibration_file else "FBCCA",
                    "trained_model": self.manifest["context"].get("trained_model"),
                    "data_scope": "Recorded trial windows only; not a continuous session recording",
                    "array_layout": "channels x samples; each trial uses trial_NNNN_ prefixed keys",
                    "label_convention": "1-based true class; -1 means unknown (including free mode)",
                    "source_data_scope": "Selected LSL channels before receiver filtering; upstream processing may apply",
                }
                metadata.pop("export_status", None)
                metadata.pop("manifest_file", None)
                for key, value in {
                    "trial_ids": np.asarray([t["trial_id"] for t in trials], dtype=np.int64),
                    "labels": np.asarray([t["true_class"] if t["true_class"] is not None else -1
                                          for t in trials], dtype=np.int64),
                    "has_eeg": np.asarray(has_eeg, dtype=bool),
                    "channel_indices": np.asarray(self.cfg.channel_indices, dtype=np.int64),
                    "channel_positions": np.asarray(self.cfg.channel_positions, dtype=str),
                    "sampling_rates_hz": np.asarray([t["sampling_rate_hz"] or math.nan for t in trials]),
                    "target_fs_hz": self.cfg.target_fs,
                    "metadata_json": json.dumps(_json_safe(metadata), ensure_ascii=False, allow_nan=False),
                }.items():
                    self._npz_entry(raw, key, value)
            with zipfile.ZipFile(raw_path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError("会话NPZ校验失败，保留逐试次文件")
            with raw_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_path, self.session_path)
        self.export_artifacts["session"] = str(self.session_path)
        return str(self.session_path)

    def finalize(self, *, summary: dict[str, Any], events: Sequence[EventStamp],
                 acquisition_metadata: dict[str, Any], acquisition_review: dict[str, Any],
                 clock_updates: Sequence[Any], display_info: dict[str, Any],
                 typed_text: str, stop_reason: str) -> Optional[str]:
        if not self._started:
            return None
        payload = _json_safe({
            "schema_version": self.schema_version,
            "manifest_file": self.manifest_path.name,
            "config": vars(self.cfg), "summary": summary,
            "decoder_settings": {"filter_bank_profile": self.cfg.filter_bank_profile,
                                 "filter_bands_hz": self.cfg.filter_bands,
                                 "n_harmonics": self.cfg.n_harmonics,
                                 "line_regression_hz": self.cfg.line_regression_hz},
            "events": events, "acquisition_metadata": acquisition_metadata,
            "acquisition_review": acquisition_review, "clock_updates": clock_updates,
            "display_info": display_info, "typed_text": typed_text,
            "stop_reason": stop_reason,
            "program": self.program, "export_status": "pending",
        })
        self._atomic_json(self.progress_path, payload)
        self.manifest["session_file"] = self.progress_path.name
        self._atomic_json(self.manifest_path, self.manifest)
        try:
            self._export_session(payload)
            if self.cfg.session_mode == "cued":
                sidecars = export_cued_sidecars(self.session_path)
                package_cued_session(self.session_path, destination=self.archive_path)
                self.export_artifacts = {"session": str(self.archive_path)}
        except Exception as exc:
            payload["export_status"] = "failed"
            payload["export_error"] = f"{type(exc).__name__}: {exc}"
            self._atomic_json(self.progress_path, payload)
            self.manifest["export_status"] = "failed"
            self._atomic_json(self.manifest_path, self.manifest)
            raise
        for record in self._records:
            record.artifact_refs = (
                {"session_archive": str(self.archive_path),
                 "session_member": self.session_path.name,
                 "array_prefix": f"trial_{record.trial_id:04d}_"}
                if self.cfg.session_mode == "cued" else
                {"session_file": str(self.session_path),
                 "array_prefix": f"trial_{record.trial_id:04d}_"})
        if self.cfg.session_mode == "cued":
            for path in (self.session_path, *(Path(path) for path in sidecars.values())):
                path.unlink()
        for entry in self.manifest["trials"]:
            (self.session_dir / entry["metadata_file"]).unlink()
            (self.session_dir / entry["arrays_file"]).unlink()
        self.progress_path.unlink()
        self.manifest_path.unlink()
        self.session_dir.rmdir()
        return str(self.archive_path if self.cfg.session_mode == "cued" else self.session_path)


def export_cued_sidecars(session_path: str | Path) -> dict[str, str]:
    """为模式2会话NPZ导出质控JSON、可加载模型PKL和事件CSV。

    可对旧版单文件会话NPZ再次运行；原NPZ保持原位且不会被修改。
    """
    session_path = Path(session_path)
    with np.load(session_path, allow_pickle=False) as archive:
        session = json.loads(archive["metadata_json"].item())
        if session.get("config", {}).get("session_mode") != "cued":
            raise ValueError("只有模式2提示测试会话可生成质控和个人模型文件")
        cfg = Config(**session["config"])
        bundle = _cued_model_bundle(archive, session, cfg)
    hasher = hashlib.sha256()
    with session_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    bundle["source_npz_sha256"] = digest
    qc = _cued_quality_report(session, bundle, digest)
    stem = session_path.stem
    qc_path = session_path.with_name(stem + "_qc.json")
    model_path = session_path.with_name(stem + "_model.pkl")
    events_path = session_path.with_name(stem + "_events.csv")
    SessionRecorder._atomic_json(qc_path, qc)

    temp = model_path.with_name(model_path.name + f".tmp-{uuid.uuid4().hex}")
    with temp.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, model_path)

    trial_by_id = {t["trial_id"]: t for t in session["trials"]}
    fieldnames = ("timestamp_lsl", "event", "trial_id", "block_id", "target_id",
                  "target_symbol", "target_frequency_hz", "status", "prediction",
                  "prediction_symbol", "cca_prediction", "details_json")
    temp = events_path.with_name(events_path.name + f".tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in session["events"]:
            trial = trial_by_id.get(event["trial_id"], {})
            target_id = event.get("target_id")
            prediction = trial.get("prediction")
            writer.writerow({
                "timestamp_lsl": event.get("timestamp"),
                "event": event["kind"],
                "trial_id": event["trial_id"],
                "block_id": trial.get("block_id"),
                "target_id": target_id,
                "target_symbol": TARGETS[target_id - 1].symbol if target_id else None,
                "target_frequency_hz": TARGETS[target_id - 1].frequency_hz if target_id else None,
                "status": trial.get("status"),
                "prediction": prediction,
                "prediction_symbol": TARGETS[prediction - 1].symbol if prediction else None,
                "cca_prediction": trial.get("cca_prediction"),
                "details_json": json.dumps(event.get("details") or {}, ensure_ascii=False,
                                           separators=(",", ":")),
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, events_path)
    return {"qc": str(qc_path), "model": str(model_path), "events": str(events_path)}


def package_cued_session(session_path: str | Path, *,
                         destination: Optional[str | Path] = None,
                         remove_sources: bool = False) -> str:
    """将一场模式2会话的四个文件封装为ZIP，逐项核对后才可移除原文件。"""
    session_path = Path(session_path)
    stem = session_path.stem
    sources = (session_path,
               session_path.with_name(stem + "_qc.json"),
               session_path.with_name(stem + "_model.pkl"),
               session_path.with_name(stem + "_events.csv"))
    archive_path = Path(destination) if destination is not None else session_path.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(f"会话ZIP已存在，不覆盖：{archive_path}")
    if not all(path.is_file() for path in sources):
        missing = [str(path) for path in sources if not path.is_file()]
        raise FileNotFoundError("会话文件不齐，无法封装：" + ", ".join(missing))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp = archive_path.with_name(archive_path.name + f".tmp-{uuid.uuid4().hex}")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in sources:
                archive.write(path, arcname=path.name)
        with zipfile.ZipFile(temp) as archive:
            if archive.namelist() != [path.name for path in sources] or archive.testzip() is not None:
                raise RuntimeError("会话ZIP目录或CRC校验失败")
            for path in sources:
                original_hash = hashlib.sha256()
                archived_hash = hashlib.sha256()
                with path.open("rb") as source, archive.open(path.name) as member:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        original_hash.update(chunk)
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        archived_hash.update(chunk)
                if original_hash.digest() != archived_hash.digest():
                    raise RuntimeError(f"会话ZIP内容核对失败：{path.name}")
        with temp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, archive_path)
    finally:
        temp.unlink(missing_ok=True)
    if remove_sources:
        for path in sources:
            path.unlink()
    return str(archive_path)


def frame_diagnostics(flip_times: Sequence[float], refresh_hz: float,
                      long_factor: float = 1.5, short_factor: float = .5,
                      max_warning_anomalies: int = 1,
                      rate_tolerance: float = .005) -> dict:
    """Check software flip timing against the refresh rate used for frame-indexed stimuli."""
    ts = np.asarray(flip_times, float)
    if (ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all()
            or not np.isfinite(refresh_hz) or refresh_hz <= 0):
        raise InvalidTrial("刺激帧时间记录不足或非法")
    if not np.isfinite(rate_tolerance) or not 0 < rate_tolerance < 1:
        raise ValueError("rate_tolerance 必须是(0, 1)内的有限比例")
    intervals = np.diff(ts)
    period = 1 / refresh_hz
    # 每个表项占一次翻转；末尾时间戳是停止闪烁的翻转，因此也要计入
    # 最后一个表项的显示时间。用整段(n-1)个间隔核对生成亮度表的时间基准。
    elapsed = float(ts[-1] - ts[0])
    observed_rate = float(len(intervals) / elapsed) if elapsed > 0 else None
    rate_error = ((observed_rate / refresh_hz - 1) if observed_rate is not None else None)
    rate_mismatch = rate_error is None or abs(rate_error) > rate_tolerance
    long = intervals > long_factor * period
    short = intervals < short_factor * period
    # A warning budget counts mild timing deviations, never long stalls or
    # reversed/duplicate frame timestamps. One 66.7 ms interval at 60 Hz
    # already loses about three refreshes even though it is only one anomaly.
    nonmonotonic = intervals <= 0
    severe = (intervals >= 2 * period) | nonmonotonic
    missed = np.maximum(1, np.rint(intervals[long] / period).astype(int) - 1).sum() if np.any(long) else 0
    anomaly_count = int(long.sum() + short.sum())
    severe_count = int(severe.sum())
    hard_invalid = (severe_count > 0 or anomaly_count > max_warning_anomalies
                    or rate_mismatch)
    return {"intervals_s": intervals, "long_intervals": int(long.sum()), "short_intervals": int(short.sum()),
            "estimated_missed_frames": int(missed), "min_interval_ms": float(intervals.min() * 1000),
            "median_interval_ms": float(np.median(intervals) * 1000),
            "max_interval_ms": float(intervals.max() * 1000),
            "anomaly_count": anomaly_count,
            "severe_interval_count": severe_count,
            "nonmonotonic_interval_count": int(nonmonotonic.sum()),
            "severe_interval_threshold_ms": float(2 * period * 1000),
            "observed_flip_rate_hz": observed_rate,
            "flip_rate_error_percent": None if rate_error is None else float(rate_error * 100),
            "flip_rate_tolerance_percent": float(rate_tolerance * 100),
            "flip_rate_mismatch": rate_mismatch,
            "valid": not hard_invalid,
            "strict_valid": anomaly_count == 0 and severe_count == 0 and not rate_mismatch,
            "warning": anomaly_count > 0 and not hard_invalid,
            "hard_invalid": hard_invalid,
            "refresh_hz": refresh_hz}


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
    # 向上取整并留一帧余量；运行时仍以实际翻转时间检查分析窗边界。
    frames = max(1, math.ceil(duration_s * refresh_hz) + 1)
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
    """三种准确率只用有效试次；回退总数计入所有留有分类结果的试次。"""
    if any(r.mode != "cued" or r.true_class is None for r in records):
        raise ValueError("自由输入没有真实标签，不可计算提示测试准确率")
    attempted = len(records)
    valid = [r for r in records if r.status == "valid"]
    n = len(TARGETS)
    counts, valid_counts, invalid_counts = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    fb_cm = np.zeros((n, n), int)
    cca_cm = np.zeros((n, n), int)
    online_cm = np.zeros((n, n), int)
    calibration_fallback_count = 0
    calibration_fallback_valid_count = 0
    online_decoder_counts: dict[str, int] = {}
    for record in records:
        t = record.true_class - 1
        counts[t] += 1
        if record.result is not None and record.result.get("calibration_fallback"):
            calibration_fallback_count += 1
            if record.status == "valid":
                calibration_fallback_valid_count += 1
        if record.status == "valid":
            if record.result is None:
                raise ValueError("有效试次缺失结果")
            valid_counts[t] += 1
            fb_cm[t, record.result["fbcca_prediction"] - 1] += 1
            cca_cm[t, record.result["cca_prediction"] - 1] += 1
            online_cm[t, record.result["prediction"] - 1] += 1
            decoder = record.result["decoder"]
            online_decoder_counts[decoder] = online_decoder_counts.get(decoder, 0) + 1
        else:
            invalid_counts[t] += 1
    active_s = sum(max(0.0, r.end - r.start) for r in records)
    wall_s = max((r.end for r in records), default=0) - min((r.start for r in records), default=0)
    started_targets = {(r.block_id, r.true_class) for r in records}
    completed_targets = {(r.block_id, r.true_class) for r in valid}
    report = {"planned": planned, "attempted": attempted,
        "not_started": max(0, planned - len(started_targets)),
        "completed_targets": len(completed_targets),
        "valid": len(valid), "rejected": sum(r.status == "rejected" for r in records),
        "invalid_or_aborted": sum(r.status in ("invalid", "aborted") for r in records),
        "active_selection_s": active_s, "wall_s_including_breaks": max(0, wall_s),
        "mean_actual_selection_s": active_s / attempted if attempted else None,
        "target_attempts": counts, "target_valid": valid_counts, "target_invalid": invalid_counts,
        "fbcca_confusion": fb_cm, "cca_confusion": cca_cm,
        "online_confusion": online_cm,
        "online_decoder_counts": online_decoder_counts,
        "calibration_fallback_count": calibration_fallback_count,
        "calibration_fallback_valid_count": calibration_fallback_valid_count}
    for name, cm in (("fbcca", fb_cm), ("cca", cca_cm), ("online", online_cm)):
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


def _brief_reason(reason: str, limit: int = 160) -> str:
    message = " ".join(reason.split())
    return message if len(message) <= limit else message[:limit - 1] + "…"


def print_summary(report: dict, ledger: TrialLedger) -> None:
    print(f"已完成目标{report['completed_targets']} | 有效试次{report['valid']} | "
          f"基础FBCCA准确率 {_pct(report['fbcca']['accuracy_valid'])} | "
          f"标准CCA准确率 {_pct(report['cca']['accuracy_valid'])} | "
          f"实际在线算法准确率 {_pct(report['online']['accuracy_valid'])} | "
          f"个人模型回退{report['calibration_fallback_count']}次"
          f"（其中有效试次{report['calibration_fallback_valid_count']}次）")


def summarize_free_trials(records: Sequence[TrialRecord], ledger: TrialLedger) -> dict:
    """自由输入只报告操作记录；不虚构正确率、混淆矩阵或ITR。"""
    if any(r.mode != "free" or r.true_class is not None for r in records):
        raise ValueError("自由输入记录不应含提示目标")
    return {
        "mode": "free", "attempted": len(records),
        "completed": sum(r.status != "aborted" for r in records),
        "valid": sum(r.status == "valid" for r in records),
        "input_actions": sum(r.text_applied for r in records),
        "rejected": sum(r.status == "rejected" for r in records),
        "invalid_or_aborted": sum(r.status in ("invalid", "aborted") for r in records),
        "typed_text": ledger.typed_text,
        "ground_truth_available": False, "accuracy": None, "itr_bits_per_minute": None,
    }


def print_free_summary(report: dict) -> None:
    print(f"已完成{report['completed']}轮 | 有效{report['valid']}轮 | "
          "准确率 N/A（自由输入无真实目标）")


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


def preflight_signal_quality(cfg: Config, collector: ContinuousLSL) -> dict[str, Any]:
    """Describe up to 5 s of recent EEG; never add a startup/rejection gate.

    Use the same timestamp validation as trial epochs. Missing data or an
    invalid time axis yields an unavailable diagnostic, not guessed values.
    This is relative spectral power; unknown physical units remain unknown.
    """
    report: dict[str, Any] = {
        "available": False,
        "input_scope": "recent uniform EEG before receiver notch/bandpass; no flicker",
        "minimum_window_s": 1.0, "maximum_window_s": 5.0,
        "used_for_rejection": False,
        "physical_units_inferred": False,
    }
    try:
        if collector.buffer is None:
            raise InvalidTrial("EEG缓冲区尚未建立")
        fs = float(collector.fs)
        _validate_fs(fs, None)
        _, _, local, _, _ = collector.buffer.snapshot()
        if len(local) < 2 or not np.isfinite(local).all():
            raise InvalidTrial("启动EEG数据不足")
        duration = math.floor(min(5.0, float(local[-1] - local[0])) * fs) / fs
        if duration < 1.0:
            raise InvalidTrial("启动EEG尚未覆盖至少1秒")
        end = float(local[-1])
        start = end - duration
        # Subtracting large LSL clock values can round one endpoint outwards.
        if start < local[0]:
            duration -= 1 / fs
            start = end - duration
        if duration < 1.0:
            raise InvalidTrial("启动EEG尚未覆盖至少1秒")
        epoch = collector.buffer.epoch(
            start, duration, fs, cfg.max_gap_factor, cfg.rate_tolerance,
            cfg.hard_gap_factor, cfg.hard_rate_tolerance, cfg.max_warning_gaps)
        mains = mains_noise_diagnostics(epoch.data, epoch.fs, cfg.notch_hz)
        report.update({
            "available": bool(mains["available"]),
            "window_start": start, "window_end": end, "window_s": duration,
            "uniform_samples": int(epoch.data.shape[1]),
            "timestamp_mode": epoch.diagnostics.get("timestamp_mode"),
            "timestamp_warnings": list(epoch.diagnostics.get("warnings", [])),
            "mains_noise": mains,
            "channels": [
                {"position": position, "lsl_index": int(lsl_index),
                 "line_fraction": fraction}
                for position, lsl_index, fraction in zip(
                    cfg.channel_positions, cfg.channel_indices,
                    mains["line_fraction_per_channel"])],
        })
        if not report["available"]:
            report["unavailable_reason"] = mains.get("unavailable_reason", "频谱诊断不可用")
    except Exception as exc:
        report["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return report


def _preflight_quality_summary(report: dict[str, Any]) -> str:
    if not report.get("available"):
        return "Unavailable: " + str(report.get("unavailable_reason", "insufficient EEG"))
    mains = report["mains_noise"]
    low, high = mains["line_band_hz"]
    reference_low, reference_high = mains["reference_band_hz"]
    return (f"{report['window_s']:.2f} s EEG; {low:g}-{high:g} / "
            f"{reference_low:g}-{reference_high:g} Hz median "
            f"{mains['median_line_fraction']:.1%}; relative power only")


def preflight_review(cfg: Config, collector: ContinuousLSL) -> dict:
    """启动核验。quick_entry_mode下跳过人工勾选，优先进入程序。"""
    meta = collector.metadata
    band_high = max(high for _, high in cfg.filter_bands)
    startup_quality = preflight_signal_quality(cfg, collector)
    startup_summary = _preflight_quality_summary(startup_quality)
    if not startup_quality["available"]:
        print("启动EEG工频检查不可用：" + str(startup_quality["unavailable_reason"]), flush=True)
    elif startup_quality["mains_noise"]["line_dominated"]:
        print("启动EEG工频占比偏高，请检查电极和采集环境。", flush=True)
    if cfg.quick_entry_mode:
        print("\n=== 快速调试模式：跳过严格EEG启动核验对话框 ===", flush=True)
        print(f"流: {meta.get('name')} | 总通道: {meta.get('channel_count')} | "
              f"声明采样率: {collector.fs:g} Hz | 当前时间轴估计: {collector.estimated_fs:.3f} Hz", flush=True)
        print("当前目标是先进入40目标键盘并验证完整链路；最终实验前再恢复严格核验。", flush=True)
        return {
            "quick_entry_mode": True,
            "startup_signal_quality": startup_quality,
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
    print(f"源时间轴模式: {collector.timestamp_mode}；LSL内建同步/去抖/单调化，"
          "不额外应用time_correction；仅减去已独立校准的设备滞后(默认未应用)。")
    print(f"信号质量门控：至少{cfg.min_valid_channels}/{len(cfg.channel_indices)}通道有效；"
          f"幅值阈值={cfg.quality_max_abs}; 设备轨值={cfg.quality_rail_min}/{cfg.quality_rail_max}。")
    print(f"未知单位/滤波不会被猜测为已确认。当前{cfg.filter_bank_profile}频带上限{band_high:g} Hz；"
          f"已知上游低通低于{band_high:g} Hz时拒绝当前频带配置。")
    choices = list(dict.fromkeys([cfg.input_unit, "AUTO", "uV", "V", "mV", "UNKNOWN"]))
    fields = {
        "Stream / nominal / observed Hz": f"{meta['name']} / {collector.fs:g} / {collector.estimated_fs:.3f}",
        "Recent EEG line power (relative only)": startup_summary,
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
        fixed=["Stream / nominal / observed Hz", "Channel map (see terminal)",
               "Recent EEG line power (relative only)"])
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
        unknowns.append(f"发布端滤波/{band_high:g} Hz有效带宽尚未完整核验")
    unknowns.extend(["屏幕真实发光与软件事件延迟未由光电二极管测量",
                     "OpenBCI设备采样到LSL时间戳的固定延迟未由本程序自动测量"])
    review = {"input_units": units, "unit_override": cfg.input_unit,
        "startup_signal_quality": startup_quality,
        "upstream_filter_description": cfg.upstream_filter_description, "upstream_lowpass_hz": cfg.upstream_lowpass_hz,
        "receiver_notch_hz": cfg.notch_hz, "wiring_user_reviewed": True, "unverified": unknowns,
        "device_timestamp_lag_applied_s": cfg.device_timestamp_lag_s}
    print("最终输入单位（分析通道顺序）：", units)
    print(f"人工声明发布端滤波: {cfg.upstream_filter_description}; 低通: {cfg.upstream_lowpass_hz}; 接收端陷波: {cfg.notch_hz}")
    print("未核验边界：\n  " + "\n  ".join(unknowns))
    return review


# ======================== 主线程：预创建刺激、逐帧更新、试次管理 ========================
def _keyboard_side_margin(width: float) -> float:
    """输出框和键盘共用左右边界，并给最外侧提示框留出空间。"""
    return max(24.0, min(64.0, width * .02))


def _display_cm_per_pixel(cfg: Config, window_size: Sequence[float],
                          display_size: Optional[Sequence[float]] = None) -> Optional[float]:
    """对角线来自人工填写；窗口模式需提供整屏的 PsychoPy pix 坐标尺寸。"""
    if cfg.screen_diagonal_inches is None:
        return None
    if display_size is None:
        if not cfg.full_screen:
            raise ValueError("窗口模式的物理尺寸换算需要整块显示器的像素尺寸")
        display_size = window_size
    if len(display_size) != 2 or not all(math.isfinite(v) and v > 0 for v in display_size):
        raise ValueError("显示器宽高必须为有限正数")
    return cfg.screen_diagonal_inches * 2.54 / math.hypot(*display_size)


def calculate_keyboard_layout(cfg: Config, window_size: Sequence[float],
                              display_size: Optional[Sequence[float]] = None
                              ) -> tuple[np.ndarray, float, tuple[float, float], float]:
    """居中网格，保留所设间距；只在放不下时等比例缩小，不拉伸填屏。"""
    cfg.validate()
    width, height = map(float, window_size)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("窗口宽高必须为有限正数")
    rows, cols = len(KEY_ROWS), max(map(len, KEY_ROWS))
    top_reserved = cfg.output_box_height_px + cfg.output_box_margin_px + 70.0
    bottom_reserved = 70.0
    available_width = width - 2 * _keyboard_side_margin(width)
    available_height = height - top_reserved - bottom_reserved
    key, gap = cfg.key_size_px, cfg.key_gap_px
    if cfg.layout_units == "degrees":
        cm_per_pixel = _display_cm_per_pixel(cfg, window_size, display_size)
        if cm_per_pixel is None:
            raise ValueError("视角模式需要填写当前显示器的实际对角线英寸数")
        key = 2 * cfg.viewing_distance_cm * math.tan(math.radians(cfg.key_size_deg) / 2) / cm_per_pixel
        gap = 2 * cfg.viewing_distance_cm * math.tan(math.radians(cfg.key_gap_deg) / 2) / cm_per_pixel
    scale = min(1.0, available_width / (cols * key + (cols - 1) * gap),
                available_height / (rows * key + (rows - 1) * gap))
    if scale <= 0 or (scale < 1 and not cfg.allow_layout_scaling):
        raise RuntimeError("屏幕不足以容纳固定按键布局；请更换分辨率或允许等比例缩放")
    key, gap = key * scale, gap * scale
    # 能容纳时以屏幕中心为注视原点；顶部输出框挤占空间时向下移动最小距离。
    grid_height = rows * key + (rows - 1) * gap
    center_y = min(0.0, height / 2 - top_reserved - grid_height / 2)
    positions = np.asarray([
        ((target.col - (cols - 1) / 2) * (key + gap),
         center_y + ((rows - 1) / 2 - target.row) * (key + gap))
        for target in TARGETS
    ])
    return positions, key, (gap, gap), scale


def keyboard_layout_info(cfg: Config, window_size: Sequence[float],
                         display_size: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """保存请求值、实际几何及估算视角；不把主观转头体验当作测量结果。"""
    positions, key, (gap_x, gap_y), scale = calculate_keyboard_layout(cfg, window_size, display_size)
    cm_per_pixel = _display_cm_per_pixel(cfg, window_size, display_size)
    corners = [target for target in TARGETS if target.row in (0, len(KEY_ROWS) - 1)
               and target.col in (0, len(KEY_ROWS[0]) - 1)]
    physical = None
    if cm_per_pixel is not None:
        def angle(length: float) -> float:
            return math.degrees(2 * math.atan(length * cm_per_pixel / (2 * cfg.viewing_distance_cm)))
        physical = {
            "basis": "declared diagonal and distance; square pixels; eye opposite screen center",
            "cm_per_pix_unit": cm_per_pixel,
            "key_size_cm": key * cm_per_pixel, "gap_cm": gap_x * cm_per_pixel,
            "key_centered_visual_angle_deg": angle(key), "gap_centered_visual_angle_deg": angle(gap_x),
            "grid_width_cm": (10 * key + 9 * gap_x) * cm_per_pixel,
            "grid_height_cm": (4 * key + 3 * gap_y) * cm_per_pixel,
            "corners": [{"class_id": t.class_id, "symbol": t.symbol,
                "center_eccentricity_deg": math.degrees(math.atan(
                    math.hypot(*positions[t.class_id - 1]) * cm_per_pixel / cfg.viewing_distance_cm)),
                "outer_edge_eccentricity_deg": math.degrees(math.atan(
                    math.hypot(*(np.abs(positions[t.class_id - 1]) + key / 2))
                    * cm_per_pixel / cfg.viewing_distance_cm))} for t in corners],
        }
    return {
        "layout_strategy": "centered_fixed_gap", "layout_schema_version": 2,
        "coordinate_space": "psychopy_pix_client",
        "requested_layout": {name: getattr(cfg, name) for name in (
            "layout_units", "key_size_px", "key_gap_px", "key_size_deg", "key_gap_deg",
            "viewing_distance_cm", "screen_diagonal_inches", "allow_layout_scaling")},
        "key_size_pix_units": key, "gap_pix_units": gap_x,
        "column_gap_pix_units": gap_x, "row_gap_pix_units": gap_y,
        "layout_scale": scale, "fit_reduced": scale < 1 - 1e-9,
        "key_positions_pix_units": positions.tolist(),
        "layout_rows": len(KEY_ROWS), "layout_cols": max(map(len, KEY_ROWS)),
        "target_viewing_distance_cm": cfg.viewing_distance_cm,
        "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
        "display_size_pix_units": list(display_size if display_size is not None else window_size),
        "physical_estimate": physical,
    }


def _create_keyboard_window(visual: Any, cfg: Config) -> Any:
    """Windows/macOS 使用目标屏幕尺寸全屏启动，布局使用 PsychoPy pix 坐标。"""
    size = cfg.window_size
    import pyglet
    screens = pyglet.canvas.get_display().get_screens()
    if not 0 <= cfg.screen_index < len(screens):
        raise ValueError(f"显示器编号 {cfg.screen_index} 不可用；检测到 {len(screens)} 个显示器")
    screen = screens[cfg.screen_index]
    if cfg.full_screen:
        # pyglet 屏幕尺寸与 PsychoPy 的客户区同属逻辑坐标。
        size = (screen.width, screen.height)
    win = visual.Window(size=size, fullscr=cfg.full_screen, screen=cfg.screen_index,
        winType="pyglet", useRetina=True,
        units="pix", color=(-.78, -.77, -.73), colorSpace="rgb", waitBlanking=True,
        allowGUI=not cfg.full_screen, checkTiming=False, autoLog=False)
    # PsychoPy 的 pix 刺激按客户区坐标定位。Retina 上 win.size 是双倍的
    # framebuffer 尺寸；用它排版会把网格和顶部文字推出可见范围。
    # 视角换算也必须采用相同坐标系中的整屏宽高。
    win.keyboard_display_size = np.asarray((screen.width, screen.height), dtype=float)
    return win


def _keyboard_coordinate_size(win: Any) -> np.ndarray:
    """返回与 PsychoPy pix 刺激位置/尺寸相同的客户区坐标范围。"""
    size = np.asarray(win.clientSize, dtype=float)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise RuntimeError(f"无效的窗口客户区尺寸：{size}")
    return size


class KeyboardLayoutPreview:
    """无EEG、无闪烁的实际尺寸预览；只在确认后交给实验使用。"""
    corner_ids = (1, 10, 31, 40)
    rating_names = ("easy", "hard_to_find", "head_turn", "hard_to_find_and_head_turn")

    def __init__(self, cfg: Config, win: Any, visual: Any):
        self.cfg, self.win, self.visual = cfg, win, visual
        self.field_index = 0
        self.corner_index = 0
        self.ratings: dict[int, str] = {}
        self.accepted = False
        self.info: dict[str, Any] = {}
        self.error = ""
        width, height = _keyboard_coordinate_size(win)
        self.title = visual.TextStim(win, pos=(0, height / 2 - 22), height=22,
            wrapWidth=width - 40, color="white", autoLog=False)
        self.settings = visual.TextStim(win, pos=(0, height / 2 - 56), height=min(20, width / 60),
            wrapWidth=width - 40, color="#FFE45C", autoLog=False)
        self.summary = visual.TextStim(win, pos=(0, height / 2 - 92), height=min(17, width / 65),
            wrapWidth=width - 40, color="white", autoLog=False)
        self.guide = visual.TextStim(win, pos=(0, height / 2 - 132), height=min(17, width / 65),
            wrapWidth=width - 40, color="white", autoLog=False)
        self.footer = visual.TextStim(win, pos=(0, -height / 2 + 31), height=min(17, width / 66),
            wrapWidth=width - 30, color="white", autoLog=False)
        self.cross = visual.TextStim(win, text="+", pos=(0, 0), height=18,
            color="#FFE45C", autoLog=False)
        self.refresh()

    def refresh(self) -> None:
        self.error = ""
        try:
            self.info = keyboard_layout_info(self.cfg, _keyboard_coordinate_size(self.win),
                                             self.win.keyboard_display_size)
        except (ValueError, RuntimeError) as exc:
            # 字段操作保持可用；无效参数不渲染旧网格、不允许进入实验。
            self.info = {}
            self.error = str(exc)
            return
        key = self.info["key_size_pix_units"]
        positions = self.info["key_positions_pix_units"]
        border = 4 * min(1.25, key / 140)
        self.borders = self.visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=positions, sizes=(key + border, key + border), elementTex=None, elementMask=None,
            colors=np.tile((-.88, -.89, -.84), (len(TARGETS), 1)), colorSpace="rgb", autoLog=False)
        self.squares = self.visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=positions, sizes=(key, key), elementTex=None, elementMask=None,
            colors=np.tile((.72, .58, .82), (len(TARGETS), 1)), colorSpace="rgb", autoLog=False)
        self.labels = [self.visual.TextStim(self.win, text=t.display_label, pos=pos,
            height=key * (.32 if len(t.display_label) > 1 else .42), units="pix",
            color=(-.64, -.67, -.58), colorSpace="rgb", autoLog=False)
            for t, pos in zip(TARGETS, positions)]
        self.outline = self.visual.Rect(self.win, width=max(1, key - 4), height=max(1, key - 4),
            fillColor=None, lineColor="#FFE45C", lineWidth=3, units="pix", autoLog=False)

    def handle_keys(self, keys: Sequence[str]) -> bool:
        """返回是否确认；参数变更会清空四角评价，防止保存旧布局的主观结果。"""
        if "escape" in keys:
            raise AbortSession("用户退出静态布局预览")
        changed = False
        if "up" in keys:
            self.field_index = (self.field_index - 1) % 4
        if "down" in keys:
            self.field_index = (self.field_index + 1) % 4
        if "u" in keys:
            self.cfg.layout_units = "degrees" if self.cfg.layout_units == "pixels" else "pixels"
            changed = True
        degree_mode = self.cfg.layout_units == "degrees"
        field_names = ("key_size_deg" if degree_mode else "key_size_px",
                       "key_gap_deg" if degree_mode else "key_gap_px",
                       "viewing_distance_cm", "screen_diagonal_inches")
        if "left" in keys or "right" in keys:
            name = field_names[self.field_index]
            step = (0.1 if degree_mode else 5.0, 0.1 if degree_mode else 5.0, 1.0, 0.1)[self.field_index]
            value = getattr(self.cfg, name)
            # 未填写的尺寸从24英寸开始编辑，但必须根据当前显示器实测/标称值修正。
            value = 24.0 if value is None else value + step * (1 if "right" in keys else -1)
            minimum = 0.0 if self.field_index == 1 else step
            maximum = 89.0 if degree_mode and self.field_index < 2 else 10000.0
            setattr(self.cfg, name, round(min(maximum, max(minimum, value)), 4))
            changed = True
        if "delete" in keys and self.field_index == 3:
            self.cfg.screen_diagonal_inches = None
            changed = True
        if changed:
            self.accepted = False
            self.ratings.clear()
            self.corner_index = 0
            self.refresh()
        if "tab" in keys:
            self.corner_index = (self.corner_index + 1) % len(self.corner_ids)
        if self.info and not changed:
            for i, rating in enumerate(self.rating_names, 1):
                if str(i) in keys or f"num_{i}" in keys:
                    self.ratings[self.corner_ids[self.corner_index]] = rating
                    self.corner_index = (self.corner_index + 1) % len(self.corner_ids)
                    break
        self.accepted = bool(not changed and self.info
                             and any(k in keys for k in ("return", "num_enter")))
        return self.accepted

    def draw(self) -> None:
        cfg = self.cfg
        degrees = cfg.layout_units == "degrees"
        unit = "deg" if degrees else "px"
        size = cfg.key_size_deg if degrees else cfg.key_size_px
        gap = cfg.key_gap_deg if degrees else cfg.key_gap_px
        diagonal = "unknown" if cfg.screen_diagonal_inches is None else f"{cfg.screen_diagonal_inches:g} in"
        fields = [f"Key: {size:g} {unit}", f"Gap: {gap:g} {unit}",
                  f"Distance: {cfg.viewing_distance_cm:g} cm", f"Screen: {diagonal}"]
        fields[self.field_index] = "[ " + fields[self.field_index] + " ]"
        self.title.text = "STATIC LAYOUT PREVIEW | No flicker"
        self.settings.text = "   |   ".join(fields)
        self.footer.text = ("UP/DOWN: field   LEFT/RIGHT: adjust   U: px/deg   DEL: clear screen size\n"
                            "TAB: next corner   1: easy   2: hard to find   3: head turn   4: both   ENTER: accept   ESC: exit")
        if self.info:
            self.borders.draw()
            self.squares.draw()
            for label in self.labels:
                label.draw()
            corner_id = self.corner_ids[self.corner_index]
            self.outline.pos = self.info["key_positions_pix_units"][corner_id - 1]
            self.outline.draw()
            self.cross.draw()
            physical = self.info["physical_estimate"]
            actual = f"Actual: key {self.info['key_size_pix_units']:.1f} px / gap {self.info['gap_pix_units']:.1f} px"
            fit = f" | FIT REDUCED to {self.info['layout_scale']:.0%}" if self.info["fit_reduced"] else ""
            if physical:
                corner = physical["corners"][self.corner_index]
                self.summary.text = (actual + fit + f" | key ~{physical['key_centered_visual_angle_deg']:.2f} deg\n"
                    f"Estimated corner center {corner['center_eccentricity_deg']:.1f} deg from screen center; "
                    "screen inches and eye distance are declared, not measured.")
            else:
                self.summary.text = actual + fit + "\nSet actual screen diagonal to estimate viewing angles."
            rating = self.ratings.get(corner_id, "unrated").replace("_", " ")
            self.guide.text = (f"From center +, find {TARGETS[corner_id - 1].symbol} at the yellow corner. "
                               f"Rate 1-4: {rating} | Checked {len(self.ratings)}/4")
        else:
            # 与实验窗口保持相同的英文界面，避免字体缺少中文字形。
            self.summary.text = "Layout unavailable. For degrees, set actual screen inches; otherwise reduce key/gap."
            self.guide.text = "Adjust settings to restore preview. ENTER is disabled."
        for text in (self.title, self.settings, self.summary, self.guide, self.footer):
            text.draw()

    def result(self) -> dict[str, Any]:
        return {**self.info, "window_size_reported": list(map(int, self.win.size)),
                "client_size_reported": list(map(int, self.win.clientSize)),
                "framebuffer_size_reported": list(map(int, self.win.frameBufferSize)),
                "content_scale_factor": self.win.getContentScaleFactor(),
                "full_screen": bool(self.win.fullscr), "screen_index": self.cfg.screen_index,
                "static_preview": {"confirmed": self.accepted,
                    "corners_checked": len(self.ratings),
                    "flicker_presented": False, "rating_source": "participant_keyboard_self_report",
                    "corner_ratings": [{"class_id": i, "symbol": TARGETS[i - 1].symbol,
                                        "rating": self.ratings.get(i)} for i in self.corner_ids]}}


def preview_keyboard_layout(cfg: Config) -> dict[str, Any]:
    from psychopy import visual, event
    win = _create_keyboard_window(visual, cfg)
    try:
        win.mouseVisible = False
        preview = KeyboardLayoutPreview(cfg, win, visual)
        event.clearEvents()
        while True:
            if preview.handle_keys(event.getKeys()):
                return preview.result()
            preview.draw()
            win.flip()
    finally:
        win.close()


class PsychoPyKeyboard:
    def __init__(self, cfg: Config, collector: ContinuousLSL, markers: Optional[EventMarkers],
                 ledger: TrialLedger, recorder: Optional[SessionRecorder] = None,
                 layout_preview: Optional[dict[str, Any]] = None, *, static_only: bool = False):
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.recorder = recorder
        self.layout_preview = layout_preview or {}
        self.static_only = static_only
        self.fatal_fault: Optional[dict] = None
        self.trial_faults: list[dict] = []
        self.visual, self.event = visual, event
        self.clock = collector.clock
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FBCCA-decoder")
        self.win: Any = None
        self.display_info: dict = {}
        self._pause_enabled = False
        self._rest_beep: Any = None
        self._active_decode_cancel: Optional[threading.Event] = None
        self._decode_future: Any = None
        self._output_cache: Optional[str] = None
        # 禁止刺激属性自动日志及重复掉帧警告占用关键时段；自己在刺激后汇报。
        logging.console.setLevel(logging.ERROR)
        try:
            self._build_window()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the original construction failure.
            raise

    def _build_window(self) -> None:
        visual, cfg = self.visual, self.cfg
        self.win = _create_keyboard_window(visual, cfg)
        self.win.mouseVisible = False
        width, height = _keyboard_coordinate_size(self.win)
        layout = keyboard_layout_info(cfg, (width, height), self.win.keyboard_display_size)
        if self.layout_preview and any(
                not np.allclose(self.layout_preview[name], layout[name])
                for name in ("key_positions_pix_units", "key_size_pix_units", "display_size_pix_units")):
            raise RuntimeError("显示器或布局在预览后发生变化；请重新启动并检查静态布局")
        self.positions = np.asarray(layout["key_positions_pix_units"])
        key, gap_y = layout["key_size_pix_units"], layout["row_gap_pix_units"]
        scale = min(1.25, key / 140.0)  # 定位框/箭头跟随实际方块大小。
        self.key_size = key
        # 箭头与外框保持小间距，底行仍留在底栏上方。
        cue_width = min(40.0, 36.0 * scale, max(0.0, gap_y - 12 * scale) * 9 / 5)
        self._show_cue_arrow = cue_width >= 4
        cue_height = cue_width * 5 / 9
        self.cue_offset = 6 * scale + cue_height / 2
        self._static_colors = np.tile((.72, .58, .82), (len(TARGETS), 1))
        # 每个目标的定位画面预先生成；进入闪烁时恢复原始40目标亮度表。
        self._cue_colors = np.tile((-.80, -.81, -.78), (len(TARGETS), len(TARGETS), 1))
        for index in range(len(TARGETS)):
            self._cue_colors[index, index] = self._static_colors[index]
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
        self.outline_shadow = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="black", lineWidth=max(4, 10 * scale), units="pix", autoLog=False)
        self.outline = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="#FFE45C", lineWidth=max(2, 6 * scale), units="pix", autoLog=False)
        self.triangle = (visual.ShapeStim(self.win,
            vertices=((-cue_width/2, -cue_height/2), (cue_width/2, -cue_height/2), (0, cue_height/2)),
            fillColor="#FFE45C", lineColor="black", lineWidth=max(1, 2 * scale), units="pix", autoLog=False)
            if self._show_cue_arrow else None)
        output_y = height/2 - cfg.output_box_margin_px - cfg.output_box_height_px/2
        output_width = width - 2 * _keyboard_side_margin(width)
        output_text_width = output_width - 48
        self.output_box = visual.Rect(self.win, width=output_width, height=cfg.output_box_height_px,
            pos=(0, output_y), fillColor=(-0.82, -0.82, -0.82), lineColor="white", lineWidth=2,
            units="pix", autoLog=False)
        output_font_height = min(28, max(16, cfg.output_box_height_px*0.42))
        self.output_text = visual.TextStim(self.win, text="", pos=(0, output_y),
            height=output_font_height, wrapWidth=output_text_width,
            font="Courier New", color="white", alignText="left", anchorHoriz="center", autoLog=False)
        # 自动绘制使顶部框在提示、闪烁、空白、反馈、暂停阶段都可见。
        # 文字只在字符串变化后重建，不在每个闪烁帧反复设置TextStim.text。
        self.output_box.autoDraw = True
        self.output_text.autoDraw = True
        self._output_max_chars = max(12, int(output_text_width / output_font_height))
        self._sync_output_text(force=True)
        self.header = visual.TextStim(self.win, text="Preparing display",
            pos=(0, output_y - cfg.output_box_height_px/2 - 22),
            height=min(24, max(12, height/40)), wrapWidth=width-40, color="white", autoLog=False)
        self.cue_title = visual.TextStim(self.win, text="TARGET", pos=self.header.pos,
            height=min(34, max(18, height/32)), wrapWidth=width-40,
            color="#FFE45C", bold=True, autoLog=False)
        self.footer = visual.TextStim(self.win, text="ESC: stop | Cued evaluation",
            pos=(0, -height/2+18), height=min(18, max(10, height/60)), wrapWidth=width-40,
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
            raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
        self.luminance = (np.empty((0, len(TARGETS))) if self.static_only else
                          make_luminance_table(self.refresh_hz, cfg.stimulus_s))
        # 论文亮度0..1 -> PsychoPy rgb -1..1；全部40目标同步逐帧更新。
        self.rgb_frames = np.repeat((2 * self.luminance - 1)[:, :, None], 3, axis=2)
        self.header.text = "Static render timing check - no flicker"
        flips = []
        for _ in range(121):
            self._check_abort()
            self._draw_keys()
            flips.append(self.win.flip())
        check = frame_diagnostics(flips, self.refresh_hz, cfg.frame_long_factor,
                                  cfg.frame_short_factor, cfg.max_warning_frame_anomalies,
                                  cfg.frame_rate_tolerance)
        if check['hard_invalid']:
            rate_detail = (f"整段翻转均速{check['observed_flip_rate_hz']:.2f}Hz，"
                           f"基准{self.refresh_hz:.2f}Hz，"
                           f"偏差{check['flip_rate_error_percent']:+.2f}%"
                           f"（限值{check['flip_rate_tolerance_percent']:.2f}%）" if
                           check['observed_flip_rate_hz'] is not None else "整段翻转速率不可计算")
            raise RuntimeError("完整键盘静态绘制时序异常："
                               f"严重间隔{check['severe_interval_count']}，"
                               f"最大{check['max_interval_ms']:.2f}ms，"
                               f"{rate_detail}；请先检查显示/负载")
        if check['warning']:
            print("[显示时序警告] 静态绘制出现一个异常帧间隔；继续前请检查负载。", flush=True)
        framebuffer = getattr(self.win, "frameBufferSize", self.win.size)
        content_scale = self.win.getContentScaleFactor() if hasattr(self.win, 'getContentScaleFactor') else None
        self.display_info = {**layout, "static_preview": self.layout_preview.get("static_preview"),
            "window_size_reported": tuple(map(int, self.win.size)),
            "client_size_reported": tuple(map(int, self.win.clientSize)),
            "full_screen": bool(self.win.fullscr), "screen_index": cfg.screen_index,
            "window_backend": self.win.winType,
            "framebuffer_size_reported": tuple(map(int, framebuffer)), "content_scale_factor": content_scale,
            "refresh_hz_measured": self.refresh_hz,
            "cued_guidance": {"timing": "automatic_within_block", "cue_s": cfg.cue_s,
                              "settle_s": cfg.cued_settle_s, "arrow_width_pix_units": cue_width,
                              "dim_others_during_preparation": True, "static_outline_during_stimulus": True},
            "target_viewing_distance_cm": cfg.viewing_distance_cm,
            "declared_screen_diagonal_inches": cfg.screen_diagonal_inches,
            "stimulus_frames": len(self.rgb_frames), "scheduled_stimulus_s": len(self.rgb_frames)/self.refresh_hz,
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}

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
                   show_outline: bool = False, dim_others: bool = False) -> None:
        if self.static_only and flicker_rgb is not None:
            raise RuntimeFault("DISPLAY", "静态自检禁止呈现闪烁帧")
        if flicker_rgb is not None:
            self.squares.colors = flicker_rgb
        elif dim_others and target_id is not None:
            self.squares.colors = self._cue_colors[target_id - 1]
        else:
            self.squares.colors = self._static_colors
        self.key_borders.draw()
        self.squares.draw()
        for label in self.labels:
            label.draw()
        if target_id is not None:
            pos = self.positions[target_id - 1]
            if self._show_cue_arrow:
                self.triangle.pos = (pos[0], pos[1] - self.key_size / 2 - self.cue_offset)
                self.triangle.draw()
            if show_outline:
                self.outline_shadow.pos = pos
                self.outline_shadow.draw()
                self.outline.pos = pos
                self.outline.draw()
        if target_id is not None:
            self.cue_title.draw()
        else:
            self.header.draw()
        self.footer.draw()

    def _prepare_cued_trial(self, record: TrialRecord, cue_event: EventStamp) -> None:
        """静态定位 → 定时稳定注视 → 闪烁；全程允许ESC和流健康检查。"""
        target = TARGETS[record.true_class - 1]
        progress = f"Block {record.block_id}/{self.cfg.blocks} | Trial {record.trial_id}/{self.cfg.blocks*len(TARGETS)}"
        self.cue_title.text = f"TARGET: {target.symbol}    |    ROW {target.row+1}   COL {target.col+1}"
        self.footer.text = f"{progress} | Find the highlighted key ({self.cfg.cue_s:g} s) | ESC: stop"
        # 菜单启动键及之后的误触都不参与单个试次的时间控制。
        self.event.getKeys(keyList=["space"])
        self._phase(record, "CUE")
        self._check_abort()
        self._draw_keys(target.class_id, show_outline=True, dim_others=True)
        self.win.callOnFlip(self.markers.mark, cue_event)
        self.win.flip()
        record.cue_onset = cue_event.timestamp
        # 使用实际翻转事件时间保证至少完整显示 cue_s。
        while self.clock() - record.cue_onset < self.cfg.cue_s:
            self._check_abort()
            self.event.getKeys(keyList=["space"])
            self._draw_keys(target.class_id, show_outline=True, dim_others=True)
            self.win.flip()
        self._phase(record, "FIXATE")
        self.footer.text = f"{progress} | Keep looking at the key center ({self.cfg.cued_settle_s:g} s) | ESC: stop"
        fixation = EventStamp("fixation_onset", record.trial_id, target.class_id,
                              {"settle_s": self.cfg.cued_settle_s,
                               "trigger": "automatic_after_cue"})
        self._check_abort()
        self._draw_keys(target.class_id, show_outline=True, dim_others=True)
        self.win.callOnFlip(self.markers.mark, fixation)
        self.win.flip()
        record.fixation_onset = fixation.timestamp
        while self.clock() - record.fixation_onset < self.cfg.cued_settle_s:
            self._check_abort()
            self.event.getKeys(keyList=["space"])
            self._draw_keys(target.class_id, show_outline=True, dim_others=True)
            self.win.flip()
        self.footer.text = f"{progress} | Keep looking at the key center | ESC: stop"

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

    def _persist_trial(self, record: TrialRecord) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.write_trial(record)
        except Exception as exc:
            raise RuntimeFault("SAVE", f"逐试次保存失败；保留 {self.recorder.session_dir}：{exc}") from exc

    def _note_fault(self, exc: BaseException, *, fatal: bool, default: str = "DISPLAY") -> None:
        fault = fault_record(exc, default)
        self.trial_faults.append(fault)
        if fatal:
            self.fatal_fault = fault
        print_fault(fault)

    @staticmethod
    def _attach_failed_epoch(record: TrialRecord, exc: BaseException) -> None:
        epoch = getattr(exc, "epoch", None)
        if epoch is not None and record.result is None:
            record.result = {"trial_id": record.trial_id, "epoch": epoch,
                             "quality": epoch.diagnostics.get("quality")}

    def _rejection_diagnostics(self, result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the free-mode evidence gate without treating scores as probabilities."""
        cfg = self.cfg
        min_score = (None if cfg.rejection_min_score is None
                     else float(cfg.rejection_min_score))
        min_margin = (None if cfg.rejection_min_margin is None
                      else float(cfg.rejection_min_margin))
        configured = cfg.rejection_thresholds_configured
        max_score: Optional[float] = None
        score_margin: Optional[float] = None
        reason_codes: list[str] = []
        reasons: list[str] = []

        if cfg.rejection_enabled and result.get("calibration_fallback"):
            reason_codes.append("calibration_fallback")
            reasons.append("个人模板回退，当前分数不适用个人模型的拒识阈值")

        if cfg.rejection_enabled and not configured:
            reason_codes.append("thresholds_unconfigured")
            reasons.append(
                "拒识阈值未完整配置；请先用明确注视目标和不打算输入的真实EEG数据标定，"
                "再用独立数据验证")

        try:
            scores = np.asarray(result.get("scores", []), dtype=float).ravel()
        except (TypeError, ValueError):
            scores = np.empty(0, dtype=float)
        if scores.size == 0 or not np.isfinite(scores).all():
            reason_codes.append("scores_missing_or_nonfinite")
            reasons.append("分类分数缺失或非有限")
        else:
            ordered = np.sort(scores)
            max_score = float(ordered[-1])
            if ordered.size < 2:
                reason_codes.append("insufficient_scores")
                reasons.append("候选分数少于两个，无法计算第一/第二名分差")
            else:
                score_margin = float(ordered[-1] - ordered[-2])
                if configured and max_score < float(min_score):
                    reason_codes.append("score_below_threshold")
                    reasons.append(
                        f"最高分{max_score:.6g}低于阈值{float(min_score):.6g}")
                if configured and score_margin < float(min_margin):
                    reason_codes.append("margin_below_threshold")
                    reasons.append(
                        f"第一/第二名分差{score_margin:.6g}低于阈值{float(min_margin):.6g}")

        accepted = bool(
            max_score is not None
            and score_margin is not None
            and (not cfg.rejection_enabled or (
                configured
                and max_score >= float(min_score)
                and score_margin >= float(min_margin)))
            and not reason_codes
        )
        if accepted:
            reason_codes = []
            reason = None
            reason_code = None
        else:
            # Keep a stable, inspectable code while retaining all contributing
            # codes for diagnostics and post-session analysis.
            reason_code = "+".join(reason_codes) or "rejected"
            reason = "拒识：" + "; ".join(reasons)
        return {
            "enabled": cfg.rejection_enabled,
            "configured": configured,
            "accepted": accepted,
            "max_score": max_score,
            "score_margin": score_margin,
            "min_score": min_score,
            "min_margin": min_margin,
            "reason_code": reason_code,
            "reason_codes": reason_codes,
            "reason": reason,
        }

    def _rejection_diagnostics_for_mode(self, mode: str,
                                        result: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Check free-mode scores; apply thresholds only when enabled."""
        return self._rejection_diagnostics(result) if mode == "free" else None

    @staticmethod
    def _rejection_calibration_notice() -> str:
        return (
            "Rejection thresholds are not configured.\n\n"
            "Before FREE INPUT can add a character, space, or delete action, "
            "calibrate both thresholds with real EEG while looking at an intended "
            "target and while not intending to enter text, then verify them on "
            "independent data.\n\n"
            "FBCCA scores are not probabilities. This gate is not validated "
            "automatic idle detection; no text will be entered until it is configured."
        )

    def _blank_until(self, when: float) -> None:
        while self.clock() < when:
            self._check_abort()
            self._draw_status()
            self.win.flip()

    def _present_trial_stimulus(self, record: TrialRecord, cue_event: EventStamp,
                                onset_event: EventStamp, offset_event: EventStamp) -> None:
        cfg = self.cfg
        free = record.mode == "free"
        target_id = record.true_class
        if free:
            self._phase(record, "READY")
            for frame in range(max(1, round(cfg.free_prepare_s * self.refresh_hz))):
                self._check_abort()
                self._draw_keys()
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, cue_event)
                self.win.flip()
            record.cue_onset = cue_event.timestamp
            self.header.text = f"FREE | Selection {record.trial_id} | Keep looking at your chosen character"
        else:
            self._prepare_cued_trial(record, cue_event)
        self._phase(record, "STIMULATE")
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            for frame, rgb in enumerate(self.rgb_frames):
                self._check_abort()
                # 静态提示留在方块外；40目标仍使用同一原始灰度正弦亮度表。
                self._draw_keys(target_id, rgb, show_outline=not free)
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
        record.frame_diagnostics = frame_diagnostics(
            record.frame_flip_times, self.refresh_hz, cfg.frame_long_factor,
            cfg.frame_short_factor, cfg.max_warning_frame_anomalies,
            cfg.frame_rate_tolerance)
        record.frame_diagnostics['psychopy_frame_intervals_s'] = np.asarray(self.win.frameIntervals).copy()

    def _await_trial_result(self, record: TrialRecord, onset_event: EventStamp,
                            offset_event: EventStamp) -> dict[str, Any]:
        cfg = self.cfg
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
            raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")
        if (not 1 <= result['prediction'] <= len(TARGETS)
                or not 1 <= result['fbcca_prediction'] <= len(TARGETS)
                or not 1 <= result['cca_prediction'] <= len(TARGETS)):
            raise InvalidTrial("预测类别超出40目标范围")
        return result

    def _apply_trial_result(self, record: TrialRecord, result: dict[str, Any],
                            target: Optional[Target]) -> None:
        free = record.mode == "free"
        record.result = result
        mains = result.get("quality", {}).get("mains_noise", {})
        if mains.get("line_dominated"):
            print(f"[试次{record.trial_id} 工频提示] " + "; ".join(mains["warnings"]), flush=True)
        rejection_diagnostics = self._rejection_diagnostics_for_mode(record.mode, result)
        if rejection_diagnostics is not None:
            result["rejection_diagnostics"] = rejection_diagnostics
        if rejection_diagnostics is not None and not rejection_diagnostics["accepted"]:
            rejection_reason = rejection_diagnostics["reason"]
            record.status = "rejected"
            record.reason = rejection_reason
            self._phase(record, "FEEDBACK")
            self.header.text = f"FREE | Selection {record.trial_id} | Rejected"
            self.show_message(rejection_reason + "\n\nNo character added.\n"
                              "Choose your next character in the next round.", self.cfg.feedback_s)
            return
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
            fb_pred = TARGETS[result['fbcca_prediction'] - 1]
            fb_correct = result['fbcca_prediction'] == target.class_id
            feedback = (f"Target: {target.symbol} ({target.frequency_hz:.1f} Hz)\n"
                        f"Online ({result['decoder']}): {pred.symbol} ({pred.frequency_hz:.1f} Hz)  {'CORRECT' if correct else 'WRONG'}\n"
                        f"FBCCA: {fb_pred.symbol} ({fb_pred.frequency_hz:.1f} Hz)  {'CORRECT' if fb_correct else 'WRONG'}\n"
                        f"CCA: {cca_pred.symbol} ({cca_pred.frequency_hz:.1f} Hz)\n\nESC: stop")
        if mains.get("line_dominated"):
            feedback += "\nHigh mains interference: check electrode / reference / BIAS contact."
        self.show_message(feedback, self.cfg.feedback_s)

    def _run_trial(self, record: TrialRecord) -> None:
        cfg = self.cfg
        self.collector.check_health()
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
            self.footer.text = "Follow the highlighted key; trials advance automatically | ESC: stop"
        self._sync_output_text()
        details = {"mode": record.mode, "window_s": cfg.window_s,
                   "response_delay_s": cfg.response_delay_s,
                   "minimum_stimulus_s": cfg.stimulus_s}
        if target is not None:
            details["frequency_hz"] = target.frequency_hz
        cue_event = EventStamp("ready_onset" if free else "cue_onset", record.trial_id, target_id,
            {} if free else {"cue_s": cfg.cue_s, "progression": "automatic",
                             "settle_s": cfg.cued_settle_s})
        onset_event = EventStamp("stimulus_onset", record.trial_id, target_id, details)
        offset_event = EventStamp("stimulus_offset", record.trial_id, target_id)
        self._present_trial_stimulus(record, cue_event, onset_event, offset_event)
        self._phase(record, "BLANK")
        if record.requested_window_end > record.stimulus_offset + 1e-6:
            self._blank_until(offset_event.timestamp + cfg.blank_min_s)
            raise InvalidTrial(
                "EEG分析窗超出实际闪烁区间："
                f"window_end={record.requested_window_end:.6f}, "
                f"stimulus_offset={record.stimulus_offset:.6f}；本试次不分类")
        if record.frame_diagnostics['hard_invalid']:
            self._blank_until(offset_event.timestamp + cfg.blank_min_s)
            observed_rate = record.frame_diagnostics['observed_flip_rate_hz']
            rate_detail = (f"整段翻转均速{observed_rate:.2f}Hz，"
                           f"基准{self.refresh_hz:.2f}Hz，"
                           f"偏差{record.frame_diagnostics['flip_rate_error_percent']:+.2f}%"
                           f"（限值{record.frame_diagnostics['flip_rate_tolerance_percent']:.2f}%）" if
                           observed_rate is not None else "整段翻转速率不可计算")
            raise InvalidTrial(f"刺激时序异常：长间隔{record.frame_diagnostics['long_intervals']}，"
                               f"短间隔{record.frame_diagnostics['short_intervals']}，"
                               f"严重间隔{record.frame_diagnostics['severe_interval_count']}"
                               "（单间隔至少2个刷新周期或时间戳非递增），"
                               f"最大{record.frame_diagnostics['max_interval_ms']:.2f}ms，"
                               f"{rate_detail}", code="DISPLAY")
        if record.frame_diagnostics['warning']:
            print(f"[显示时序警告] trial {record.trial_id} 存在一个异常帧间隔："
                  f"长间隔{record.frame_diagnostics['long_intervals']}，"
                  f"短间隔{record.frame_diagnostics['short_intervals']}；继续分类。", flush=True)
        if self.markers.error is not None:
            raise RuntimeFault("LSL_CONNECTION", f"LSL Marker发送失败：{self.markers.error}")
        self._phase(record, "WAIT_DATA_AND_CLASSIFY")
        self.header.text = (f"FREE | Selection {record.trial_id} | Processing EEG..." if free
                            else f"Trial {record.trial_id} | Processing EEG...")
        result = self._await_trial_result(record, onset_event, offset_event)
        self._apply_trial_result(record, result, target)

    def _select_mode(self) -> str:
        self._pause_enabled = False
        selected = self.cfg.session_mode
        self.event.getKeys(keyList=["1", "2", "num_1", "num_2", "space", "return", "num_enter"])
        previous = None
        self.header.text = "40-target SSVEP keyboard | Select a mode"
        self.footer.text = "ESC: exit | No flicker until you press SPACE or ENTER"
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
                    f"Both modes: at least {self.cfg.stimulus_s:g} s flicker, "
                    f"{self.cfg.window_s:g} s EEG window starting {self.cfg.response_delay_s:g} s after onset.\n"
                    "Free mode: keyboard SPACE pauses/resumes.\n"
                    f"Cued mode: SPACE starts; each target advances automatically after "
                    f"{self.cfg.cue_s:g} s cue and {self.cfg.cued_settle_s:g} s steady gaze. "
                    f"Cued mode: rest {self.cfg.cued_rest_s:g} s after every "
                    f"{self.cfg.cued_rest_every} completed targets, then resume automatically.\n"
                    "Block breaks and quality pauses require SPACE.\n"
                    "Blank bottom-left key: insert space.  <-: delete.\n"
                    + (f"Personal calibration: {self.cfg.participant_id}\n"
                       if self.cfg.calibration_file else "Decoder: uncalibrated FBCCA\n") +
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
        if self.static_only:
            raise RuntimeFault("DISPLAY", "静态自检不能启动实验")
        mode = self._select_mode()
        return self._run_free() if mode == "free" else self._run_cued()

    def _wait_free_pause(self, message: str) -> None:
        # 暂停界面不产生刺激；EEG仍连续采集。后台运算结束前不启动新一轮。
        self._pause_enabled = False
        self._cancel_decode()
        self.event.getKeys(keyList=["space", "return", "num_enter"])
        started = self.clock()
        self.markers.mark(EventStamp("pause", 0, details={"typed_text": self.ledger.typed_text}))
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
                self.markers.mark(EventStamp("resume", 0))
                return

    @staticmethod
    def _release_free_eeg(record: TrialRecord) -> None:
        # 自由输入可持续很久：保留轻量预测/时序摘要，不累积每轮EEG与逐帧数组。
        if record.result is not None:
            record.result.pop("epoch", None)
        record.frame_flip_times.clear()
        record.frame_diagnostics.pop("intervals_s", None)
        record.frame_diagnostics.pop("psychopy_frame_intervals_s", None)

    @staticmethod
    def _advance_free_invalid_streak(current: int, record: TrialRecord,
                                     *, pause_requested: bool,
                                     exit_requested: bool) -> int:
        """Count only real failed trials; normal rejection is not a fault."""
        if (pause_requested or exit_requested or record.text_applied
                or record.status == "rejected"):
            return 0
        return current + 1

    def _run_free(self) -> str:
        self._pause_enabled = False
        selection_id = 0
        consecutive_invalid = 0
        try:
            if self.cfg.rejection_enabled and not self.cfg.rejection_thresholds_configured:
                notice = self._rejection_calibration_notice()
                print("[自由输入保护] " + notice.replace("\n", " "), flush=True)
                self.show_message(notice, max(self.cfg.feedback_s, 2.0))
            self._pause_enabled = True
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
                    self._attach_failed_epoch(record, exc)
                    record.status = "valid" if record.text_applied else "invalid"
                    record.reason = f"{type(exc).__name__}: {exc}"
                    fatal = (isinstance(exc, RuntimeFault) or not isinstance(exc, InvalidTrial)
                             or self.collector.error is not None or self.markers.error is not None)
                    self._note_fault(exc, fatal=fatal)
                finally:
                    self._cancel_decode()
                    try:
                        self.win.recordFrameIntervals = False
                        self._draw_status()
                        self.win.flip()  # 清除所有闪烁目标；autoDraw保留输出框。
                    except Exception as exc:
                        fatal = True
                        record.reason += f"; display-close error: {exc}"
                        self._note_fault(exc, fatal=True, default="DISPLAY")
                        if not record.text_applied:
                            record.status = "invalid"
                    record.end = self.clock()
                    self._phase(record, "COMPLETE")
                    try:
                        self.markers.mark(EventStamp("selection_end", record.trial_id, None,
                            {"mode": "free", "status": record.status, "reason": record.reason,
                             "prediction": (record.result or {}).get("prediction"),
                             "rejection_diagnostics": (
                                 record.result.get("rejection_diagnostics")
                                 if record.result else None),
                             "text_applied": record.text_applied}))
                    except Exception as exc:
                        record.reason += f"; marker error: {exc}"
                        fatal = True
                        self._note_fault(exc, fatal=True, default="LSL_CONNECTION")
                    self.ledger.commit(record)
                    self._persist_trial(record)
                    self._release_free_eeg(record)
                if record.text_applied:
                    consecutive_invalid = 0
                elif record.status == "rejected":
                    # A normal evidence-gate rejection is not an acquisition,
                    # data, or timing fault and must not create a fault streak.
                    print(f"FREE {selection_id:04d}: rejected | {_brief_reason(record.reason)}", flush=True)
                else:
                    print(f"FREE {selection_id:04d}: no character added | {_brief_reason(record.reason)}", flush=True)
                consecutive_invalid = self._advance_free_invalid_streak(
                    consecutive_invalid, record,
                    pause_requested=pause_requested, exit_requested=exit_requested)
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
        # 在首个试次前准备音频，避免休息倒数期间才初始化音频设备。
        try:
            from psychopy import sound
            self._rest_beep = sound.Sound(880, secs=0.15, volume=0.5, autoLog=False)
        except Exception as exc:
            raise RuntimeFault("AUDIO", f"模式2休息提示音初始化失败：{exc}") from exc
        schedule = deque(make_schedule(cfg.blocks, cfg.random_seed))
        consecutive_invalid = 0
        completed_in_block = 0
        previous_block = 1
        stop_reason = "Completed planned trials"
        trial_id = 0
        while schedule:
            block_id, target_id = schedule.popleft()
            trial_id += 1
            if block_id != previous_block:
                self.show_message(f"Block {previous_block} complete. Rest your eyes.\n"
                    f"Minimum break: {cfg.block_rest_s:g} seconds.\n\nThen SPACE to continue; ESC to finish.",
                    cfg.block_rest_s, wait_space=True)
                previous_block = block_id
                completed_in_block = 0
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
                self._attach_failed_epoch(record, exc)
                record.status = "valid" if record.text_applied else "invalid"
                record.reason = f"{type(exc).__name__}: {exc}"
                fatal = (isinstance(exc, RuntimeFault) or not isinstance(exc, InvalidTrial)
                         or self.collector.error is not None or self.markers.error is not None)
                self._note_fault(exc, fatal=fatal)
                stop_reason = record.reason
            finally:
                try:
                    self.win.recordFrameIntervals = False
                    self.win.flip()  # 所有异常路径立即去掉闪烁画面
                except Exception as clear_error:
                    # 例如用户关闭了窗口：即使清屏失败，也必须留下已开始试次的记录。
                    if not record.text_applied and record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; display-close error: {clear_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                    self._note_fault(clear_error, fatal=True, default="DISPLAY")
                record.end = self.clock()
                self._phase(record, "COMPLETE")
                try:
                    self.markers.mark(EventStamp("trial_end", record.trial_id, record.true_class,
                        {"status": record.status, "reason": record.reason,
                         "prediction": (record.result or {}).get("prediction")}))
                except Exception as marker_error:
                    if not record.text_applied and record.status != "aborted":
                        record.status = "invalid"
                    record.reason = (record.reason + f"; marker error: {marker_error}").strip("; ")
                    stop_reason, fatal = record.reason, True
                    self._note_fault(marker_error, fatal=True, default="LSL_CONNECTION")
                self.ledger.commit(record)
                self._persist_trial(record)
            if record.status == 'valid':
                consecutive_invalid = 0
                completed_in_block += 1
            else:
                consecutive_invalid += 1
                print(f"Trial {trial_id:03d}: {record.status.upper()} - {_brief_reason(record.reason)}", flush=True)
                if not fatal:
                    # Preserve the scheduled target. A failed attempt gets a new
                    # trial ID, so every target can still obtain one valid epoch.
                    schedule.appendleft((block_id, target_id))
            if fatal:
                break
            if consecutive_invalid >= cfg.max_consecutive_invalid:
                print(f"[采集暂停] 连续{consecutive_invalid}次无效：{record.reason}", flush=True)
                self.show_message(
                    "Several attempts were invalid. The current target will be retried.\n"
                    "Check the EEG stream, then press SPACE to continue; ESC to stop.",
                    cfg.feedback_s, wait_space=True)
                consecutive_invalid = 0
            if (record.status == 'valid' and completed_in_block % cfg.cued_rest_every == 0
                    and schedule and schedule[0][0] == block_id):
                self.header.text = (f"Block {block_id}/{cfg.blocks} | "
                                    f"{completed_in_block}/{len(TARGETS)} targets complete")
                self.footer.text = "Resting | No flicker | ESC: stop"
                print(f"[模式2休息] 已完成{completed_in_block}个目标，休息{cfg.cued_rest_s:g}秒。",
                      flush=True)
                self._wait_cued_rest(completed_in_block)
        else:
            stop_reason = "Completed planned trials"
        return stop_reason

    def _wait_cued_rest(self, completed_in_block: int) -> None:
        """静态休息30秒；最后3、2、1秒各响一次，结束后自动继续。"""
        duration = self.cfg.cued_rest_s
        self._sync_output_text()
        self.win.recordFrameIntervals = False
        started = self.clock()
        last_beep_second = None
        try:
            while True:
                self._check_abort()
                remaining = duration - (self.clock() - started)
                if remaining <= 0:
                    return
                seconds_left = math.ceil(remaining)
                if duration >= 3 and seconds_left in (3, 2, 1):
                    if seconds_left != last_beep_second:
                        try:
                            self._rest_beep.play()
                        except Exception as exc:
                            raise RuntimeFault("AUDIO", f"模式2休息提示音播放失败：{exc}") from exc
                        last_beep_second = seconds_left
                    countdown = f"Resuming in {seconds_left}..."
                else:
                    countdown = f"Resuming automatically after {duration:g} seconds."
                message = (f"{completed_in_block}/{len(TARGETS)} targets complete. "
                           f"Rest your eyes.\n{countdown}\n\nESC: stop.")
                if self.message.text != message:
                    self.message.text = message
                self._draw_status()
                self.message.draw()
                self.win.flip()
        finally:
            if self._rest_beep is not None:
                self._rest_beep.stop()

    def close(self) -> None:
        self.cancel.set()
        self._cancel_decode()
        if self.win is not None:
            try:
                self.win.recordFrameIntervals = False
                self.win.flip()
            except Exception:
                pass
        try:
            if self.win is not None:
                self.win.close()
                self.win = None
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


def static_ui_self_test(cfg: Config, *, seconds: float = 5.0) -> dict:
    """Render the production keyboard, fixed cue and output; never start LSL or decoding."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("自检时长必须为有限正数")
    app = None
    result = {"status": "cancelled", "exit_code": 130,
              "eeg_connected": False, "flicker_presented": False,
              "scope": "静态布局、字体、窗口与软件刷新时序；不验证EEG、识别准确率或物理显示延迟"}
    try:
        # An unstarted collector only supplies the monotonic clock/health interface.
        # Do not construct EventMarkers, a recorder, or a fake EEG stream.
        app = PsychoPyKeyboard(replace(cfg), ContinuousLSL(cfg), None, TrialLedger(), static_only=True)
        app.ledger.typed_text = "STATIC UI CHECK  ABC 123"
        app._sync_output_text(force=True)
        app.cue_title.text = "STATIC UI CHECK - no EEG / no flicker"
        app.footer.text = "40 keys + output + fixed cue | ESC: cancel | Closes automatically"
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            app._check_abort()
            app._draw_keys(1, show_outline=True)
            app.win.flip()
        result.update(status="passed", exit_code=0, display_info=app.display_info)
    except AbortSession:
        print("静态界面自检已取消，未记为通过。", flush=True)
    except Exception as exc:
        raise RuntimeFault("DISPLAY", str(exc)) from exc
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as exc:
                fault = fault_record(exc, "DISPLAY")
                result.setdefault("errors", []).append(fault)
                result.update(status="failed", exit_code=40)
                print_fault(fault)
    if result["status"] == "passed":
        print("静态界面自检通过：40键、文字、输出框、固定提示与静态绘制时序已检查。", flush=True)
    return result


def main(*, cfg: Optional[Config] = None) -> dict:
    """唯一运行入口；不要求CLI参数；返回全部内存记录。"""
    global LAST_SESSION
    # 启动对话框与模式菜单会修改配置；每场会话从用户设定的默认值复制。
    cfg = replace(cfg if cfg is not None else CONFIG)
    if os.environ.get("FBCCA_QUICK_ENTRY") == "1":
        cfg.quick_entry_mode = True
    cfg.validate()
    calibration = None
    if cfg.calibration_file:
        model_path = Path(cfg.calibration_file)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        calibration = PersonalCalibration.load(model_path, cfg)
        print(f"个人模型已加载：受试者 {cfg.participant_id}，"
              f"每类 {calibration.counts.min()}–{calibration.counts.max()} 次校准，"
              f"模型 {calibration.model_id[:12]}", flush=True)
    print("[1/4] 运行环境由启动入口核验；准备静态布局与保存目录检查…", flush=True)
    print(f"安全提示：此程序呈现{BENCHMARK_FREQUENCIES_HZ.min():g}–{BENCHMARK_FREQUENCIES_HZ.max():g}Hz视觉闪烁。"
          "对闪光敏感、有光敏性癫痫史者不要自行测试；"
          "出现眼部不适、头痛、眩晕等应立即按ESC停止。先阅读风险并核对接线，程序不会直接开始闪烁。")
    collector = ContinuousLSL(cfg)
    collector.calibration = calibration
    ledger = TrialLedger()
    markers: Optional[EventMarkers] = None
    app: Optional[PsychoPyKeyboard] = None
    recorder: Optional[SessionRecorder] = None
    layout_preview: dict[str, Any] = {}
    review: dict = {}
    reason = "Not started"
    errors: list[dict] = []
    stage = "SAVE"

    def add_error(exc: BaseException, default: str) -> None:
        fault = fault_record(exc, default)
        errors.append(fault)
        print_fault(fault)

    try:
        probe_save_directory(cfg.record_root)
        stage = "DISPLAY"
        print("先检查静态布局：方向键调整尺寸/间距/距离/屏幕英寸，U切换像素/视角；"
              "按1–4评价四角，ENTER确认；此阶段无闪烁、无需EEG。", flush=True)
        layout_preview = preview_keyboard_layout(cfg)
        stage = "SAVE"
        recorder = SessionRecorder(cfg, context={"mode_at_start": cfg.session_mode,
                                                  "quick_entry_mode": cfg.quick_entry_mode,
                                                  "trained_model": ({
                                                      "model_id": calibration.model_id,
                                                      **calibration.metadata,
                                                  } if calibration is not None else None),
                                                  "layout_preview": layout_preview})
        print("\n[2/4] 连接EEG LSL并检查采样时间轴（请保持采集软件的LSL输出开启）…", flush=True)
        stage = "LSL_CONNECTION"
        collector.start()
        print("[3/4] EEG已接收，快速模式直接继续…" if cfg.quick_entry_mode
              else "[3/4] EEG已接收，打开启动核验对话框…", flush=True)
        stage = "DISPLAY"
        review = preflight_review(cfg, collector)
        if calibration is not None:
            calibration.validate_config(cfg)  # 启动核验可能修改陷波设置。
        stage = "LSL_CONNECTION"
        markers = EventMarkers(cfg, collector.clock)
        markers.mark(EventStamp("session_start", 0, details={"mode_at_start": cfg.session_mode}))
        # 预热SciPy/BLAS与分类代码，但不使用真实目标，不计入准确率。
        stage = "INTERNAL"
        t = np.arange(sample_count(cfg.window_s, cfg.target_fs)) / cfg.target_fs
        warm = np.tile(np.sin(2*np.pi*10*t), (len(cfg.channel_indices), 1))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            classify_eeg_window(warm, cfg.target_fs, cfg.window_s, target_fs=cfg.target_fs,
                notch_hz=cfg.notch_hz, n_harmonics=cfg.n_harmonics,
                a=cfg.weight_a, b=cfg.weight_b, regularization=cfg.cca_regularization,
                filter_bands=cfg.filter_bands, line_regression_hz=cfg.line_regression_hz,
                calibration=calibration)
        print("[4/4] 创建键盘窗口；在菜单按1自由输入、2提示测试，再按SPACE开始…", flush=True)
        stage = "DISPLAY"
        app = PsychoPyKeyboard(cfg, collector, markers, ledger, recorder, layout_preview)
        reason = app.run()
        if app.fatal_fault is not None:
            errors.append(app.fatal_fault)
    except (AbortSession, KeyboardInterrupt) as exc:
        reason = str(exc) or "用户中止"
    except Exception as exc:
        add_error(exc, stage)
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        # 窗口/设备关闭异常不能跳过已采集数据的最终导出。
        for resource in (app, collector):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    add_error(exc, "DISPLAY" if resource is app else "ACQUISITION")
                    reason += f"; {type(resource).__name__}.close: {type(exc).__name__}: {exc}"
        if markers is not None:
            try:
                markers.mark(EventStamp("session_end", 0, details={"stop_reason": reason}))
            except Exception as exc:
                add_error(exc, "LSL_CONNECTION")
                reason += f"; session_end marker: {type(exc).__name__}: {exc}"
            try:
                markers.close()
            except Exception as exc:
                add_error(exc, "LSL_CONNECTION")
                reason += f"; markers.close: {type(exc).__name__}: {exc}"
        report = (summarize_free_trials(ledger.records, ledger) if cfg.session_mode == "free"
                  else summarize_trials(ledger.records, cfg.blocks * len(TARGETS)))
        session_artifact = None
        if recorder is not None:
            try:
                session_artifact = recorder.finalize(
                    summary=report, events=markers.events if markers else [],
                    acquisition_metadata=collector.metadata, acquisition_review=review,
                    clock_updates=collector.clock_updates,
                    display_info=app.display_info if app else layout_preview,
                    typed_text=ledger.typed_text, stop_reason=reason)
            except Exception as exc:
                add_error(exc, "SAVE")
                reason += f"; 保存失败：{exc}"
                if recorder.archive_path.exists() and cfg.session_mode == "cued":
                    session_artifact = str(recorder.archive_path)
                    print(f"会话ZIP已保存，但临时文件清理未完成：{type(exc).__name__}: {exc}", flush=True)
                elif recorder.session_path.exists():
                    session_artifact = str(recorder.session_path)
                    print(f"会话NPZ已保存，但配套文件封装未完成：{type(exc).__name__}: {exc}", flush=True)
                else:
                    print(f"实验记录导出失败，已有逐试次文件保留：{type(exc).__name__}: {exc}", flush=True)
        if collector.error is not None and not any(f["code"] == "ACQUISITION" for f in errors):
            add_error(collector.error, "ACQUISITION")
        if markers is not None and markers.error is not None and not any(f["code"] == "LSL_CONNECTION" for f in errors):
            add_error(markers.error, "LSL_CONNECTION")
        LAST_SESSION = {"mode": cfg.session_mode, "config": cfg, "targets": TARGETS, "records": ledger.records,
            "events": markers.events if markers else [], "acquisition_metadata": collector.metadata,
            "acquisition_review": review, "clock_updates": collector.clock_updates,
            "display_info": app.display_info if app else layout_preview, "summary": report,
            "typed_text": ledger.typed_text, "stop_reason": reason,
            "exit_code": (50 if any(f["code"] == "SAVE" for f in errors)
                          else errors[0]["exit_code"] if errors else 0),
            "errors": errors, "trial_faults": app.trial_faults if app else [],
            "session_artifact": session_artifact,
            "export_artifacts": dict(recorder.export_artifacts) if recorder else {},
            "marker_error": str(markers.error) if markers and markers.error else None,
            "validation_scope": "software events only; physical screen/device delays unmeasured"}
        if cfg.session_mode == "free":
            print_free_summary(report)
        else:
            print_summary(report, ledger)
        saved_path = session_artifact
        if saved_path is None and recorder is not None and recorder.session_dir.exists():
            saved_path = str(recorder.session_dir)
        print(f"结束原因：{reason}")
        print(f"保存路径：{saved_path or '未生成'}")
        if recorder is not None:
            for path in recorder.export_artifacts.values():
                if path != session_artifact:
                    print(f"导出文件：{path}")
        if markers is not None and markers.error is not None:
            print(f"外部Marker发送存在错误：{markers.error}；请勿把本次外部Marker流视为完整。")
    return LAST_SESSION


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds 必须是有限正数")
    return seconds


def cli(argv=None) -> int:
    """Windows 与 macOS 共用的单文件命令行入口。"""
    configure_console()
    parser = argparse.ArgumentParser(description="40目标 EEG 键盘")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--diagnose", action="store_true", help="检查依赖、liblsl 和保存目录")
    modes.add_argument("--check-lsl-timing", action="store_true",
                       help="不显示闪烁，连接EEG并检查LSL时间轴")
    modes.add_argument("--self-test-ui", action="store_true", help="运行静态键盘自检")
    modes.add_argument("--preview-layout", action="store_true", help="交互调整静态布局")
    parser.add_argument("--seconds", type=positive_seconds, default=5.0,
                        help="静态自检展示或LSL预热后持续检查的秒数")
    parser.add_argument("--windowed", action="store_true", help="使用1280×800窗口")
    parser.add_argument("--screen", type=int, help="显示器编号，从0开始")
    parser.add_argument("--record-root", help="会话保存目录")
    parser.add_argument("--participant", help="匿名受试者编号；加载个人模型时必填")
    parser.add_argument("--calibration", help="个人校准模型NPZ、完整模式2会话PKL或ZIP路径（受试者与配置须匹配）")
    args = parser.parse_args(argv)
    if args.screen is not None and args.screen < 0:
        parser.error("screen 必须非负")
    report = {
        "mode": "static_ui" if args.self_test_ui else "layout_preview" if args.preview_layout
                else "lsl_timing" if args.check_lsl_timing else "diagnose" if args.diagnose else "eeg",
        "status": "failed", "errors": [], "exit_code": 0,
    }
    try:
        report["checks"] = check_environment(static_ui=args.self_test_ui or args.preview_layout)
        cfg = replace(CONFIG)
        if args.windowed:
            cfg.full_screen, cfg.window_size = False, (1280, 800)
        if args.screen is not None:
            cfg.screen_index = args.screen
        if args.record_root:
            cfg.record_root = args.record_root
        if args.participant:
            cfg.participant_id = args.participant
        if args.calibration:
            cfg.calibration_file = args.calibration
        cfg.validate()
        if args.preview_layout:
            try:
                report["display_info"] = preview_keyboard_layout(cfg)
                report["status"] = "finished"
            except AbortSession:
                report.update(status="cancelled", exit_code=130)
            except Exception as exc:
                raise RuntimeFault("DISPLAY", str(exc)) from exc
        elif args.self_test_ui:
            report.update(static_ui_self_test(cfg, seconds=args.seconds))
        elif args.diagnose:
            report["checks"]["save_directory"] = probe_save_directory(cfg.record_root)
            report["status"] = "passed"
            print("环境诊断通过；尚未验证 EEG 流、真实采样或显示窗口。", flush=True)
        elif args.check_lsl_timing:
            collector = ContinuousLSL(cfg)
            try:
                collector.start()
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    collector.check_health()
                    collector.stop_event.wait(min(.05, deadline - time.monotonic()))
                collector.check_health()
                assert collector.buffer is not None
                _, _, local, _, _ = collector.buffer.snapshot()
                age = collector.clock() - local[-1]
                report["checks"]["lsl_timing"] = {
                    "stream": collector.metadata["name"],
                    "declared_fs_hz": collector.fs,
                    "observed_fs_hz": collector.estimated_fs,
                    "latest_sample_age_s": age,
                    "post_warmup_observation_s": args.seconds,
                    "timestamp_processing": collector.metadata["timestamp_processing"],
                }
                report["status"] = "passed"
                print(f"LSL时间轴检查通过：{collector.metadata['name']}；"
                      f"估计{collector.estimated_fs:.3f}Hz；最新样本年龄{age:.3f}s。"
                      "此检查不测量设备到屏幕的物理延迟。", flush=True)
            finally:
                collector.close()
        else:
            result = main(cfg=cfg)
            report.update({key: result[key] for key in
                           ("exit_code", "errors", "trial_faults", "stop_reason", "session_artifact")})
            report["status"] = "failed" if report["exit_code"] else "finished"
    except KeyboardInterrupt:
        report.update(status="cancelled", exit_code=130)
    except Exception as exc:
        fault = fault_record(exc)
        report["errors"].append(fault)
        report["exit_code"] = fault["exit_code"]
        print_fault(fault)
    if write_diagnostic(report) is None and not report["exit_code"]:
        report["exit_code"] = 50
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(cli())
