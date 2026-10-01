"""40 目标 SSVEP 键盘：自由输入与提示测试。

启动时自动检查OpenBCI GUI的LSL EEG传输；1=自由输入，2=提示测试（默认）；SPACE/ENTER开始，ESC退出。
键盘布局固定为截图中的4×10排列，按窗口大小等比例适配，无启动布局预览。
正常启动先收集10秒数据检查时间轴；--check-lsl-timing 默认预热30秒，再持续检查EEG流。
--warmup-seconds可指定预热时长（不低于检查窗及陷波历史所需数据）；--seconds指定观察时长。
预检同时报告各电极的工频能量占比；--protocol standard_5s 可做5秒窗复测，--seed更换目标顺序。
提示测试：每个目标测试3轮；轮间静态休息3分钟，最后3秒逐秒提示音后自动继续。
自由输入：自行注视字符，每轮真实LSL EEG经FBCCA识别后写入顶部OUTPUT框。
电脑键盘SPACE暂停/继续；屏幕内SPACE目标插入空格，BACK目标删除上一字符。
采用4×10 QWERTY布局；按新行列重排40类及频率（8–15.8 Hz）。
默认使用论文M3的7子带/5谐波；--fbcca-profile line_robust 可选择原48Hz/3谐波对照。
--notch-mode history 可用前4秒真实源数据辅助陷波；分析窗不变，保存上下文，不兼容旧个人模板。
论文M3要求原始采样率支持90Hz上限；--protocol paper_window_1p25s 仅设置分析窗，不复刻整套论文时序。
可通过 --calibration 和 --participant 加载个人模板；没有自动空闲检测。
不包含迷宫，不注入操作系统按键；首个已开始试次结束后暂存逐试次数据，包括无EEG的失败试次。
两种模式均导出会话ZIP，包含连续EEG、同步信号、事件、逐帧时间及同步报告。
模式2另含质控JSON和个人模板PKL；导出失败时保留逐试次及连续数据分块。
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
import gc
import importlib
import importlib.metadata as metadata
import os
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
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np


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
    """将标准输出和标准错误流配置为容错的 UTF-8 编码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


class RuntimeFault(RuntimeError):
    """携带标准故障代码的运行时异常。"""

    def __init__(self, code: str, detail: str):
        """使用故障代码和可读详情初始化异常。"""
        self.code = code
        super().__init__(detail)


def fault_record(exc: BaseException, default: str = "INTERNAL") -> dict:
    """将异常转换为可序列化的标准故障记录。"""
    code = getattr(exc, "code", default)
    exit_code, title, action = FAULTS[code]
    return {"code": code, "exit_code": exit_code, "title": title, "detail": str(exc),
            "action": action,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def print_fault(fault: dict) -> None:
    """将标准故障记录及处理建议输出到标准错误流。"""
    print(f"[{fault['code']}] {fault['title']}：{fault['detail']}\n处理建议：{fault['action']}",
          file=sys.stderr, flush=True)


def environment_snapshot() -> dict:
    """采集 Python、平台、依赖版本和 LSL 库覆盖配置。"""
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "platform": platform.platform(), "machine": platform.machine(),
            "bits": struct.calcsize("P") * 8,
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()
                         if d.metadata.get("Name")},
            "mne_lsl_lib_override": os.environ.get("MNE_LSL_LIB") or os.environ.get("PYLSL_LIB")}


def check_environment(*, static_ui: bool = False) -> dict:
    """检查解释器和运行依赖，静态 UI 模式可跳过 LSL 检查。"""
    issues = []
    if (sys.version_info[:3] != PYTHON_VERSION or struct.calcsize("P") != 8
            or platform.python_implementation() != "CPython"):
        issues.append(f"需要 64 位 CPython {'.'.join(map(str, PYTHON_VERSION))}；当前 {platform.python_version()}")
    for name in ("numpy", "scipy", "threadpoolctl", "psychopy.visual", "psychopy.event"):
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
    """通过写入、同步、原子替换和回读验证保存目录。"""
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


@dataclass
class Config:
    """集中保存会话、采集、解码、显示和持久化配置。"""

    session_mode: str = "cued"
    free_prepare_s: float = 1.0
    protocol: str = "standard_2s"
    blocks: int = 3
    random_seed: int = 20260928
    cue_s: float = 2.0
    cued_settle_s: float = 0.5
    response_delay_s: float = 0.14
    blank_min_s: float = 0.5
    feedback_s: float = 0.5
    cued_rest_every: int = 40
    cued_rest_s: float = 30.0
    block_rest_s: float = 180.0
    cued_offline_max_delay_s: float = 0.30
    max_consecutive_invalid: int = 3

    eeg_stream_name: Optional[str] = "obci_eeg1"
    eeg_stream_type: str = "EEG"
    timestamp_mode: str = "lsl"
    continue_on_timestamp_anomaly: bool = True
    timestamp_max_jitter_s: float = 0.05
    timestamp_jitter_grace_s: float = 0.5
    timestamp_max_discontinuity_s: float = 0.25
    timestamp_slew_rate_s_per_s: float = 0.01
    channel_indices: tuple[int, ...] = (7, 6, 5, 4, 3, 1, 2, 0)
    channel_positions: tuple[str, ...] = ("P1", "P2", "PO3", "POz", "PO4", "O1", "Oz", "O2")
    buffer_s: float = 60.0
    connect_timeout_s: float = 10.0
    startup_timeout_s: float = 45.0
    warmup_s: float = 10.0
    data_wait_timeout_s: float = 8.0
    clock_refresh_s: float = 1.0
    max_clock_step_s: float = 0.002
    clock_slew_rate_s_per_s: float = 0.001
    max_receive_age_s: float = 2.0
    max_gap_factor: float = 1.5
    hard_gap_factor: float = 2.0
    rate_tolerance: float = 0.002
    hard_rate_tolerance: float = 0.005
    max_warning_gaps: int = 1
    max_warning_frame_anomalies: int = 1
    min_valid_channels: int = 6
    max_warning_bad_channels: int = 2
    quality_max_abs: Optional[float] = None
    quality_rail_min: Optional[float] = None
    quality_rail_max: Optional[float] = None
    quality_rail_margin: float = 0.0
    quality_max_clip_fraction: float = 0.0
    quality_jump_z: float = 12.0
    rejection_enabled: bool = False
    rejection_min_score: Optional[float] = None
    rejection_min_margin: Optional[float] = None
    record_root: str = "session_records"
    input_unit: str = "AUTO"
    upstream_filter_description: str = "OpenBCI GUI TimeSeriesRaw（用户指定；LSL元数据未必可核实）"
    upstream_lowpass_hz: Optional[float] = None
    device_timestamp_lag_s: float = 0.0
    sample_counter_channel: Optional[int] = None
    hardware_timestamp_channel: Optional[int] = None
    photodiode_channel: Optional[int] = None
    trigger_channel: Optional[int] = None
    timing_calibration_file: Optional[str] = None

    target_fs: float = 250.0
    n_harmonics: int = 5
    filter_bank_profile: str = "m3"
    weight_a: float = 1.25
    weight_b: float = 0.25
    notch_hz: Optional[float] = 50.0
    notch_mode: str = "epoch"
    notch_history_s: float = 4.0
    line_regression_hz: Optional[float] = None
    cca_regularization: float = 1e-10
    participant_id: Optional[str] = None
    calibration_file: Optional[str] = None
    rejection_calibration_id: Optional[str] = None

    full_screen: bool = True
    screen_index: int = 0
    window_size: tuple[int, int] = (1920, 1080)
    viewing_distance_cm: float = 70.0
    screen_diagonal_inches: Optional[float] = None
    frame_long_factor: float = 1.5
    frame_short_factor: float = 0.5
    frame_rate_tolerance: float = 0.005
    marker_stream_name: str = "FBCCA_40_Keyboard_Events"
    output_box_height_px: float = 72.0
    output_box_margin_px: float = 24.0

    @property
    def window_s(self) -> float:
        """返回当前实验协议对应的 EEG 分析窗长度（秒）。"""
        presets = {"standard_5s": 5.0, "standard_2s": 2.0, "debug_2s": 2.0,
                   "paper_window_1p25s": 1.25, "paper_offline_5s": 5.0}
        if self.protocol not in presets:
            raise ValueError(f"未知 protocol: {self.protocol}; 可选 {tuple(presets)}")
        return presets[self.protocol]

    @property
    def startup_buffer_s(self) -> float:
        """返回启动检查前必须积累的最短数据时长（秒）。"""
        history_s = self.notch_history_s if self.notch_mode == "history" else 0.0
        return max(self.warmup_s, history_s + 2.1)

    @property
    def stimulus_s(self) -> float:
        """返回响应延迟与分析窗之和，即单次刺激时长。"""
        return self.response_delay_s + self.window_s

    @property
    def cued_stimulus_s(self) -> float:
        """返回模式2为离线延迟分析保留的完整闪烁时长。"""
        return self.window_s + max(self.response_delay_s, self.cued_offline_max_delay_s)

    @property
    def filter_bands(self) -> tuple[tuple[float, float], ...]:
        """返回当前 FBCCA 配置使用的滤波器组频带。"""
        profiles = {"m3": M3_BANDS, "line_robust": LINE_ROBUST_BANDS}
        if self.filter_bank_profile not in profiles:
            raise ValueError("filter_bank_profile 必须为 m3 或 line_robust")
        return profiles[self.filter_bank_profile]

    @property
    def rejection_thresholds_configured(self) -> bool:
        """判断自由输入模式是否完整启用了双阈值拒识。"""
        return bool(
            self.rejection_enabled
            and self.rejection_min_score is not None
            and self.rejection_min_margin is not None
            and math.isfinite(float(self.rejection_min_score))
            and math.isfinite(float(self.rejection_min_margin))
        )

    def validate(self) -> None:
        """验证所有配置项及跨字段约束，不合法时抛出 ``ValueError``。"""
        _ = self.window_s
        if self.calibration_file and not (self.participant_id and self.participant_id.strip()):
            raise ValueError("加载个人模型必须设置 participant_id / --participant")
        if self.notch_mode not in ("epoch", "history"):
            raise ValueError("notch_mode 必须为 epoch 或 history")
        if (not math.isfinite(self.notch_history_s) or self.notch_history_s <= 0):
            raise ValueError("notch_history_s 必须为有限正数")
        if self.notch_mode == "history":
            if self.notch_hz is None or not math.isfinite(self.notch_hz) or not 0 < self.notch_hz < self.target_fs / 2:
                raise ValueError("history 模式必须设置有效 notch_hz")
            if self.calibration_file:
                raise ValueError("history 模式目前仅支持基础FBCCA；旧个人模板使用epoch预处理，不能混用")
            if self.line_regression_hz is not None:
                raise ValueError("history 模式不能叠加未验证的单窗工频回归")
            if self.rejection_enabled:
                raise ValueError("history 模式的拒识分数尚未标定，不能套用epoch模式阈值")
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
                     "cued_rest_s", "block_rest_s", "cued_offline_max_delay_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须非负且有限")
        for name in ("connect_timeout_s", "startup_timeout_s", "warmup_s", "data_wait_timeout_s",
                     "buffer_s", "clock_refresh_s", "max_clock_step_s", "max_receive_age_s",
                     "viewing_distance_cm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.timestamp_mode == "lsl" and self.startup_timeout_s <= self.startup_buffer_s:
            raise ValueError("LSL启动超时必须长于启动检查所需数据时长")
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
        sync_indices = []
        for name in ("sample_counter_channel", "hardware_timestamp_channel",
                     "photodiode_channel", "trigger_channel"):
            index = getattr(self, name)
            if index is not None:
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"{name} 必须为空或从0开始的通道整数索引")
                if index in self.channel_indices or index in sync_indices:
                    raise ValueError(f"{name} 不得与分类EEG或其他同步通道重叠")
                sync_indices.append(index)
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
        if (not math.isfinite(self.hard_rate_tolerance)
                or not self.hard_rate_tolerance >= self.rate_tolerance):
            raise ValueError("hard_rate_tolerance 必须有限且大于或等于 rate_tolerance")
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
        history_s = self.notch_history_s if self.notch_mode == "history" else 0.0
        if self.buffer_s <= self.window_s + self.response_delay_s + self.data_wait_timeout_s + 2 + history_s:
            raise ValueError("内存缓冲区过短，可能覆盖仍待分类的 EEG")
        _validate_harmonics(self.n_harmonics, BENCHMARK_FREQUENCIES_HZ, self.target_fs)
        if self.upstream_lowpass_hz is not None:
            if not math.isfinite(self.upstream_lowpass_hz) or self.upstream_lowpass_hz < band_high:
                raise ValueError(f"已知发布端低通小于{band_high:g} Hz，不能使用当前频带")


CONFIG = Config()


def configure_fbcca_profile(cfg: Config, profile: str) -> Config:
    """同时选择子带、谐波和论文融合权重，避免只切换M3频带却仍用3次谐波。

    窗长和刺激流程独立于算法配置；m3不表示设备、空间布局和时序已严格复现。
    原始采样率在接收/分类入口校验，不能通过升采样绕过Nyquist限制。
    """
    harmonics = {"m3": 5, "line_robust": 3}
    if profile not in harmonics:
        raise ValueError("FBCCA配置必须为 m3 或 line_robust")
    configured = replace(cfg, filter_bank_profile=profile,
                         n_harmonics=harmonics[profile], weight_a=1.25, weight_b=.25,
                         line_regression_hz=None)
    configured.validate()
    return configured


@dataclass(frozen=True)
class Target:
    """描述一个键盘目标及其网格位置和刺激频率。"""

    class_id: int
    symbol: str
    row: int
    col: int
    frequency_hz: float
    phase_rad: float = 0.0

    @property
    def display_label(self) -> str:
        """返回适合显示在键帽上的文本。"""
        return {"SPACE": "", "BACK": "<-"}.get(self.symbol, self.symbol)


KEY_ROWS = (
    tuple("1234567890"),
    tuple("QWERTYUIOP"),
    (*tuple("ASDFGHJKL"), "BACK"),
    ("SPACE", *tuple("ZXCVBNM,.")),
)
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
LINE_ROBUST_BANDS = tuple((float(8 * n - 2), 48.0) for n in range(1, 4))
DEFAULT_WINDOW_S = 2.0
TARGET_FS = 250.0
N_HARMONICS = 5
LAST_SESSION: Optional[dict[str, Any]] = None


class InvalidTrial(RuntimeError):
    """数据/时序未通过质量检查；必须计入无效试次，而不是静默跳过。"""

    def __init__(self, message: str, *, epoch: Optional["Epoch"] = None,
                 code: str = "ACQUISITION"):
        """保存失败原因、可选的部分事件窗及标准故障代码。"""
        super().__init__(message)
        self.epoch = epoch
        self.code = code


class DegenerateSignalError(ValueError):
    """没有足够的变化信号，不能产生有意义的 CCA 结果。"""


class AbortSession(RuntimeError):
    """用户取消或急停。"""


class PauseSelection(RuntimeError):
    """自由输入暂停：立即停止闪烁，取消尚未提交的选择；不关闭EEG采集。"""


def _normalise_channel_indices(channel_indices: Sequence[int]) -> np.ndarray:
    """规范化并验证非负且不重复的 EEG 通道索引。"""
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
    """按内部统一的“通道 × 采样点”布局验证并返回 EEG 矩阵。"""
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
    """验证采样率为正数，并在需要时检查奈奎斯特上限。"""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须为有限正数")
    if upper_hz is not None and fs <= 2 * upper_hz:
        raise ValueError(f"原始/处理采样率必须大于{2 * upper_hz:g} Hz，以支持{upper_hz:g} Hz子带上限")


def _validate_filter_bands(filter_bands: Optional[Sequence[Sequence[float]]],
                           fs: float, target_fs: float) -> tuple[tuple[float, float], ...]:
    """验证滤波器组频带及原始、目标采样率。"""
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
    """验证谐波数，并保证最高参考谐波低于奈奎斯特频率。"""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("谐波数必须为正整数")
    if not np.isfinite(fs) or fs <= 0 or np.max(frequencies) * n >= fs / 2:
        raise ValueError("最高参考谐波必须低于Nyquist频率")


def _validate_line_regression_hz(value: Optional[float], fs: float) -> None:
    """验证可选的工频回归频率。"""
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
    """描述接收端滤波前的窄带功率，但不据此拒绝 EEG。

    汉宁窗周期图的功率比与通道单位无关。通道中位比例不低于 50% 仅表示
    输入由工频成分主导，并非经过验证的 SSVEP 质量或拒识阈值，也不能据此
    判定物理噪声源。关闭接收端陷波时仍观察默认的 49--51 Hz 频带；分母频带
    上限为 90 Hz 与奈奎斯特频率中的较小值，且不含奈奎斯特频点。分母频带
    不同时，所得比例不可直接比较。
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
    """对输入行进行标准化和协方差白化。"""
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization 必须非负且有限")
    z, _ = _normalised_rows(data)
    covariance = z @ z.T / (z.shape[1] - 1)
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
    """创建并缓存只读的陷波器和 Chebyshev-I 滤波器组系数。"""
    from scipy.signal import cheby1, iirnotch
    notch = None if notch_hz is None else iirnotch(notch_hz, 30.0, fs=fs)
    bank = tuple(cheby1(4, .5, [low, high], btype="bandpass", fs=fs, output="sos")
                 for low, high in bands)
    for coefficient in (*(() if notch is None else notch), *bank):
        coefficient.setflags(write=False)
    return notch, bank


def _slice_eeg_window(data: np.ndarray, fs: float, window_s: float,
                      onset_s: float, *, short_message: str) -> np.ndarray:
    """在调用方完成布局和采样率检查后，截取一个完整 EEG 窗。"""
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


@dataclass
class NotchContext:
    """真实历史原始样本及其时间轴；末端包含分析窗但不含 epoch_end 后样本。

    构造时仅验证结构，允许保留含 NaN 或异常时间戳的原始块供诊断。
    source_offset 指向原事件窗左插值支撑样本，之前是要求的历史样本。
    """
    data: np.ndarray
    local_timestamps: np.ndarray
    uniform_timestamps: np.ndarray
    source_offset: int
    epoch_end: float
    requested_history_samples: int

    def __post_init__(self) -> None:
        """规范化数组字段，并立即验证历史上下文维度。"""
        self.data = _validate_eeg_ct(
            self.data, name="NotchContext.data", allow_empty_samples=True)
        self.local_timestamps = np.asarray(self.local_timestamps, dtype=float)
        self.uniform_timestamps = np.asarray(self.uniform_timestamps, dtype=float)
        self.validate_dimensions()

    def validate_dimensions(self) -> None:
        """验证原始数据、时间轴、偏移量和历史样本数的一致性。"""
        if self.local_timestamps.ndim != 1 or len(self.local_timestamps) != self.data.shape[1]:
            raise ValueError("NotchContext.local_timestamps 必须与原始块采样点数相同")
        if self.uniform_timestamps.ndim != 1:
            raise ValueError("NotchContext.uniform_timestamps 必须为一维")
        for name in ("source_offset", "requested_history_samples"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"NotchContext.{name} 必须为非负整数")
        if self.source_offset > self.data.shape[1]:
            raise ValueError("NotchContext.source_offset 不得超出原始块")


def apply_history_notch(window: np.ndarray, fs: float, notch_hz: float,
                        context: NotchContext) -> tuple[np.ndarray, dict[str, Any]]:
    """真实历史块按声明采样率陷波，再插回原网格；不归一化、不做子带。

    使用 SciPy iirnotch(Q=30) 和 filtfilt 默认 odd/padlen=9，与固定离线
    实验一致。拒绝未来真实样本；右端反射填充仍可能带来滤波边缘瞬态。
    """
    from scipy.signal import filtfilt, iirnotch

    _validate_fs(fs, None)
    if not isinstance(context, NotchContext):
        raise ValueError("历史陷波需要 NotchContext")
    context.validate_dimensions()
    x = _validate_eeg_ct(window, name="历史陷波原窗口")
    raw, stamps, grid = context.data, context.local_timestamps, context.uniform_timestamps
    if (not np.isfinite(x).all() or not np.isfinite(raw).all()
            or not np.isfinite(stamps).all() or not np.isfinite(grid).all()
            or not np.isfinite(context.epoch_end)):
        raise ValueError("历史陷波数据、时间轴及终点须全部有限")
    if (not isinstance(notch_hz, (int, float, np.integer, np.floating))
            or isinstance(notch_hz, (bool, np.bool_)) or not np.isfinite(notch_hz)
            or not 0 < notch_hz < fs / 2):
        raise ValueError("历史陷波频率须位于0与Nyquist之间")
    if x.shape[0] != raw.shape[0] or len(grid) != x.shape[1]:
        raise ValueError("历史陷波原始块通道数及均匀网格长度须与窗口一致")
    if context.source_offset != context.requested_history_samples or context.source_offset < 1:
        raise ValueError("历史陷波必须有完整请求历史，不得用缩短块静默回退")
    if len(stamps) - context.source_offset < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError("历史陷波时间戳须严格递增且有至少两个事件窗支撑样本")
    if np.any(stamps > context.epoch_end):
        raise ValueError("历史陷波上下文含 epoch_end 之后的真实样本")
    if len(grid) < 32 or np.any(np.diff(grid) <= 0):
        raise ValueError("历史陷波需要至少32点的完整等间隔分析窗口")
    time_atol = max(1e-10, 8 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(grid)))))
    expected_grid = grid[0] + np.arange(len(grid), dtype=float) / fs
    duration = context.epoch_end - grid[0]
    count_atol = max(1e-9, time_atol * fs)
    expected_count = int(math.ceil(duration * fs - count_atol))
    if (not np.allclose(grid, expected_grid, rtol=0, atol=time_atol)
            or duration <= 0 or grid[-1] >= context.epoch_end or len(grid) != expected_count):
        raise ValueError("历史陷波网格或终点不符合原窗口采样定义")
    source_stamps = stamps[context.source_offset:]
    if grid[0] < source_stamps[0] or grid[-1] > source_stamps[-1]:
        raise ValueError("历史陷波事件窗支撑不足，不允许端点外推或补未来样本")
    rebuilt = np.asarray([
        np.interp(grid, source_stamps, channel)
        for channel in raw[:, context.source_offset:]], dtype=float)
    row_scale = np.maximum(np.max(np.abs(x), axis=1, keepdims=True),
                           np.max(np.abs(rebuilt), axis=1, keepdims=True))
    mismatch = np.abs(rebuilt - x)
    tolerance = 1e-10 * np.abs(x) + 128 * np.finfo(float).eps * row_scale
    if np.any(mismatch > tolerance):
        raise ValueError("历史陷波上下文与原始均匀窗口不匹配")
    bn, an = iirnotch(float(notch_hz), 30.0, fs=fs)
    padlen = 3 * max(len(bn), len(an))
    if raw.shape[1] <= padlen:
        raise ValueError("历史陷波原始块过短，无法使用规定的反射边界")
    filtered = filtfilt(bn, an, raw, axis=-1)
    uniform = np.asarray([
        np.interp(grid, source_stamps, channel)
        for channel in filtered[:, context.source_offset:]], dtype=float)
    if not np.isfinite(uniform).all():
        raise ValueError("历史陷波产生非有限结果")
    diagnostics = {
        "mode": "history", "notch_mode": "history",
        "notch_hz": float(notch_hz), "notch_q": 30.0,
        "native_fs_hz": float(fs), "context_samples": int(raw.shape[1]),
        "history_samples": int(context.source_offset),
        "requested_history_samples": int(context.requested_history_samples),
        "history_nominal_s": float(context.source_offset / fs),
        "context_first_timestamp": float(stamps[0]),
        "context_last_timestamp": float(stamps[-1]),
        "source_first_timestamp": float(source_stamps[0]),
        "epoch_end": float(context.epoch_end), "future_samples_used": 0,
        "uniform_samples": int(len(grid)), "filtfilt_padtype": "odd",
        "filtfilt_padlen_samples": int(padlen),
        "raw_window_reconstruction_max_abs_error": float(np.max(mismatch)),
        "normalization": "none; classifier normalizes after filtered interpolation",
        "processing_order": "native raw history notch -> original uniform grid -> classifier",
    }
    return uniform, diagnostics


def _preprocess_validated_window(window: np.ndarray, fs: float, window_s: float,
                                 target_fs: float, notch_hz: Optional[float],
                                 bands: tuple[tuple[float, float], ...]
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """对已截取且验证过的窗口进行重采样、陷波和滤波器组处理。"""
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
    _normalised_rows(window)
    return _preprocess_validated_window(window, fs, window_s, target_fs, notch_hz, bands)


def classify_eeg_window(data_ch_samples: np.ndarray, fs: float,
                        window_s: float = DEFAULT_WINDOW_S, onset_s: float = 0.0,
                        target_frequencies_hz: Optional[Sequence[float]] = None,
                        n_harmonics: int = N_HARMONICS, *, target_fs: float = TARGET_FS,
                        notch_hz: Optional[float] = 50.0, a: float = 1.25, b: float = .25,
                        regularization: float = 1e-10, min_valid_channels: int = 1,
                        filter_bands: Optional[Sequence[Sequence[float]]] = None,
                        line_regression_hz: Optional[float] = None,
                        calibration: Optional[PersonalCalibration] = None,
                        notch_context: Optional[NotchContext] = None) -> dict:
    """只接收EEG/算法参数，不接受真实目标，防止标签泄漏。

    FBCCA和标准CCA共享完全相同的EEG窗、通道、参考和第一子带。
    分数不是概率；本方法仍强制选择最高分，没有空闲检测或自由输入可靠性保证。
    """
    frequencies = (BENCHMARK_FREQUENCIES_HZ if target_frequencies_hz is None
                   else np.asarray(target_frequencies_hz, float))
    bands = _validate_filter_bands(filter_bands, fs, target_fs)
    _validate_harmonics(n_harmonics, frequencies, fs)
    _validate_line_regression_hz(line_regression_hz, min(fs, target_fs))
    if notch_context is not None and calibration is not None:
        raise ValueError("history 模式不能套用使用epoch预处理的个人模板")
    if notch_context is not None and line_regression_hz is not None:
        raise ValueError("history 模式不能叠加单窗工频回归")
    data = _validate_eeg_ct(data_ch_samples, name="EEG")
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(onset_s) or onset_s < 0:
        raise ValueError("EEG须二维，窗长为正，偏移非负")
    window = _slice_eeg_window(data, fs, window_s, onset_s,
                               short_message="EEG时间窗覆盖不足")
    notch_processing = {"mode": "epoch", "notch_hz": notch_hz, "Q": 30.0,
                        "order": "uniform raw window -> normalize -> epoch notch -> subbands"}
    history_notched_data = None
    original_active = None
    if notch_context is not None:
        _, original_active = _normalised_rows(window)
        window, notch_processing = apply_history_notch(window, fs, notch_hz, notch_context)
        history_notched_data = window.copy()
    if line_regression_hz is not None:
        window = _regress_line_noise(window, fs, line_regression_hz)
    normalized, active = _normalised_rows(window if original_active is None else window[original_active])
    if original_active is not None:
        active = original_active[active]
    if isinstance(min_valid_channels, bool) or not isinstance(min_valid_channels, int) or min_valid_channels < 1:
        raise ValueError("min_valid_channels须为正整数")
    if len(active) < min_valid_channels:
        raise DegenerateSignalError(f"有效变化通道仅{len(active)}个，至少需要{min_valid_channels}个")
    epoch_notch = notch_hz if notch_context is None else None
    window, bank = _preprocess_validated_window(normalized, fs, window_s, target_fs, epoch_notch, bands)
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
        "notch_processing": notch_processing,
        "history_notched_data": history_notched_data,
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
    """标准化校准矩阵，并要求所有通道均有效。"""
    z, active = _normalised_rows(x)
    if len(active) != len(x):
        raise DegenerateSignalError("个人模板要求所有校准通道有效")
    return z


def _calibration_whitener(x: np.ndarray, reg: float) -> np.ndarray:
    """计算个人校准特征使用的协方差白化矩阵。"""
    cov = x @ x.T / (x.shape[-1] - 1)
    cov += reg * np.trace(cov) / len(cov) * np.eye(len(cov))
    return _inverse_sqrt_covariance(cov, reg)


def _projected_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """按最后一个轴计算成对投影信号的相关系数。"""
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
        """验证并冻结个人模板，同时预计算评分所需的空间滤波量。"""
        self._metadata = json.loads(json.dumps(metadata, allow_nan=False))
        m = self._metadata
        if m.get("schema_version") != self.schema_version or m.get("algorithm") != "FB-eCCA-4":
            raise ValueError("不支持的个人模型版本")
        if not isinstance(m.get("participant_id"), str) or not m["participant_id"].strip():
            raise ValueError("个人模型缺少受试者编号")
        s = m["settings"]
        self._settings = s
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
        """返回模型元数据的深拷贝，避免外部修改内部状态。"""
        return json.loads(json.dumps(self._metadata))

    @classmethod
    def fit(cls, windows: Sequence[np.ndarray], labels: Sequence[int],
            fs: float | Sequence[float],
            cfg: Config, *, sources: Sequence[dict] = ()) -> PersonalCalibration:
        """从覆盖全部目标的已对齐校准窗训练个人 eCCA 模板。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持个人模板训练；2秒窗不能代替完整历史预处理")
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
        """确认运行配置与模型的受试者及预处理契约一致。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持个人模型；旧模板仅适用于 epoch 预处理")
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
        """确认一次解码调用的参数与模型训练设置完全一致。"""
        s = self._settings
        actual = [list(frequencies), fs, window_s, harmonics, notch, [list(v) for v in bands],
                  reg, line, a, b, channels]
        expected = [s["frequencies"], s["target_fs"], s["window_s"], s["n_harmonics"],
                    s["notch_hz"], s["filter_bands"], s["regularization"],
                    s["line_regression_hz"], s["a"], s["b"], len(s["channel_positions"])]
        if actual != expected:
            raise ValueError("分类参数与个人模型不匹配，拒绝套用模板")

    def score(self, bank: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算滤波器组 eCCA 融合分数及四类中间特征。"""
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
        """以原子 NPZ 文件保存模板、计数、元数据和完整性标识。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        SessionRecorder._atomic_npz(path, {
            "templates": self.templates, "counts": self.counts,
            "metadata_json": np.array(json.dumps(self._metadata, ensure_ascii=False, allow_nan=False)),
            "model_id": np.array(self.model_id),
        })

    @classmethod
    def load(cls, path: str | Path, cfg: Config) -> PersonalCalibration:
        """从 NPZ、会话模型 PKL 或会话 ZIP 加载并验证个人模型。"""
        if getattr(cfg, "notch_mode", "epoch") != "epoch":
            raise ValueError("history 陷波暂不支持加载个人模型；请使用无模板 FBCCA")
        path = Path(path)
        if path.suffix.lower() in (".pkl", ".zip"):
            class DataOnlyUnpickler(pickle.Unpickler):
                """禁止通过 pickle 反序列化任意 Python 类。"""

                def find_class(self, module: str, name: str) -> Any:
                    """拒绝所有需要导入类或全局对象的 pickle 条目。"""
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


@dataclass
class Epoch:
    """保存一个对齐后的 EEG 事件窗及其源时间轴和诊断信息。"""

    data: np.ndarray
    fs: float
    raw_timestamps: np.ndarray
    local_timestamps: np.ndarray
    clock_corrections: np.ndarray
    diagnostics: dict[str, Any]
    source_data: np.ndarray
    uniform_timestamps: np.ndarray
    source_timestamps: Optional[np.ndarray] = None
    notch_context: Optional[NotchContext] = None

    def __post_init__(self) -> None:
        """规范化数组字段，并验证数据与各时间轴的维度。"""
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
        """验证 C×T 数据矩阵及对应的一维时间轴。"""
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
        if self.notch_context is not None:
            self.notch_context.validate_dimensions()
            if self.notch_context.data.shape[0] != self.source_data.shape[0]:
                raise ValueError("Epoch.notch_context 与 source_data 通道数不一致")


def _parse_stream_metadata(info: Any, indices: Sequence[int]) -> dict:
    """解析 LSL 流 XML 元数据并提取选定通道与发布端滤波信息。"""
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
    filter_nodes = []
    for element in root.findall("./desc//*"):
        if any(s in element.tag.lower() for s in ("filter", "lowpass", "highpass", "notch")):
            text = " ".join(element.itertext()).strip()
            if text:
                filter_nodes.append(f"{element.tag}: {text}")
    def value(name: str, legacy_name: str) -> Any:
        """兼容读取新版属性和旧版零参数访问器。"""
        item = getattr(info, name, None)
        return item if item is not None else getattr(info, legacy_name)()
    return {"name": value("name", "name"), "type": value("stype", "type"),
            "source_id": value("source_id", "source_id"),
            "channel_count": value("n_channels", "channel_count"),
            "nominal_fs": float(value("sfreq", "nominal_srate")),
            "selected_channels": selected, "publisher_filter_metadata": filter_nodes or ["UNKNOWN"],
            "raw_xml": xml}


class MNEStreamWindow:
    """从 MNE-LSL 环形缓冲读取事件窗；生产环境与 acquire 共用同一把锁。"""

    timestamp_mode = "lsl"

    def __init__(self, stream: Any, indices: Sequence[int], device_lag_s: float,
                 *, lock: Any = None):
        """绑定 MNE-LSL 流、分类通道和可选的设备时延补偿。"""
        self.stream = stream
        self.indices = list(indices)
        self.n_channels = len(indices)
        self.device_lag_s = device_lag_s
        self.total_received = 0
        self.lock = threading.RLock() if lock is None else lock

    def snapshot(self) -> tuple[np.ndarray, ...]:
        """在线程锁内复制当前环形缓冲及其处理后时间戳。"""
        with self.lock:
            data, timestamps = self.stream.get_data(picks=self.indices)
            x = np.asarray(data, dtype=float).copy()
            received = np.asarray(timestamps, dtype=float).copy()
        if (received.ndim != 1
                or x.shape != (self.n_channels, len(received))):
            raise InvalidTrial("MNE-LSL缓冲数据与时间戳形状不一致")
        nonempty = np.flatnonzero(received != 0)
        x, received = x[:, nonempty], received[nonempty]
        local = received - self.device_lag_s
        corrections = np.zeros(len(received), dtype=float)
        valid = np.isfinite(x).all(axis=0)
        return x, received, local, corrections, valid

    @property
    def latest(self) -> float:
        """返回缓冲区最新样本的本地时间；无数据时返回负无穷。"""
        with self.lock:
            _, received = self.stream.get_data(
                winsize=1 / float(self.stream.info["sfreq"]), picks=[self.indices[0]])
            return float(received[-1] - self.device_lag_s) if len(received) and received[-1] else -math.inf

    def window_snapshot(self, start: float, end: float) -> tuple[np.ndarray, ...]:
        """从一次缓冲快照中返回覆盖指定半开区间的支撑样本。"""
        return self._window_from_snapshot(self.snapshot(), start, end)

    @staticmethod
    def _window_from_snapshot(snapshot: tuple[np.ndarray, ...], start: float,
                              end: float) -> tuple[np.ndarray, ...]:
        """从已复制的快照中截取事件窗及左右插值支撑点。"""
        x, received, local, corrections, valid = snapshot
        if len(local) < 2 or local[0] > start or local[-1] < end:
            raise InvalidTrial("事件窗口数据未到齐或已被MNE-LSL缓冲覆盖")
        left = int(np.searchsorted(local, start, side="right") - 1)
        right = int(np.searchsorted(local, end, side="left"))
        support = slice(left, right + 1)
        return (x[:, support], received[support], local[support],
                corrections[support], valid[support], received[support].copy())


    def epoch(self, start: float, duration_s: float, nominal_fs: float,
              max_gap_factor: float = 1.5, rate_tolerance: float = .002,
              hard_gap_factor: float = 2.0, hard_rate_tolerance: float = .005,
              max_warning_gaps: int = 1, *, notch_history_s: float = 0.0) -> Epoch:
        """校验指定事件窗的时间轴，并插值到声明采样率的均匀网格。"""
        _validate_fs(nominal_fs, None)
        if not np.isfinite(start) or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("事件时间/窗口长度不合法")
        if not np.isfinite(notch_history_s) or notch_history_s < 0:
            raise ValueError("陷波历史长度必须非负且有限")
        end = start + duration_s
        snapshot = self.snapshot()
        x, raw, local, corrections, valid, source = self._window_from_snapshot(snapshot, start, end)
        grid = start + np.arange(sample_count(duration_s, nominal_fs)) / nominal_fs
        notch_context = None
        if notch_history_s > 0:
            requested_history_samples = int(round(notch_history_s * nominal_fs))
            if requested_history_samples < 1:
                raise ValueError("陷波历史长度不足一个采样点")
            all_data, _, all_local, _, _ = snapshot
            source_left = int(np.searchsorted(all_local, start, side="right") - 1)
            history_left = max(0, source_left - requested_history_samples)
            history_end = int(np.searchsorted(all_local, end, side="right"))
            notch_context = NotchContext(
                data=all_data[:, history_left:history_end].copy(),
                local_timestamps=all_local[history_left:history_end].copy(),
                uniform_timestamps=grid.copy(), source_offset=source_left - history_left,
                epoch_end=float(end), requested_history_samples=requested_history_samples)
        diagnostics = {
            "timestamp_mode": self.timestamp_mode,
            "nominal_fs_hz": float(nominal_fs),
            "requested_window_start": float(start), "requested_window_end": float(end),
            "source_support_samples": len(local),
            "source_samples_within_window": int(np.count_nonzero((local >= start) & (local < end))),
            "expected_uniform_samples": sample_count(duration_s, nominal_fs),
            "rate_warning_tolerance": float(rate_tolerance),
            "rate_hard_tolerance": float(hard_rate_tolerance),
        }
        if notch_context is not None:
            history_stamps = notch_context.local_timestamps
            diagnostics["notch_history"] = {
                "passed": False, "checked": False,
                "requested_history_samples": notch_context.requested_history_samples,
                "available_history_samples": notch_context.source_offset,
                "raw_block_samples": len(history_stamps),
                "source_start": float(history_stamps[0]) if len(history_stamps) else None,
                "source_end": float(history_stamps[-1]) if len(history_stamps) else None,
                "epoch_end": float(end),
                "reason": "原分析窗须先通过校验；历史尚未判定合格",
            }
        def failure_epoch(message: str) -> Epoch:
            """构造保留源数据和失败诊断的空事件窗。"""
            return Epoch(
                data=np.empty((self.n_channels, 0), dtype=float), fs=nominal_fs,
                raw_timestamps=raw.copy(), local_timestamps=local.copy(),
                clock_corrections=corrections.copy(),
                diagnostics={**diagnostics, "hard_reason": message, "uniform_samples": 0},
                source_data=x.copy(), uniform_timestamps=np.empty(0, dtype=float),
                source_timestamps=source.copy(), notch_context=notch_context)
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
        warning_gap_count = int(np.count_nonzero(gap_mask))
        hard_gap_count = int(np.count_nonzero(hard_gap_mask))
        if hard_gap_count:
            message = ("LSL处理后窗口存在明确缺样级时间戳间隔："
                       f"warning_gaps={warning_gap_count}, hard_gaps={hard_gap_count}")
            raise InvalidTrial(message, epoch=failure_epoch(message))
        if warning_gap_count:
            level = "较多" if warning_gap_count > max_warning_gaps else "轻微"
            warnings.append(f"存在{warning_gap_count}个{level}异常采样间隔；未达到明确缺样阈值，继续处理")
        measured_fs = (len(local) - 1) / (local[-1] - local[0])
        source_measured_fs = (len(source) - 1) / (source[-1] - source[0])
        local_rate_error = abs(measured_fs / nominal_fs - 1)
        source_rate_error = abs(source_measured_fs / nominal_fs - 1)
        diagnostics.update(
            timestamp_estimated_fs=float(measured_fs),
            source_clock_estimated_fs=float(source_measured_fs),
            local_rate_error_fraction=float(local_rate_error),
            source_rate_error_fraction=float(source_rate_error),
            timestamp_nominal_grid_offset_peak_to_peak_s=float(np.ptp(
                local - local[0] - np.arange(len(local)) / nominal_fs)),
            sample_count_interpretation=("count of received samples within postprocessed timestamps; "
                                         "does not establish device sample loss"),
        )
        maximum_rate_error = max(local_rate_error, source_rate_error)
        if maximum_rate_error > hard_rate_tolerance:
            warnings.append(
                f"时间戳采样率偏差较大但仅警告：source={source_measured_fs:.3f}Hz, "
                f"local={measured_fs:.3f}Hz, declared={nominal_fs:g}Hz")
        elif maximum_rate_error > rate_tolerance:
            warnings.append(
                f"时间戳采样率轻微偏差：source={source_measured_fs:.3f}Hz, local={measured_fs:.3f}Hz")
        _validate_fs(min(measured_fs, source_measured_fs), None)
        if grid.size < 32 or grid[-1] > local[-1] or grid[0] < local[0]:
            message = "窗口不满足重采样时间覆盖"
            raise InvalidTrial(message, epoch=failure_epoch(message))
        uniform = np.asarray([np.interp(grid, local, channel) for channel in x], dtype=float)
        epoch = Epoch(data=uniform, fs=nominal_fs, raw_timestamps=raw.copy(),
                     local_timestamps=local.copy(), clock_corrections=corrections.copy(), diagnostics={
            **diagnostics,
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
            "warning_gap_count": warning_gap_count,
            "hard_gap_count": hard_gap_count,
            "rate_mismatch_handling": "warning_only",
        }, source_data=x.copy(), uniform_timestamps=grid.copy(), source_timestamps=source.copy(),
                     notch_context=notch_context)
        if notch_context is not None:
            self._check_notch_history(epoch, nominal_fs, max_gap_factor, rate_tolerance,
                                      hard_gap_factor, hard_rate_tolerance, max_warning_gaps)
        return epoch

    @staticmethod
    def _check_notch_history(epoch: Epoch, nominal_fs: float, max_gap_factor: float,
                             rate_tolerance: float, hard_gap_factor: float,
                             hard_rate_tolerance: float, max_warning_gaps: int) -> None:
        """只检查已复制的历史；失败仍保留原分析窗及可获得的原始历史。"""
        context = epoch.notch_context
        assert context is not None
        stamps = context.local_timestamps
        diagnostics = epoch.diagnostics["notch_history"] = {
            "passed": False, "checked": True, "nominal_fs_hz": float(nominal_fs),
            "requested_history_samples": context.requested_history_samples,
            "available_history_samples": context.source_offset,
            "raw_block_samples": len(stamps),
            "source_offset": context.source_offset,
            "source_start": float(stamps[0]) if len(stamps) else None,
            "source_end": float(stamps[-1]) if len(stamps) else None,
            "epoch_end": context.epoch_end,
            "max_gap_factor": float(max_gap_factor), "hard_gap_factor": float(hard_gap_factor),
            "max_warning_gaps": int(max_warning_gaps),
            "rate_warning_tolerance": float(rate_tolerance),
            "rate_hard_tolerance": float(hard_rate_tolerance),
            "warnings": [],
            "scope": "LSL-postprocessed timestamps; device sample loss and physical timing unverified",
        }

        def reject(message: str) -> None:
            """记录历史检查失败原因并抛出携带事件窗的异常。"""
            diagnostics["hard_reason"] = message
            epoch.diagnostics["hard_reason"] = message
            raise InvalidTrial(message, epoch=epoch)

        if context.source_offset != context.requested_history_samples:
            reject("陷波历史不足或已被环形缓冲覆盖；不缩短历史或补零")
        if (len(stamps) < 3 or not np.isfinite(stamps).all()
                or not np.isfinite(context.data).all()):
            reject("陷波历史含NaN/Inf或样本不足")
        intervals = np.diff(stamps)
        if np.any(intervals <= 0):
            reject("陷波历史时间戳不严格递增")
        if (np.any(stamps > context.epoch_end)
                or context.source_offset >= len(stamps)
                or stamps[context.source_offset] > context.uniform_timestamps[0]
                or stamps[-1] < context.uniform_timestamps[-1]):
            reject("陷波历史在窗口终点前的样本不能覆盖完整分析网格；不使用未来样本或外推")
        gap_mask = intervals > max_gap_factor / nominal_fs
        hard_gap_mask = intervals >= hard_gap_factor / nominal_fs
        observed_fs = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        rate_error = abs(observed_fs / nominal_fs - 1)
        diagnostics.update(
            timestamp_estimated_fs=float(observed_fs), rate_error_fraction=float(rate_error),
            max_interval_s=float(intervals.max()), warning_gap_count=int(np.count_nonzero(gap_mask)),
            hard_gap_count=int(np.count_nonzero(hard_gap_mask)),
            timestamp_nominal_grid_offset_peak_to_peak_s=float(np.ptp(
                stamps - stamps[0] - np.arange(len(stamps)) / nominal_fs)))
        warning_gap_count = int(np.count_nonzero(gap_mask))
        hard_gap_count = int(np.count_nonzero(hard_gap_mask))
        if hard_gap_count:
            reject(f"陷波历史存在{hard_gap_count}个明确缺样级时间戳间隔；不跨缺口补齐")
        if warning_gap_count:
            level = "较多" if warning_gap_count > max_warning_gaps else "轻微"
            diagnostics["warnings"].append(
                f"陷波历史存在{warning_gap_count}个{level}异常采样间隔；未达到明确缺样阈值，继续处理")
        if rate_error > hard_rate_tolerance:
            diagnostics["warnings"].append(
                f"陷波历史时间戳采样率偏差较大但仅警告：{observed_fs:.3f}Hz")
        elif rate_error > rate_tolerance:
            diagnostics["warnings"].append(f"陷波历史时间戳采样率轻微偏差：{observed_fs:.3f}Hz")
        diagnostics["rate_mismatch_handling"] = "warning_only"
        diagnostics["passed"] = True
        epoch.diagnostics["warnings"].extend(diagnostics["warnings"])

class ContinuousLSL:
    """管理 MNE-LSL 连续采集、缓冲、健康检查和事件窗读取。"""

    def __init__(self, cfg: Config):
        """根据配置初始化尚未连接的连续采集器。"""
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
        self._stream_lock = threading.RLock()
        self._acquisition_thread: Optional[threading.Thread] = None
        self.recorder: Optional[SessionRecorder] = None

    def _timestamp_summary(self) -> dict[str, Any]:
        """汇总当前 LSL 时间戳处理方式及其可验证范围。"""
        return {
            "mode": "mne_lsl_streamlsl",
            "flags": ["clocksync", "dejitter", "monotonize"],
            "clock_domain": "mne_lsl.lsl.local_clock",
            "manual_time_correction_applied": False,
            "sample_index_reconstruction": False,
            "acquisition_buffer": "MNE-LSL StreamLSL",
            "buffer_consistency": "manual acquire and copied snapshots share one lock",
            "original_outlet_timestamps_available": False,
            "saved_raw_timestamp_field": "legacy name; contains LSL-postprocessed timestamps",
            "device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
            "accepted_samples_total": self.buffer.total_received if self.buffer is not None else 0,
            "sample_integrity": ("configured device counter saved separately; semantics and continuity unverified"
                                 if self.cfg.sample_counter_channel is not None else
                                 "device sample counter unavailable in selected EEG stream"),
            "physical_latency": "unmeasured; GUI and display delays require independent validation",
        }

    def start(self) -> None:
        """连接唯一匹配的 EEG 流，启动采集线程并完成预热检查。"""
        if self.cfg.timestamp_mode != "lsl":
            raise RuntimeFault("ACQUISITION", "实时采集仅支持LSL内建时间处理；旧重建模式已停用")
        try:
            from mne_lsl.lsl import local_clock, resolve_streams
            from mne_lsl.stream import StreamLSL
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeFault("LSL_LIBRARY", "不能加载MNE-LSL/liblsl：" + str(exc)) from exc
        self.clock = local_clock
        name = self.cfg.eeg_stream_name or None
        stype = self.cfg.eeg_stream_type
        streams = resolve_streams(timeout=self.cfg.connect_timeout_s, name=name, stype=stype)
        if not streams:
            visible = resolve_streams(timeout=min(1.0, self.cfg.connect_timeout_s))
            details = "; ".join(
                f"{sinfo.name!r}(type={sinfo.stype!r}, channels={sinfo.n_channels}, "
                f"fs={sinfo.sfreq:g}Hz)"
                for sinfo in visible
            ) or "无"
            raise RuntimeFault("LSL_NOT_FOUND",
                f"未找到LSL EEG流（name={name or '任意'}, type={stype}）。当前可见流：{details}\n"
                "设备连接成功不代表已开启LSL输出。本程序通过LSL接收EEG，不直接连接USB设备。\n"
                "若使用OpenBCI GUI：先Start Data Stream，再在Networking选择LSL，"
                "数据选择TimeSeriesRaw、Type填EEG，点击Start LSL Stream，保持GUI运行后重试。"
            )
        if len(streams) != 1:
            names = ", ".join(repr(sinfo.name) for sinfo in streams)
            raise RuntimeFault("LSL_CONNECTION",
                f"找到多个匹配的EEG流（{names}），请确认OpenBCI GUI只发布一条{name or stype}流"
            )
        self.stream = StreamLSL(self.cfg.buffer_s, name=streams[0].name,
                                stype=streams[0].stype,
                                source_id=streams[0].source_id or None)
        try:
            self.stream.connect(acquisition_delay=None,
                                processing_flags=("clocksync", "dejitter", "monotize"),
                                timeout=self.cfg.connect_timeout_s)
            info = self.stream.sinfo
            self.fs = float(info.sfreq)
            _validate_fs(self.fs, max(high for _, high in self.cfg.filter_bands))
            if max(self.cfg.channel_indices) >= info.n_channels:
                raise RuntimeError("所选LSL通道索引超出实际流通道数")
            self.metadata = _parse_stream_metadata(info, self.cfg.channel_indices)
            self.metadata["all_channels"] = _parse_stream_metadata(
                info, range(info.n_channels))["selected_channels"]
            if self.recorder is not None:
                self.recorder.start_continuous(self.metadata)
            self.metadata["timestamp_processing"] = self._timestamp_summary()
            print("[时间轴] MNE-LSL StreamLSL后台采集与环形缓冲；"
                  "LSL内建时钟同步、去抖和单调化。\n"
                  f"收集{self.cfg.startup_buffer_s:g}秒数据用于启动检查，收齐即继续；"
                  "原始发布端时间戳在此模式下不可恢复。"
                  "设备和屏幕的物理延迟仍需独立测量。", flush=True)
            self.buffer = MNEStreamWindow(self.stream, self.cfg.channel_indices,
                                          self.cfg.device_timestamp_lag_s,
                                          lock=self._stream_lock)
            self.last_arrival = time.monotonic()
            self.stream.add_callback(self._observe_chunk)
            if self.recorder is not None:
                self.recorder.capture_active = True
            self._acquisition_thread = threading.Thread(
                target=self._acquire_loop, name="EEG-acquisition", daemon=True)
            self._acquisition_thread.start()
        except BaseException:
            self.close()
            raise
        try:
            deadline = time.monotonic() + self.cfg.startup_timeout_s
            while time.monotonic() < deadline:
                self.check_health()
                with self._stream_lock:
                    first = self.first_timestamp
                    last = self.last_timestamp
                    received = self.buffer.total_received
                if (first is not None and received > 2
                        and last - first >= self.cfg.startup_buffer_s):
                    self.estimated_fs = (received - 1) / (last - first)
                    startup_error = abs(self.estimated_fs / self.fs - 1)
                    if startup_error > self.cfg.hard_rate_tolerance:
                        print(f"[时间轴警告] 启动LSL采样率{self.estimated_fs:.3f}Hz，"
                              f"与声明值{self.fs:g}Hz偏差较大；仅警告并继续。", flush=True)
                    elif startup_error > self.cfg.rate_tolerance:
                        print(f"[时间轴警告] 启动LSL采样率{self.estimated_fs:.3f}Hz，"
                              f"与声明值{self.fs:g}Hz轻微偏差。", flush=True)
                    _validate_fs(self.estimated_fs, max(high for _, high in self.cfg.filter_bands))
                    return
                self.stop_event.wait(.02)
            raise RuntimeFault("ACQUISITION", "已找到LSL流，但启动阶段未收到足够的连续EEG数据")
        except BaseException:
            self.close()
            raise

    def _acquire_loop(self) -> None:
        """串行更新 MNE-LSL 缓冲；不在 acquire 的中间状态发布快照。"""
        while not self.stop_event.is_set():
            try:
                with self._stream_lock:
                    if self.stop_event.is_set():
                        return
                    if self.stream is None or not self.stream.connected:
                        raise RuntimeError("MNE-LSL采集流已断开")
                    self.stream.acquire()
                    if not self.stream.connected:
                        raise RuntimeError("MNE-LSL采集异常后断开")
                    if self.error is not None:
                        return
            except BaseException as exc:
                if self.error is None:
                    self.error = exc
                return
            self.stop_event.wait(.01)

    def _observe_chunk(self, data: np.ndarray, ts: np.ndarray, info: Any) -> tuple[np.ndarray, np.ndarray]:
        """保存回调所见的所有通道并检查健康状态；磁盘写入在独立线程完成。"""
        try:
            stamps = np.asarray(ts, dtype=float)
            if (data.ndim != 2 or len(stamps) != len(data)
                    or data.shape[1] <= max(self.cfg.channel_indices)):
                raise InvalidTrial("MNE-LSL采集块形状或时间戳无效")
            if len(stamps) and self.recorder is not None:
                self.recorder.record_continuous(data, stamps, self.clock())
            if not np.isfinite(stamps).all():
                raise InvalidTrial("MNE-LSL时间戳含NaN/Inf")
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
        """检查采集、记录线程、流连接状态和最近样本年龄。"""
        if self.recorder is not None:
            self.recorder.check_recording_health()
        if self.error is not None:
            raise RuntimeFault("ACQUISITION", f"采集线程异常：{self.error}") from self.error
        if self.stop_event.is_set():
            raise RuntimeFault("ACQUISITION", "采集已停止")
        with self._stream_lock:
            if self.stream is not None:
                if not self.stream.connected:
                    raise RuntimeFault("ACQUISITION", "MNE-LSL采集线程已断开")
                if time.monotonic() - self.last_arrival > self.cfg.max_receive_age_s:
                    raise RuntimeFault("ACQUISITION", "EEG流停止发送，已停止实验")

    def wait_epoch(self, onset: float, cancel: threading.Event) -> Epoch:
        """等待目标事件窗到齐，随后返回经过时间轴校验的事件窗。"""
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
            self.cfg.hard_rate_tolerance, self.cfg.max_warning_gaps,
            notch_history_s=(self.cfg.notch_history_s if self.cfg.notch_mode == "history" else 0.0))

    def close(self) -> None:
        """停止采集线程并安全断开 LSL 流；重复调用不会重复断开。"""
        self.stop_event.set()
        if self._acquisition_thread is not None:
            if self._acquisition_thread.ident is not None:
                self._acquisition_thread.join(timeout=max(2.0, self.cfg.connect_timeout_s))
            if self._acquisition_thread.is_alive():
                raise RuntimeFault("ACQUISITION", "EEG接收线程未及时退出；未并发断开仍在使用的流")
            self._acquisition_thread = None
        if self.recorder is not None:
            self.recorder.capture_active = False
        with self._stream_lock:
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
    notch_context = epoch.notch_context
    if cfg.notch_mode == "history":
        if (notch_context is None
                or notch_context.requested_history_samples != round(cfg.notch_history_s * epoch.fs)):
            raise InvalidTrial("history 模式缺少匹配的历史上下文，不回退到短窗陷波", epoch=epoch)
    elif notch_context is not None:
        raise InvalidTrial("epoch 模式收到history上下文，拒绝混用预处理设置", epoch=epoch)
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
    if cancel.is_set():
        raise AbortSession("已取消分类")
    started = time.monotonic()
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(limits=1):
            result = classify_eeg_window(epoch.data, epoch.fs, window_s=cfg.window_s, onset_s=0.0,
                target_frequencies_hz=BENCHMARK_FREQUENCIES_HZ, n_harmonics=cfg.n_harmonics,
                target_fs=cfg.target_fs, notch_hz=cfg.notch_hz, a=cfg.weight_a, b=cfg.weight_b,
                regularization=cfg.cca_regularization, min_valid_channels=cfg.min_valid_channels,
                filter_bands=cfg.filter_bands, line_regression_hz=cfg.line_regression_hz,
                calibration=getattr(collector, "calibration", None), notch_context=notch_context)
    except (ValueError, DegenerateSignalError, np.linalg.LinAlgError) as exc:
        if cfg.notch_mode != "history":
            raise
        epoch.diagnostics["notch_processing_error"] = str(exc)
        raise InvalidTrial(f"history预处理/分类失败：{exc}", epoch=epoch) from exc
    if cancel.is_set():
        raise AbortSession("已取消分类，未提交结果")
    epoch.diagnostics["notch_processing"] = result["notch_processing"]
    result.update(trial_id=trial_id, epoch=epoch, quality=quality,
                  computation_s=time.monotonic() - started)
    return result


@dataclass
class EventStamp:
    """记录一个实验事件及其试次、目标、详情和 LSL 时间戳。"""

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
        """创建标记流，并启动异步发送线程。"""
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
        """在调用线程立即记时，并将事件加入异步发送队列。"""
        if self.closed or event.timestamp is not None:
            raise RuntimeError("事件重复标记或事件模块已经关闭")
        event.timestamp = self.clock()
        self.events.append(event)
        self.pending.put(event)

    def _send(self) -> None:
        """持续发送排队事件，并保留首次发送异常供主线程检查。"""
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
        """通知标记线程退出并等待其结束。"""
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = RuntimeError("Marker发送线程未及时退出")


@dataclass
class TrialRecord:
    """保存单个试次从呈现到分类、质控和落盘的完整状态。"""

    trial_id: int
    block_id: int
    true_class: Optional[int]
    status: str = "started"
    reason: str = ""
    start: float = 0.0
    end: float = 0.0
    cue_onset: Optional[float] = None
    stimulus_onset: Optional[float] = None
    stimulus_offset: Optional[float] = None
    phase_times: list[tuple[str, float]] = field(default_factory=list)
    frame_flip_times: list[float] = field(default_factory=list)
    frame_lsl_times: list[float] = field(default_factory=list)
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
        """初始化空试次集合和自由输入文本。"""
        self.records: list[TrialRecord] = []
        self.ids: set[int] = set()
        self.typed_text = ""

    def apply_prediction(self, pred: int) -> None:
        """将预测类别对应的字符、空格或退格应用到输入文本。"""
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
        """校验已结束试次的状态与模式约束，并保证只提交一次。"""
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
    """返回有限数值的中位数；无可用值时返回 ``None``。"""
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
    if getattr(cfg, "notch_mode", "epoch") != "epoch":
        bundle["training_status"] = "unsupported"
        bundle["training_reason"] = (
            "history 陷波本版仅支持无模板 FBCCA；个人模板训练尚未接入完整历史预处理，未训练模型")
        bundle["decoder_settings"] = {
            **bundle["decoder_settings"],
            "notch_mode": getattr(cfg, "notch_mode", "epoch"),
            "notch_history_s": getattr(cfg, "notch_history_s", 4.0),
            "notch_Q": 30.0,
        }
        return bundle
    if missing:
        return bundle
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
    """汇总提示测试的逐目标频谱、呈现和信号质量指标。"""
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


SYNC_CHANNELS = ("sample_counter", "hardware_timestamp", "photodiode", "trigger")


class ContinuousRecording:
    """有界队列 + 约一秒的原子分块；不占用呈现线程，不插值、不滤波。

    recording_sample_index 仅为本程序接收到的数据编号，不冒充设备采样计数器。
    分块在成功封装前一直保留；进程意外退出后也可单独读取已完成的分块。
    """

    def __init__(self, folder: Path, channels: int, fs: float):
        """创建分块目录、有界队列和后台写入线程。"""
        self.folder, self.channels, self.fs = folder, channels, fs
        self.folder.mkdir(parents=True, exist_ok=False)
        self.pending: queue.Queue = queue.Queue(maxsize=512)
        self.error: Optional[BaseException] = None
        self.closed = False
        self.accepted_samples = 0
        self.submitted_samples = 0
        self.saved_samples = 0
        self.chunk_count = 0
        self.data_dtype: Optional[np.dtype] = None
        self.parts: list[Path] = []
        self.thread = threading.Thread(target=self._write, name="EEG-file-writer", daemon=True)
        self.thread.start()

    def append(self, data: np.ndarray, stamps: np.ndarray, arrival: float) -> None:
        """复制一个采集回调块并将其无阻塞地提交给写入线程。"""
        if self.closed or self.error is not None:
            raise RuntimeFault("SAVE", f"连续数据写入已停止：{self.error}")
        if data.shape != (len(stamps), self.channels):
            raise RuntimeFault("SAVE", "连续数据通道数变化，拒绝错误拼接")
        if self.data_dtype is None:
            self.data_dtype = data.dtype
        if data.dtype != self.data_dtype:
            raise RuntimeFault("SAVE", "连续数据类型变化，拒绝有损转换")
        self.submitted_samples += len(stamps)
        item = (np.array(data, copy=True), stamps.copy(),
                float(arrival), self.accepted_samples, self.chunk_count)
        try:
            self.pending.put_nowait(item)
        except queue.Full as exc:
            self.error = RuntimeError("连续录制队列已满；停止采集，不能宣称完整保存")
            raise RuntimeFault("SAVE", str(self.error)) from exc
        self.accepted_samples += len(stamps)
        self.chunk_count += 1

    def _write(self) -> None:
        """将队列中的采集块聚合为约一秒一个的原子 NPZ 分块。"""
        batch: list = []
        count = 0

        def flush() -> None:
            """将当前批次原子写入磁盘并更新已保存计数。"""
            nonlocal count
            if not batch:
                return
            path = self.folder / f"block_{len(self.parts):06d}.npz"
            SessionRecorder._atomic_npz(path, {
                "source_data": np.concatenate([item[0] for item in batch]).T,
                "lsl_postprocessed_timestamps": np.concatenate([item[1] for item in batch]),
                "callback_lsl_clock": np.concatenate([np.full(len(item[1]), item[2]) for item in batch]),
                "recording_sample_index": np.concatenate([
                    np.arange(item[3], item[3] + len(item[1]), dtype=np.int64) for item in batch]),
                "receive_chunk_index": np.concatenate([
                    np.full(len(item[1]), item[4], dtype=np.int64) for item in batch]),
            })
            self.parts.append(path)
            self.saved_samples += count
            batch.clear()
            count = 0
        try:
            while True:
                try:
                    item = self.pending.get(timeout=.5)
                except queue.Empty:
                    flush()
                    continue
                if item is None:
                    flush()
                    return
                batch.append(item)
                count += len(item[1])
                if count >= max(1, round(self.fs)):
                    flush()
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        """排空队列并等待写入线程结束，保留失败时的已有分块。"""
        if self.closed:
            if self.thread.is_alive():
                raise RuntimeFault("SAVE", "连续录制线程仍在写入，未开始封装")
            return
        self.closed = True
        if self.thread.is_alive():
            try:
                self.pending.put(None, timeout=5)
            except queue.Full as exc:
                raise RuntimeFault("SAVE", "连续录制队列无法排空；保留分块") from exc
            self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeFault("SAVE", "连续录制线程未及时退出；保留分块")

    def summary(self) -> dict:
        """返回连续录制的样本、分块、数据类型和完整性摘要。"""
        return {"submitted_samples": self.submitted_samples,
                "queued_samples": self.accepted_samples, "saved_samples": self.saved_samples,
                "received_chunks": self.chunk_count, "saved_blocks": len(self.parts),
                "source_dtype": str(self.data_dtype) if self.data_dtype is not None else None,
                "complete_for_submitted_samples": bool(self.closed and not self.thread.is_alive()
                    and self.error is None and self.submitted_samples == self.saved_samples),
                "error": str(self.error) if self.error else None,
                "scope": "Samples delivered to the MNE-LSL callback; upstream loss unverified"}


class SessionRecorder:
    """连续数据与逐试次落盘；两种模式均封装完整会话ZIP。"""

    schema_version = 5

    def __init__(self, cfg: Config, *, root: Optional[str | Path] = None,
                 context: Optional[dict[str, Any]] = None):
        """准备会话路径、清单、程序快照和可选的时间校准证据。"""
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
        self.continuous: Optional[ContinuousRecording] = None
        self.capture_active = False
        self.continuous_metadata: dict = {}
        self.timing_calibration: dict = {"status": "not_provided", "measurement": None}
        if cfg.timing_calibration_file:
            path = Path(cfg.timing_calibration_file).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            content = path.read_bytes()
            measurement = json.loads(content)
            json.dumps(measurement, allow_nan=False)
            if not isinstance(measurement, dict):
                raise ValueError("时间校准文件必须为JSON对象")
            self.timing_calibration = {"status": "provided_not_independently_verified",
                "source_file": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                "measurement": measurement}
        self.program = {
            "program_version": "dual-mode-continuous-recording-6", "export_schema_version": 6,
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

    def start_continuous(self, metadata: dict) -> None:
        """验证流元数据并启动包含全部 LSL 通道的连续录制。"""
        if self.continuous is not None:
            raise RuntimeError("连续录制不能重复启动")
        channels = int(metadata["channel_count"])
        for key in SYNC_CHANNELS:
            index = getattr(self.cfg, key + "_channel")
            if index is not None and index >= channels:
                raise RuntimeFault("SAVE", f"{key}通道{index}超出当前LSL流的{channels}列")
        self._start_on_first_trial()
        self.continuous_metadata = {"stream": metadata, "config": vars(self.cfg).copy(),
            "timing_calibration": self.timing_calibration, "program": self.program,
            "array_layout": "channels x samples; all source columns in original order",
            "timestamp_semantics": "LSL clocksync/dejitter/monotonize; NOT hardware capture time",
            "callback_lsl_clock_semantics": "one local LSL clock read per callback, repeated per sample",
            "recording_sample_index_semantics": "receiver row number; NOT device sample counter",
            "recording_scope": "first acquisition callback through collector shutdown, including warmup",
            "source_data_scope": "all LSL channels before receiver filtering; upstream processing may apply"}
        self._atomic_json(self.session_dir / "continuous_metadata.json", self.continuous_metadata)
        self.continuous = ContinuousRecording(self.session_dir / "continuous_blocks",
                                               channels, float(metadata["nominal_fs"]))

    def record_continuous(self, data: np.ndarray, stamps: np.ndarray, arrival: float) -> None:
        """将一个原始 LSL 回调块转交给连续录制器。"""
        if self.continuous is None:
            raise RuntimeFault("SAVE", "连续录制尚未初始化")
        self.continuous.append(data, stamps, arrival)

    def check_recording_health(self) -> None:
        """在后台连续写入失败时抛出标准保存故障。"""
        if self.continuous is not None and self.continuous.error is not None:
            raise RuntimeFault("SAVE", f"连续数据保存失败：{self.continuous.error}")

    def _start_on_first_trial(self) -> None:
        """惰性创建会话目录和初始清单。"""
        if self._started:
            return
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._atomic_json(self.manifest_path, self.manifest)
        self._started = True

    @staticmethod
    def _atomic_csv(path: Path, columns: Sequence[str], rows: Any) -> None:
        """写入并同步临时 CSV，完成后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _export_timing(self, payload: dict) -> dict[str, Path]:
        """磁盘映射合并连续分块；缺失硬件信息以空数组和状态记录，绝不推算填充。"""
        stem = self.session_dir.name
        paths = {key: self.session_dir / f"{stem}_{suffix}" for key, suffix in {
            "continuous": "continuous_eeg.npz", "signals": "sync_signals.npz",
            "timing_calibration": "timing_calibration.json", "sync_report": "sync_report.json",
            "frames": "frames.csv", "events": "events.csv", "metadata": "metadata.json",
            "readme": "recording_readme.txt"}.items()}
        recording = self.continuous
        summary = recording.summary() if recording is not None else {
            "submitted_samples": 0, "queued_samples": 0, "saved_samples": 0,
            "complete_for_submitted_samples": False, "error": "continuous acquisition not started"}
        n = summary["saved_samples"]
        channels = recording.channels if recording is not None else 0
        stream = payload["acquisition_metadata"]
        signal_status = {}
        timing_stats: dict = {"available": False, "physical_latency_measured": False}
        with tempfile.TemporaryDirectory(prefix=".continuous-export-", dir=self.session_dir) as folder, ExitStack() as mappings:
            def array(name: str, shape: tuple, dtype: Any = float) -> np.ndarray:
                """为大型导出数组创建空数组或临时内存映射。"""
                if not all(shape):
                    return np.empty(shape, dtype=dtype)
                mapped = np.lib.format.open_memmap(Path(folder) / (name + ".npy"), mode="w+",
                                                   dtype=dtype, shape=shape)
                mappings.callback(mapped._mmap.close)
                return mapped
            data = array("source_data", (channels, n),
                         recording.data_dtype if recording is not None and recording.data_dtype is not None else float)
            stamps = array("lsl_postprocessed_timestamps", (n,))
            local = array("lag_corrected_lsl_timestamps", (n,))
            arrivals = array("callback_lsl_clock", (n,))
            indices = array("recording_sample_index", (n,), np.int64)
            chunks = array("receive_chunk_index", (n,), np.int64)
            offset = 0
            for part in recording.parts if recording is not None else []:
                with np.load(part, allow_pickle=False) as saved:
                    count = len(saved["lsl_postprocessed_timestamps"])
                    end = offset + count
                    data[:, offset:end] = saved["source_data"]
                    for name, dest in (("lsl_postprocessed_timestamps", stamps),
                                       ("callback_lsl_clock", arrivals),
                                       ("recording_sample_index", indices),
                                       ("receive_chunk_index", chunks)):
                        dest[offset:end] = saved[name]
                    local[offset:end] = stamps[offset:end] - self.cfg.device_timestamp_lag_s
                    offset = end
            if offset != n:
                raise RuntimeError("连续分块样本总数不符，保留原始分块")
            continuous_metadata = {**self.continuous_metadata, "recording": summary,
                "stream": stream, "selected_eeg_channel_indices": self.cfg.channel_indices,
                "applied_device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
                "original_outlet_timestamps_available": False}
            self._atomic_npz(paths["continuous"], {
                "source_data": data, "lsl_postprocessed_timestamps": stamps,
                "lag_corrected_lsl_timestamps": local, "callback_lsl_clock": arrivals,
                "recording_sample_index": indices, "receive_chunk_index": chunks,
                "selected_eeg_channel_indices": np.asarray(self.cfg.channel_indices, dtype=np.int64),
                "metadata_json": np.array(json.dumps(_json_safe(continuous_metadata), ensure_ascii=False)),
            })
            signals: dict[str, Any] = {"recording_sample_index": indices,
                "lsl_postprocessed_timestamps": stamps, "lag_corrected_lsl_timestamps": local}
            channel_metadata = stream.get("all_channels", [])
            for key in SYNC_CHANNELS:
                index = getattr(self.cfg, key + "_channel")
                available = index is not None and index < channels and n > 0
                values = data[index] if available else np.empty(0, dtype=float)
                signals[key] = values
                finite = values[np.isfinite(values)]
                signal_status[key] = {
                    "status": "recorded_unvalidated" if available else "unavailable",
                    "lsl_channel_index": index, "samples": len(values),
                    "channel_metadata": channel_metadata[index] if available and index < len(channel_metadata) else None,
                    "source": "user-configured column in the same EEG LSL stream" if available else None,
                    "reason": None if available else ("channel not configured" if index is None else "no recorded samples"),
                    "finite_samples": len(finite),
                    "minimum": float(finite.min()) if len(finite) else None,
                    "maximum": float(finite.max()) if len(finite) else None,
                    "hardware_semantics_verified": False,
                }
            signals["metadata_json"] = np.array(json.dumps({
                "signals": signal_status, "hardware_values": "as received, no unit/clock conversion",
                "missing_values": "empty arrays mean unavailable, never synthesized",
                "alignment": "same received rows; device sampling relationship requires validation",
            }, ensure_ascii=False))
            self._atomic_npz(paths["signals"], signals)
            if n:
                delta = np.diff(stamps)
                finite_delta = delta[np.isfinite(delta)]
                monotonic = bool(np.isfinite(stamps).all() and np.all(delta > 0))
                timing_stats = {"available": True, "physical_latency_measured": False,
                    "first_lsl_timestamp": float(stamps[0]), "last_lsl_timestamp": float(stamps[-1]),
                    "strictly_increasing": monotonic,
                    "nonfinite_timestamp_count": int((~np.isfinite(stamps)).sum()),
                    "nonpositive_interval_count": int((delta <= 0).sum()),
                    "median_interval_s": float(np.median(finite_delta)) if len(finite_delta) else None,
                    "maximum_interval_s": float(finite_delta.max()) if len(finite_delta) else None,
                    "timestamp_gap_count": int((delta > self.cfg.max_gap_factor / recording.fs).sum()),
                    "estimated_rate_hz": float((n - 1) / (stamps[-1] - stamps[0])) if monotonic and n > 1 else None,
                    "scope": "postprocessed LSL timestamps; not a hardware loss or jitter measurement"}
            else:
                monotonic = False
            trials = []
            for record in self._records:
                start, end = record.requested_window_start, record.requested_window_end
                covered = bool(monotonic and start is not None and end is not None
                               and local[0] <= start < end <= local[-1])
                trials.append({"trial_id": record.trial_id, "status": record.status,
                    "stimulus_onset_lsl": record.stimulus_onset,
                    "stimulus_offset_lsl": record.stimulus_offset,
                    "requested_window_start_lsl": start, "requested_window_end_lsl": end,
                    "software_window_bounds_covered": covered,
                    "recording_start_index": int(np.searchsorted(local, start)) if covered else None,
                    "recording_end_index_exclusive": int(np.searchsorted(local, end)) if covered else None,
                    "frame_timing_valid": record.frame_diagnostics.get("valid"),
                    "physical_alignment": "unverified"})
        calibration = {**self.timing_calibration,
            "applied_device_timestamp_lag_s": self.cfg.device_timestamp_lag_s,
            "correction_formula": "lag_corrected_lsl_timestamp = lsl_postprocessed_timestamp - device_timestamp_lag_s",
            "zero_lag_meaning": "no correction applied; not a measured zero latency",
            "calibration_file_auto_applied": False,
            "display_latency_correction_applied_s": 0.0,
            "response_delay_s": self.cfg.response_delay_s,
            "response_delay_meaning": "analysis window parameter; not a measured hardware delay",
            "measured_physical_offset_s": None, "measured_physical_jitter_s": None,
            "measured_clock_drift_ppm": None,
            "measurement_scope": "provided evidence is archived verbatim; no automatic physical calibration"}
        self._atomic_json(paths["timing_calibration"], calibration)
        self._atomic_json(paths["sync_report"], {
            "schema_version": 1, "physical_alignment": "unverified",
            "physical_alignment_reason": "需验证光电/触发通道含义、设备采样关系，并进行事件匹配与延迟校准",
            "recording": summary, "software_timestamp_diagnostics": timing_stats,
            "signals": signal_status, "trials": trials,
            "original_outlet_timestamps": {"status": "unavailable", "reason": "MNE-LSL postprocessing enabled"},
            "calibration_file": paths["timing_calibration"].name,
            "acquisition_timestamp_processing": stream.get("timestamp_processing", {}),
            "limitations": ["软件时间戳正常不证明实际出光与采样对齐", "硬件通道未经配置时不推算、不填零冒充测量",
                            "recording_sample_index不是硬件采样序号", "软件间隔统计不能证明设备无丢样"]})
        self._atomic_csv(paths["events"], ("timestamp_lsl", "event", "trial_id", "target_id", "details_json"),
            ({"timestamp_lsl": e.get("timestamp"), "event": e["kind"], "trial_id": e["trial_id"],
              "target_id": e.get("target_id"), "details_json": json.dumps(e.get("details", {}), ensure_ascii=False)}
             for e in payload["events"]))
        def frame_rows():
            """逐行生成每个试次的显示帧时间记录。"""
            for entry in self.manifest["trials"]:
                record = json.loads((self.session_dir / entry["metadata_file"]).read_text(encoding="utf-8"))
                with np.load(self.session_dir / entry["arrays_file"], allow_pickle=False) as saved:
                    flips = saved["frame_flip_times"]
                    clocks = saved["frame_lsl_times"]
                target = record["true_class"]
                for i, stamp in enumerate(flips):
                    yield {"trial_id": record["trial_id"], "frame_index": i,
                        "flip_time_psychopy": stamp,
                        "flip_callback_lsl": clocks[i] if i < len(clocks) else None,
                        "interval_s": stamp - flips[i-1] if i else None,
                        "is_offset_flip": bool(record["stimulus_offset"] is not None and i == len(flips)-1),
                        "stimulus_onset_lsl": record["stimulus_onset"],
                        "target_id": target,
                        "target_frequency_hz": TARGETS[target-1].frequency_hz if target else None}
        self._atomic_csv(paths["frames"], ("trial_id", "frame_index", "flip_time_psychopy", "flip_callback_lsl", "interval_s",
            "is_offset_flip", "stimulus_onset_lsl", "target_id", "target_frequency_hz"), frame_rows())
        self._atomic_json(paths["metadata"], {**payload, "targets": _json_safe(TARGETS),
            "session_id": stem, "created_utc": self.manifest["created_utc"],
            "recording": summary, "timing_calibration": calibration,
            "files": {key: path.name for key, path in paths.items()},
            "frame_clock": "PsychoPy flip return and local LSL clock in callOnFlip are separate readings; neither measures photons"})
        paths["readme"].write_text(
            "EEG会话录制说明\n"
            "同名会话NPZ：原有逐试次EEG、处理网格、分类结果和完整试次元数据。\n"
            "continuous_eeg.npz：从首个采集回调到采集停止的所有LSL通道，包含预热、刺激及间歇。\n"
            "模式2可用events.csv中的stimulus_onset从连续EEG截取每次闪烁开始后的数据；程序不自动估计延迟。\n"
            "模式2闪烁覆盖2秒分析窗及最多300毫秒离线候选起点。\n"
            "source_data为通道×样本，按原流列顺序；selected_eeg_channel_indices给出分类通道。\n"
            "lsl_postprocessed_timestamps已由LSL校时、去抖和单调化，不是原始发布端或硬件时间戳。\n"
            "callback_lsl_clock是采集回调时读取的本机时钟，同一回调内重复，不是逐样本到达或采样时间。\n"
            "recording_sample_index只是接收记录序号，不代表设备采样序号。\n"
            "sync_signals.npz：配置的硬件计数、硬件时间戳、光电和触发列；未获取时为空数组。\n"
            "同步列保留收到的值，不擅自换算单位、校时或解释计数回绕。独立AUX流本版本未连接。\n"
            "events.csv：LSL软件事件；frames.csv：PsychoPy逐帧时间及翻帧回调LSL时间，均非实际出光时刻。\n"
            "timing_calibration.json：外部校准证据及本次实际补偿；0补偿不代表测得0延迟。\n"
            "sync_report.json：记录完整性、软件时间间隔、硬件字段可用性和逐试次覆盖检查。\n"
            "metadata.json：配置、设备元数据、显示信息、程序版本和源代码哈希。\n"
            "光电信号存在也不自动代表已校准；本程序不根据分类准确率猜测物理延迟。\n"
            "异常退出：session目录中的continuous_blocks分块可单独用numpy.load读取。\n"
            "这些分块最多保留到最近一次成功落盘，崩溃前内存中的队列可能未写入。\n",
            encoding="utf-8")
        return paths

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        """以 UTF-8 写入并同步临时 JSON，随后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    @staticmethod
    def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        """写入并同步临时压缩 NPZ，随后原子替换目标文件。"""
        temp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def write_trial(self, record: TrialRecord) -> dict[str, str]:
        """原子保存单个试次的数组、元数据并更新会话清单。"""
        self._start_on_first_trial()
        trial_stem = f"trial_{record.trial_id:04d}"
        metadata_path = self.session_dir / f"{trial_stem}.json"
        arrays_path = self.session_dir / f"{trial_stem}.npz"
        record.artifact_refs = {"metadata": str(metadata_path), "arrays": str(arrays_path)}
        result = record.result or {}
        epoch = result.get("epoch")
        notch_context_metadata = None
        arrays: dict[str, np.ndarray] = {
            "frame_flip_times": np.asarray(record.frame_flip_times, dtype=float),
            "frame_lsl_times": np.asarray(record.frame_lsl_times, dtype=float),
        }
        frame_intervals = record.frame_diagnostics.get("psychopy_frame_intervals_s")
        if frame_intervals is not None:
            arrays["psychopy_frame_intervals_s"] = np.asarray(frame_intervals, dtype=float)
        if epoch is not None:
            epoch.validate_dimensions()
            arrays.update({
                "source_data": np.asarray(epoch.source_data, dtype=float),
                "source_lsl_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                "source_raw_timestamps": np.asarray(epoch.raw_timestamps, dtype=float),
                "source_timestamps": np.asarray(epoch.source_timestamps, dtype=float),
                "timestamp_adjustments": epoch.source_timestamps - epoch.raw_timestamps,
                "source_local_timestamps": np.asarray(epoch.local_timestamps, dtype=float),
                "clock_corrections": np.asarray(epoch.clock_corrections, dtype=float),
                "uniform_data": np.asarray(epoch.data, dtype=float),
                "uniform_timestamps": np.asarray(epoch.uniform_timestamps, dtype=float),
            })
            notch_context = getattr(epoch, "notch_context", None)
            if notch_context is not None:
                arrays.update({
                    "notch_context_data": np.asarray(notch_context.data, dtype=float),
                    "notch_context_local_timestamps": np.asarray(notch_context.local_timestamps, dtype=float),
                    "notch_context_uniform_timestamps": np.asarray(notch_context.uniform_timestamps, dtype=float),
                    "notch_context_source_offset": np.asarray(notch_context.source_offset, dtype=np.int64),
                    "notch_context_epoch_end": np.asarray(notch_context.epoch_end, dtype=float),
                    "notch_context_requested_history_samples": np.asarray(
                        notch_context.requested_history_samples, dtype=np.int64),
                })
                context_ts = arrays["notch_context_local_timestamps"]
                notch_context_metadata = {
                    "source_offset": int(notch_context.source_offset),
                    "epoch_end": float(notch_context.epoch_end),
                    "requested_history_samples": int(notch_context.requested_history_samples),
                    "saved_samples": int(notch_context.data.shape[1]),
                    "saved_start": float(context_ts[0]) if len(context_ts) else None,
                    "saved_end": float(context_ts[-1]) if len(context_ts) else None,
                    "data_scope": "Selected LSL channels before receiver filtering; past context and current epoch",
                    "timestamp_semantics": "LSL-postprocessed local timestamps; not device hardware timestamps",
                }
        for key in ("scores", "correlations", "weights", "cca_scores", "fbcca_scores", "ecca_features",
                    "history_notched_data"):
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
            "notch_mode": getattr(self.cfg, "notch_mode", "epoch"),
            "notch_history_s": getattr(self.cfg, "notch_history_s", 4.0),
            "notch_processing": result.get("notch_processing"),
            "notch_context": notch_context_metadata,
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
        """将一个无 pickle 的 NPY 数组成员写入 NPZ 归档。"""
        with archive.open(name + ".npy", "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)

    def _export_session(self, payload: dict[str, Any]) -> str:
        """逐试次合并原始EEG与元数据，生成可直接读取的NPZ。"""
        stem = self.session_dir.name
        self.session_path = self.session_dir / f"{stem}.npz"
        trials = []
        for entry in self.manifest["trials"]:
            with (self.session_dir / entry["metadata_file"]).open(encoding="utf-8") as handle:
                trials.append(json.load(handle))
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
                    "data_scope": (
                        "Recorded raw trial windows plus per-trial raw notch history; not a continuous session recording"
                        if getattr(self.cfg, "notch_mode", "epoch") == "history" else
                        "Recorded trial windows only; not a continuous session recording"),
                    "array_layout": "channels x samples; each trial uses trial_NNNN_ prefixed keys",
                    "label_convention": "1-based true class; -1 means unknown (including free mode)",
                    "source_data_scope": "Selected LSL channels before receiver filtering; upstream processing may apply",
                    "continuous_data_file": stem + "_continuous_eeg.npz",
                    "sync_report_file": stem + "_sync_report.json",
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
        """停止写入、导出会话文件、封装 ZIP，并清理已成功封装的临时文件。"""
        if not self._started:
            return None
        if self.capture_active:
            raise RuntimeFault("SAVE", "采集线程尚未停止，不能封装正在写入的连续记录；保留分块")
        if self.continuous is not None:
            self.continuous.close()
        payload = _json_safe({
            "schema_version": self.schema_version,
            "manifest_file": self.manifest_path.name,
            "config": vars(self.cfg), "summary": summary,
            "decoder_settings": {"filter_bank_profile": self.cfg.filter_bank_profile,
                                 "filter_bands_hz": self.cfg.filter_bands,
                                 "n_harmonics": self.cfg.n_harmonics,
                                 "line_regression_hz": self.cfg.line_regression_hz,
                                 "notch_mode": getattr(self.cfg, "notch_mode", "epoch"),
                                 "notch_history_s": getattr(self.cfg, "notch_history_s", 4.0),
                                 "notch_hz": self.cfg.notch_hz,
                                 "notch_Q": 30.0,
                                 "notch_pipeline": (
                                     "raw sample-order history through epoch end -> notch filtfilt -> "
                                     "interpolation to epoch grid -> row normalization -> "
                                     "epoch filter bank; no second epoch notch"
                                     if getattr(self.cfg, "notch_mode", "epoch") == "history" else
                                     "uniform raw epoch -> optional line regression -> row normalization -> "
                                     "epoch notch filtfilt -> epoch filter bank")},
            "events": events, "acquisition_metadata": acquisition_metadata,
            "acquisition_review": acquisition_review, "clock_updates": clock_updates,
            "display_info": display_info, "typed_text": typed_text,
            "stop_reason": stop_reason,
            "program": self.program, "export_status": "pending",
            "continuous_recording": self.continuous.summary() if self.continuous else None,
        })
        self._atomic_json(self.progress_path, payload)
        self.manifest["session_file"] = self.progress_path.name
        self._atomic_json(self.manifest_path, self.manifest)
        try:
            timing_files = self._export_timing(payload)
            self._export_session(payload)
            sidecars: dict[str, str] = {}
            if self.cfg.session_mode == "cued":
                sidecars = export_cued_sidecars(self.session_path)
                package_cued_session(self.session_path, destination=self.archive_path,
                    extra_sources=[path for key, path in timing_files.items() if key != "events"])
            else:
                package_recording_files([self.session_path, *timing_files.values()], self.archive_path)
            self.export_artifacts = {"session": str(self.archive_path)}
        except Exception as exc:
            payload["export_status"] = "failed"
            payload["export_error"] = f"{type(exc).__name__}: {exc}"
            self._atomic_json(self.progress_path, payload)
            self.manifest["export_status"] = "failed"
            self._atomic_json(self.manifest_path, self.manifest)
            raise
        for record in self._records:
            record.artifact_refs = {"session_archive": str(self.archive_path),
                 "session_member": self.session_path.name,
                 "array_prefix": f"trial_{record.trial_id:04d}_"}
        if self.continuous is not None and self.continuous.error is not None:
            return str(self.archive_path)
        for path in {self.session_path, *timing_files.values(), *(Path(p) for p in sidecars.values())}:
            path.unlink()
        if self.continuous is not None:
            for path in self.continuous.parts:
                path.unlink()
            self.continuous.folder.rmdir()
            (self.session_dir / "continuous_metadata.json").unlink()
        for entry in self.manifest["trials"]:
            (self.session_dir / entry["metadata_file"]).unlink()
            (self.session_dir / entry["arrays_file"]).unlink()
        self.progress_path.unlink()
        self.manifest_path.unlink()
        self.session_dir.rmdir()
        return str(self.archive_path)


def export_cued_sidecars(session_path: str | Path) -> dict[str, str]:
    """为模式2会话NPZ导出质控JSON、可加载模型PKL和事件CSV。

    可对旧版单文件会话NPZ再次运行；原NPZ保持原位且不会被修改。
    """
    session_path = Path(session_path)
    with np.load(session_path, allow_pickle=False) as archive:
        session = json.loads(archive["metadata_json"].item())
        if session.get("config", {}).get("session_mode") != "cued":
            raise ValueError("只有模式2提示测试会话可生成质控和个人模型文件")
        saved_config = dict(session["config"])
        saved_config.pop("quick_entry_mode", None)
        cfg = Config(**saved_config)
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
                         remove_sources: bool = False,
                         extra_sources: Sequence[Path] = ()) -> str:
    """兼容旧四文件会话，也可附加完整连续记录，核对后才可移除源文件。"""
    session_path = Path(session_path)
    stem = session_path.stem
    sources = (session_path,
               session_path.with_name(stem + "_qc.json"),
               session_path.with_name(stem + "_model.pkl"),
               session_path.with_name(stem + "_events.csv"), *extra_sources)
    archive_path = Path(destination) if destination is not None else session_path.with_suffix(".zip")
    return package_recording_files(sources, archive_path, remove_sources=remove_sources)


def package_recording_files(sources: Sequence[Path], archive_path: Path, *,
                            remove_sources: bool = False) -> str:
    """封装会话文件，逐项核对哈希后可选择移除源文件。"""
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("会话ZIP存在重复文件名")
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
    """根据逐帧刺激使用的刷新率检查软件翻帧时间。"""
    ts = np.asarray(flip_times, float)
    if (ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all()
            or not np.isfinite(refresh_hz) or refresh_hz <= 0):
        raise InvalidTrial("刺激帧时间记录不足或非法")
    if not np.isfinite(rate_tolerance) or not 0 < rate_tolerance < 1:
        raise ValueError("rate_tolerance 必须是(0, 1)内的有限比例")
    intervals = np.diff(ts)
    period = 1 / refresh_hz
    elapsed = float(ts[-1] - ts[0])
    observed_rate = float(len(intervals) / elapsed) if elapsed > 0 else None
    rate_error = ((observed_rate / refresh_hz - 1) if observed_rate is not None else None)
    rate_mismatch = rate_error is None or abs(rate_error) > rate_tolerance
    long = intervals > long_factor * period
    short = intervals < short_factor * period
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
    """按随机种子为每个区组生成一次覆盖全部目标的顺序。"""
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError("blocks必须是正整数")
    rng = np.random.default_rng(seed)
    return [(b + 1, int(class_id)) for b in range(blocks)
            for class_id in rng.permutation(np.arange(1, len(TARGETS) + 1))]


def make_luminance_table(refresh_hz: float, duration_s: float) -> np.ndarray:
    """预计算所有目标在每个显示帧上的正弦亮度值。"""
    if not np.isfinite(refresh_hz) or refresh_hz <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
        raise ValueError("刷新率无法支持当前最高刺激频率")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("刺激时长必须为正")
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
    """保留有效试次准确率；另计每个块/目标首尝试，失败与拒识不能靠重试抹去。"""
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
    first_records: dict[tuple[int, int], TrialRecord] = {}
    for record in records:
        key = (record.block_id, record.true_class)
        previous = first_records.get(key)
        if previous is None or record.trial_id < previous.trial_id:
            first_records[key] = record
    first_started = len(first_records)
    first_attempt = {
        "selection_rule": "lowest actual trial_id per (block_id, true_class)",
        "started_targets": first_started,
        "planned_targets": planned,
        "coverage_rate": first_started / planned if planned > 0 else None,
        "all_planned_targets_started": planned > 0 and first_started == planned,
        "retry_attempts": attempted - first_started,
        "trial_ids": sorted(r.trial_id for r in first_records.values()),
        "nonvalid_first_attempts": sum(r.status != "valid" for r in first_records.values()),
    }
    for name, prediction_key in (("online", "prediction"),
                                 ("fbcca", "fbcca_prediction"), ("cca", "cca_prediction")):
        first_correct = sum(
            r.status == "valid" and r.result is not None
            and r.result[prediction_key] == r.true_class
            for r in first_records.values())
        first_attempt[name] = {
            "correct": int(first_correct),
            "accuracy_started": first_correct / first_started if first_started else None,
            "success_planned": first_correct / planned if planned > 0 else None,
        }
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
        "calibration_fallback_valid_count": calibration_fallback_valid_count,
        "first_attempt": first_attempt}
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
    """将可选比例格式化为百分比或 ``N/A``。"""
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _brief_reason(reason: str, limit: int = 160) -> str:
    """压缩空白并截短过长的单行原因文本。"""
    message = " ".join(reason.split())
    return message if len(message) <= limit else message[:limit - 1] + "…"


def print_summary(report: dict, ledger: TrialLedger) -> None:
    """输出提示测试的首尝试覆盖率和各解码器准确率摘要。"""
    first = report["first_attempt"]
    coverage_note = ("已覆盖全部计划目标" if first["all_planned_targets_started"]
                     else "尚未覆盖全部计划目标；不能作为完整一场验收")
    print(f"首尝试覆盖 {first['started_targets']}/{first['planned_targets']} "
          f"({_pct(first['coverage_rate'])}) | 重试 {first['retry_attempts']} 次 | {coverage_note}")
    for name, label in (("online", "实际在线"), ("fbcca", "基础FBCCA"), ("cca", "标准CCA")):
        metric = first[name]
        print(f"首尝试{label}：已开始目标 {metric['correct']}/{first['started_targets']} "
              f"({_pct(metric['accuracy_started'])})；按全部计划 "
              f"{metric['correct']}/{first['planned_targets']} "
              f"({_pct(metric['success_planned'])})；无效/中止/拒识均计失败")
    print(f"取得有效结果的目标{report['completed_targets']} | 有效试次{report['valid']} | "
          f"基础FBCCA有效试次准确率 {_pct(report['fbcca']['accuracy_valid'])} | "
          f"标准CCA有效试次准确率 {_pct(report['cca']['accuracy_valid'])} | "
          f"实际在线算法有效试次准确率 {_pct(report['online']['accuracy_valid'])} | "
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
    """输出不含虚构准确率的自由输入会话摘要。"""
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
    """将常见电压单位写法规范为 ``uV``、``mV`` 或 ``V``。"""
    unit = str(value).strip().replace("μ", "u").replace("µ", "u").lower()
    return {"uv": "uV", "microvolt": "uV", "microvolts": "uV", "v": "V", "volt": "V", "volts": "V",
            "mv": "mV", "millivolt": "mV", "millivolts": "mV"}.get(unit, "UNKNOWN")


def preflight_transport(cfg: Config, collector: ContinuousLSL) -> dict[str, Any]:
    """自动确认最近的LSL EEG可用于试次；电极质量由OpenBCI GUI检查。"""
    collector.check_health()
    if collector.buffer is None:
        raise RuntimeFault("ACQUISITION", "LSL EEG缓冲区尚未建立")
    fs = float(collector.fs)
    try:
        _, _, local, _, _ = collector.buffer.snapshot()
    except InvalidTrial as exc:
        raise RuntimeFault("ACQUISITION", f"启动LSL缓冲读取失败：{exc}") from exc
    if len(local) < 2 or not np.isfinite(local).all():
        raise RuntimeFault("ACQUISITION", "启动LSL尚未收到足够的有效EEG时间戳")
    if np.any(np.diff(local) <= 0):
        raise RuntimeFault("ACQUISITION", "启动LSL时间戳回退或重复")
    available_s = float(local[-1] - local[0])
    if cfg.notch_mode == "history":
        available_s -= cfg.notch_history_s + 0.1
    duration = math.floor(min(5.0, available_s) * fs) / fs
    end = float(local[-1])
    start = end - duration
    if start < local[0]:
        duration -= 1 / fs
        start = end - duration
    if duration < 1.0:
        raise RuntimeFault("ACQUISITION", "最近LSL EEG未覆盖至少1秒连续数据")
    try:
        epoch = collector.buffer.epoch(
            start, duration, fs, cfg.max_gap_factor, cfg.rate_tolerance,
            cfg.hard_gap_factor, cfg.hard_rate_tolerance, cfg.max_warning_gaps,
            **({"notch_history_s": cfg.notch_history_s} if cfg.notch_mode == "history" else {}))
    except (InvalidTrial, ValueError) as exc:
        raise RuntimeFault("ACQUISITION", f"启动LSL数据不连续或包含无效样本：{exc}") from exc
    collector.check_health()
    age = float(collector.clock() - local[-1])
    if not math.isfinite(age) or abs(age) > cfg.max_receive_age_s:
        raise RuntimeFault("ACQUISITION", f"最新LSL EEG时间戳不新鲜或时钟异常：age={age:.3f}秒")

    meta = collector.metadata
    units = ([_canonical_unit(item["unit"]) for item in meta["selected_channels"]]
             if cfg.input_unit == "AUTO" else
             [_canonical_unit(cfg.input_unit)] * len(cfg.channel_indices))
    unverified = ["电极质量及物理接线由OpenBCI GUI/操作者检查，本程序未自动验证",
                  "TimeSeriesRaw及GUI滤波设置来自用户配置，LSL元数据未独立核实"]
    if "UNKNOWN" in units:
        unverified.append("LSL通道单位未全部声明；不推断物理幅度")
    startup_mains = mains_noise_diagnostics(epoch.data, fs, cfg.notch_hz)
    configuration_notes = []
    if meta["channel_count"] == 16 and max(cfg.channel_indices) < 8:
        configuration_notes.append(
            "当前LSL声明16路、程序只取前8路。若使用Cyton+Daisy，GUI关闭9–16路不等于"
            "切换到8通道采集模式；请核对GUI设备模式、原始记录采样率与LSL声明。"
            "仅凭LSL元数据不能判断实际板型或采样率是否错误。")
    review = {
        "review_method": "automatic_lsl_transport",
        "passed": True,
        "expected_openbci_output": "TimeSeriesRaw",
        "source_output_verified_from_lsl": False,
        "stream_name": meta["name"],
        "stream_type": meta["type"],
        "channel_count": meta["channel_count"],
        "selected_channel_indices": list(cfg.channel_indices),
        "input_units": units,
        "unit_override": cfg.input_unit,
        "upstream_filter_description": cfg.upstream_filter_description,
        "upstream_lowpass_hz": cfg.upstream_lowpass_hz,
        "receiver_notch_hz": cfg.notch_hz,
        "receiver_notch_mode": cfg.notch_mode,
        "receiver_notch_history_s": cfg.notch_history_s if cfg.notch_mode == "history" else 0.0,
        "wiring_user_reviewed": False,
        "electrode_quality_checked_by_app": False,
        "startup_transport": {
            "window_s": duration,
            "uniform_samples": int(epoch.data.shape[1]),
            "nominal_fs_hz": fs,
            "observed_fs_hz": float(collector.estimated_fs),
            "recent_fs_hz": epoch.diagnostics["timestamp_estimated_fs"],
            "latest_sample_age_s": age,
            "timestamp_warnings": list(epoch.diagnostics.get("warnings", [])),
            "warning_gap_count": epoch.diagnostics["warning_gap_count"],
        },
        "startup_mains_noise": startup_mains,
        "configuration_notes": configuration_notes,
        "unverified": unverified,
        "device_timestamp_lag_applied_s": cfg.device_timestamp_lag_s,
    }
    print(f"LSL传输检查通过：{meta['name']} / {meta['type']}；"
          f"通道{len(cfg.channel_indices)}/{meta['channel_count']}；"
          f"声明{fs:g}Hz，观测{collector.estimated_fs:.3f}Hz；"
          f"最近{duration:.2f}秒通过连续性检查，最新样本{age:.3f}秒前。", flush=True)
    for warning in epoch.diagnostics.get("warnings", []):
        if not warning.startswith("LSL时间戳已同步/去抖"):
            print("LSL时间戳提示：" + warning, flush=True)
    if startup_mains["available"]:
        low, high = startup_mains["line_band_hz"]
        ref_low, ref_high = startup_mains["reference_band_hz"]
        print(f"[采集质量] 滤波前{low:g}–{high:g}Hz / {ref_low:g}–{ref_high:g}Hz能量占比，"
              f"通道中位数={startup_mains['median_line_fraction']:.1%}。", flush=True)
        values = startup_mains["line_fraction_per_channel"]
        print("[各通道] " + "；".join(
            f"{position}(CH{index + 1})=" + (f"{value:.1%}" if value is not None else "不可用")
            for position, index, value in zip(cfg.channel_positions, cfg.channel_indices, values)),
            flush=True)
        if startup_mains["line_dominated"]:
            print("[采集质量提示] 工频附近能量占比高，建议先检查参考/BIAS、电极接触与周围电源。"
                  "此占比不是识别准确率，也不能单独确定干扰来源。", flush=True)
    for note in configuration_notes:
        print("[设备配置提示] " + note, flush=True)
    print("电极质量与实际接线沿用OpenBCI GUI检查；程序不会据此声称自动验证。", flush=True)
    return review


KEY_BASE_SIZE = 140.0
KEY_BASE_GAP = 30.0
GRID_WIDTH_FRACTION = .96
GRID_HEIGHT_FRACTION = .58
LAYOUT_REFERENCE_SIZE = (1470.0, 960.0)


def _keyboard_ui_scale(width: float, height: float) -> float:
    """顶部输出框和辅助文字随客户区等比例缩放。"""
    return min(width / LAYOUT_REFERENCE_SIZE[0], height / LAYOUT_REFERENCE_SIZE[1])


def _keyboard_side_margin(width: float) -> float:
    """输出框和键盘共用左右边界，并给最外侧提示框留出空间。"""
    return width * (1 - GRID_WIDTH_FRACTION) / 2


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
    """按截图的键间比例居中排布；不同屏幕只做统一等比例缩放。"""
    cfg.validate()
    width, height = map(float, window_size)
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("窗口宽高必须为有限正数")
    rows, cols = len(KEY_ROWS), max(map(len, KEY_ROWS))
    ui_scale = _keyboard_ui_scale(width, height)
    top_reserved = (cfg.output_box_height_px + cfg.output_box_margin_px + 70.0) * ui_scale
    bottom_reserved = 70.0 * ui_scale
    available_width = width - 2 * _keyboard_side_margin(width)
    available_height = height - top_reserved - bottom_reserved
    nominal_width = cols * KEY_BASE_SIZE + (cols - 1) * KEY_BASE_GAP
    nominal_height = rows * KEY_BASE_SIZE + (rows - 1) * KEY_BASE_GAP
    scale = min(available_width / nominal_width,
                GRID_HEIGHT_FRACTION * height / nominal_height,
                available_height / nominal_height)
    if scale <= 0:
        raise RuntimeError("屏幕客户区太小，无法容纳键盘与顶部输出框")
    key, gap = KEY_BASE_SIZE * scale, KEY_BASE_GAP * scale
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
            """将屏幕上的像素长度换算为居中视角。"""
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
        "layout_strategy": "centered_proportional_fit", "layout_schema_version": 3,
        "coordinate_space": "psychopy_pix_client",
        "requested_layout": {"rows": len(KEY_ROWS), "cols": max(map(len, KEY_ROWS)),
            "gap_to_key_ratio": KEY_BASE_GAP / KEY_BASE_SIZE,
            "max_grid_width_fraction": GRID_WIDTH_FRACTION,
            "max_grid_height_fraction": GRID_HEIGHT_FRACTION},
        "key_size_pix_units": key, "gap_pix_units": gap_x,
        "column_gap_pix_units": gap_x, "row_gap_pix_units": gap_y,
        "layout_scale_from_140px_reference": scale,
        "ui_scale_from_reference": _keyboard_ui_scale(*map(float, window_size)),
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
        size = (screen.width, screen.height)
    win = visual.Window(size=size, fullscr=cfg.full_screen, screen=cfg.screen_index,
        winType="pyglet", useRetina=True,
        units="pix", color=(-.78, -.77, -.73), colorSpace="rgb", waitBlanking=True,
        allowGUI=not cfg.full_screen, checkTiming=False, autoLog=False)
    win.keyboard_display_size = np.asarray((screen.width, screen.height), dtype=float)
    return win


def _keyboard_coordinate_size(win: Any) -> np.ndarray:
    """返回与 PsychoPy pix 刺激位置/尺寸相同的客户区坐标范围。"""
    size = np.asarray(win.clientSize, dtype=float)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise RuntimeError(f"无效的窗口客户区尺寸：{size}")
    return size


class MainThreadGraphicsGC:
    """窗口存活时，将循环垃圾回收限制在有 GL 上下文的主线程。

    pyglet 1.5 的纹理析构可能直接调用 glDeleteTextures。自动 GC 若被
    EEG/分类线程触发，会在没有当前 GL 上下文的线程内造成原生崩溃。
    引用计数仍正常工作；循环垃圾在试次间主动回收，避免长期输入累积。
    """

    def __init__(self) -> None:
        """确认当前位于主线程，并暂时关闭自动循环垃圾回收。"""
        self._check_thread()
        self.was_enabled = gc.isenabled()
        self.closed = False
        gc.disable()

    @staticmethod
    def _check_thread() -> None:
        """要求图形资源操作发生在 Python 主线程。"""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("图形资源必须在主线程回收")

    def collect(self, win: Any) -> None:
        """在窗口 GL 上下文有效且未记录帧间隔时主动回收循环垃圾。"""
        self._check_thread()
        if self.closed:
            return
        if win.recordFrameIntervals:
            raise RuntimeError("刺激时序记录期间不得执行图形垃圾回收")
        win._setCurrent()
        gc.collect()

    def close(self) -> None:
        """恢复进入上下文前的垃圾回收启用状态。"""
        self._check_thread()
        if self.closed:
            return
        self.closed = True
        if self.was_enabled:
            gc.enable()


class PsychoPyKeyboard:
    """负责 SSVEP 键盘的显示、交互、试次编排和结果反馈。"""

    def __init__(self, cfg: Config, collector: ContinuousLSL, markers: Optional[EventMarkers],
                 ledger: TrialLedger, recorder: Optional[SessionRecorder] = None,
                 *, static_only: bool = False):
        """绑定会话组件并在主线程创建 PsychoPy 窗口和视觉刺激。"""
        MainThreadGraphicsGC._check_thread()
        from psychopy import visual, event, logging
        self.cfg, self.collector, self.markers, self.ledger = cfg, collector, markers, ledger
        self.recorder = recorder
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
        self._closed = False
        self._graphics_gc = MainThreadGraphicsGC()
        try:
            logging.console.setLevel(logging.ERROR)
            self._build_window()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _build_window(self) -> None:
        """创建全部视觉元素，测量刷新率并完成静态渲染时序检查。"""
        visual, cfg = self.visual, self.cfg
        self.win = _create_keyboard_window(visual, cfg)
        self.win.mouseVisible = False
        width, height = _keyboard_coordinate_size(self.win)
        layout = keyboard_layout_info(cfg, (width, height), self.win.keyboard_display_size)
        self.positions = np.asarray(layout["key_positions_pix_units"])
        key, gap_y = layout["key_size_pix_units"], layout["row_gap_pix_units"]
        scale = key / KEY_BASE_SIZE
        ui_scale = layout["ui_scale_from_reference"]
        self.key_size = key
        cue_width = min(36.0 * scale, max(0.0, gap_y - 12 * scale) * 9 / 5)
        self._show_cue_arrow = cue_width >= 4
        cue_height = cue_width * 5 / 9
        self.cue_offset = 6 * scale + cue_height / 2
        self._static_colors = np.tile((.72, .58, .82), (len(TARGETS), 1))
        self._cue_colors = np.tile((-.33, -.37, -.25), (len(TARGETS), len(TARGETS), 1))
        for index in range(len(TARGETS)):
            self._cue_colors[index, index] = self._static_colors[index]
        self.key_borders = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key + 4 * scale, key + 4 * scale), elementTex=None, elementMask=None,
            colors=np.tile((-.88, -.89, -.84), (len(TARGETS), 1)), colorSpace="rgb", autoLog=False)
        self.squares = visual.ElementArrayStim(self.win, nElements=len(TARGETS), units="pix",
            xys=self.positions, sizes=(key, key), elementTex=None, elementMask=None,
            colors=self._static_colors, colorSpace="rgb", autoLog=False)
        self._static_label_color = (-.64, -.67, -.58)
        self._cue_label_color = (.63, .57, .69)
        self._label_cue_target_id: Optional[int] = None
        self.labels = []
        for target, pos in zip(TARGETS, self.positions):
            label_height = key * (.32 if len(target.display_label) > 1 else .42)
            self.labels.append(visual.TextStim(self.win, text=target.display_label, pos=pos,
                height=label_height, units="pix", color=self._static_label_color,
                colorSpace="rgb", bold=False, autoLog=False))
        self.outline_shadow = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="black", lineWidth=max(4, 10 * scale), units="pix", autoLog=False)
        self.outline = visual.Rect(self.win, width=key + 14 * scale, height=key + 14 * scale,
            fillColor=None, lineColor="#FFE45C", lineWidth=max(2, 6 * scale), units="pix", autoLog=False)
        self.triangle = (visual.ShapeStim(self.win,
            vertices=((-cue_width/2, -cue_height/2), (cue_width/2, -cue_height/2), (0, cue_height/2)),
            fillColor="#FFE45C", lineColor="black", lineWidth=max(1, 2 * scale), units="pix", autoLog=False)
            if self._show_cue_arrow else None)
        output_height = cfg.output_box_height_px * ui_scale
        output_margin = cfg.output_box_margin_px * ui_scale
        output_y = height/2 - output_margin - output_height/2
        output_width = width - 2 * _keyboard_side_margin(width)
        output_text_width = output_width - 48 * ui_scale
        self.output_box = visual.Rect(self.win, width=output_width, height=output_height,
            pos=(0, output_y), fillColor=(-0.82, -0.82, -0.82), lineColor="white", lineWidth=2*ui_scale,
            units="pix", autoLog=False)
        output_font_height = min(28, max(16, cfg.output_box_height_px*0.42)) * ui_scale
        self.output_text = visual.TextStim(self.win, text="", pos=(0, output_y),
            height=output_font_height, wrapWidth=output_text_width,
            font="Courier New", color="white", alignText="left", anchorHoriz="center", autoLog=False)
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        self._output_max_chars = max(12, int(output_text_width / output_font_height))
        self._sync_output_text(force=True)
        self.header = visual.TextStim(self.win, text="",
            pos=(0, output_y - output_height/2 - 22*ui_scale),
            height=24*ui_scale, wrapWidth=width-40*ui_scale, color="white", autoLog=False)
        self.cue_title = visual.TextStim(self.win, text="TARGET", pos=self.header.pos,
            height=34*ui_scale, wrapWidth=width-40*ui_scale,
            color="#FFE45C", bold=True, autoLog=False)
        self.footer = visual.TextStim(self.win, text="",
            pos=(0, -height/2+18*ui_scale), height=18*ui_scale, wrapWidth=width-40*ui_scale,
            color="white", autoLog=False)
        self.message = visual.TextStim(self.win, text="", pos=(0, -15*ui_scale),
            height=26*ui_scale, wrapWidth=width*.88, color="white", autoLog=False)
        menu_font = "Arial"
        self.mode_title = visual.TextStim(self.win, text="Select mode", pos=(0, 165*ui_scale),
            height=34*ui_scale, font=menu_font, color="white", autoLog=False)
        self.mode_positions = {"free": (0, 48*ui_scale), "cued": (0, -48*ui_scale)}
        self.mode_outline = visual.Rect(self.win, width=min(460*ui_scale, width*.8),
            height=76*ui_scale, fillColor=None, lineColor="#FFE45C",
            lineWidth=max(1, 2*ui_scale), units="pix", autoLog=False)
        self.mode_labels = {
            "free": visual.TextStim(self.win, text="1  Free spelling", pos=self.mode_positions["free"],
                height=28*ui_scale, font=menu_font, color="white", autoLog=False),
            "cued": visual.TextStim(self.win, text="2  Cued test", pos=self.mode_positions["cued"],
                height=28*ui_scale, font=menu_font, color="white", autoLog=False),
        }
        self.mode_hint = visual.TextStim(self.win, text="Enter / Space: start  ·  Esc: exit",
            pos=(0, -height/2 + 82*ui_scale), height=20*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.mode_warning = visual.TextStim(self.win, text="Warning: starting will show 8–15.8 Hz visual flicker · Esc: stop",
            pos=(0, -height/2 + 46*ui_scale), height=18*ui_scale,
            font=menu_font, color="#FFCB52", autoLog=False)
        self.startup_title = visual.TextStim(self.win, text="Checking display",
            pos=(0, height/2 - 64*ui_scale), height=34*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.startup_dot = visual.Circle(self.win, radius=7*ui_scale,
            pos=(-158*ui_scale, height/2 - 64*ui_scale),
            fillColor="#FFCB52", lineColor="#FFCB52", units="pix", autoLog=False)
        self.startup_subtitle = visual.TextStim(self.win, text="Please wait · No flicker",
            pos=(0, height/2 - 105*ui_scale), height=19*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        processing_key_size = min(120*ui_scale, key*1.1)
        self._processing_key_size = processing_key_size
        self._processing_cued_positions = ((0, -height*.005), (0, -height*.075),
                                          (0, -height*.135))
        self._processing_free_positions = ((0, height*.055), (0, -height*.015),
                                          (0, -height*.075))
        self.processing_key = visual.Rect(self.win, width=processing_key_size,
            height=processing_key_size, pos=(0, height*.10),
            fillColor=(-.20, -.24, -.10), lineColor="#BEB8C5",
            lineWidth=max(1, 2*ui_scale), units="pix", autoLog=False)
        self.processing_key_label = visual.TextStim(self.win, text="",
            pos=self.processing_key.pos, height=processing_key_size*.43,
            color="white", font=menu_font, autoLog=False)
        self.processing_dot = visual.Circle(self.win, radius=7*ui_scale,
            pos=(0, -height*.005), fillColor="#FFCB52", lineColor="#FFCB52",
            units="pix", autoLog=False)
        self.processing_title = visual.TextStim(self.win, text="Processing EEG",
            pos=(0, -height*.075), height=34*ui_scale,
            font=menu_font, color="white", bold=True, autoLog=False)
        self.processing_subtitle = visual.TextStim(self.win, text="Please wait for the result",
            pos=(0, -height*.135), height=20*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_verdict = visual.TextStim(self.win, text="",
            pos=(0, height*.11), height=68*ui_scale,
            font=menu_font, color="#92E5B6", bold=True, autoLog=False)
        self.result_pair = visual.TextStim(self.win, text="",
            pos=(0, height*.005), height=34*ui_scale, wrapWidth=width*.9,
            font=menu_font, color="white", autoLog=False)
        self.result_frequency = visual.TextStim(self.win, text="",
            pos=(0, -height*.055), height=19*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_divider = visual.Line(self.win,
            start=(-min(width*.13, 160*ui_scale), -height*.105),
            end=(min(width*.13, 160*ui_scale), -height*.105),
            lineColor="#777781", lineWidth=max(1, ui_scale), units="pix", autoLog=False)
        self.result_details = visual.TextStim(self.win, text="",
            pos=(0, -height*.18), height=19*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.result_warning = visual.TextStim(self.win, text="",
            pos=(0, -height*.29), height=16*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="#FFCB52", autoLog=False)
        self.pause_title = visual.TextStim(self.win, text="Paused",
            pos=(0, height*.015), height=56*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.pause_subtitle = visual.TextStim(self.win, text="",
            pos=(0, -height*.055), height=20*ui_scale, wrapWidth=width*.85,
            font=menu_font, color="white", autoLog=False)
        self.rest_title = visual.TextStim(self.win, text="Rest your eyes",
            pos=(0, height*.10), height=44*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.rest_timer = visual.TextStim(self.win, text="",
            pos=(0, -height*.015), height=112*ui_scale,
            font=menu_font, color="white", autoLog=False)
        self.rest_remaining_label = visual.TextStim(self.win, text="remaining",
            pos=(0, -height*.12), height=22*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        self.rest_auto_label = visual.TextStim(self.win, text="Resumes automatically",
            pos=(0, -height*.19), height=18*ui_scale,
            font=menu_font, color="#BDBDC6", autoLog=False)
        for _ in range(30):
            self._check_abort()
            self._draw_startup_check()
            self.win.flip()
        refresh = self._measure_refresh_rate()
        if refresh is None or not np.isfinite(refresh) or refresh <= 2 * BENCHMARK_FREQUENCIES_HZ.max():
            raise RuntimeError("未测得足够稳定的刷新率；不以猜测的60Hz继续实验")
        self.refresh_hz = float(refresh)
        self.win.refreshThreshold = cfg.frame_long_factor / self.refresh_hz
        durations = {"free": cfg.stimulus_s, "cued": cfg.cued_stimulus_s}
        self.luminance_by_mode = {
            mode: (np.empty((0, len(TARGETS))) if self.static_only else
                   make_luminance_table(self.refresh_hz, duration))
            for mode, duration in durations.items()
        }
        self.rgb_frames_by_mode = {
            mode: np.repeat((2 * luminance - 1)[:, :, None], 3, axis=2)
            for mode, luminance in self.luminance_by_mode.items()
        }
        flips = []
        for _ in range(121):
            self._check_abort()
            self._draw_startup_check()
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
        self.display_info = {**layout,
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
            "stimulus_frames": len(self.rgb_frames_by_mode["cued"]),
            "scheduled_stimulus_s": len(self.rgb_frames_by_mode["cued"])/self.refresh_hz,
            "stimulus_by_mode": {
                mode: {"frames": len(frames), "scheduled_s": len(frames) / self.refresh_hz}
                for mode, frames in self.rgb_frames_by_mode.items()
            },
            "static_render_check": check, "physical_timing_measured": False, "luminance_gamma_calibrated": False}

    def _measure_refresh_rate(self) -> Optional[float]:
        """在静态键盘上测量刷新率，避免 PsychoPy 绘制独立的等待画面。"""
        previous_recording = self.win.recordFrameIntervals
        self.win.frameIntervals = []
        self.win.recordFrameIntervals = True
        try:
            for _ in range(180):
                self._check_abort()
                self._draw_startup_check()
                self.win.flip()
                if len(self.win.frameIntervals) >= 30:
                    recent = np.asarray(self.win.frameIntervals[-30:], dtype=float)
                    if np.all(np.isfinite(recent)) and np.std(recent) < .0005:
                        return float(1.0 / np.mean(recent))
        finally:
            self.win.recordFrameIntervals = previous_recording
            self.win.frameIntervals = []
        return None

    def _cancel_decode(self) -> None:
        """通知当前解码任务取消，并尝试取消尚未开始的任务。"""
        if self._active_decode_cancel is not None:
            self._active_decode_cancel.set()
        if self._decode_future is not None:
            self._decode_future.cancel()

    def _check_abort(self) -> None:
        """处理退出或暂停按键，并同步检查 EEG 采集健康状态。"""
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
        """仅在文本变化或强制刷新时更新顶部输出框。"""
        text = self.ledger.typed_text
        if force or text != self._output_cache:
            self.output_text.text = output_display_text(text, self._output_max_chars)
            self._output_cache = text

    def _draw_status(self) -> None:
        """绘制当前页面通用的页眉和页脚。"""
        self.header.draw()
        self.footer.draw()

    def _draw_processing_status(self, record: TrialRecord) -> None:
        """绘制等待 EEG 解码结果的状态页。"""
        self._draw_status()
        if record.true_class is not None:
            self.processing_key.draw()
            self.processing_key_label.draw()
        self.processing_dot.draw()
        self.processing_title.draw()
        self.processing_subtitle.draw()

    def _draw_cued_result_feedback(self) -> None:
        """绘制提示测试的识别结果和可选警告。"""
        self._draw_status()
        self.result_verdict.draw()
        self.result_pair.draw()
        self.result_frequency.draw()
        self.result_divider.draw()
        self.result_details.draw()
        if self.result_warning.text:
            self.result_warning.draw()

    def _prepare_free_pause_display(self, message: str) -> None:
        """设置自由输入暂停页的文字内容。"""
        self.header.text = "Free spelling"
        self.footer.text = "SPACE or ENTER: resume   ·   ESC: exit"
        self.pause_subtitle.text = message

    def _draw_free_pause_display(self) -> None:
        """绘制自由输入暂停页。"""
        self._draw_status()
        self.pause_title.draw()
        self.pause_subtitle.draw()

    def _prepare_cued_rest_display(self, block_id: int, completed_in_block: int) -> None:
        """设置提示测试定时休息页的进度文字。"""
        self.header.text = (f"Block {block_id}/{self.cfg.blocks}  ·  "
                            f"{completed_in_block}/{len(TARGETS)} targets complete")
        self.footer.text = "No flicker  ·  ESC: stop"

    def _draw_cued_rest_display(self) -> None:
        """绘制提示测试定时休息页。"""
        self._draw_status()
        self.rest_title.draw()
        self.rest_timer.draw()
        self.rest_remaining_label.draw()
        self.rest_auto_label.draw()

    def _prepare_cued_result_feedback(self, record: TrialRecord, result: dict[str, Any],
                                      target: Target, mains: dict[str, Any]) -> None:
        """根据真实目标和解码结果设置提示测试反馈内容。"""
        pred = TARGETS[result['prediction'] - 1]
        fb_pred = TARGETS[result['fbcca_prediction'] - 1]
        cca_pred = TARGETS[result['cca_prediction'] - 1]
        correct = pred.class_id == target.class_id
        self.header.text = f"Trial {record.trial_id} of {self.cfg.blocks*len(TARGETS)}  ·  Result"
        self.footer.text = "Continuing automatically  ·  ESC: stop"
        self.result_verdict.text = "CORRECT" if correct else "WRONG"
        self.result_verdict.color = "#92E5B6" if correct else "#FF9A94"
        self.result_pair.text = f"Target {target.symbol}   →   Recognized {pred.symbol}"
        self.result_frequency.text = (f"{target.frequency_hz:.1f} Hz" if correct else
            f"Target {target.frequency_hz:.1f} Hz  ·  Recognized {pred.frequency_hz:.1f} Hz")
        self.result_details.text = (f"FBCCA  {fb_pred.symbol}  ({fb_pred.frequency_hz:.1f} Hz)\n"
                                    f"CCA  {cca_pred.symbol}  ({cca_pred.frequency_hz:.1f} Hz)")
        self.result_warning.text = ("High mains interference: check electrode contact" if
                                    mains.get("line_dominated") else "")

    def _show_cued_result_feedback(self, record: TrialRecord, result: dict[str, Any],
                                   target: Target, mains: dict[str, Any]) -> None:
        """在配置的反馈时长内持续显示提示测试结果。"""
        self._prepare_cued_result_feedback(record, result, target, mains)
        started = self.clock()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            self._draw_cued_result_feedback()
            self.win.flip()
            if self.clock() - started >= self.cfg.feedback_s:
                return

    def _prepare_processing_status(self, record: TrialRecord) -> None:
        """根据自由输入或提示测试模式准备处理状态页。"""
        free = record.mode == "free"
        self.header.text = (f"Selection {record.trial_id}" if free else
                            f"Trial {record.trial_id} of {self.cfg.blocks*len(TARGETS)}")
        self.footer.text = "SPACE: pause  ·  ESC: stop" if free else "ESC: stop"
        target = None if free else TARGETS[record.true_class - 1]
        self.processing_key_label.text = ("" if target is None else
                                          target.display_label or "SPACE")
        self.processing_key_label.height = (self._processing_key_size*.25 if
                                            target is not None and target.symbol == "SPACE" else
                                            self._processing_key_size*.43)
        positions = (self._processing_free_positions if free else
                     self._processing_cued_positions)
        self.processing_dot.pos, self.processing_title.pos, self.processing_subtitle.pos = positions

    def _draw_startup_check(self) -> None:
        """绘制无闪烁的启动显示检查页。"""
        self._draw_keys()
        self.startup_dot.draw()
        self.startup_title.draw()
        self.startup_subtitle.draw()

    def _set_cue_label_target(self, target_id: Optional[int]) -> None:
        """仅在目标变化时更新键帽文字的提示配色。"""
        if target_id == self._label_cue_target_id:
            return
        for index, label in enumerate(self.labels, start=1):
            label.color = (self._static_label_color if target_id is None or index == target_id
                           else self._cue_label_color)
        self._label_cue_target_id = target_id

    def _draw_keys(self, target_id: Optional[int] = None, flicker_rgb: Optional[np.ndarray] = None,
                   show_outline: bool = False, dim_others: bool = False) -> None:
        """绘制静态或闪烁键盘，并可突出显示指定目标。"""
        if self.static_only and flicker_rgb is not None:
            raise RuntimeFault("DISPLAY", "静态自检禁止呈现闪烁帧")
        self._set_cue_label_target(target_id if dim_others and flicker_rgb is None else None)
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
        self.event.getKeys(keyList=["space"])
        self._phase(record, "CUE")
        self._check_abort()
        self._draw_keys(target.class_id, show_outline=True, dim_others=True)
        self.win.callOnFlip(self.markers.mark, cue_event)
        self.win.flip()
        record.cue_onset = cue_event.timestamp
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
        self._set_cue_label_target(None)
        self.footer.text = f"{progress} | Keep looking at the key center | ESC: stop"

    def show_message(self, text: str, min_duration_s: float = 0.0, wait_space: bool = False) -> None:
        """显示消息至少指定时长，并可等待用户按空格继续。"""
        self.message.text = text
        self._sync_output_text()
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
        """将当前 LSL 时钟时间记录为试次阶段切换点。"""
        record.phase_times.append((name, self.clock()))

    def _persist_trial(self, record: TrialRecord) -> None:
        """在启用记录器时立即保存已结束试次。"""
        if self.recorder is None:
            return
        try:
            self.recorder.write_trial(record)
        except Exception as exc:
            raise RuntimeFault("SAVE", f"逐试次保存失败；保留 {self.recorder.session_dir}：{exc}") from exc

    def _note_fault(self, exc: BaseException, *, fatal: bool, default: str = "DISPLAY") -> None:
        """登记并打印试次故障，必要时标记为会话致命错误。"""
        fault = fault_record(exc, default)
        self.trial_faults.append(fault)
        if fatal:
            self.fatal_fault = fault
        print_fault(fault)

    @staticmethod
    def _attach_failed_epoch(record: TrialRecord, exc: BaseException) -> None:
        """将异常携带的部分事件窗附加到试次，以便失败后诊断。"""
        epoch = getattr(exc, "epoch", None)
        if epoch is not None and record.result is None:
            record.result = {"trial_id": record.trial_id, "epoch": epoch,
                             "quality": epoch.diagnostics.get("quality")}

    def _rejection_diagnostics(self, result: dict[str, Any]) -> dict[str, Any]:
        """评估自由输入证据门控，但不把分类分数解释为概率。"""
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
        """仅为自由输入模式计算拒识诊断，并按配置决定是否应用阈值。"""
        return self._rejection_diagnostics(result) if mode == "free" else None

    @staticmethod
    def _rejection_calibration_notice() -> str:
        """返回自由输入拒识阈值尚未标定时的安全提示。"""
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
        """保持静态状态页，直到指定的 LSL 时钟时间。"""
        while self.clock() < when:
            self._check_abort()
            self._draw_status()
            self.win.flip()

    def _present_trial_stimulus(self, record: TrialRecord, cue_event: EventStamp,
                                onset_event: EventStamp, offset_event: EventStamp) -> None:
        """呈现准备和闪烁阶段，并记录事件、逐帧时间及显示诊断。"""
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
            stimulus_frames = self.rgb_frames_by_mode["free" if free else "cued"]
            for frame, rgb in enumerate(stimulus_frames):
                self._check_abort()
                self._draw_keys(target_id, rgb, show_outline=not free)
                if frame == 0:
                    self.win.callOnFlip(self.markers.mark, onset_event)
                self.win.callOnFlip(lambda: record.frame_lsl_times.append(self.clock()))
                flip_time = self.win.flip()
                record.frame_flip_times.append(float(flip_time))
                if frame == 0:
                    record.stimulus_onset = onset_event.timestamp
            self.win.callOnFlip(self.markers.mark, offset_event)
            self.win.callOnFlip(lambda: record.frame_lsl_times.append(self.clock()))
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
        """异步等待分类及最短空白期结束，并校验返回结果归属。"""
        cfg = self.cfg
        self._decode_future = self.executor.submit(decode_trial, record.trial_id, onset_event.timestamp,
                                                   self.collector, cfg, self._active_decode_cancel)
        while not self._decode_future.done() or self.clock() < offset_event.timestamp + cfg.blank_min_s:
            self._check_abort()
            if self.clock() < offset_event.timestamp + cfg.blank_min_s:
                self._draw_status()
            else:
                self._draw_processing_status(record)
            self.win.flip()
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
        """执行拒识门控、写入字符并显示与模式匹配的反馈。"""
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
        if free:
            self.header.text = f"FREE | Selection {record.trial_id} | Recognized: {pred.symbol}"
            feedback = (f"Recognized: {pred.symbol} ({pred.frequency_hz:.1f} Hz)\n\n"
                        "Choose your next character in the next round.\n"
                        "Keyboard SPACE: pause   ESC: exit")
        else:
            self._show_cued_result_feedback(record, result, target, mains)
            return
        if mains.get("line_dominated"):
            feedback += "\nHigh mains interference: check electrode / reference / BIAS contact."
        self.show_message(feedback, self.cfg.feedback_s)

    def _run_trial(self, record: TrialRecord) -> None:
        """运行单个试次的准备、刺激、时序质控、解码和反馈流程。"""
        cfg = self.cfg
        self.collector.check_health()
        self._graphics_gc.collect(self.win)
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
            self.footer.text = "Keyboard SPACE: pause   ·   ESC: exit   ·   <-: delete"
        else:
            self.header.text = f"Block {record.block_id}/{cfg.blocks} | Trial {record.trial_id}/{cfg.blocks*40} | Look at {target.symbol}"
            self.footer.text = "Follow the highlighted key; trials advance automatically | ESC: stop"
        self._sync_output_text()
        details = {"mode": record.mode, "window_s": cfg.window_s,
                   "response_delay_s": cfg.response_delay_s,
                   "minimum_stimulus_s": (cfg.stimulus_s if free else cfg.cued_stimulus_s),
                   "continuous_eeg_saved_from_stimulus_onset": True}
        if not free:
            details["offline_max_delay_s"] = cfg.cued_offline_max_delay_s
        if target is not None:
            details["frequency_hz"] = target.frequency_hz
        cue_event = EventStamp("ready_onset" if free else "cue_onset", record.trial_id, target_id,
            {} if free else {"cue_s": cfg.cue_s, "progression": "automatic",
                             "settle_s": cfg.cued_settle_s})
        onset_event = EventStamp("stimulus_onset", record.trial_id, target_id, details)
        offset_event = EventStamp("stimulus_offset", record.trial_id, target_id)
        self._present_trial_stimulus(record, cue_event, onset_event, offset_event)
        self._phase(record, "BLANK")
        self._prepare_processing_status(record)
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
        result = self._await_trial_result(record, onset_event, offset_event)
        self._apply_trial_result(record, result, target)

    def _draw_mode_menu(self, selected: str) -> None:
        """绘制模式选择菜单并标出当前选项。"""
        self.mode_title.draw()
        self.mode_outline.pos = self.mode_positions[selected]
        self.mode_outline.draw()
        for label in self.mode_labels.values():
            label.draw()
        self.mode_hint.draw()
        self.mode_warning.draw()

    def _select_mode(self) -> str:
        """等待用户选择自由输入或提示测试模式。"""
        self._pause_enabled = False
        selected = self.cfg.session_mode
        self.event.getKeys(keyList=["1", "2", "num_1", "num_2", "space", "return", "num_enter"])
        self.output_box.autoDraw = False
        self.output_text.autoDraw = False
        try:
            while True:
                self._check_abort()
                keys = self.event.getKeys(keyList=["1", "2", "num_1", "num_2", "space", "return", "num_enter"])
                if "1" in keys or "num_1" in keys:
                    selected = "free"
                if "2" in keys or "num_2" in keys:
                    selected = "cued"
                self._draw_mode_menu(selected)
                self.win.flip()
                if any(k in keys for k in ("space", "return", "num_enter")):
                    self.cfg.session_mode = selected
                    print("\nSelected mode: " + ("Free spelling" if selected == "free" else "Cued test"), flush=True)
                    return selected
        finally:
            self.output_box.autoDraw = True
            self.output_text.autoDraw = True

    def run(self) -> str:
        """进入模式选择并运行所选会话，返回结束原因。"""
        if self.static_only:
            raise RuntimeFault("DISPLAY", "静态自检不能启动实验")
        mode = self._select_mode()
        return self._run_free() if mode == "free" else self._run_cued()

    def _wait_free_pause(self, message: str) -> None:
        """暂停自由输入，等待解码任务结束并由用户恢复。"""
        self._pause_enabled = False
        self._cancel_decode()
        self.event.getKeys(keyList=["space", "return", "num_enter"])
        started = self.clock()
        self.markers.mark(EventStamp("pause", 0, details={"typed_text": self.ledger.typed_text}))
        self._prepare_free_pause_display(message)
        self._sync_output_text()
        self.win.recordFrameIntervals = False
        while True:
            self._check_abort()
            keys = self.event.getKeys(keyList=["space", "return", "num_enter"])
            ready = self._decode_future is None or self._decode_future.done()
            self._draw_free_pause_display()
            self.win.flip()
            if ready and self.clock() - started >= .35 and keys:
                self._decode_future = None
                self._active_decode_cancel = None
                self._pause_enabled = True
                self.markers.mark(EventStamp("resume", 0))
                return

    @staticmethod
    def _release_free_eeg(record: TrialRecord) -> None:
        """在试次已持久化后释放自由输入记录中的大型数组。"""
        if record.result is not None:
            record.result.pop("epoch", None)
            record.result.pop("history_notched_data", None)
        record.frame_flip_times.clear()
        record.frame_lsl_times.clear()
        record.frame_diagnostics.pop("intervals_s", None)
        record.frame_diagnostics.pop("psychopy_frame_intervals_s", None)

    @staticmethod
    def _advance_free_invalid_streak(current: int, record: TrialRecord,
                                     *, pause_requested: bool,
                                     exit_requested: bool) -> int:
        """仅累计真实失败试次；正常拒识不视为故障。"""
        if (pause_requested or exit_requested or record.text_applied
                or record.status == "rejected"):
            return 0
        return current + 1

    def _run_free(self) -> str:
        """循环运行自由输入选择，处理暂停、拒识和连续无效保护。"""
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
                        self.win.flip()
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
                    print(f"FREE {selection_id:04d}: rejected | {_brief_reason(record.reason)}", flush=True)
                else:
                    print(f"FREE {selection_id:04d}: no character added | {_brief_reason(record.reason)}", flush=True)
                consecutive_invalid = self._advance_free_invalid_streak(
                    consecutive_invalid, record,
                    pause_requested=pause_requested, exit_requested=exit_requested)
                if exit_requested or fatal:
                    return record.reason or "Free spelling stopped"
                if pause_requested:
                    self._wait_free_pause("Your text is kept. Unfinished selection discarded.")
                    consecutive_invalid = 0
                elif consecutive_invalid >= self.cfg.max_consecutive_invalid:
                    self._wait_free_pause("Several selections could not be decoded.\n"
                                          "No character was inserted; check the EEG stream before resuming.")
                    consecutive_invalid = 0
        finally:
            self._pause_enabled = False
            self._cancel_decode()

    def _run_cued(self) -> str:
        """按随机计划运行提示测试，并处理重试、分段休息和终止。"""
        cfg = self.cfg
        self._pause_enabled = False
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
                print(f"[模式2轮间休息] 第{previous_block}轮完成，休息{cfg.block_rest_s:g}秒。",
                      flush=True)
                self._wait_cued_rest(previous_block, completed_in_block,
                                      duration_s=cfg.block_rest_s)
                previous_block = block_id
                completed_in_block = 0
            record = TrialRecord(trial_id, block_id, target_id, start=self.clock())
            fatal = False
            try:
                self._run_trial(record)
            except (AbortSession, KeyboardInterrupt) as exc:
                record.status = "valid" if record.text_applied else "aborted"
                record.reason = str(exc) or "KeyboardInterrupt"
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
                    self.win.flip()
                except Exception as clear_error:
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
                print(f"[模式2休息] 已完成{completed_in_block}个目标，休息{cfg.cued_rest_s:g}秒。",
                      flush=True)
                self._wait_cued_rest(block_id, completed_in_block)
        else:
            stop_reason = "Completed planned trials"
        return stop_reason

    def _wait_cued_rest(self, block_id: int, completed_in_block: int,
                        *, duration_s: Optional[float] = None) -> None:
        """静态休息期间持续显示剩余时间；最后3、2、1秒各响一次。"""
        duration = self.cfg.cued_rest_s if duration_s is None else duration_s
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("休息时长必须非负且有限")
        self._prepare_cued_rest_display(block_id, completed_in_block)
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
                minutes, seconds = divmod(seconds_left, 60)
                timer_text = f"{minutes:02d}:{seconds:02d}"
                if self.rest_timer.text != timer_text:
                    self.rest_timer.text = timer_text
                if duration >= 3 and seconds_left in (3, 2, 1):
                    if seconds_left != last_beep_second:
                        try:
                            self._rest_beep.play()
                        except Exception as exc:
                            raise RuntimeFault("AUDIO", f"模式2休息提示音播放失败：{exc}") from exc
                        last_beep_second = seconds_left
                self._draw_cued_rest_display()
                self.win.flip()
        finally:
            if self._rest_beep is not None:
                self._rest_beep.stop()

    def _release_visual_stimuli(self) -> None:
        """先释放刺激及其循环引用，再销毁它们所属的 GL 上下文。"""
        from psychopy.visual.basevisual import MinimalStim

        for name, value in list(vars(self).items()):
            if isinstance(value, MinimalStim):
                value.autoDraw = False
                setattr(self, name, None)
            elif isinstance(value, (list, dict)):
                items = list(value.values()) if isinstance(value, dict) else value
                if items and all(isinstance(item, MinimalStim) for item in items):
                    for item in items:
                        item.autoDraw = False
                    value.clear()

    def close(self) -> None:
        """取消解码、关闭执行器，并在主线程释放视觉资源和窗口。"""
        MainThreadGraphicsGC._check_thread()
        if self._closed:
            return
        self.cancel.set()
        self._cancel_decode()
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
            if self.win is not None:
                try:
                    self.win.recordFrameIntervals = False
                    self.win._setCurrent()
                    self._release_visual_stimuli()
                    self._graphics_gc.collect(self.win)
                finally:
                    self.win.close()
                    self.win = None
        finally:
            self._closed = True
            self._graphics_gc.close()


def static_ui_self_test(cfg: Config, *, seconds: float = 5.0) -> dict:
    """渲染正式键盘、固定提示和输出框，但不启动 LSL 或解码。"""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("自检时长必须为有限正数")
    app = None
    result = {"status": "cancelled", "exit_code": 130,
              "eeg_connected": False, "flicker_presented": False,
              "scope": "静态布局、字体、窗口与软件刷新时序；不验证EEG、识别准确率或物理显示延迟"}
    try:
        app = PsychoPyKeyboard(replace(cfg), ContinuousLSL(cfg), None, TrialLedger(), static_only=True)
        app.output_box.autoDraw = True
        app.output_text.autoDraw = True
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
    cfg = replace(cfg if cfg is not None else CONFIG)
    cfg.validate()
    print(f"[算法] {cfg.filter_bank_profile}；{len(cfg.filter_bands)}个子带，"
          f"上限{max(high for _, high in cfg.filter_bands):g}Hz，{cfg.n_harmonics}次谐波；"
          f"权重 a={cfg.weight_a:g}, b={cfg.weight_b:g}；分析窗{cfg.window_s:g}秒。\n"
          f"当前{len(cfg.channel_indices)}路设备/键盘时序适配不等同论文完整实验复现；"
          "LSL声明采样率不能独立证明硬件有效带宽。", flush=True)
    if cfg.notch_mode == "history":
        print(f"[预处理] history：{cfg.notch_history_s:g}秒历史辅助{cfg.notch_hz:g}Hz陷波，"
              f"分类仍使用{cfg.window_s:g}秒；上下文不足即拒绝，不能加载旧个人模板。"
              "此模式需新的在线数据验证准确率。", flush=True)
    else:
        print("[预处理] epoch：沿用当前分析窗独立陷波。", flush=True)
    calibration = None
    if cfg.calibration_file:
        model_path = Path(cfg.calibration_file)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        calibration = PersonalCalibration.load(model_path, cfg)
        print(f"个人模型已加载：受试者 {cfg.participant_id}，"
              f"每类 {calibration.counts.min()}–{calibration.counts.max()} 次校准，"
              f"模型 {calibration.model_id[:12]}", flush=True)
    print("[1/3] 运行环境由启动入口核验；准备保存目录检查…", flush=True)
    print(f"安全提示：此程序呈现{BENCHMARK_FREQUENCIES_HZ.min():g}–{BENCHMARK_FREQUENCIES_HZ.max():g}Hz视觉闪烁。"
          "对闪光敏感、有光敏性癫痫史者不要自行测试；"
          "出现眼部不适、头痛、眩晕等应立即按ESC停止。先在OpenBCI GUI检查电极与接线，程序不会直接开始闪烁。")
    collector = ContinuousLSL(cfg)
    collector.calibration = calibration
    ledger = TrialLedger()
    markers: Optional[EventMarkers] = None
    app: Optional[PsychoPyKeyboard] = None
    recorder: Optional[SessionRecorder] = None
    review: dict = {}
    reason = "Not started"
    errors: list[dict] = []
    stage = "SAVE"

    def add_error(exc: BaseException, default: str) -> None:
        """将异常登记为标准故障并立即输出。"""
        fault = fault_record(exc, default)
        errors.append(fault)
        print_fault(fault)

    try:
        probe_save_directory(cfg.record_root)
        stage = "SAVE"
        recorder = SessionRecorder(cfg, context={"mode_at_start": cfg.session_mode,
                                                  "trained_model": ({
                                                      "model_id": calibration.model_id,
                                                      **calibration.metadata,
                                                  } if calibration is not None else None)})
        collector.recorder = recorder
        print("\n[2/3] 连接EEG LSL并检查采样时间轴（请保持采集软件的LSL输出开启）…", flush=True)
        stage = "LSL_CONNECTION"
        collector.start()
        stage = "ACQUISITION"
        review = preflight_transport(cfg, collector)
        if calibration is not None:
            calibration.validate_config(cfg)
        stage = "LSL_CONNECTION"
        markers = EventMarkers(cfg, collector.clock)
        markers.mark(EventStamp("session_start", 0, details={"mode_at_start": cfg.session_mode}))
        stage = "INTERNAL"
        t = np.arange(sample_count(cfg.window_s, cfg.target_fs)) / cfg.target_fs
        warm = np.tile(np.sin(2*np.pi*10*t), (len(cfg.channel_indices), 1))
        warm_context = None
        if cfg.notch_mode == "history":
            count = round(cfg.notch_history_s * cfg.target_fs)
            history_t = np.arange(count + len(t)) / cfg.target_fs
            history_x = np.tile(np.sin(2*np.pi*10*history_t), (len(cfg.channel_indices), 1))
            warm = history_x[:, count:].copy()
            warm_context = NotchContext(
                data=history_x, local_timestamps=history_t,
                uniform_timestamps=history_t[count:].copy(), source_offset=count,
                epoch_end=count/cfg.target_fs + cfg.window_s,
                requested_history_samples=count)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            classify_eeg_window(warm, cfg.target_fs, cfg.window_s, target_fs=cfg.target_fs,
                notch_hz=cfg.notch_hz, n_harmonics=cfg.n_harmonics,
                a=cfg.weight_a, b=cfg.weight_b, regularization=cfg.cca_regularization,
                filter_bands=cfg.filter_bands, line_regression_hz=cfg.line_regression_hz,
                calibration=calibration, notch_context=warm_context)
        print("[3/3] 创建键盘窗口；在菜单按1自由输入、2提示测试，再按SPACE开始…", flush=True)
        stage = "DISPLAY"
        app = PsychoPyKeyboard(cfg, collector, markers, ledger, recorder)
        reason = app.run()
        if app.fatal_fault is not None:
            errors.append(app.fatal_fault)
    except (AbortSession, KeyboardInterrupt) as exc:
        reason = str(exc) or "用户中止"
    except Exception as exc:
        add_error(exc, stage)
        reason = f"{type(exc).__name__}: {exc}"
    finally:
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
                    display_info=app.display_info if app else {},
                    typed_text=ledger.typed_text, stop_reason=reason)
            except Exception as exc:
                add_error(exc, "SAVE")
                reason += f"; 保存失败：{exc}"
                if recorder.archive_path.exists():
                    session_artifact = str(recorder.archive_path)
                    print(f"会话ZIP已保存，但临时文件清理未完成：{type(exc).__name__}: {exc}", flush=True)
                elif recorder.session_path.exists():
                    session_artifact = str(recorder.session_path)
                    print(f"会话NPZ已保存，但配套文件封装未完成：{type(exc).__name__}: {exc}", flush=True)
                else:
                    print(f"实验记录导出失败，已有逐试次文件保留：{type(exc).__name__}: {exc}", flush=True)
        if collector.error is not None and not any(f["code"] == "ACQUISITION" for f in errors):
            add_error(collector.error, "ACQUISITION")
        if (recorder is not None and recorder.continuous is not None
                and recorder.continuous.error is not None
                and not any(f["code"] == "SAVE" for f in errors)):
            add_error(recorder.continuous.error, "SAVE")
        if markers is not None and markers.error is not None and not any(f["code"] == "LSL_CONNECTION" for f in errors):
            add_error(markers.error, "LSL_CONNECTION")
        LAST_SESSION = {"mode": cfg.session_mode, "config": cfg, "targets": TARGETS, "records": ledger.records,
            "events": markers.events if markers else [], "acquisition_metadata": collector.metadata,
            "acquisition_review": review, "clock_updates": collector.clock_updates,
            "display_info": app.display_info if app else {}, "summary": report,
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


def observe_lsl_timing(cfg: Config, collector: ContinuousLSL, seconds: float,
                       *, report: Optional[dict] = None) -> dict:
    """预热后每秒检查最近2秒；保留中途异常，不能只用末窗宣称全程通过。

    只读接收缓冲，不开启闪烁，不产生准确率。相邻窗口重叠；此诊断
    不能代替设备计数器或物理延迟测量。工频占比仅作描述，不判失败。
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("观察时长必须为有限正数")
    if collector.buffer is None:
        raise RuntimeFault("ACQUISITION", "LSL缓冲区尚未建立")
    result = report if report is not None else {}
    result.update(requested_observation_s=float(seconds), window_s=2.0,
                  check_interval_s=1.0, completed=False, passed=False,
                  observations=[], invalid_window_count=0,
                  scope="每秒检查最近2秒重叠窗；首窗包含预热末段；不测物理延迟或识别准确率")
    started = time.monotonic()
    deadline = started + seconds
    next_check = started
    last_end = -math.inf
    while True:
        collector.check_health()
        now = time.monotonic()
        if now >= next_check or now >= deadline:
            end = collector.buffer.latest
            if end > last_end:
                item = {"elapsed_s": float(now - started),
                        "requested_window_start": float(end - 2.0),
                        "requested_window_end": float(end)}
                try:
                    epoch = collector.buffer.epoch(
                        end - 2.0, 2.0, collector.fs, cfg.max_gap_factor,
                        cfg.rate_tolerance, cfg.hard_gap_factor,
                        cfg.hard_rate_tolerance, cfg.max_warning_gaps,
                        **({"notch_history_s": cfg.notch_history_s} if cfg.notch_mode == "history" else {}))
                    item.update(valid=True, diagnostics=epoch.diagnostics,
                                mains_noise=mains_noise_diagnostics(
                                    epoch.data, epoch.fs, cfg.notch_hz))
                except InvalidTrial as exc:
                    item.update(valid=False, reason=str(exc),
                                diagnostics=exc.epoch.diagnostics if exc.epoch is not None else {})
                    result["invalid_window_count"] += 1
                    print(f"[LSL持续检查 {now - started:.1f}s] 窗口异常：{exc}", flush=True)
                result["observations"].append(item)
                last_end = end
            next_check = now + 1.0
        if now >= deadline:
            break
        collector.stop_event.wait(max(0.0, min(.05, deadline - now, next_check - now)))
    result.update(completed=True, elapsed_s=float(time.monotonic() - started),
                  checked_window_count=len(result["observations"]))
    result["passed"] = bool(result["observations"] and result["invalid_window_count"] == 0)
    fractions = [item["mains_noise"]["median_line_fraction"]
                 for item in result["observations"]
                 if item["valid"] and item["mains_noise"]["available"]]
    result["median_line_fraction_across_windows"] = float(np.median(fractions)) if fractions else None
    print(f"[LSL持续检查] {result['checked_window_count']}个重叠窗，"
          f"{result['invalid_window_count']}个时间轴异常窗；"
          "工频逐通道占比和每窗诊断写入检查报告。", flush=True)
    return result


def positive_seconds(value: str) -> float:
    """将命令行文本解析为有限正秒数。"""
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
                       help="不显示闪烁，连接EEG并检查LSL时间轴及滤波前工频占比")
    modes.add_argument("--self-test-ui", action="store_true", help="运行静态键盘自检")
    parser.add_argument("--seconds", type=positive_seconds, default=5.0,
                        help="静态自检展示或LSL预热后持续检查的秒数")
    parser.add_argument("--warmup-seconds", type=positive_seconds,
                        help="EEG预热秒数；普通启动默认10秒，时间轴诊断默认30秒；"
                             "history模式至少收齐4秒历史及2.1秒检查数据")
    parser.add_argument("--windowed", action="store_true", help="使用1280×800窗口")
    parser.add_argument("--screen", type=int, help="显示器编号，从0开始")
    parser.add_argument("--record-root", help="会话保存目录")
    for signal in SYNC_CHANNELS:
        parser.add_argument("--" + signal.replace("_", "-") + "-channel", type=int,
            help=f"{signal}在同一EEG LSL流内的列号（从0开始）；未提供时记为未获取")
    parser.add_argument("--timing-calibration", help="随会话保存的实测时间校准JSON；不自动应用补偿")
    parser.add_argument("--fbcca-profile", choices=("m3", "line_robust"),
                        help="m3=论文7子带/90Hz/5谐波，需原始采样率>180Hz；"
                             "line_robust=48Hz/3子带/3谐波适配方案")
    parser.add_argument("--notch-mode", choices=("epoch", "history"),
                        help="epoch=原短窗陷波；history=使用前4秒真实源数据辅助陷波，"
                             "分析窗不变，仅支持无个人模板的FBCCA，需独立实测")
    parser.add_argument("--protocol", choices=("standard_2s", "standard_5s", "paper_window_1p25s"),
                        help="分析窗长：2秒、5秒或论文1.25秒；仅改变窗长，不复刻论文完整时序")
    parser.add_argument("--seed", type=int, help="提示目标随机种子；不同复测可使用不同非负整数")
    parser.add_argument("--participant", help="匿名受试者编号；加载个人模型时必填")
    parser.add_argument("--calibration", help="个人校准模型NPZ、完整模式2会话PKL或ZIP路径（受试者与配置须匹配）")
    args = parser.parse_args(argv)
    if args.screen is not None and args.screen < 0:
        parser.error("screen 必须非负")
    if args.seed is not None and args.seed < 0:
        parser.error("seed 必须非负")
    report = {
        "mode": "static_ui" if args.self_test_ui else
                "lsl_timing" if args.check_lsl_timing else "diagnose" if args.diagnose else "eeg",
        "status": "failed", "errors": [], "exit_code": 0,
    }
    try:
        print("[启动] 正在加载运行依赖…", flush=True)
        report["checks"] = check_environment(static_ui=args.self_test_ui)
        cfg = replace(CONFIG)
        if args.fbcca_profile:
            cfg = configure_fbcca_profile(cfg, args.fbcca_profile)
        if args.protocol:
            cfg.protocol = args.protocol
        if args.notch_mode:
            cfg.notch_mode = args.notch_mode
        if args.seed is not None:
            cfg.random_seed = args.seed
        if args.windowed:
            cfg.full_screen, cfg.window_size = False, (1280, 800)
        if args.screen is not None:
            cfg.screen_index = args.screen
        if args.record_root:
            cfg.record_root = args.record_root
        for signal in SYNC_CHANNELS:
            index = getattr(args, signal + "_channel")
            if index is not None:
                setattr(cfg, signal + "_channel", index)
        if args.timing_calibration:
            cfg.timing_calibration_file = args.timing_calibration
        if args.participant:
            cfg.participant_id = args.participant
        if args.calibration:
            cfg.calibration_file = args.calibration
        if args.warmup_seconds is not None:
            cfg.warmup_s = args.warmup_seconds
        elif args.check_lsl_timing:
            cfg.warmup_s = max(cfg.warmup_s, 30.0)
        if args.warmup_seconds is not None or args.check_lsl_timing:
            cfg.startup_timeout_s = max(cfg.startup_timeout_s, cfg.startup_buffer_s + 15.0)
        cfg.validate()
        if args.self_test_ui:
            report.update(static_ui_self_test(cfg, seconds=args.seconds))
        elif args.diagnose:
            report["checks"]["save_directory"] = probe_save_directory(cfg.record_root)
            report["status"] = "passed"
            print("环境诊断通过；尚未验证 EEG 流、真实采样或显示窗口。", flush=True)
        elif args.check_lsl_timing:
            collector = ContinuousLSL(cfg)
            try:
                collector.start()
                observation = report["checks"]["continuous_observation"] = {}
                observe_lsl_timing(cfg, collector, args.seconds, report=observation)
                collector.check_health()
                report["checks"]["acquisition_review"] = preflight_transport(cfg, collector)
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
                if not observation["passed"]:
                    raise RuntimeFault("ACQUISITION",
                        f"观察期间有{observation['invalid_window_count']}个异常窗；"
                        "完整逐窗诊断已保留，不能仅凭最后窗口通过而判定全程通过")
            finally:
                collector.close()
            report["status"] = "passed"
            print(f"LSL时间轴检查通过：{collector.metadata['name']}；"
                  f"估计{collector.estimated_fs:.3f}Hz；最新样本年龄{age:.3f}s。"
                  "此检查不测量设备到屏幕的物理延迟。", flush=True)
        else:
            result = main(cfg=cfg)
            report.update({key: result[key] for key in
                           ("exit_code", "errors", "trial_faults", "stop_reason", "session_artifact")})
            report["status"] = "failed" if report["exit_code"] else "finished"
    except KeyboardInterrupt:
        report.update(status="cancelled", exit_code=130)
    except Exception as exc:
        report["status"] = "failed"
        fault = fault_record(exc)
        report["errors"].append(fault)
        report["exit_code"] = fault["exit_code"]
        print_fault(fault)
    if write_diagnostic(report) is None and not report["exit_code"]:
        report["exit_code"] = 50
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(cli())
