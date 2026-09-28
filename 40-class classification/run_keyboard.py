"""Safe CLI entry: parse arguments and check the environment before NumPy import."""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
import sys

from runtime_support import (check_environment, configure_console, fault_record, print_fault,
                             probe_save_directory, write_diagnostic)


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds 必须是有限正数")
    return seconds


def cli(argv=None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="40目标 EEG 键盘；只检查依赖，绝不自动安装。")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--diagnose", action="store_true", help="检查固定依赖、liblsl 和保存目录；不连接 EEG/不打开窗口")
    modes.add_argument("--self-test-ui", action="store_true", help="完整静态键盘自检；不加载 LSL、不采样、不闪烁")
    modes.add_argument("--preview-layout", action="store_true", help="交互调整静态布局；不加载 LSL、不闪烁")
    parser.add_argument("--seconds", type=positive_seconds, default=5.0, help="静态自检展示秒数（默认5秒，另加显示初始化时间）")
    parser.add_argument("--windowed", action="store_true", help="使用1280×800窗口；默认全屏")
    parser.add_argument("--screen", type=int, help="显示器编号，从0开始")
    parser.add_argument("--record-root", help="会话保存目录；相对路径以程序目录为基准")
    args = parser.parse_args(argv)
    if args.screen is not None and args.screen < 0:
        parser.error("screen 必须非负")
    report = {"mode": "static_ui" if args.self_test_ui else "layout_preview" if args.preview_layout else "diagnose" if args.diagnose else "eeg",
              "status": "failed", "errors": [], "exit_code": 0}
    try:
        report["checks"] = check_environment(static_ui=args.self_test_ui or args.preview_layout)
        import fbcca_keyboard_dual_mode as keyboard
        cfg = replace(keyboard.CONFIG)
        if args.windowed:
            cfg.full_screen, cfg.window_size = False, (1280, 800)
        if args.screen is not None:
            cfg.screen_index = args.screen
        if args.record_root:
            cfg.record_root = args.record_root
        cfg.validate()
        if args.preview_layout:
            try:
                report["display_info"] = keyboard.preview_keyboard_layout(cfg)
                report["status"] = "finished"
            except keyboard.AbortSession:
                report.update(status="cancelled", exit_code=130)
            except Exception as exc:
                from runtime_support import RuntimeFault
                raise RuntimeFault("DISPLAY", str(exc)) from exc
        elif args.self_test_ui:
            report.update(keyboard.static_ui_self_test(cfg, seconds=args.seconds))
        elif args.diagnose:
            report["checks"]["save_directory"] = probe_save_directory(cfg.record_root)
            report["status"] = "passed"
            print("环境诊断通过；尚未验证 EEG 流、真实采样或显示窗口。", flush=True)
        else:
            result = keyboard.main(cfg=cfg)
            report.update({key: result[key] for key in ("exit_code", "errors", "trial_faults", "stop_reason", "session_artifact")})
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
