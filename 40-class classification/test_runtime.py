"""Run with python -B -m unittest test_runtime; no EEG, native display, or network."""
import contextlib
from dataclasses import replace
import io
import itertools
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import fbcca_keyboard_dual_mode as keyboard
import run_keyboard
import runtime_support as runtime


class EnvironmentTests(unittest.TestCase):
    def test_locks_separate_platform_dependencies(self):
        windows = runtime.pinned_packages(runtime.ROOT / "requirements/windows-py310.lock")
        mac = runtime.pinned_packages(runtime.ROOT / "requirements/macos-py310.lock")
        self.assertEqual(windows["pyglet"], "1.4.11")
        self.assertEqual(mac["pyglet"], "1.5.27")
        self.assertIn("pyobjc-core", mac)
        for name in ("pypiwin32", "pywin32", "pywinhook", "pyparallel"):
            self.assertNotIn(name, mac)
        self.assertIn("pywinhook", windows)

    def test_missing_dependency_is_read_only(self):
        with patch.object(runtime, "pinned_packages", return_value={"numpy": "2.2.6"}), \
             patch.object(runtime.metadata, "version", side_effect=runtime.metadata.PackageNotFoundError), \
             patch.object(runtime.importlib, "import_module") as imports:
            with self.assertRaises(runtime.RuntimeFault) as error:
                runtime.check_environment()
        self.assertEqual(error.exception.code, "ENVIRONMENT")
        imports.assert_not_called()

    def test_ui_check_never_imports_lsl(self):
        with patch.object(runtime, "pinned_packages", return_value={"pylsl": "1.18.2"}), \
             patch.object(runtime.metadata, "version", side_effect=AssertionError("Must skip pylsl")), \
             patch.object(runtime.importlib, "import_module") as imports:
            runtime.check_environment(static_ui=True)
        self.assertNotIn("pylsl", [call.args[0] for call in imports.call_args_list])

    def test_native_library_failure_has_distinct_code(self):
        def import_module(name):
            if name == "pylsl":
                raise OSError("wrong architecture or missing liblsl")
        with patch.object(runtime, "pinned_packages", return_value={}), \
             patch.object(runtime.importlib, "import_module", side_effect=import_module):
            with self.assertRaises(runtime.RuntimeFault) as error:
                runtime.check_environment()
        self.assertEqual(error.exception.code, "LSL_LIBRARY")

    def test_diagnostic_falls_back_when_project_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = root / "file"
            blocked.write_text("not a directory")
            with patch.object(runtime.tempfile, "gettempdir", return_value=tmp), \
                 patch.object(runtime, "environment_snapshot", return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                path = runtime.write_diagnostic({"errors": [{"code": "DISPLAY"}]}, blocked)
            self.assertEqual(json.loads(Path(path).read_text())["errors"][0]["code"], "DISPLAY")


class AcquisitionTests(unittest.TestCase):
    def test_no_stream_reports_not_found(self):
        fake = SimpleNamespace(local_clock=lambda: 0, resolve_byprop=Mock(return_value=[]),
                               resolve_streams=Mock(return_value=[]))
        with patch.dict("sys.modules", pylsl=fake):
            with self.assertRaises(runtime.RuntimeFault) as error:
                keyboard.ContinuousLSL(keyboard.CONFIG).start()
        self.assertEqual(error.exception.code, "LSL_NOT_FOUND")

    def test_stalled_source_reports_acquisition(self):
        collector = keyboard.ContinuousLSL(replace(keyboard.CONFIG, clock_refresh_s=1000,
                                                  max_receive_age_s=.5))
        collector.inlet = SimpleNamespace(pull_chunk=lambda **kw: ([], []))
        collector.buffer = keyboard.TimestampBuffer(1024, 8)
        collector.fs = 250
        with patch.object(keyboard.time, "monotonic", side_effect=itertools.count()):
            collector._run(0.0)
        with self.assertRaises(runtime.RuntimeFault) as error:
            collector.check_health()
        self.assertEqual(error.exception.code, "ACQUISITION")
        self.assertIn("停止发送", str(error.exception))


class SessionFaultTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.cfg = replace(keyboard.CONFIG, record_root=self.tmp)

    def test_save_probe_failure(self):
        target = Path(self.tmp) / "occupied"
        target.write_text("keep me")
        with self.assertRaises(runtime.RuntimeFault) as error:
            runtime.probe_save_directory(target)
        self.assertEqual(error.exception.code, "SAVE")
        self.assertEqual(target.read_text(), "keep me")

    def test_trial_save_failure_is_not_display_failure(self):
        app = object.__new__(keyboard.PsychoPyKeyboard)
        app.recorder = SimpleNamespace(session_dir=Path(self.tmp),
                                       write_trial=Mock(side_effect=PermissionError("disk full")))
        with self.assertRaises(runtime.RuntimeFault) as error:
            app._persist_trial(keyboard.TrialRecord(1, 1, 1))
        self.assertEqual(error.exception.code, "SAVE")

    def test_caught_display_error_is_fatal_in_both_modes(self):
        for mode in ("free", "cued"):
            with self.subTest(mode=mode):
                app = object.__new__(keyboard.PsychoPyKeyboard)
                app.cfg, app.ledger = replace(self.cfg, session_mode=mode), keyboard.TrialLedger()
                app.clock = lambda: 1.0
                app.collector = SimpleNamespace(error=None)
                app.markers = SimpleNamespace(error=None, mark=Mock())
                app.win = SimpleNamespace(flip=Mock(), recordFrameIntervals=False)
                app.recorder, app.fatal_fault, app.trial_faults = None, None, []
                app.cancel = threading.Event()
                app._run_trial = Mock(side_effect=RuntimeError("GL context lost"))
                app._draw_status, app._cancel_decode = Mock(), Mock()
                (app._run_free if mode == "free" else app._run_cued)()
                self.assertEqual(app.fatal_fault["code"], "DISPLAY")
                self.assertEqual(len(app.ledger.records), 1)
                self.assertEqual(app.ledger.typed_text, "")

    def test_final_save_failure_retains_primary_error_and_files(self):
        sentinel = Path(self.tmp) / "trial_0001.npz"
        sentinel.write_bytes(b"previous trial")
        recorder = SimpleNamespace(session_dir=Path(self.tmp), session_path=Path(self.tmp) / "session.npz",
                                   export_artifacts={}, finalize=Mock(side_effect=OSError("disk full")))
        with patch.object(keyboard, "preview_keyboard_layout", return_value={}), \
             patch.object(keyboard, "SessionRecorder", return_value=recorder), \
             patch.object(keyboard.ContinuousLSL, "start", side_effect=runtime.RuntimeFault("LSL_NOT_FOUND", "no EEG")):
            result = keyboard.main(cfg=self.cfg)
        self.assertEqual(result["exit_code"], 50)
        self.assertEqual([f["code"] for f in result["errors"]], ["LSL_NOT_FOUND", "SAVE"])
        self.assertEqual(sentinel.read_bytes(), b"previous trial")

    def test_display_failure_reaches_session_exit_code(self):
        display_fault = runtime.fault_record(RuntimeError("GL context lost"), "DISPLAY")
        fake_app = SimpleNamespace(run=lambda: "display failed", close=Mock(),
                                   fatal_fault=display_fault, trial_faults=[display_fault], display_info={})
        fake_markers = SimpleNamespace(mark=Mock(), close=Mock(), events=[], error=None)
        with patch.object(keyboard, "preview_keyboard_layout", return_value={}), \
             patch.object(keyboard.ContinuousLSL, "start"), \
             patch.object(keyboard, "preflight_review", return_value={}), \
             patch.object(keyboard, "EventMarkers", return_value=fake_markers), \
             patch.object(keyboard, "classify_eeg_window"), \
             patch.object(keyboard, "PsychoPyKeyboard", return_value=fake_app):
            result = keyboard.main(cfg=self.cfg)
        self.assertEqual(result["exit_code"], 40)
        self.assertEqual(result["errors"][0]["code"], "DISPLAY")
        fake_app.close.assert_called_once()

    def test_saved_trial_files_survive_export_failure(self):
        recorder = keyboard.SessionRecorder(self.cfg)
        recorder.session_dir.mkdir()
        recorder._started = True
        trial_file = recorder.session_dir / "trial_0001.npz"
        trial_file.write_bytes(b"recorded data")
        with patch.object(recorder, "_export_session", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                recorder.finalize(summary={}, events=[], acquisition_metadata={}, acquisition_review={},
                                  clock_updates=[], display_info={}, typed_text="", stop_reason="test")
        self.assertEqual(trial_file.read_bytes(), b"recorded data")
        self.assertEqual(json.loads(recorder.progress_path.read_text(encoding="utf-8"))["export_status"], "failed")


class StaticModeTests(unittest.TestCase):
    def test_static_mode_cannot_enter_experiment_or_flash(self):
        app = object.__new__(keyboard.PsychoPyKeyboard)
        app.static_only = True
        with self.assertRaises(runtime.RuntimeFault):
            app.run()
        with self.assertRaises(runtime.RuntimeFault):
            app._draw_keys(flicker_rgb=object())

    def test_self_test_does_not_start_acquisition_markers_or_save(self):
        fake_app = Mock()
        fake_app.display_info = {"static_render_check": "passed"}
        with patch.object(keyboard, "PsychoPyKeyboard", return_value=fake_app), \
             patch.object(keyboard.ContinuousLSL, "start", side_effect=AssertionError("EEG forbidden")), \
             patch.object(keyboard, "EventMarkers", side_effect=AssertionError("Markers forbidden")), \
             patch.object(keyboard, "SessionRecorder", side_effect=AssertionError("EEG save forbidden")), \
             patch.object(keyboard.time, "monotonic", side_effect=[0, 0, 2]), \
             contextlib.redirect_stdout(io.StringIO()):
            result = keyboard.static_ui_self_test(keyboard.CONFIG, seconds=1)
        self.assertEqual(result["status"], "passed")
        fake_app._draw_keys.assert_called_once_with(1, show_outline=True)
        fake_app.run.assert_not_called()
        fake_app.close.assert_called_once()

    def test_cancelled_self_test_is_not_passed(self):
        with patch.object(keyboard, "PsychoPyKeyboard", side_effect=keyboard.AbortSession("ESC")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = keyboard.static_ui_self_test(keyboard.CONFIG)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["exit_code"], 130)

    def test_invalid_duration_cannot_hang(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(Exception):
                run_keyboard.positive_seconds(value)


if __name__ == "__main__":
    runtime.configure_console()
    unittest.main()
