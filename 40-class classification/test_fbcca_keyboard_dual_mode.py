"""Hardware-free regression tests for the FBCCA keyboard helpers.

The application module is loaded by path so these tests work even though the
directory name is not a Python package.  No LSL, PsychoPy, display, or device
is required: all fixtures are synthetic NumPy arrays and timestamps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("fbcca_keyboard_dual_mode.py")
SPEC = importlib.util.spec_from_file_location("fbcca_keyboard_dual_mode_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure is reported by unittest
    raise ImportError(f"Cannot load application module from {MODULE_PATH}")
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


class ConfigAndQualityTests(unittest.TestCase):
    def test_default_mode_keeps_strict_checks_enabled(self) -> None:
        cfg = APP.Config()
        self.assertFalse(cfg.quick_entry_mode)
        self.assertEqual(cfg.min_valid_channels, 6)
        self.assertEqual(cfg.max_warning_bad_channels, 2)
        self.assertFalse(cfg.rejection_enabled)

    def test_quality_gate_runs_before_normalisation(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 0.2, size=(8, 200))
        data[0] = 0.0  # constant channel is a warning, not silently discarded
        data[1, 75] = 20.0  # physical amplitude violation

        report = APP.assess_signal_quality(
            data,
            unit="uV",
            max_abs=5.0,
            min_valid_channels=6,
            max_warning_bad_channels=2,
        )

        self.assertTrue(report["hard_fail"])
        self.assertIn(0, report["bad_channels"])
        self.assertTrue(any("幅值" in reason for reason in report["hard_reasons"]))

    def test_unknown_units_are_recorded_as_warning_without_physical_gate(self) -> None:
        data = np.tile(np.linspace(-1.0, 1.0, 100), (8, 1))
        report = APP.assess_signal_quality(data, unit="UNKNOWN", min_valid_channels=6)
        self.assertFalse(report["hard_fail"])
        self.assertTrue(any("单位" in warning for warning in report["warnings"]))

    def test_nan_is_a_hard_quality_failure(self) -> None:
        data = np.ones((8, 32), dtype=float)
        data[0, 0] = np.nan
        report = APP.assess_signal_quality(data, unit="uV")
        self.assertTrue(report["hard_fail"])
        self.assertTrue(any("NaN" in reason for reason in report["hard_reasons"]))

    def test_configured_device_rail_clipping_is_a_hard_failure(self) -> None:
        t = np.arange(64, dtype=float)
        data = np.vstack([np.sin(t / 7.0) for _ in range(8)])
        data[2, 10:12] = 1.0
        report = APP.assess_signal_quality(
            data, unit="uV", rail_min=-1.0, rail_max=1.0, clip_tolerance=0.0)
        self.assertTrue(report["hard_fail"])
        self.assertGreater(report["clip_fraction"][2], 0.0)


class TimestampBufferTests(unittest.TestCase):
    def _samples(self, count: int) -> np.ndarray:
        t = np.arange(count, dtype=float) / 250.0
        return np.vstack([np.sin(2 * np.pi * (8 + i) * t) for i in range(8)]).T

    def test_snapshot_preserves_raw_timestamps_and_per_sample_correction(self) -> None:
        base = 100.0
        first = self._samples(100)
        first_ts = base + np.arange(len(first), dtype=float) / 250.0
        second = self._samples(40)
        second_ts = base + (100 + np.arange(len(second), dtype=float)) / 250.0
        buffer = APP.TimestampBuffer(256, 8)

        buffer.append(first, first_ts, correction=0.010, device_lag_s=0.002)
        buffer.append(second, second_ts, correction=0.011, device_lag_s=0.002)
        samples, raw, local, corrections, valid = buffer.snapshot()

        np.testing.assert_allclose(samples, np.vstack([first, second]))
        np.testing.assert_allclose(raw, np.concatenate([first_ts, second_ts]))
        np.testing.assert_allclose(local[:100], first_ts + 0.008)
        np.testing.assert_allclose(local[100:], second_ts + 0.009)
        np.testing.assert_allclose(corrections[:100], 0.010)
        np.testing.assert_allclose(corrections[100:], 0.011)
        self.assertTrue(np.all(valid))

    def test_duplicate_or_reversed_source_timestamp_is_rejected(self) -> None:
        buffer = APP.TimestampBuffer(32, 8)
        data = self._samples(3)
        with self.assertRaises(APP.InvalidTrial):
            buffer.append(data, np.array([1.0, 1.004, 1.004]), correction=0.0)

        buffer = APP.TimestampBuffer(32, 8)
        buffer.append(data[:2], np.array([1.0, 1.004]), correction=0.0)
        with self.assertRaises(APP.InvalidTrial):
            buffer.append(data[2:], np.array([1.003]), correction=0.0)

    def test_epoch_retains_raw_and_corrected_time_axes(self) -> None:
        count = 800
        base = 100.0
        data = self._samples(count)
        timestamps = base + np.arange(count, dtype=float) / 250.0
        buffer = APP.TimestampBuffer(1024, 8)
        buffer.append(data[:400], timestamps[:400], correction=0.010)
        buffer.append(data[400:], timestamps[400:], correction=0.0105)

        epoch = buffer.epoch(base + 1.0, duration_s=1.0, nominal_fs=250.0)
        self.assertEqual(epoch.data.shape[0], 8)
        self.assertEqual(epoch.data.shape[1], APP.sample_count(1.0, 250.0))
        self.assertTrue(np.all(np.diff(epoch.raw_timestamps) > 0))
        self.assertTrue(np.all(np.diff(epoch.local_timestamps) > 0))
        self.assertGreater(np.unique(epoch.clock_corrections).size, 1)
        self.assertEqual(len(epoch.raw_timestamps), len(epoch.local_timestamps))
        self.assertEqual(len(epoch.raw_timestamps), len(epoch.clock_corrections))

    def test_clock_correction_target_is_applied_with_a_slew_limit(self) -> None:
        state = APP.ClockCorrectionState(
            observed=0.0, target=0.0, applied=0.0, slew_rate_s_per_s=0.001
        )
        state.observe(0.002, max_step_s=0.002)
        raw = np.arange(250, dtype=float) / 250.0
        applied = state.apply(raw)

        self.assertLessEqual(np.max(np.diff(applied)), 0.001 / 250.0 + 1e-12)
        self.assertGreater(applied[-1], 0.0009)
        self.assertLess(applied[-1], 0.002)
        with self.assertRaises(APP.InvalidTrial):
            state.apply(np.array([raw[-1], raw[-1] + 0.004]))

    def test_one_small_gap_is_recorded_as_warning_and_two_gaps_are_rejected(self) -> None:
        data = self._samples(800)
        base = 100.0
        intervals = np.full(799, 1.0 / 250.0)
        intervals[300] = 1.6 / 250.0
        timestamps = base + np.concatenate(([0.0], np.cumsum(intervals)))
        buffer = APP.TimestampBuffer(1024, 8)
        buffer.append(data, timestamps, correction=0.0)
        epoch = buffer.epoch(base + 1.0, 1.0, 250.0, max_warning_gaps=1)
        self.assertEqual(epoch.diagnostics["warning_gap_count"], 1)
        self.assertTrue(epoch.diagnostics["warnings"])

        intervals[350] = 1.6 / 250.0
        timestamps = base + np.concatenate(([0.0], np.cumsum(intervals)))
        buffer = APP.TimestampBuffer(1024, 8)
        buffer.append(data, timestamps, correction=0.0)
        with self.assertRaises(APP.InvalidTrial):
            buffer.epoch(base + 1.0, 1.0, 250.0, max_warning_gaps=1)


class TrialLedgerTests(unittest.TestCase):
    def test_rejected_trial_is_recorded_without_modifying_output(self) -> None:
        ledger = APP.TrialLedger()
        record = APP.TrialRecord(
            trial_id=1,
            block_id=1,
            true_class=None,
            status="rejected",
            reason="insufficient evidence",
            mode="free",
        )
        ledger.commit(record)
        self.assertEqual(ledger.typed_text, "")
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0].status, "rejected")
        report = APP.summarize_free_trials(ledger.records, ledger)
        self.assertEqual(report["attempted"], 1)
        self.assertEqual(report["valid"], 0)
        self.assertEqual(report["input_actions"], 0)

    def test_invalid_and_aborted_trials_never_apply_a_prediction(self) -> None:
        ledger = APP.TrialLedger()
        for trial_id, status in ((1, "invalid"), (2, "aborted")):
            record = APP.TrialRecord(
                trial_id=trial_id,
                block_id=1,
                true_class=None,
                status=status,
                result={"trial_id": trial_id, "prediction": 1, "cca_prediction": 1},
                mode="free",
            )
            ledger.commit(record)
        self.assertEqual(ledger.typed_text, "")

    def test_valid_record_must_already_have_text_application(self) -> None:
        ledger = APP.TrialLedger()
        ledger.typed_text = "MANUAL"
        record = APP.TrialRecord(
            trial_id=1,
            block_id=1,
            true_class=None,
            status="valid",
            result={"trial_id": 1, "prediction": 1, "cca_prediction": 1},
            text_applied=True,
            mode="free",
        )
        ledger.commit(record)
        self.assertEqual(ledger.typed_text, "MANUAL")


class RejectionGateTests(unittest.TestCase):
    def _keyboard_without_display(self, **config_overrides):
        keyboard = object.__new__(APP.PsychoPyKeyboard)
        keyboard.cfg = APP.Config(**config_overrides)
        return keyboard

    def test_rejection_is_opt_in_and_uses_score_and_margin(self) -> None:
        disabled = self._keyboard_without_display(
            rejection_enabled=False, rejection_min_score=0.9, rejection_min_margin=0.2
        )
        self.assertIsNone(disabled._rejection_reason({"scores": np.array([0.01, 0.02])}))

        keyboard = self._keyboard_without_display(
            rejection_enabled=True, rejection_min_score=0.5, rejection_min_margin=0.1
        )
        self.assertIsNotNone(keyboard._rejection_reason({"scores": np.array([0.1, 0.2, 0.25])}))
        self.assertIsNone(keyboard._rejection_reason({"scores": np.array([0.6, 0.2, 0.1])}))

    def test_rejection_handles_missing_or_nonfinite_scores(self) -> None:
        keyboard = self._keyboard_without_display(rejection_enabled=True)
        for scores in ([], np.array([np.nan, 0.2])):
            reason = keyboard._rejection_reason({"scores": scores})
            self.assertIsNotNone(reason)
            self.assertIn("拒识", reason)


class FrameDiagnosticsTests(unittest.TestCase):
    def test_single_display_anomaly_is_warning_and_multiple_are_hard_invalid(self) -> None:
        one = APP.frame_diagnostics(
            [0.00, 0.010, 0.026, 0.036], refresh_hz=100.0, max_warning_anomalies=1
        )
        self.assertTrue(one["warning"])
        self.assertFalse(one["hard_invalid"])
        self.assertTrue(one["valid"])

        many = APP.frame_diagnostics(
            [0.00, 0.010, 0.026, 0.042, 0.052], refresh_hz=100.0, max_warning_anomalies=1
        )
        self.assertFalse(many["warning"])
        self.assertTrue(many["hard_invalid"])
        self.assertFalse(many["valid"])


class PersistenceContractTests(unittest.TestCase):
    """Exercise the recorder once the implementation exposes its public helper.

    The recorder API is intentionally discovered by name so this test module
    remains usable during the staged rollout.  The accepted names are
    ``SessionRecorder`` or ``save_trial_record``; once either is present the
    helper must write JSON metadata and a compressed NPZ containing the raw and
    corrected time axes.
    """

    def test_trial_record_persistence_round_trip(self) -> None:
        recorder_cls = getattr(APP, "SessionRecorder", None)
        save_trial_record = getattr(APP, "save_trial_record", None)
        if recorder_cls is None and save_trial_record is None:
            self.skipTest("persistence helper is not exposed yet")

        with tempfile.TemporaryDirectory() as temp_dir:
            record = APP.TrialRecord(
                trial_id=7,
                block_id=1,
                true_class=3,
                status="valid",
                start=10.0,
                end=12.0,
                result={
                    "trial_id": 7,
                    "prediction": 4,
                    "cca_prediction": 4,
                    "scores": np.array([0.1, 0.2]),
                    "correlations": np.array([[0.3, 0.4]]),
                    "epoch": APP.Epoch(
                        data=np.ones((2, 4)),
                        fs=250.0,
                        requested_start=10.0,
                        requested_end=12.0,
                        raw_timestamps=np.array([10.0, 10.004, 10.008, 10.012]),
                        local_timestamps=np.array([10.01, 10.014, 10.018, 10.022]),
                        clock_corrections=np.full(4, 0.01),
                        diagnostics={"ok": True},
                        source_data=np.ones((4, 2)),
                        uniform_timestamps=np.array([10.0, 10.004, 10.008, 10.012]),
                    ),
                },
                mode="cued",
            )

            if recorder_cls is not None:
                cfg = APP.Config()
                recorder = recorder_cls(cfg, root=temp_dir)
                result = recorder.write_trial(record)
            else:
                result = save_trial_record(temp_dir, record)

            paths = [Path(p) for p in result] if isinstance(result, (tuple, list)) else []
            if not paths:
                paths = list(Path(temp_dir).rglob("*"))
            json_files = [p for p in paths if p.suffix == ".json" and p.stem.startswith("trial_")]
            npz_files = [p for p in paths if p.suffix == ".npz" and p.stem.startswith("trial_")]
            self.assertTrue(json_files, f"no JSON artifact returned in {paths}")
            self.assertTrue(npz_files, f"no NPZ artifact returned in {paths}")
            metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata.get("trial_id"), 7)
            with np.load(npz_files[0]) as arrays:
                keys = ("source_raw_timestamps", "source_local_timestamps", "clock_corrections")
                for key in keys:
                    self.assertIn(key, arrays.files)
                    self.assertEqual(len(arrays[key]), 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
