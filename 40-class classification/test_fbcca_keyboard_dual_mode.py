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
import time
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
        # All EEG arrays use channels x samples (C x T).  Timestamp arrays
        # remain one-dimensional and therefore index the second axis.
        return np.vstack([np.sin(2 * np.pi * (8 + i) * t) for i in range(8)])

    def test_snapshot_preserves_raw_timestamps_and_per_sample_correction(self) -> None:
        base = 100.0
        first = self._samples(100)
        first_ts = base + np.arange(first.shape[1], dtype=float) / 250.0
        second = self._samples(40)
        second_ts = base + (100 + np.arange(second.shape[1], dtype=float)) / 250.0
        buffer = APP.TimestampBuffer(256, 8)

        buffer.append(first, first_ts, correction=0.010, device_lag_s=0.002)
        buffer.append(second, second_ts, correction=0.011, device_lag_s=0.002)
        samples, raw, local, corrections, valid = buffer.snapshot()

        np.testing.assert_allclose(samples, np.hstack([first, second]))
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
        buffer.append(data[:, :2], np.array([1.0, 1.004]), correction=0.0)
        with self.assertRaises(APP.InvalidTrial):
            buffer.append(data[:, 2:3], np.array([1.003]), correction=0.0)

    def test_epoch_retains_raw_and_corrected_time_axes(self) -> None:
        count = 800
        base = 100.0
        data = self._samples(count)
        timestamps = base + np.arange(count, dtype=float) / 250.0
        buffer = APP.TimestampBuffer(1024, 8)
        buffer.append(data[:, :400], timestamps[:400], correction=0.010)
        buffer.append(data[:, 400:], timestamps[400:], correction=0.0105)

        epoch = buffer.epoch(base + 1.0, duration_s=1.0, nominal_fs=250.0)
        self.assertEqual(epoch.data.shape, (8, APP.sample_count(1.0, 250.0)))
        self.assertTrue(np.all(np.diff(epoch.raw_timestamps) > 0))
        self.assertTrue(np.all(np.diff(epoch.local_timestamps) > 0))
        self.assertGreater(np.unique(epoch.clock_corrections).size, 1)
        self.assertEqual(len(epoch.raw_timestamps), len(epoch.local_timestamps))
        self.assertEqual(len(epoch.raw_timestamps), len(epoch.clock_corrections))

    def test_append_rejects_time_by_channel_layout(self) -> None:
        buffer = APP.TimestampBuffer(32, 8)
        # A T x C matrix has the same two dimensions as a valid C x T input
        # only when T happens to equal C.  Use a distinct sample count so the
        # contract is unambiguous.
        txc = np.zeros((3, 8), dtype=float)
        with self.assertRaises((ValueError, APP.InvalidTrial)):
            buffer.append(txc, np.arange(3, dtype=float), correction=0.0)

    def test_ring_buffer_wrap_preserves_channel_sample_order(self) -> None:
        buffer = APP.TimestampBuffer(5, 2)
        first = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=float)
        second = np.array([[14, 15, 16], [24, 25, 26]], dtype=float)
        buffer.append(first, np.arange(4, dtype=float), correction=0.0)
        buffer.append(second, np.arange(4, 7, dtype=float), correction=0.0)
        samples, raw, local, corrections, valid = buffer.snapshot()
        np.testing.assert_allclose(samples, np.array([[12, 13, 14, 15, 16],
                                                       [22, 23, 24, 25, 26]], dtype=float))
        np.testing.assert_allclose(raw, np.arange(2, 7, dtype=float))
        np.testing.assert_allclose(local, raw)
        self.assertEqual(samples.shape, (2, 5))
        self.assertEqual(valid.shape, (5,))
        self.assertTrue(np.all(valid))

    def test_chunk_larger_than_capacity_keeps_latest_samples(self) -> None:
        buffer = APP.TimestampBuffer(4, 2)
        data = np.arange(14, dtype=float).reshape(2, 7)
        ts = np.arange(7, dtype=float)
        buffer.append(data, ts, correction=0.0)
        samples, raw, local, corrections, valid = buffer.snapshot()
        np.testing.assert_allclose(samples, data[:, -4:])
        np.testing.assert_allclose(raw, ts[-4:])
        self.assertEqual(samples.shape, (2, 4))
        self.assertEqual(local.shape, (4,))
        self.assertEqual(corrections.shape, (4,))
        self.assertEqual(valid.shape, (4,))

    def test_empty_append_is_a_noop_with_cx0_input(self) -> None:
        buffer = APP.TimestampBuffer(8, 2)
        empty = np.empty((2, 0), dtype=float)
        buffer.append(empty, np.empty(0, dtype=float), correction=0.0)
        samples, raw, local, corrections, valid = buffer.snapshot()
        self.assertEqual(samples.shape, (2, 0))
        for axis in (raw, local, corrections, valid):
            self.assertEqual(axis.shape, (0,))

    def test_validity_is_per_sample_when_one_channel_is_nan(self) -> None:
        buffer = APP.TimestampBuffer(8, 2)
        data = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
        buffer.append(data, np.arange(3, dtype=float), correction=0.0)
        samples, raw, local, corrections, valid = buffer.snapshot()
        np.testing.assert_allclose(samples[:, [0, 2]], data[:, [0, 2]])
        self.assertEqual(valid.tolist(), [True, False, True])

    def test_epoch_interpolates_each_channel_along_sample_axis(self) -> None:
        # Deliberately jitter the source timestamps while retaining the same
        # channel-wise values.  The epoch result must be C x T and use the
        # timestamp grid for each channel independently.
        nominal_fs = 250.0
        raw = 100.0 + np.arange(650, dtype=float) / nominal_fs
        raw[5:] += 0.002
        data = np.vstack((2.0 * raw + 1.0, -3.0 * raw + 7.0))
        buffer = APP.TimestampBuffer(1024, 2)
        buffer.append(data, raw, correction=0.0)
        start, duration = 101.0, 1.0
        epoch = buffer.epoch(start, duration, nominal_fs)
        expected_grid = start + np.arange(APP.sample_count(duration, nominal_fs)) / nominal_fs
        expected = np.vstack([np.interp(expected_grid, raw, row) for row in data])
        self.assertEqual(epoch.data.shape, (2, len(expected_grid)))
        np.testing.assert_allclose(epoch.uniform_timestamps, expected_grid)
        np.testing.assert_allclose(epoch.data, expected, rtol=0, atol=1e-10)

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


class EpochDimensionTests(unittest.TestCase):
    def _epoch(self, channels: int = 3, source_samples: int = 5,
               uniform_samples: int = 4) -> object:
        source = np.arange(channels * source_samples, dtype=float).reshape(channels, source_samples)
        data = np.arange(channels * uniform_samples, dtype=float).reshape(channels, uniform_samples)
        raw = np.arange(source_samples, dtype=float)
        return APP.Epoch(
            data=data,
            fs=250.0,
            requested_start=0.0,
            requested_end=uniform_samples / 250.0,
            raw_timestamps=raw,
            local_timestamps=raw + 0.01,
            clock_corrections=np.full(source_samples, 0.01),
            diagnostics={},
            source_data=source,
            uniform_timestamps=np.arange(uniform_samples, dtype=float) / 250.0,
        )

    def test_epoch_accepts_cx_t_and_validates_time_axes(self) -> None:
        epoch = self._epoch()
        self.assertEqual(epoch.data.shape, (3, 4))
        self.assertEqual(epoch.source_data.shape, (3, 5))
        self.assertEqual(epoch.raw_timestamps.shape, (5,))
        self.assertEqual(epoch.local_timestamps.shape, (5,))
        self.assertEqual(epoch.clock_corrections.shape, (5,))
        self.assertEqual(epoch.uniform_timestamps.shape, (4,))
        self.assertIsNone(epoch.validate_dimensions())

    def test_epoch_rejects_wrong_channel_or_timestamp_dimensions(self) -> None:
        valid = self._epoch()
        cases = [
            {"data": np.zeros((2, 4))},
            {"source_data": np.zeros((4, 5))},
            {"raw_timestamps": np.zeros((2, 5))},
            {"raw_timestamps": np.zeros(4)},
            {"local_timestamps": np.zeros(4)},
            {"clock_corrections": np.zeros(4)},
            {"uniform_timestamps": np.zeros(5)},
        ]
        for overrides in cases:
            values = {
                "data": valid.data.copy(), "fs": valid.fs,
                "requested_start": valid.requested_start,
                "requested_end": valid.requested_end,
                "raw_timestamps": valid.raw_timestamps.copy(),
                "local_timestamps": valid.local_timestamps.copy(),
                "clock_corrections": valid.clock_corrections.copy(),
                "diagnostics": {}, "source_data": valid.source_data.copy(),
                "uniform_timestamps": valid.uniform_timestamps.copy(),
            }
            values.update(overrides)
            with self.subTest(field=next(iter(overrides))):
                with self.assertRaises((ValueError, APP.InvalidTrial, TypeError)):
                    APP.Epoch(**values)


class ContinuousLSL入口Tests(unittest.TestCase):
    def test_run_selects_channels_and_transposes_once_at_ingress(self) -> None:
        cfg = APP.Config(
            channel_indices=(2, 0),
            max_receive_age_s=2.0,
            clock_refresh_s=3600.0,
        )
        collector = APP.ContinuousLSL(cfg)
        collector.fs = 250.0
        collector.buffer = APP.TimestampBuffer(32, len(cfg.channel_indices))
        base = time.monotonic()
        raw_chunk = np.array([
            [100.0, 101.0, 102.0, 103.0],
            [110.0, 111.0, 112.0, 113.0],
            [120.0, 121.0, 122.0, 123.0],
            [130.0, 131.0, 132.0, 133.0],
        ])
        timestamps = base + np.arange(raw_chunk.shape[0], dtype=float) / collector.fs

        class FakeInlet:
            def __init__(self) -> None:
                self.calls = 0

            def pull_chunk(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return raw_chunk.tolist(), timestamps.tolist()
                collector.stop_event.set()
                return [], []

            def was_clock_reset(self) -> bool:
                return False

        collector.inlet = FakeInlet()
        collector.clock = lambda: base + 0.1
        collector._run(0.0)
        self.assertIsNone(collector.error)
        samples, raw, local, corrections, valid = collector.buffer.snapshot()
        np.testing.assert_allclose(samples, raw_chunk[:, cfg.channel_indices].T)
        np.testing.assert_allclose(raw, timestamps)
        np.testing.assert_allclose(local, timestamps)
        np.testing.assert_allclose(corrections, 0.0)
        self.assertTrue(np.all(valid))
        self.assertEqual(samples.shape, (len(cfg.channel_indices), len(timestamps)))


class AlgorithmLayoutTests(unittest.TestCase):
    def _signal(self, samples: int = 500) -> np.ndarray:
        t = np.arange(samples, dtype=float) / 250.0
        return np.vstack([np.sin(2 * np.pi * (10 + i) * t) for i in range(8)])

    def test_preprocess_and_classify_keep_channel_major_layout(self) -> None:
        data = self._signal()
        window, bank = APP.preprocess_lsl_window(
            data, 250.0, window_s=2.0, target_fs=250.0, notch_hz=50.0)
        self.assertEqual(window.shape, (8, 500))
        self.assertEqual(bank.shape, (len(APP.M3_BANDS), 8, 500))

        result = APP.classify_eeg_window(
            data, 250.0, window_s=2.0, target_fs=250.0, notch_hz=50.0)
        self.assertEqual(result["window_shape"], (8, 500))
        self.assertEqual(result["bank_shape"], (len(APP.M3_BANDS), 8, 500))
        self.assertEqual(result["reference_shape"][-1], 500)

    def test_time_major_input_is_rejected_in_algorithm_entrypoints(self) -> None:
        data = self._signal()
        with self.assertRaises(ValueError):
            APP.preprocess_lsl_window(data.T, 250.0, window_s=2.0, target_fs=250.0)
        with self.assertRaises(ValueError):
            APP.classify_eeg_window(data.T, 250.0, window_s=2.0, target_fs=250.0)


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

    def test_both_thresholds_and_both_evidence_checks_are_required(self) -> None:
        keyboard = self._keyboard_without_display(
            rejection_enabled=True, rejection_min_score=0.5, rejection_min_margin=0.1
        )
        accepted = keyboard._rejection_diagnostics({"scores": np.array([0.6, 0.2, 0.1])})
        self.assertTrue(accepted["configured"])
        self.assertTrue(accepted["accepted"])
        self.assertAlmostEqual(accepted["max_score"], 0.6)
        self.assertAlmostEqual(accepted["score_margin"], 0.4)
        self.assertIsNone(accepted["reason"])
        self.assertIsNone(keyboard._rejection_reason({"scores": np.array([0.6, 0.2, 0.1])}))

        for scores, code in (
                (np.array([0.49, 0.2, 0.1]), "score_below_threshold"),
                (np.array([0.55, 0.5, 0.1]), "margin_below_threshold"),
                (np.array([0.49, 0.45, 0.1]), "score_below_threshold+margin_below_threshold")):
            with self.subTest(scores=scores):
                diagnostics = keyboard._rejection_diagnostics({"scores": scores})
                self.assertFalse(diagnostics["accepted"])
                self.assertEqual(diagnostics["reason_code"], code)
                self.assertIsNotNone(diagnostics["reason"])

    def test_disabled_or_partially_configured_gate_rejects_without_bypassing(self) -> None:
        for overrides in (
                {},
                {"rejection_enabled": False, "rejection_min_score": 0.5,
                 "rejection_min_margin": 0.1},
                {"rejection_enabled": True, "rejection_min_score": 0.5},
                {"rejection_enabled": True, "rejection_min_margin": 0.1}):
            with self.subTest(overrides=overrides):
                keyboard = self._keyboard_without_display(**overrides)
                diagnostics = keyboard._rejection_diagnostics(
                    {"scores": np.array([0.9, 0.1])})
                self.assertFalse(diagnostics["configured"])
                self.assertFalse(diagnostics["accepted"])
                self.assertIn("thresholds_unconfigured", diagnostics["reason_codes"])
                self.assertIsNotNone(diagnostics["reason"])

    def test_rejection_handles_missing_or_nonfinite_scores(self) -> None:
        keyboard = self._keyboard_without_display(
            rejection_enabled=True, rejection_min_score=0.5, rejection_min_margin=0.1)
        for scores, code in (
                ([], "scores_missing_or_nonfinite"),
                (np.array([np.nan, 0.2]), "scores_missing_or_nonfinite"),
                (np.array([0.6]), "insufficient_scores")):
            diagnostics = keyboard._rejection_diagnostics({"scores": scores})
            self.assertFalse(diagnostics["accepted"])
            self.assertIn(code, diagnostics["reason_codes"])
            self.assertIsNone(diagnostics["score_margin"])
            self.assertIsNotNone(diagnostics["reason"])

    def test_cued_baseline_does_not_apply_rejection_gate(self) -> None:
        keyboard = self._keyboard_without_display(
            rejection_enabled=False, rejection_min_score=None, rejection_min_margin=None)
        self.assertIsNone(
            keyboard._rejection_diagnostics_for_mode(
                "cued", {"scores": np.array([0.01, 0.02])}))
        free = keyboard._rejection_diagnostics_for_mode(
            "free", {"scores": np.array([0.01, 0.02])})
        self.assertIsNotNone(free)

    def test_rejected_trials_do_not_advance_fault_streak(self) -> None:
        keyboard = self._keyboard_without_display(max_consecutive_invalid=2)
        rejected = APP.TrialRecord(1, 1, None, status="rejected", mode="free")
        invalid = APP.TrialRecord(2, 1, None, status="invalid", mode="free")
        self.assertEqual(
            keyboard._advance_free_invalid_streak(
                0, rejected, pause_requested=False, exit_requested=False), 0)
        self.assertEqual(
            keyboard._advance_free_invalid_streak(
                1, rejected, pause_requested=False, exit_requested=False), 0)
        self.assertEqual(
            keyboard._advance_free_invalid_streak(
                0, invalid, pause_requested=False, exit_requested=False), 1)
        self.assertEqual(
            keyboard._advance_free_invalid_streak(
                1, invalid, pause_requested=False, exit_requested=False), 2)

    def test_missing_threshold_notice_requires_calibration_and_disclaims_probability(self) -> None:
        notice = APP.PsychoPyKeyboard._rejection_calibration_notice()
        self.assertIn("independent data", notice)
        self.assertIn("not probabilities", notice)
        self.assertIn("automatic idle detection", notice)


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
                    "rejection_diagnostics": {
                        "configured": True, "accepted": False,
                        "max_score": 0.2, "score_margin": 0.1,
                        "min_score": 0.5, "min_margin": 0.2,
                        "reason_code": "score_below_threshold",
                        "reason": "拒识：最高分低于阈值",
                    },
                    "epoch": APP.Epoch(
                        data=np.ones((2, 4)),
                        fs=250.0,
                        requested_start=10.0,
                        requested_end=12.0,
                        raw_timestamps=np.array([10.0, 10.004, 10.008, 10.012]),
                        local_timestamps=np.array([10.01, 10.014, 10.018, 10.022]),
                        clock_corrections=np.full(4, 0.01),
                        diagnostics={"ok": True},
                        source_data=np.ones((2, 4)),
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
            self.assertEqual(metadata["rejection_diagnostics"]["max_score"], 0.2)
            self.assertEqual(metadata["rejection_diagnostics"]["score_margin"], 0.1)
            self.assertEqual(metadata["rejection_diagnostics"]["min_score"], 0.5)
            self.assertEqual(metadata["rejection_diagnostics"]["min_margin"], 0.2)
            self.assertEqual(metadata["rejection_diagnostics"]["reason_code"],
                             "score_below_threshold")
            with np.load(npz_files[0]) as arrays:
                keys = ("source_raw_timestamps", "source_local_timestamps", "clock_corrections")
                for key in keys:
                    self.assertIn(key, arrays.files)
                    self.assertEqual(len(arrays[key]), 4)
                self.assertEqual(arrays["source_data"].shape, (2, 4))
                self.assertEqual(arrays["uniform_data"].shape, (2, 4))
                self.assertEqual(arrays["uniform_timestamps"].shape, (4,))
                np.testing.assert_allclose(arrays["source_data"], np.ones((2, 4)))
                np.testing.assert_allclose(arrays["uniform_data"], np.ones((2, 4)))

    def test_failed_epoch_persistence_keeps_cx_t_and_empty_uniform_window(self) -> None:
        recorder_cls = getattr(APP, "SessionRecorder", None)
        save_trial_record = getattr(APP, "save_trial_record", None)
        if recorder_cls is None and save_trial_record is None:
            self.skipTest("persistence helper is not exposed yet")

        with tempfile.TemporaryDirectory() as temp_dir:
            source = np.arange(2 * 4, dtype=float).reshape(2, 4)
            epoch = APP.Epoch(
                data=np.empty((2, 0), dtype=float), fs=250.0,
                requested_start=10.0, requested_end=12.0,
                raw_timestamps=np.arange(4, dtype=float),
                local_timestamps=np.arange(4, dtype=float),
                clock_corrections=np.zeros(4), diagnostics={"hard_reason": "test"},
                source_data=source, uniform_timestamps=np.empty(0),
            )
            record = APP.TrialRecord(
                trial_id=8, block_id=1, true_class=None, status="invalid",
                reason="test failure", result={"epoch": epoch}, mode="free",
            )
            if recorder_cls is not None:
                result = recorder_cls(APP.Config(), root=temp_dir).write_trial(record)
            else:
                result = save_trial_record(temp_dir, record)
            paths = [Path(p) for p in result] if isinstance(result, (tuple, list)) else []
            if not paths:
                paths = list(Path(temp_dir).rglob("*"))
            npz_files = [p for p in paths if p.suffix == ".npz" and p.stem.startswith("trial_")]
            self.assertTrue(npz_files)
            with np.load(npz_files[0]) as arrays:
                self.assertEqual(arrays["source_data"].shape, (2, 4))
                self.assertEqual(arrays["uniform_data"].shape, (2, 0))
                self.assertEqual(arrays["uniform_timestamps"].shape, (0,))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
