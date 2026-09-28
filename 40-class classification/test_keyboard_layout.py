"""Geometry, preview interaction and record round-trip; no display or EEG required."""
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from fbcca_keyboard_dual_mode import (
    AbortSession, Config, Epoch, KeyboardLayoutPreview, SessionRecorder, TrialRecord,
    calculate_keyboard_layout, keyboard_layout_info,
)


class Stim:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def draw(self):
        pass


def preview(cfg=None):
    win = SimpleNamespace(size=(1920, 1080), keyboard_display_size=(1920, 1080), fullscr=True)
    visual = SimpleNamespace(TextStim=Stim, Rect=Stim, ElementArrayStim=Stim)
    return KeyboardLayoutPreview(cfg or Config(), win, visual)


class KeyboardLayoutTests(unittest.TestCase):
    def test_large_screen_does_not_inflate_gaps(self):
        cfg = Config()
        small = calculate_keyboard_layout(cfg, (1920, 1080))
        large = calculate_keyboard_layout(cfg, (3840, 2160))
        np.testing.assert_allclose(small[0], large[0])
        self.assertEqual(large[1:], (140.0, (30.0, 30.0), 1.0))

    def test_small_screen_fits_without_changing_spacing_ratio(self):
        cfg = Config()
        for width, height in ((800, 600), (1366, 768), (1080, 1920), (2560, 1080)):
            with self.subTest(size=(width, height)):
                positions, key, (gap_x, gap_y), scale = calculate_keyboard_layout(cfg, (width, height))
                self.assertLessEqual(np.max(np.abs(positions[:, 0])) + key / 2, width / 2 - 24 + 1e-8)
                self.assertLessEqual(np.max(positions[:, 1]) + key / 2, height / 2 - 166 + 1e-8)
                self.assertGreaterEqual(np.min(positions[:, 1]) - key / 2, -height / 2 + 70 - 1e-8)
                self.assertAlmostEqual(gap_x / key, cfg.key_gap_px / cfg.key_size_px)
                self.assertEqual(gap_x, gap_y)
                self.assertLessEqual(scale, 1)
        with self.assertRaises(RuntimeError):
            calculate_keyboard_layout(replace(cfg, allow_layout_scaling=False), (800, 600))

    def test_angular_size_is_stable_across_displays_and_distances(self):
        base = Config(layout_units="degrees")
        for diagonal, distance, resolution in ((15.6, 50, (1920, 1080)), (27, 70, (3840, 2160)),
                                               (32, 90, (2560, 1440))):
            cfg = replace(base, screen_diagonal_inches=diagonal, viewing_distance_cm=distance)
            info = keyboard_layout_info(cfg, resolution)
            self.assertFalse(info["fit_reduced"])
            physical = info["physical_estimate"]
            self.assertAlmostEqual(physical["key_centered_visual_angle_deg"], cfg.key_size_deg)
            self.assertAlmostEqual(physical["gap_centered_visual_angle_deg"], cfg.key_gap_deg)
            self.assertAlmostEqual(physical["key_size_cm"], 2 * distance * math.tan(math.radians(1)))

    def test_window_uses_monitor_dimensions_and_hidpi_doubles_pixels(self):
        cfg = Config(layout_units="degrees", screen_diagonal_inches=27, key_size_deg=1)
        normal = keyboard_layout_info(cfg, (1920, 1080))
        retina = keyboard_layout_info(cfg, (3840, 2160))
        self.assertAlmostEqual(retina["key_size_pix_units"], normal["key_size_pix_units"] * 2)
        self.assertAlmostEqual(retina["physical_estimate"]["key_size_cm"], normal["physical_estimate"]["key_size_cm"])
        window = keyboard_layout_info(replace(cfg, full_screen=False), (1000, 700), (1920, 1080))
        self.assertAlmostEqual(window["key_size_pix_units"], normal["key_size_pix_units"])
        with self.assertRaises(ValueError):
            keyboard_layout_info(replace(cfg, full_screen=False), (1000, 700))

    def test_unknown_dimensions_and_invalid_values(self):
        self.assertIsNone(keyboard_layout_info(Config(), (1920, 1080))["physical_estimate"])
        with self.assertRaises(ValueError):
            calculate_keyboard_layout(Config(layout_units="degrees"), (1920, 1080))
        for settings in ({"viewing_distance_cm": 0}, {"viewing_distance_cm": math.nan},
                         {"screen_diagonal_inches": -1}, {"screen_diagonal_inches": math.inf},
                         {"key_gap_px": -1}, {"key_size_deg": 90}, {"key_gap_deg": math.nan}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                calculate_keyboard_layout(Config(**settings), (1920, 1080))

    def test_corner_angles_and_fit_are_actual_not_requested(self):
        cfg = Config(screen_diagonal_inches=15.6, key_size_px=400)
        info = keyboard_layout_info(cfg, (1366, 768))
        self.assertTrue(info["fit_reduced"])
        p = info["physical_estimate"]
        self.assertAlmostEqual(p["key_size_cm"], info["key_size_pix_units"] * p["cm_per_pix_unit"])
        first = p["corners"][0]
        pos = info["key_positions_pix_units"][0]
        self.assertAlmostEqual(first["center_eccentricity_deg"],
                               math.degrees(math.atan(math.hypot(*pos) * p["cm_per_pix_unit"] / 70)))
        self.assertGreater(first["outer_edge_eccentricity_deg"], first["center_eccentricity_deg"])
        self.assertEqual([c["class_id"] for c in p["corners"]], [1, 10, 31, 40])

    def test_zero_gap_and_too_small_window(self):
        _, key, gaps, _ = calculate_keyboard_layout(Config(key_gap_px=0), (1920, 1080))
        self.assertEqual(gaps, (0, 0))
        self.assertEqual(key, 140)
        with self.assertRaises(RuntimeError):
            calculate_keyboard_layout(Config(), (100, 100))

    def test_preview_marks_unchecked_corners_and_invalidates_old_ratings(self):
        p = preview()
        self.assertTrue(p.handle_keys(["return"]))
        self.assertEqual(p.result()["static_preview"]["corners_checked"], 0)
        self.assertTrue(all(c["rating"] is None for c in p.result()["static_preview"]["corner_ratings"]))
        for rating in ("1", "2", "3", "4"):
            p.handle_keys([rating])
        self.assertTrue(p.handle_keys(["return"]))
        self.assertEqual(list(p.ratings.values()), list(p.rating_names))
        p.handle_keys(["right", "return"])
        self.assertFalse(p.result()["static_preview"]["confirmed"])
        self.assertEqual(p.ratings, {})
        p.handle_keys(["u"])
        self.assertFalse(p.info)  # no assumed screen inches
        self.assertFalse(p.handle_keys(["return", "1"]))
        p.field_index = 3
        p.handle_keys(["right"])
        self.assertTrue(p.info)
        p.draw()
        with self.assertRaises(AbortSession):
            p.handle_keys(["escape"])

    def test_session_npz_contains_geometry_and_corner_reports(self):
        cfg = Config(screen_diagonal_inches=24)
        p = preview(cfg)
        for rating in ("1", "2", "3", "4"):
            p.handle_keys([rating])
        p.handle_keys(["return"])
        info = p.result()
        t = np.arange(10) / 250
        data = np.zeros((8, 10))
        epoch = Epoch(data, 250, t, t, np.zeros(10), {}, data, t)
        record = TrialRecord(1, 1, 1, status="invalid", result={"epoch": epoch})
        with tempfile.TemporaryDirectory() as folder:
            recorder = SessionRecorder(cfg, root=folder, context={"layout_preview": info})
            recorder.write_trial(record)
            with recorder.manifest_path.open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["context"]["layout_preview"], info)
            path = recorder.finalize(summary={}, events=[], acquisition_metadata={}, acquisition_review={},
                clock_updates=[], display_info=info, typed_text="", stop_reason="test")
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
            self.assertEqual(metadata["display_info"], info)
            self.assertEqual(metadata["config"]["key_gap_px"], cfg.key_gap_px)
            self.assertEqual(metadata["display_info"]["static_preview"]["corner_ratings"][2]["rating"], "head_turn")
            self.assertEqual(list(Path(folder).iterdir()), [Path(path)])


if __name__ == "__main__":
    unittest.main()
