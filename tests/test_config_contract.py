"""Config contract tests — thresholds, windows, primary_window, ranges."""

from __future__ import annotations

import pytest

from src.config_schema import validate_config, validate_ranges


def base_config():
    return {
        "thresholds": {
            "corr_threshold_high": 0.7,
            "corr_threshold_low": 0.2,
            "beta_min": 3.0,
            "beta_max": 5.0,
            "alpha": 0.05,
        },
        "windows": [30, 90, 180],
        "primary_window": 180,
    }


class TestThresholdContract:
    def test_valid_config_passes(self):
        cfg = base_config()
        assert validate_config(cfg) == cfg

    def test_missing_threshold_key_raises(self):
        cfg = base_config()
        del cfg["thresholds"]["alpha"]
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config(cfg)

    def test_missing_thresholds_top_level_raises(self):
        cfg = base_config()
        del cfg["thresholds"]
        with pytest.raises(ValueError, match="missing required key"):
            validate_config(cfg)

    def test_missing_windows_raises(self):
        cfg = base_config()
        del cfg["windows"]
        with pytest.raises(ValueError, match="missing required key"):
            validate_config(cfg)

    def test_windows_must_be_ascending(self):
        cfg = base_config()
        cfg["windows"] = [90, 30, 180]
        with pytest.raises(ValueError, match="strictly ascending"):
            validate_config(cfg)

    def test_windows_duplicates_raises(self):
        cfg = base_config()
        cfg["windows"] = [30, 30, 90]
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_primary_window_must_be_in_windows(self):
        cfg = base_config()
        cfg["primary_window"] = 999
        with pytest.raises(ValueError, match="primary_window"):
            validate_config(cfg)

    def test_no_silent_defaults_for_thresholds(self):
        # classifier should raise if thresholds missing, not use .get defaults
        from src.classifier import build_classification_table
        cfg = {"windows": [30, 90], "primary_window": 30, "thresholds": {"corr_threshold_high": 0.7}}  # incomplete
        with pytest.raises(ValueError, match="missing required keys"):
            build_classification_table({}, cfg)


class TestValidateRanges:
    def test_beta_min_gt_max_raises(self):
        cfg = base_config()
        cfg["thresholds"]["beta_min"] = 5.0
        cfg["thresholds"]["beta_max"] = 3.0
        with pytest.raises(ValueError, match="beta_min"):
            validate_ranges(cfg)

    def test_alpha_out_of_range(self):
        cfg = base_config()
        cfg["thresholds"]["alpha"] = 1.5
        with pytest.raises(ValueError, match="alpha"):
            validate_ranges(cfg)
        cfg["thresholds"]["alpha"] = 0
        with pytest.raises(ValueError, match="alpha"):
            validate_ranges(cfg)

    def test_corr_low_ge_high_raises(self):
        cfg = base_config()
        cfg["thresholds"]["corr_threshold_low"] = 0.8
        cfg["thresholds"]["corr_threshold_high"] = 0.7
        with pytest.raises(ValueError, match="corr_threshold_low"):
            validate_ranges(cfg)

    def test_corr_out_of_0_1_raises(self):
        cfg = base_config()
        cfg["thresholds"]["corr_threshold_low"] = -0.1
        with pytest.raises(ValueError):
            validate_ranges(cfg)

    def test_category_b_flag_optional(self):
        cfg = base_config()
        cfg["thresholds"]["category_b_require_nonsignificant"] = True
        validate_config(cfg)  # should not raise
        cfg["thresholds"]["category_b_require_nonsignificant"] = False
        validate_config(cfg)
