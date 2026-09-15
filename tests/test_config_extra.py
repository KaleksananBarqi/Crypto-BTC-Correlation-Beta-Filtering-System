"""Extra tests for config_schema validation guards and classifier fallback paths."""

from __future__ import annotations

import pytest

from src.classifier import build_classification_table
from src.config_schema import validate_config

_BASE_THRESHOLDS = {
    "corr_threshold_high": 0.7,
    "corr_threshold_low": 0.2,
    "beta_min": 3.0,
    "beta_max": 5.0,
    "alpha": 0.05,
}


def test_validate_config_non_dict_raises():
    with pytest.raises(ValueError):
        validate_config(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_config_thresholds_not_mapping_raises():
    with pytest.raises(ValueError):
        validate_config({"thresholds": [1, 2], "windows": [30]})


def test_validate_config_windows_must_be_positive_ints():
    cfg = {"thresholds": dict(_BASE_THRESHOLDS), "windows": [0]}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_build_classification_table_missing_thresholds_raises():
    with pytest.raises(ValueError):
        build_classification_table({}, {"windows": [30]})


def test_build_classification_table_missing_threshold_keys_raises():
    with pytest.raises(ValueError):
        build_classification_table({}, {"thresholds": {"alpha": 0.05}, "windows": [30]})


def test_build_classification_table_empty_metrics(sample_config):
    df = build_classification_table({}, sample_config)
    assert df.empty


def test_build_classification_table_primary_window_not_in_windows_warns(sample_config):
    cfg = dict(sample_config)
    cfg["primary_window"] = 999
    df = build_classification_table({}, cfg)
    assert df.empty
