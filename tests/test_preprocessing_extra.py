"""Extra tests for src/preprocessing.py — validation guards and outlier-handling paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    align_ohlcv,
    apply_outlier_handling,
    compute_log_returns,
    prepare_aligned_returns,
)


def _ohlcv(n: int = 40, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    prices = start_price * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
            "volume": np.random.uniform(1e7, 1e8, size=n),
        }
    )


def test_align_missing_timestamp_column_raises():
    with pytest.raises(ValueError):
        align_ohlcv(_ohlcv(5).drop(columns=["timestamp"]), _ohlcv(5))


def test_align_missing_close_column_raises():
    with pytest.raises(ValueError):
        align_ohlcv(_ohlcv(5).drop(columns=["close"]), _ohlcv(5))


def test_align_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["timestamp", "close"])
    btc, alt = align_ohlcv(empty, _ohlcv(5))
    assert btc.empty


def test_align_unknown_policy_raises():
    with pytest.raises(ValueError):
        align_ohlcv(_ohlcv(5), _ohlcv(5), missing_policy="bogus")


def test_compute_log_returns_missing_price_column_raises():
    with pytest.raises(ValueError):
        compute_log_returns(_ohlcv(5), price_col="does_not_exist")


def test_compute_log_returns_non_positive_price_is_dropped():
    df = _ohlcv(5)
    df.loc[0, "close"] = -1.0
    out = compute_log_returns(df)
    assert "log_return" in out.columns
    assert len(out) <= 4


def test_apply_outlier_unknown_method_raises():
    with pytest.raises(ValueError):
        apply_outlier_handling(pd.Series([0.1, 0.2]), method="bogus")


def test_apply_outlier_winsorize_requires_two_limits():
    with pytest.raises(ValueError):
        apply_outlier_handling(
            pd.Series([0.1, 0.2, 0.3]), method="winsorize", winsorize_limits=None
        )


def test_apply_outlier_robust_is_passthrough():
    s = pd.Series([0.1, 0.2, 0.3])
    pd.testing.assert_series_equal(apply_outlier_handling(s, method="robust"), s)


def test_apply_outlier_none_on_empty_series():
    empty = pd.Series(dtype=float)
    assert apply_outlier_handling(empty, method="winsorize").empty


def test_apply_outlier_winsorize_clips_tails():
    s = pd.Series([0.0] * 98 + [100.0, -100.0])
    out = apply_outlier_handling(s, method="winsorize", winsorize_limits=[0.05, 0.05])
    assert out.max() < 100.0
    assert out.min() > -100.0


def test_prepare_aligned_returns_winsorize_path():
    cfg = {
        "preprocessing": {
            "missing_data_policy": "drop",
            "forward_fill_limit": 2,
            "outlier_handling": "winsorize",
            "winsorize_limits": [0.05, 0.05],
        }
    }
    btc, alt, merged = prepare_aligned_returns(_ohlcv(60, seed=1), _ohlcv(60, seed=2), cfg)
    assert len(btc) == len(alt)
    assert not merged.empty


def test_prepare_aligned_returns_empty_input():
    cfg = {"preprocessing": {"missing_data_policy": "drop"}}
    empty = pd.DataFrame(columns=["timestamp", "close"])
    btc, alt, merged = prepare_aligned_returns(empty, empty, cfg)
    assert btc.empty
    assert alt.empty
    assert merged.empty
