"""Extra tests for src/metrics.py — parallel executor paths, rolling metrics, robust OLS.

Complements tests/test_metrics.py (which covers the core math + look-ahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_all_coins_metrics,
    compute_all_coins_metrics_v2,
    compute_metrics,
    compute_rolling_metrics,
)


def _series(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.02, n))


def test_compute_all_coins_metrics_thread_executor():
    btc = _series(100, 0)
    alt = btc * 2.0 + pd.Series(np.random.default_rng(1).normal(0, 0.002, 100))
    result = compute_all_coins_metrics(
        btc_returns=btc,
        alt_returns_map={"AAA": (btc, alt)},
        windows=[30, 90],
        min_data_points=10,
        max_workers=2,
        executor="thread",
    )
    assert "AAA" in result
    assert set(result["AAA"].keys()) == {30, 90}


def test_compute_all_coins_metrics_non_tuple_fallback():
    btc = _series(50, 3)
    alt = btc * 1.5
    result = compute_all_coins_metrics(
        btc_returns=btc,
        alt_returns_map={"BBB": alt},
        windows=[30],
        min_data_points=5,
        max_workers=1,
    )
    assert "BBB" in result


def test_compute_all_coins_metrics_empty_map():
    result = compute_all_coins_metrics(
        btc_returns=_series(10), alt_returns_map={}, windows=[30]
    )
    assert result == {}


def test_compute_all_coins_metrics_unknown_executor_falls_back():
    btc = _series(60, 4)
    result = compute_all_coins_metrics(
        btc_returns=btc,
        alt_returns_map={"CCC": (btc, btc * 2.0)},
        windows=[30],
        min_data_points=5,
        max_workers=1,
        executor="bogus",
    )
    assert "CCC" in result


def test_compute_all_coins_metrics_v2_delegates():
    btc = _series(60, 5)
    result = compute_all_coins_metrics_v2(
        {"DDD": (btc, btc * 2.0)}, windows=[30], min_data_points=5, max_workers=1
    )
    assert "DDD" in result


def test_compute_rolling_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_rolling_metrics(_series(20), _series(10), window=5)


def test_compute_rolling_metrics_len_lt_window_returns_empty():
    df = compute_rolling_metrics(_series(5), _series(5), window=10)
    assert df.empty


def test_compute_rolling_metrics_datetime_index():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    btc = pd.Series(np.random.default_rng(0).normal(0, 0.02, n), index=idx)
    alt = btc * 2.0
    df = compute_rolling_metrics(btc, alt, window=30)
    assert not df.empty
    assert {"r", "beta", "p_value"}.issubset(df.columns)


def test_compute_metrics_single_point_returns_nan():
    m = compute_metrics(pd.Series([0.01]), pd.Series([0.02]))
    assert np.isnan(m["r"])
    assert np.isnan(m["beta"])


def test_compute_metrics_robust_uses_rlm_huber():
    btc = _series(100, 7)
    alt = btc * 2.0 + pd.Series(np.random.default_rng(8).normal(0, 0.001, 100))
    m = compute_metrics(btc, alt, robust=True)
    assert m["regression_method"] == "rlm_huber"
    assert not np.isnan(m["beta"])
