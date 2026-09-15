"""Shared fixtures for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def btc_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.02, 100))


@pytest.fixture
def alt_returns_correlated(btc_returns):
    # alt = 2*btc + small noise
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(0, 0.005, len(btc_returns)))
    return btc_returns * 2.0 + noise


@pytest.fixture
def sample_config():
    return {
        "thresholds": {
            "corr_threshold_high": 0.7,
            "corr_threshold_low": 0.2,
            "beta_min": 3.0,
            "beta_max": 5.0,
            "alpha": 0.05,
            "category_b_require_nonsignificant": False,
        },
        "windows": [30, 90, 180],
        "primary_window": 180,
        "universe": {"top_n": 10, "exclude_symbols": ["USDT"], "liquidity": {"min_24h_volume_usd": 1_000_000}, "min_data_points": 30},
        "data": {"cache_dir": "data", "vs_currency": "USDT", "interval": "1d", "coingecko": {"fallback_on_binance_failure": True}},
        "preprocessing": {"missing_data_policy": "drop", "forward_fill_limit": 2, "outlier_handling": "none", "winsorize_limits": [0.01, 0.01]},
        "performance": {"max_workers": 2, "executor": "thread"},
        "output": {"csv_path": "output/classified_coins.csv", "scatter_path": "output/scatter_beta_corr.png"},
        "logging": {"level": "INFO", "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    }


@pytest.fixture
def ohlcv_df():
    n = 10
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)))
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(1e7, 1e8, size=n),
    })
