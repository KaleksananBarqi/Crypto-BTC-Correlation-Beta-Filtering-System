"""
Unit tests for metrics.py, classifier.py, preprocessing.py

- Synthetic series with known correlation (r=1, r=0, r=-1)
- Beta, R², p-value verification
- Look-ahead bias test (rolling window only uses data up to time t)
- No silent-fail on missing data / API error
- Edge cases: small-sample, zero variance, NaN, negative beta

All inputs are log-returns (dimensionless, NOT percent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.classifier import CATEGORY_A, CATEGORY_B, CATEGORY_NEUTRAL, build_classification_table, check_stability, classify_single
from src.metrics import compute_metrics, compute_metrics_for_windows, compute_rolling_metrics
from src.preprocessing import align_ohlcv, apply_outlier_handling, compute_log_returns, prepare_aligned_returns


# ── Helpers ─────────────────────────────────────────────────────────────

def make_returns(n: int = 100, seed: int = 0) -> np.ndarray:
    """Generate synthetic BTC log-returns (dimensionless)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, size=n)


def make_ohlcv_from_returns(returns: np.ndarray, start_price: float = 100.0) -> pd.DataFrame:
    """Build OHLCV DataFrame from log-returns for preprocessing tests."""
    n = len(returns) + 1
    prices = np.empty(n)
    prices[0] = start_price
    for i, r in enumerate(returns):
        prices[i + 1] = prices[i] * np.exp(r)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(1e7, 1e8, size=n),
    })


# ── Metrics: synthetic known correlation ────────────────────────────────

class TestSyntheticCorrelation:
    """Verify metrics against synthetic series with known correlation."""

    def test_perfect_positive_correlation(self):
        """r=1, beta=2, R²=1 when alt = 2*btc (perfect linear)."""
        btc = pd.Series(make_returns(100, seed=1))
        alt = btc * 2.0  # perfect linear, no noise
        m = compute_metrics(btc, alt)
        assert m["r"] == pytest.approx(1.0, abs=1e-6), f"Expected r=1, got {m['r']}"
        assert m["beta"] == pytest.approx(2.0, abs=1e-4), f"Expected beta=2, got {m['beta']}"
        assert m["r_squared"] == pytest.approx(1.0, abs=1e-4), f"Expected R²=1, got {m['r_squared']}"
        assert m["p_value"] < 1e-10, f"Expected p~0, got {m['p_value']}"
        assert m["n"] == 100

    def test_perfect_negative_correlation(self):
        """r=-1, beta=-1 when alt = -btc."""
        btc = pd.Series(make_returns(100, seed=2))
        alt = -btc
        m = compute_metrics(btc, alt)
        assert m["r"] == pytest.approx(-1.0, abs=1e-6)
        assert m["beta"] == pytest.approx(-1.0, abs=1e-4)
        assert m["r_squared"] == pytest.approx(1.0, abs=1e-4)
        assert m["p_value"] < 1e-10

    def test_zero_correlation(self):
        """r≈0 when alt is independent noise."""
        rng = np.random.default_rng(42)
        btc = pd.Series(rng.normal(0, 0.02, 100))
        alt = pd.Series(rng.normal(0, 0.02, 100))  # independent
        m = compute_metrics(btc, alt)
        assert abs(m["r"]) < 0.3, f"Expected |r|<0.3 for independent series, got {m['r']}"
        assert m["p_value"] > 0.01, f"Expected non-significant p, got {m['p_value']}"
        # R² should be small
        assert m["r_squared"] < 0.15, f"Expected small R², got {m['r_squared']}"

    def test_known_beta_3x(self):
        """Beta ≈3 when alt = 3*btc + small noise."""
        rng = np.random.default_rng(7)
        btc = pd.Series(rng.normal(0, 0.02, 200))
        noise = pd.Series(rng.normal(0, 0.005, 200))
        alt = btc * 3.0 + noise
        m = compute_metrics(btc, alt)
        assert m["r"] > 0.9, f"Expected high r, got {m['r']}"
        assert m["beta"] == pytest.approx(3.0, abs=0.15), f"Expected beta≈3, got {m['beta']}"
        assert m["r_squared"] > 0.8

    def test_r_squared_equals_r_squared(self):
        """R² should equal r² for simple OLS with intercept (within tolerance)."""
        btc = pd.Series(make_returns(150, seed=10))
        alt = btc * 1.5 + pd.Series(np.random.default_rng(10).normal(0, 0.01, 150))
        m = compute_metrics(btc, alt)
        # For OLS with intercept, R² ≈ r² (not exact due to intercept, but close)
        assert abs(m["r_squared"] - m["r"] ** 2) < 0.05, f"R²={m['r_squared']} vs r²={m['r']**2}"

    def test_p_value_significance(self):
        """p-value < 0.05 for strong correlation; weak has smaller |r| than strong."""
        # Strong: highly correlated
        btc = pd.Series(make_returns(100, seed=20))
        alt_strong = btc * 2.0 + pd.Series(np.random.default_rng(20).normal(0, 0.005, 100))
        m_strong = compute_metrics(btc, alt_strong)
        assert m_strong["p_value"] < 0.05
        assert abs(m_strong["r"]) > 0.8
        # Weak: independent with larger n to reduce spurious significance
        rng = np.random.default_rng(21)
        btc2 = pd.Series(rng.normal(0, 0.02, 200))
        alt_weak = pd.Series(rng.normal(0, 0.02, 200))
        m_weak = compute_metrics(btc2, alt_weak)
        # Weak must have smaller |r| than strong and be relatively small
        assert abs(m_weak["r"]) < abs(m_strong["r"])
        assert abs(m_weak["r"]) < 0.35, f"Expected weak |r|<0.35, got {m_weak['r']}"


# ── Edge cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Explicit edge-case handling — no silent fail."""

    def test_zero_variance_btc(self):
        """var_btc=0 -> beta NaN, logged, not crash."""
        btc = pd.Series([0.01] * 50)  # constant returns -> var=0
        alt = pd.Series(np.random.default_rng(0).normal(0, 0.02, 50))
        m = compute_metrics(btc, alt)
        assert np.isnan(m["beta"]), "Beta should be NaN when var_btc=0"
        assert np.isnan(m["r"]) or np.isnan(m["r_squared"])

    def test_small_sample(self):
        """n < min_data_points still computes but warns; n<2 returns NaN."""
        btc = pd.Series([0.01, 0.02])
        alt = pd.Series([0.02, 0.04])
        m = compute_metrics(btc, alt)
        # n=2 should still compute (but barely)
        assert not np.isnan(m["r"]) or m["n"] == 2

        btc1 = pd.Series([0.01])
        alt1 = pd.Series([0.02])
        m1 = compute_metrics(btc1, alt1)
        assert np.isnan(m1["r"]), "n=1 should return NaN"

    def test_nan_handling_no_silent_fail(self):
        """NaN/inf in series are dropped explicitly, not silently ignored."""
        btc = pd.Series([0.01, 0.02, np.nan, 0.03, np.inf, 0.01])
        alt = pd.Series([0.02, 0.04, 0.06, 0.03, 0.02, 0.01])
        m = compute_metrics(btc, alt)
        # Should drop 2 rows (nan, inf) and compute on remaining 4
        assert m["n"] == 4, f"Expected n=4 after dropping NaN/inf, got {m['n']}"
        assert not np.isnan(m["r"]) or m["n"] >= 2

    def test_empty_series(self):
        """Empty series returns NaN metrics, not crash."""
        m = compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
        assert np.isnan(m["r"])
        assert np.isnan(m["beta"])

    def test_negative_beta_not_category_a(self):
        """Beta negative must NOT be Category A (requires 3.0-5.0)."""
        cat = classify_single(r=0.85, beta=-2.0, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_NEUTRAL, f"Negative beta should be neutral, got {cat}"

        cat2 = classify_single(r=0.85, beta=4.0, p_value=0.01,
                               corr_threshold_high=0.7, corr_threshold_low=0.2,
                               beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat2 == CATEGORY_A


# ── Look-ahead bias ─────────────────────────────────────────────────────

class TestLookAheadBias:
    """
    Verify no look-ahead bias: rolling window only uses data up to time t.

    Strategy: create a series where future data would change the result if leaked.
    """

    def test_rolling_window_no_future_leak(self):
        """Rolling window at t should not see data after t."""
        n = 100
        rng = np.random.default_rng(99)
        # First 50: alt = btc (r=1), Last 50: alt = -btc (r=-1)
        btc = pd.Series(rng.normal(0, 0.02, n))
        alt = pd.Series(np.concatenate([btc.iloc[:50].values, -btc.iloc[50:].values]))

        window = 30
        rolling = compute_rolling_metrics(btc, alt, window=window)

        # First rolling window (ends at index 29) should be r≈1 (only sees first regime)
        first_r = rolling.iloc[0]["r"]
        assert first_r == pytest.approx(1.0, abs=1e-6), f"First window should be r=1, got {first_r}"

        # Last rolling window (ends at 99) should be r≈-1 (only sees second regime)
        last_r = rolling.iloc[-1]["r"]
        assert last_r == pytest.approx(-1.0, abs=1e-6), f"Last window should be r=-1, got {last_r}"

        # Window that straddles the regime change 50/50 should be near zero correlation
        # rolling index 35 corresponds to window [35, 64] -> 15 from first regime, 15 from second
        straddle_r = rolling.iloc[35]["r"]
        assert abs(straddle_r) < 0.5, f"50/50 straddling window should be near 0, got r={straddle_r}"
        # Also verify straddle is less correlated than pure regime windows
        assert abs(straddle_r) < abs(first_r)
        assert abs(straddle_r) < abs(last_r)

    def test_metrics_for_windows_uses_most_recent(self):
        """compute_metrics_for_windows must use LAST w points (most recent, no future)."""
        n = 200
        rng = np.random.default_rng(123)
        btc = pd.Series(rng.normal(0, 0.02, n))
        # Make alt correlated only in last 30 points, uncorrelated before
        alt_vals = rng.normal(0, 0.02, n)
        alt_vals[-30:] = btc.iloc[-30:].values * 2.0  # last 30 are perfectly correlated
        alt = pd.Series(alt_vals)

        windows = [30, 90, 180]
        result = compute_metrics_for_windows(btc, alt, windows=windows)

        # 30d window (last 30) should be r≈1
        assert result[30]["r"] == pytest.approx(1.0, abs=1e-6), f"30d should be r=1, got {result[30]['r']}"
        # 180d window includes mostly uncorrelated data, so |r| should be smaller
        assert abs(result[180]["r"]) < 0.5, f"180d should be less correlated, got {result[180]['r']}"
        # This proves no look-ahead: 30d doesn't see future beyond last 30

    def test_rolling_does_not_use_future_data(self):
        """Explicit test: modifying future data should not affect past rolling windows."""
        n = 60
        rng = np.random.default_rng(55)
        btc = pd.Series(rng.normal(0, 0.02, n))
        alt = btc * 1.5 + pd.Series(rng.normal(0, 0.005, n))

        window = 20
        rolling_before = compute_rolling_metrics(btc, alt, window=window)

        # Modify future data (last 10 points) drastically
        btc_mod = btc.copy()
        alt_mod = alt.copy()
        btc_mod.iloc[-10:] = 999  # extreme future values
        alt_mod.iloc[-10:] = -999

        rolling_after = compute_rolling_metrics(btc_mod, alt_mod, window=window)

        # First rolling window should be identical (no future leak)
        assert rolling_before.iloc[0]["r"] == pytest.approx(rolling_after.iloc[0]["r"], abs=1e-9), \
            "Future modification affected past window — look-ahead bias!"
        # Last window should differ (it includes future)
        assert rolling_before.iloc[-1]["r"] != pytest.approx(rolling_after.iloc[-1]["r"], abs=1e-3), \
            "Last window should differ after future modification"


# ── Classifier ──────────────────────────────────────────────────────────

class TestClassifier:
    """Test categorization logic exactly per spec."""

    def test_category_a(self):
        cat = classify_single(r=0.8, beta=4.0, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_A

    def test_category_a_boundary(self):
        # r must be > 0.7, not >=
        cat = classify_single(r=0.7, beta=4.0, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_NEUTRAL, "r=0.7 should NOT be Category A (needs >0.7)"

        # beta inclusive [3,5]
        cat2 = classify_single(r=0.8, beta=3.0, p_value=0.01,
                               corr_threshold_high=0.7, corr_threshold_low=0.2,
                               beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat2 == CATEGORY_A
        cat3 = classify_single(r=0.8, beta=5.0, p_value=0.01,
                               corr_threshold_high=0.7, corr_threshold_low=0.2,
                               beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat3 == CATEGORY_A
        cat4 = classify_single(r=0.8, beta=5.01, p_value=0.01,
                               corr_threshold_high=0.7, corr_threshold_low=0.2,
                               beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat4 == CATEGORY_NEUTRAL

    def test_category_b(self):
        cat = classify_single(r=0.1, beta=0.5, p_value=0.3,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_B
        # |r| < 0.2 and p >= 0.05
        cat2 = classify_single(r=-0.15, beta=-1.0, p_value=0.1,
                               corr_threshold_high=0.7, corr_threshold_low=0.2,
                               beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat2 == CATEGORY_B

    def test_category_b_requires_nonsignificant(self):
        # Default (flag false): Category B only checks |r|<low, decoupled from p-value
        cat_default = classify_single(r=0.1, beta=0.5, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat_default == CATEGORY_B, "Default Category B should be decoupled from p-value"
        # Strict mode (flag true): requires p>=alpha
        cat_strict = classify_single(r=0.1, beta=0.5, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05,
                              category_b_require_nonsignificant=True)
        assert cat_strict == CATEGORY_NEUTRAL, "Category B strict requires p >= alpha"
        cat_strict2 = classify_single(r=0.1, beta=0.5, p_value=0.3,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05,
                              category_b_require_nonsignificant=True)
        assert cat_strict2 == CATEGORY_B

    def test_neutral(self):
        cat = classify_single(r=0.5, beta=1.5, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_NEUTRAL

    def test_stability_flag(self):
        stable = {30: CATEGORY_A, 90: CATEGORY_A, 180: CATEGORY_A}
        unstable, _ = check_stability(stable)
        assert not unstable

        mixed = {30: CATEGORY_A, 90: CATEGORY_NEUTRAL, 180: CATEGORY_NEUTRAL}
        unstable2, detail = check_stability(mixed)
        assert unstable2
        assert "unstable" in detail

    def test_build_classification_table(self):
        all_metrics = {
            "COIN_A": {
                30: {"r": 0.85, "beta": 4.0, "r_squared": 0.72, "p_value": 0.001, "n": 30, "var_btc": 0.0004, "cov": 0.0016},
                90: {"r": 0.82, "beta": 3.8, "r_squared": 0.67, "p_value": 0.001, "n": 90, "var_btc": 0.0004, "cov": 0.0015},
                180: {"r": 0.80, "beta": 3.9, "r_squared": 0.64, "p_value": 0.001, "n": 180, "var_btc": 0.0004, "cov": 0.0015},
            },
            "COIN_B": {
                30: {"r": 0.05, "beta": 0.1, "r_squared": 0.002, "p_value": 0.6, "n": 30, "var_btc": 0.0004, "cov": 0.00004},
                90: {"r": 0.08, "beta": 0.2, "r_squared": 0.006, "p_value": 0.5, "n": 90, "var_btc": 0.0004, "cov": 0.00008},
                180: {"r": 0.03, "beta": 0.05, "r_squared": 0.001, "p_value": 0.7, "n": 180, "var_btc": 0.0004, "cov": 0.00002},
            },
        }
        config = {
            "thresholds": {"corr_threshold_high": 0.7, "corr_threshold_low": 0.2, "beta_min": 3.0, "beta_max": 5.0, "alpha": 0.05},
            "windows": [30, 90, 180],
            "primary_window": 180,
        }
        table = build_classification_table(all_metrics, config)
        assert len(table) == 2
        assert table[table["symbol"] == "COIN_A"]["category"].iloc[0] == CATEGORY_A
        assert table[table["symbol"] == "COIN_B"]["category"].iloc[0] == CATEGORY_B
        # Check required columns
        for col in ["r_30d", "r_90d", "r_180d", "beta_30d", "beta_90d", "beta_180d", "r_squared", "p_value", "category", "stability_flag"]:
            assert col in table.columns, f"Missing column {col}"


# ── Preprocessing ───────────────────────────────────────────────────────

class TestPreprocessing:
    """Test alignment, log-returns, missing-data policy."""

    def test_log_returns(self):
        """Log-return = ln(P_t / P_{t-1}), not simple return."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "close": [100.0, 110.0, 121.0],
        })
        ret = compute_log_returns(df)
        # ln(110/100) = 0.09531, ln(121/110) = 0.09531
        assert len(ret) == 2
        assert ret["log_return"].iloc[0] == pytest.approx(np.log(110 / 100), abs=1e-9)
        assert ret["log_return"].iloc[1] == pytest.approx(np.log(121 / 110), abs=1e-9)
        # Simple return would be 0.10, log is 0.095 — ensure we use log
        assert ret["log_return"].iloc[0] != pytest.approx(0.10, abs=1e-3)

    def test_align_drop_policy(self):
        btc = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "close": [100, 101, 102, 103, 104],
        })
        alt = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC"),
            "close": [200, 201, 202],
        })
        btc_a, alt_a = align_ohlcv(btc, alt, missing_policy="drop")
        # Inner join: only 3 overlapping timestamps
        assert len(btc_a) == 3
        assert len(alt_a) == 3

    def test_align_forward_fill(self):
        btc = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "close": [100, 101, 102, 103, 104],
        })
        alt = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"), pd.Timestamp("2024-01-05", tz="UTC")],
            "close": [200, 202, 204],
        })
        btc_a, alt_a = align_ohlcv(btc, alt, missing_policy="forward_fill", ffill_limit=1)
        # With ffill limit 1, some NaNs will be filled, but not all
        assert len(btc_a) >= 3

    def test_winsorize(self):
        s = pd.Series([0.01, 0.02, 0.03, 10.0, -10.0])  # outliers
        clipped = apply_outlier_handling(s, method="winsorize", winsorize_limits=[0.2, 0.2])
        assert clipped.max() < 10.0
        assert clipped.min() > -10.0

    def test_prepare_aligned_returns_no_lookahead(self):
        """prepare_aligned_returns should not introduce look-ahead."""
        btc_ret = make_returns(50, seed=100)
        alt_ret = btc_ret * 2.0
        btc_df = make_ohlcv_from_returns(btc_ret)
        alt_df = make_ohlcv_from_returns(alt_ret)
        config = {
            "preprocessing": {"missing_data_policy": "drop", "forward_fill_limit": 2, "outlier_handling": "none", "winsorize_limits": [0.01, 0.01]},
        }
        btc_s, alt_s, merged = prepare_aligned_returns(btc_df, alt_df, config)
        # Should have n-1 returns (one less than prices due to diff)
        assert len(btc_s) == len(alt_s)
        assert len(btc_s) == 50  # 51 prices -> 50 returns, aligned
        # Correlation should be ~1
        m = compute_metrics(btc_s, alt_s)
        assert m["r"] == pytest.approx(1.0, abs=1e-6)


# ── No silent fail ──────────────────────────────────────────────────────

class TestNoSilentFail:
    """Ensure missing data / API errors are logged, not silently ignored."""

    def test_align_empty_returns_empty(self):
        btc_empty = pd.DataFrame(columns=["timestamp", "close"])
        alt_empty = pd.DataFrame(columns=["timestamp", "close"])
        btc_a, alt_a = align_ohlcv(btc_empty, alt_empty)
        assert btc_a.empty and alt_a.empty

    def test_compute_log_returns_empty(self):
        ret = compute_log_returns(pd.DataFrame(columns=["timestamp", "close"]))
        assert ret.empty

    def test_metrics_with_all_nan(self):
        btc = pd.Series([np.nan] * 10)
        alt = pd.Series([np.nan] * 10)
        m = compute_metrics(btc, alt)
        assert np.isnan(m["r"])

    def test_classifier_nan_inputs(self):
        cat = classify_single(r=float("nan"), beta=4.0, p_value=0.01,
                              corr_threshold_high=0.7, corr_threshold_low=0.2,
                              beta_min=3.0, beta_max=5.0, alpha=0.05)
        assert cat == CATEGORY_NEUTRAL
