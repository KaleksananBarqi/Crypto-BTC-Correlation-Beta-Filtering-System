"""Tests for src/visualize.py — scatter plot generation, column resolution, edge cases.

Uses the headless Agg backend so figures render without a display (CI-safe).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.visualize import plot_beta_vs_correlation  # noqa: E402


def _classification_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "category": ["A_levered_btc", "B_uncorrelated", "neutral", "A_levered_btc"],
            "r_180d": [0.9, 0.05, 0.5, 0.85],
            "beta_180d": [4.0, 0.5, 1.2, 3.5],
        }
    )


def test_plot_creates_file(tmp_path, sample_config):
    out = tmp_path / "scatter.png"
    path = plot_beta_vs_correlation(_classification_df(), sample_config, output_path=str(out))
    assert path == out
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_falls_back_when_primary_window_columns_missing(tmp_path, sample_config):
    df = _classification_df().rename(columns={"r_180d": "r_90d", "beta_180d": "beta_90d"})
    out = tmp_path / "fallback.png"
    path = plot_beta_vs_correlation(df, sample_config, output_path=str(out))
    assert path.exists()


def test_plot_explicit_columns_override(tmp_path, sample_config):
    df = _classification_df().rename(columns={"r_180d": "r_30d", "beta_180d": "beta_30d"})
    out = tmp_path / "explicit.png"
    path = plot_beta_vs_correlation(
        df, sample_config, output_path=str(out), r_col="r_30d", beta_col="beta_30d"
    )
    assert path.exists()


def test_plot_missing_r_beta_columns_raises(tmp_path, sample_config):
    df = pd.DataFrame({"symbol": ["AAA"], "category": ["neutral"], "unrelated": [1.0]})
    with pytest.raises(ValueError):
        plot_beta_vs_correlation(df, sample_config, output_path=str(tmp_path / "x.png"))


def test_plot_all_nan_category_is_skipped(tmp_path, sample_config):
    df = _classification_df()
    # Wipe the only Category A point -> that category has no plottable rows.
    df.loc[0, "r_180d"] = float("nan")
    df.loc[0, "beta_180d"] = float("nan")
    out = tmp_path / "nan.png"
    path = plot_beta_vs_correlation(df, sample_config, output_path=str(out))
    assert path.exists()


def test_plot_empty_dataframe(tmp_path, sample_config):
    df = pd.DataFrame(columns=["symbol", "category", "r_180d", "beta_180d"])
    out = tmp_path / "empty.png"
    path = plot_beta_vs_correlation(df, sample_config, output_path=str(out))
    assert path.exists()


def test_plot_without_category_column(tmp_path, sample_config):
    df = _classification_df().drop(columns=["category"])
    out = tmp_path / "nocat.png"
    path = plot_beta_vs_correlation(df, sample_config, output_path=str(out))
    assert path.exists()
