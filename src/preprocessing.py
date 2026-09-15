"""
Preprocessing: timestamp alignment, missing-data policy, log-returns, outlier handling.

All thresholds/options come from config.yaml — no hardcoding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def align_ohlcv(
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    missing_policy: str = "drop",
    ffill_limit: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align BTC and alt OHLCV on timestamp (inner join on timestamp).

    Missing-data policy (explicit, never silent):
      - "drop": inner join, drop any row where either close is NaN.
      - "forward_fill": forward-fill with limit, then drop remaining NaNs.

    Args:
        btc_df: DataFrame with [timestamp, close, ...].
        alt_df: DataFrame with [timestamp, close, ...].
        missing_policy: "drop" or "forward_fill".
        ffill_limit: max consecutive NaNs to forward-fill (only if policy == forward_fill).

    Returns:
        (btc_aligned, alt_aligned) DataFrames indexed by timestamp, sorted, with close column.

    Units:
        close in quote currency (e.g. USDT); timestamp is UTC datetime.
    """
    if btc_df.empty or alt_df.empty:
        logger.warning("align_ohlcv: one or both DataFrames empty (btc=%d, alt=%d) — returning empty", len(btc_df), len(alt_df))
        return btc_df.copy(), alt_df.copy()

    # Ensure timestamp is datetime and sorted
    for df in (btc_df, alt_df):
        if "timestamp" not in df.columns:
            raise ValueError("align_ohlcv: DataFrame missing 'timestamp' column")
        if "close" not in df.columns:
            raise ValueError("align_ohlcv: DataFrame missing 'close' column")

    btc = btc_df[["timestamp", "close"]].copy()
    alt = alt_df[["timestamp", "close"]].copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    alt["timestamp"] = pd.to_datetime(alt["timestamp"], utc=True)

    # Full outer join to detect missing timestamps explicitly
    merged = pd.merge(btc, alt, on="timestamp", how="outer", suffixes=("_btc", "_alt"), sort=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    n_before = len(merged)
    n_missing_btc = merged["close_btc"].isna().sum()
    n_missing_alt = merged["close_alt"].isna().sum()
    if n_missing_btc > 0 or n_missing_alt > 0:
        logger.info(
            "align_ohlcv: outer join %d rows — missing btc=%d alt=%d (%.1f%% total missing)",
            n_before, n_missing_btc, n_missing_alt,
            100 * (n_missing_btc + n_missing_alt) / (2 * n_before) if n_before else 0,
        )

    if missing_policy == "forward_fill":
        # Forward-fill with limit, then drop remaining NaNs
        merged["close_btc"] = merged["close_btc"].ffill(limit=ffill_limit)
        merged["close_alt"] = merged["close_alt"].ffill(limit=ffill_limit)
        filled_btc = n_missing_btc - merged["close_btc"].isna().sum()
        filled_alt = n_missing_alt - merged["close_alt"].isna().sum()
        logger.info(
            "align_ohlcv forward_fill(limit=%d): filled btc=%d alt=%d, remaining NaN btc=%d alt=%d",
            ffill_limit, filled_btc, filled_alt,
            merged["close_btc"].isna().sum(), merged["close_alt"].isna().sum(),
        )
        # Drop any remaining NaNs (explicit, not silent — logged)
        before_drop = len(merged)
        merged = merged.dropna(subset=["close_btc", "close_alt"])
        dropped = before_drop - len(merged)
        if dropped > 0:
            logger.info("align_ohlcv forward_fill: dropped %d rows still containing NaN after ffill", dropped)
    elif missing_policy == "drop":
        before_drop = len(merged)
        merged = merged.dropna(subset=["close_btc", "close_alt"])
        dropped = before_drop - len(merged)
        if dropped > 0:
            logger.info("align_ohlcv drop policy: dropped %d rows with missing data (%.1f%%)", dropped, 100 * dropped / before_drop if before_drop else 0)
    else:
        raise ValueError(f"Unknown missing_data_policy: {missing_policy!r} (expected 'drop' or 'forward_fill')")

    # Split back
    btc_aligned = merged[["timestamp", "close_btc"]].rename(columns={"close_btc": "close"}).reset_index(drop=True)
    alt_aligned = merged[["timestamp", "close_alt"]].rename(columns={"close_alt": "close"}).reset_index(drop=True)

    logger.debug("align_ohlcv result: %d aligned rows", len(btc_aligned))
    return btc_aligned, alt_aligned


def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute log-returns: R_t = ln(P_t / P_{t-1}).

    Args:
        df: DataFrame with timestamp and price_col.
        price_col: column name for price.

    Returns:
        DataFrame with columns [timestamp, log_return] (first row dropped due to NaN).

    Units:
        log_return is dimensionless (log-price change, NOT percent). E.g. 0.01 ≈ 1% move.
    """
    if df.empty:
        logger.warning("compute_log_returns: empty DataFrame — returning empty")
        return pd.DataFrame(columns=["timestamp", "log_return"])
    if price_col not in df.columns:
        raise ValueError(f"compute_log_returns: missing column {price_col!r}")
    if (df[price_col] <= 0).any():
        n_bad = (df[price_col] <= 0).sum()
        logger.warning("compute_log_returns: %d non-positive prices — will produce NaN/inf, dropping", n_bad)

    out = df[["timestamp", price_col]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    # Use numpy log for precision
    prices = out[price_col].to_numpy(dtype=float)
    # Guard against non-positive
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(prices[1:] / prices[:-1])
    # First row has no prior price
    result = pd.DataFrame({
        "timestamp": out["timestamp"].iloc[1:].reset_index(drop=True),
        "log_return": log_ret,
    })
    # Drop inf/nan from bad prices
    before = len(result)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return"]).reset_index(drop=True)
    dropped = before - len(result)
    if dropped > 0:
        logger.info("compute_log_returns: dropped %d rows with inf/NaN returns (bad prices)", dropped)
    return result


def apply_outlier_handling(
    series: pd.Series,
    method: str = "none",
    winsorize_limits: list[float] | None = None,
) -> pd.Series:
    """
    Apply outlier handling to a return series.

    Args:
        series: pd.Series of log-returns (dimensionless).
        method: "none" | "winsorize" | "robust" (robust is handled in metrics.py via RLM).
        winsorize_limits: [lower_quantile, upper_quantile] e.g. [0.01, 0.01] clips 1% tails.

    Returns:
        Series with outliers handled (or original if method == "none").

    Units:
        Input/output are log-returns (dimensionless).
    """
    if method == "none" or series.empty:
        return series
    if method == "winsorize":
        if winsorize_limits is None or len(winsorize_limits) != 2:
            raise ValueError("winsorize_limits must be [lower_q, upper_q] e.g. [0.01, 0.01]")
        lo_q, hi_q = winsorize_limits
        lo = series.quantile(lo_q)
        hi = series.quantile(1 - hi_q)
        clipped = series.clip(lower=lo, upper=hi)
        n_clipped = (clipped != series).sum()
        logger.info("winsorize: clipped %d/%d values (%.1f%%) to [%.4f, %.4f]", n_clipped, len(series), 100 * n_clipped / len(series) if len(series) else 0, lo, hi)
        return clipped
    if method == "robust":
        # Robust handling is done at regression time (Huber RLM); here we just log
        logger.info("outlier_handling=robust: no clipping here — robust regression will be used in metrics")
        return series
    raise ValueError(f"Unknown outlier_handling method: {method!r}")


def prepare_aligned_returns(
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Full preprocessing pipeline for one alt vs BTC.

    Steps (in order, no look-ahead):
      1. Align timestamps (explicit missing-data policy).
      2. Compute log-returns for both.
      3. Re-align returns on timestamp (inner join).
      4. Apply outlier handling (winsorize if configured).

    Args:
        btc_df: BTC OHLCV DataFrame.
        alt_df: Alt OHLCV DataFrame.
        config: parsed config.yaml dict.

    Returns:
        (btc_returns, alt_returns, merged_returns_df) where merged has [timestamp, btc, alt].

    Units:
        Returns are log-returns (dimensionless, NOT percent).
    """
    prep_cfg = config.get("preprocessing", {})
    missing_policy: str = prep_cfg.get("missing_data_policy", "drop")
    ffill_limit: int = prep_cfg.get("forward_fill_limit", 2)
    outlier_method: str = prep_cfg.get("outlier_handling", "none")
    winsorize_limits: list[float] | None = prep_cfg.get("winsorize_limits", [0.01, 0.01])

    # 1. Align OHLCV
    btc_aligned, alt_aligned = align_ohlcv(btc_df, alt_df, missing_policy=missing_policy, ffill_limit=ffill_limit)

    # 2. Log returns
    btc_ret_df = compute_log_returns(btc_aligned, price_col="close")
    alt_ret_df = compute_log_returns(alt_aligned, price_col="close")

    if btc_ret_df.empty or alt_ret_df.empty:
        logger.warning("prepare_aligned_returns: empty returns (btc=%d, alt=%d)", len(btc_ret_df), len(alt_ret_df))
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    # 3. Re-align returns on timestamp
    merged = pd.merge(btc_ret_df, alt_ret_df, on="timestamp", how="inner", suffixes=("_btc", "_alt"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.rename(columns={"log_return_btc": "btc", "log_return_alt": "alt"})
    # Drop any NaN/inf that slipped through
    before = len(merged)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(merged) < before:
        logger.info("prepare_aligned_returns: dropped %d rows with NaN/inf after merge", before - len(merged))

    # 4. Outlier handling (applied to both series independently)
    if outlier_method == "winsorize":
        merged["btc"] = apply_outlier_handling(merged["btc"], method="winsorize", winsorize_limits=winsorize_limits)
        merged["alt"] = apply_outlier_handling(merged["alt"], method="winsorize", winsorize_limits=winsorize_limits)

    btc_series = merged["btc"]
    alt_series = merged["alt"]
    return btc_series, alt_series, merged
