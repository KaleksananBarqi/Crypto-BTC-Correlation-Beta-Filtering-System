"""
Metrics: Pearson r, Beta (OLS slope), R², p-value.

Definitions (EXACT per spec):
  - Correlation r = Pearson correlation of log-returns BTC vs alt, [-1, 1]
  - Beta β = Cov(R_alt, R_btc) / Var(R_btc) = OLS slope, unbounded
  - R² = coefficient of determination [0, 1]
  - p-value = statistical significance of r (from scipy.stats.pearsonr)

All functions use log-returns (dimensionless, NOT percent) as input.
Thresholds are NEVER hardcoded — passed via config dict.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_metrics(
    btc_returns: pd.Series,
    alt_returns: pd.Series,
    robust: bool = False,
) -> Dict[str, float]:
    """
    Compute r, beta, R², p-value for aligned log-return series.

    Args:
        btc_returns: pd.Series of BTC log-returns (dimensionless, NOT percent).
        alt_returns: pd.Series of alt log-returns (dimensionless, NOT percent).
        robust: if True, use Huber RLM for beta instead of OLS.

    Returns:
        Dict with keys: r, beta, r_squared, p_value, n, var_btc, cov.

    Units:
        Inputs are log-returns (dimensionless). Outputs: r in [-1,1], beta unbounded,
        r_squared in [0,1], p_value in [0,1].

    Notes:
        - Uses scipy.stats.pearsonr for r and p_value.
        - Uses statsmodels OLS (or RLM if robust) for beta and R².
        - Returns NaN for all metrics if n < 2 or var_btc == 0 (logged explicitly).
    """
    # Align and drop NaN/inf explicitly (no silent fail)
    df = pd.DataFrame({"btc": btc_returns, "alt": alt_returns})
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("compute_metrics: dropped %d NaN/inf rows before calculation", dropped)

    n = len(df)
    if n < 2:
        logger.warning("compute_metrics: n=%d < 2 — insufficient data, returning NaN", n)
        return {"r": float("nan"), "beta": float("nan"), "r_squared": float("nan"), "p_value": float("nan"), "n": float(n), "var_btc": float("nan"), "cov": float("nan")}

    btc = df["btc"].to_numpy(dtype=float)
    alt = df["alt"].to_numpy(dtype=float)

    var_btc = float(np.var(btc, ddof=1)) if n > 1 else float("nan")
    cov = float(np.cov(alt, btc, ddof=1)[0, 1]) if n > 1 else float("nan")

    if var_btc == 0 or np.isnan(var_btc):
        logger.warning("compute_metrics: var_btc=%.6g — beta undefined (zero variance), returning NaN", var_btc)
        return {"r": float("nan"), "beta": float("nan"), "r_squared": float("nan"), "p_value": float("nan"), "n": float(n), "var_btc": var_btc, "cov": cov}

    # Pearson r and p-value
    try:
        r, p_value = pearsonr(btc, alt)
    except Exception as exc:  # noqa: BLE001
        logger.error("pearsonr failed (n=%d): %s — returning NaN", n, exc)
        r, p_value = float("nan"), float("nan")

    # Beta and R² via statsmodels
    beta: float = float("nan")
    r_squared: float = float("nan")
    try:
        import statsmodels.api as sm  # type: ignore

        X = sm.add_constant(btc)  # adds intercept
        if robust:
            # Robust regression (Huber)
            try:
                rlm = sm.RLM(alt, X, M=sm.robust.norms.HuberT())
                res = rlm.fit()
                beta = float(res.params[1]) if len(res.params) > 1 else float("nan")
                # RLM has no rsquared; compute pseudo-R² as 1 - SSR/SST
                y_pred = res.fittedvalues
                ss_res = float(np.sum((alt - y_pred) ** 2))
                ss_tot = float(np.sum((alt - np.mean(alt)) ** 2))
                r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
                # Clamp R² to [0,1] for reporting (pseudo-R² can be negative)
                r_squared = max(0.0, min(1.0, r_squared)) if not np.isnan(r_squared) else float("nan")
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLM failed, falling back to OLS: %s", exc)
                robust = False  # fallback

        if not robust:
            model = sm.OLS(alt, X)
            res = model.fit()
            beta = float(res.params[1]) if len(res.params) > 1 else float("nan")
            r_squared = float(res.rsquared) if hasattr(res, "rsquared") else float("nan")

        # Cross-check: beta should equal cov/var_btc (within tolerance) for OLS
        beta_cov = cov / var_btc if var_btc != 0 else float("nan")
        if not np.isnan(beta) and not np.isnan(beta_cov) and abs(beta - beta_cov) > 1e-6:
            logger.debug("beta OLS=%.6f vs cov/var=%.6f diff=%.2e (expected small diff due to intercept)", beta, beta_cov, beta - beta_cov)

    except ImportError as exc:
        logger.error("statsmodels not installed — cannot compute beta/R²: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("OLS/RLM failed: %s — returning NaN for beta/R²", exc)
        beta, r_squared = float("nan"), float("nan")

    return {
        "r": float(r) if not np.isnan(r) else float("nan"),
        "beta": float(beta),
        "r_squared": float(r_squared),
        "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
        "n": float(n),
        "var_btc": float(var_btc),
        "cov": float(cov),
    }


def compute_metrics_for_windows(
    btc_series: pd.Series,
    alt_series: pd.Series,
    windows: List[int],
    robust: bool = False,
    min_data_points: int = 30,
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for each window using the MOST RECENT window (no look-ahead).

    For each window w, uses the last w points of the aligned series (data up to time t only).

    Args:
        btc_series: pd.Series of BTC log-returns (dimensionless), sorted by time ascending.
        alt_series: pd.Series of alt log-returns (dimensionless), same index/length as btc_series.
        windows: list of window sizes in days (e.g. [30, 90, 180]).
        robust: use robust regression if True.
        min_data_points: minimum n required per window (from config).

    Returns:
        Dict mapping window -> metrics dict (as from compute_metrics).

    Units:
        Inputs are log-returns (dimensionless). Outputs per compute_metrics.

    No look-ahead bias:
        Each window only uses data up to time t (the end of the series). No future data.
    """
    result: Dict[int, Dict[str, float]] = {}
    total_n = len(btc_series)
    for w in windows:
        if total_n < w:
            logger.warning(
                "compute_metrics_for_windows: total_n=%d < window %d — using all available (%d) but flagging small-sample",
                total_n, w, total_n,
            )
        # Use last w points (most recent window, no look-ahead)
        btc_w = btc_series.iloc[-w:] if total_n >= w else btc_series
        alt_w = alt_series.iloc[-w:] if total_n >= w else alt_series

        if len(btc_w) < min_data_points:
            logger.warning(
                "Window %d: n=%d < min_data_points %d — metrics will be computed but p-value check will fail significance",
                w, len(btc_w), min_data_points,
            )
        metrics = compute_metrics(btc_w, alt_w, robust=robust)
        result[w] = metrics
        logger.debug("Window %d: r=%.3f beta=%.3f R²=%.3f p=%.3g n=%d", w, metrics["r"], metrics["beta"], metrics["r_squared"], metrics["p_value"], int(metrics["n"]))
    return result


def compute_rolling_metrics(
    btc_series: pd.Series,
    alt_series: pd.Series,
    window: int,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Compute rolling metrics over time (for look-ahead bias testing).

    Each rolling window [t-window+1, t] only uses data up to time t (no future).

    Args:
        btc_series: pd.Series of BTC log-returns (dimensionless), indexed by timestamp or integer.
        alt_series: pd.Series of alt log-returns (dimensionless).
        window: rolling window size.
        robust: use robust regression.

    Returns:
        DataFrame with columns [timestamp/index, r, beta, r_squared, p_value] for each window end.

    Units:
        Inputs log-returns dimensionless; outputs as per compute_metrics.
    """
    if len(btc_series) != len(alt_series):
        raise ValueError(f"Series length mismatch: btc {len(btc_series)} vs alt {len(alt_series)}")
    if len(btc_series) < window:
        logger.warning("compute_rolling_metrics: len %d < window %d — returning empty", len(btc_series), window)
        return pd.DataFrame(columns=["r", "beta", "r_squared", "p_value"])

    # Preserve timestamp if series has it
    has_timestamp = isinstance(btc_series.index, pd.DatetimeIndex) or "timestamp" in str(btc_series.index.name or "")

    rows = []
    for end in range(window, len(btc_series) + 1):
        start = end - window
        btc_w = btc_series.iloc[start:end]
        alt_w = alt_series.iloc[start:end]
        m = compute_metrics(btc_w, alt_w, robust=robust)
        # Use end timestamp if available
        ts = btc_series.index[end - 1] if hasattr(btc_series.index, "__getitem__") else end - 1
        rows.append({"timestamp": ts, "r": m["r"], "beta": m["beta"], "r_squared": m["r_squared"], "p_value": m["p_value"], "n": m["n"]})

    return pd.DataFrame(rows)


def compute_all_coins_metrics(
    btc_returns: pd.Series,
    alt_returns_map: Dict[str, Tuple[pd.Series, pd.Series]],
    windows: List[int],
    robust: bool = False,
    min_data_points: int = 30,
    max_workers: int = 8,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute metrics for all coins in parallel.

    Args:
        btc_returns: not used directly — alt_returns_map already contains aligned series.
                     Kept for API symmetry; if alt_returns_map values are (btc_series, alt_series) tuples, uses those.
        alt_returns_map: Dict symbol -> (btc_series, alt_series) aligned log-returns.
        windows: window sizes.
        robust: robust regression flag.
        min_data_points: minimum n per window.
        max_workers: thread pool size.

    Returns:
        Dict symbol -> Dict window -> metrics.

    Units:
        All returns are log-returns dimensionless.
    """
    # alt_returns_map is Dict[str, Tuple[pd.Series, pd.Series]] where each tuple is (btc_series, alt_series)
    # For backwards compat, also support Dict[str, pd.Series] where btc_series is passed separately
    result: Dict[str, Dict[int, Dict[str, float]]] = {}

    def _task(symbol: str, btc_s: pd.Series, alt_s: pd.Series) -> Tuple[str, Dict[int, Dict[str, float]]]:
        metrics_by_window = compute_metrics_for_windows(btc_s, alt_s, windows, robust=robust, min_data_points=min_data_points)
        return symbol, metrics_by_window

    # Prepare tasks
    tasks = []
    for symbol, val in alt_returns_map.items():
        if isinstance(val, tuple) and len(val) == 2:
            btc_s, alt_s = val
        else:
            # Fallback: val is alt_series, use btc_returns
            btc_s, alt_s = btc_returns, val  # type: ignore
        tasks.append((symbol, btc_s, alt_s))

    if not tasks:
        logger.warning("compute_all_coins_metrics: no coins to process")
        return result

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(_task, sym, btc_s, alt_s): sym for sym, btc_s, alt_s in tasks}
        for fut in as_completed(future_to_sym):
            sym = future_to_sym[fut]
            try:
                symbol, metrics_by_window = fut.result()
                result[symbol] = metrics_by_window
            except Exception as exc:  # noqa: BLE001
                logger.error("Metrics failed for %s: %s — returning NaN metrics for all windows", sym, exc)
                result[sym] = {w: {"r": float("nan"), "beta": float("nan"), "r_squared": float("nan"), "p_value": float("nan"), "n": 0, "var_btc": float("nan"), "cov": float("nan")} for w in windows}

    logger.info("Computed metrics for %d coins across windows %s", len(result), windows)
    return result
