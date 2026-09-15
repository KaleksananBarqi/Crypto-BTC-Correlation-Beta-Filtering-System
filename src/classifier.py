"""
Classifier: categorization + stability check.

Definitions (EXACT per spec, thresholds from config.yaml):
  Category A = (r > CORR_THRESHOLD_HIGH) AND (BETA_MIN ≤ β ≤ BETA_MAX) AND (p < ALPHA)
  Category B = (|r| < CORR_THRESHOLD_LOW) [AND (p ≥ ALPHA) if category_b_require_nonsignificant=true]
  Else       = neutral/unclassified (still shown)

Beta negative: NOT Category A (Category A requires positive 3.0-5.0x). Computed and shown as neutral.
Stability: compare classification across windows; unstable=True if any window differs.
Primary classification = most recent window (max(windows)), others as stability context.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Category labels
CATEGORY_A = "A_levered_btc"
CATEGORY_B = "B_uncorrelated"
CATEGORY_NEUTRAL = "neutral"


def classify_single(
    r: float,
    beta: float,
    p_value: float,
    corr_threshold_high: float,
    corr_threshold_low: float,
    beta_min: float,
    beta_max: float,
    alpha: float,
    category_b_require_nonsignificant: bool = False,
) -> str:
    """
    Classify a single window's metrics.

    Args:
        r: Pearson correlation [-1, 1] of log-returns (dimensionless).
        beta: OLS slope Cov(R_alt,R_btc)/Var(R_btc), unbounded (dimensionless).
        p_value: significance of r in [0, 1].
        corr_threshold_high: threshold for Category A (e.g. 0.7).
        corr_threshold_low: threshold for Category B (e.g. 0.2).
        beta_min: lower bound for Category A beta (e.g. 3.0).
        beta_max: upper bound for Category A beta (e.g. 5.0).
        alpha: significance level (e.g. 0.05).
        category_b_require_nonsignificant: if True, Category B requires |r|<low AND p>=alpha;
            if False (default), Category B only checks |r|<low (decoupled).

    Returns:
        Category string: "A_levered_btc" | "B_uncorrelated" | "neutral".

    Units:
        r dimensionless [-1,1], beta dimensionless unbounded, p_value dimensionless [0,1].

    Notes:
        - NaN inputs -> neutral (explicit, logged).
        - Beta negative never qualifies for Category A (requires beta in [3,5]).
    """
    # NaN guard — no silent fail, explicit neutral
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (r, beta, p_value)):
        logger.debug("classify_single: NaN input r=%s beta=%s p=%s -> neutral", r, beta, p_value)
        return CATEGORY_NEUTRAL

    # Category A: strong positive correlation + levered beta + significant
    if (r > corr_threshold_high) and (beta_min <= beta <= beta_max) and (p_value < alpha):
        return CATEGORY_A

    # Category B: uncorrelated (|r| < low) [+ not significant if flag true]
    if abs(r) < corr_threshold_low:
        if category_b_require_nonsignificant:
            if p_value >= alpha:
                return CATEGORY_B
        else:
            return CATEGORY_B

    return CATEGORY_NEUTRAL


def classify_all_windows(
    metrics_by_window: dict[int, dict[str, float]],
    thresholds: dict[str, float],
) -> dict[int, str]:
    """
    Classify each window's metrics.

    Args:
        metrics_by_window: Dict window -> metrics dict with keys r, beta, p_value.
        thresholds: dict with keys corr_threshold_high, corr_threshold_low, beta_min, beta_max, alpha,
            and optionally category_b_require_nonsignificant (bool, default False).

    Returns:
        Dict window -> category string.

    Units:
        Thresholds dimensionless; metrics as per classify_single.
    """
    result: dict[int, str] = {}
    require_nonsig = bool(thresholds.get("category_b_require_nonsignificant", False))
    for w, m in metrics_by_window.items():
        cat = classify_single(
            r=m.get("r", float("nan")),
            beta=m.get("beta", float("nan")),
            p_value=m.get("p_value", float("nan")),
            corr_threshold_high=thresholds["corr_threshold_high"],
            corr_threshold_low=thresholds["corr_threshold_low"],
            beta_min=thresholds["beta_min"],
            beta_max=thresholds["beta_max"],
            alpha=thresholds["alpha"],
            category_b_require_nonsignificant=require_nonsig,
        )
        result[w] = cat
    return result


def check_stability(categories_by_window: dict[int, str]) -> tuple[bool, str]:
    """
    Check stability across windows.

    Args:
        categories_by_window: Dict window -> category.

    Returns:
        (unstable, detail) where unstable=True if classifications differ across windows.
        detail is a human-readable string e.g. "30d:A 90d:neutral 180d:neutral -> unstable".

    Units:
        Categories are strings; no numeric units.
    """
    if not categories_by_window:
        return False, "no windows"
    unique_cats = set(categories_by_window.values())
    unstable = len(unique_cats) > 1
    # Build detail string sorted by window
    parts = [f"{w}d:{categories_by_window[w]}" for w in sorted(categories_by_window)]
    detail = " ".join(parts) + (" -> unstable" if unstable else " -> stable")
    if unstable:
        logger.info("Stability check: %s (non-stationarity expected)", detail)
    return unstable, detail


def build_classification_table(
    all_metrics: dict[str, dict[int, dict[str, float]]],
    config: dict,
) -> pd.DataFrame:
    """
    Build the final classification DataFrame for output CSV.

    Args:
        all_metrics: Dict symbol -> Dict window -> metrics dict.
        config: parsed config.yaml dict (for thresholds, windows, primary_window).

    Returns:
        DataFrame with columns:
          symbol, r_30d, r_90d, r_180d, beta_30d, beta_90d, beta_180d,
          r_squared, p_value, category, stability_flag, stability_detail
        (r_squared and p_value are from primary_window).

    Units:
        r in [-1,1], beta unbounded, r_squared in [0,1], p_value in [0,1],
        all derived from log-returns (dimensionless).
    """
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("config missing 'thresholds' mapping — no silent defaults allowed")
    # Strict contract: no silent .get defaults for required keys
    required_keys = ["corr_threshold_high", "corr_threshold_low", "beta_min", "beta_max", "alpha"]
    missing = [k for k in required_keys if k not in thresholds]
    if missing:
        raise ValueError(f"config thresholds missing required keys {missing} — no silent defaults allowed")
    thresh: dict[str, object] = {
        "corr_threshold_high": thresholds["corr_threshold_high"],
        "corr_threshold_low": thresholds["corr_threshold_low"],
        "beta_min": thresholds["beta_min"],
        "beta_max": thresholds["beta_max"],
        "alpha": thresholds["alpha"],
        "category_b_require_nonsignificant": bool(thresholds.get("category_b_require_nonsignificant", False)),
    }
    windows: list[int] = config.get("windows", [30, 90, 180])
    if not windows:
        raise ValueError("config 'windows' must be non-empty")
    primary_window: int = config.get("primary_window", max(windows) if windows else 180)
    if primary_window not in windows:
        logger.warning("primary_window %d not in windows %s — using max(windows) %d", primary_window, windows, max(windows))
        primary_window = max(windows)

    rows = []
    for symbol, metrics_by_window in all_metrics.items():
        # Classify each window
        cats_by_window = classify_all_windows(metrics_by_window, thresh)  # type: ignore[arg-type]
        primary_cat = cats_by_window.get(primary_window, CATEGORY_NEUTRAL)
        unstable, detail = check_stability(cats_by_window)

        # Primary window metrics for r_squared and p_value
        primary_metrics = metrics_by_window.get(primary_window, {})
        r_squared = primary_metrics.get("r_squared", float("nan"))
        p_value = primary_metrics.get("p_value", float("nan"))

        row: dict[str, object] = {"symbol": symbol}
        for w in windows:
            m = metrics_by_window.get(w, {})
            row[f"r_{w}d"] = m.get("r", float("nan"))
            row[f"beta_{w}d"] = m.get("beta", float("nan"))
        row["r_squared"] = r_squared
        row["p_value"] = p_value
        row["category"] = primary_cat
        row["stability_flag"] = unstable
        row["stability_detail"] = detail
        # Also include per-window categories for debugging
        for w in windows:
            row[f"cat_{w}d"] = cats_by_window.get(w, CATEGORY_NEUTRAL)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort: Category A first, then neutral, then B (or alphabetical)
    cat_order = {CATEGORY_A: 0, CATEGORY_NEUTRAL: 1, CATEGORY_B: 2}
    if not df.empty:
        df["_sort"] = df["category"].map(cat_order).fillna(99)
        df = df.sort_values(["_sort", "symbol"]).drop(columns=["_sort"]).reset_index(drop=True)

    logger.info("Classification table: %d coins -> A=%d B=%d neutral=%d unstable=%d",
                len(df),
                (df["category"] == CATEGORY_A).sum() if not df.empty else 0,
                (df["category"] == CATEGORY_B).sum() if not df.empty else 0,
                (df["category"] == CATEGORY_NEUTRAL).sum() if not df.empty else 0,
                df["stability_flag"].sum() if not df.empty else 0)
    return df
