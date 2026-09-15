"""
Visualization: scatter beta (x) vs correlation (y), color per category.

Uses matplotlib (no plotly dependency required). All styling from config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

# Category colors (colorblind-friendly)
COLORS = {
    "A_levered_btc": "#d62728",   # red — levered BTC play
    "B_uncorrelated": "#2ca02c",  # green — uncorrelated
    "neutral": "#7f7f7f",         # grey — neutral
}
LABELS = {
    "A_levered_btc": "A: Levered BTC (r>high, β 3-5, p<α)",
    "B_uncorrelated": "B: Uncorrelated (|r|<low, p≥α)",
    "neutral": "Neutral / Unclassified",
}


def plot_beta_vs_correlation(
    df: pd.DataFrame,
    config: Dict,
    output_path: Optional[str] = None,
    r_col: Optional[str] = None,
    beta_col: Optional[str] = None,
) -> Path:
    """
    Create scatter plot: beta (x-axis) vs correlation (y-axis).

    Args:
        df: classification DataFrame with at least columns [symbol, category, r_*, beta_*].
        config: parsed config.yaml dict (for thresholds, windows, output paths).
        output_path: override output path (else from config output.scatter_path).
        r_col: override r column (else r_{primary_window}d).
        beta_col: override beta column (else beta_{primary_window}d).

    Returns:
        Path to saved PNG.

    Units:
        r in [-1, 1] (y-axis), beta unbounded (x-axis), both from log-returns.
    """
    thresholds = config.get("thresholds", {})
    corr_high = thresholds.get("corr_threshold_high", 0.7)
    corr_low = thresholds.get("corr_threshold_low", 0.2)
    beta_min = thresholds.get("beta_min", 3.0)
    beta_max = thresholds.get("beta_max", 5.0)

    windows: List[int] = config.get("windows", [30, 90, 180])
    primary_window: int = config.get("primary_window", max(windows) if windows else 180)
    out_cfg = config.get("output", {})
    scatter_path = Path(output_path or out_cfg.get("scatter_path", "output/scatter_beta_corr.png"))
    dpi = out_cfg.get("scatter_dpi", 150)
    figsize = tuple(out_cfg.get("scatter_figsize", [10, 7]))

    # Resolve columns
    if r_col is None:
        r_col = f"r_{primary_window}d"
    if beta_col is None:
        beta_col = f"beta_{primary_window}d"

    # Fallback if primary window columns missing
    if r_col not in df.columns or beta_col not in df.columns:
        # Try any r_*/beta_* column
        r_candidates = [c for c in df.columns if c.startswith("r_")]
        b_candidates = [c for c in df.columns if c.startswith("beta_")]
        if r_candidates and b_candidates:
            r_col = r_candidates[0]
            beta_col = b_candidates[0]
            logger.warning("Primary window columns missing — using %s vs %s", beta_col, r_col)
        else:
            raise ValueError(f"DataFrame missing r/beta columns: has {list(df.columns)}")

    scatter_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot per category
    for cat in ["A_levered_btc", "B_uncorrelated", "neutral"]:
        sub = df[df["category"] == cat] if "category" in df.columns else df
        if sub.empty:
            continue
        # Drop NaN for plotting
        plot_df = sub.dropna(subset=[beta_col, r_col])
        if plot_df.empty:
            logger.warning("No valid points for category %s (all NaN)", cat)
            continue
        ax.scatter(
            plot_df[beta_col],
            plot_df[r_col],
            c=COLORS.get(cat, "#7f7f7f"),
            label=f"{LABELS.get(cat, cat)} (n={len(plot_df)})",
            alpha=0.75,
            s=55,
            edgecolors="white",
            linewidths=0.6,
        )
        # Annotate Category A points with symbol
        if cat == "A_levered_btc":
            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row.get("symbol", "")),
                    (row[beta_col], row[r_col]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    alpha=0.9,
                )

    # Reference lines
    ax.axhline(y=corr_high, color="#d62728", linestyle="--", linewidth=1, alpha=0.6, label=f"r = {corr_high} (A threshold)")
    ax.axhline(y=corr_low, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(y=-corr_low, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.6, label=f"|r| = {corr_low} (B threshold)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.7, alpha=0.4)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.7, alpha=0.4)
    ax.axvline(x=beta_min, color="#d62728", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=beta_max, color="#d62728", linestyle="--", linewidth=1, alpha=0.5)
    # Shade Category A region
    ax.axhspan(corr_high, 1.0, xmin=0, xmax=1, alpha=0.04, color="#d62728")
    ax.axvspan(beta_min, beta_max, alpha=0.04, color="#d62728")

    ax.set_xlabel(f"Beta β  (Cov(R_alt,R_btc)/Var(R_btc), OLS slope)  —  window {primary_window}d", fontsize=10)
    ax.set_ylabel(f"Pearson r  (log-returns)  —  window {primary_window}d", fontsize=10)
    ax.set_title(f"BTC Correlation vs Beta — {len(df)} coins  (primary window {primary_window}d, log-returns)", fontsize=12, pad=12)
    ax.set_ylim(-1.05, 1.05)
    # Beta limits: auto with padding, but clamp to show A region
    if not df.empty and beta_col in df.columns:
        b_vals = df[beta_col].dropna()
        if not b_vals.empty:
            lo, hi = float(b_vals.min()), float(b_vals.max())
            pad = max(0.5, (hi - lo) * 0.1)
            ax.set_xlim(min(lo - pad, beta_min - 1), max(hi + pad, beta_max + 1))

    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # Footer with methodology note
    fig.text(
        0.01, 0.01,
        f"Method: log-returns (not simple), Pearson r + OLS beta, p-value from pearsonr.  "
        f"A: r>{corr_high} & β∈[{beta_min},{beta_max}] & p<α  |  B: |r|<{corr_low} & p≥α  |  Unstable if category changes across windows.",
        fontsize=6, color="#555", ha="left", va="bottom", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scatter saved to %s (%d points)", scatter_path, len(df))
    return scatter_path
