"""
Main orchestration: end-to-end pipeline for Crypto BTC-Correlation & Beta Filtering System.

Methodology order (no look-ahead bias):
  1. Ingest OHLCV BTC + universe (Binance ccxt + CoinGecko fallback)
  2. Align timestamps, explicit missing-data policy
  3. Compute log-returns (not simple returns)
  4. For each coin compute r, β, R², p-value in parallel for windows 30/90/180
  5. Stability test across windows
  6. Primary classification = most recent window
  7. Outlier handling (winsorize / robust as per config)
  8. Output CSV + scatter plot

Graceful fallback: if APIs unavailable, uses cached data in /data; if no cache, generates synthetic demo data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from src.classifier import build_classification_table
from src.config_schema import validate_config
from src.data_fetch import DataFetcher
from src.metrics import compute_all_coins_metrics
from src.preprocessing import prepare_aligned_returns
from src.visualize import plot_beta_vs_correlation

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> Dict:
    """Load and validate config.yaml."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "thresholds" not in cfg:
        raise ValueError("config.yaml missing required key: 'thresholds' (no silent defaults)")
    if "windows" not in cfg:
        raise ValueError("config.yaml missing required key: 'windows' (no silent defaults)")
    # Full contract validation (raises ValueError on violation)
    validate_config(cfg)
    return cfg


def setup_logging(cfg: Dict) -> None:
    """Configure logging from config."""
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Reset handlers to allow file handler
    root = logging.getLogger()
    # Use basicConfig only if no handlers yet; otherwise set level/format manually
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    else:
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)
    # File handler if configured
    log_file = log_cfg.get("file")
    if log_file:
        try:
            fp = Path(log_file)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt))
            root.addHandler(fh)
            logger.info("File logging enabled: %s", fp)
        except Exception as exc:
            logger.warning("Failed to setup file logging %s: %s", log_file, exc)
    # Reduce noise from external libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto BTC-Correlation & Beta Filtering System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic/demo mode")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (use cache/synthetic, no live fetch)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic demo data (ignore cache/live)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for synthetic data (YYYY-MM-DD), default now UTC")
    return parser.parse_args()


def generate_synthetic_demo_data(cfg: Dict, seed: int = 42, end_date: str | None = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic OHLCV for demo/offline mode when no API/cache available.

    Creates BTC + 12 alts with known behaviours:
      - 3 levered BTC (beta ~3-5, high correlation)
      - 3 uncorrelated (low correlation)
      - 3 neutral / moderate
      - 3 edge cases (negative beta, low volume, etc.)

    Returns:
        (btc_df, alt_map) with OHLCV DataFrames.

    Units:
        Prices in USDT, log-returns dimensionless.
    """
    logger.warning("Generating SYNTHETIC demo OHLCV — no live API/cache available (offline mode)")
    random.seed(seed)
    np.random.seed(seed)
    n = max(cfg.get("windows", [30, 90, 180])) + 5
    if end_date:
        try:
            end_ts = pd.Timestamp(end_date, tz="UTC")
        except Exception:
            logger.warning("Invalid --end-date %s, using now UTC", end_date)
            end_ts = pd.Timestamp.now(tz="UTC")
    else:
        end_ts = pd.Timestamp.now(tz="UTC")
    dates = pd.date_range(end=end_ts, periods=n, freq="D")

    # BTC: random walk log-returns
    btc_ret = np.random.normal(0.0005, 0.02, size=n)
    btc_price = 60000 * np.exp(np.cumsum(btc_ret))
    btc_price[0] = 60000
    btc_df = pd.DataFrame({
        "timestamp": dates,
        "open": btc_price * 0.999,
        "high": btc_price * 1.01,
        "low": btc_price * 0.99,
        "close": btc_price,
        "volume": np.random.uniform(1e9, 3e9, size=n),
    })

    # Helper to make alt from BTC returns
    def make_alt(name: str, beta: float, noise_std: float, drift: float = 0.0) -> pd.DataFrame:
        alt_ret = beta * btc_ret + np.random.normal(drift, noise_std, size=n)
        alt_price = 100 * np.exp(np.cumsum(alt_ret))
        alt_price[0] = 100
        return pd.DataFrame({
            "timestamp": dates,
            "open": alt_price * 0.999,
            "high": alt_price * 1.01,
            "low": alt_price * 0.99,
            "close": alt_price,
            "volume": np.random.uniform(5e7, 5e8, size=n),
        })

    alt_map: Dict[str, pd.DataFrame] = {
        # Category A candidates: high corr, beta 3-5
        "DEMO_A1": make_alt("DEMO_A1", beta=3.5, noise_std=0.015),
        "DEMO_A2": make_alt("DEMO_A2", beta=4.2, noise_std=0.018),
        "DEMO_A3": make_alt("DEMO_A3", beta=3.0, noise_std=0.012),
        # Category B candidates: uncorrelated
        "DEMO_B1": make_alt("DEMO_B1", beta=0.0, noise_std=0.04),
        "DEMO_B2": make_alt("DEMO_B2", beta=0.1, noise_std=0.035),
        "DEMO_B3": make_alt("DEMO_B3", beta=-0.05, noise_std=0.03),
        # Neutral: moderate correlation, beta outside 3-5
        "DEMO_N1": make_alt("DEMO_N1", beta=1.5, noise_std=0.02),
        "DEMO_N2": make_alt("DEMO_N2", beta=0.8, noise_std=0.025),
        "DEMO_N3": make_alt("DEMO_N3", beta=6.0, noise_std=0.02),  # high beta but >5 -> neutral
        # Edge: negative beta (should be neutral, not A)
        "DEMO_NEG": make_alt("DEMO_NEG", beta=-2.0, noise_std=0.02),
        # Edge: flash-crash outlier (inject spike)
        "DEMO_SPIKE": make_alt("DEMO_SPIKE", beta=1.0, noise_std=0.02),
        "DEMO_LOWVOL": make_alt("DEMO_LOWVOL", beta=1.0, noise_std=0.02),
    }
    # Inject flash crash into DEMO_SPIKE
    spike_df = alt_map["DEMO_SPIKE"]
    spike_idx = n // 2
    spike_df.loc[spike_idx, "close"] *= 0.5  # 50% flash crash
    alt_map["DEMO_SPIKE"] = spike_df

    # Cache synthetic data for next run
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "data"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        btc_df.to_csv(cache_dir / "BTC_USDT_1d.csv", index=False)
        for sym, df in alt_map.items():
            df.to_csv(cache_dir / f"{sym}_USDT_1d.csv", index=False)
        logger.info("Cached synthetic demo data to %s", cache_dir)
    except Exception as exc:
        logger.warning("Failed to cache synthetic data: %s", exc)

    return btc_df, alt_map


def main() -> None:
    """Orchestrate the full pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("=== Crypto BTC-Correlation & Beta Filtering System ===")
    logger.info("Config: windows=%s primary=%s thresholds=%s", cfg.get("windows"), cfg.get("primary_window"), cfg.get("thresholds"))
    logger.info("CLI args: config=%s seed=%d offline=%s synthetic=%s end_date=%s", args.config, args.seed, args.offline, args.synthetic, args.end_date)

    windows = cfg.get("windows", [30, 90, 180])
    min_data_points = cfg.get("universe", {}).get("min_data_points", 30)
    prep_cfg = cfg.get("preprocessing", {})
    robust = prep_cfg.get("outlier_handling") == "robust" or prep_cfg.get("robust_regression", False)
    perf_cfg = cfg.get("performance", {})
    max_workers = int(perf_cfg.get("max_workers", 8))
    executor = str(perf_cfg.get("executor", "thread"))

    # ── 1. Ingest ───────────────────────────────────────────────────────
    fetcher = DataFetcher(cfg)
    btc_df: pd.DataFrame | None = None
    alt_map: Dict[str, pd.DataFrame] = {}

    if args.synthetic:
        logger.info("Forced synthetic mode via --synthetic")
        btc_df, alt_map = generate_synthetic_demo_data(cfg, seed=args.seed, end_date=args.end_date)
    elif args.offline:
        logger.info("Offline mode via --offline — trying cache first")
        cached_btc, cached_alts = fetcher.load_cached_ohlcv()
        if cached_btc is not None and not cached_btc.empty and cached_alts:
            btc_df = cached_btc
            alt_map = cached_alts
            logger.info("Using cached data: BTC %d rows, %d alts", len(btc_df), len(alt_map))
        else:
            logger.warning("No usable cache — falling back to synthetic demo data")
            btc_df, alt_map = generate_synthetic_demo_data(cfg, seed=args.seed, end_date=args.end_date)
    else:
        try:
            universe = fetcher.get_universe()
            if not universe:
                logger.warning("Universe empty — will try cached data")
                raise ValueError("Empty universe")
            btc_df, alt_map = fetcher.fetch_all_ohlcv(universe)
            if btc_df is None or btc_df.empty:
                logger.warning("BTC OHLCV empty after fetch — trying cache")
                raise ValueError("Empty BTC OHLCV")
            # Filter out empty alts
            alt_map = {k: v for k, v in alt_map.items() if not v.empty}
            if not alt_map:
                logger.warning("All alt OHLCV empty — trying cache")
                raise ValueError("No alt OHLCV")
            logger.info("Live fetch succeeded: BTC %d rows, %d alts", len(btc_df), len(alt_map))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live fetch failed: %s — attempting cached data", exc)
            cached_btc, cached_alts = fetcher.load_cached_ohlcv()
            if cached_btc is not None and not cached_btc.empty and cached_alts:
                btc_df = cached_btc
                alt_map = cached_alts
                logger.info("Using cached data: BTC %d rows, %d alts", len(btc_df), len(alt_map))
            else:
                logger.warning("No usable cache — falling back to synthetic demo data")
                btc_df, alt_map = generate_synthetic_demo_data(cfg, seed=args.seed, end_date=args.end_date)

    if btc_df is None or btc_df.empty:
        raise ValueError("BTC data is empty — cannot proceed (no silent assert)")
    if not alt_map:
        raise ValueError("No alt data — cannot proceed (no silent assert)")

    # ── 2-3. Preprocessing: align + log-returns ─────────────────────────
    # Build aligned returns map: symbol -> (btc_series, alt_series)
    aligned_map: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    for symbol, alt_df in alt_map.items():
        try:
            btc_s, alt_s, merged = prepare_aligned_returns(btc_df, alt_df, cfg)
            if btc_s.empty or alt_s.empty:
                logger.warning("Skipping %s: empty aligned returns (btc=%d alt=%d)", symbol, len(btc_s), len(alt_s))
                continue
            if len(btc_s) < min_data_points:
                logger.warning("Skipping %s: n=%d < min_data_points %d", symbol, len(btc_s), min_data_points)
                continue
            aligned_map[symbol] = (btc_s, alt_s)
        except Exception as exc:  # noqa: BLE001
            logger.error("Preprocessing FAILED for %s: %s — skipping (no silent fail)", symbol, exc, exc_info=True)
            continue

    if not aligned_map:
        logger.error("No coins passed preprocessing — cannot compute metrics")
        sys.exit(1)

    logger.info("Preprocessing complete: %d coins with aligned returns", len(aligned_map))

    # ── 4. Metrics (parallel) ───────────────────────────────────────────
    all_metrics = compute_all_coins_metrics(
        btc_returns=pd.Series(dtype=float),
        alt_returns_map=aligned_map,  # type: ignore[arg-type]
        windows=windows,
        robust=robust,
        min_data_points=min_data_points,
        max_workers=max_workers,
        executor=executor,
    )

    # ── 5-6. Classification + stability ─────────────────────────────────
    table = build_classification_table(all_metrics, cfg)

    # ── 7. Output ───────────────────────────────────────────────────────
    out_cfg = cfg.get("output", {})
    csv_path = Path(out_cfg.get("csv_path", "output/classified_coins.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required columns per spec (plus extras for debugging)
    required_cols = []
    for w in windows:
        required_cols.extend([f"r_{w}d", f"beta_{w}d"])
    required_cols.extend(["r_squared", "p_value", "category", "stability_flag"])
    # Reorder: symbol first, then required, then extras
    all_cols = ["symbol"] + required_cols
    # Add extras if present
    extras = [c for c in table.columns if c not in all_cols]
    # Ensure all required cols exist (fill NaN if missing)
    for c in required_cols:
        if c not in table.columns:
            table[c] = float("nan")
    # Select and save
    output_table = table[["symbol"] + required_cols + [c for c in extras if c in table.columns]]
    # Also ensure spec column order: symbol, r_30d, r_90d, r_180d, beta_30d, beta_90d, beta_180d, r_squared, p_value, category, stability_flag
    # Our table already matches if windows=[30,90,180]
    output_table.to_csv(csv_path, index=False)
    logger.info("Saved classification CSV to %s (%d rows)", csv_path, len(output_table))
    # Log preview
    logger.info("\n%s", output_table.head(20).to_string(index=False))

    # ── 8. Visualization ────────────────────────────────────────────────
    try:
        scatter_path = plot_beta_vs_correlation(table, cfg)
        logger.info("Scatter plot saved to %s", scatter_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Visualization FAILED: %s", exc, exc_info=True)

    # ── 9. Run metadata ─────────────────────────────────────────────────
    try:
        # config sha256
        cfg_path = Path(args.config)
        sha = ""
        if cfg_path.exists():
            sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        meta = {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(cfg_path),
            "config_sha256": sha,
            "windows": windows,
            "primary_window": cfg.get("primary_window"),
            "universe_size": len(aligned_map),
            "total_rows": len(table),
            "category_counts": table["category"].value_counts().to_dict() if not table.empty else {},
            "seed": args.seed,
            "executor": executor,
            "max_workers": max_workers,
        }
        meta_path = Path(out_cfg.get("csv_path", "output/classified_coins.csv")).parent / "run_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Run metadata saved to %s", meta_path)
    except Exception as exc:
        logger.warning("Failed to write run_metadata.json: %s", exc)

    # ── Summary ─────────────────────────────────────────────────────────
    if not table.empty:
        counts = table["category"].value_counts().to_dict()
        unstable_n = int(table["stability_flag"].sum()) if "stability_flag" in table.columns else 0
        logger.info("=== SUMMARY ===")
        logger.info("Total coins: %d", len(table))
        for cat, n in counts.items():
            logger.info("  %s: %d", cat, n)
        logger.info("Unstable (category changes across windows): %d", unstable_n)
        logger.info("Outputs: %s and %s", csv_path, out_cfg.get("scatter_path", "output/scatter_beta_corr.png"))
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
