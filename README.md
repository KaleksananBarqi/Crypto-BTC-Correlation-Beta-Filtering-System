# Crypto BTC-Correlation & Beta Filtering System

Filters altcoins by **BTC correlation** and **beta** to identify:
- **Category A — Levered BTC play**: strong positive correlation + beta 3–5× + statistically significant
- **Category B — Uncorrelated**: |r| < threshold (and optionally p ≥ alpha if `category_b_require_nonsignificant=true`)
- **Neutral** — everything else (including negative beta, still shown)

All thresholds are in [`config.yaml`](config.yaml:1) — no hardcoding in logic.

---

## Methodology

1. **Ingest** OHLCV for BTC + universe (top 100 by market cap/volume) via Binance `ccxt` (public, no API key) with CoinGecko fallback for universe discovery.
2. **Align** timestamps (explicit missing-data policy: `drop` or `forward_fill` with limit — logged, never silent).
3. **Compute log-returns**: `R_t = ln(P_t / P_{t-1})` (not simple returns).
4. **Metrics per window** (30d / 90d / 180d, daily, rolling, parallel):
   - `r` = Pearson correlation of log-returns BTC vs alt, in [-1, 1] (`scipy.stats.pearsonr`)
   - `β` = `Cov(R_alt,R_btc) / Var(R_btc)` = OLS slope, unbounded (`statsmodels` OLS; optional Huber RLM)
   - `R²` = coefficient of determination [0, 1] (with `r_squared_raw` before clamp + `regression_method` for diagnostics)
   - `p-value` = significance of `r`
5. **Stability test**: compare classification across windows; `unstable=True` if category changes (e.g. correlated 30d but neutral 180d).
6. **Primary classification** = most recent window (`primary_window` in config); other windows are stability context.
7. **Outlier handling**: `none` | `winsorize` (clip tails) | `robust` (Huber regression) — configurable.
8. **No look-ahead bias**: each rolling window only uses data up to time `t`.

### Classification (exact definitions)

```
Category A = (r > CORR_THRESHOLD_HIGH) AND (BETA_MIN ≤ β ≤ BETA_MAX) AND (p < ALPHA)
Category B = (|r| < CORR_THRESHOLD_LOW) [AND (p ≥ ALPHA) if category_b_require_nonsignificant=true]
Else       = neutral / unclassified (still shown)
```

- **Category B flag**: `thresholds.category_b_require_nonsignificant` (default `false`). If `false`, Category B only checks `|r| < low` (decoupled from p-value). If `true`, requires both `|r| < low` AND `p ≥ alpha` (strict, old behaviour).
- **Beta negative**: NOT Category A (Category A requires positive 3.0–5.0×). Computed and shown as neutral — explicit decision per spec.
- **Small-sample noise**: `p-value` required; `n ≥ 30` per window enforced (configurable `min_data_points`).

---

## Project Structure

```
/data                  # cache raw OHLCV (gitignored)
/src
  data_fetch.py        # abstraction: Binance ccxt + CoinGecko fallback, universe + liquidity filter
  preprocessing.py     # alignment, log-return, missing-data handling
  metrics.py           # r, beta, R², p_value (scipy + statsmodels, parallel)
  classifier.py        # categorization + stability check
  config_schema.py     # config validation (no silent defaults)
  visualize.py         # scatter beta (x) vs correlation (y), color per category
/output
  classified_coins.csv # symbol, r_30d, r_90d, r_180d, beta_30d, beta_90d, beta_180d, r_squared, p_value, category, stability_flag
  scatter_beta_corr.png
  run_metadata.json    # run_utc, config_sha256, windows, universe_size
  run.log              # file logging if enabled
/tests
  test_metrics.py      # synthetic series (r=1,0,-1), look-ahead bias, no silent-fail
  test_data_fetch.py   # universe filter, OHLCV fallback, cache handling (mocked)
  test_config_contract.py # threshold contract, windows, primary_window
  conftest.py          # shared fixtures
config.yaml            # ALL thresholds & windows
main.py                # orchestrates pipeline end-to-end (CLI: --config --seed --offline --synthetic --end-date)
requirements.txt       # runtime only
requirements-dev.txt   # dev tools
pyproject.toml         # build, pytest, ruff, mypy
```

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
# CLI options:
python main.py --config config.yaml --seed 42 --offline --synthetic --end-date 2024-12-31
```

- On first run with internet: fetches live data via Binance + CoinGecko, caches to `data/`.
- Offline / API unavailable: automatically falls back to cached `data/*.csv`; if no cache, generates synthetic demo data (12 coins with known behaviours) so the pipeline always produces output.
- Outputs: [`output/classified_coins.csv`](output/classified_coins.csv), [`output/scatter_beta_corr.png`](output/scatter_beta_corr.png), [`output/run_metadata.json`](output/run_metadata.json).

### Run Tests

```bash
pytest tests/ -v
# with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
# lint & type check
ruff check src/ tests/
mypy src/
```

Tests cover:
- Synthetic series with known `r` (1, 0, -1) and known beta
- `R²`, `p-value` verification (including `r_squared_raw` + `regression_method`)
- Look-ahead bias (rolling window only uses data up to `t`; future mutation must not affect past windows)
- No silent-fail on missing data / NaN / empty series
- Edge cases: zero variance, small sample, negative beta, winsorize, alignment policies
- Config contract (thresholds, windows, primary_window, ranges)
- Data fetch (universe filter, OHLCV fallback, cache)

---

## Configuration

All thresholds in [`config.yaml`](config.yaml:1):

| Key | Default | Meaning |
|-----|---------|---------|
| `thresholds.corr_threshold_high` | 0.7 | Category A: `r >` this |
| `thresholds.corr_threshold_low` | 0.2 | Category B: `|r| <` this |
| `thresholds.beta_min` / `beta_max` | 3.0 / 5.0 | Category A beta range |
| `thresholds.alpha` | 0.05 | Significance level |
| `thresholds.category_b_require_nonsignificant` | false | If true, Category B requires `|r|<low` AND `p>=alpha`; if false, only `|r|<low` |
| `windows` | [30, 90, 180] | Rolling windows (days) |
| `primary_window` | 180 | Window used for primary classification |
| `universe.top_n` | 100 | Universe size |
| `universe.liquidity.min_24h_volume_usd` | 1_000_000 | Liquidity filter |
| `preprocessing.missing_data_policy` | drop | `drop` or `forward_fill` |
| `preprocessing.outlier_handling` | none | `none` / `winsorize` / `robust` |
| `performance.max_workers` | 8 | Thread/process pool size |
| `performance.executor` | thread | `thread` or `process` |
| `logging.file` | output/run.log | File handler (empty to disable) |

No threshold is hardcoded in `src/` — all read from config. Missing required keys raise `ValueError` (no silent `.get` defaults).

---

## Outputs

### `output/classified_coins.csv`

Columns: `symbol, r_30d, r_90d, r_180d, beta_30d, beta_90d, beta_180d, r_squared, p_value, category, stability_flag, stability_detail, cat_30d, cat_90d, cat_180d`

- `r_squared` and `p_value` are from `primary_window`.
- `stability_flag` = `True` if category differs across windows.

### `output/scatter_beta_corr.png`

Scatter: beta (x) vs correlation (y), color per category, with threshold reference lines and shaded Category A region. Category A points are annotated with symbol.

### `output/run_metadata.json`

`run_utc`, `config_sha256`, `windows`, `universe_size`, `category_counts`, `seed`, `executor`.

---

## Limitations & Disclosures

### Non-stationarity (expected)

Correlation and beta are **non-stationary** — they change over time with market regimes. A coin that is Category A in the 30d window may be neutral in the 180d window. This is not a bug; it is flagged via `stability_flag` and `stability_detail`. Always compare across windows and re-estimate periodically. The README and CSV explicitly surface this.

### Survivorship Bias

The universe is built from the **current** top-N by market cap. Coins that were delisted, rugged, or fell out of the top-N are not in the sample. Backtests on this universe overstate performance. To mitigate, document the universe snapshot date and consider point-in-time historical constituents if available.

### Small-Sample Noise

Short windows (e.g. 30d) have high estimation variance. The pipeline enforces `min_data_points ≥ 30` and requires `p < ALPHA` for Category A, but users should still treat short-window classifications as noisy.

### Look-ahead Bias

The pipeline uses **rolling windows that only include data up to time `t`** — no future data. Tests explicitly verify this by mutating future data and asserting past windows are unchanged.

### Other

- **Beta negative**: shown as neutral, not Category A (levered BTC play is defined as positive 3–5×).
- **Flash crashes / hacks / delistings**: detected via outlier handling (`winsorize` or `robust` regression) — configurable, not silent.
- **Missing data / API errors**: never silent — logged at `WARNING`/`ERROR` with explicit counts; empty DataFrames propagate and are skipped with a log line.
- **Log-returns vs simple returns**: all calculations use log-returns; units are dimensionless (not percent).

---

## Data Sources

- **Binance** via `ccxt` — public OHLCV (`fetch_ohlcv`), no API key, rate-limited.
- **CoinGecko** — ` /coins/markets` for universe (market cap / volume) and `/coins/{id}/market_chart` as OHLCV fallback.
- Abstraction: [`src/data_fetch.py`](src/data_fetch.py:1) exposes `DataFetcher` so the source can be swapped without changing downstream logic.

---

## License

MIT — see [LICENSE](LICENSE).
