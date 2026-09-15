"""
Data fetching abstraction: Binance via ccxt (public OHLCV) + CoinGecko fallback.

Design: DataFetcher is the single abstraction so the source can be swapped
without changing downstream logic. All thresholds come from config.yaml.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """Abstract OHLCV data source."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV and return DataFrame with columns [timestamp, open, high, low, close, volume]."""

    @abstractmethod
    def fetch_universe(self, top_n: int, vs_currency: str = "usd") -> list[dict]:
        """Return list of coin dicts with at least keys: symbol, id, market_cap, volume_24h."""


# ---------------------------------------------------------------------------
# Binance via ccxt
# ---------------------------------------------------------------------------

class BinanceDataSource(DataSource):
    """Binance public OHLCV via ccxt (no API key required)."""

    def __init__(
        self,
        exchange_id: str = "binance",
        rate_limit_ms: int = 200,
        timeout_ms: int = 15000,
        max_retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialise Binance data source.

        Args:
            exchange_id: ccxt exchange id (e.g. "binance").
            rate_limit_ms: milliseconds to sleep between calls.
            timeout_ms: ccxt request timeout.
            max_retries: retries on transient failure.
            cache_dir: directory to cache raw OHLCV as CSV.
        """
        self.exchange_id = exchange_id
        self.rate_limit_ms = rate_limit_ms
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None  # lazy init so import doesn't fail if ccxt missing

    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ccxt is required for BinanceDataSource. Install with: pip install ccxt"
            ) from exc
        klass = getattr(ccxt, self.exchange_id)
        self._exchange = klass({
            "enableRateLimit": True,
            "timeout": self.timeout_ms,
        })
        return self._exchange

    # -- OHLCV ---------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a symbol (e.g. "BTC/USDT").

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume].
            timestamp is UTC datetime. Empty DataFrame on failure (logged, not silent).

        Units:
            Prices in quote currency (e.g. USDT), volume in base currency.
        """
        cache_path = None
        if self.cache_dir:
            safe = symbol.replace("/", "_")
            cache_path = self.cache_dir / f"{safe}_{interval}.csv"
            if cache_path.exists():
                try:
                    df = pd.read_csv(cache_path, parse_dates=["timestamp"])
                    # If cached data covers requested limit, return it
                    if len(df) >= limit * 0.8:  # allow slight staleness
                        logger.info("Cache hit for %s (%d rows) -> %s", symbol, len(df), cache_path)
                        return df.tail(limit).reset_index(drop=True)
                except Exception as exc:
                    logger.warning("Failed to read cache %s: %s — refetching", cache_path, exc)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ex = self._get_exchange()
                # ccxt expects milliseconds since epoch for `since` if we want history
                # For simplicity fetch most recent `limit` candles
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
                if not ohlcv:
                    raise ValueError(f"Empty OHLCV returned for {symbol}")
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                # Cache
                if cache_path is not None:
                    try:
                        df.to_csv(cache_path, index=False)
                        logger.info("Cached %s -> %s", symbol, cache_path)
                    except Exception as exc:
                        logger.warning("Failed to write cache %s: %s", cache_path, exc)
                time.sleep(self.rate_limit_ms / 1000.0)
                return df
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "fetch_ohlcv attempt %d/%d failed for %s: %s",
                    attempt, self.max_retries, symbol, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
                else:
                    logger.error("All retries exhausted for %s: %s", symbol, exc)

        # Explicit failure — caller must handle empty DataFrame
        logger.error("fetch_ohlcv FAILED for %s after %d retries: %s", symbol, self.max_retries, last_exc)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # -- Universe (Binance doesn't have market-cap ranking; delegate to CoinGecko) --

    def fetch_universe(self, top_n: int, vs_currency: str = "usd") -> list[dict]:
        """
        Binance has no market-cap ranking endpoint. This method raises and
        signals the caller to use CoinGecko fallback.

        Raises:
            NotImplementedError: always — caller should fallback.
        """
        raise NotImplementedError(
            "BinanceDataSource does not support universe ranking; use CoinGeckoDataSource."
        )


# ---------------------------------------------------------------------------
# CoinGecko fallback (universe + optional OHLCV fallback)
# ---------------------------------------------------------------------------

class CoinGeckoDataSource(DataSource):
    """CoinGecko public API for universe discovery and fallback OHLCV."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        vs_currency: str = "usd",
        per_page: int = 250,
        timeout_ms: int = 15000,
        max_retries: int = 3,
        cache_dir: Path | None = None,
    ) -> None:
        self.vs_currency = vs_currency
        self.per_page = per_page
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_universe(self, top_n: int, vs_currency: str = "usd") -> list[dict]:
        """
        Fetch top-N coins by market cap from CoinGecko.

        Returns:
            List of dicts with keys: id, symbol, name, market_cap, volume_24h.

        Raises:
            Logs and returns empty list on failure (no silent fail — caller checks).
        """
        vs = vs_currency or self.vs_currency
        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": vs,
            "order": "market_cap_desc",
            "per_page": min(self.per_page, top_n),
            "page": 1,
            "sparkline": "false",
        }
        # Paginate if top_n > per_page
        results: list[dict] = []
        pages = (top_n + self.per_page - 1) // self.per_page
        for page in range(1, pages + 1):
            params["page"] = page
            params["per_page"] = min(self.per_page, top_n - len(results))
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.get(url, params=params, timeout=self.timeout_ms / 1000)
                    if resp.status_code == 429:
                        logger.warning("CoinGecko rate-limited (429) page %d attempt %d — sleeping 10s", page, attempt)
                        time.sleep(10)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    for coin in data:
                        results.append({
                            "id": coin.get("id"),
                            "symbol": (coin.get("symbol") or "").upper(),
                            "name": coin.get("name"),
                            "market_cap": coin.get("market_cap"),
                            "volume_24h": coin.get("total_volume"),
                        })
                    logger.info("CoinGecko universe page %d: %d coins", page, len(data))
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("CoinGecko fetch_universe page %d attempt %d failed: %s", page, attempt, exc)
                    if attempt == self.max_retries:
                        logger.error("CoinGecko fetch_universe FAILED page %d: %s", page, exc)
                    else:
                        time.sleep(2 * attempt)
            if len(results) >= top_n:
                break
            time.sleep(1.2)  # be nice to public API
        return results[:top_n]

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fallback OHLCV via CoinGecko /coins/{id}/market_chart.

        Note:
            `symbol` here is expected to be a CoinGecko id (e.g. "bitcoin").
            For generic use, this is a best-effort fallback; Binance is preferred.
        """
        # Map interval to CoinGecko `days` param
        days_map = {"1d": limit, "1h": max(1, limit // 24)}
        days = days_map.get(interval, limit)
        url = f"{self.BASE_URL}/coins/{symbol.lower()}/market_chart"
        params = {"vs_currency": self.vs_currency, "days": days, "interval": "daily"}
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout_ms / 1000)
                if resp.status_code == 429:
                    logger.warning("CoinGecko rate-limited (429) for %s — sleeping 10s", symbol)
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])
                if not prices:
                    raise ValueError(f"No prices returned for {symbol}")
                df = pd.DataFrame(prices, columns=["timestamp", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                # CoinGecko market_chart doesn't give OHLC, synthesize
                df["open"] = df["close"]
                df["high"] = df["close"]
                df["low"] = df["close"]
                if volumes:
                    vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                    vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms", utc=True)
                    df = pd.merge_asof(df.sort_values("timestamp"), vol_df.sort_values("timestamp"), on="timestamp")
                else:
                    df["volume"] = 0.0
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                return df.tail(limit).reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CoinGecko fetch_ohlcv attempt %d/%d for %s failed: %s", attempt, self.max_retries, symbol, exc)
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    logger.error("CoinGecko fetch_ohlcv FAILED for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# High-level DataFetcher (abstraction that orchestrates sources + filters)
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    High-level fetcher that abstracts Binance + CoinGecko.

    Usage:
        fetcher = DataFetcher(config)
        universe = fetcher.get_universe()          # filtered symbols
        btc_df, alt_map = fetcher.fetch_all_ohlcv(universe)
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: parsed config.yaml dict.
        """
        self.config = config
        data_cfg = config.get("data", {})
        uni_cfg = config.get("universe", {})

        cache_dir = Path(data_cfg.get("cache_dir", "data"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.binance = BinanceDataSource(
            exchange_id=data_cfg.get("exchange", "binance"),
            rate_limit_ms=data_cfg.get("rate_limit_ms", 200),
            timeout_ms=data_cfg.get("timeout_ms", 15000),
            max_retries=data_cfg.get("max_retries", 3),
            cache_dir=cache_dir,
        )
        cg_cfg = data_cfg.get("coingecko", {})
        self.coingecko = CoinGeckoDataSource(
            vs_currency=cg_cfg.get("vs_currency", "usd"),
            per_page=cg_cfg.get("per_page", 250),
            timeout_ms=data_cfg.get("timeout_ms", 15000),
            max_retries=data_cfg.get("max_retries", 3),
            cache_dir=cache_dir,
        )
        self.vs_currency = data_cfg.get("vs_currency", "USDT")
        self.interval = data_cfg.get("interval", "1d")
        self.top_n = uni_cfg.get("top_n", 100)
        self.exclude_symbols = {s.upper() for s in uni_cfg.get("exclude_symbols", [])}
        self.min_volume = uni_cfg.get("liquidity", {}).get("min_24h_volume_usd", 1_000_000)
        self.min_data_points = uni_cfg.get("min_data_points", 30)
        # windows determine how much history we need
        self.windows: list[int] = config.get("windows", [30, 90, 180])
        self.ohlcv_limit = max(self.windows) + 5  # + buffer for returns

    @staticmethod
    def _cache_filename(symbol: str, interval: str) -> str:
        """Single source of truth for cache filename. Symbol may contain '/'."""
        safe = symbol.replace("/", "_")
        return f"{safe}_{interval}.csv"

    @staticmethod
    def _parse_symbol_from_cache(cache_path: Path) -> str:
        """Derive symbol from cache filename. E.g. ETH_USDT_1d.csv -> ETH, BTC_USDT_1d.csv -> BTC."""
        stem = cache_path.stem  # e.g. ETH_USDT_1d
        # Symbol is first part before '_' (handles ETH_USDT_1d -> ETH)
        # For symbols like BTC, ETH, etc.
        return stem.split("_")[0].upper()

    # -- Universe ------------------------------------------------------------

    def get_universe(self) -> list[dict]:
        """
        Fetch and filter universe.

        Filters applied (in order, all logged):
          1. Exclude symbols in exclude_symbols (stablecoins, WBTC, BTC).
          2. Liquidity filter: volume_24h < min_24h_volume_usd -> excluded.
          3. Returns filtered list sorted by market_cap desc.

        Returns:
            Filtered list of coin dicts.
        """
        logger.info("Fetching universe top_n=%d via CoinGecko", self.top_n)
        # Fetch more than top_n to account for filtering
        raw = self.coingecko.fetch_universe(top_n=self.top_n * 2, vs_currency="usd")
        if not raw:
            logger.error("Universe fetch returned 0 coins — API failure or rate limit. No silent fallback.")
            return []

        filtered: list[dict] = []
        for coin in raw:
            sym = (coin.get("symbol") or "").upper()
            if sym in self.exclude_symbols:
                logger.info("Universe filter: excluded %s (%s) — in exclude_symbols", sym, coin.get("id"))
                continue
            vol = coin.get("volume_24h") or 0
            if vol < self.min_volume:
                logger.info(
                    "Universe filter: excluded %s — volume %.0f < threshold %.0f",
                    sym, vol, self.min_volume,
                )
                continue
            filtered.append(coin)
            if len(filtered) >= self.top_n:
                break

        logger.info("Universe: %d raw -> %d after filters (target %d)", len(raw), len(filtered), self.top_n)
        if len(filtered) < self.top_n:
            logger.warning(
                "Universe smaller than requested: %d < %d (filters too strict or API limited)",
                len(filtered), self.top_n,
            )
        return filtered

    # -- OHLCV ---------------------------------------------------------------

    def fetch_all_ohlcv(
        self,
        universe: list[dict],
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Fetch BTC OHLCV + each alt's OHLCV.

        Tries Binance first (symbol = "{SYMBOL}/{VS_CURRENCY}"), falls back to
        CoinGecko id on failure if coingecko.fallback_on_binance_failure is true.

        Returns:
            (btc_df, alt_map) where alt_map is {symbol: DataFrame}.
            Empty DataFrames indicate fetch failure (logged explicitly).
        """
        data_cfg = self.config.get("data", {})
        fallback = data_cfg.get("coingecko", {}).get("fallback_on_binance_failure", True)

        # BTC first
        btc_symbol = f"BTC/{self.vs_currency}"
        logger.info("Fetching BTC OHLCV: %s interval=%s limit=%d", btc_symbol, self.interval, self.ohlcv_limit)
        btc_df = self.binance.fetch_ohlcv(btc_symbol, interval=self.interval, limit=self.ohlcv_limit)
        if btc_df.empty and fallback:
            logger.warning("BTC Binance fetch empty — trying CoinGecko fallback (id=bitcoin)")
            btc_df = self.coingecko.fetch_ohlcv("bitcoin", interval=self.interval, limit=self.ohlcv_limit)
        if btc_df.empty:
            logger.error("CRITICAL: BTC OHLCV is empty — pipeline cannot proceed without BTC data")

        alt_map: dict[str, pd.DataFrame] = {}
        for coin in tqdm(universe, desc="Fetching alt OHLCV", unit="coin"):
            sym = (coin.get("symbol") or "").upper()
            cg_id = coin.get("id") or sym.lower()
            binance_symbol = f"{sym}/{self.vs_currency}"
            df = self.binance.fetch_ohlcv(binance_symbol, interval=self.interval, limit=self.ohlcv_limit)
            if df.empty and fallback:
                logger.warning("Binance empty for %s — trying CoinGecko fallback id=%s", sym, cg_id)
                df = self.coingecko.fetch_ohlcv(cg_id, interval=self.interval, limit=self.ohlcv_limit)
            if df.empty:
                logger.error("OHLCV FAILED for %s (binance=%s, coingecko=%s) — will be skipped downstream", sym, binance_symbol, cg_id)
            else:
                # Liquidity sanity: if volume column is all zero, warn
                if "volume" in df.columns and (df["volume"] == 0).all():
                    logger.warning("OHLCV for %s has zero volume throughout — possible illiquid/delisted", sym)
                # Enforce min_data_points
                if len(df) < self.min_data_points:
                    logger.warning(
                        "OHLCV for %s has only %d rows < min_data_points %d — will be skipped in metrics",
                        sym, len(df), self.min_data_points,
                    )
            alt_map[sym] = df

        logger.info("Fetched OHLCV: BTC %d rows, %d alts", len(btc_df), len(alt_map))
        return btc_df, alt_map

    def load_cached_ohlcv(self) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
        """
        Load cached OHLCV from cache_dir without hitting APIs.

        Returns:
            (btc_df or None, alt_map). Used for offline / graceful fallback.
        """
        cache_dir = Path(self.config.get("data", {}).get("cache_dir", "data"))
        if not cache_dir.exists():
            logger.warning("Cache dir %s does not exist — no cached data", cache_dir)
            return None, {}
        # Use single source of truth for BTC cache filename
        btc_cache_name = self._cache_filename(f"BTC/{self.vs_currency}", self.interval)
        candidates = [
            cache_dir / btc_cache_name,
            cache_dir / f"BTC_USDT_{self.interval}.csv",
            cache_dir / "BTC_USDT_1d.csv",
        ]
        btc_df: pd.DataFrame | None = None
        for p in candidates:
            if p.exists():
                try:
                    btc_df = pd.read_csv(p, parse_dates=["timestamp"])
                    logger.info("Loaded cached BTC from %s (%d rows)", p, len(btc_df))
                    break
                except Exception as exc:
                    logger.warning("Failed to read cached BTC %s: %s", p, exc)
        # Also try generic scan
        if btc_df is None:
            for p in cache_dir.glob("BTC*.csv"):
                try:
                    btc_df = pd.read_csv(p, parse_dates=["timestamp"])
                    logger.info("Loaded cached BTC from %s (%d rows)", p, len(btc_df))
                    break
                except Exception:
                    continue

        alt_map: dict[str, pd.DataFrame] = {}
        for csv_path in cache_dir.glob("*.csv"):
            name = csv_path.stem
            # Skip BTC files already handled
            if name.startswith("BTC"):
                continue
            try:
                df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                sym = self._parse_symbol_from_cache(csv_path)
                alt_map[sym] = df
            except Exception as exc:
                logger.warning("Failed to read cached %s: %s", csv_path, exc)
        logger.info("Loaded cached alt OHLCV: %d symbols", len(alt_map))
        return btc_df, alt_map
