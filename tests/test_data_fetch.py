"""Tests for data_fetch.py — universe filter, OHLCV fallback, cache handling (mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_fetch import DataFetcher


@pytest.fixture
def mock_config(tmp_path):
    return {
        "windows": [30, 90, 180],
        "primary_window": 180,
        "thresholds": {"corr_threshold_high": 0.7, "corr_threshold_low": 0.2, "beta_min": 3.0, "beta_max": 5.0, "alpha": 0.05},
        "universe": {"top_n": 5, "exclude_symbols": ["USDT", "BTC"], "liquidity": {"min_24h_volume_usd": 1_000_000}, "min_data_points": 30},
        "data": {"cache_dir": str(tmp_path / "data"), "vs_currency": "USDT", "interval": "1d", "exchange": "binance", "rate_limit_ms": 0, "timeout_ms": 5000, "max_retries": 1, "coingecko": {"vs_currency": "usd", "per_page": 250, "fallback_on_binance_failure": True}},
        "preprocessing": {"missing_data_policy": "drop", "forward_fill_limit": 2, "outlier_handling": "none", "winsorize_limits": [0.01, 0.01]},
    }


class TestUniverseFilter:
    def test_exclude_symbols_filtered(self, mock_config):
        fetcher = DataFetcher(mock_config)
        raw = [
            {"id": "bitcoin", "symbol": "BTC", "market_cap": 1e12, "volume_24h": 5e10},
            {"id": "tether", "symbol": "USDT", "market_cap": 1e11, "volume_24h": 5e10},
            {"id": "ethereum", "symbol": "ETH", "market_cap": 5e11, "volume_24h": 2e10},
            {"id": "solana", "symbol": "SOL", "market_cap": 1e11, "volume_24h": 3e9},
        ]
        with patch.object(fetcher.coingecko, "fetch_universe", return_value=raw):
            universe = fetcher.get_universe()
        symbols = [c["symbol"] for c in universe]
        assert "BTC" not in symbols
        assert "USDT" not in symbols
        assert "ETH" in symbols

    def test_liquidity_filter(self, mock_config):
        fetcher = DataFetcher(mock_config)
        raw = [
            {"id": "coin-a", "symbol": "COINA", "market_cap": 1e9, "volume_24h": 500_000},  # below threshold
            {"id": "coin-b", "symbol": "COINB", "market_cap": 1e9, "volume_24h": 5_000_000},
        ]
        with patch.object(fetcher.coingecko, "fetch_universe", return_value=raw):
            universe = fetcher.get_universe()
        symbols = [c["symbol"] for c in universe]
        assert "COINA" not in symbols
        assert "COINB" in symbols

    def test_empty_universe_returns_empty(self, mock_config):
        fetcher = DataFetcher(mock_config)
        with patch.object(fetcher.coingecko, "fetch_universe", return_value=[]):
            universe = fetcher.get_universe()
        assert universe == []


class TestOHLCVFallback:
    def test_btc_fallback_to_coingecko(self, mock_config):
        fetcher = DataFetcher(mock_config)
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        btc_df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"), "open": [1]*5, "high": [1]*5, "low": [1]*5, "close": [1]*5, "volume": [1]*5})
        with patch.object(fetcher.binance, "fetch_ohlcv", return_value=empty):
            with patch.object(fetcher.coingecko, "fetch_ohlcv", return_value=btc_df):
                universe = [{"id": "ethereum", "symbol": "ETH"}]
                # Mock alt also
                with patch.object(fetcher.binance, "fetch_ohlcv", side_effect=[empty, empty]):
                    with patch.object(fetcher.coingecko, "fetch_ohlcv", side_effect=[btc_df, btc_df]):
                        btc, alts = fetcher.fetch_all_ohlcv(universe)
                        assert not btc.empty

    def test_alt_fallback(self, mock_config):
        fetcher = DataFetcher(mock_config)
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        alt_df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"), "open": [1]*5, "high": [1]*5, "low": [1]*5, "close": [2]*5, "volume": [1]*5})
        btc_df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"), "open": [1]*5, "high": [1]*5, "low": [1]*5, "close": [1]*5, "volume": [1]*5})
        # First call is BTC, second is alt
        with patch.object(fetcher.binance, "fetch_ohlcv", side_effect=[btc_df, empty]):
            with patch.object(fetcher.coingecko, "fetch_ohlcv", return_value=alt_df):
                btc, alts = fetcher.fetch_all_ohlcv([{"id": "ethereum", "symbol": "ETH"}])
                assert "ETH" in alts
                assert not alts["ETH"].empty


class TestCacheHandling:
    def test_cache_filename_single_source(self):
        assert DataFetcher._cache_filename("BTC/USDT", "1d") == "BTC_USDT_1d.csv"
        assert DataFetcher._cache_filename("ETH/USDT", "1d") == "ETH_USDT_1d.csv"

    def test_parse_symbol_from_cache(self, tmp_path):
        p = tmp_path / "ETH_USDT_1d.csv"
        assert DataFetcher._parse_symbol_from_cache(p) == "ETH"
        p2 = tmp_path / "BTC_USDT_1d.csv"
        assert DataFetcher._parse_symbol_from_cache(p2) == "BTC"

    def test_load_cached_ohlcv(self, mock_config, tmp_path):
        cache_dir = Path(mock_config["data"]["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Create BTC cache
        btc = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"), "open": [1,2,3], "high": [1,2,3], "low": [1,2,3], "close": [1,2,3], "volume": [1,1,1]})
        btc.to_csv(cache_dir / "BTC_USDT_1d.csv", index=False)
        eth = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"), "open": [1,2,3], "high": [1,2,3], "low": [1,2,3], "close": [2,3,4], "volume": [1,1,1]})
        eth.to_csv(cache_dir / "ETH_USDT_1d.csv", index=False)
        fetcher = DataFetcher(mock_config)
        btc_df, alt_map = fetcher.load_cached_ohlcv()
        assert btc_df is not None and not btc_df.empty
        assert "ETH" in alt_map

    def test_load_cached_empty_dir(self, mock_config):
        fetcher = DataFetcher(mock_config)
        # Ensure empty
        btc_df, alt_map = fetcher.load_cached_ohlcv()
        # May be empty if no files yet, but should not crash
        assert isinstance(alt_map, dict)
