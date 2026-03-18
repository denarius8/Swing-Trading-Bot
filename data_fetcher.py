"""Fetch and cache SPX historical data."""

import os
import time
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config


def _cache_path():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, "spx_daily.csv")


def _cache_meta_path():
    return os.path.join(config.CACHE_DIR, "meta.json")


def _cache_is_fresh():
    meta = _cache_meta_path()
    if not os.path.exists(meta):
        return False
    with open(meta) as f:
        info = json.load(f)
    fetched = datetime.fromisoformat(info["fetched_at"])
    return (datetime.now() - fetched).total_seconds() < config.CACHE_EXPIRY_HOURS * 3600


def fetch_spx_data(force_refresh=False):
    """Fetch 5 years of daily SPX OHLCV data. Uses cache if fresh."""
    cache = _cache_path()

    if not force_refresh and _cache_is_fresh() and os.path.exists(cache):
        print("[DATA] Loading from cache...")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[DATA] {len(df)} trading days loaded (cached)")
        return df

    print("[DATA] Fetching SPX data from Yahoo Finance...")
    end = datetime.now()
    start = end - timedelta(days=config.DATA_PERIOD_YEARS * 365)

    ticker = yf.Ticker(config.TICKER)
    df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    if df.empty:
        raise RuntimeError("Failed to fetch SPX data. Check your internet connection.")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    # Save cache
    df.to_csv(cache)
    with open(_cache_meta_path(), "w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "rows": len(df)}, f)

    print(f"[DATA] {len(df)} trading days fetched ({df.index[0].date()} to {df.index[-1].date()})")
    return df
