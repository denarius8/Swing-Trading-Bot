"""Fetch and cache SPX historical data."""

import os
import time
import json
import subprocess
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config


def _safe_remove(path):
    """Delete a file before rewriting it.
    macOS com.apple.provenance xattr cannot be stripped — the only reliable
    fix is to remove the old file so the new write creates a clean one.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


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

    # Check if cache includes the most recent trading day's close
    # If it's after 4:30pm ET on a weekday and cache was fetched before today, it's stale
    now = datetime.now()
    age_hours = (now - fetched).total_seconds() / 3600

    # Always stale if older than cache expiry
    if age_hours > config.CACHE_EXPIRY_HOURS:
        return False

    # Check if a new trading day has closed since cache was fetched
    # Load cached data to check the last date
    cache = _cache_path()
    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            last_cached_date = df.index[-1].date()
            # Fetch minimal fresh data to see if there's a newer close
            fresh = yf.Ticker(config.TICKER).history(period="5d")
            if not fresh.empty:
                latest_date = fresh.index[-1]
                if hasattr(latest_date, 'tz') and latest_date.tz:
                    latest_date = latest_date.tz_localize(None) if hasattr(latest_date, 'tz_localize') else latest_date.replace(tzinfo=None)
                if latest_date.date() > last_cached_date:
                    return False  # New trading day available
        except Exception:
            pass

    return True


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

    # Delete old files first — macOS com.apple.provenance blocks in-place overwrites
    _safe_remove(cache)
    _safe_remove(_cache_meta_path())
    df.to_csv(cache)
    with open(_cache_meta_path(), "w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "rows": len(df)}, f)

    print(f"[DATA] {len(df)} trading days fetched ({df.index[0].date()} to {df.index[-1].date()})")
    return df
