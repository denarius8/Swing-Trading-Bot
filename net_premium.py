"""
SPX/NDX Net Premium Tracker
Approximates net dollar flow into options (bullish vs bearish positioning).
Auto-calculates from yfinance options chains with manual Unusual Whales override.
"""

import os
import json
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


CACHE_FILE = os.path.join("cache", "net_premium.json")
NDX_CACHE_FILE = os.path.join("cache", "ndx_net_premium.json")


def _get_cache_file(index='SPX'):
    """Return the cache file path for the given index."""
    if index == 'NDX':
        return NDX_CACHE_FILE
    return CACHE_FILE


def _load_history(index='SPX'):
    """Load premium history from disk."""
    cache_file = _get_cache_file(index)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {"history": []}


def _save_history(data, index='SPX'):
    """Save premium history to disk."""
    os.makedirs("cache", exist_ok=True)
    cache_file = _get_cache_file(index)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def calculate_net_premium(index='SPX'):
    """
    Approximate net premium from yfinance options chain.
    Net premium = call_flow - put_flow (positive = bullish, negative = bearish).
    Weights shorter-dated expirations higher.

    For SPX: uses ^SPX options chain.
    For NDX: uses QQQ options chain (more liquid than ^NDX options).
    """
    try:
        if index == 'NDX':
            ticker = yf.Ticker("QQQ")
        else:
            ticker = yf.Ticker("^SPX")

        expirations = ticker.options
        if not expirations:
            return None

        # Use next 5 expirations
        exp_list = list(expirations[:5])

        total_call_flow = 0
        total_put_flow = 0
        per_expiry = []

        today = datetime.now().date()

        for exp_str in exp_list:
            try:
                chain = ticker.option_chain(exp_str)
                calls = chain.calls
                puts = chain.puts

                # Calculate days to expiration for weighting
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < 0:
                    continue

                # Weight: 0-7 DTE = 2x, 7-30 DTE = 1x, 30+ DTE = 0.5x
                if dte <= 7:
                    weight = 2.0
                elif dte <= 30:
                    weight = 1.0
                else:
                    weight = 0.5

                # Call flow: sum(volume × mid_price × 100) for strikes with volume
                call_flow = 0
                for _, row in calls.iterrows():
                    vol = row.get("volume", 0)
                    if vol is None or np.isnan(vol) or vol <= 0:
                        continue
                    bid = row.get("bid", 0) or 0
                    ask = row.get("ask", 0) or 0
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else (row.get("lastPrice", 0) or 0)
                    if mid > 0:
                        call_flow += vol * mid * 100

                # Put flow
                put_flow = 0
                for _, row in puts.iterrows():
                    vol = row.get("volume", 0)
                    if vol is None or np.isnan(vol) or vol <= 0:
                        continue
                    bid = row.get("bid", 0) or 0
                    ask = row.get("ask", 0) or 0
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else (row.get("lastPrice", 0) or 0)
                    if mid > 0:
                        put_flow += vol * mid * 100

                weighted_call = call_flow * weight
                weighted_put = put_flow * weight
                total_call_flow += weighted_call
                total_put_flow += weighted_put

                per_expiry.append({
                    "expiration": exp_str,
                    "dte": dte,
                    "weight": weight,
                    "call_flow": round(call_flow),
                    "put_flow": round(put_flow),
                    "net_flow": round(call_flow - put_flow),
                })

            except Exception:
                continue

        net_premium = round(total_call_flow - total_put_flow)
        total_premium = round(total_call_flow + total_put_flow)

        return {
            "net_premium": net_premium,
            "total_premium": total_premium,
            "call_flow": round(total_call_flow),
            "put_flow": round(total_put_flow),
            "per_expiry": per_expiry,
            "expirations_used": len(per_expiry),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception as e:
        print(f"[NET PREMIUM] Error: {e}")
        return None


def save_daily_premium(date_str, net_premium, total_premium, source="auto",
                       spx_open=None, spx_close=None, change_pct=None, index='SPX'):
    """Append or update a day's premium data in history."""
    data = _load_history(index)
    history = data["history"]

    # Check if entry for this date already exists
    existing = None
    for i, entry in enumerate(history):
        if entry["date"] == date_str:
            existing = i
            break

    entry = {
        "date": date_str,
        "open": spx_open,
        "close": spx_close,
        "change_pct": change_pct,
        "net_premium": net_premium,
        "total_premium": total_premium,
        "source": source,
    }

    # Keep auto_estimate if overriding with manual
    if source == "manual" and existing is not None:
        entry["auto_estimate"] = history[existing].get("net_premium") \
            if history[existing].get("source") == "auto" \
            else history[existing].get("auto_estimate")
    elif source == "auto":
        entry["auto_estimate"] = net_premium

    if existing is not None:
        # Preserve manual override if auto is updating
        if source == "auto" and history[existing].get("source") == "manual":
            entry["auto_estimate"] = net_premium
            entry["net_premium"] = history[existing]["net_premium"]
            entry["total_premium"] = history[existing].get("total_premium", total_premium)
            entry["source"] = "manual"
        history[existing] = entry
    else:
        history.append(entry)

    # Sort by date descending
    history.sort(key=lambda x: x["date"], reverse=True)

    # Keep last 60 days
    data["history"] = history[:60]
    _save_history(data, index)
    return entry


def update_manual_premium(date_str, net_premium_value, total_premium_value=None, index='SPX'):
    """Override a day's net premium with the real Unusual Whales value."""
    data = _load_history(index)
    history = data["history"]

    # Find existing entry
    for entry in history:
        if entry["date"] == date_str:
            entry["auto_estimate"] = entry.get("auto_estimate") or entry.get("net_premium")
            entry["net_premium"] = net_premium_value
            if total_premium_value is not None:
                entry["total_premium"] = total_premium_value
            entry["source"] = "manual"
            _save_history(data, index)
            return entry

    # No existing entry — create one with manual data
    entry = {
        "date": date_str,
        "open": None,
        "close": None,
        "change_pct": None,
        "net_premium": net_premium_value,
        "total_premium": total_premium_value,
        "source": "manual",
        "auto_estimate": None,
    }
    history.append(entry)
    history.sort(key=lambda x: x["date"], reverse=True)
    data["history"] = history[:60]
    _save_history(data, index)
    return entry


def get_premium_table(days=20, index='SPX'):
    """
    Return last N days of premium data for display.
    Merges stored history with OHLC data from the appropriate index.
    """
    data = _load_history(index)
    history = data["history"][:days]

    # Fill in any missing OHLC data using the appropriate index ticker
    ohlc_ticker = "^NDX" if index == 'NDX' else "^GSPC"
    if history:
        try:
            tk = yf.Ticker(ohlc_ticker)
            daily = tk.history(period="2mo")
            if not daily.empty:
                for entry in history:
                    if entry.get("close") is None or entry.get("open") is None:
                        date_str = entry["date"]
                        for idx in daily.index:
                            idx_date = idx.date() if hasattr(idx, 'date') else idx
                            if str(idx_date) == date_str:
                                entry["open"] = round(float(daily.loc[idx, "Open"]), 2)
                                entry["close"] = round(float(daily.loc[idx, "Close"]), 2)
                                if len(daily.index) > 1:
                                    loc = daily.index.get_loc(idx)
                                    if loc > 0:
                                        prev_close = float(daily.iloc[loc - 1]["Close"])
                                        entry["change_pct"] = round(
                                            (entry["close"] - prev_close) / prev_close * 100, 2)
                                break
        except Exception:
            pass

    # Calculate streak
    streak = 0
    streak_dir = None
    for entry in history:
        np_val = entry.get("net_premium")
        if np_val is None:
            break
        if np_val > 0:
            if streak_dir is None:
                streak_dir = "positive"
            if streak_dir == "positive":
                streak += 1
            else:
                break
        elif np_val < 0:
            if streak_dir is None:
                streak_dir = "negative"
            if streak_dir == "negative":
                streak += 1
            else:
                break
        else:
            break

    return {
        "history": history,
        "streak": streak,
        "streak_direction": streak_dir or "neutral",
        "days_shown": len(history),
    }


def fetch_net_premium_signal(index='SPX'):
    """
    Return a signal for the confidence system based on net premium streak.
    4+ consecutive positive → bullish (+1)
    4+ consecutive negative → bearish (-1)
    Otherwise → neutral (0)
    """
    table = get_premium_table(days=10, index=index)
    streak = table["streak"]
    direction = table["streak_direction"]

    if streak >= 4 and direction == "positive":
        signal = 1
        label = f"{streak} days positive net premium"
        detail = "Sustained bullish options flow — buying pressure likely"
    elif streak >= 4 and direction == "negative":
        signal = -1
        label = f"{streak} days negative net premium"
        detail = "Sustained bearish options flow — selling pressure likely"
    elif streak >= 2 and direction == "positive":
        signal = 0
        label = f"{streak} days positive (building)"
        detail = "Bullish flow emerging but not yet sustained (need 4+ days)"
    elif streak >= 2 and direction == "negative":
        signal = 0
        label = f"{streak} days negative (building)"
        detail = "Bearish flow emerging but not yet sustained (need 4+ days)"
    else:
        signal = 0
        label = "No clear premium trend"
        detail = "Net premium direction is mixed — no sustained flow signal"

    # Get latest values for display
    latest = table["history"][0] if table["history"] else None
    net_prem = latest["net_premium"] if latest else None
    source = latest.get("source", "auto") if latest else "none"

    return {
        "signal": signal,
        "label": label,
        "detail": detail,
        "streak": streak,
        "streak_direction": direction,
        "latest_net_premium": net_prem,
        "latest_source": source,
    }


def auto_update_today(index='SPX'):
    """Calculate today's net premium and save it. Called by the API."""
    result = calculate_net_premium(index=index)
    if result is None:
        return None

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Get today's OHLC for the appropriate index
    ohlc_ticker = "^NDX" if index == 'NDX' else "^GSPC"
    spx_open = None
    spx_close = None
    change_pct = None
    try:
        tk = yf.Ticker(ohlc_ticker)
        daily = tk.history(period="5d")
        if not daily.empty:
            spx_close = round(float(daily["Close"].iloc[-1]), 2)
            spx_open = round(float(daily["Open"].iloc[-1]), 2)
            if len(daily) >= 2:
                prev_close = float(daily["Close"].iloc[-2])
                change_pct = round((spx_close - prev_close) / prev_close * 100, 2)
    except Exception:
        pass

    entry = save_daily_premium(
        date_str=today_str,
        net_premium=result["net_premium"],
        total_premium=result["total_premium"],
        source="auto",
        spx_open=spx_open,
        spx_close=spx_close,
        change_pct=change_pct,
        index=index,
    )

    return {
        "entry": entry,
        "calculation": result,
    }
