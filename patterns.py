"""
AskLivermore-Style Chart Pattern Scanner
Detects 15 technical chart patterns with quality grading (A+ to B).
Phase 1: VCP, Bull Flag, New Uptrend, Golden Pocket, Livermore Breakout
"""

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _find_swing_points(prices, order=5):
    """
    Find local swing highs and lows.
    order: a point must be the max/min within [i-order, i+order].
    Returns (highs, lows) as lists of (index_position, price).
    """
    highs = []
    lows = []
    arr = np.array(prices)
    n = len(arr)

    for i in range(order, n - order):
        window = arr[i - order:i + order + 1]
        if arr[i] == window.max():
            highs.append((i, float(arr[i])))
        if arr[i] == window.min():
            lows.append((i, float(arr[i])))

    return highs, lows


def _fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels from a swing high to low."""
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.500": high - diff * 0.500,
        "0.618": high - diff * 0.618,
        "0.650": high - diff * 0.650,
        "0.786": high - diff * 0.786,
        "1.0": low,
    }


def _volume_trend(volumes, window=10):
    """Check if volume is contracting over window. Returns ratio of recent vs earlier."""
    if len(volumes) < window * 2:
        return 1.0
    recent = np.mean(volumes[-window:])
    earlier = np.mean(volumes[-window * 2:-window])
    return recent / earlier if earlier > 0 else 1.0


def _trend_template(df):
    """
    Minervini Trend Template pre-filter.
    Returns True if stock passes basic trend criteria for long setups.
    """
    if len(df) < 200:
        return False

    close = df["Close"].values
    price = close[-1]

    sma50 = np.mean(close[-50:])
    sma150 = np.mean(close[-150:])
    sma200 = np.mean(close[-200:])

    # Price above key MAs
    if price < sma50 or price < sma150 or price < sma200:
        return False

    # 150 SMA > 200 SMA
    if sma150 < sma200:
        return False

    # 200 SMA trending up (current > 1 month ago)
    sma200_prev = np.mean(close[-230:-30]) if len(close) >= 230 else sma200
    if sma200 < sma200_prev:
        return False

    # Price within 25% of 52-week high
    high_52w = max(df["High"].values[-252:]) if len(df) >= 252 else max(df["High"].values)
    if price < high_52w * 0.75:
        return False

    # Price at least 25% above 52-week low
    low_52w = min(df["Low"].values[-252:]) if len(df) >= 252 else min(df["Low"].values)
    if price < low_52w * 1.25:
        return False

    return True


def _grade_score(score, max_score):
    """Convert numeric score to letter grade."""
    pct = score / max_score if max_score > 0 else 0
    if pct >= 0.9:
        return "A+"
    elif pct >= 0.75:
        return "A"
    elif pct >= 0.6:
        return "B+"
    else:
        return "B"


# ---------------------------------------------------------------------------
# Pattern Detectors - Phase 1
# ---------------------------------------------------------------------------

def detect_vcp(df):
    """
    Volatility Contraction Pattern (Mark Minervini).
    Detects successive tightening price ranges with declining volume.
    """
    result = {"pattern": "VCP", "detected": False, "direction": "LONG"}

    if len(df) < 60:
        return result

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    price = close[-1]

    # Find the base: highest high in last 120 days
    lookback = min(120, len(df))
    base_high = max(high[-lookback:])
    base_high_idx = len(high) - lookback + np.argmax(high[-lookback:])

    # Price must be within 25% of the base high
    if price < base_high * 0.75:
        return result

    # Find contractions: measure successive pullback depths from the base high
    contractions = []
    swing_highs, swing_lows = _find_swing_points(close[-lookback:], order=3)

    if len(swing_lows) < 2:
        return result

    for i, (idx, low_price) in enumerate(swing_lows):
        pullback_pct = (base_high - low_price) / base_high * 100
        if pullback_pct > 3:  # Meaningful pullback
            contractions.append({
                "idx": idx,
                "low": low_price,
                "pullback_pct": pullback_pct,
            })

    if len(contractions) < 2:
        return result

    # Check for volatility contraction: each pullback should be smaller
    contracting = 0
    for i in range(1, len(contractions)):
        if contractions[i]["pullback_pct"] < contractions[i - 1]["pullback_pct"]:
            contracting += 1

    if contracting == 0:
        return result

    # Volume should decline through the base
    vol_ratio = _volume_trend(volume[-lookback:], window=10)

    # Grade scoring
    score = 0
    max_score = 6

    if contracting >= 2:
        score += 2  # 3+ contractions
    elif contracting >= 1:
        score += 1

    # Tight final contraction
    final_pullback = contractions[-1]["pullback_pct"]
    if final_pullback < 5:
        score += 2
    elif final_pullback < 10:
        score += 1

    # Volume declining
    if vol_ratio < 0.7:
        score += 1
    elif vol_ratio < 0.9:
        score += 0.5

    # Price near breakout (within 3% of base high)
    proximity = (base_high - price) / base_high * 100
    if proximity < 3:
        score += 0.5

    if score < 2:
        return result

    # Calculate entry/stop/target
    entry_low = price * 0.99
    entry_high = base_high * 1.01
    stop = contractions[-1]["low"] * 0.98
    target = price + (base_high - contractions[-1]["low"])  # Measured move
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "base_high": round(base_high, 2),
            "contractions": len(contractions),
            "contracting_count": contracting,
            "final_pullback_pct": round(final_pullback, 1),
            "volume_contraction": round(vol_ratio, 2),
            "breakout_proximity_pct": round(proximity, 1),
        },
        "entry_zone": [round(entry_low, 2), round(entry_high, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_bull_flag(df):
    """
    Bull Flag pattern.
    Strong upward pole followed by tight consolidation channel with declining volume.
    """
    result = {"pattern": "Bull Flag", "detected": False, "direction": "LONG"}

    if len(df) < 30:
        return result

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    price = close[-1]

    # Look for a strong pole in the last 30 bars
    best_pole = None
    for start in range(max(0, len(close) - 30), len(close) - 5):
        for end in range(start + 3, min(start + 12, len(close) - 3)):
            gain_pct = (close[end] - close[start]) / close[start] * 100
            bars = end - start
            if gain_pct >= 8 and bars <= 10:  # 8%+ gain in <=10 bars
                if best_pole is None or gain_pct > best_pole["gain_pct"]:
                    best_pole = {
                        "start": start,
                        "end": end,
                        "start_price": close[start],
                        "end_price": close[end],
                        "gain_pct": gain_pct,
                        "bars": bars,
                    }

    if best_pole is None:
        return result

    # Check consolidation after the pole
    pole_end = best_pole["end"]
    pole_high = max(high[best_pole["start"]:pole_end + 1])
    consol_bars = close[pole_end:]

    if len(consol_bars) < 3:
        return result

    # Consolidation checks
    consol_high = max(high[pole_end:])
    consol_low = min(low[pole_end:])
    consol_range_pct = (consol_high - consol_low) / consol_high * 100
    retracement = (pole_high - consol_low) / (pole_high - best_pole["start_price"]) * 100

    # Consolidation should retrace < 50% of pole and be tight
    if retracement > 50 or consol_range_pct > 8:
        return result

    # Volume should decline during consolidation
    pole_vol = np.mean(volume[best_pole["start"]:pole_end + 1])
    consol_vol = np.mean(volume[pole_end:])
    vol_decline = consol_vol / pole_vol if pole_vol > 0 else 1

    # Grade scoring
    score = 0
    max_score = 6

    # Strong pole
    if best_pole["gain_pct"] >= 15:
        score += 2
    elif best_pole["gain_pct"] >= 10:
        score += 1.5
    else:
        score += 1

    # Tight consolidation
    if consol_range_pct < 3:
        score += 1.5
    elif consol_range_pct < 5:
        score += 1

    # Low retracement
    if retracement < 30:
        score += 1
    elif retracement < 40:
        score += 0.5

    # Volume decline
    if vol_decline < 0.5:
        score += 1
    elif vol_decline < 0.7:
        score += 0.5

    if score < 2.5:
        return result

    # Entry/stop/target
    entry_low = consol_high * 0.99
    entry_high = consol_high * 1.01
    stop = consol_low * 0.98
    target = consol_high + (pole_high - best_pole["start_price"])  # Measured move
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "pole_gain_pct": round(best_pole["gain_pct"], 1),
            "pole_bars": best_pole["bars"],
            "consolidation_bars": len(consol_bars),
            "consolidation_range_pct": round(consol_range_pct, 1),
            "retracement_pct": round(retracement, 1),
            "volume_decline": round(vol_decline, 2),
        },
        "entry_zone": [round(entry_low, 2), round(entry_high, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_new_uptrend(df):
    """
    New Uptrend Detection.
    Price > rising 50 SMA > rising 200 SMA with volume confirmation.
    """
    result = {"pattern": "New Uptrend", "detected": False, "direction": "LONG"}

    if len(df) < 200:
        return result

    close = df["Close"].values
    volume = df["Volume"].values
    price = close[-1]

    sma50 = pd.Series(close).rolling(50).mean().values
    sma200 = pd.Series(close).rolling(200).mean().values

    # Current values
    sma50_now = sma50[-1]
    sma200_now = sma200[-1]

    # Must be: price > SMA50 > SMA200
    if not (price > sma50_now > sma200_now):
        return result

    # SMA50 must be rising (compare to 10 bars ago)
    if sma50[-1] <= sma50[-10]:
        return result

    # SMA200 must be rising or at least flat
    if sma200[-1] < sma200[-20] * 0.998:
        return result

    # Check if this is a NEW uptrend (recently crossed)
    # SMA50 crossed above SMA200 in last 30 bars
    recent_cross = False
    for i in range(-30, -1):
        if sma50[i - 1] <= sma200[i - 1] and sma50[i] > sma200[i]:
            recent_cross = True
            break

    # Or price recently reclaimed SMA50 from below
    price_reclaim = False
    for i in range(-15, -1):
        if close[i - 1] <= sma50[i - 1] and close[i] > sma50[i]:
            price_reclaim = True
            break

    # Volume above average
    vol_sma20 = np.mean(volume[-20:])
    vol_recent = np.mean(volume[-5:])
    vol_ratio = vol_recent / vol_sma20 if vol_sma20 > 0 else 1

    # Grade scoring
    score = 0
    max_score = 6

    # Price structure
    score += 1  # Base: price > SMA50 > SMA200

    # Golden cross (recent)
    if recent_cross:
        score += 1.5

    # Price reclaim
    if price_reclaim:
        score += 1

    # Volume confirmation
    if vol_ratio > 1.3:
        score += 1
    elif vol_ratio > 1.1:
        score += 0.5

    # Distance from 52-week low (should be well above)
    low_52w = min(df["Low"].values[-252:])
    dist_from_low = (price - low_52w) / low_52w * 100
    if dist_from_low > 30:
        score += 1
    elif dist_from_low > 15:
        score += 0.5

    if score < 2.5:
        return result

    # Entry/stop/target
    stop = sma50_now * 0.97
    target = price * 1.15  # 15% upside target
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "sma50": round(sma50_now, 2),
            "sma200": round(sma200_now, 2),
            "golden_cross_recent": recent_cross,
            "price_reclaim_recent": price_reclaim,
            "volume_ratio": round(vol_ratio, 2),
            "dist_from_52w_low_pct": round(dist_from_low, 1),
        },
        "entry_zone": [round(price * 0.99, 2), round(price * 1.01, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_golden_pocket(df):
    """
    Fibonacci .618 Golden Pocket Entry.
    Price is at the 61.8%-65% retracement of the most recent significant swing.
    """
    result = {"pattern": "Golden Pocket (.618)", "detected": False, "direction": "LONG"}

    if len(df) < 30:
        return result

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    price = close[-1]

    # Find the most recent significant swing (high then low, or low then high)
    swing_highs, swing_lows = _find_swing_points(close, order=5)

    if not swing_highs or not swing_lows:
        return result

    # Look for a pullback setup: recent swing high followed by current price near .618
    # Use the highest high and subsequent lowest low
    recent_high = None
    for h_idx, h_price in reversed(swing_highs):
        if h_idx < len(close) - 3:  # Not the very last bars
            recent_high = (h_idx, h_price)
            break

    if recent_high is None:
        return result

    # Find the swing low before the high (the start of the move up)
    swing_low = None
    for l_idx, l_price in reversed(swing_lows):
        if l_idx < recent_high[0]:
            swing_low = (l_idx, l_price)
            break

    if swing_low is None:
        return result

    # The move must be significant (at least 5%)
    move_pct = (recent_high[1] - swing_low[1]) / swing_low[1] * 100
    if move_pct < 5:
        return result

    # Calculate Fibonacci levels
    fib = _fibonacci_levels(recent_high[1], swing_low[1])

    # Check if current price is in the golden pocket (61.8% - 65%)
    fib_618 = fib["0.618"]
    fib_650 = fib["0.650"]

    # Allow some tolerance (within 2% of the zone)
    zone_top = fib_618 * 1.02
    zone_bottom = fib_650 * 0.98

    if not (zone_bottom <= price <= zone_top):
        return result

    # Check for bounce confirmation (price showing support at this level)
    rsi = ta.momentum.rsi(pd.Series(close), window=14).values
    rsi_val = rsi[-1] if not np.isnan(rsi[-1]) else 50

    # Grade scoring
    score = 0
    max_score = 6

    # Price in golden pocket
    score += 2

    # RSI oversold at the level
    if rsi_val < 35:
        score += 1.5
    elif rsi_val < 45:
        score += 0.5

    # Clean bounce (price today > yesterday)
    if close[-1] > close[-2]:
        score += 1

    # Volume pickup on bounce
    vol_today = df["Volume"].values[-1]
    vol_avg = np.mean(df["Volume"].values[-20:])
    if vol_today > vol_avg * 1.2:
        score += 1

    # Significance of the original move
    if move_pct > 15:
        score += 0.5

    if score < 3:
        return result

    # Entry/stop/target
    stop = fib["0.786"] * 0.98
    target = recent_high[1]  # Back to the swing high
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "swing_high": round(recent_high[1], 2),
            "swing_low": round(swing_low[1], 2),
            "move_pct": round(move_pct, 1),
            "fib_618": round(fib_618, 2),
            "fib_650": round(fib_650, 2),
            "current_retracement": round((recent_high[1] - price) / (recent_high[1] - swing_low[1]) * 100, 1),
            "rsi": round(rsi_val, 1),
        },
        "entry_zone": [round(fib_650, 2), round(fib_618, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_livermore_breakout(df):
    """
    Livermore Pivotal Point Breakout.
    Price breaking out of consolidation to new highs with volume surge.
    """
    result = {"pattern": "Livermore Breakout", "detected": False, "direction": "LONG"}

    if len(df) < 40:
        return result

    close = df["Close"].values
    high = df["High"].values
    volume = df["Volume"].values
    price = close[-1]

    # Find consolidation range (last 20-40 bars)
    lookback = min(40, len(close) - 5)
    consol_period = close[-lookback:-3]
    consol_high = max(high[-lookback:-3])
    consol_low = min(df["Low"].values[-lookback:-3])

    # Consolidation must be relatively tight (< 15% range)
    consol_range_pct = (consol_high - consol_low) / consol_high * 100
    if consol_range_pct > 15 or consol_range_pct < 2:
        return result

    # Price must be breaking above the consolidation high (within last 3 bars)
    breakout = False
    for i in range(-3, 0):
        if close[i] > consol_high:
            breakout = True
            break

    if not breakout and price <= consol_high:
        return result

    # Volume surge on breakout
    vol_avg = np.mean(volume[-lookback:-3])
    vol_breakout = np.mean(volume[-3:])
    vol_surge = vol_breakout / vol_avg if vol_avg > 0 else 1

    # Must be at or near 20-day high
    high_20 = max(high[-20:])
    near_high = (high_20 - price) / high_20 * 100 < 2

    # Grade scoring
    score = 0
    max_score = 6

    # Breakout confirmed
    if price > consol_high:
        score += 1.5

    # Near 20-day high
    if near_high:
        score += 1

    # Volume surge
    if vol_surge > 2.0:
        score += 2
    elif vol_surge > 1.5:
        score += 1.5
    elif vol_surge > 1.2:
        score += 1

    # Tight consolidation (tighter = better)
    if consol_range_pct < 6:
        score += 1
    elif consol_range_pct < 10:
        score += 0.5

    if score < 2.5:
        return result

    # Entry/stop/target
    stop = consol_low * 0.98
    target = price + (consol_high - consol_low)  # Measured move
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "consol_high": round(consol_high, 2),
            "consol_low": round(consol_low, 2),
            "consol_range_pct": round(consol_range_pct, 1),
            "consolidation_bars": lookback,
            "volume_surge": round(vol_surge, 2),
            "at_20d_high": near_high,
        },
        "entry_zone": [round(consol_high, 2), round(consol_high * 1.01, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


# ---------------------------------------------------------------------------
# Pattern Detectors - Phase 2
# ---------------------------------------------------------------------------

def detect_stage1_base(df):
    """
    Weinstein Stage 1 Base.
    30-week MA flattening, price consolidating above it, volume quiet.
    Uses daily data: 150-day MA as proxy for 30-week.
    """
    result = {"pattern": "Stage 1 Base", "detected": False, "direction": "LONG"}

    if len(df) < 150:
        return result

    close = df["Close"].values
    volume = df["Volume"].values
    price = close[-1]

    ma150 = pd.Series(close).rolling(150).mean().values
    ma50  = pd.Series(close).rolling(50).mean().values

    ma150_now  = ma150[-1]
    ma150_prev = ma150[-20]   # 4 weeks ago

    if np.isnan(ma150_now) or np.isnan(ma150_prev):
        return result

    # Stage 1: 150-day MA must be flattening (< 2% change in 20 days)
    ma_slope_pct = abs(ma150_now - ma150_prev) / ma150_prev * 100
    if ma_slope_pct > 2.5:
        return result

    # Price must be near or above the 150 MA (within 10%)
    if price < ma150_now * 0.90 or price > ma150_now * 1.20:
        return result

    # Price should be below the 52-week high (still basing, not broken out)
    high_52w = max(df["High"].values[-252:]) if len(df) >= 252 else max(df["High"].values)
    if price > high_52w * 0.95:
        return result  # Already at highs — not a base

    # Volume contracting (quiet accumulation)
    vol_recent = np.mean(volume[-20:])
    vol_earlier = np.mean(volume[-60:-20])
    vol_ratio = vol_recent / vol_earlier if vol_earlier > 0 else 1.0

    # Grade scoring
    score = 0
    max_score = 6

    # MA flat
    if ma_slope_pct < 1.0:
        score += 2
    elif ma_slope_pct < 2.0:
        score += 1

    # Price above MA (bullish structure)
    if price > ma150_now:
        score += 1

    # MA50 also flattening / turning up
    ma50_slope = abs(ma50[-1] - ma50[-10]) / ma50[-10] * 100 if not np.isnan(ma50[-10]) else 99
    if ma50_slope < 1.5:
        score += 1

    # Volume contraction
    if vol_ratio < 0.75:
        score += 1.5
    elif vol_ratio < 0.90:
        score += 0.75

    if score < 2.5:
        return result

    stop = ma150_now * 0.95
    target = high_52w
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "ma150": round(ma150_now, 2),
            "ma_slope_pct": round(ma_slope_pct, 2),
            "price_vs_ma150_pct": round((price - ma150_now) / ma150_now * 100, 1),
            "vol_ratio": round(vol_ratio, 2),
            "pct_from_52w_high": round((high_52w - price) / high_52w * 100, 1),
        },
        "entry_zone": [round(price, 2), round(high_52w * 0.95, 2)],
        "stop_loss": round(stop, 2),
        "target": round(high_52w, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_parabolic_short(df):
    """
    Parabolic Short Setup.
    Price extended >2 std devs above 50 SMA, RSI >75 — mean reversion short.
    """
    result = {"pattern": "Parabolic Short", "detected": False, "direction": "SHORT"}

    if len(df) < 50:
        return result

    close = df["Close"].values
    high  = df["High"].values
    price = close[-1]

    sma50 = np.mean(close[-50:])
    std50 = np.std(close[-50:])

    # Price must be >2 std devs above 50 SMA
    z_score = (price - sma50) / std50 if std50 > 0 else 0
    if z_score < 2.0:
        return result

    # RSI overbought
    rsi = ta.momentum.rsi(pd.Series(close), window=14).values
    rsi_val = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
    if rsi_val < 70:
        return result

    # Volume spike on recent candles (blow-off top)
    vol = df["Volume"].values
    vol_avg = np.mean(vol[-20:])
    vol_recent = np.mean(vol[-3:])
    vol_spike = vol_recent / vol_avg if vol_avg > 0 else 1.0

    # Recent candle showing exhaustion (upper wick > body)
    last_open  = df["Open"].values[-1]
    last_close = close[-1]
    last_high  = high[-1]
    body = abs(last_close - last_open)
    upper_wick = last_high - max(last_close, last_open)
    exhaustion = upper_wick > body * 0.5

    # Grade scoring
    score = 0
    max_score = 6

    if z_score >= 3.0:
        score += 2
    elif z_score >= 2.5:
        score += 1.5
    else:
        score += 1

    if rsi_val >= 80:
        score += 1.5
    elif rsi_val >= 75:
        score += 1

    if vol_spike > 2.0:
        score += 1.5
    elif vol_spike > 1.5:
        score += 1

    if exhaustion:
        score += 1

    if score < 3.0:
        return result

    stop = max(high[-5:]) * 1.01       # Stop above recent 5-day high
    target = sma50                      # Revert to 50 SMA
    rr = (price - target) / (stop - price) if stop > price else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "z_score": round(z_score, 2),
            "sma50": round(sma50, 2),
            "rsi": round(rsi_val, 1),
            "vol_spike": round(vol_spike, 2),
            "exhaustion_candle": bool(exhaustion),
            "pct_above_sma50": round((price - sma50) / sma50 * 100, 1),
        },
        "entry_zone": [round(price * 0.99, 2), round(price, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_close_to_bottom(df):
    """
    Close to Bottom / Reversal Setup.
    Within 10-15% of 52-week low with early signs of reversal.
    """
    result = {"pattern": "Close to Bottom", "detected": False, "direction": "LONG"}

    if len(df) < 60:
        return result

    close  = df["Close"].values
    low    = df["Low"].values
    volume = df["Volume"].values
    price  = close[-1]

    low_52w = min(low[-252:]) if len(df) >= 252 else min(low)
    high_52w = max(df["High"].values[-252:]) if len(df) >= 252 else max(df["High"].values)

    dist_from_low = (price - low_52w) / low_52w * 100

    # Must be within 15% of 52-week low
    if dist_from_low > 15:
        return result

    # Must not already be at the low (need a small bounce)
    if dist_from_low < 0.5:
        return result

    # RSI must be oversold or recovering from oversold
    rsi = ta.momentum.rsi(pd.Series(close), window=14).values
    rsi_val = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
    if rsi_val > 45:
        return result

    # Volume: recent pickup vs prior week (accumulation signal)
    vol_recent = np.mean(volume[-5:])
    vol_prior  = np.mean(volume[-20:-5])
    vol_ratio  = vol_recent / vol_prior if vol_prior > 0 else 1.0

    # Price action: recent candle is up day (close > open)
    up_day = close[-1] > df["Open"].values[-1]

    # Grade scoring
    score = 0
    max_score = 6

    # Proximity to low (closer = higher urgency)
    if dist_from_low < 5:
        score += 2
    elif dist_from_low < 10:
        score += 1.5
    else:
        score += 1

    # RSI deeply oversold
    if rsi_val < 25:
        score += 1.5
    elif rsi_val < 35:
        score += 1

    # Volume pickup (accumulation)
    if vol_ratio > 1.5:
        score += 1.5
    elif vol_ratio > 1.2:
        score += 0.75

    # Up day candle
    if up_day:
        score += 0.5

    if score < 3.0:
        return result

    stop = low_52w * 0.97
    target = price * 1.15   # 15% recovery target
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "low_52w": round(low_52w, 2),
            "dist_from_52w_low_pct": round(dist_from_low, 1),
            "rsi": round(rsi_val, 1),
            "vol_ratio": round(vol_ratio, 2),
            "up_day": bool(up_day),
        },
        "entry_zone": [round(price, 2), round(price * 1.02, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_bx_momentum(df):
    """
    BX Momentum Setup.
    ADX >25 confirming trend strength, EMA stack aligned, momentum building.
    """
    result = {"pattern": "BX Momentum", "detected": False, "direction": "LONG"}

    if len(df) < 50:
        return result

    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    volume = df["Volume"].values
    price  = close[-1]

    # EMA stack: 8 > 21 > 50
    ema8  = pd.Series(close).ewm(span=8,  adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values

    stack_aligned = price > ema8[-1] > ema21[-1] > ema50[-1]
    if not stack_aligned:
        # Check SHORT: price < ema8 < ema21 < ema50
        short_stack = price < ema8[-1] < ema21[-1] < ema50[-1]
        if short_stack:
            result["direction"] = "SHORT"
        else:
            return result

    direction = result["direction"]

    # ADX > 25 (strong trend)
    adx_series = ta.trend.ADXIndicator(
        pd.Series(high), pd.Series(low), pd.Series(close), window=14
    )
    adx_val = float(adx_series.adx().iloc[-1])
    if np.isnan(adx_val) or adx_val < 20:
        return result

    # RSI momentum: 50-70 for long, 30-50 for short
    rsi = ta.momentum.rsi(pd.Series(close), window=14).values
    rsi_val = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50

    if direction == "LONG" and not (45 <= rsi_val <= 75):
        return result
    if direction == "SHORT" and not (25 <= rsi_val <= 55):
        return result

    # Volume above average (momentum confirmation)
    vol_avg = np.mean(volume[-20:])
    vol_recent = np.mean(volume[-3:])
    vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1.0

    # Grade scoring
    score = 0
    max_score = 6

    # EMA stack quality
    score += 1.5  # Stack aligned

    # ADX strength
    if adx_val >= 35:
        score += 2
    elif adx_val >= 28:
        score += 1.5
    else:
        score += 1

    # RSI in trend zone (not overbought/oversold)
    score += 0.5

    # Volume confirmation
    if vol_ratio > 1.3:
        score += 1.5
    elif vol_ratio > 1.1:
        score += 0.75

    if score < 3.0:
        return result

    if direction == "LONG":
        stop = ema21[-1] * 0.98
        target = price * 1.12
    else:
        stop = ema21[-1] * 1.02
        target = price * 0.88

    rr = abs(target - price) / abs(price - stop) if abs(price - stop) > 0 else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "adx": round(adx_val, 1),
            "ema8": round(float(ema8[-1]), 2),
            "ema21": round(float(ema21[-1]), 2),
            "ema50": round(float(ema50[-1]), 2),
            "rsi": round(rsi_val, 1),
            "vol_ratio": round(vol_ratio, 2),
            "stack_aligned": bool(stack_aligned),
        },
        "entry_zone": [round(price * 0.99, 2), round(price * 1.005, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


def detect_earnings_gap(df):
    """
    Earnings Gap-Up Setup.
    Large gap (>4%) on above-average volume — buyable if holding above gap open.
    """
    result = {"pattern": "Earnings Gap", "detected": False, "direction": "LONG"}

    if len(df) < 10:
        return result

    close  = df["Close"].values
    open_  = df["Open"].values
    volume = df["Volume"].values
    price  = close[-1]

    # Look for a large gap in the last 5 bars
    vol_avg_20 = np.mean(volume[-20:])
    best_gap = None

    for i in range(-5, 0):
        gap_pct = (open_[i] - close[i - 1]) / close[i - 1] * 100
        if gap_pct >= 4.0 and volume[i] > vol_avg_20 * 1.5:
            if best_gap is None or abs(gap_pct) > abs(best_gap["gap_pct"]):
                best_gap = {
                    "bar_idx": i,
                    "gap_pct": gap_pct,
                    "gap_open": float(open_[i]),
                    "prev_close": float(close[i - 1]),
                    "vol_surge": float(volume[i] / vol_avg_20),
                }

    if best_gap is None:
        return result

    # Price must still be holding above the gap open (not failed)
    if price < best_gap["gap_open"] * 0.98:
        return result

    # Grade scoring
    score = 0
    max_score = 6

    if best_gap["gap_pct"] >= 10:
        score += 2
    elif best_gap["gap_pct"] >= 7:
        score += 1.5
    else:
        score += 1

    if best_gap["vol_surge"] >= 3.0:
        score += 2
    elif best_gap["vol_surge"] >= 2.0:
        score += 1.5
    else:
        score += 1

    # Price holding well above gap open
    hold_pct = (price - best_gap["gap_open"]) / best_gap["gap_open"] * 100
    if hold_pct >= 2:
        score += 1
    elif hold_pct >= 0:
        score += 0.5

    if score < 3.0:
        return result

    stop = best_gap["gap_open"] * 0.97
    target = price * 1.12
    rr = (target - price) / (price - stop) if price > stop else 0

    result.update({
        "detected": True,
        "grade": _grade_score(score, max_score),
        "details": {
            "gap_pct": round(best_gap["gap_pct"], 1),
            "gap_open": round(best_gap["gap_open"], 2),
            "prev_close": round(best_gap["prev_close"], 2),
            "vol_surge": round(best_gap["vol_surge"], 2),
            "price_vs_gap_open_pct": round(hold_pct, 1),
        },
        "entry_zone": [round(best_gap["gap_open"], 2), round(price * 1.01, 2)],
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_reward": round(rr, 1),
    })
    return result


# ---------------------------------------------------------------------------
# Pattern Registry & Scanner
# ---------------------------------------------------------------------------

PATTERN_REGISTRY = {
    # Phase 1 — Core Long Setups
    "VCP":               detect_vcp,
    "Bull Flag":         detect_bull_flag,
    "New Uptrend":       detect_new_uptrend,
    "Golden Pocket":     detect_golden_pocket,
    "Livermore Breakout": detect_livermore_breakout,
    # Phase 2 — Expanded Coverage
    "Stage 1 Base":      detect_stage1_base,
    "Parabolic Short":   detect_parabolic_short,
    "Close to Bottom":   detect_close_to_bottom,
    "BX Momentum":       detect_bx_momentum,
    "Earnings Gap":      detect_earnings_gap,
}


def scan_patterns(symbol, df=None, patterns=None):
    """
    Run selected patterns on one ticker.
    Returns list of detected patterns with grades.
    """
    if df is None:
        try:
            tk = yf.Ticker(symbol)
            df = tk.history(period="1y")
            if df.empty:
                return []
        except Exception:
            return []

    if len(df) < 30:
        return []

    price = float(df["Close"].iloc[-1])
    detectors = patterns or PATTERN_REGISTRY

    results = []
    for name, func in (detectors.items() if isinstance(detectors, dict) else
                        [(k, PATTERN_REGISTRY[k]) for k in detectors if k in PATTERN_REGISTRY]):
        try:
            r = func(df)
            if r.get("detected"):
                r["symbol"] = symbol
                r["price"] = round(price, 2)
                results.append(r)
        except Exception:
            continue

    return results


def scan_universe(symbols, patterns=None, min_grade="B", max_workers=8):
    """
    Scan multiple tickers for patterns.
    Uses batch yfinance download for performance.
    Returns detected patterns sorted by grade.
    """
    results = []

    # Batch download data
    batch_size = 100
    all_data = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            data = yf.download(batch, period="1y", group_by="ticker",
                               progress=False, threads=True)
            if data is not None and not data.empty:
                if len(batch) == 1:
                    # Single ticker: data is not grouped
                    sym = batch[0]
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    all_data[sym] = data
                else:
                    for sym in batch:
                        try:
                            ticker_data = data[sym].dropna(how="all")
                            if not ticker_data.empty and len(ticker_data) >= 30:
                                all_data[sym] = ticker_data
                        except (KeyError, Exception):
                            continue
        except Exception:
            continue

    # Parallel pattern detection
    grade_order = {"A+": 0, "A": 1, "B+": 2, "B": 3}

    def _scan_one(sym, df):
        return scan_patterns(sym, df, patterns)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_scan_one, sym, df): sym
                   for sym, df in all_data.items()}

        for future in as_completed(futures):
            try:
                detected = future.result()
                for r in detected:
                    if grade_order.get(r.get("grade"), 99) <= grade_order.get(min_grade, 3):
                        results.append(r)
            except Exception:
                continue

    # Sort by grade (A+ first), then by risk/reward
    results.sort(key=lambda r: (
        grade_order.get(r.get("grade"), 99),
        -r.get("risk_reward", 0),
    ))

    return results
