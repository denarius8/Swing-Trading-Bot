"""
Confluence-Based Swing Trade Signal Engine
Scans any ticker for alignment of 10 high-conviction indicators.
Only fires ENTER LONG / ENTER SHORT when 7+ indicators align.

Also includes a parallel Reversal Entry panel (6 indicators) that
fires independently of the trend system — catches capitulation bottoms
and overbought tops that the trend system misses.
"""

import os
import json
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime
from net_premium import fetch_net_premium_signal


# Default watchlist for scanner
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META",
    "GOOG", "AMD", "NFLX", "AVGO", "CRM", "ORCL", "COST",
    "V", "JPM", "UNH", "MA", "HD",
    "XOM", "LLY", "ABBV", "PG", "MRK",
    "COIN", "MARA", "PLTR", "SOFI", "RIVN",
    "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT",
]

CONFLUENCE_THRESHOLD = 8   # Need 8+ out of 12 for ENTER signal (67% bar)
STAY_THRESHOLD       = 6   # 6-7/12 = STAY LONG/SHORT (trend intact, not fresh entry)
LEAN_THRESHOLD       = 5   # 5/12 = LEAN (informational only)


def _fetch_ticker_data(symbol, period="6mo"):
    """Fetch OHLCV data for any ticker."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    except Exception:
        return None


def _calculate_indicators(df):
    """Calculate all confluence indicators on a dataframe."""
    d = {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Current values
    price = float(close.iloc[-1])
    d["price"] = price
    d["prev_close"] = float(close.iloc[-2])

    # --- 1. RSI ---
    rsi = ta.momentum.rsi(close, window=14)
    d["rsi"] = float(rsi.iloc[-1])
    # Check for RSI divergence (price making new low but RSI making higher low)
    if len(close) >= 20:
        price_low5 = close.iloc[-5:].min()
        price_low20 = close.iloc[-20:-5].min()
        rsi_low5 = rsi.iloc[-5:].min()
        rsi_low20 = rsi.iloc[-20:-5].min()
        d["rsi_bull_divergence"] = (price_low5 < price_low20) and (rsi_low5 > rsi_low20)
        price_high5 = close.iloc[-5:].max()
        price_high20 = close.iloc[-20:-5].max()
        rsi_high5 = rsi.iloc[-5:].max()
        rsi_high20 = rsi.iloc[-20:-5].max()
        d["rsi_bear_divergence"] = (price_high5 > price_high20) and (rsi_high5 < rsi_high20)
    else:
        d["rsi_bull_divergence"] = False
        d["rsi_bear_divergence"] = False

    # --- 2. VWAP (approx using cumulative typical price * volume) ---
    typical_price = (high + low + close) / 3
    # Use last 20 bars as "session" for daily data
    tp_vol = (typical_price * volume).rolling(20).sum()
    vol_sum = volume.rolling(20).sum()
    vwap = tp_vol / vol_sum
    d["vwap"] = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else price
    d["price_vs_vwap"] = price - d["vwap"]

    # --- 3. EMA Stack (9, 21, 50) ---
    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    ema50 = close.ewm(span=50).mean()
    d["ema9"] = float(ema9.iloc[-1])
    d["ema21"] = float(ema21.iloc[-1])
    d["ema50"] = float(ema50.iloc[-1])
    d["ema_bull_stack"] = d["ema9"] > d["ema21"] > d["ema50"]
    d["ema_bear_stack"] = d["ema9"] < d["ema21"] < d["ema50"]

    # --- 4. MACD Crossover ---
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()
    d["macd"] = float(macd_line.iloc[-1])
    d["macd_signal"] = float(macd_signal.iloc[-1])
    d["macd_hist"] = float(macd_hist.iloc[-1])
    d["macd_hist_prev"] = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0
    # Fresh crossover: MACD crossed signal in last 3 bars
    d["macd_bull_cross"] = (macd_line.iloc[-1] > macd_signal.iloc[-1]) and (macd_line.iloc[-3] < macd_signal.iloc[-3])
    d["macd_bear_cross"] = (macd_line.iloc[-1] < macd_signal.iloc[-1]) and (macd_line.iloc[-3] > macd_signal.iloc[-3])
    d["macd_hist_expanding"] = abs(d["macd_hist"]) > abs(d["macd_hist_prev"])

    # --- 5. Bollinger Band Squeeze & Expansion ---
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    bb_pct = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    d["bb_width"] = float(bb_width.iloc[-1])
    d["bb_width_prev"] = float(bb_width.iloc[-5]) if len(bb_width) >= 5 else d["bb_width"]
    d["bb_pct"] = float(bb_pct.iloc[-1])
    d["bb_squeeze"] = d["bb_width"] < d["bb_width_prev"] * 0.8  # Width contracted 20%+
    d["bb_expanding"] = d["bb_width"] > d["bb_width_prev"]

    # --- 6. Volume Confirmation ---
    vol_sma20 = volume.rolling(20).mean()
    d["vol_ratio"] = float(volume.iloc[-1] / vol_sma20.iloc[-1]) if vol_sma20.iloc[-1] > 0 else 1.0
    d["vol_above_avg"] = d["vol_ratio"] > 1.2

    # --- 7. Key Level Test (20-day high/low) ---
    high_20 = float(high.rolling(20).max().iloc[-1])
    low_20 = float(low.rolling(20).min().iloc[-1])
    d["high_20"] = high_20
    d["low_20"] = low_20
    d["near_20d_high"] = (high_20 - price) / price < 0.01  # Within 1%
    d["near_20d_low"] = (price - low_20) / price < 0.01

    # --- 8. ADX Trend Strength ---
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    d["adx"] = float(adx_ind.adx().iloc[-1])
    d["adx_pos"] = float(adx_ind.adx_pos().iloc[-1])
    d["adx_neg"] = float(adx_ind.adx_neg().iloc[-1])
    d["adx_trending"] = d["adx"] > 25
    d["adx_bull"] = d["adx_pos"] > d["adx_neg"]

    # --- 9. Stochastic Crossover in Extreme Zones ---
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    d["stoch_k"] = float(stoch_k.iloc[-1])
    d["stoch_d"] = float(stoch_d.iloc[-1])
    d["stoch_bull_cross"] = (d["stoch_k"] > d["stoch_d"]) and (d["stoch_k"] < 30)
    d["stoch_bear_cross"] = (d["stoch_k"] < d["stoch_d"]) and (d["stoch_k"] > 70)

    # --- 10. Trend Direction (price vs key MAs) ---
    sma200 = close.rolling(200).mean()
    d["sma200"] = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else price
    d["above_200sma"] = price > d["sma200"]

    # 50 SMA for trend context
    sma50 = close.rolling(50).mean()
    d["sma50"] = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else price
    d["above_50sma"] = price > d["sma50"]

    # Recovery detection — RSI was at extreme within last 15 bars, now recovering
    rsi_window = rsi.iloc[-15:] if len(rsi) >= 15 else rsi
    d["rsi_recovery_bullish"] = bool(rsi_window.min() < 32) and float(rsi.iloc[-1]) > 50
    d["rsi_recovery_bearish"] = bool(rsi_window.max() > 68) and float(rsi.iloc[-1]) < 50

    # Trend context flags (used by context-aware scoring)
    d["confirmed_uptrend"]   = (d["above_200sma"] and d["above_50sma"]
                                and d["adx_trending"] and d["adx_bull"])
    d["confirmed_downtrend"] = (not d["above_200sma"] and not d["above_50sma"]
                                and d["adx_trending"] and not d["adx_bull"])

    # Extra context
    d["change_1d"] = (price - d["prev_close"]) / d["prev_close"]
    ret_5d = close.pct_change(5)
    d["change_5d"] = float(ret_5d.iloc[-1]) if not np.isnan(ret_5d.iloc[-1]) else 0

    # Where today's close sits within the day's high-low range (0 = at low, 1 = at high)
    # Used by the reversal candle detector
    day_high = float(high.iloc[-1])
    day_low  = float(low.iloc[-1])
    day_range = day_high - day_low
    d["close_range_pct"] = (price - day_low) / day_range if day_range > 0 else 0.5
    d["day_high"] = day_high
    d["day_low"]  = day_low

    return d


# ─── Reversal Entry Helpers ───────────────────────────────────────────────────

def _get_vix():
    """Fetch current VIX closing level."""
    try:
        vix = yf.Ticker("^VIX").history(period="3d")
        if not vix.empty:
            return round(float(vix["Close"].iloc[-1]), 1)
    except Exception:
        pass
    return None


def _get_net_premium_data(index='SPX'):
    """
    Read the last two net premium entries from cache.
    Returns (latest_np, prev_np, streak_direction) or (None, None, None).
    Does NOT trigger a live calculation — reads cache only (fast).
    """
    try:
        cache_file = "ndx_net_premium.json" if index == 'NDX' else "net_premium.json"
        cache_path = os.path.join("cache", cache_file)
        if not os.path.exists(cache_path):
            return None, None, None
        with open(cache_path) as f:
            data = json.load(f)
        history = data.get("history", [])
        if not history:
            return None, None, None

        latest_np = history[0].get("net_premium")
        prev_np   = history[1].get("net_premium") if len(history) >= 2 else None

        # Streak direction (how many consecutive days in the same sign)
        streak_dir = None
        for entry in history:
            np_val = entry.get("net_premium")
            if np_val is None:
                break
            sign = "positive" if np_val > 0 else "negative"
            if streak_dir is None:
                streak_dir = sign
            elif sign != streak_dir:
                break

        return latest_np, prev_np, streak_dir
    except Exception:
        return None, None, None


# ─── Reversal Entry Scoring ───────────────────────────────────────────────────

REVERSAL_THRESHOLD = 4   # Need 4/6 indicators for a reversal signal


def score_reversal(indicators, vix=None, net_premium_data=None):
    """
    Score 6 reversal-specific indicators that fire independently of the
    main 10-indicator trend system.

    Designed to catch:
      - Capitulation bottoms (all trend indicators still bearish at the low)
      - Overbought tops (all trend indicators still bullish at the high)

    Returns dict with scores, signal, and regime classification.
    """
    price  = indicators["price"]
    sma200 = indicators["sma200"]
    rsi    = indicators["rsi"]

    # ── Regime Detection ─────────────────────────────────────────────
    below_200_pct = (sma200 - price) / sma200 * 100 if sma200 > 0 else 0
    above_200_pct = (price - sma200) / sma200 * 100 if sma200 > 0 else 0

    vix_extreme  = vix is not None and vix > 35
    vix_elevated = vix is not None and vix > 28
    price_oversold    = below_200_pct > 7
    price_overbought  = above_200_pct > 10
    rsi_extreme_low   = rsi < 35
    rsi_extreme_high  = rsi > 65

    if vix_extreme or below_200_pct > 15:
        regime = "EXTREME"
        regime_class = "regime-extreme"
        regime_note = "Crisis-level readings — reversal entries highest priority"
    elif vix_elevated or price_oversold or price_overbought or rsi_extreme_low or rsi_extreme_high:
        regime = "ELEVATED"
        regime_class = "regime-elevated"
        regime_note = "Extended market — reversal signals are actionable"
    else:
        regime = "NORMAL"
        regime_class = "regime-normal"
        regime_note = "Normal market — reversal signals are informational"

    scores = {}

    # ── 1. RSI Extreme (broader zone: <35 LONG, >65 SHORT) ───────────
    if rsi < 35:
        scores["rsi_extreme"] = {
            "score": 1,
            "label": "RSI Extreme Oversold",
            "detail": f"RSI: {rsi:.1f} — below 35 (deeply oversold)",
            "reason": "Broad oversold zone signals exhausted selling — reversal probability high",
        }
    elif rsi > 65:
        scores["rsi_extreme"] = {
            "score": -1,
            "label": "RSI Extreme Overbought",
            "detail": f"RSI: {rsi:.1f} — above 65 (deeply overbought)",
            "reason": "Broad overbought zone signals exhausted buying — pullback probability high",
        }
    else:
        scores["rsi_extreme"] = {
            "score": 0,
            "label": "RSI Neutral Zone",
            "detail": f"RSI: {rsi:.1f} (35–65 range)",
            "reason": "RSI not at an extreme — no reversal pressure from momentum",
        }

    # ── 2. Stochastic Extreme (broader: <25 LONG, >75 SHORT) ─────────
    sk = indicators["stoch_k"]
    sd = indicators["stoch_d"]
    if sk < 25:
        scores["stoch_extreme"] = {
            "score": 1,
            "label": "Stochastic Deep Oversold",
            "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — below 25",
            "reason": "Stochastic at extreme oversold — sellers exhausted, bounce typically follows",
        }
    elif sk > 75:
        scores["stoch_extreme"] = {
            "score": -1,
            "label": "Stochastic Deep Overbought",
            "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — above 75",
            "reason": "Stochastic at extreme overbought — buyers exhausted, fade setup",
        }
    else:
        scores["stoch_extreme"] = {
            "score": 0,
            "label": "Stochastic Neutral",
            "detail": f"%K: {sk:.1f} / %D: {sd:.1f}",
            "reason": "Stochastic in neutral zone — no extreme reading",
        }

    # ── 3. Bollinger Extreme (NO expansion check — fixed crash bias) ──
    bb_pct = indicators["bb_pct"]
    if bb_pct < 0.15:
        scores["bb_extreme"] = {
            "score": 1,
            "label": "At/Below Lower Bollinger Band",
            "detail": f"BB%: {bb_pct:.1%} — at or below lower band",
            "reason": "Price at statistical extreme low — mean reversion expected regardless of trend",
        }
    elif bb_pct > 0.85:
        scores["bb_extreme"] = {
            "score": -1,
            "label": "At/Above Upper Bollinger Band",
            "detail": f"BB%: {bb_pct:.1%} — at or above upper band",
            "reason": "Price at statistical extreme high — mean reversion expected regardless of trend",
        }
    else:
        scores["bb_extreme"] = {
            "score": 0,
            "label": "BB Mid-Range",
            "detail": f"BB%: {bb_pct:.1%}",
            "reason": "Price within Bollinger Bands — no statistical extreme",
        }

    # ── 4. Net Premium (institutional options flow) ───────────────────
    if net_premium_data:
        latest_np, prev_np, streak_dir = net_premium_data
    else:
        latest_np = prev_np = streak_dir = None

    if latest_np is not None:
        np_b = latest_np / 1e9
        reversed_pos = prev_np is not None and prev_np < 0 and latest_np > 0
        reversed_neg = prev_np is not None and prev_np > 0 and latest_np < 0

        if latest_np > 0:
            reversal_tag = " — Reversed from negative!" if reversed_pos else ""
            scores["net_premium"] = {
                "score": 1,
                "label": f"Net Premium Positive{reversal_tag}",
                "detail": f"NP: ${np_b:.2f}B ({streak_dir} streak)",
                "reason": "Positive net premium = call buying dominant — institutional money is long",
            }
        else:
            reversal_tag = " — Reversed from positive!" if reversed_neg else ""
            scores["net_premium"] = {
                "score": -1,
                "label": f"Net Premium Negative{reversal_tag}",
                "detail": f"NP: ${np_b:.2f}B ({streak_dir} streak)",
                "reason": "Negative net premium = put buying dominant — institutional money is hedging/short",
            }
    else:
        scores["net_premium"] = {
            "score": 0,
            "label": "Net Premium — No Data",
            "detail": "Load Confluence tab → Net Premium section to populate",
            "reason": "No net premium history in cache. Run /api/net-premium to calculate.",
        }

    # ── 5. VIX Regime ─────────────────────────────────────────────────
    if vix is not None:
        if vix > 35:
            scores["vix_regime"] = {
                "score": 1,
                "label": f"VIX Panic Zone ({vix:.1f})",
                "detail": f"VIX: {vix:.1f} — historically marks generational buy zones",
                "reason": "VIX above 35 = extreme fear/panic. Historically within days of major bottoms",
            }
        elif vix > 28:
            scores["vix_regime"] = {
                "score": 1,
                "label": f"VIX Elevated ({vix:.1f})",
                "detail": f"VIX: {vix:.1f} — above 28 = capitulation zone",
                "reason": "Elevated VIX signals market fear — contrarian long setup when price stabilizes",
            }
        elif vix < 14:
            scores["vix_regime"] = {
                "score": -1,
                "label": f"VIX Complacency ({vix:.1f})",
                "detail": f"VIX: {vix:.1f} — below 14 = extreme complacency",
                "reason": "Very low VIX = no fear priced in — markets vulnerable, fade extended rallies",
            }
        else:
            scores["vix_regime"] = {
                "score": 0,
                "label": f"VIX Normal ({vix:.1f})",
                "detail": f"VIX: {vix:.1f} — normal range (14–28)",
                "reason": "VIX in normal range — no extreme fear or complacency signal",
            }
    else:
        scores["vix_regime"] = {
            "score": 0,
            "label": "VIX — Unavailable",
            "detail": "Could not fetch VIX",
            "reason": "VIX data unavailable",
        }

    # ── 6. Volume Reversal Candle ─────────────────────────────────────
    vol_ratio  = indicators["vol_ratio"]
    chg1d      = indicators["change_1d"]
    close_rng  = indicators.get("close_range_pct", 0.5)

    # Bullish reversal: elevated vol + up day + strong close (top 40% of range)
    if vol_ratio > 1.1 and chg1d > 0 and close_rng >= 0.40:
        scores["reversal_candle"] = {
            "score": 1,
            "label": "Bullish Reversal Candle",
            "detail": f"Vol {vol_ratio:.1f}× | +{chg1d*100:.1f}% | closed {close_rng:.0%} up in range",
            "reason": "High-volume up day closing strong — buyers absorbed all sellers, reversal confirmed",
        }
    # Bearish reversal: elevated vol + down day + weak close (bottom 40% of range)
    elif vol_ratio > 1.1 and chg1d < 0 and close_rng <= 0.40:
        scores["reversal_candle"] = {
            "score": -1,
            "label": "Bearish Reversal Candle",
            "detail": f"Vol {vol_ratio:.1f}× | {chg1d*100:.1f}% | closed {close_rng:.0%} in range",
            "reason": "High-volume down day closing weak — sellers overwhelmed all buyers, breakdown confirmed",
        }
    elif vol_ratio > 1.3:
        scores["reversal_candle"] = {
            "score": 0,
            "label": "High Volume — No Clear Close",
            "detail": f"Vol {vol_ratio:.1f}× | {chg1d*100:+.1f}% | closed {close_rng:.0%} in range",
            "reason": "Elevated volume but close mid-range — indecision. Watch next candle for direction",
        }
    else:
        scores["reversal_candle"] = {
            "score": 0,
            "label": "Normal Volume — No Reversal Candle",
            "detail": f"Vol {vol_ratio:.1f}× | {chg1d*100:+.1f}%",
            "reason": "Volume not elevated enough to confirm institutional reversal activity",
        }

    # ── Totals & Signal ───────────────────────────────────────────────
    long_count    = sum(1 for s in scores.values() if s["score"] == 1)
    short_count   = sum(1 for s in scores.values() if s["score"] == -1)
    neutral_count = sum(1 for s in scores.values() if s["score"] == 0)

    if long_count >= REVERSAL_THRESHOLD:
        signal = "ENTER LONG"
        signal_class = "enter-long"
    elif short_count >= REVERSAL_THRESHOLD:
        signal = "ENTER SHORT"
        signal_class = "enter-short"
    elif long_count == 3:
        signal = "WATCH LONG"
        signal_class = "lean-bull"
    elif short_count == 3:
        signal = "WATCH SHORT"
        signal_class = "lean-bear"
    else:
        signal = "NO SIGNAL"
        signal_class = "no-signal"

    return {
        "scores": scores,
        "long_count":    long_count,
        "short_count":   short_count,
        "neutral_count": neutral_count,
        "signal":        signal,
        "signal_class":  signal_class,
        "regime":        regime,
        "regime_class":  regime_class,
        "regime_note":   regime_note,
        "vix":           vix,
        "threshold":     REVERSAL_THRESHOLD,
    }


def get_fast_pullback_alert(indicators, vix=None, vix_prev=None, gex_signal=None, np_signal=None):
    """
    Fast Pullback Alert — fires at 4+ fear triggers before 8/12 confluence confirms.
    Selloffs are faster than rallies; this catches them before full confirmation.

    Symmetric: also detects fast breakout / capitulation bottom conditions.
    Returns dict with triggers, count, and alert level.
    """
    triggers = {}
    price  = indicators["price"]
    sma50  = indicators.get("sma50", price)

    # ── 1. VIX Spike (fear entering fast) ────────────────────────────
    if vix is not None and vix_prev is not None and vix_prev > 0:
        vix_chg = (vix - vix_prev) / vix_prev
        if vix_chg >= 0.15:
            triggers["vix_spike"] = {
                "triggered": True, "direction": "bearish",
                "label": f"VIX Spike +{vix_chg*100:.0f}%",
                "detail": f"VIX: {vix_prev:.1f} → {vix:.1f} — fear entering market fast",
            }
        elif vix_chg <= -0.15:
            triggers["vix_spike"] = {
                "triggered": True, "direction": "bullish",
                "label": f"VIX Collapse -{abs(vix_chg)*100:.0f}%",
                "detail": f"VIX: {vix_prev:.1f} → {vix:.1f} — fear leaving market fast",
            }
        else:
            triggers["vix_spike"] = {"triggered": False, "direction": "neutral",
                                      "label": f"VIX Stable ({vix:.1f})", "detail": ""}
    else:
        triggers["vix_spike"] = {"triggered": False, "direction": "neutral",
                                  "label": "VIX — Unavailable", "detail": ""}

    # ── 2. EMA 9 crossed EMA 21 ───────────────────────────────────────
    ema9  = indicators["ema9"]
    ema21 = indicators["ema21"]
    if indicators["ema_bear_stack"] and not indicators.get("prev_ema_bear", False):
        triggers["ema_cross"] = {
            "triggered": True, "direction": "bearish",
            "label": "EMA 9 Crossed Below 21",
            "detail": f"EMA9 ({ema9:.1f}) < EMA21 ({ema21:.1f}) — short-term momentum flip",
        }
    elif indicators["ema_bull_stack"] and not indicators.get("prev_ema_bull", False):
        triggers["ema_cross"] = {
            "triggered": True, "direction": "bullish",
            "label": "EMA 9 Crossed Above 21",
            "detail": f"EMA9 ({ema9:.1f}) > EMA21 ({ema21:.1f}) — short-term momentum flip",
        }
    else:
        triggers["ema_cross"] = {"triggered": False, "direction": "neutral",
                                  "label": f"EMA Stable (9:{ema9:.0f} / 21:{ema21:.0f})", "detail": ""}

    # ── 3. Price broke above/below 50 SMA on volume ───────────────────
    if indicators["vol_above_avg"]:
        if price < sma50 and indicators.get("above_50sma_prev", True):
            triggers["sma50_break"] = {
                "triggered": True, "direction": "bearish",
                "label": "Broke Below 50 SMA on Volume",
                "detail": f"Price {price:.2f} < SMA50 {sma50:.2f} on {indicators['vol_ratio']:.1f}x avg vol",
            }
        elif price > sma50 and not indicators.get("above_50sma_prev", True):
            triggers["sma50_break"] = {
                "triggered": True, "direction": "bullish",
                "label": "Reclaimed 50 SMA on Volume",
                "detail": f"Price {price:.2f} > SMA50 {sma50:.2f} on {indicators['vol_ratio']:.1f}x avg vol",
            }
        else:
            triggers["sma50_break"] = {"triggered": False, "direction": "neutral",
                                        "label": f"50 SMA Stable ({sma50:.0f})", "detail": ""}
    else:
        triggers["sma50_break"] = {"triggered": False, "direction": "neutral",
                                    "label": f"50 SMA — Low Volume", "detail": ""}

    # ── 4. GEX Flip Event ─────────────────────────────────────────────
    if gex_signal and gex_signal.get("flip_event"):
        flip_dir = gex_signal.get("flip_direction", "")
        direction = "bullish" if flip_dir == "LONG" else "bearish"
        triggers["gex_flip"] = {
            "triggered": True, "direction": direction,
            "label": f"GEX Flip {flip_dir} ⚡",
            "detail": f"Dealers flipped to {'LONG' if direction == 'bullish' else 'SHORT'} gamma — hedging mechanics reversed",
        }
    else:
        gex_regime = gex_signal.get("regime", "UNKNOWN") if gex_signal else "UNKNOWN"
        triggers["gex_flip"] = {"triggered": False, "direction": "neutral",
                                 "label": f"No GEX Flip ({gex_regime})", "detail": ""}

    # ── 5. Net Premium Flip ───────────────────────────────────────────
    if np_signal and np_signal.get("flip_event"):
        flip_dir = np_signal.get("flip_direction", "")
        direction = "bullish" if flip_dir == "positive" else "bearish"
        triggers["np_flip"] = {
            "triggered": True, "direction": direction,
            "label": f"Options Flow Flip {'↑' if direction == 'bullish' else '↓'}",
            "detail": np_signal.get("detail", "Net premium changed direction"),
        }
    else:
        triggers["np_flip"] = {"triggered": False, "direction": "neutral",
                                "label": "No Flow Flip", "detail": ""}

    # ── 6. Volume Climax (high volume day reversing) ──────────────────
    vol_ratio = indicators["vol_ratio"]
    chg1d     = indicators["change_1d"]
    close_rng = indicators.get("close_range_pct", 0.5)
    if vol_ratio > 1.5 and chg1d < -0.015 and close_rng < 0.3:
        triggers["volume_climax"] = {
            "triggered": True, "direction": "bearish",
            "label": f"Bearish Volume Climax ({vol_ratio:.1f}x)",
            "detail": f"{chg1d*100:.1f}% on {vol_ratio:.1f}x avg vol, closed weak — capitulation selling",
        }
    elif vol_ratio > 1.5 and chg1d > 0.015 and close_rng > 0.7:
        triggers["volume_climax"] = {
            "triggered": True, "direction": "bullish",
            "label": f"Bullish Volume Climax ({vol_ratio:.1f}x)",
            "detail": f"+{chg1d*100:.1f}% on {vol_ratio:.1f}x avg vol, closed strong — capitulation buying",
        }
    else:
        triggers["volume_climax"] = {"triggered": False, "direction": "neutral",
                                      "label": f"Volume Normal ({vol_ratio:.1f}x)", "detail": ""}

    # ── Tally ─────────────────────────────────────────────────────────
    active    = [t for t in triggers.values() if t["triggered"]]
    bearish_n = sum(1 for t in active if t["direction"] == "bearish")
    bullish_n = sum(1 for t in active if t["direction"] == "bullish")
    total_n   = len(active)

    ALERT_THRESHOLD = 3  # 3+ triggers = alert (fast moves don't always hit 4)

    if bearish_n >= ALERT_THRESHOLD:
        alert_level = "FAST PULLBACK ALERT"
        alert_class = "fast-pullback"
        alert_dir   = "bearish"
    elif bullish_n >= ALERT_THRESHOLD:
        alert_level = "FAST BREAKOUT ALERT"
        alert_class = "fast-breakout"
        alert_dir   = "bullish"
    elif total_n >= 2:
        dominant = "bearish" if bearish_n >= bullish_n else "bullish"
        alert_level = f"{'PULLBACK' if dominant == 'bearish' else 'BREAKOUT'} WATCH"
        alert_class = "pullback-watch"
        alert_dir   = dominant
    else:
        alert_level = "NO ALERT"
        alert_class = "no-alert"
        alert_dir   = "neutral"

    return {
        "triggers":     triggers,
        "alert_level":  alert_level,
        "alert_class":  alert_class,
        "alert_dir":    alert_dir,
        "bearish_count": bearish_n,
        "bullish_count": bullish_n,
        "total_active": total_n,
        "threshold":    ALERT_THRESHOLD,
    }


def score_confluence(indicators, gex_signal=None, np_signal=None):
    """
    Score 12 confluence indicators with context-aware trend mode.

    Indicators 1-10: existing (with context-aware fixes for RSI, BB, Key Levels, Stochastic)
    Indicator 11:    GEX Regime (long/short gamma)
    Indicator 12:    Net Premium Flow (Day-1 flip + streak)

    Threshold: 8+/12 = ENTER LONG/SHORT (67% bar)
               6-7/12 = STAY LONG/SHORT (trend intact)
               5/12   = LEAN (informational)
    """
    scores = {}
    uptrend   = indicators.get("confirmed_uptrend",   False)
    downtrend = indicators.get("confirmed_downtrend", False)

    # ── 1. RSI — context-aware ────────────────────────────────────────
    rsi = indicators["rsi"]
    if uptrend:
        # In confirmed uptrend: RSI 50+ = bullish momentum; RSI < 40 = pullback entry; bear div = caution
        if rsi >= 50 and not indicators["rsi_bear_divergence"]:
            scores["rsi"] = {"score": 1, "label": "RSI Bullish (Trend Mode)",
                             "detail": f"RSI: {rsi:.1f} — above 50 in uptrend",
                             "reason": "RSI 50+ in confirmed uptrend = momentum confirmation"}
        elif rsi < 40 or indicators["rsi_bull_divergence"]:
            scores["rsi"] = {"score": 1, "label": "RSI Pullback Entry",
                             "detail": f"RSI: {rsi:.1f}" + (" + Bull Divergence" if indicators["rsi_bull_divergence"] else ""),
                             "reason": "RSI pullback in uptrend = re-entry opportunity"}
        elif indicators["rsi_bear_divergence"]:
            scores["rsi"] = {"score": -1, "label": "RSI Bearish Divergence",
                             "detail": f"RSI: {rsi:.1f} — diverging from price high",
                             "reason": "Bearish divergence in uptrend — warning signal"}
        else:
            scores["rsi"] = {"score": 0, "label": "RSI Neutral (Trend Mode)",
                             "detail": f"RSI: {rsi:.1f}", "reason": "RSI between 40-50 in uptrend"}
    elif downtrend:
        # In confirmed downtrend: RSI <= 50 = bearish; RSI > 60 = dead cat bounce
        if rsi <= 50 and not indicators["rsi_bull_divergence"]:
            scores["rsi"] = {"score": -1, "label": "RSI Bearish (Trend Mode)",
                             "detail": f"RSI: {rsi:.1f} — below 50 in downtrend",
                             "reason": "RSI below 50 in confirmed downtrend = momentum confirmation"}
        elif rsi > 60 or indicators["rsi_bear_divergence"]:
            scores["rsi"] = {"score": -1, "label": "RSI Dead Cat / Bear Div",
                             "detail": f"RSI: {rsi:.1f}" + (" + Bear Divergence" if indicators["rsi_bear_divergence"] else ""),
                             "reason": "RSI bounce in downtrend = short re-entry opportunity"}
        elif indicators["rsi_bull_divergence"]:
            scores["rsi"] = {"score": 1, "label": "RSI Bullish Divergence",
                             "detail": f"RSI: {rsi:.1f} — diverging from price low",
                             "reason": "Bullish divergence in downtrend — warning signal"}
        else:
            scores["rsi"] = {"score": 0, "label": "RSI Neutral (Trend Mode)",
                             "detail": f"RSI: {rsi:.1f}", "reason": "RSI between 50-60 in downtrend"}
    else:
        # Range-bound: traditional oversold/overbought
        if rsi < 30 or indicators["rsi_bull_divergence"]:
            scores["rsi"] = {"score": 1, "label": "RSI Oversold / Bull Divergence",
                             "detail": f"RSI: {rsi:.1f}" + (" + Bullish Divergence" if indicators["rsi_bull_divergence"] else ""),
                             "reason": "RSI below 30 signals oversold bounce potential"}
        elif rsi > 70 or indicators["rsi_bear_divergence"]:
            scores["rsi"] = {"score": -1, "label": "RSI Overbought / Bear Divergence",
                             "detail": f"RSI: {rsi:.1f}" + (" + Bearish Divergence" if indicators["rsi_bear_divergence"] else ""),
                             "reason": "RSI above 70 signals overbought pullback risk"}
        else:
            scores["rsi"] = {"score": 0, "label": "RSI Neutral",
                             "detail": f"RSI: {rsi:.1f}", "reason": "RSI in neutral zone (30-70)"}

    # ── 2. VWAP ──────────────────────────────────────────────────────
    if indicators["price_vs_vwap"] > 0:
        scores["vwap"] = {"score": 1, "label": "Above VWAP",
                          "detail": f"Price {indicators['price']:.2f} > VWAP {indicators['vwap']:.2f}",
                          "reason": "Price above VWAP = institutional buying bias"}
    else:
        scores["vwap"] = {"score": -1, "label": "Below VWAP",
                          "detail": f"Price {indicators['price']:.2f} < VWAP {indicators['vwap']:.2f}",
                          "reason": "Price below VWAP = institutional selling bias"}

    # ── 3. EMA Stack ─────────────────────────────────────────────────
    if indicators["ema_bull_stack"]:
        scores["ema_stack"] = {"score": 1, "label": "Bullish EMA Stack",
                               "detail": f"9 ({indicators['ema9']:.1f}) > 21 ({indicators['ema21']:.1f}) > 50 ({indicators['ema50']:.1f})",
                               "reason": "All EMAs aligned bullish = strong uptrend"}
    elif indicators["ema_bear_stack"]:
        scores["ema_stack"] = {"score": -1, "label": "Bearish EMA Stack",
                               "detail": f"9 ({indicators['ema9']:.1f}) < 21 ({indicators['ema21']:.1f}) < 50 ({indicators['ema50']:.1f})",
                               "reason": "All EMAs aligned bearish = strong downtrend"}
    else:
        scores["ema_stack"] = {"score": 0, "label": "EMAs Mixed",
                               "detail": "EMA 9/21/50 not fully aligned",
                               "reason": "Mixed EMAs = no clear directional trend"}

    # ── 4. MACD ───────────────────────────────────────────────────────
    if indicators["macd_hist"] > 0 and (indicators["macd_bull_cross"] or indicators["macd_hist_expanding"]):
        scores["macd"] = {"score": 1, "label": "MACD Bullish",
                          "detail": f"Hist: {indicators['macd_hist']:.2f}" + (" (Cross)" if indicators["macd_bull_cross"] else " (Expanding)"),
                          "reason": "Bullish MACD crossover or expanding positive histogram"}
    elif indicators["macd_hist"] < 0 and (indicators["macd_bear_cross"] or indicators["macd_hist_expanding"]):
        scores["macd"] = {"score": -1, "label": "MACD Bearish",
                          "detail": f"Hist: {indicators['macd_hist']:.2f}" + (" (Cross)" if indicators["macd_bear_cross"] else " (Expanding)"),
                          "reason": "Bearish MACD crossover or expanding negative histogram"}
    else:
        scores["macd"] = {"score": 0, "label": "MACD Neutral",
                          "detail": f"Hist: {indicators['macd_hist']:.2f}",
                          "reason": "No fresh crossover or histogram not expanding"}

    # ── 5. Bollinger Bands — context-aware ───────────────────────────
    bb_pct      = indicators["bb_pct"]
    bb_expand   = indicators["bb_expanding"]
    if uptrend:
        # Uptrend: riding upper band = momentum; lower band touch = pullback entry
        if bb_pct > 0.75:
            scores["bollinger"] = {"score": 1, "label": "Riding Upper Band (Trend)",
                                   "detail": f"BB%: {bb_pct:.1%} — momentum continuation",
                                   "reason": "Price at upper band in confirmed uptrend = strong momentum, not overbought"}
        elif bb_pct < 0.25:
            scores["bollinger"] = {"score": 1, "label": "Lower Band Pullback Entry",
                                   "detail": f"BB%: {bb_pct:.1%} — pullback to lower band in uptrend",
                                   "reason": "Price at lower band in uptrend = re-entry opportunity"}
        else:
            scores["bollinger"] = {"score": 0, "label": "BB Mid-Range (Trend)",
                                   "detail": f"BB%: {bb_pct:.1%}", "reason": "Price in Bollinger mid-range during uptrend"}
    elif downtrend:
        # Downtrend: riding lower band = momentum; upper band touch = dead cat/short entry
        if bb_pct < 0.25:
            scores["bollinger"] = {"score": -1, "label": "Riding Lower Band (Trend)",
                                   "detail": f"BB%: {bb_pct:.1%} — momentum breakdown",
                                   "reason": "Price at lower band in confirmed downtrend = strong selling pressure"}
        elif bb_pct > 0.75:
            scores["bollinger"] = {"score": -1, "label": "Upper Band Dead Cat",
                                   "detail": f"BB%: {bb_pct:.1%} — bounce to upper band in downtrend",
                                   "reason": "Price at upper band in downtrend = short re-entry opportunity"}
        else:
            scores["bollinger"] = {"score": 0, "label": "BB Mid-Range (Trend)",
                                   "detail": f"BB%: {bb_pct:.1%}", "reason": "Price in Bollinger mid-range during downtrend"}
    else:
        # Range: traditional mean-reversion logic
        if bb_pct > 0.8 and bb_expand:
            scores["bollinger"] = {"score": 1, "label": "BB Upper Breakout",
                                   "detail": f"BB%: {bb_pct:.1%} (Expanding)",
                                   "reason": "Price breaking above upper band with expansion = momentum breakout"}
        elif bb_pct < 0.2 and bb_expand:
            scores["bollinger"] = {"score": -1, "label": "BB Lower Breakdown",
                                   "detail": f"BB%: {bb_pct:.1%} (Expanding)",
                                   "reason": "Price breaking below lower band with expansion = momentum breakdown"}
        elif bb_pct < 0.2:
            scores["bollinger"] = {"score": 1, "label": "BB Oversold Bounce",
                                   "detail": f"BB%: {bb_pct:.1%}",
                                   "reason": "Price at lower band = mean reversion bounce likely"}
        elif bb_pct > 0.8:
            scores["bollinger"] = {"score": -1, "label": "BB Overbought Fade",
                                   "detail": f"BB%: {bb_pct:.1%}",
                                   "reason": "Price at upper band = mean reversion pullback likely"}
        else:
            scores["bollinger"] = {"score": 0, "label": "BB Neutral",
                                   "detail": f"BB%: {bb_pct:.1%}", "reason": "Price in middle of Bollinger Bands"}

    # ── 6. Volume ─────────────────────────────────────────────────────
    if indicators["vol_above_avg"] and indicators["change_1d"] > 0:
        scores["volume"] = {"score": 1, "label": "High Volume Up",
                            "detail": f"Volume {indicators['vol_ratio']:.1f}x average",
                            "reason": "Above-average volume on up move = institutional conviction"}
    elif indicators["vol_above_avg"] and indicators["change_1d"] < 0:
        scores["volume"] = {"score": -1, "label": "High Volume Down",
                            "detail": f"Volume {indicators['vol_ratio']:.1f}x average",
                            "reason": "Above-average volume on down move = institutional selling"}
    else:
        scores["volume"] = {"score": 0, "label": "Volume Normal",
                            "detail": f"Volume {indicators['vol_ratio']:.1f}x average",
                            "reason": "Volume near average = no clear conviction"}

    # ── 7. Key Levels — context-aware ────────────────────────────────
    near_high = indicators["near_20d_high"]
    near_low  = indicators["near_20d_low"]
    if near_high:
        if uptrend:
            scores["key_level"] = {"score": 1, "label": "Breakout to 20-Day High",
                                   "detail": f"At/above {indicators['high_20']:.2f} in confirmed uptrend",
                                   "reason": "New 20-day high in uptrend = breakout confirmation, not resistance"}
        else:
            scores["key_level"] = {"score": -1, "label": "Testing 20-Day Resistance",
                                   "detail": f"Near {indicators['high_20']:.2f}",
                                   "reason": "Price at key resistance = rejection risk"}
    elif near_low:
        if downtrend:
            scores["key_level"] = {"score": -1, "label": "Breakdown to 20-Day Low",
                                   "detail": f"At/below {indicators['low_20']:.2f} in confirmed downtrend",
                                   "reason": "New 20-day low in downtrend = breakdown confirmation, not support"}
        else:
            scores["key_level"] = {"score": 1, "label": "Testing 20-Day Support",
                                   "detail": f"Near {indicators['low_20']:.2f}",
                                   "reason": "Price at key support level = bounce opportunity"}
    else:
        scores["key_level"] = {"score": 0, "label": "Mid-Range",
                               "detail": f"{indicators['low_20']:.2f} – {indicators['high_20']:.2f}",
                               "reason": "Price not testing any key level"}

    # ── 8. ADX ───────────────────────────────────────────────────────
    if indicators["adx_trending"] and indicators["adx_bull"]:
        scores["adx"] = {"score": 1, "label": "Strong Bullish Trend",
                         "detail": f"ADX: {indicators['adx']:.1f} (+DI > -DI)",
                         "reason": "ADX > 25 with +DI leading = confirmed bullish trend"}
    elif indicators["adx_trending"] and not indicators["adx_bull"]:
        scores["adx"] = {"score": -1, "label": "Strong Bearish Trend",
                         "detail": f"ADX: {indicators['adx']:.1f} (-DI > +DI)",
                         "reason": "ADX > 25 with -DI leading = confirmed bearish trend"}
    else:
        scores["adx"] = {"score": 0, "label": "Weak/No Trend",
                         "detail": f"ADX: {indicators['adx']:.1f}",
                         "reason": "ADX < 25 = no strong trend in either direction"}

    # ── 9. Stochastic — context-aware ────────────────────────────────
    sk = indicators["stoch_k"]
    sd = indicators["stoch_d"]
    if uptrend:
        # Uptrend: K<30 = pullback re-entry (+1); K>80 = neutral (0, momentum in trend)
        if sk < 30:
            scores["stochastic"] = {"score": 1, "label": "Stochastic Pullback Entry (Trend)",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — oversold in uptrend",
                                    "reason": "Stochastic pullback in uptrend = re-entry opportunity, not reversal"}
        elif sk > 80:
            scores["stochastic"] = {"score": 0, "label": "Stochastic High — Trend Momentum",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — elevated in uptrend",
                                    "reason": "High stochastic in uptrend = momentum, not overbought signal"}
        else:
            scores["stochastic"] = {"score": 0, "label": "Stochastic Neutral (Trend)",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f}", "reason": "Stochastic in mid-range during uptrend"}
    elif downtrend:
        # Downtrend: K>70 = dead cat re-entry (-1); K<20 = neutral (0, momentum in trend)
        if sk > 70:
            scores["stochastic"] = {"score": -1, "label": "Stochastic Dead Cat (Trend)",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — overbought in downtrend",
                                    "reason": "High stochastic in downtrend = dead cat bounce, short re-entry"}
        elif sk < 20:
            scores["stochastic"] = {"score": 0, "label": "Stochastic Low — Trend Momentum",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f} — low in downtrend",
                                    "reason": "Low stochastic in downtrend = momentum, oversold can get worse"}
        else:
            scores["stochastic"] = {"score": 0, "label": "Stochastic Neutral (Trend)",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f}", "reason": "Stochastic in mid-range during downtrend"}
    else:
        # Range-bound: traditional crossover logic
        if indicators["stoch_bull_cross"] or sk < 20:
            scores["stochastic"] = {"score": 1, "label": "Stochastic Oversold",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f}",
                                    "reason": "Stochastic in oversold zone = bounce signal"}
        elif indicators["stoch_bear_cross"] or sk > 80:
            scores["stochastic"] = {"score": -1, "label": "Stochastic Overbought",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f}",
                                    "reason": "Stochastic in overbought zone = pullback signal"}
        else:
            scores["stochastic"] = {"score": 0, "label": "Stochastic Neutral",
                                    "detail": f"%K: {sk:.1f} / %D: {sd:.1f}",
                                    "reason": "Stochastic in neutral zone"}

    # ── 10. 200 SMA ──────────────────────────────────────────────────
    if indicators["above_200sma"]:
        scores["trend_200"] = {"score": 1, "label": "Above 200 SMA",
                               "detail": f"Price {indicators['price']:.2f} > SMA200 {indicators['sma200']:.2f}",
                               "reason": "Above 200-day MA = long-term uptrend intact"}
    else:
        scores["trend_200"] = {"score": -1, "label": "Below 200 SMA",
                               "detail": f"Price {indicators['price']:.2f} < SMA200 {indicators['sma200']:.2f}",
                               "reason": "Below 200-day MA = long-term downtrend"}

    # ── 11. GEX Regime ────────────────────────────────────────────────
    if gex_signal and gex_signal.get("signal") != 0:
        g = gex_signal
        regime = g.get("regime", "UNKNOWN")
        flip_tag = ""
        if g.get("flip_event"):
            flip_dir = g.get("flip_direction", "")
            flip_tag = f" — ⚡ GEX FLIP {flip_dir}!"
        if g.get("signal") == 1:
            scores["gex_regime"] = {
                "score": 1,
                "label": f"Long Gamma{flip_tag}",
                "detail": (f"Total GEX: {g.get('total_gex', 0):,.0f} | "
                           f"Flip level: {g.get('flip_level', 'N/A')} | Spot {'above' if g.get('above_flip') else 'below'} flip"),
                "reason": "Dealers are long gamma — they buy dips and sell rips, supporting upward drift",
            }
        else:
            scores["gex_regime"] = {
                "score": -1,
                "label": f"Short Gamma{flip_tag}",
                "detail": (f"Total GEX: {g.get('total_gex', 0):,.0f} | "
                           f"Flip level: {g.get('flip_level', 'N/A')} | Spot {'above' if g.get('above_flip') else 'below'} flip"),
                "reason": "Dealers are short gamma — they sell weakness and buy strength, amplifying moves",
            }
    else:
        scores["gex_regime"] = {
            "score": 0,
            "label": "GEX — No Data",
            "detail": "Load GEX tab to populate dealer gamma data",
            "reason": "GEX data unavailable — cannot assess dealer positioning",
        }

    # ── 12. Net Premium Flow ─────────────────────────────────────────
    if np_signal and np_signal.get("signal") != 0:
        n = np_signal
        tier_labels = {"flip": "⚡ FLOW FLIP", "conviction": "CONVICTION", "sustained": "SUSTAINED", "early": ""}
        tier_tag = tier_labels.get(n.get("tier", ""), "")
        scores["net_premium_flow"] = {
            "score": n["signal"],
            "label": f"Net Premium {'+' if n['signal'] > 0 else '-'} {'Bullish' if n['signal'] > 0 else 'Bearish'}{' — ' + tier_tag if tier_tag else ''}",
            "detail": n.get("detail", n.get("label", "")),
            "reason": ("Positive net premium = call flow dominant — institutional money is long"
                       if n["signal"] > 0 else
                       "Negative net premium = put flow dominant — institutional money is hedging/short"),
        }
    else:
        scores["net_premium_flow"] = {
            "score": 0,
            "label": "Net Premium — Neutral / No Data",
            "detail": "Load Confluence → Net Premium section to populate",
            "reason": "No sustained net premium signal",
        }

    # ── Totals & Signal State ─────────────────────────────────────────
    long_count    = sum(1 for s in scores.values() if s["score"] == 1)
    short_count   = sum(1 for s in scores.values() if s["score"] == -1)
    neutral_count = sum(1 for s in scores.values() if s["score"] == 0)
    total_indicators = len(scores)

    # Recovery signals (fire as advisory even below threshold)
    recovery_bullish = indicators.get("rsi_recovery_bullish", False)
    recovery_bearish = indicators.get("rsi_recovery_bearish", False)

    if long_count >= CONFLUENCE_THRESHOLD:
        signal       = "ENTER LONG"
        signal_class = "enter-long"
        strength     = long_count
    elif short_count >= CONFLUENCE_THRESHOLD:
        signal       = "ENTER SHORT"
        signal_class = "enter-short"
        strength     = short_count
    elif long_count >= STAY_THRESHOLD and long_count > short_count + 1:
        signal       = "STAY LONG"
        signal_class = "stay-long"
        strength     = long_count
    elif short_count >= STAY_THRESHOLD and short_count > long_count + 1:
        signal       = "STAY SHORT"
        signal_class = "stay-short"
        strength     = short_count
    elif long_count >= LEAN_THRESHOLD and long_count > short_count:
        signal       = "LEAN LONG"
        signal_class = "lean-bull"
        strength     = long_count
    elif short_count >= LEAN_THRESHOLD and short_count > long_count:
        signal       = "LEAN SHORT"
        signal_class = "lean-bear"
        strength     = short_count
    else:
        signal       = "NO SIGNAL"
        signal_class = "no-signal"
        strength     = max(long_count, short_count)

    return {
        "scores":            scores,
        "long_count":        long_count,
        "short_count":       short_count,
        "neutral_count":     neutral_count,
        "signal":            signal,
        "signal_class":      signal_class,
        "strength":          strength,
        "threshold":         CONFLUENCE_THRESHOLD,
        "total_indicators":  total_indicators,
        "trend_context":     ("UPTREND" if uptrend else "DOWNTREND" if downtrend else "RANGE"),
        "recovery_bullish":  recovery_bullish,
        "recovery_bearish":  recovery_bearish,
    }


EXIT_THRESHOLD = 3  # Need 3+ out of 6 exit reasons to fire


def score_exit(indicators, position_type, entry_price):
    """
    Score 6 exit conditions for an open position.
    position_type: "long" or "short"
    entry_price: price where the trade was entered

    Returns dict with exit reasons, scores, and final EXIT signal.
    """
    price = indicators["price"]
    reasons = {}
    is_long = position_type == "long"

    pnl_pct = ((price - entry_price) / entry_price) if is_long else ((entry_price - price) / entry_price)

    # Calculate exit price targets
    if is_long:
        stop_price = entry_price * 0.95       # -5%
        partial_price = entry_price * 1.05     # +5%
        full_target_price = entry_price * 1.10 # +10%
    else:
        stop_price = entry_price * 1.05        # +5% (price going up = loss for short)
        partial_price = entry_price * 0.95     # -5% (price going down = profit for short)
        full_target_price = entry_price * 0.90 # -10%

    # --- 1. Profit Target Hit ---
    if pnl_pct >= 0.10:
        reasons["profit_target"] = {
            "triggered": True,
            "label": "Full Profit Target Hit",
            "detail": f"P&L: {pnl_pct:+.2%} (entry: {entry_price:.2f}, now: {price:.2f})",
            "reason": f"10%+ profit reached — full take-profit target hit (target was {full_target_price:.2f})",
            "urgency": "high"
        }
    elif pnl_pct >= 0.05:
        reasons["profit_target"] = {
            "triggered": True,
            "label": "Partial Profit Target Zone",
            "detail": f"P&L: {pnl_pct:+.2%} (entry: {entry_price:.2f}, now: {price:.2f})",
            "reason": f"5%+ profit — consider partial take-profit (full target: {full_target_price:.2f})",
            "urgency": "medium"
        }
    else:
        reasons["profit_target"] = {
            "triggered": False,
            "label": "Below Profit Target",
            "detail": f"P&L: {pnl_pct:+.2%} (partial: {partial_price:.2f} / full: {full_target_price:.2f})",
            "reason": "Profit target not yet reached",
            "urgency": "none"
        }

    # --- 2. Stop Loss Breached ---
    if pnl_pct <= -0.05:
        reasons["stop_loss"] = {
            "triggered": True,
            "label": "STOP LOSS HIT",
            "detail": f"P&L: {pnl_pct:+.2%} — exceeds -5% max loss (stop: {stop_price:.2f})",
            "reason": "Position exceeded maximum acceptable loss — exit immediately",
            "urgency": "critical"
        }
    elif pnl_pct <= -0.03:
        reasons["stop_loss"] = {
            "triggered": True,
            "label": "Approaching Stop",
            "detail": f"P&L: {pnl_pct:+.2%} — nearing -5% stop (stop: {stop_price:.2f})",
            "reason": "Position approaching stop loss level — tighten or exit",
            "urgency": "high"
        }
    else:
        reasons["stop_loss"] = {
            "triggered": False,
            "label": "Within Risk Tolerance",
            "detail": f"P&L: {pnl_pct:+.2%} (stop at {stop_price:.2f})",
            "reason": "Position within acceptable risk range",
            "urgency": "none"
        }

    # --- 3. Momentum Exhaustion (RSI reversal) ---
    rsi = indicators["rsi"]
    if is_long and rsi > 70:
        reasons["momentum_exhaustion"] = {
            "triggered": True,
            "label": "RSI Overbought — Momentum Exhausting",
            "detail": f"RSI: {rsi:.1f} (>70)",
            "reason": "RSI overbought on a long = upside momentum fading, pullback likely",
            "urgency": "high"
        }
    elif not is_long and rsi < 30:
        reasons["momentum_exhaustion"] = {
            "triggered": True,
            "label": "RSI Oversold — Momentum Exhausting",
            "detail": f"RSI: {rsi:.1f} (<30)",
            "reason": "RSI oversold on a short = downside momentum fading, bounce likely",
            "urgency": "high"
        }
    elif is_long and rsi > 60:
        reasons["momentum_exhaustion"] = {
            "triggered": False,
            "label": "RSI Elevated — Watch Closely",
            "detail": f"RSI: {rsi:.1f}",
            "reason": "RSI rising but not yet extreme",
            "urgency": "low"
        }
    elif not is_long and rsi < 40:
        reasons["momentum_exhaustion"] = {
            "triggered": False,
            "label": "RSI Declining — Watch Closely",
            "detail": f"RSI: {rsi:.1f}",
            "reason": "RSI falling but not yet extreme",
            "urgency": "low"
        }
    else:
        reasons["momentum_exhaustion"] = {
            "triggered": False,
            "label": "Momentum Intact",
            "detail": f"RSI: {rsi:.1f}",
            "reason": "RSI in healthy range for the position direction",
            "urgency": "none"
        }

    # --- 4. Confluence Breakdown (indicators flipping against position) ---
    # Re-run the entry confluence to see if conditions have deteriorated
    entry_scores = score_confluence(indicators)
    if is_long:
        opposing = entry_scores["short_count"]
        supporting = entry_scores["long_count"]
    else:
        opposing = entry_scores["long_count"]
        supporting = entry_scores["short_count"]

    if opposing >= 5:
        reasons["confluence_breakdown"] = {
            "triggered": True,
            "label": "Confluence Reversed Against Position",
            "detail": f"Only {supporting}/10 indicators still support your {'long' if is_long else 'short'}, {opposing} now oppose",
            "reason": "The indicators that supported entry have flipped — thesis is broken",
            "urgency": "high"
        }
    elif opposing >= 3 and supporting < 5:
        reasons["confluence_breakdown"] = {
            "triggered": True,
            "label": "Confluence Weakening",
            "detail": f"{supporting}/10 supporting, {opposing} opposing",
            "reason": "Entry thesis weakening — indicators no longer aligned",
            "urgency": "medium"
        }
    else:
        reasons["confluence_breakdown"] = {
            "triggered": False,
            "label": "Confluence Holding",
            "detail": f"{supporting}/10 still supporting your position",
            "reason": "Indicators still support the trade direction",
            "urgency": "none"
        }

    # --- 5. Key Level Rejection / Support Break ---
    if is_long:
        # Long: exit if price rejected at resistance or broke below support
        if indicators["near_20d_high"] and indicators["change_1d"] < -0.003:
            reasons["key_level_break"] = {
                "triggered": True,
                "label": "Rejected at 20-Day High",
                "detail": f"Hit {indicators['high_20']:.2f} and reversed ({indicators['change_1d']*100:+.2f}%)",
                "reason": "Price rejected at key resistance — sellers stepping in",
                "urgency": "high"
            }
        elif price < indicators.get("ema21", price):
            reasons["key_level_break"] = {
                "triggered": True,
                "label": "Lost 21 EMA Support",
                "detail": f"Price {price:.2f} < EMA21 {indicators['ema21']:.2f}",
                "reason": "Price broke below key moving average support",
                "urgency": "medium"
            }
        else:
            reasons["key_level_break"] = {
                "triggered": False,
                "label": "Holding Above Support",
                "detail": f"Price {price:.2f} > EMA21 {indicators['ema21']:.2f}",
                "reason": "Price still holding above key support levels",
                "urgency": "none"
            }
    else:
        # Short: exit if price bounced off support or broke above resistance
        if indicators["near_20d_low"] and indicators["change_1d"] > 0.003:
            reasons["key_level_break"] = {
                "triggered": True,
                "label": "Bounced Off 20-Day Low",
                "detail": f"Hit {indicators['low_20']:.2f} and bounced ({indicators['change_1d']*100:+.2f}%)",
                "reason": "Price bounced at key support — buyers stepping in",
                "urgency": "high"
            }
        elif price > indicators.get("ema21", price):
            reasons["key_level_break"] = {
                "triggered": True,
                "label": "Reclaimed 21 EMA Resistance",
                "detail": f"Price {price:.2f} > EMA21 {indicators['ema21']:.2f}",
                "reason": "Price broke above key moving average resistance",
                "urgency": "medium"
            }
        else:
            reasons["key_level_break"] = {
                "triggered": False,
                "label": "Holding Below Resistance",
                "detail": f"Price {price:.2f} < EMA21 {indicators['ema21']:.2f}",
                "reason": "Price still holding below key resistance levels",
                "urgency": "none"
            }

    # --- 6. Stochastic / MACD Reversal Signal ---
    stoch_reversing = False
    macd_reversing = False
    if is_long:
        stoch_reversing = indicators["stoch_k"] > 80 and indicators["stoch_k"] < indicators["stoch_d"]
        macd_reversing = indicators["macd_hist"] < indicators["macd_hist_prev"] and indicators["macd_hist"] < 0
    else:
        stoch_reversing = indicators["stoch_k"] < 20 and indicators["stoch_k"] > indicators["stoch_d"]
        macd_reversing = indicators["macd_hist"] > indicators["macd_hist_prev"] and indicators["macd_hist"] > 0

    if stoch_reversing and macd_reversing:
        reasons["reversal_signal"] = {
            "triggered": True,
            "label": "Stochastic + MACD Both Reversing",
            "detail": f"Stoch %K: {indicators['stoch_k']:.1f}, MACD Hist: {indicators['macd_hist']:.2f}",
            "reason": "Both momentum oscillators confirming reversal against your position",
            "urgency": "high"
        }
    elif stoch_reversing or macd_reversing:
        reasons["reversal_signal"] = {
            "triggered": True,
            "label": "Early Reversal Warning",
            "detail": f"{'Stochastic' if stoch_reversing else 'MACD'} showing reversal",
            "reason": "One momentum oscillator flipping — watch for confirmation",
            "urgency": "medium"
        }
    else:
        reasons["reversal_signal"] = {
            "triggered": False,
            "label": "No Reversal Signals",
            "detail": f"Stoch %K: {indicators['stoch_k']:.1f}, MACD Hist: {indicators['macd_hist']:.2f}",
            "reason": "Momentum oscillators still support position direction",
            "urgency": "none"
        }

    # Calculate totals
    triggered_count = sum(1 for r in reasons.values() if r["triggered"])
    has_critical = any(r["urgency"] == "critical" for r in reasons.values())

    if has_critical:
        exit_signal = "EXIT NOW"
        exit_class = "exit-now"
    elif triggered_count >= EXIT_THRESHOLD:
        exit_signal = "EXIT POSITION"
        exit_class = "exit-position"
    elif triggered_count >= 2:
        exit_signal = "TIGHTEN STOP"
        exit_class = "tighten-stop"
    else:
        exit_signal = "HOLD POSITION"
        exit_class = "hold-position"

    return {
        "reasons": reasons,
        "triggered_count": triggered_count,
        "total_checks": len(reasons),
        "exit_signal": exit_signal,
        "exit_class": exit_class,
        "pnl_pct": round(pnl_pct * 100, 2),
        "position_type": position_type,
        "entry_price": entry_price,
        "current_price": price,
        "stop_price": round(stop_price, 2),
        "partial_target": round(partial_price, 2),
        "full_target": round(full_target_price, 2),
    }


def analyze_exit(symbol, position_type, entry_price):
    """Full exit analysis for an open position."""
    df = _fetch_ticker_data(symbol)
    if df is None or len(df) < 50:
        return None

    try:
        indicators = _calculate_indicators(df)
        result = score_exit(indicators, position_type, entry_price)
        result["symbol"] = symbol
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        return result
    except Exception:
        return None


def analyze_ticker(symbol, include_reversal=False):
    """Full confluence analysis for a single ticker.
    Automatically includes GEX regime and net premium flow in the 12-indicator system.
    Pass include_reversal=True to also run the reversal entry panel.
    """
    df = _fetch_ticker_data(symbol)
    if df is None or len(df) < 50:
        return None

    try:
        indicators = _calculate_indicators(df)

        # Fetch GEX signal (cached, fast)
        gex_sig = None
        try:
            from gex import get_gex_signal
            # Use NDX GEX for NDX symbols, SPX for everything else
            gex_index = 'NDX' if symbol in ('^NDX', 'QQQ') else 'SPX'
            gex_sig = get_gex_signal(index=gex_index)
        except Exception:
            pass

        # Read net premium signal from cache (no live fetch)
        np_sig = None
        try:
            np_index = 'NDX' if symbol in ('^NDX', 'QQQ') else 'SPX'
            np_sig = fetch_net_premium_signal(index=np_index)
        except Exception:
            pass

        result = score_confluence(indicators, gex_signal=gex_sig, np_signal=np_sig)
        result["symbol"]    = symbol
        result["price"]     = indicators["price"]
        result["change_1d"] = round(indicators["change_1d"] * 100, 2)
        result["change_5d"] = round(indicators["change_5d"] * 100, 2)
        result["rsi"]       = round(indicators["rsi"], 1)
        result["adx"]       = round(indicators["adx"], 1)
        result["vol_ratio"] = round(indicators["vol_ratio"], 2)
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        result["gex_signal"] = gex_sig
        result["np_signal"]  = np_sig

        if include_reversal:
            vix    = _get_vix()
            np_data = _get_net_premium_data()
            result["reversal"] = score_reversal(indicators, vix=vix, net_premium_data=np_data)

            # Fast pullback alert
            try:
                result["fast_pullback"] = get_fast_pullback_alert(
                    indicators, vix=vix, gex_signal=gex_sig, np_signal=np_sig)
            except Exception:
                result["fast_pullback"] = None

        return result
    except Exception:
        return None


def scan_watchlist(symbols=None):
    """Scan a list of tickers and return those with confluence signals."""
    if symbols is None:
        symbols = DEFAULT_WATCHLIST

    results = []
    for symbol in symbols:
        result = analyze_ticker(symbol)
        if result is not None:
            results.append(result)

    # Sort: signals first (ENTER LONG/SHORT), then by strength descending
    def sort_key(r):
        if r["signal"] != "NO SIGNAL":
            return (0, -r["strength"])
        return (1, -r["strength"])

    results.sort(key=sort_key)
    return results


def analyze_ticker_with_confidence(symbol):
    """Full confluence analysis + confidence overlay + reversal panel."""
    result = analyze_ticker(symbol, include_reversal=True)
    if result is None:
        return None

    try:
        from confidence import assess_confidence
        result["confidence"] = assess_confidence(result)
    except Exception as e:
        result["confidence"] = {
            "grade": "UNKNOWN",
            "grade_class": "neutral",
            "supporting": 0,
            "conflicting": 0,
            "neutral": 0,
            "warnings": [f"Confidence check failed: {str(e)}"],
            "details": {},
        }

    return result
