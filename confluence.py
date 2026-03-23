"""
Confluence-Based Swing Trade Signal Engine
Scans any ticker for alignment of 10 high-conviction indicators.
Only fires ENTER LONG / ENTER SHORT when 7+ indicators align.
"""

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime


# Default watchlist for scanner
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META",
    "GOOG", "AMD", "NFLX", "AVGO", "CRM", "ORCL", "COST",
    "V", "JPM", "UNH", "MA", "HD",
    "XOM", "LLY", "ABBV", "PG", "MRK",
    "COIN", "MARA", "PLTR", "SOFI", "RIVN",
    "IWM", "DIA", "XLF", "XLE", "XLK", "GLD", "SLV", "TLT",
]

CONFLUENCE_THRESHOLD = 7  # Need 7+ out of 10 for a signal


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

    # Extra context
    d["change_1d"] = (price - d["prev_close"]) / d["prev_close"]
    ret_5d = close.pct_change(5)
    d["change_5d"] = float(ret_5d.iloc[-1]) if not np.isnan(ret_5d.iloc[-1]) else 0

    return d


def score_confluence(indicators):
    """
    Score 10 confluence indicators.
    Returns: dict with scores (+1=long, -1=short, 0=neutral) per indicator,
    total long/short scores, and final signal.
    """
    scores = {}

    # 1. RSI Zone + Divergence
    rsi = indicators["rsi"]
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

    # 2. VWAP Reclaim/Rejection
    if indicators["price_vs_vwap"] > 0:
        scores["vwap"] = {"score": 1, "label": "Above VWAP",
                          "detail": f"Price {indicators['price']:.2f} > VWAP {indicators['vwap']:.2f}",
                          "reason": "Price above VWAP = institutional buying bias"}
    else:
        scores["vwap"] = {"score": -1, "label": "Below VWAP",
                          "detail": f"Price {indicators['price']:.2f} < VWAP {indicators['vwap']:.2f}",
                          "reason": "Price below VWAP = institutional selling bias"}

    # 3. EMA Stack (9 > 21 > 50)
    if indicators["ema_bull_stack"]:
        scores["ema_stack"] = {"score": 1, "label": "Bullish EMA Stack",
                               "detail": f"EMA 9 ({indicators['ema9']:.1f}) > 21 ({indicators['ema21']:.1f}) > 50 ({indicators['ema50']:.1f})",
                               "reason": "All short-term MAs aligned bullish = strong uptrend"}
    elif indicators["ema_bear_stack"]:
        scores["ema_stack"] = {"score": -1, "label": "Bearish EMA Stack",
                               "detail": f"EMA 9 ({indicators['ema9']:.1f}) < 21 ({indicators['ema21']:.1f}) < 50 ({indicators['ema50']:.1f})",
                               "reason": "All short-term MAs aligned bearish = strong downtrend"}
    else:
        scores["ema_stack"] = {"score": 0, "label": "EMAs Mixed",
                               "detail": "EMA 9/21/50 not fully aligned",
                               "reason": "Mixed EMAs = no clear directional trend"}

    # 4. MACD Crossover + Histogram
    if indicators["macd_hist"] > 0 and (indicators["macd_bull_cross"] or indicators["macd_hist_expanding"]):
        scores["macd"] = {"score": 1, "label": "MACD Bullish",
                          "detail": f"Histogram: {indicators['macd_hist']:.2f}" + (" (Fresh Cross)" if indicators["macd_bull_cross"] else " (Expanding)"),
                          "reason": "Bullish MACD crossover or expanding positive histogram"}
    elif indicators["macd_hist"] < 0 and (indicators["macd_bear_cross"] or indicators["macd_hist_expanding"]):
        scores["macd"] = {"score": -1, "label": "MACD Bearish",
                          "detail": f"Histogram: {indicators['macd_hist']:.2f}" + (" (Fresh Cross)" if indicators["macd_bear_cross"] else " (Expanding)"),
                          "reason": "Bearish MACD crossover or expanding negative histogram"}
    else:
        scores["macd"] = {"score": 0, "label": "MACD Neutral",
                          "detail": f"Histogram: {indicators['macd_hist']:.2f}",
                          "reason": "No fresh crossover or histogram not expanding"}

    # 5. Bollinger Band Squeeze → Expansion
    if indicators["bb_pct"] > 0.8 and indicators["bb_expanding"]:
        scores["bollinger"] = {"score": 1, "label": "BB Upper Breakout",
                               "detail": f"BB%: {indicators['bb_pct']:.1%} (Bands Expanding)",
                               "reason": "Price breaking above upper band with expansion = momentum breakout"}
    elif indicators["bb_pct"] < 0.2 and indicators["bb_expanding"]:
        scores["bollinger"] = {"score": -1, "label": "BB Lower Breakdown",
                               "detail": f"BB%: {indicators['bb_pct']:.1%} (Bands Expanding)",
                               "reason": "Price breaking below lower band with expansion = momentum breakdown"}
    elif indicators["bb_pct"] < 0.2 and not indicators["bb_expanding"]:
        scores["bollinger"] = {"score": 1, "label": "BB Oversold Bounce",
                               "detail": f"BB%: {indicators['bb_pct']:.1%} (Bands Tight)",
                               "reason": "Price at lower band in squeeze = mean reversion bounce likely"}
    elif indicators["bb_pct"] > 0.8 and not indicators["bb_expanding"]:
        scores["bollinger"] = {"score": -1, "label": "BB Overbought Fade",
                               "detail": f"BB%: {indicators['bb_pct']:.1%} (Bands Tight)",
                               "reason": "Price at upper band in squeeze = mean reversion pullback likely"}
    else:
        scores["bollinger"] = {"score": 0, "label": "BB Neutral",
                               "detail": f"BB%: {indicators['bb_pct']:.1%}",
                               "reason": "Price in middle of Bollinger Bands"}

    # 6. Volume Confirmation
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

    # 7. Key Level Test
    if indicators["near_20d_low"]:
        scores["key_level"] = {"score": 1, "label": "Testing 20-Day Low (Support)",
                               "detail": f"Near {indicators['low_20']:.2f}",
                               "reason": "Price at key support level = bounce opportunity"}
    elif indicators["near_20d_high"]:
        scores["key_level"] = {"score": -1, "label": "Testing 20-Day High (Resistance)",
                               "detail": f"Near {indicators['high_20']:.2f}",
                               "reason": "Price at key resistance = rejection risk"}
    else:
        scores["key_level"] = {"score": 0, "label": "Mid-Range",
                               "detail": f"Range: {indicators['low_20']:.2f} - {indicators['high_20']:.2f}",
                               "reason": "Price not testing any key level"}

    # 8. ADX Trend Strength
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

    # 9. Stochastic Crossover in Extreme Zones
    if indicators["stoch_bull_cross"] or (indicators["stoch_k"] < 20):
        scores["stochastic"] = {"score": 1, "label": "Stochastic Oversold",
                                "detail": f"%K: {indicators['stoch_k']:.1f} / %D: {indicators['stoch_d']:.1f}",
                                "reason": "Stochastic in oversold zone = bounce signal"}
    elif indicators["stoch_bear_cross"] or (indicators["stoch_k"] > 80):
        scores["stochastic"] = {"score": -1, "label": "Stochastic Overbought",
                                "detail": f"%K: {indicators['stoch_k']:.1f} / %D: {indicators['stoch_d']:.1f}",
                                "reason": "Stochastic in overbought zone = pullback signal"}
    else:
        scores["stochastic"] = {"score": 0, "label": "Stochastic Neutral",
                                "detail": f"%K: {indicators['stoch_k']:.1f} / %D: {indicators['stoch_d']:.1f}",
                                "reason": "Stochastic in neutral zone"}

    # 10. Price vs 200 SMA (Big Picture Trend)
    if indicators["above_200sma"]:
        scores["trend_200"] = {"score": 1, "label": "Above 200 SMA",
                               "detail": f"Price {indicators['price']:.2f} > SMA200 {indicators['sma200']:.2f}",
                               "reason": "Above 200-day MA = long-term uptrend intact"}
    else:
        scores["trend_200"] = {"score": -1, "label": "Below 200 SMA",
                               "detail": f"Price {indicators['price']:.2f} < SMA200 {indicators['sma200']:.2f}",
                               "reason": "Below 200-day MA = long-term downtrend"}

    # Calculate totals
    long_count = sum(1 for s in scores.values() if s["score"] == 1)
    short_count = sum(1 for s in scores.values() if s["score"] == -1)
    neutral_count = sum(1 for s in scores.values() if s["score"] == 0)

    if long_count >= CONFLUENCE_THRESHOLD:
        signal = "ENTER LONG"
        signal_class = "enter-long"
        strength = long_count
    elif short_count >= CONFLUENCE_THRESHOLD:
        signal = "ENTER SHORT"
        signal_class = "enter-short"
        strength = short_count
    else:
        signal = "NO SIGNAL"
        signal_class = "no-signal"
        strength = max(long_count, short_count)

    return {
        "scores": scores,
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "signal": signal,
        "signal_class": signal_class,
        "strength": strength,
        "threshold": CONFLUENCE_THRESHOLD,
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


def analyze_ticker(symbol):
    """Full confluence analysis for a single ticker."""
    df = _fetch_ticker_data(symbol)
    if df is None or len(df) < 50:
        return None

    try:
        indicators = _calculate_indicators(df)
        result = score_confluence(indicators)
        result["symbol"] = symbol
        result["price"] = indicators["price"]
        result["change_1d"] = round(indicators["change_1d"] * 100, 2)
        result["change_5d"] = round(indicators["change_5d"] * 100, 2)
        result["rsi"] = round(indicators["rsi"], 1)
        result["adx"] = round(indicators["adx"], 1)
        result["vol_ratio"] = round(indicators["vol_ratio"], 2)
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        return result
    except Exception as e:
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
    """Full confluence analysis + confidence overlay from leading indicators."""
    result = analyze_ticker(symbol)
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
