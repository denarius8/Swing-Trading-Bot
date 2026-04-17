"""
Scaled Entry Checklist — 5-point trend confirmation scoring system.

Replaces a binary pass/fail gate with a continuous 0-5 score.
Score determines position size tier (NO TRADE / STARTER / ADD / FULL).
Auto-populates all inputs from live market data.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── Account & Score Configuration ───────────────────────────────────

ACCOUNT_CONFIG = {
    "balance": 10000,           # Default — users set their own balance in the UI
    "max_risk_pct": 0.02,       # 2% max risk per trade
    "daily_stop_pct": 0.03,     # 3% daily stop-out
}

SCORE_THRESHOLDS = {
    "starter_min": 2.5,
    "add_min": 3.5,
    "full_min": 4.5,
}

TIER_OUTLAY_PCT = {
    "NO TRADE":     0.00,
    "STARTER":      0.25,
    "ADD":          0.50,
    "FULL":         1.00,
}


# ─── Helpers ─────────────────────────────────────────────────────────

def _ha_score(df, label=""):
    """Score a single timeframe's Heikin-Ashi trend. Returns dict."""
    if df is None or len(df) < 5:
        return {"trend": "NEUTRAL", "signal": 0, "streak": 0, "vol_confirm": False}

    df = df.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (float(df["Open"].iloc[0]) + float(df["Close"].iloc[0])) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + float(ha_close.iloc[i - 1])) / 2

    green_streak = red_streak = 0
    for i in range(len(df) - 1, max(len(df) - 10, -1), -1):
        if ha_close.iloc[i] > ha_open[i]:
            if red_streak > 0: break
            green_streak += 1
        elif ha_close.iloc[i] < ha_open[i]:
            if green_streak > 0: break
            red_streak += 1
        else:
            break

    vol_confirm = False
    if "Volume" in df.columns and len(df) >= 6:
        try:
            rv = df["Volume"].iloc[-3:].mean()
            pv = df["Volume"].iloc[-6:-3].mean()
            vol_confirm = bool(rv > pv * 1.1) if pv > 0 else False
        except Exception:
            pass

    streak = green_streak if green_streak > 0 else -red_streak

    if green_streak >= 3:
        trend, signal = "BULLISH", 1
    elif red_streak >= 3:
        trend, signal = "BEARISH", -1
    elif green_streak >= 2:
        trend, signal = "LEAN BULLISH", 1
    elif red_streak >= 2:
        trend, signal = "LEAN BEARISH", -1
    else:
        trend, signal = "NEUTRAL", 0

    return {"trend": trend, "signal": signal, "streak": streak, "vol_confirm": vol_confirm}


def _macd_state(close):
    """Return MACD state dict from a Close price Series."""
    if len(close) < 35:
        return {"hist": 0, "cross": "Neutral", "expanding": False}
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    cross = "Bullish" if float(hist.iloc[-1]) > 0 else "Bearish"
    expanding = bool(abs(float(hist.iloc[-1])) > abs(float(hist.iloc[-2])))
    return {"hist": round(float(hist.iloc[-1]), 4), "cross": cross, "expanding": expanding}


def _fetch_tf(symbol, interval, period):
    """Fetch OHLCV for a timeframe, normalize column names."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = [c.title() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


# ─── Live Data Fetcher ────────────────────────────────────────────────

def fetch_checklist_data(symbol="^GSPC"):
    """
    Fetch all live data needed for the 5-check scoring system.
    Returns a flat dict of scored inputs.
    """
    yf_symbol = symbol if symbol != "SPX" else "^GSPC"

    # ── Timeframe data ────────────────────────────────────────────────
    weekly_df  = _fetch_tf(yf_symbol, "1wk",  "1y")
    daily_df   = _fetch_tf(yf_symbol, "1d",   "1y")
    h4_df_raw  = _fetch_tf(yf_symbol, "60m",  "5d")
    h1_df      = _fetch_tf(yf_symbol, "60m",  "5d")
    m15_df     = _fetch_tf(yf_symbol, "15m",  "5d")

    # Resample 60m → 4H
    h4_df = None
    if h4_df_raw is not None and len(h4_df_raw) >= 4:
        try:
            h4_df = h4_df_raw.resample("4h").agg({
                "Open": "first", "High": "max",
                "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
        except Exception:
            h4_df = h4_df_raw

    # ── HA scores ─────────────────────────────────────────────────────
    weekly_ha = _ha_score(weekly_df, "Weekly")
    daily_ha  = _ha_score(daily_df,  "Daily")
    h4_ha     = _ha_score(h4_df,     "4H")
    h1_ha     = _ha_score(h1_df,     "1H")
    m15_ha    = _ha_score(m15_df,    "15M")

    # ── MACD states ───────────────────────────────────────────────────
    weekly_macd = _macd_state(weekly_df["Close"]) if weekly_df is not None else {"cross": "Neutral"}
    m15_macd    = _macd_state(m15_df["Close"])    if m15_df   is not None else {"cross": "Neutral"}

    # ── Daily indicators: EMA20 distance, golden cross ────────────────
    price_vs_ema20_pct = 0.0
    golden_cross = False
    try:
        if daily_df is not None and len(daily_df) >= 20:
            close = daily_df["Close"].dropna()
            # EMA20 distance
            ema20 = close.ewm(span=20, adjust=False).mean()
            last_close = float(close.iloc[-1])
            last_ema20 = float(ema20.iloc[-1])
            if last_ema20 > 0:
                price_vs_ema20_pct = round((last_close - last_ema20) / last_ema20 * 100, 2)
            # Golden cross: SMA50 > SMA200 (needs ~200 bars — use "1y" period)
            if len(close) >= 200:
                sma50  = float(close.rolling(50).mean().iloc[-1])
                sma200 = float(close.rolling(200).mean().iloc[-1])
                golden_cross = sma50 > sma200
            elif len(close) >= 50:
                # Fallback: SMA50 > SMA100 if not enough history
                sma50  = float(close.rolling(50).mean().iloc[-1])
                sma100 = float(close.rolling(min(100, len(close))).mean().iloc[-1])
                golden_cross = sma50 > sma100
    except Exception:
        pass

    # ── Macro context: VIX + GLD ──────────────────────────────────────
    vix_change_pct = 0.0
    gld_change_pct = 0.0
    try:
        vix = yf.Ticker("^VIX").history(period="5d")
        if len(vix) >= 2:
            vix_change_pct = round((float(vix["Close"].iloc[-1]) - float(vix["Close"].iloc[-2]))
                                   / float(vix["Close"].iloc[-2]) * 100, 2)
    except Exception:
        pass
    try:
        gld = yf.Ticker("GLD").history(period="5d")
        if len(gld) >= 2:
            gld_change_pct = round((float(gld["Close"].iloc[-1]) - float(gld["Close"].iloc[-2]))
                                   / float(gld["Close"].iloc[-2]) * 100, 2)
    except Exception:
        pass

    # ── 1H trend strength ─────────────────────────────────────────────
    streak = abs(h1_ha["streak"])
    if streak >= 5 or h1_ha["vol_confirm"]:
        h1_trend_strength = "Very Strong"
    elif streak >= 3:
        h1_trend_strength = "Strong"
    else:
        h1_trend_strength = "Weak/No Trend"

    return {
        "symbol": yf_symbol,
        # Timeframe trends
        "weekly_bias":          weekly_ha["trend"],
        "weekly_signal":        weekly_ha["signal"],
        "daily_bias":           daily_ha["trend"],
        "daily_signal":         daily_ha["signal"],
        "h4_bias":              h4_ha["trend"],
        "h4_signal":            h4_ha["signal"],
        "h4_momentum_aligned":  h4_ha["vol_confirm"],
        "h1_bias":              h1_ha["trend"],
        "h1_signal":            h1_ha["signal"],
        "h1_trend_strength":    h1_trend_strength,
        "m15_bias":             m15_ha["trend"],
        "m15_signal":           m15_ha["signal"],
        "m15_momentum_aligned": m15_ha["vol_confirm"],
        "m15_macd_cross":       m15_macd["cross"],
        # Price extension
        "price_vs_ema20_pct":   price_vs_ema20_pct,
        "golden_cross":         golden_cross,
        # Divergence signals
        "weekly_macd_cross":    weekly_macd["cross"],
        "vix_change_pct":       vix_change_pct,
        "gld_change_pct":       gld_change_pct,
    }


# ─── Scoring Engine ───────────────────────────────────────────────────

def score_checklist(data):
    """
    Score all 5 checks from live data dict.
    Returns full result including score, tier, direction, and per-check breakdown.
    """
    # Determine intended trade direction from daily bias
    daily_sig = data.get("daily_signal", 0)
    weekly_sig = data.get("weekly_signal", 0)
    if daily_sig > 0 or weekly_sig > 0:
        direction = "BULLISH"
    elif daily_sig < 0 or weekly_sig < 0:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    checks = []

    # ── CHECK 1: Structural trend ─────────────────────────────────────
    weekly = data.get("weekly_bias", "Neutral")
    daily  = data.get("daily_bias",  "Neutral")
    gc     = data.get("golden_cross", False)

    w_bull = "BULLISH" in weekly.upper()
    w_bear = "BEARISH" in weekly.upper()
    d_bull = "BULLISH" in daily.upper()
    d_bear = "BEARISH" in daily.upper()

    if (w_bull and d_bull) or (w_bear and d_bear):
        c1 = 1.0 if gc or (w_bull and d_bull) else 0.5
        c1_detail = f"Weekly {weekly} + Daily {daily}" + (" + Golden Cross" if gc else "")
    elif (w_bull or d_bull) or (w_bear or d_bear):
        c1 = 0.5
        c1_detail = f"Weekly {weekly} vs Daily {daily} — partial alignment"
    else:
        c1 = 0.0
        c1_detail = f"Weekly {weekly} vs Daily {daily} — diverging"

    checks.append({"id": 1, "name": "Structural Trend", "score": c1,
                   "detail": c1_detail, "pass": c1 >= 0.5})

    # ── CHECK 2: Momentum alignment ───────────────────────────────────
    h4    = data.get("h4_bias", "Neutral")
    h4_ok = data.get("h4_momentum_aligned", False)
    h1    = data.get("h1_bias", "Neutral")
    h1_ts = data.get("h1_trend_strength", "Weak/No Trend")

    h4_bull = "BULLISH" in h4.upper()
    h4_bear = "BEARISH" in h4.upper()
    h1_bull = "BULLISH" in h1.upper()
    h1_bear = "BEARISH" in h1.upper()

    # Aligned = both point same way as daily
    if direction == "BULLISH":
        both_confirm = h4_bull and h1_bull
        one_confirms = h4_bull or h1_bull
    elif direction == "BEARISH":
        both_confirm = h4_bear and h1_bear
        one_confirms = h4_bear or h1_bear
    else:
        both_confirm = (h4_bull and h1_bull) or (h4_bear and h1_bear)
        one_confirms = h4_bull or h1_bull or h4_bear or h1_bear

    if both_confirm and (h4_ok or h1_ts in ("Very Strong", "Strong")):
        c2 = 1.0
        c2_detail = f"4H {h4} + 1H {h1} both confirm ({h1_ts})"
    elif both_confirm or one_confirms:
        c2 = 0.5
        c2_detail = f"4H {h4} + 1H {h1} — partial momentum"
    else:
        c2 = 0.0
        c2_detail = f"4H {h4} / 1H {h1} — contradicts {direction}"

    checks.append({"id": 2, "name": "Momentum Alignment", "score": c2,
                   "detail": c2_detail, "pass": c2 >= 0.5})

    # ── CHECK 3: Execution layer (15M) ────────────────────────────────
    m15       = data.get("m15_bias", "Neutral")
    m15_mom   = data.get("m15_momentum_aligned", False)
    m15_macd  = data.get("m15_macd_cross", "Neutral")

    m15_bull  = "BULLISH" in m15.upper()
    m15_bear  = "BEARISH" in m15.upper()
    m15_neut  = not m15_bull and not m15_bear

    if direction == "BULLISH":
        aligned = m15_bull
        against = m15_bear
    elif direction == "BEARISH":
        aligned = m15_bear
        against = m15_bull
    else:
        aligned = m15_bull or m15_bear
        against = False

    if aligned and m15_mom and m15_macd == "Bullish" if direction != "BEARISH" else m15_macd == "Bearish":
        c3 = 1.0
        c3_detail = f"15M {m15} + momentum + MACD {m15_macd}"
    elif aligned or (m15_neut and (m15_macd == "Bullish" and direction == "BULLISH"
                                    or m15_macd == "Bearish" and direction == "BEARISH")):
        c3 = 0.5
        c3_detail = f"15M {m15} — neutral/partial (MACD {m15_macd})"
    elif against:
        c3 = 0.0
        c3_detail = f"15M {m15} actively against {direction}"
    else:
        c3 = 0.5
        c3_detail = f"15M {m15} — neutral, no active resistance"

    checks.append({"id": 3, "name": "Execution Layer (15M)", "score": c3,
                   "detail": c3_detail, "pass": c3 >= 0.5})

    # ── CHECK 4: Extension risk ───────────────────────────────────────
    ext = abs(data.get("price_vs_ema20_pct", 0))
    direction_sign = data.get("price_vs_ema20_pct", 0)

    if ext <= 3.0:
        c4 = 1.0
        c4_detail = f"Price {direction_sign:+.2f}% from EMA20 — ideal pullback zone"
    elif ext <= 5.0:
        c4 = 0.5
        c4_detail = f"Price {direction_sign:+.2f}% from EMA20 — elevated but tradeable"
    else:
        c4 = 0.0
        c4_detail = f"Price {direction_sign:+.2f}% from EMA20 — overextended, snap-back risk"

    checks.append({"id": 4, "name": "Extension Risk", "score": c4,
                   "detail": c4_detail, "pass": c4 >= 0.5})

    # ── CHECK 5: Divergence warnings ─────────────────────────────────
    weekly_macd = data.get("weekly_macd_cross", "Bullish")
    vix_chg     = data.get("vix_change_pct", 0)
    gld_chg     = data.get("gld_change_pct", 0)

    warnings_list = []
    if weekly_macd == "Bearish" and direction == "BULLISH":
        warnings_list.append(f"Weekly MACD bearish cross")
    if gld_chg > 1.5:
        warnings_list.append(f"GLD +{gld_chg:.1f}% (hedging signal)")
    if vix_chg > 10:
        warnings_list.append(f"VIX +{vix_chg:.1f}% (fear spike)")
    elif vix_chg > 5:
        warnings_list.append(f"VIX +{vix_chg:.1f}% (elevated)")
    if weekly_macd == "Bullish" and direction == "BEARISH":
        warnings_list.append(f"Weekly MACD bullish (counter-trend)")

    n_warn = len(warnings_list)
    if n_warn == 0:
        c5 = 1.0
        c5_detail = f"No warnings — VIX {vix_chg:+.1f}%, GLD {gld_chg:+.1f}%"
    elif n_warn == 1:
        c5 = 0.5
        c5_detail = f"1 warning: {warnings_list[0]}"
    else:
        c5 = 0.0
        c5_detail = f"{n_warn} warnings: {' | '.join(warnings_list)}"

    checks.append({"id": 5, "name": "Divergence Warnings", "score": c5,
                   "detail": c5_detail, "pass": c5 >= 0.5})

    # ── Total score + tier ────────────────────────────────────────────
    total = round(sum(c["score"] for c in checks), 1)

    if total < SCORE_THRESHOLDS["starter_min"]:
        tier = "NO TRADE"
        tier_class = "no-trade"
        verdict = "Cash is correct. Neither direction has sufficient confirmation."
    elif total < SCORE_THRESHOLDS["add_min"]:
        tier = "STARTER"
        tier_class = "starter"
        verdict = "25% of max outlay. Defined small risk — gets you in the trade."
    elif total < SCORE_THRESHOLDS["full_min"]:
        tier = "ADD"
        tier_class = "add"
        verdict = "50% total (add 25% to starter). Only after starter is live and moving in your favor."
    else:
        tier = "FULL"
        tier_class = "full"
        verdict = "100% outlay. All checks confirmed. Strongest setups only."

    balance = data.get("_account_balance", ACCOUNT_CONFIG["balance"])
    max_risk = round(balance * ACCOUNT_CONFIG["max_risk_pct"], 0)
    recommended_outlay = round(max_risk / ACCOUNT_CONFIG["max_risk_pct"]
                               * TIER_OUTLAY_PCT.get(tier, 0) * ACCOUNT_CONFIG["max_risk_pct"], 0)

    return {
        "symbol":               data.get("symbol", "^GSPC"),
        "score":                total,
        "max_score":            5.0,
        "tier":                 tier,
        "tier_class":           tier_class,
        "direction":            direction,
        "verdict":              verdict,
        "checks":               checks,
        "warnings":             warnings_list,
        "max_risk":             max_risk,
        "recommended_outlay":   recommended_outlay,
        "account_balance":      balance,
        "raw_data":             {k: v for k, v in data.items() if k != "symbol"},
    }


# ─── Position Management Checks ──────────────────────────────────────

def check_add_triggers(option_pct_gain, spx_held_level, m15_bull_continuation,
                       h1_ema20_bounce):
    """
    Evaluate 4 add-to triggers. Returns True + list if 2+ are met.
    """
    triggers = []
    if option_pct_gain >= 20:
        triggers.append(f"Option up {option_pct_gain:.0f}% from starter entry")
    if spx_held_level:
        triggers.append("SPX holding above key structural level since entry")
    if m15_bull_continuation:
        triggers.append("15M closed bullish continuation candle with expanding volume")
    if h1_ema20_bounce:
        triggers.append("Price held 1H EMA20 on pullback and bounced")

    met = len(triggers) >= 2
    return {"met": met, "confirmed": triggers, "count": len(triggers), "required": 2}


def check_full_position_triggers(checklist_score, spx_broke_level,
                                  volume_1_5x, all_macd_aligned):
    """
    Evaluate 4 full-size triggers. Returns True + list if 3+ are met.
    """
    triggers = []
    if checklist_score >= 4.5:
        triggers.append(f"Checklist score {checklist_score}/5.0 reached 4.5+")
    if spx_broke_level:
        triggers.append("SPX broke above/below key structural level on this candle")
    if volume_1_5x:
        triggers.append("Volume 1.5x+ normal on breakout candle")
    if all_macd_aligned:
        triggers.append("All MACD timeframes (1H, 4H, daily) aligned same direction")

    met = len(triggers) >= 3
    return {"met": met, "confirmed": triggers, "count": len(triggers), "required": 3}


def check_exit_triggers(option_pct_gain, rsi_1h, first_red_after_rsi,
                        hit_t1_resistance, hit_t2_resistance,
                        spx_below_prior_day_low, daily_below_h1_ema50):
    """
    Evaluate T1/T2/T3 exit conditions. Returns list of active exit signals.
    """
    exits = []

    # T1: close 25%
    if hit_t1_resistance:
        exits.append({"tier": "T1", "action": "Close 25%", "reason": "First key resistance hit"})
    if option_pct_gain >= 50:
        exits.append({"tier": "T1", "action": "Close 25%", "reason": f"Option up {option_pct_gain:.0f}% from entry"})

    # T2: close 50%
    if hit_t2_resistance:
        exits.append({"tier": "T2", "action": "Close 50%", "reason": "Second key resistance hit"})
    if rsi_1h and rsi_1h > 85 and first_red_after_rsi:
        exits.append({"tier": "T2", "action": "Close 50%", "reason": f"1H RSI {rsi_1h} then first red candle"})

    # T3: trail + structural exit
    if spx_below_prior_day_low:
        exits.append({"tier": "T3", "action": "Close remaining 25%", "reason": "SPX daily candle below prior day's low"})
    if daily_below_h1_ema50:
        exits.append({"tier": "T3", "action": "EXIT ALL", "reason": "SPX daily candle below 1H EMA50 — structural failure"})

    return exits


# ─── Main entry point ─────────────────────────────────────────────────

def run_checklist(symbol="^GSPC", account_balance=None):
    """Fetch live data and return full scored checklist result."""
    data = fetch_checklist_data(symbol)
    if account_balance is not None:
        data["_account_balance"] = float(account_balance)
    return score_checklist(data)
