"""
Signal Confidence Overlay — Leading Indicators System

Grades confluence signals as HIGH / MEDIUM / LOW confidence based on
4 leading indicators that detect conditions BEFORE price moves:
  1. News Sentiment (headlines from yfinance)
  2. Crude Oil Correlation (Brent/WTI direction vs SPX)
  3. Dealer Positioning (GEX + options flow)
  4. Multi-Timeframe Heikin-Ashi Trends

When confluence fires ENTER SHORT but leading indicators conflict,
the grade drops to LOW with specific warnings.
"""

import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def _sanitize(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ─── In-memory TTL cache (5-minute expiry) ───────────────────────────
_cache = {}
CACHE_TTL = 300  # seconds


def _get_cached(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def _set_cached(key, data):
    _cache[key] = (data, time.time())


# ─── 1. News Sentiment ───────────────────────────────────────────────

BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "record high", "all-time high", "beat expectations",
    "rate cut", "stimulus", "jobs beat", "strong earnings", "bullish", "recovery",
    "rebound", "breakout", "optimism", "upgrade", "buy",
]
BEARISH_KEYWORDS = [
    "crash", "plunge", "tumble", "sell-off", "selloff", "recession", "war",
    "tariff", "rate hike", "miss", "downgrade", "bearish", "fears", "crisis",
    "layoffs", "default", "inflation spike", "collapse", "slump", "decline",
]
HIGH_IMPACT_KEYWORDS = [
    "fed", "fomc", "cpi", "ppi", "jobs report", "nonfarm", "non-farm",
    "earnings", "gdp", "oil", "crude", "geopolitical", "election",
    "debt ceiling", "government shutdown", "banking crisis", "opec",
]


def fetch_news_sentiment(symbol="^GSPC"):
    """Fetch and score recent news headlines for sentiment."""
    cached = _get_cached("news_sentiment")
    if cached is not None:
        return cached

    result = {
        "headlines": [],
        "net_sentiment": 0.0,
        "high_impact_pending": False,
        "signal": 0,
        "warning": None,
    }

    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            _set_cached("news_sentiment", result)
            return result

        # Handle different yfinance news formats
        items = news
        if isinstance(news, dict) and "news" in news:
            items = news["news"]

        scored_headlines = []
        total_score = 0
        high_impact = False

        for item in items[:15]:  # Check up to 15 headlines
            title = ""
            pub_time = None

            # Handle different item formats
            if isinstance(item, dict):
                title = item.get("title", item.get("headline", ""))
                # Try different timestamp fields
                ts = item.get("providerPublishTime", item.get("publish_time", None))
                if ts and isinstance(ts, (int, float)):
                    pub_time = datetime.fromtimestamp(ts)
                elif ts and isinstance(ts, str):
                    try:
                        pub_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        pass
            elif isinstance(item, str):
                title = item

            if not title:
                continue

            # Filter to last 48 hours
            if pub_time and (datetime.now() - pub_time).total_seconds() > 48 * 3600:
                continue

            title_lower = title.lower()

            # Score the headline
            bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in title_lower)
            bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in title_lower)
            impact_hits = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in title_lower]

            score = bull_hits - bear_hits
            if impact_hits:
                high_impact = True

            scored_headlines.append({
                "title": title,
                "time": pub_time.strftime("%m/%d %H:%M") if pub_time else "Unknown",
                "sentiment": "BULLISH" if score > 0 else ("BEARISH" if score < 0 else "NEUTRAL"),
                "score": score,
                "high_impact": bool(impact_hits),
                "impact_tags": impact_hits,
            })
            total_score += score

        # Normalize sentiment to -1 to +1
        if scored_headlines:
            net = total_score / max(len(scored_headlines), 1)
            net = max(-1.0, min(1.0, net))
        else:
            net = 0.0

        result["headlines"] = scored_headlines
        result["net_sentiment"] = round(net, 2)
        result["high_impact_pending"] = high_impact

        if net > 0.3:
            result["signal"] = 1
        elif net < -0.3:
            result["signal"] = -1
        else:
            result["signal"] = 0

        if high_impact:
            impact_tags = set()
            for h in scored_headlines:
                impact_tags.update(h.get("impact_tags", []))
            result["warning"] = f"High-impact event detected: {', '.join(sorted(impact_tags))}"

    except Exception as e:
        result["error"] = str(e)

    result = _sanitize(result)
    _set_cached("news_sentiment", result)
    return result


# ─── 2. Crude Oil Correlation ────────────────────────────────────────

def fetch_crude_correlation():
    """Fetch Brent & WTI crude oil trends as a macro leading indicator."""
    cached = _get_cached("crude_correlation")
    if cached is not None:
        return cached

    result = {
        "brent_price": None,
        "brent_1d_change": None,
        "brent_5d_change": None,
        "brent_trend": "UNKNOWN",
        "wti_price": None,
        "wti_1d_change": None,
        "wti_trend": "UNKNOWN",
        "signal": 0,
        "warning": None,
    }

    def _analyze_crude(symbol):
        try:
            tk = yf.Ticker(symbol)
            df = tk.history(period="1mo")
            if df.empty or len(df) < 10:
                return None

            price = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            change_1d = (price - prev) / prev
            change_5d = (price - float(df["Close"].iloc[-6])) / float(df["Close"].iloc[-6]) if len(df) >= 6 else 0

            ema9 = df["Close"].ewm(span=9).mean()
            ema21 = df["Close"].ewm(span=21).mean()
            trend = "BULLISH" if float(ema9.iloc[-1]) > float(ema21.iloc[-1]) else "BEARISH"

            return {
                "price": round(price, 2),
                "change_1d": round(change_1d * 100, 2),
                "change_5d": round(change_5d * 100, 2),
                "trend": trend,
            }
        except Exception:
            return None

    brent = _analyze_crude("BZ=F")
    wti = _analyze_crude("CL=F")

    if brent:
        result["brent_price"] = brent["price"]
        result["brent_1d_change"] = brent["change_1d"]
        result["brent_5d_change"] = brent["change_5d"]
        result["brent_trend"] = brent["trend"]

    if wti:
        result["wti_price"] = wti["price"]
        result["wti_1d_change"] = wti["change_1d"]
        result["wti_trend"] = wti["trend"]

    # Determine crude signal — crude direction often aligns with SPX (risk-on/risk-off)
    trends = []
    if brent:
        trends.append(1 if brent["trend"] == "BULLISH" else -1)
    if wti:
        trends.append(1 if wti["trend"] == "BULLISH" else -1)

    if trends:
        avg = sum(trends) / len(trends)
        if avg > 0:
            result["signal"] = 1  # Crude bullish = risk-on = SPX bullish bias
        elif avg < 0:
            result["signal"] = -1  # Crude bearish = risk-off = SPX bearish bias
        else:
            result["signal"] = 0

    # Generate warning for strong moves
    if brent and abs(brent["change_1d"]) > 2.0:
        direction = "surging" if brent["change_1d"] > 0 else "plunging"
        result["warning"] = f"Brent crude {direction} ({brent['change_1d']:+.1f}% today)"

    result = _sanitize(result)
    _set_cached("crude_correlation", result)
    return result


# ─── 3. Dealer Positioning ───────────────────────────────────────────

def fetch_dealer_positioning():
    """Analyze dealer positioning via GEX and options flow data."""
    cached = _get_cached("dealer_positioning")
    if cached is not None:
        return cached

    result = {
        "dealer_position": "UNKNOWN",
        "total_gex": 0,
        "pc_volume_ratio": None,
        "pc_oi_ratio": None,
        "iv_rank": None,
        "skew_ratio": None,
        "signal": 0,
        "warning": None,
    }

    # Fetch GEX data
    try:
        from gex import fetch_gex_data
        gex_data = fetch_gex_data()
        result["dealer_position"] = gex_data.get("dealer_position", "UNKNOWN")
        result["total_gex"] = gex_data.get("total_gex", 0)
    except Exception:
        pass

    # Fetch options data for put/call ratios and IV
    try:
        from options_analyzer import analyze_spx_options
        opts = analyze_spx_options()
        if opts:
            result["pc_volume_ratio"] = opts.get("pc_volume_ratio")
            result["pc_oi_ratio"] = opts.get("pc_oi_ratio")
            result["iv_rank"] = opts.get("iv_rank")
            result["skew_ratio"] = opts.get("skew", {}).get("ratio") if isinstance(opts.get("skew"), dict) else None
    except Exception:
        pass

    # Score dealer positioning
    bearish_signals = 0
    bullish_signals = 0

    # Short gamma = trend-following (amplifies moves) = volatile
    if result["dealer_position"] == "SHORT GAMMA":
        bearish_signals += 1  # Higher volatility environment
    elif result["dealer_position"] == "LONG GAMMA":
        bullish_signals += 1  # Mean-reverting, calmer

    # High put/call ratio = more hedging demand = bearish bias
    if result["pc_volume_ratio"] and result["pc_volume_ratio"] > 1.2:
        bearish_signals += 1
    elif result["pc_volume_ratio"] and result["pc_volume_ratio"] < 0.8:
        bullish_signals += 1

    # High IV rank = fear elevated = contrarian bullish, but supports bearish positioning
    if result["iv_rank"] and result["iv_rank"] > 60:
        bearish_signals += 1
    elif result["iv_rank"] and result["iv_rank"] < 30:
        bullish_signals += 1

    if bearish_signals >= 2:
        result["signal"] = -1
    elif bullish_signals >= 2:
        result["signal"] = 1
    else:
        result["signal"] = 0

    # Generate warning for extreme positioning
    if result["pc_volume_ratio"] and result["pc_volume_ratio"] > 1.5:
        result["warning"] = f"Extreme put/call ratio ({result['pc_volume_ratio']:.2f}) — heavy hedging"
    elif result["dealer_position"] == "SHORT GAMMA":
        result["warning"] = "Dealers SHORT GAMMA — expect amplified moves in either direction"

    result = _sanitize(result)
    _set_cached("dealer_positioning", result)
    return result


# ─── 4. Multi-Timeframe Heikin-Ashi ──────────────────────────────────

def calculate_heikin_ashi(df):
    """Convert OHLCV DataFrame to Heikin-Ashi candles."""
    ha = pd.DataFrame(index=df.index)
    ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # HA Open: sequential calculation
    ha_open = np.zeros(len(df))
    ha_open[0] = (float(df["Open"].iloc[0]) + float(df["Close"].iloc[0])) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + float(ha["Close"].iloc[i - 1])) / 2
    ha["Open"] = ha_open

    ha["High"] = pd.concat([df["High"], ha["Open"], ha["Close"]], axis=1).max(axis=1)
    ha["Low"] = pd.concat([df["Low"], ha["Open"], ha["Close"]], axis=1).min(axis=1)

    if "Volume" in df.columns:
        ha["Volume"] = df["Volume"]

    return ha


def _score_ha_timeframe(df, label):
    """Score a single timeframe's Heikin-Ashi trend + volume."""
    if df is None or len(df) < 5:
        return {"label": label, "trend": "UNKNOWN", "ha_streak": 0, "signal": 0, "vol_confirm": False}

    ha = calculate_heikin_ashi(df)

    # Count consecutive green/red HA candles from most recent
    green_streak = 0
    red_streak = 0
    for i in range(len(ha) - 1, max(len(ha) - 10, -1), -1):
        if float(ha["Close"].iloc[i]) > float(ha["Open"].iloc[i]):
            if red_streak > 0:
                break
            green_streak += 1
        elif float(ha["Close"].iloc[i]) < float(ha["Open"].iloc[i]):
            if green_streak > 0:
                break
            red_streak += 1
        else:
            break

    # Volume confirmation: is volume trending up during the streak?
    vol_confirm = False
    if "Volume" in ha.columns and len(ha) >= 5:
        try:
            recent_vol = ha["Volume"].iloc[-3:].mean()
            prior_vol = ha["Volume"].iloc[-6:-3].mean()
            vol_confirm = recent_vol > prior_vol * 1.1 if prior_vol > 0 else False
        except Exception:
            pass

    streak = green_streak if green_streak > 0 else -red_streak

    if green_streak >= 3:
        trend = "BULLISH"
        signal = 1
    elif red_streak >= 3:
        trend = "BEARISH"
        signal = -1
    elif green_streak >= 2:
        trend = "LEAN BULLISH"
        signal = 1 if vol_confirm else 0
    elif red_streak >= 2:
        trend = "LEAN BEARISH"
        signal = -1 if vol_confirm else 0
    else:
        trend = "NEUTRAL"
        signal = 0

    return {
        "label": label,
        "trend": trend,
        "ha_streak": streak,
        "signal": signal,
        "vol_confirm": vol_confirm,
    }


def fetch_multi_timeframe_signals():
    """Fetch and score Heikin-Ashi trends across multiple timeframes."""
    cached = _get_cached("mtf_signals")
    if cached is not None:
        return cached

    result = {
        "timeframes": {},
        "alignment": 0,
        "total_timeframes": 0,
        "dominant_trend": "NEUTRAL",
        "signal": 0,
        "warning": None,
    }

    # (label, symbol, interval, period, weight)
    # Weekly 3x, Daily 2x, 4-Hour 1x, 90-Min 1x — higher timeframes dominate
    timeframe_configs = [
        ("Weekly", "^GSPC", "1wk", "6mo", 3),
        ("Daily", "^GSPC", "1d", "3mo", 2),
        ("4-Hour", "^GSPC", "60m", "5d", 1),    # Aggregate 60m into 4h
        ("90-Min", "^GSPC", "90m", "5d", 1),
    ]

    for label, symbol, interval, period, weight in timeframe_configs:
        try:
            tk = yf.Ticker(symbol)
            df = tk.history(period=period, interval=interval)

            if df.empty:
                continue

            # For 4-Hour: aggregate 60m bars into 4-bar groups
            if label == "4-Hour" and interval == "60m" and len(df) >= 4:
                df_4h = df.resample("4h").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }).dropna()
                df = df_4h

            tf_result = _score_ha_timeframe(df, label)
            tf_result["weight"] = weight
            result["timeframes"][label] = tf_result

        except Exception:
            continue

    # Calculate weighted alignment
    # Weekly=3x, Daily=2x, 4-Hour=1x, 90-Min=1x (total possible = 7)
    result["total_timeframes"] = len(result["timeframes"])
    total_weight = sum(tf.get("weight", 1) for tf in result["timeframes"].values())

    bull_weight = sum(tf.get("weight", 1) for tf in result["timeframes"].values() if tf["signal"] == 1)
    bear_weight = sum(tf.get("weight", 1) for tf in result["timeframes"].values() if tf["signal"] == -1)
    bull_count = sum(1 for tf in result["timeframes"].values() if tf["signal"] == 1)
    bear_count = sum(1 for tf in result["timeframes"].values() if tf["signal"] == -1)

    result["bull_weight"] = bull_weight
    result["bear_weight"] = bear_weight
    result["total_weight"] = total_weight
    result["alignment"] = max(bull_count, bear_count)

    if bull_weight > 0 or bear_weight > 0:
        if bull_weight > bear_weight:
            result["dominant_trend"] = "BULLISH"
            result["signal"] = 1
        elif bear_weight > bull_weight:
            result["dominant_trend"] = "BEARISH"
            result["signal"] = -1
        else:
            result["dominant_trend"] = "MIXED"
            result["signal"] = 0

        # Warning if timeframes disagree
        if bull_count > 0 and bear_count > 0:
            result["warning"] = f"Timeframes split: {bull_count} bullish, {bear_count} bearish"
    else:
        result["dominant_trend"] = "NEUTRAL"

    result = _sanitize(result)
    _set_cached("mtf_signals", result)
    return result


# ─── Master Confidence Assessment ────────────────────────────────────

def assess_confidence(confluence_result):
    """
    Grade a confluence signal's confidence using 4 leading indicators.

    Returns:
        dict with grade (HIGH/MEDIUM/LOW), warnings, and full details.
    """
    direction = confluence_result.get("signal", "NO SIGNAL")

    # Import net premium signal
    try:
        from net_premium import fetch_net_premium_signal
        net_prem = fetch_net_premium_signal()
    except Exception:
        net_prem = {"signal": 0, "label": "Unavailable", "detail": "Could not load net premium data"}

    if direction == "ENTER LONG":
        expected_sign = 1
    elif direction == "ENTER SHORT":
        expected_sign = -1
    else:
        # No signal fired — still provide indicator data but grade is N/A
        news = fetch_news_sentiment()
        crude = fetch_crude_correlation()
        positioning = fetch_dealer_positioning()
        mtf = fetch_multi_timeframe_signals()
        return _sanitize({
            "grade": "N/A",
            "grade_class": "neutral",
            "supporting": 0,
            "conflicting": 0,
            "neutral": 0,
            "warnings": [],
            "details": {
                "news": news,
                "crude": crude,
                "positioning": positioning,
                "multi_timeframe": mtf,
                "net_premium": net_prem,
            },
        })

    # Fetch all leading indicators
    news = fetch_news_sentiment()
    crude = fetch_crude_correlation()
    positioning = fetch_dealer_positioning()
    mtf = fetch_multi_timeframe_signals()

    # Count conflicts
    indicators = [
        ("News Sentiment", news),
        ("Crude Oil", crude),
        ("Dealer Positioning", positioning),
        ("Multi-Timeframe", mtf),
        ("Net Premium Flow", net_prem),
    ]

    conflicts = 0
    supporting = 0
    neutral = 0
    warnings = []

    for name, indicator in indicators:
        sig = indicator.get("signal", 0)
        if sig == -expected_sign:
            conflicts += 1
            # Generate specific conflict message
            if name == "News Sentiment" and news.get("net_sentiment", 0) != 0:
                sent_dir = "bullish" if news["net_sentiment"] > 0 else "bearish"
                warnings.append(f"News sentiment is {sent_dir} — conflicts with {direction}")
            elif name == "Crude Oil":
                crude_dir = crude.get("brent_trend", "UNKNOWN")
                if crude_dir != "UNKNOWN":
                    warnings.append(f"Crude oil {crude_dir.lower()} — conflicts with {direction}")
            elif name == "Dealer Positioning":
                warnings.append(f"Dealer positioning conflicts with {direction}")
            elif name == "Multi-Timeframe":
                dom = mtf.get("dominant_trend", "UNKNOWN")
                warnings.append(f"Multi-timeframe trend is {dom} — conflicts with {direction}")
            elif name == "Net Premium Flow":
                np_dir = net_prem.get("streak_direction", "unknown")
                warnings.append(f"Net premium flow is {np_dir} — conflicts with {direction}")
        elif sig == expected_sign:
            supporting += 1
        else:
            neutral += 1

        # Add indicator-specific warnings
        if indicator.get("warning"):
            warnings.append(indicator["warning"])

    # Grade assignment
    if conflicts == 0:
        grade = "HIGH"
        grade_class = "high"
    elif conflicts == 1:
        grade = "MEDIUM"
        grade_class = "medium"
    else:
        grade = "LOW"
        grade_class = "low"

    # Special override: high-impact news always drops to LOW
    if news.get("high_impact_pending"):
        if grade != "LOW":
            grade = "LOW"
            grade_class = "low"
            if news.get("warning") and news["warning"] not in warnings:
                warnings.insert(0, news["warning"])

    return _sanitize({
        "grade": grade,
        "grade_class": grade_class,
        "supporting": supporting,
        "conflicting": conflicts,
        "neutral": neutral,
        "total": len(indicators),
        "warnings": warnings,
        "details": {
            "news": news,
            "crude": crude,
            "positioning": positioning,
            "multi_timeframe": mtf,
            "net_premium": net_prem,
        },
    })
