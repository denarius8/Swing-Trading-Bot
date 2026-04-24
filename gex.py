"""
Gamma Exposure (GEX) estimation using yfinance options data.

Calculates dealer gamma exposure by strike to identify:
- Net GEX (are dealers long or short gamma?)
- GEX flip level (where dealers switch from long to short gamma)
- Key gamma strikes (support/resistance from dealer hedging)
- Expected move implied by options pricing

Uses Black-Scholes gamma calculation on SPX/QQQ options chain data.
Data is delayed ~15-20 min via Yahoo Finance.
"""

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# SPX contract multiplier
CONTRACT_MULT = 100
# Risk-free rate estimate
RISK_FREE_RATE = 0.045


def black_scholes_gamma(S, K, T, r, sigma):
    """Calculate Black-Scholes gamma for an option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return gamma
    except (ValueError, ZeroDivisionError):
        return 0.0


def implied_vol_from_price(S, K, T, r, market_price, option_type="call"):
    """Simple bisection IV solver."""
    if market_price <= 0 or T <= 0:
        return 0.3  # default fallback

    low, high = 0.01, 5.0
    for _ in range(50):
        mid = (low + high) / 2
        price = bs_price(S, K, T, r, mid, option_type)
        if price > market_price:
            high = mid
        else:
            low = mid
    return mid


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def fetch_gex_data(index='SPX'):
    """
    Fetch options data and calculate gamma exposure by strike.

    For index='NDX': uses QQQ options (more liquid), scaled to NDX levels.
    For index='SPX': uses SPX/^SPX options directly.

    Returns dict with GEX analysis results.
    """
    if index == 'NDX':
        return _fetch_gex_ndx()

    # --- SPX path (existing logic) ---
    ticker = yf.Ticker("^SPX")

    # Get current price
    hist = ticker.history(period="1d")
    if hist.empty:
        # Fallback to GSPC
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period="1d")

    if hist.empty:
        raise RuntimeError("Could not fetch SPX price data")

    spot = float(hist["Close"].iloc[-1])

    # Get available expiration dates
    try:
        expirations = ticker.options
    except Exception:
        # If ^SPX doesn't work for options, try SPY and scale
        return _fetch_gex_via_spy(spot)

    if not expirations:
        return _fetch_gex_via_spy(spot)

    now = datetime.now()
    target_expirations = []

    # Get expirations within next 45 days (captures weeklies and monthlies)
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if 0 < dte <= 45:
            target_expirations.append(exp_str)

    if not target_expirations:
        # Take first 4 available
        target_expirations = list(expirations[:4])

    return _calculate_gex(ticker, spot, target_expirations, now)


def _fetch_gex_ndx():
    """NDX GEX via QQQ options (more liquid), scaled to NDX."""
    # Get NDX spot
    ndx_tk = yf.Ticker("^NDX")
    ndx_hist = ndx_tk.history(period="1d")
    if ndx_hist.empty:
        raise RuntimeError("Could not fetch NDX price")
    ndx_spot = float(ndx_hist["Close"].iloc[-1])

    # Get QQQ price for scaling
    qqq_tk = yf.Ticker("QQQ")
    qqq_hist = qqq_tk.history(period="1d")
    if qqq_hist.empty:
        raise RuntimeError("Could not fetch QQQ price")
    qqq_price = float(qqq_hist["Close"].iloc[-1])
    scale_factor = ndx_spot / qqq_price  # typically ~40x

    now = datetime.now()
    try:
        expirations = qqq_tk.options
    except Exception:
        raise RuntimeError("Could not fetch QQQ options")

    target_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if 0 < dte <= 45:
            target_expirations.append(exp_str)
    if not target_expirations:
        target_expirations = list(expirations[:4])

    result = _calculate_gex(qqq_tk, qqq_price, target_expirations, now)

    # Scale QQQ values to NDX equivalent
    result["spot"] = round(ndx_spot, 2)
    result["data_source"] = "QQQ (NDX proxy)"
    for s in result["strikes_data"]:
        s["strike"] = round(s["strike"] * scale_factor, 0)
        s["call_gex"] = round(s["call_gex"] * scale_factor, 0)
        s["put_gex"] = round(s["put_gex"] * scale_factor, 0)
        s["net_gex"] = round(s["net_gex"] * scale_factor, 0)
    result["total_gex"] = round(result["total_gex"] * scale_factor, 0)
    if result["gex_flip"]:
        result["gex_flip"] = round(result["gex_flip"] * scale_factor, 2)
    if result["gamma_resistance"]:
        result["gamma_resistance"] = round(result["gamma_resistance"] * scale_factor, 0)
    if result["gamma_support"]:
        result["gamma_support"] = round(result["gamma_support"] * scale_factor, 0)
    result["top_call_gamma"] = [
        {"strike": round(s["strike"] * scale_factor, 0), "gex": round(s["gex"] * scale_factor, 0)}
        for s in result.get("top_call_gamma", [])
    ]
    result["top_put_gamma"] = [
        {"strike": round(s["strike"] * scale_factor, 0), "gex": round(s["gex"] * scale_factor, 0)}
        for s in result.get("top_put_gamma", [])
    ]
    return result


def _fetch_gex_via_spy(spx_spot):
    """Fallback: use SPY options and scale to SPX."""
    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="1d")

    if hist.empty:
        raise RuntimeError("Could not fetch SPY data")

    spy_price = float(hist["Close"].iloc[-1])
    scale_factor = spx_spot / spy_price  # ~10x

    now = datetime.now()
    expirations = ticker.options

    target_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if 0 < dte <= 45:
            target_expirations.append(exp_str)

    if not target_expirations:
        target_expirations = list(expirations[:4])

    result = _calculate_gex(ticker, spy_price, target_expirations, now)

    # Scale SPY strikes and values to SPX equivalent
    result["spot"] = spx_spot
    result["data_source"] = "SPY (scaled to SPX)"
    scaled_strikes = []
    for s in result["strikes_data"]:
        s["strike"] = round(s["strike"] * scale_factor, 0)
        s["call_gex"] = s["call_gex"] * scale_factor
        s["put_gex"] = s["put_gex"] * scale_factor
        s["net_gex"] = s["net_gex"] * scale_factor
        scaled_strikes.append(s)
    result["strikes_data"] = scaled_strikes

    if result["gex_flip"]:
        result["gex_flip"] = round(result["gex_flip"] * scale_factor, 0)
    result["total_gex"] = result["total_gex"] * scale_factor

    # Rescale top strikes
    result["top_call_gamma"] = [
        {"strike": round(s["strike"] * scale_factor, 0), "gex": s["gex"] * scale_factor}
        for s in result.get("top_call_gamma_raw", [])
    ]
    result["top_put_gamma"] = [
        {"strike": round(s["strike"] * scale_factor, 0), "gex": s["gex"] * scale_factor}
        for s in result.get("top_put_gamma_raw", [])
    ]

    return result


def _calculate_gex(ticker, spot, expirations, now):
    """Core GEX calculation across expirations."""
    all_strikes = {}  # strike -> {call_gex, put_gex}
    per_expiry = {}   # exp_str -> {net_gex, call_gex, put_gex, dte, top_strike, dealer_position}

    for exp_str in expirations:
        try:
            chain = ticker.option_chain(exp_str)
        except Exception:
            continue

        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        T = max(dte / 365.0, 1 / 365.0)

        calls = chain.calls
        puts = chain.puts

        # Filter to strikes within reasonable range (80% - 120% of spot)
        strike_low = spot * 0.85
        strike_high = spot * 1.15

        exp_call_gex = 0
        exp_put_gex = 0
        exp_strikes = {}  # per-expiry strike tracking

        # Process calls
        for _, row in calls.iterrows():
            K = float(row["strike"])
            if K < strike_low or K > strike_high:
                continue

            oi_raw = row.get("openInterest", 0)
            if pd.isna(oi_raw) or oi_raw is None:
                continue
            oi = int(oi_raw)
            if oi <= 0:
                continue

            # Use implied vol if available, otherwise estimate
            iv = float(row.get("impliedVolatility", 0.2))
            if iv <= 0:
                mid = (float(row.get("bid", 0)) + float(row.get("ask", 0))) / 2
                if mid > 0:
                    iv = implied_vol_from_price(spot, K, T, RISK_FREE_RATE, mid, "call")
                else:
                    iv = 0.2

            gamma = black_scholes_gamma(spot, K, T, RISK_FREE_RATE, iv)

            # Dealer is short calls (sold to customer), so dealer gamma from calls is positive
            # when they hedge. GEX = gamma * OI * spot * contract_mult * spot / 100
            call_gex = gamma * oi * CONTRACT_MULT * spot
            exp_call_gex += call_gex

            if K not in all_strikes:
                all_strikes[K] = {"call_gex": 0, "put_gex": 0}
            all_strikes[K]["call_gex"] += call_gex

            if K not in exp_strikes:
                exp_strikes[K] = 0
            exp_strikes[K] += call_gex

        # Process puts
        for _, row in puts.iterrows():
            K = float(row["strike"])
            if K < strike_low or K > strike_high:
                continue

            oi_raw = row.get("openInterest", 0)
            if pd.isna(oi_raw) or oi_raw is None:
                continue
            oi = int(oi_raw)
            if oi <= 0:
                continue

            iv = float(row.get("impliedVolatility", 0.2))
            if iv <= 0:
                mid = (float(row.get("bid", 0)) + float(row.get("ask", 0))) / 2
                if mid > 0:
                    iv = implied_vol_from_price(spot, K, T, RISK_FREE_RATE, mid, "put")
                else:
                    iv = 0.2

            gamma = black_scholes_gamma(spot, K, T, RISK_FREE_RATE, iv)

            # Dealer is short puts (sold to customer), so dealer gamma from puts is negative
            put_gex = -gamma * oi * CONTRACT_MULT * spot
            exp_put_gex += put_gex

            if K not in all_strikes:
                all_strikes[K] = {"call_gex": 0, "put_gex": 0}
            all_strikes[K]["put_gex"] += put_gex

            if K not in exp_strikes:
                exp_strikes[K] = 0
            exp_strikes[K] += put_gex

        # Per-expiry summary
        exp_net = exp_call_gex + exp_put_gex
        top_strike = max(exp_strikes, key=lambda k: abs(exp_strikes[k])) if exp_strikes else None
        is_monthly = exp_date.weekday() == 4 and 15 <= exp_date.day <= 21  # 3rd Friday

        per_expiry[exp_str] = {
            "date": exp_str,
            "dte": dte,
            "call_gex": round(exp_call_gex, 0),
            "put_gex": round(exp_put_gex, 0),
            "net_gex": round(exp_net, 0),
            "dealer_position": "LONG GAMMA" if exp_net > 0 else "SHORT GAMMA",
            "top_strike": top_strike,
            "is_monthly": is_monthly,
            "label": "MONTHLY OPEX" if is_monthly else f"{dte}DTE",
        }

    if not all_strikes:
        raise RuntimeError("No options data available. Market may be closed.")

    # Build strike-level data
    strikes_data = []
    for strike in sorted(all_strikes.keys()):
        d = all_strikes[strike]
        net = d["call_gex"] + d["put_gex"]
        strikes_data.append({
            "strike": strike,
            "call_gex": round(d["call_gex"], 0),
            "put_gex": round(d["put_gex"], 0),
            "net_gex": round(net, 0),
        })

    # Total GEX
    total_gex = sum(s["net_gex"] for s in strikes_data)

    # Find GEX flip level (where cumulative GEX crosses zero from positive to negative)
    gex_flip = None
    for i in range(len(strikes_data) - 1):
        if strikes_data[i]["net_gex"] > 0 and strikes_data[i + 1]["net_gex"] <= 0:
            # Interpolate
            s1 = strikes_data[i]
            s2 = strikes_data[i + 1]
            if s1["net_gex"] != s2["net_gex"]:
                ratio = s1["net_gex"] / (s1["net_gex"] - s2["net_gex"])
                gex_flip = s1["strike"] + ratio * (s2["strike"] - s1["strike"])
            else:
                gex_flip = s1["strike"]
            break

    # If no crossover found, check if all positive or all negative
    if gex_flip is None:
        # Look for the strike nearest to zero
        closest = min(strikes_data, key=lambda s: abs(s["net_gex"]))
        gex_flip = closest["strike"]

    # Top gamma strikes
    sorted_by_call = sorted(strikes_data, key=lambda s: s["call_gex"], reverse=True)
    sorted_by_put = sorted(strikes_data, key=lambda s: s["put_gex"])  # most negative

    top_call_gamma = [{"strike": s["strike"], "gex": s["call_gex"]} for s in sorted_by_call[:5]]
    top_put_gamma = [{"strike": s["strike"], "gex": s["put_gex"]} for s in sorted_by_put[:5]]

    # Dealer positioning
    if total_gex > 0:
        dealer_position = "LONG GAMMA"
        dealer_implication = "Dealers hedge by selling rallies / buying dips. Expect MEAN REVERSION and LOWER volatility."
    else:
        dealer_position = "SHORT GAMMA"
        dealer_implication = "Dealers hedge by buying rallies / selling dips. Expect TREND CONTINUATION and HIGHER volatility."

    # Nearest major gamma levels (support/resistance)
    above_spot = [s for s in strikes_data if s["strike"] > spot and abs(s["net_gex"]) > 0]
    below_spot = [s for s in strikes_data if s["strike"] < spot and abs(s["net_gex"]) > 0]

    # Highest absolute GEX above and below spot
    gamma_resistance = None
    gamma_support = None

    if above_spot:
        top_above = max(above_spot, key=lambda s: abs(s["net_gex"]))
        gamma_resistance = top_above["strike"]
    if below_spot:
        top_below = max(below_spot, key=lambda s: abs(s["net_gex"]))
        gamma_support = top_below["strike"]

    return {
        "spot": round(spot, 2),
        "total_gex": round(total_gex, 0),
        "gex_flip": round(gex_flip, 2) if gex_flip else None,
        "dealer_position": dealer_position,
        "dealer_implication": dealer_implication,
        "gamma_resistance": gamma_resistance,
        "gamma_support": gamma_support,
        "top_call_gamma": top_call_gamma,
        "top_put_gamma": top_put_gamma,
        "top_call_gamma_raw": top_call_gamma,  # for SPY scaling
        "top_put_gamma_raw": top_put_gamma,
        "strikes_data": strikes_data,
        "per_expiry": sorted(per_expiry.values(), key=lambda e: e["dte"]),
        "expirations_used": len(expirations),
        "data_source": "SPX (^SPX)" if "SPX" in str(expirations) else "yfinance",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
