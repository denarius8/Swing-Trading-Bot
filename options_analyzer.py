"""
SPX Options Analyzer - 0DTE to 30DTE specific indicators
For short-term SPX option trades with high volatility
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import ta


def _fetch_options_data(symbol="^SPX"):
    """Fetch current options chain and price data."""
    for sym in [symbol, "^GSPC", "^SPX"]:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="6mo")
            if hist.empty:
                continue
            hist.index = pd.to_datetime(hist.index, utc=True).tz_localize(None)

            # Get available expirations
            try:
                expirations = ticker.options
            except Exception:
                expirations = []

            return {
                "ticker": ticker,
                "hist": hist,
                "expirations": list(expirations),
                "symbol": sym,
            }
        except Exception:
            continue
    return None


def _black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate option Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% move in IV

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


def analyze_spx_options():
    """
    Full SPX options analysis for 0DTE-30DTE trading.
    Returns comprehensive data for the dashboard.
    """
    data = _fetch_options_data()
    if data is None:
        return None

    hist = data["hist"]
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    spot = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    result = {
        "spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "data_source": data["symbol"],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    # --- IV Rank & IV Percentile ---
    # Calculate 20-day historical volatility (HV)
    returns = np.log(close / close.shift(1)).dropna()
    hv_20 = float(returns.iloc[-20:].std() * np.sqrt(252))
    hv_5 = float(returns.iloc[-5:].std() * np.sqrt(252))

    # Historical HV range for IV rank approximation
    rolling_hv = returns.rolling(20).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna()
    hv_min = float(rolling_hv.min())
    hv_max = float(rolling_hv.max())
    hv_median = float(rolling_hv.median())

    # IV Rank: where current HV sits in its range (0-100)
    iv_rank = ((hv_20 - hv_min) / (hv_max - hv_min)) * 100 if hv_max > hv_min else 50
    iv_rank = max(0, min(100, iv_rank))

    # IV Percentile: % of days HV was below current
    iv_percentile = (rolling_hv < hv_20).sum() / len(rolling_hv) * 100

    result["iv_rank"] = round(iv_rank, 1)
    result["iv_percentile"] = round(iv_percentile, 1)
    result["hv_20"] = round(hv_20 * 100, 1)  # as percentage
    result["hv_5"] = round(hv_5 * 100, 1)
    result["hv_median"] = round(hv_median * 100, 1)

    # IV label
    if iv_rank > 70:
        result["iv_label"] = "HIGH"
        result["iv_strategy"] = "Sell premium — IV is elevated, options are expensive"
    elif iv_rank > 40:
        result["iv_label"] = "NORMAL"
        result["iv_strategy"] = "Neutral — IV is fair value, both buying and selling viable"
    else:
        result["iv_label"] = "LOW"
        result["iv_strategy"] = "Buy premium — IV is cheap, options are underpriced"

    # --- Expected Move ---
    # 1-day expected move (1 standard deviation)
    em_1d = spot * hv_20 / np.sqrt(252)
    em_5d = spot * hv_20 / np.sqrt(252 / 5)
    em_30d = spot * hv_20 / np.sqrt(252 / 30)

    result["expected_move_1d"] = round(em_1d, 2)
    result["expected_move_5d"] = round(em_5d, 2)
    result["expected_move_30d"] = round(em_30d, 2)
    result["em_1d_pct"] = round(em_1d / spot * 100, 2)
    result["em_5d_pct"] = round(em_5d / spot * 100, 2)
    result["em_30d_pct"] = round(em_30d / spot * 100, 2)

    # Expected move ranges
    result["em_1d_range"] = [round(spot - em_1d, 2), round(spot + em_1d, 2)]
    result["em_5d_range"] = [round(spot - em_5d, 2), round(spot + em_5d, 2)]
    result["em_30d_range"] = [round(spot - em_30d, 2), round(spot + em_30d, 2)]

    # --- Put/Call Skew ---
    # Try to get actual options data
    skew_data = None
    if data["expirations"]:
        try:
            # Find nearest expiration
            today = datetime.now().date()
            nearest_exp = None
            for exp_str in data["expirations"][:5]:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                if exp_date >= today:
                    nearest_exp = exp_str
                    break

            if nearest_exp:
                chain = data["ticker"].option_chain(nearest_exp)
                calls = chain.calls
                puts = chain.puts

                # ATM strike (closest to spot)
                all_strikes = sorted(calls["strike"].unique())
                atm_strike = min(all_strikes, key=lambda x: abs(x - spot))

                # Get ATM call and put IV
                atm_call = calls[calls["strike"] == atm_strike]
                atm_put = puts[puts["strike"] == atm_strike]

                if not atm_call.empty and not atm_put.empty:
                    call_iv = float(atm_call.iloc[0].get("impliedVolatility", 0))
                    put_iv = float(atm_put.iloc[0].get("impliedVolatility", 0))

                    if call_iv > 0 and put_iv > 0:
                        skew = put_iv - call_iv
                        skew_ratio = put_iv / call_iv

                        skew_data = {
                            "call_iv": round(call_iv * 100, 1),
                            "put_iv": round(put_iv * 100, 1),
                            "skew": round(skew * 100, 2),
                            "skew_ratio": round(skew_ratio, 3),
                            "atm_strike": atm_strike,
                            "expiration": nearest_exp,
                        }

                # P/C volume ratio
                total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
                total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
                total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
                total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0

                if pd.notna(total_call_vol) and pd.notna(total_put_vol) and total_call_vol > 0:
                    result["pc_volume_ratio"] = round(float(total_put_vol) / float(total_call_vol), 3)
                else:
                    result["pc_volume_ratio"] = None

                if pd.notna(total_call_oi) and pd.notna(total_put_oi) and total_call_oi > 0:
                    result["pc_oi_ratio"] = round(float(total_put_oi) / float(total_call_oi), 3)
                else:
                    result["pc_oi_ratio"] = None
        except Exception:
            pass

    result["skew"] = skew_data
    if "pc_volume_ratio" not in result:
        result["pc_volume_ratio"] = None
    if "pc_oi_ratio" not in result:
        result["pc_oi_ratio"] = None

    # --- Optimal Strike Selection ---
    # Suggest strikes for different strategies
    atm = round(spot / 5) * 5  # Round to nearest 5

    result["suggested_strikes"] = {
        "atm": atm,
        "otm_call_1": atm + round(em_1d / 5) * 5,  # ~1 SD OTM
        "otm_put_1": atm - round(em_1d / 5) * 5,
        "otm_call_2": atm + round(em_5d / 5) * 5,  # ~1 SD for weekly
        "otm_put_2": atm - round(em_5d / 5) * 5,
    }

    # --- Greeks Snapshot for ATM options ---
    T_0dte = 1 / 252  # ~1 trading day
    T_weekly = 5 / 252
    T_monthly = 21 / 252
    r = 0.045
    sigma = hv_20

    result["greeks_0dte"] = {
        "call": _calculate_greeks(spot, atm, T_0dte, r, sigma, "call"),
        "put": _calculate_greeks(spot, atm, T_0dte, r, sigma, "put"),
        "dte": 0,
    }
    result["greeks_weekly"] = {
        "call": _calculate_greeks(spot, atm, T_weekly, r, sigma, "call"),
        "put": _calculate_greeks(spot, atm, T_weekly, r, sigma, "put"),
        "dte": 5,
    }
    result["greeks_monthly"] = {
        "call": _calculate_greeks(spot, atm, T_monthly, r, sigma, "call"),
        "put": _calculate_greeks(spot, atm, T_monthly, r, sigma, "put"),
        "dte": 21,
    }

    # --- Key Technical Levels for Options ---
    # Support/Resistance from recent price action
    high_5d = float(high.iloc[-5:].max())
    low_5d = float(low.iloc[-5:].min())
    high_20d = float(high.iloc[-20:].max())
    low_20d = float(low.iloc[-20:].min())

    # Pivot points (classic)
    prev_h = float(high.iloc[-1])
    prev_l = float(low.iloc[-1])
    prev_c = float(close.iloc[-1])
    pivot = (prev_h + prev_l + prev_c) / 3
    r1 = 2 * pivot - prev_l
    r2 = pivot + (prev_h - prev_l)
    s1 = 2 * pivot - prev_h
    s2 = pivot - (prev_h - prev_l)

    result["levels"] = {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "high_5d": round(high_5d, 2),
        "low_5d": round(low_5d, 2),
        "high_20d": round(high_20d, 2),
        "low_20d": round(low_20d, 2),
    }

    # --- 0DTE Specific Signals ---
    # RSI on recent bars
    rsi = ta.momentum.rsi(close, window=14)
    rsi_val = float(rsi.iloc[-1])

    # Intraday momentum (using daily as proxy)
    day_range = prev_h - prev_l
    day_body = abs(prev_c - float(hist["Open"].iloc[-1]))
    body_ratio = day_body / day_range if day_range > 0 else 0

    # ATR for position sizing
    atr_14 = float(ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])
    atr_5 = float(ta.volatility.average_true_range(high, low, close, window=5).iloc[-1])

    result["technicals"] = {
        "rsi": round(rsi_val, 1),
        "atr_14": round(atr_14, 2),
        "atr_5": round(atr_5, 2),
        "atr_pct": round(atr_14 / spot * 100, 2),
        "day_range": round(day_range, 2),
        "body_ratio": round(body_ratio * 100, 1),
    }

    # --- Options Strategy Suggestions ---
    strategies = []

    if iv_rank > 60:
        strategies.append({
            "name": "Iron Condor / Credit Spread",
            "bias": "NEUTRAL",
            "reason": "IV is elevated — sell premium, collect theta",
            "risk": "Defined risk, capped profit",
        })

    if iv_rank < 30:
        strategies.append({
            "name": "Long Calls / Puts (Directional)",
            "bias": "DIRECTIONAL",
            "reason": "IV is low — options are cheap, good for directional bets",
            "risk": "Loss limited to premium paid",
        })

    if rsi_val < 30:
        strategies.append({
            "name": "Bull Call Spread / Long Calls",
            "bias": "BULLISH",
            "reason": f"RSI oversold ({rsi_val:.0f}) — bounce setup",
            "risk": "Defined risk spread recommended",
        })
    elif rsi_val > 70:
        strategies.append({
            "name": "Bear Put Spread / Long Puts",
            "bias": "BEARISH",
            "reason": f"RSI overbought ({rsi_val:.0f}) — pullback setup",
            "risk": "Defined risk spread recommended",
        })

    if hv_5 * 100 < hv_20 * 100 * 0.7:
        strategies.append({
            "name": "Straddle / Strangle (Long Vol)",
            "bias": "VOLATILITY",
            "reason": "Short-term vol collapsed — expecting expansion",
            "risk": "Needs a big move to profit",
        })

    if not strategies:
        strategies.append({
            "name": "Wait for Setup",
            "bias": "NEUTRAL",
            "reason": "No high-conviction option strategy right now",
            "risk": "Cash is a position",
        })

    result["strategies"] = strategies

    # --- Risk Calculator Data ---
    result["risk_calc"] = {
        "max_risk_1_pct": round(spot * 0.01, 2),   # 1% of spot
        "max_risk_2_pct": round(spot * 0.02, 2),    # 2% of spot
        "atr_stop": round(spot - 2 * atr_14, 2),    # 2 ATR stop below
        "atr_target": round(spot + 3 * atr_14, 2),  # 3 ATR target above
    }

    return result


def _implied_vol_from_premium(spot, strike, T, r, premium, option_type):
    """
    Solve for implied volatility given a market premium using bisection.
    This is critical: the market price tells us what vol the market is pricing in.
    """
    if premium <= 0 or T <= 0:
        return 0.20  # fallback

    low, high = 0.01, 5.0  # IV range 1% to 500%
    for _ in range(100):
        mid = (low + high) / 2
        price = _black_scholes_price(spot, strike, T, r, mid, option_type)
        if price < premium:
            low = mid
        else:
            high = mid
        if abs(price - premium) < 0.001:
            break
    return mid


def _fetch_live_option_price(strike, expiration, option_type):
    """
    Fetch the LIVE market price and IV for a specific SPX option from the options chain.
    Returns (mid_price, market_iv, bid, ask, last_price, volume, open_interest) or None if unavailable.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^SPX")

        # Check if expiration date is available
        available_dates = ticker.options
        if expiration not in available_dates:
            return None

        chain = ticker.option_chain(expiration)
        options = chain.calls if option_type == "call" else chain.puts

        row = options[options["strike"] == float(strike)]
        if row.empty:
            # Try finding closest strike
            closest_idx = (options["strike"] - float(strike)).abs().idxmin()
            row = options.loc[[closest_idx]]
            if abs(row["strike"].values[0] - float(strike)) > 5:
                return None

        bid = float(row["bid"].values[0])
        ask = float(row["ask"].values[0])
        last = float(row["lastPrice"].values[0])
        iv = float(row["impliedVolatility"].values[0])
        volume = int(row["volume"].values[0]) if not np.isnan(row["volume"].values[0]) else 0
        oi = int(row["openInterest"].values[0]) if not np.isnan(row["openInterest"].values[0]) else 0

        # Use mid price if bid/ask available, otherwise last price
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        else:
            mid_price = last

        return {
            "mid_price": mid_price,
            "bid": bid,
            "ask": ask,
            "last_price": last,
            "market_iv": iv,
            "volume": volume,
            "open_interest": oi,
        }
    except Exception:
        return None


def analyze_contract(strike, expiration, option_type, premium, contracts=1, target_exit=None, current_price=None, current_spx=None):
    """
    Analyze probability of profit and key scenarios for a specific SPX contract.
    Auto-fetches LIVE market price and IV from the options chain for maximum accuracy.
    Falls back to user-provided current_price, then entry premium.

    Args:
        strike: Strike price (e.g. 6400)
        expiration: Expiration date string "YYYY-MM-DD"
        option_type: "call" or "put"
        premium: Entry premium paid per contract (e.g. 18.80)
        contracts: Number of contracts
        target_exit: Optional target premium to sell at
        current_price: Optional override for current market price
        current_spx: Optional override for current SPX spot price
    """
    data = _fetch_options_data()
    if data is None:
        return None

    hist = data["hist"]
    close = hist["Close"]

    # Get best available spot price
    if current_spx and current_spx > 0:
        # User override — most accurate during market hours
        spot = float(current_spx)
        spot_source = "manual"
    else:
        # Try intraday data first (more current than daily close)
        try:
            import yfinance as yf
            intraday = yf.download("^GSPC", period="1d", interval="2m", progress=False)
            if intraday is not None and len(intraday) > 0:
                spot = float(intraday["Close"].iloc[-1])
                spot_source = "intraday"
            else:
                spot = float(close.iloc[-1])
                spot_source = "daily"
        except Exception:
            spot = float(close.iloc[-1])
            spot_source = "daily"

    # Calculate DTE
    today = datetime.now().date()
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    dte = (exp_date - today).days
    if dte < 0:
        dte = 0
    T = max(dte / 365, 1 / 365)  # Time in years, min 1 day

    r = 0.045
    multiplier = 100  # SPX options multiplier

    # --- AUTO-FETCH live option price and IV from options chain ---
    live_data = _fetch_live_option_price(strike, expiration, option_type)
    iv_source = "entry_premium"  # Track where IV came from

    if live_data and live_data["mid_price"] > 0:
        # Best source: live market data from options chain
        live_price = live_data["mid_price"]
        iv_source = "live_chain"

        # Derive IV from live mid price using OUR BS model (ensures internal consistency)
        # This way theoretical_price at current spot = live mid price exactly
        implied_vol = _implied_vol_from_premium(spot, strike, T, r, live_price, option_type)

        # Override current_price with live data if user didn't provide one
        if not current_price:
            current_price = live_price
    elif current_price:
        # Second best: user-provided current price
        implied_vol = _implied_vol_from_premium(spot, strike, T, r, current_price, option_type)
        iv_source = "current_price"
        live_data = None
    else:
        # Fallback: entry premium (least accurate if spot has moved)
        implied_vol = _implied_vol_from_premium(spot, strike, T, r, premium, option_type)
        live_data = None

    # Also get historical vol for reference
    returns = np.log(close / close.shift(1)).dropna()
    hist_vol = float(returns.iloc[-20:].std() * np.sqrt(252))
    if hist_vol <= 0:
        hist_vol = 0.15

    # Use implied vol for all calculations (this is what the market is actually pricing)
    sigma = implied_vol

    # --- Current contract value using implied vol ---
    theoretical_price = _black_scholes_price(spot, strike, T, r, sigma, option_type)
    greeks = _calculate_greeks(spot, strike, T, r, sigma, option_type)

    # --- Breakeven at expiration (intrinsic only) ---
    if option_type == "call":
        breakeven_expiry = strike + premium
    else:
        breakeven_expiry = strike - premium

    # --- Swing Trade Breakeven (NOW, with time value) ---
    # Find the SPX spot price where BS option value = entry premium
    # This is the price at which your position goes from profit to loss RIGHT NOW
    def _find_breakeven_now():
        # For puts: as SPX rises, put value drops. Find where it equals entry premium.
        # For calls: as SPX drops, call value drops. Find where it equals entry premium.
        low_spot = spot * 0.80
        high_spot = spot * 1.20
        for _ in range(100):
            mid_spot = (low_spot + high_spot) / 2
            val = _black_scholes_price(mid_spot, strike, T, r, sigma, option_type)
            if option_type == "put":
                # Put value decreases as spot increases
                if val > premium:
                    low_spot = mid_spot  # need higher spot to decrease value
                else:
                    high_spot = mid_spot
            else:
                # Call value increases as spot increases
                if val < premium:
                    low_spot = mid_spot  # need higher spot to increase value
                else:
                    high_spot = mid_spot
            if abs(val - premium) < 0.01:
                break
        return mid_spot

    breakeven_now = _find_breakeven_now()
    breakeven_now_move = (breakeven_now - spot) / spot * 100

    # --- Probability calculations using implied vol ---
    def prob_above(target, time=T):
        if time <= 0:
            return 1.0 if spot > target else 0.0
        d2 = (np.log(spot / target) + (r - 0.5 * sigma**2) * time) / (sigma * np.sqrt(time))
        return float(norm.cdf(d2))

    def prob_below(target, time=T):
        return 1.0 - prob_above(target, time)

    # Probability of profit at expiration
    if option_type == "call":
        prob_profit = prob_above(breakeven_expiry)
        prob_itm = prob_above(strike)
        prob_max_loss = prob_below(strike)
    else:
        prob_profit = prob_below(breakeven_expiry)
        prob_itm = prob_below(strike)
        prob_max_loss = prob_above(strike)

    # --- P&L scenarios ---
    total_premium = premium * contracts * multiplier
    max_loss = total_premium

    # Target exit analysis
    target_data = None
    if target_exit and target_exit > premium:
        target_profit = (target_exit - premium) * contracts * multiplier
        target_pct = (target_exit - premium) / premium * 100

        # Use delta to estimate what spot move is needed NOW to hit target premium
        delta = greeks["delta"]
        if abs(delta) > 0.001:
            spot_move_needed = (target_exit - theoretical_price) / abs(delta)
        else:
            spot_move_needed = 999999

        if option_type == "call":
            spot_needed_now = spot + spot_move_needed
            prob_target_now = prob_above(spot_needed_now)
            spot_needed_exp = strike + target_exit
            prob_target_exp = prob_above(spot_needed_exp)
        else:
            spot_needed_now = spot - spot_move_needed
            prob_target_now = prob_below(spot_needed_now)
            spot_needed_exp = strike - target_exit
            prob_target_exp = prob_below(spot_needed_exp)

        target_data = {
            "target_premium": target_exit,
            "target_profit": round(target_profit, 2),
            "target_pct": round(target_pct, 1),
            "spot_needed_now": round(spot_needed_now, 2),
            "spot_move_needed": round(spot_move_needed, 2),
            "prob_swing_trade": round(prob_target_now * 100, 1),
            "spot_needed_at_expiry": round(spot_needed_exp, 2),
            "prob_at_expiry": round(prob_target_exp * 100, 1),
        }

    # --- FIXED: Scenario table shows CURRENT option value (with time value) ---
    # This is what matters for swing traders - what is the option worth NOW if SPX moves
    scenarios = []
    move_points = [
        -round(spot * 0.03), -round(spot * 0.02), -round(spot * 0.01),
        -round(spot * 0.005), 0, round(spot * 0.005),
        round(spot * 0.01), round(spot * 0.02), round(spot * 0.03)
    ]
    for move in move_points:
        scenario_spot = spot + move
        # Price the option at the new spot WITH remaining time value (not intrinsic only)
        option_value = _black_scholes_price(scenario_spot, strike, T, r, sigma, option_type)
        pnl_per_contract = (option_value - premium) * multiplier
        total_pnl = pnl_per_contract * contracts
        pnl_pct = (option_value - premium) / premium * 100 if premium > 0 else 0

        # Also show intrinsic for reference
        if option_type == "call":
            intrinsic = max(scenario_spot - strike, 0)
        else:
            intrinsic = max(strike - scenario_spot, 0)

        scenarios.append({
            "spot": round(scenario_spot, 2),
            "move": round(move, 2),
            "move_pct": round(move / spot * 100, 2),
            "option_value": round(option_value, 2),
            "intrinsic": round(intrinsic, 2),
            "time_value": round(option_value - intrinsic, 2),
            "pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 1),
        })

    # --- Theta decay schedule ---
    decay = []
    for days_ahead in [0, 1, 2, 3, 5, 7, 10, 14, 21]:
        if days_ahead > dte:
            break
        T_future = max((dte - days_ahead) / 365, 0.001)
        future_price = _black_scholes_price(spot, strike, T_future, r, sigma, option_type)
        value_lost = theoretical_price - future_price
        decay.append({
            "day": days_ahead,
            "dte_remaining": dte - days_ahead,
            "value": round(future_price, 2),
            "theta_lost": round(value_lost, 2),
            "pct_remaining": round(future_price / theoretical_price * 100, 1) if theoretical_price > 0 else 0,
        })

    # --- Expected value calculation ---
    ev_raw = theoretical_price - premium
    ev_total = ev_raw * contracts * multiplier

    # --- Current P&L based on theoretical price vs entry premium ---
    current_pnl_per = (theoretical_price - premium) * multiplier
    current_pnl = current_pnl_per * contracts
    current_pnl_pct = (theoretical_price - premium) / premium * 100 if premium > 0 else 0

    # If user provided current market price, also show P&L based on that
    if current_price:
        market_pnl_per = (current_price - premium) * multiplier
        market_pnl = market_pnl_per * contracts
        market_pnl_pct = (current_price - premium) / premium * 100 if premium > 0 else 0
    else:
        market_pnl = current_pnl
        market_pnl_pct = current_pnl_pct

    # Live chain data to include in response
    chain_info = None
    if live_data:
        chain_info = {
            "bid": live_data["bid"],
            "ask": live_data["ask"],
            "last_price": live_data["last_price"],
            "mid_price": round(live_data["mid_price"], 2),
            "volume": live_data["volume"],
            "open_interest": live_data["open_interest"],
        }

    return {
        "spot": round(spot, 2),
        "strike": strike,
        "option_type": option_type,
        "expiration": expiration,
        "dte": dte,
        "premium": premium,
        "current_price": round(current_price, 2) if current_price else None,
        "contracts": contracts,
        "multiplier": multiplier,
        "total_cost": round(total_premium, 2),
        "theoretical_price": round(theoretical_price, 2),
        "greeks": greeks,
        "breakeven": round(breakeven_now, 2),
        "breakeven_move": round(breakeven_now_move, 2),
        "breakeven_expiry": round(breakeven_expiry, 2),
        "breakeven_expiry_move": round((breakeven_expiry - spot) / spot * 100, 2),
        "prob_profit": round(prob_profit * 100, 1),
        "prob_itm": round(prob_itm * 100, 1),
        "prob_max_loss": round(prob_max_loss * 100, 1),
        "max_loss": round(max_loss, 2),
        "expected_value": round(ev_total, 2),
        "ev_per_contract": round(ev_raw * multiplier, 2),
        "implied_vol": round(implied_vol * 100, 1),
        "hist_vol": round(hist_vol * 100, 1),
        "sigma_used": round(sigma * 100, 1),
        "iv_source": iv_source,
        "current_pnl": round(current_pnl, 2),
        "current_pnl_pct": round(current_pnl_pct, 1),
        "market_pnl": round(market_pnl, 2),
        "market_pnl_pct": round(market_pnl_pct, 1),
        "live_chain": chain_info,
        "target": target_data,
        "scenarios": scenarios,
        "theta_decay": decay,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
