#!/usr/bin/env python3
"""
SPX Trading Bot - Local Web Dashboard
Run: python3 app.py
Open: http://localhost:5050
"""

import os
import sys
import json
import warnings
import traceback
from datetime import datetime

warnings.filterwarnings("ignore")

from flask import Flask, render_template, jsonify

# Ensure imports work from project dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
import numpy as np
import pandas as pd
from model import train_model, train_trend_model, predict_next_day, predict_trend, load_model, prepare_data
from data_fetcher import fetch_spx_data
from indicators import add_all_features, get_feature_columns
from gex import fetch_gex_data
from confluence import analyze_ticker, analyze_ticker_with_confidence, analyze_exit, scan_watchlist, DEFAULT_WATCHLIST
from options_analyzer import analyze_spx_options, analyze_contract
from portfolio import (add_position, remove_position, get_portfolio_status,
                       update_account_size, calculate_position_size)
from net_premium import (auto_update_today, get_premium_table, update_manual_premium,
                         fetch_net_premium_signal)
from patterns import scan_universe, scan_patterns, PATTERN_REGISTRY
from universe import get_universe, get_universe_info
from scaled_checklist import run_checklist, ACCOUNT_CONFIG, SCORE_THRESHOLDS
from trade_card import save_trade_card, get_recent_trades
import yfinance as yf

app = Flask(__name__, template_folder="templates", static_folder="static")


def signal_label(bull_prob):
    if bull_prob >= config.STRONG_BULL_THRESHOLD:
        return "STRONG BULLISH"
    elif bull_prob >= config.BULL_THRESHOLD:
        return "LEAN BULLISH"
    elif bull_prob > config.BEAR_THRESHOLD:
        return "NEUTRAL"
    elif bull_prob > config.STRONG_BEAR_THRESHOLD:
        return "LEAN BEARISH"
    else:
        return "STRONG BEARISH"


def signal_class(bull_prob):
    if bull_prob >= config.STRONG_BULL_THRESHOLD:
        return "strong-bull"
    elif bull_prob >= config.BULL_THRESHOLD:
        return "lean-bull"
    elif bull_prob > config.BEAR_THRESHOLD:
        return "neutral"
    elif bull_prob > config.STRONG_BEAR_THRESHOLD:
        return "lean-bear"
    else:
        return "strong-bear"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict")
def api_predict():
    try:
        prediction = predict_next_day()
        bull_prob = prediction["bull_probability"]
        f = prediction["features"]

        raw_df = prediction["raw_df"]

        # Start with cached values as fallback
        high_20 = float(raw_df["High"].tail(20).max())
        low_20 = float(raw_df["Low"].tail(20).min())
        prev_high = float(raw_df["High"].iloc[-1])
        prev_low = float(raw_df["Low"].iloc[-1])
        prev_close = float(raw_df["Close"].iloc[-1])
        prev_close_date = raw_df.index[-1]
        live_price = None
        live_change = None
        live_change_pct = None

        try:
            # Fetch 30 days of fresh daily data — refreshes ALL key levels, not just close
            fresh_daily = yf.download("^GSPC", period="30d", progress=False)
            if fresh_daily is not None and not fresh_daily.empty:
                if isinstance(fresh_daily.columns, pd.MultiIndex):
                    fresh_daily.columns = fresh_daily.columns.get_level_values(0)
                latest_date = fresh_daily.index[-1]
                cached_date_naive = prev_close_date.tz_localize(None) if prev_close_date.tzinfo else prev_close_date
                latest_date_naive = latest_date.tz_localize(None) if latest_date.tzinfo else latest_date
                if latest_date_naive >= cached_date_naive:
                    # Update close
                    prev_close = round(float(fresh_daily["Close"].iloc[-1]), 2)
                    prev_close_date = latest_date
                    # Update High/Low key levels from fresh data
                    prev_high = round(float(fresh_daily["High"].iloc[-1]), 2)
                    prev_low = round(float(fresh_daily["Low"].iloc[-1]), 2)
                    high_20 = round(float(fresh_daily["High"].tail(20).max()), 2)
                    low_20 = round(float(fresh_daily["Low"].tail(20).min()), 2)

            # Fetch intraday for live price during market hours
            intra = yf.download("^GSPC", period="5d", interval="2m", progress=False)
            if intra is not None and not intra.empty:
                if isinstance(intra.columns, pd.MultiIndex):
                    intra.columns = intra.columns.get_level_values(0)
                live_price = round(float(intra["Close"].iloc[-1]), 2)
                live_change = round(live_price - prev_close, 2)
                live_change_pct = round((live_change / prev_close) * 100, 2)
        except Exception as e:
            print(f"[PREDICT] Live price fetch error: {e}")

        consec_up = int(f.get("consec_up", 0))
        consec_down = int(f.get("consec_down", 0))
        streak = f"{consec_up} up" if consec_up > 0 else f"{consec_down} down"

        rsi = f.get("rsi", 50)
        rsi_label = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
        macd_dir = "EXPANDING" if f.get("macd_hist_change", 0) > 0 else "CONTRACTING"
        bb_pct = f.get("bb_pct", 0.5)
        bb_label = "UPPER BAND" if bb_pct > 0.8 else ("LOWER BAND" if bb_pct < 0.2 else "MID RANGE")
        vol_ratio = f.get("vol_ratio", 1)
        vol_label = "HIGH" if vol_ratio > 1.3 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
        trend_sma20 = "ABOVE" if f.get("dist_sma_20", 0) > 0 else "BELOW"
        trend_sma200 = "ABOVE" if f.get("dist_sma_200", 0) > 0 else "BELOW"

        # Determine prediction target date
        data_date = prediction["date"]
        prediction_for = datetime.now().strftime("%A, %B %d, %Y")
        prev_close_label = prev_close_date.strftime("%A, %B %d, %Y")

        # 5-day trend prediction
        try:
            trend = predict_trend()
            trend_bull_prob = trend["bull_probability"]
            trend_data = {
                "bull_prob": round(trend_bull_prob * 100, 1),
                "bear_prob": round(trend["bear_probability"] * 100, 1),
                "signal": signal_label(trend_bull_prob),
                "signal_class": signal_class(trend_bull_prob),
                "ml_prob": trend.get("ml_prob"),
                "trend_score": trend.get("trend_score"),
                "trend_details": trend.get("trend_details", {}),
                "adx": trend.get("adx"),
                "ret_20d": trend.get("ret_20d"),
            }
        except Exception as te:
            trend_data = {"bull_prob": None, "bear_prob": None, "signal": "UNAVAILABLE", "signal_class": "neutral", "error": str(te)}

        return jsonify({
            "success": True,
            "today": prediction_for,
            "data_date": prev_close_label,
            "prediction_for": prediction_for,
            "close": prev_close,
            "prev_close": prev_close,
            "live_price": live_price,
            "live_change": live_change,
            "live_change_pct": live_change_pct,
            "bull_prob": round(bull_prob * 100, 1),
            "bear_prob": round(prediction["bear_probability"] * 100, 1),
            "signal": signal_label(bull_prob),
            "signal_class": signal_class(bull_prob),
            "trend_5d": trend_data,
            "context": {
                "sma20": {"dir": trend_sma20, "dist": f"{f.get('dist_sma_20', 0):+.2%}"},
                "sma200": {"dir": trend_sma200, "dist": f"{f.get('dist_sma_200', 0):+.2%}"},
                "golden_cross": "YES" if f.get("sma_50_200_cross", 0) == 1 else "NO",
                "rsi": round(rsi, 1),
                "rsi_label": rsi_label,
                "macd_hist": round(f.get("macd_hist", 0), 2),
                "macd_dir": macd_dir,
                "stoch_k": round(f.get("stoch_k", 50), 1),
                "bb_pct": f"{bb_pct:.1%}",
                "bb_label": bb_label,
                "atr_pct": f"{f.get('atr_pct', 0):.2%}",
                "hvol_20": f"{f.get('hvol_20', 0):.1%}",
                "ret_1d": f"{f.get('returns_1d', 0):+.2%}",
                "ret_5d": f"{f.get('returns_5d', 0):+.2%}",
                "gap": f"{f.get('gap', 0):+.2%}",
                "streak": streak,
                "vol_ratio": round(vol_ratio, 2),
                "vol_label": vol_label,
                "adx": round(f.get("adx", 0), 1),
                "williams_r": round(f.get("williams_r", 0), 1),
                "cci": round(f.get("cci", 0), 1),
            },
            "levels": {
                "high_20": round(high_20, 2),
                "low_20": round(low_20, 2),
                "prev_high": round(prev_high, 2),
                "prev_low": round(prev_low, 2),
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/backtest")
def api_backtest():
    try:
        model, scaler, feature_cols = load_model()
        raw_df = fetch_spx_data()
        df = prepare_data(raw_df)

        n_days = 30
        recent = df.iloc[-n_days:]
        X = recent[feature_cols].values
        X_scaled = scaler.transform(X)
        probas = model.predict_proba(X_scaled)[:, 1]
        preds = (probas >= 0.5).astype(int)
        actuals = recent["target"].values

        rows = []
        correct = 0
        for i in range(len(recent)):
            pred = int(preds[i])
            actual = int(actuals[i])
            hit = pred == actual
            if hit:
                correct += 1
            rows.append({
                "date": recent.index[i].strftime("%Y-%m-%d"),
                "close": round(float(recent.iloc[i]["Close"]), 2),
                "prob": round(float(probas[i]) * 100, 1),
                "signal": signal_label(probas[i]),
                "signal_class": signal_class(probas[i]),
                "actual": "BULL" if actual == 1 else "BEAR",
                "hit": hit
            })

        return jsonify({
            "success": True,
            "rows": rows,
            "accuracy": round(correct / len(recent) * 100, 1),
            "correct": correct,
            "total": len(recent)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/train")
def api_train():
    try:
        train_model(force_refresh_data=True)
        train_trend_model(force_refresh_data=False)  # data already fresh from above
        return jsonify({"success": True, "message": "Daily + 5-day trend models retrained with latest market data."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/gex")
def api_gex():
    try:
        data = fetch_gex_data()

        # Build the chart data (top 20 strikes around spot for the bar chart)
        spot = data["spot"]
        strikes = data["strikes_data"]
        # Filter to strikes near spot and sort by absolute GEX
        near_spot = [s for s in strikes if abs(s["strike"] - spot) / spot < 0.05]
        near_spot.sort(key=lambda s: s["strike"])

        chart_strikes = [s["strike"] for s in near_spot]
        chart_call_gex = [s["call_gex"] for s in near_spot]
        chart_put_gex = [s["put_gex"] for s in near_spot]
        chart_net_gex = [s["net_gex"] for s in near_spot]

        return jsonify({
            "success": True,
            "spot": data["spot"],
            "total_gex": data["total_gex"],
            "gex_flip": data["gex_flip"],
            "dealer_position": data["dealer_position"],
            "dealer_implication": data["dealer_implication"],
            "gamma_resistance": data["gamma_resistance"],
            "gamma_support": data["gamma_support"],
            "top_call_gamma": data["top_call_gamma"],
            "top_put_gamma": data["top_put_gamma"],
            "chart_strikes": chart_strikes,
            "chart_call_gex": chart_call_gex,
            "chart_put_gex": chart_put_gex,
            "chart_net_gex": chart_net_gex,
            "per_expiry": data["per_expiry"],
            "expirations_used": data["expirations_used"],
            "data_source": data["data_source"],
            "timestamp": data["timestamp"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/confluence")
def api_confluence():
    try:
        symbol = "^GSPC"  # SPX
        result = analyze_ticker_with_confidence(symbol)
        if result is None:
            return jsonify({"success": False, "error": "Could not fetch SPX data"})

        # Convert scores to serializable format
        scores_list = []
        for key, val in result["scores"].items():
            scores_list.append({
                "id": key,
                "score": val["score"],
                "label": val["label"],
                "detail": val["detail"],
                "reason": val["reason"],
            })

        # Fetch live/intraday data for current session info
        live = {}
        try:
            # SPX indices support 2m interval, not 1m
            for sym, interval in [("^GSPC", "2m"), ("^SPX", "2m"), ("SPY", "1m")]:
                tk = yf.Ticker(sym)
                intra = tk.history(period="5d", interval=interval, prepost=True)
                daily = tk.history(period="5d")
                if intra.empty or daily.empty:
                    continue

                current = float(intra["Close"].iloc[-1])
                prev_close = float(daily["Close"].iloc[-2])

                # Today's open from daily (more reliable than intraday)
                today_open = float(daily["Open"].iloc[-1])
                if today_open <= 0:
                    today_open = float(intra["Open"].iloc[0])

                # Today's intraday high/low
                # Filter to today's bars only
                import pandas as pd
                last_date = intra.index[-1].date() if hasattr(intra.index[-1], 'date') else pd.Timestamp(intra.index[-1]).date()
                today_bars = intra[intra.index.date == last_date] if hasattr(intra.index, 'date') else intra.tail(200)
                if today_bars.empty:
                    today_bars = intra.tail(100)

                today_high = float(today_bars["High"].max())
                today_low = float(today_bars["Low"].min())

                # If using SPY, scale to SPX (~10x)
                scale = 1.0
                if sym == "SPY":
                    scale = result["price"] / current if current > 0 else 10.0
                    current *= scale
                    today_open *= scale
                    prev_close *= scale
                    today_high *= scale
                    today_low *= scale

                live = {
                    "current": round(current, 2),
                    "open": round(today_open, 2),
                    "prev_close": round(prev_close, 2),
                    "high": round(today_high, 2),
                    "low": round(today_low, 2),
                    "change_from_close": round(current - prev_close, 2),
                    "change_from_close_pct": round((current - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0,
                    "change_from_open": round(current - today_open, 2),
                    "change_from_open_pct": round((current - today_open) / today_open * 100, 2) if today_open > 0 else 0,
                    "day_range_pct": round((current - today_low) / (today_high - today_low) * 100, 1) if today_high != today_low else 50.0,
                    "source": sym,
                }
                break
        except Exception:
            pass

        # Serialize reversal scores the same way
        reversal_scores_list = []
        if result.get("reversal"):
            rev = result["reversal"]
            for key, val in rev["scores"].items():
                reversal_scores_list.append({
                    "id":     key,
                    "score":  val["score"],
                    "label":  val["label"],
                    "detail": val["detail"],
                    "reason": val["reason"],
                })

        resp = {
            "success": True,
            "symbol": "SPX",
            "price": result["price"],
            "change_1d": result["change_1d"],
            "signal": result["signal"],
            "signal_class": result["signal_class"],
            "long_count": result["long_count"],
            "short_count": result["short_count"],
            "neutral_count": result["neutral_count"],
            "strength": result["strength"],
            "threshold": result["threshold"],
            "scores": scores_list,
            "timestamp": result["timestamp"],
        }
        if live:
            resp["live"] = live
        if result.get("confidence"):
            resp["confidence"] = result["confidence"]
        if result.get("reversal"):
            rev = result["reversal"]
            resp["reversal"] = {
                "signal":        rev["signal"],
                "signal_class":  rev["signal_class"],
                "long_count":    rev["long_count"],
                "short_count":   rev["short_count"],
                "neutral_count": rev["neutral_count"],
                "regime":        rev["regime"],
                "regime_class":  rev["regime_class"],
                "regime_note":   rev["regime_note"],
                "vix":           rev["vix"],
                "threshold":     rev["threshold"],
                "scores":        reversal_scores_list,
            }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/confidence")
def api_confidence():
    try:
        result = analyze_ticker("^GSPC")
        if result is None:
            return jsonify({"success": False, "error": "Could not fetch SPX data"})
        from confidence import assess_confidence
        conf = assess_confidence(result)
        conf["success"] = True
        return jsonify(conf)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/exit")
def api_exit():
    try:
        from flask import request
        symbol = request.args.get("symbol", "^GSPC")
        position_type = request.args.get("type", "long")  # "long" or "short"
        entry_price = request.args.get("entry", None)

        if entry_price is None:
            return jsonify({"success": False, "error": "Missing entry price. Use ?entry=XXXX"})

        entry_price = float(entry_price)
        display_symbol = "SPX" if symbol in ("^GSPC", "^SPX") else symbol

        result = analyze_exit(symbol, position_type, entry_price)
        if result is None:
            return jsonify({"success": False, "error": f"Could not fetch data for {symbol}"})

        reasons_list = []
        for key, val in result["reasons"].items():
            reasons_list.append({
                "id": key,
                "triggered": val["triggered"],
                "label": val["label"],
                "detail": val["detail"],
                "reason": val["reason"],
                "urgency": val["urgency"],
            })

        return jsonify({
            "success": True,
            "symbol": display_symbol,
            "position_type": result["position_type"],
            "entry_price": result["entry_price"],
            "current_price": result["current_price"],
            "pnl_pct": result["pnl_pct"],
            "exit_signal": result["exit_signal"],
            "exit_class": result["exit_class"],
            "triggered_count": result["triggered_count"],
            "total_checks": result["total_checks"],
            "reasons": reasons_list,
            "stop_price": result["stop_price"],
            "partial_target": result["partial_target"],
            "full_target": result["full_target"],
            "timestamp": result["timestamp"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/scan")
def api_scan():
    try:
        from flask import request
        custom = request.args.get("tickers", None)
        symbols = None
        if custom:
            symbols = [t.strip().upper() for t in custom.split(",") if t.strip()]
        results = scan_watchlist(symbols)
        rows = []
        for r in results:
            scores_list = []
            for key, val in r["scores"].items():
                scores_list.append({
                    "id": key,
                    "score": val["score"],
                    "label": val["label"],
                    "detail": val["detail"],
                    "reason": val["reason"],
                })
            rows.append({
                "symbol": r["symbol"],
                "price": r["price"],
                "change_1d": r["change_1d"],
                "change_5d": r["change_5d"],
                "signal": r["signal"],
                "signal_class": r["signal_class"],
                "long_count": r["long_count"],
                "short_count": r["short_count"],
                "neutral_count": r["neutral_count"],
                "strength": r["strength"],
                "rsi": r["rsi"],
                "adx": r["adx"],
                "vol_ratio": r["vol_ratio"],
                "scores": scores_list,
            })

        signals_found = sum(1 for r in rows if r["signal"] != "NO SIGNAL")
        return jsonify({
            "success": True,
            "rows": rows,
            "total_scanned": len(rows),
            "signals_found": signals_found,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/options/contract")
def api_options_contract():
    try:
        from flask import request
        strike = float(request.args.get("strike", 0))
        expiration = request.args.get("expiration", "")
        option_type = request.args.get("type", "call")
        premium = float(request.args.get("premium", 0))
        contracts = int(request.args.get("contracts", 1))
        target_exit = request.args.get("target", None)
        current_price = request.args.get("current_price", None)
        current_spx = request.args.get("current_spx", None)
        if target_exit:
            target_exit = float(target_exit)
        if current_price:
            current_price = float(current_price)
        if current_spx:
            current_spx = float(current_spx)

        if strike <= 0 or not expiration or premium <= 0:
            return jsonify({"success": False, "error": "Strike, expiration, and premium are required"})

        result = analyze_contract(strike, expiration, option_type, premium, contracts, target_exit, current_price, current_spx)
        if result is None:
            return jsonify({"success": False, "error": "Could not fetch SPX data"})
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/options")
def api_options():
    try:
        result = analyze_spx_options()
        if result is None:
            return jsonify({"success": False, "error": "Could not fetch SPX options data"})
        result["success"] = True

        # Fetch live/intraday data (same as confluence)
        live = {}
        try:
            import pandas as pd
            for sym, interval in [("^GSPC", "2m"), ("^SPX", "2m"), ("SPY", "1m")]:
                tk = yf.Ticker(sym)
                intra = tk.history(period="5d", interval=interval, prepost=True)
                daily = tk.history(period="5d")
                if intra.empty or daily.empty:
                    continue

                current = float(intra["Close"].iloc[-1])
                prev_close = float(daily["Close"].iloc[-2])

                today_open = float(daily["Open"].iloc[-1])
                if today_open <= 0:
                    today_open = float(intra["Open"].iloc[0])

                last_date = intra.index[-1].date() if hasattr(intra.index[-1], 'date') else pd.Timestamp(intra.index[-1]).date()
                today_bars = intra[intra.index.date == last_date] if hasattr(intra.index, 'date') else intra.tail(200)
                if today_bars.empty:
                    today_bars = intra.tail(100)

                today_high = float(today_bars["High"].max())
                today_low = float(today_bars["Low"].min())

                scale = 1.0
                if sym == "SPY":
                    scale = result["spot"] / current if current > 0 else 10.0
                    current *= scale
                    today_open *= scale
                    prev_close *= scale
                    today_high *= scale
                    today_low *= scale

                live = {
                    "current": round(current, 2),
                    "open": round(today_open, 2),
                    "prev_close": round(prev_close, 2),
                    "high": round(today_high, 2),
                    "low": round(today_low, 2),
                    "change_from_close": round(current - prev_close, 2),
                    "change_from_close_pct": round((current - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0,
                    "change_from_open": round(current - today_open, 2),
                    "change_from_open_pct": round((current - today_open) / today_open * 100, 2) if today_open > 0 else 0,
                    "day_range_pct": round((current - today_low) / (today_high - today_low) * 100, 1) if today_high != today_low else 50.0,
                    "source": sym,
                }
                break
        except Exception:
            pass

        if live:
            result["live"] = live

        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/portfolio")
def api_portfolio():
    try:
        status = get_portfolio_status()
        status["success"] = True
        return jsonify(status)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/portfolio/add", methods=["POST"])
def api_portfolio_add():
    try:
        from flask import request
        data = request.get_json() if request.is_json else {}
        # Also support form/query params
        if not data:
            data = request.args.to_dict()

        symbol = data.get("symbol", "").upper()
        entry_price = float(data.get("entry_price", 0))
        shares = float(data.get("shares", 0))
        position_type = data.get("position_type", "long")
        target_low = float(data["target_low"]) if data.get("target_low") else None
        target_high = float(data["target_high"]) if data.get("target_high") else None
        stop_loss = float(data["stop_loss"]) if data.get("stop_loss") else None
        notes = data.get("notes", "")

        if not symbol or entry_price <= 0 or shares <= 0:
            return jsonify({"success": False, "error": "Symbol, entry price, and shares are required"})

        pos = add_position(symbol, entry_price, shares, position_type,
                          target_low, target_high, stop_loss, notes)
        return jsonify({"success": True, "position": pos})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/portfolio/remove", methods=["POST"])
def api_portfolio_remove():
    try:
        from flask import request
        data = request.get_json() if request.is_json else {}
        if not data:
            data = request.args.to_dict()

        pos_id = data.get("id", "")
        exit_price = float(data["exit_price"]) if data.get("exit_price") else None

        if not pos_id:
            return jsonify({"success": False, "error": "Position ID required"})

        removed = remove_position(pos_id, exit_price)
        if removed is None:
            return jsonify({"success": False, "error": "Position not found"})
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/portfolio/account", methods=["POST"])
def api_portfolio_account():
    try:
        from flask import request
        data = request.get_json() if request.is_json else {}
        if not data:
            data = request.args.to_dict()

        size = float(data.get("size", 0))
        if size <= 0:
            return jsonify({"success": False, "error": "Account size must be positive"})
        update_account_size(size)
        return jsonify({"success": True, "account_size": size})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/risk-calc")
def api_risk_calc():
    try:
        from flask import request
        account = float(request.args.get("account", 10000))
        entry = float(request.args.get("entry", 0))
        stop = float(request.args.get("stop", 0))
        risk_pct = float(request.args.get("risk", 2.0))

        if entry <= 0 or stop <= 0:
            return jsonify({"success": False, "error": "Entry and stop price required"})

        result = calculate_position_size(account, entry, stop, risk_pct)
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/live")
def api_live():
    """Fetch current SPX price using 2-minute intraday bars (same as Confluence tab)."""
    try:
        import pandas as pd
        price = prev_close = day_open = day_high = day_low = None
        volume = 0
        source = None

        # --- Primary: 2-minute intraday bars (most current during market hours) ---
        for sym, interval in [("^GSPC", "2m"), ("^SPX", "2m"), ("SPY", "1m")]:
            try:
                tk = yf.Ticker(sym)
                intra = tk.history(period="5d", interval=interval, prepost=True)
                daily = tk.history(period="5d")
                if intra.empty or daily.empty or len(daily) < 2:
                    continue

                price = float(intra["Close"].iloc[-1])
                prev_close = float(daily["Close"].iloc[-2])
                day_open = float(daily["Open"].iloc[-1])
                volume = int(daily["Volume"].iloc[-1])

                # Intraday high/low filtered to today only
                last_date = intra.index[-1].date() if hasattr(intra.index[-1], "date") else pd.Timestamp(intra.index[-1]).date()
                today_bars = intra[intra.index.date == last_date] if hasattr(intra.index, "date") else intra.tail(200)
                if today_bars.empty:
                    today_bars = intra.tail(100)

                day_high = float(today_bars["High"].max())
                day_low = float(today_bars["Low"].min())

                # Scale SPY to SPX equivalent
                if sym == "SPY":
                    ref_daily = yf.Ticker("^GSPC").history(period="2d")
                    spx_ref = float(ref_daily["Close"].iloc[-1]) if not ref_daily.empty else price * 10
                    scale = spx_ref / price
                    price *= scale; day_open *= scale
                    prev_close *= scale; day_high *= scale; day_low *= scale

                source = sym
                break
            except Exception:
                continue

        # --- Fallback: daily bars (works outside market hours) ---
        if price is None:
            for sym in ["^GSPC", "^SPX"]:
                try:
                    tk = yf.Ticker(sym)
                    hist = tk.history(period="5d")
                    if hist.empty or len(hist) < 2:
                        continue
                    today = hist.iloc[-1]
                    price = float(today["Close"])
                    day_open = float(today["Open"])
                    day_high = float(today["High"])
                    day_low = float(today["Low"])
                    volume = int(today["Volume"])
                    prev_close = float(hist.iloc[-2]["Close"])
                    source = sym + "_daily"
                    break
                except Exception:
                    continue

        if price is None:
            return jsonify({"success": False, "error": "Could not fetch SPX price from any source"})

        change = price - prev_close
        change_pct = change / prev_close * 100 if prev_close > 0 else 0
        day_range_pct = (price - day_low) / (day_high - day_low) * 100 if day_high != day_low else 50.0

        return jsonify({
            "success": True,
            "price": round(price, 2),
            "open": round(day_open, 2),
            "high": round(day_high, 2),
            "low": round(day_low, 2),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "day_range_pct": round(day_range_pct, 1),
            "source": source,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def _sanitize(obj):
    """Recursively convert numpy scalars to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


@app.route("/api/patterns")
def api_patterns():
    try:
        from flask import request
        import time
        custom = request.args.get("tickers", None)
        mode = request.args.get("mode", "default")
        min_grade = request.args.get("min_grade", "B")
        pattern_filter = request.args.get("patterns", None)

        if custom:
            symbols = [t.strip().upper() for t in custom.split(",") if t.strip()]
        else:
            symbols = get_universe(mode)

        patterns = None
        if pattern_filter:
            pattern_names = [p.strip() for p in pattern_filter.split(",")]
            patterns = {k: v for k, v in PATTERN_REGISTRY.items() if k in pattern_names}

        t0 = time.time()
        results = scan_universe(symbols, patterns=patterns, min_grade=min_grade)
        elapsed = round(time.time() - t0, 1)

        return jsonify({
            "success": True,
            "results": _sanitize(results),
            "total_scanned": len(symbols),
            "patterns_found": len(results),
            "scan_time_sec": elapsed,
            "available_patterns": list(PATTERN_REGISTRY.keys()),
            "universe_info": get_universe_info(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/net-premium")
def api_net_premium():
    try:
        # Auto-calculate today's net premium and save
        today_result = auto_update_today()

        # Get historical table
        table = get_premium_table(days=20)

        return jsonify({
            "success": True,
            "today": today_result.get("calculation") if today_result else None,
            "table": table,
            "signal": fetch_net_premium_signal(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/net-premium/update", methods=["POST"])
def api_net_premium_update():
    try:
        from flask import request
        data = request.get_json() if request.is_json else {}
        if not data:
            data = request.args.to_dict()

        date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        net_premium = data.get("net_premium")
        total_premium = data.get("total_premium")

        if net_premium is None:
            return jsonify({"success": False, "error": "net_premium value required"})

        net_premium = float(str(net_premium).replace(",", "").replace("$", ""))
        if total_premium:
            total_premium = float(str(total_premium).replace(",", "").replace("$", ""))

        entry = update_manual_premium(date_str, net_premium, total_premium)
        return jsonify({"success": True, "entry": entry})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/checklist")
def api_checklist():
    try:
        from flask import request
        symbol = request.args.get("symbol", "^GSPC").strip().upper()
        if symbol == "SPX":
            symbol = "^GSPC"
        raw_balance = request.args.get("balance", None)
        account_balance = float(raw_balance) if raw_balance else None
        result = run_checklist(symbol, account_balance=account_balance)
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/trade-card", methods=["POST"])
def api_save_trade_card():
    try:
        from flask import request
        card = request.get_json()
        if not card:
            return jsonify({"success": False, "error": "No trade card data received"})
        result = save_trade_card(card)
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/trade-log")
def api_trade_log():
    try:
        from flask import request
        n = int(request.args.get("n", 20))
        trades = get_recent_trades(n)
        return jsonify({"success": True, "trades": trades, "total": len(trades)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    print("\n  SPX Trading Bot Dashboard")
    print("  Open in your browser: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
