#!/usr/bin/env python3
"""
SPX Predictive Trading Bot - Daily Research Signal Generator

Usage:
    python3 bot.py              # Run daily prediction
    python3 bot.py --train      # Train/retrain the model
    python3 bot.py --refresh    # Force refresh data and retrain
    python3 bot.py --backtest   # Show recent backtest results
"""

import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

import config
from model import train_model, predict_next_day, load_model, prepare_data
from data_fetcher import fetch_spx_data
from indicators import add_all_features, get_feature_columns

W = 53  # Box inner width


def row(label, value):
    """Format a box row with label and value."""
    content = f"  {label:<18s} {value}"
    return f"│{content:<{W}s}│"


def signal_label(bull_prob):
    """Convert probability to a readable signal."""
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


def signal_bar(bull_prob, width=30):
    """Visual probability bar."""
    filled = int(bull_prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"BEAR |{bar}| BULL"


def box_top():
    print("┌" + "─" * W + "┐")

def box_mid():
    print("├" + "─" * W + "┤")

def box_bot():
    print("└" + "─" * W + "┘")

def box_title(title):
    print(f"│{title:^{W}s}│")


def print_market_context(prediction):
    """Print key indicator readings for context."""
    f = prediction["features"]

    print()
    box_top()
    box_title("MARKET CONTEXT")
    box_mid()

    # Trend
    trend_sma = "ABOVE" if f.get("dist_sma_20", 0) > 0 else "BELOW"
    trend_200 = "ABOVE" if f.get("dist_sma_200", 0) > 0 else "BELOW"
    print(row("Price vs SMA(20)", f"{trend_sma} ({f.get('dist_sma_20', 0):+.2%})"))
    print(row("Price vs SMA(200)", f"{trend_200} ({f.get('dist_sma_200', 0):+.2%})"))
    golden = "YES" if f.get("sma_50_200_cross", 0) == 1 else "NO"
    print(row("Golden Cross", golden))

    box_mid()

    # Momentum
    rsi = f.get("rsi", 50)
    rsi_label = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL")
    print(row("RSI(14)", f"{rsi:.1f} ({rsi_label})"))

    macd_hist = f.get("macd_hist", 0)
    macd_dir = "EXPANDING" if f.get("macd_hist_change", 0) > 0 else "CONTRACTING"
    print(row("MACD Hist", f"{macd_hist:+.2f} ({macd_dir})"))

    stoch_k = f.get("stoch_k", 50)
    print(row("Stochastic %K", f"{stoch_k:.1f}"))

    box_mid()

    # Volatility
    bb_pct = f.get("bb_pct", 0.5)
    bb_label = "UPPER BAND" if bb_pct > 0.8 else ("LOWER BAND" if bb_pct < 0.2 else "MID RANGE")
    print(row("BB Position", f"{bb_pct:.1%} ({bb_label})"))
    print(row("ATR(%)", f"{f.get('atr_pct', 0):.2%}"))
    print(row("HVol(20d)", f"{f.get('hvol_20', 0):.1%}"))

    box_mid()

    # Recent action
    ret_1d = f.get("returns_1d", 0)
    ret_5d = f.get("returns_5d", 0)
    gap = f.get("gap", 0)
    consec_up = int(f.get("consec_up", 0))
    consec_down = int(f.get("consec_down", 0))
    streak = f"{consec_up} up" if consec_up > 0 else f"{consec_down} down"

    print(row("1-Day Return", f"{ret_1d:+.2%}"))
    print(row("5-Day Return", f"{ret_5d:+.2%}"))
    print(row("Last Gap", f"{gap:+.2%}"))
    print(row("Streak", streak))

    vol_ratio = f.get("vol_ratio", 1)
    vol_label = "HIGH" if vol_ratio > 1.3 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
    print(row("Volume Ratio", f"{vol_ratio:.2f}x avg ({vol_label})"))

    box_bot()


def run_backtest(n_days=20):
    """Show model predictions vs actual results for recent days."""
    model, scaler, feature_cols = load_model()
    raw_df = fetch_spx_data()
    df = prepare_data(raw_df)

    recent = df.iloc[-n_days:]
    X = recent[feature_cols].values
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]
    preds = (probas >= 0.5).astype(int)
    actuals = recent["target"].values

    print("\n" + "=" * 70)
    print(f"  BACKTEST - Last {n_days} Trading Days")
    print("=" * 70)
    print(f"  {'Date':<12s} {'Close':>9s} {'Prob':>6s} {'Signal':<16s} {'Actual':<8s} {'Result'}")
    print("  " + "-" * 64)

    correct = 0
    for i in range(len(recent)):
        date = recent.index[i].strftime("%Y-%m-%d")
        close = recent.iloc[i]["Close"]
        prob = probas[i]
        pred = preds[i]
        actual = actuals[i]
        signal = signal_label(prob)
        actual_label = "BULL" if actual == 1 else "BEAR"
        hit = pred == actual
        if hit:
            correct += 1
        result = "  HIT" if hit else " MISS"

        print(f"  {date:<12s} {close:>9.2f} {prob:>5.1%} {signal:<16s} {actual_label:<8s} {result}")

    accuracy = correct / len(recent)
    print("  " + "-" * 64)
    print(f"  Accuracy: {correct}/{len(recent)} ({accuracy:.1%})")
    print()


def run_prediction():
    """Run the daily prediction and display results."""
    prediction = predict_next_day()

    bull_prob = prediction["bull_probability"]
    bear_prob = prediction["bear_probability"]
    signal = signal_label(bull_prob)
    close = prediction["close"]
    date = prediction["date"]

    print("\n" + "=" * 55)
    print("  SPX PREDICTIVE TRADING BOT - DAILY SIGNAL")
    print("  For research purposes only - not financial advice")
    print("=" * 55)

    print(f"\n  Data as of:   {date.strftime('%A, %B %d, %Y')}")
    print(f"  SPX Close:    {close:,.2f}")

    box_top()
    box_title(f"NEXT SESSION SIGNAL:  {signal}")
    box_bot()

    print(f"\n  Bull Probability: {bull_prob:.1%}")
    print(f"  Bear Probability: {bear_prob:.1%}")
    print(f"\n  {signal_bar(bull_prob)}")

    print_market_context(prediction)

    # Key levels
    raw_df = prediction["raw_df"]
    high_20 = raw_df["High"].tail(20).max()
    low_20 = raw_df["Low"].tail(20).min()
    prev_high = raw_df["High"].iloc[-1]
    prev_low = raw_df["Low"].iloc[-1]

    print()
    box_top()
    box_title("KEY LEVELS")
    box_mid()
    print(row("20-Day High", f"{high_20:,.2f}"))
    print(row("Prior Day High", f"{prev_high:,.2f}"))
    print(row("Prior Day Low", f"{prev_low:,.2f}"))
    print(row("20-Day Low", f"{low_20:,.2f}"))
    box_bot()

    print("\n  DISCLAIMER: This is a research tool only.")
    print("  Past performance does not predict future results.")
    print("  Do not use this as your sole basis for trading.\n")


def main():
    args = sys.argv[1:]

    if "--train" in args:
        train_model()
    elif "--refresh" in args:
        train_model(force_refresh_data=True)
    elif "--backtest" in args:
        run_backtest()
    else:
        run_prediction()


if __name__ == "__main__":
    main()
