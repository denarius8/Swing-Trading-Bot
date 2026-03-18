"""
Portfolio Tracker - Track open positions with exit alerts
Stores positions locally in portfolio.json
"""

import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from confluence import _fetch_ticker_data, _calculate_indicators, score_exit

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")


def _load_portfolio():
    """Load portfolio from JSON file."""
    if not os.path.exists(PORTFOLIO_FILE):
        return {"positions": [], "closed": [], "account_size": 10000}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
        # Ensure all required keys exist
        if "positions" not in data:
            data["positions"] = []
        if "closed" not in data:
            data["closed"] = []
        if "account_size" not in data:
            data["account_size"] = 10000
        return data
    except:
        return {"positions": [], "closed": [], "account_size": 10000}


def _save_portfolio(data):
    """Save portfolio to JSON file."""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_position(symbol, entry_price, shares, position_type="long",
                 target_low=None, target_high=None, stop_loss=None, notes=""):
    """Add a new position to the portfolio."""
    portfolio = _load_portfolio()

    # Generate unique ID
    pos_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    total_cost = entry_price * shares

    # Default stop and targets if not provided
    if stop_loss is None:
        stop_loss = entry_price * 0.95 if position_type == "long" else entry_price * 1.05
    if target_low is None:
        target_low = entry_price * 1.05 if position_type == "long" else entry_price * 0.95
    if target_high is None:
        target_high = entry_price * 1.10 if position_type == "long" else entry_price * 0.90

    position = {
        "id": pos_id,
        "symbol": symbol.upper(),
        "entry_price": entry_price,
        "shares": shares,
        "total_cost": round(total_cost, 2),
        "position_type": position_type,
        "target_low": target_low,
        "target_high": target_high,
        "stop_loss": stop_loss,
        "notes": notes,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "status": "open",
    }

    portfolio["positions"].append(position)
    _save_portfolio(portfolio)
    return position


def remove_position(pos_id, exit_price=None):
    """Close/remove a position. Optionally log the exit."""
    portfolio = _load_portfolio()

    removed = None
    remaining = []
    for p in portfolio["positions"]:
        if p["id"] == pos_id:
            removed = p
        else:
            remaining.append(p)

    if removed is None:
        return None

    portfolio["positions"] = remaining

    # Log to closed positions
    if exit_price:
        removed["exit_price"] = exit_price
        removed["exit_date"] = datetime.now().strftime("%Y-%m-%d")
        if removed["position_type"] == "long":
            removed["pnl"] = round((exit_price - removed["entry_price"]) * removed["shares"], 2)
            removed["pnl_pct"] = round((exit_price - removed["entry_price"]) / removed["entry_price"] * 100, 2)
        else:
            removed["pnl"] = round((removed["entry_price"] - exit_price) * removed["shares"], 2)
            removed["pnl_pct"] = round((removed["entry_price"] - exit_price) / removed["entry_price"] * 100, 2)
        removed["status"] = "closed"
        portfolio["closed"].append(removed)

    _save_portfolio(portfolio)
    return removed


def update_account_size(size):
    """Update account size for risk calculations."""
    portfolio = _load_portfolio()
    portfolio["account_size"] = size
    _save_portfolio(portfolio)
    return size


def get_portfolio_status():
    """Get full portfolio status with live prices and exit signals."""
    portfolio = _load_portfolio()

    positions_status = []
    total_invested = 0
    total_current = 0
    total_pnl = 0
    alerts = []

    for pos in portfolio["positions"]:
        symbol = pos["symbol"]

        # Fetch current price
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
            current_price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
        except:
            continue

        entry_price = pos["entry_price"]
        shares = pos["shares"]
        is_long = pos["position_type"] == "long"

        # P&L
        if is_long:
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * shares
            pnl_pct = (entry_price - current_price) / entry_price * 100

        current_value = current_price * shares
        total_invested += pos["total_cost"]
        total_current += current_value
        total_pnl += pnl

        # Day change
        day_change = current_price - prev_close
        day_change_pct = day_change / prev_close * 100

        # Progress toward target
        target_low = pos.get("target_low", entry_price * 1.05)
        target_high = pos.get("target_high", entry_price * 1.10)
        stop_loss = pos.get("stop_loss", entry_price * 0.95)

        if is_long:
            if target_high > entry_price:
                progress = (current_price - entry_price) / (target_high - entry_price) * 100
            else:
                progress = 0
        else:
            if entry_price > target_high:
                progress = (entry_price - current_price) / (entry_price - target_high) * 100
            else:
                progress = 0
        progress = max(-100, min(200, progress))

        # Run exit analysis
        exit_data = None
        try:
            from confluence import analyze_exit
            yahoo_symbol = symbol
            if symbol == "SPX":
                yahoo_symbol = "^GSPC"
            exit_result = analyze_exit(yahoo_symbol, pos["position_type"], entry_price)
            if exit_result:
                exit_data = {
                    "exit_signal": exit_result["exit_signal"],
                    "exit_class": exit_result["exit_class"],
                    "triggered_count": exit_result["triggered_count"],
                    "total_checks": exit_result["total_checks"],
                }
        except:
            pass

        # Generate alerts
        if is_long:
            if current_price <= stop_loss:
                alerts.append({"symbol": symbol, "type": "STOP HIT", "urgency": "critical",
                              "message": f"{symbol} hit stop loss at ${stop_loss:.2f}"})
            elif current_price >= target_high:
                alerts.append({"symbol": symbol, "type": "TARGET HIT", "urgency": "high",
                              "message": f"{symbol} reached full target at ${target_high:.2f}"})
            elif current_price >= target_low:
                alerts.append({"symbol": symbol, "type": "PARTIAL TARGET", "urgency": "medium",
                              "message": f"{symbol} in profit target zone (${target_low:.2f}-${target_high:.2f})"})
        else:
            if current_price >= stop_loss:
                alerts.append({"symbol": symbol, "type": "STOP HIT", "urgency": "critical",
                              "message": f"{symbol} hit stop loss at ${stop_loss:.2f}"})
            elif current_price <= target_high:
                alerts.append({"symbol": symbol, "type": "TARGET HIT", "urgency": "high",
                              "message": f"{symbol} reached full target at ${target_high:.2f}"})
            elif current_price <= target_low:
                alerts.append({"symbol": symbol, "type": "PARTIAL TARGET", "urgency": "medium",
                              "message": f"{symbol} in profit target zone"})

        # Days held
        try:
            entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_date).days
        except:
            days_held = 0

        positions_status.append({
            "id": pos["id"],
            "symbol": symbol,
            "position_type": pos["position_type"],
            "entry_price": entry_price,
            "current_price": round(current_price, 2),
            "shares": shares,
            "total_cost": pos["total_cost"],
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "day_change": round(day_change, 2),
            "day_change_pct": round(day_change_pct, 2),
            "target_low": target_low,
            "target_high": target_high,
            "stop_loss": stop_loss,
            "progress": round(progress, 1),
            "days_held": days_held,
            "entry_date": pos["entry_date"],
            "notes": pos.get("notes", ""),
            "exit_analysis": exit_data,
        })

    # Risk calculator
    account_size = portfolio["account_size"]
    risk_1_pct = account_size * 0.01
    risk_2_pct = account_size * 0.02
    risk_5_pct = account_size * 0.05

    return {
        "positions": positions_status,
        "alerts": alerts,
        "summary": {
            "total_positions": len(positions_status),
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / total_invested * 100, 2) if total_invested > 0 else 0,
        },
        "account": {
            "size": account_size,
            "risk_1_pct": round(risk_1_pct, 2),
            "risk_2_pct": round(risk_2_pct, 2),
            "risk_5_pct": round(risk_5_pct, 2),
            "invested": round(total_invested, 2),
            "available": round(account_size - total_invested, 2),
            "exposure_pct": round(total_invested / account_size * 100, 1) if account_size > 0 else 0,
        },
        "closed": portfolio["closed"][-10:],  # Last 10 closed trades
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def calculate_position_size(account_size, entry_price, stop_price, risk_pct=2.0):
    """Calculate optimal position size based on risk percentage."""
    risk_amount = account_size * (risk_pct / 100)
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return {"error": "Stop price must differ from entry price"}

    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry_price

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "max_loss": round(shares * risk_per_share, 2),
        "position_pct": round(position_value / account_size * 100, 1),
    }
