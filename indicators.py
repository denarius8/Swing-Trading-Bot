"""Technical indicator calculations and feature engineering for SPX."""

import numpy as np
import pandas as pd
import ta
import config


def add_all_features(df):
    """Add all technical indicators and engineered features to the dataframe."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    # --- Price Action Features ---
    df["returns_1d"] = df["Close"].pct_change()
    df["returns_2d"] = df["Close"].pct_change(2)
    df["returns_5d"] = df["Close"].pct_change(5)
    df["returns_10d"] = df["Close"].pct_change(10)

    # Intraday range and body
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["body"] = (df["Close"] - df["Open"]) / df["Close"]
    df["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Gap (open vs previous close)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["gap_filled"] = ((df["gap"] > 0) & (df["Low"] <= df["Close"].shift(1)) |
                        (df["gap"] < 0) & (df["High"] >= df["Close"].shift(1))).astype(int)

    # --- Moving Averages ---
    for period in config.LOOKBACK_PERIODS:
        df[f"sma_{period}"] = df["Close"].rolling(period).mean()
        df[f"ema_{period}"] = df["Close"].ewm(span=period).mean()
        df[f"dist_sma_{period}"] = (df["Close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]
        df[f"dist_ema_{period}"] = (df["Close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]

    # MA crossovers
    df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)
    df["sma_10_50_cross"] = (df["sma_10"] > df["sma_50"]).astype(int)
    df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
    df["ema_5_20_cross"] = (df["ema_5"] > df["ema_20"]).astype(int)

    # --- Momentum Indicators ---
    df["rsi"] = ta.momentum.rsi(df["Close"], window=config.RSI_PERIOD)
    df["rsi_5"] = ta.momentum.rsi(df["Close"], window=5)

    macd_ind = ta.trend.MACD(df["Close"], window_slow=config.MACD_SLOW,
                              window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()
    df["macd_hist_change"] = df["macd_hist"].diff()

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"],
                                              window=config.STOCH_PERIOD)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"],
                                               lbp=config.WILLIAMS_PERIOD)
    df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=config.CCI_PERIOD)
    df["roc_10"] = ta.momentum.roc(df["Close"], window=10)
    df["roc_20"] = ta.momentum.roc(df["Close"], window=20)

    # --- Volatility Indicators ---
    bb = ta.volatility.BollingerBands(df["Close"], window=config.BB_PERIOD, window_dev=config.BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"],
                                                  window=config.ATR_PERIOD)
    df["atr_pct"] = df["atr"] / df["Close"]

    # Historical volatility
    df["hvol_10"] = df["returns_1d"].rolling(10).std() * np.sqrt(252)
    df["hvol_20"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)

    # --- Trend Indicators ---
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=config.ADX_PERIOD)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # --- Volume Features ---
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]
    df["vol_change"] = df["Volume"].pct_change()

    # --- Day-of-Week and Calendar Features ---
    df["dow"] = df.index.dayofweek  # Mon=0, Fri=4
    df["dom"] = df.index.day
    df["month"] = df.index.month
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_month_start"] = (df["dom"] <= 3).astype(int)
    df["is_month_end"] = (df["dom"] >= 27).astype(int)

    # --- Pattern Features ---
    # Consecutive up/down days
    up = (df["Close"] > df["Close"].shift(1)).astype(int)
    down = (df["Close"] < df["Close"].shift(1)).astype(int)
    df["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumsum()
    df["consec_down"] = down.groupby((down != down.shift()).cumsum()).cumsum()

    # Distance from N-day high/low
    for period in [10, 20, 50]:
        df[f"dist_high_{period}"] = (df["Close"] - df["High"].rolling(period).max()) / df["Close"]
        df[f"dist_low_{period}"] = (df["Close"] - df["Low"].rolling(period).min()) / df["Close"]

    # --- Mean Reversion Features ---
    df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()
    df["zscore_50"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).std()

    # --- Swing / Trend-Specific Features (5-day model) ---
    # Longer-horizon returns
    df["returns_20d"] = df["Close"].pct_change(20)
    df["returns_50d"] = df["Close"].pct_change(50)

    # Longer RSI — better for swing trend identification
    df["rsi_21"] = ta.momentum.rsi(df["Close"], window=21)

    # MA spread: how far is 50 SMA from 200 SMA (trend strength/direction)
    df["ma_spread_50_200"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

    # Trend regime: is the market in a trending or ranging environment
    df["trend_regime"] = (df["adx"] > 25).astype(int)
    df["strong_trend_regime"] = (df["adx"] > 40).astype(int)

    # Higher highs / higher lows over 10 and 20 days (swing structure)
    df["higher_high_10"] = (df["High"] > df["High"].shift(10)).astype(int)
    df["higher_low_10"] = (df["Low"] > df["Low"].shift(10)).astype(int)
    df["higher_high_20"] = (df["High"] > df["High"].shift(20)).astype(int)
    df["higher_low_20"] = (df["Low"] > df["Low"].shift(20)).astype(int)

    # Trend score: sum of higher highs + higher lows over recent swings
    df["swing_trend_score"] = (df["higher_high_10"] + df["higher_low_10"] +
                                df["higher_high_20"] + df["higher_low_20"])

    # Momentum acceleration: is the 5d return accelerating vs 10d?
    df["momentum_accel"] = df["returns_5d"] - df["returns_10d"]

    # Volume trend: is volume higher on up days vs down days (accumulation)?
    # Use a simpler calculation: rolling sum of signed volume
    df["signed_volume"] = df["Volume"] * np.sign(df["returns_1d"].fillna(0))
    df["vol_accumulation"] = df["signed_volume"].rolling(10).sum() / df["Volume"].rolling(10).mean().replace(0, np.nan)
    df["vol_accumulation"] = df["vol_accumulation"].fillna(0)

    # Clean up inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def get_feature_columns(df):
    """Return list of feature column names (excludes OHLCV and target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "target",
               "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
               "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
               "bb_upper", "bb_lower", "bb_mid", "vol_sma_20", "atr"}
    return [c for c in df.columns if c not in exclude]
