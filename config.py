"""Configuration for SPX Predictive Trading Bot."""

TICKER = "^GSPC"  # S&P 500 Index
DATA_PERIOD_YEARS = 5
TRADING_WINDOW_HOURS = 2  # First 2 hours from open (9:30 - 11:30 ET)

# Feature engineering
LOOKBACK_PERIODS = [5, 10, 20, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
STOCH_PERIOD = 14
WILLIAMS_PERIOD = 14
CCI_PERIOD = 20
ADX_PERIOD = 14

# Model — Daily (next-day candle direction)
MODEL_PATH = "model/spx_model.pkl"
SCALER_PATH = "model/spx_scaler.pkl"
FEATURE_PATH = "model/spx_features.pkl"

# Model — 5-Day Trend (will price be higher in 5 trading days?)
TREND_MODEL_PATH = "model/spx_trend_model.pkl"
TREND_SCALER_PATH = "model/spx_trend_scaler.pkl"
TREND_FEATURE_PATH = "model/spx_trend_features.pkl"
TREND_HORIZON = 5  # trading days forward
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 500
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_DEPTH = 12

# Prediction thresholds
STRONG_BULL_THRESHOLD = 0.65
BULL_THRESHOLD = 0.55
BEAR_THRESHOLD = 0.45
STRONG_BEAR_THRESHOLD = 0.35

# Data cache
CACHE_DIR = "cache"
CACHE_EXPIRY_HOURS = 4
