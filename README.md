# SPX Swing Trading Dashboard

A real-time S&P 500 trading research dashboard combining machine learning predictions, 10-indicator confluence signals, options analytics, gamma exposure mapping, chart pattern detection, and portfolio tracking — all in a single browser-based interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

| Tab | Description |
|-----|-------------|
| **Confluence** | 10-indicator consensus signal (RSI, VWAP, EMA Stack, MACD, Bollinger Bands, Volume, Key Levels, ADX, Stochastic, 200 SMA). Fires ENTER LONG/SHORT when 7+ indicators align. Includes 5 leading confidence indicators and net premium tracking. |
| **SPX Options** | Full options chain analysis — Greeks (delta, gamma, theta, vega), IV surface, 0DTE-30DTE premium decay, implied move, key strike levels. |
| **Portfolio** | Track open positions with real-time P&L, exit signals, stop/target alerts, position sizing calculator, and trade history. |
| **Scanner** | Batch confluence scan across 38-570 tickers. Filter by signal type (ENTER LONG, ENTER SHORT, HOLD). |
| **Patterns** | AskLivermore-style chart pattern detection — VCP, Bull Flag, New Uptrend, Golden Pocket, Livermore Breakout — with A+ to B quality grading. |
| **ML Signal** | Ensemble ML model (Random Forest + Gradient Boosting) predicting next-day SPX direction with probability scores and market context. |
| **GEX** | Dealer gamma exposure by strike. Shows GEX flip level, key gamma concentrations, and whether dealers are long/short gamma. |
| **Backtest** | Last 30 days of ML predictions vs. actual outcomes with cumulative accuracy tracking. |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/denarius8/Swing-Trading-Bot.git
cd Swing-Trading-Bot
```

### 2. Install dependencies

```bash
pip install flask yfinance pandas numpy scikit-learn ta scipy joblib
```

### 3. Run the dashboard

```bash
python3 app.py
```

Open **http://localhost:5050** in your browser.

On first load, the ML model will auto-train using 5 years of SPX daily data from Yahoo Finance (~1,250 trading days). This takes about 30 seconds.

## How It Works

### Confluence System (10 Indicators)

The core signal engine scores 10 technical indicators on a -1 to +1 scale:

1. **RSI** — Oversold/overbought zones + divergence detection
2. **VWAP** — Institutional flow (price vs. volume-weighted average)
3. **EMA Stack** — 9 > 21 > 50 alignment (bullish) or inverse (bearish)
4. **MACD** — Crossovers + histogram expansion
5. **Bollinger Bands** — Squeeze breakouts, band touches, mean reversion
6. **Volume** — Confirmation on up/down moves vs. 20-day average
7. **Key Levels** — Testing 20-day highs (resistance) or lows (support)
8. **ADX** — Trend strength (>25) with directional index
9. **Stochastic** — Crossovers in extreme zones (<20 or >80)
10. **200 SMA** — Long-term trend filter

**Signal threshold**: 7+ indicators aligned = actionable signal.

### Confidence Overlay (5 Leading Indicators)

Grades confluence signals as HIGH / MEDIUM / LOW confidence:

1. **News Sentiment** — Scans headlines for high-impact events (Fed, CPI, earnings)
2. **Crude Oil Correlation** — Brent/WTI trend vs. SPX direction
3. **Dealer Positioning** — GEX-derived long/short gamma context
4. **Multi-Timeframe Heikin-Ashi** — Weekly (3x weight), Daily (2x), 4-Hour (1x), 90-Min (1x)
5. **Net Premium Flow** — SPX options net dollar flow streak (4+ consecutive days = signal)

### ML Model

- **Algorithm**: VotingClassifier (Random Forest 500 trees + Gradient Boosting 300 trees)
- **Features**: 70+ engineered indicators (momentum, volatility, trend, volume, calendar)
- **Target**: Next-day close > open (binary: bullish/bearish)
- **Training**: 5 years daily data, 80/20 time-series split (no data leakage)
- **Accuracy**: Typically 52-58% on out-of-sample data

### Pattern Scanner

Detects chart patterns with quality grades (A+ to B):

| Pattern | Description |
|---------|-------------|
| **VCP** | Volatility Contraction Pattern — successive tighter ranges with declining volume |
| **Bull Flag** | Strong pole (8%+) followed by tight consolidation channel |
| **New Uptrend** | Price > rising 50 SMA > rising 200 SMA with volume confirmation |
| **Golden Pocket** | Price at 61.8%-65% Fibonacci retracement of recent swing |
| **Livermore Breakout** | Breakout from tight consolidation to new highs on volume surge |

Each detection includes entry zone, stop loss, target price, and risk/reward ratio.

### Net Premium Tracker

Tracks net dollar flow into SPX options as a leading indicator:

- **Auto-calculation**: Approximates net premium from yfinance options chains (volume x mid-price, weighted by DTE)
- **Manual override**: Enter real values from Unusual Whales for higher accuracy
- **Signal**: 4+ consecutive positive days = bullish, 4+ negative = bearish
- **Display**: 20-day rolling table with streak counter and source badges (AUTO/UW)

## Project Structure

```
.
├── app.py                 # Flask server + API routes (localhost:5050)
├── config.py              # All tunable parameters
├── model.py               # ML model training and prediction
├── data_fetcher.py        # Yahoo Finance data + caching
├── indicators.py          # 70+ technical indicator calculations
├── confluence.py          # 10-indicator signal engine + exit scoring
├── confidence.py          # Leading indicators confidence overlay
├── options_analyzer.py    # SPX options Greeks + analysis
├── gex.py                 # Gamma exposure calculation
├── net_premium.py         # Net premium tracking + manual override
├── patterns.py            # Chart pattern detection + grading
├── universe.py            # Ticker lists (S&P 500, popular, ETFs)
├── portfolio.py           # Position tracking + P&L
├── bot.py                 # CLI interface for signals
├── templates/
│   └── index.html         # Dashboard UI (single-page app)
├── cache/                 # Auto-generated market data cache
└── model/                 # Auto-generated trained model files
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/confluence` | GET | 10-indicator confluence signal |
| `/api/confidence` | GET | Leading indicators confidence grade |
| `/api/predict` | GET | ML next-day prediction + probability |
| `/api/backtest` | GET | Last 30 days prediction vs. actual |
| `/api/train` | GET | Retrain model with latest data |
| `/api/options` | GET | Full SPX options analysis |
| `/api/options/contract` | GET | Single contract Greeks + P&L |
| `/api/gex` | GET | Gamma exposure by strike |
| `/api/scan` | GET | Batch confluence scan |
| `/api/patterns` | GET | Chart pattern scan |
| `/api/net-premium` | GET | Net premium table + signal |
| `/api/net-premium/update` | POST | Manual premium data entry |
| `/api/portfolio` | GET | Portfolio status + positions |
| `/api/portfolio/add` | POST | Add position |
| `/api/portfolio/remove` | POST | Close position |
| `/api/portfolio/account` | POST | Update account size |
| `/api/risk-calc` | GET | Position sizing calculator |
| `/api/live` | GET | Current SPX price + day stats |
| `/api/exit` | GET | Exit analysis for open position |

## CLI Usage

```bash
python3 bot.py              # Print today's ML signal
python3 bot.py --train      # Train/retrain the model
python3 bot.py --refresh    # Force refresh data + retrain
python3 bot.py --backtest   # Show 20-day backtest results
```

## Configuration

All tunable parameters are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_PERIOD_YEARS` | 5 | Historical data lookback |
| `N_ESTIMATORS` | 500 | Random Forest trees |
| `MAX_DEPTH` | 12 | Tree depth limit |
| `STRONG_BULL_THRESHOLD` | 0.65 | Strong bull probability cutoff |
| `CACHE_EXPIRY_HOURS` | 4 | Market data cache duration |

## Data Sources

All market data comes from **Yahoo Finance** via the `yfinance` library (free, no API key required):

- Daily/intraday OHLCV for SPX (`^GSPC`) and individual tickers
- SPX options chains (`^SPX`) for Greeks, GEX, and net premium
- News headlines for sentiment analysis
- Crude oil prices (`BZ=F`, `CL=F`) for correlation

## Disclaimer

This tool is for **research and educational purposes only**. It is not financial advice. Past performance of signals and predictions does not guarantee future results. Always do your own analysis and manage risk appropriately.
