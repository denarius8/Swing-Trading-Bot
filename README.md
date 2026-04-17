# SPX Swing Trading Dashboard

A real-time S&P 500 trading research dashboard combining dual machine learning signals, 10-indicator confluence, options analytics, gamma exposure mapping, chart pattern detection, and portfolio tracking — all in a single browser-based interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

| Tab | Description |
|-----|-------------|
| **Confluence** | 10-indicator consensus signal (RSI, VWAP, EMA Stack, MACD, Bollinger Bands, Volume, Key Levels, ADX, Stochastic, 200 SMA). Fires ENTER LONG/SHORT when 7+ indicators align. Includes 5 leading confidence indicators and net premium tracking. |
| **SPX Options** | Full options chain analysis — Greeks (delta, gamma, theta, vega), IV surface, 0DTE-30DTE premium decay, implied move, key strike levels. Includes Scaled Entry Checklist and Trade Card Enforcer. |
| **Portfolio** | Track open positions with real-time P&L, exit signals, stop/target alerts, position sizing calculator, and trade history. |
| **Scanner** | Batch confluence scan across 38-570 tickers. Filter by signal type (ENTER LONG, ENTER SHORT, HOLD). |
| **Patterns** | AskLivermore-style chart pattern detection — VCP, Bull Flag, New Uptrend, Golden Pocket, Livermore Breakout — with A+ to B quality grading. |
| **ML Signal** | **Dual ML system**: tomorrow's candle direction (55%+ accuracy) + 5-day swing trend (hybrid ML + 8-factor rules score). Both signals shown side-by-side with alignment indicator. |
| **GEX** | Dealer gamma exposure by strike with full-price labels, gridlines, and hover tooltips. Shows GEX flip level, key gamma concentrations, and dealer positioning. |
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

On first load, both ML models will auto-train using 5 years of SPX daily data from Yahoo Finance (~1,250 trading days). This takes about 60 seconds.

**macOS tip**: Double-click `start_dashboard.command` in Finder to start the server and auto-open the browser.

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

**Signal threshold**: 7+ indicators aligned = actionable ENTER LONG or ENTER SHORT signal.

### Confidence Overlay (5 Leading Indicators)

Grades confluence signals as HIGH / MEDIUM / LOW confidence before price moves:

1. **News Sentiment** — Live headline scanning for high-impact events (Fed, CPI, earnings, geopolitical). Parses the current yfinance news format with bullish/bearish keyword scoring.
2. **Crude Oil Correlation** — Brent/WTI trend vs. SPX direction (risk-on/risk-off)
3. **Dealer Positioning** — GEX-derived long/short gamma context + put/call ratio + IV rank
4. **Multi-Timeframe Heikin-Ashi** — Weekly (3x weight), Daily (2x), 4-Hour (1x), 90-Min (1x). Dominant trend determined by weighted score, not vote count.
5. **Net Premium Flow** — SPX options net dollar flow streak (4+ consecutive days triggers signal)

### Dual ML Signal

The ML Signal tab shows **two independent predictions** side-by-side:

#### Tomorrow (Daily Candle Model)
- **Target**: Will tomorrow's close be above its open?
- **Algorithm**: VotingClassifier (Random Forest 500 trees + Gradient Boosting 300 trees)
- **Features**: 79 engineered indicators (momentum, volatility, trend, volume, calendar)
- **Training**: 5 years daily data, 80/20 time-series split (no data leakage)
- **Accuracy**: ~55-58% on out-of-sample data
- **Use case**: Short-term entry timing. Reads overbought/oversold conditions for the next candle.

#### 5-Day Trend (Swing Model)
- **Target**: Hybrid — 40% ML probability + 60% rules-based 8-factor trend score
- **8 Trend Factors**: Price above 50 SMA, Price above 200 SMA, Golden Cross (50/200), ADX > 25, +DI > -DI, 20-day return positive, Higher highs (20-day), MACD histogram positive
- **Use case**: Swing trading context. Shows whether the larger trend supports a long or short bias regardless of short-term overbought readings.
- **Display**: Trend score (e.g. 8/8), color-coded factor badges, ML probability, ADX, 20-day return

#### Signal Alignment Indicator
When both signals agree → **"✓ Both signals agree — higher conviction"**

When they diverge → **"⚠ Signals diverge — daily pullback likely within larger trend"**

Example: SPX at all-time highs with RSI 69, Stoch 99.5:
- Daily: **STRONG BEARISH** (35% bull) — overbought, tomorrow's candle likely pulls back
- 5-Day: **STRONG BULLISH** (77% bull) — trend score 8/8, all factors aligned
- Alignment: Signals diverge — normal bull market consolidation, not a reversal

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

- **Auto-calculation**: Approximates net premium from yfinance options chains (volume × mid-price × 100, weighted by DTE: 0-7DTE = 2x, 7-30DTE = 1x, 30+ = 0.5x)
- **Manual override**: Enter real values from Unusual Whales for higher accuracy. Source badge shows AUTO or UW per row.
- **Signal**: 4+ consecutive positive days = bullish (+1), 4+ negative = bearish (-1)
- **Display**: 20-day rolling table with streak counter matching Unusual Whales format

### Scaled Entry Checklist

A 5-point scoring system that auto-populates from live market data and determines position size tier before entry:

| Score | Tier | Action |
|-------|------|--------|
| < 2.5 | **NO TRADE** | Cash is correct. Neither direction confirmed. |
| 2.5–3.4 | **STARTER** | 25% of max outlay. Defined small risk. |
| 3.5–4.4 | **ADD** | 50% total. Only after starter is live and in your favor. |
| 4.5–5.0 | **FULL** | 100% outlay. All checks confirmed. |

**5 Checks (auto-populated from live data):**
1. **Structural Trend** — Weekly + Daily Heikin-Ashi alignment + Golden Cross (SMA50 > SMA200)
2. **Momentum Alignment** — 4H + 1H HA confirmation with volume
3. **Execution Layer (15M)** — 15M HA + MACD direction for optimal entry timing
4. **Extension Risk** — Distance from EMA20: ≤3% ideal, ≤5% elevated, >5% overextended
5. **Divergence Warnings** — Weekly MACD cross, VIX spike (>5%), GLD hedge flow (>1.5%)

Supports any ticker (default: `^GSPC` / SPX). Works alongside any confluence signal.

### Trade Card Enforcer

Forces a written plan before every entry. Unlocked automatically when checklist score ≥ 2.5 (STARTER). Required fields:
- Contract details (ticker, direction, expiry, strike, entry price, number of contracts)
- Trade thesis (free text, required)
- Exit targets: SPX level at T1, T2, T3 + stop SPX level
- Auto-validates max risk ($) against account 2% rule

Saved to `trade_log.json` (git-ignored, stays local). Last 20 trades visible in the Trade Log table.

### GEX Chart

Net GEX by Strike chart with:
- **Full strike price labels** — shows 6,800, 6,900, 7,000 etc. (~10 evenly spaced, not truncated)
- **Dashed vertical gridlines** at each labeled strike for alignment
- **Hover tooltips** — exact strike + Net GEX value with green/red coloring

## Project Structure

```
.
├── app.py                 # Flask server + API routes (localhost:5050)
├── config.py              # All tunable parameters
├── model.py               # Dual ML model: daily candle + 5-day trend
├── data_fetcher.py        # Yahoo Finance data + caching + macOS xattr fix
├── indicators.py          # 79 technical indicators including swing-specific features
├── confluence.py          # 10-indicator signal engine + exit scoring
├── confidence.py          # 5 leading indicators confidence overlay
├── options_analyzer.py    # SPX options Greeks + analysis
├── gex.py                 # Gamma exposure calculation
├── net_premium.py         # Net premium tracking + manual Unusual Whales override
├── patterns.py            # Chart pattern detection + grading
├── universe.py            # Ticker lists (S&P 500, popular, ETFs)
├── portfolio.py           # Position tracking + P&L
├── scaled_checklist.py    # 5-point entry scoring: STARTER / ADD / FULL / NO TRADE
├── trade_card.py          # Pre-trade plan enforcer + trade log storage
├── bot.py                 # CLI interface for signals
├── start_dashboard.command # Double-click to launch server + open browser (macOS)
├── templates/
│   └── index.html         # Dashboard UI (single-page app)
├── cache/                 # Auto-generated market data cache (git-ignored)
└── model/                 # Auto-generated trained model files (git-ignored)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/confluence` | GET | 10-indicator confluence signal |
| `/api/confidence` | GET | Leading indicators confidence grade |
| `/api/predict` | GET | Dual ML signal: daily + 5-day trend |
| `/api/backtest` | GET | Last 30 days prediction vs. actual |
| `/api/train` | GET | Retrain both models with latest data |
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
| `/api/checklist` | GET | Scaled entry checklist score + tier (add `?symbol=AAPL` for any ticker) |
| `/api/trade-card` | POST | Save a trade card to trade_log.json |
| `/api/trade-log` | GET | Last N trade cards (`?n=20`) |

## CLI Usage

```bash
python3 bot.py              # Print today's ML signal
python3 bot.py --train      # Train/retrain both models
python3 bot.py --refresh    # Force refresh data + retrain
python3 bot.py --backtest   # Show 20-day backtest results
```

## Configuration

All tunable parameters are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_PERIOD_YEARS` | 5 | Historical data lookback |
| `N_ESTIMATORS` | 500 | Random Forest trees |
| `MAX_DEPTH` | 12 | Daily model tree depth |
| `TREND_HORIZON` | 5 | 5-day trend model forward window |
| `STRONG_BULL_THRESHOLD` | 0.65 | Strong bull probability cutoff |
| `CACHE_EXPIRY_HOURS` | 4 | Market data cache duration |

## Data Sources

All market data comes from **Yahoo Finance** via the `yfinance` library (free, no API key required):

- Daily/intraday OHLCV for SPX (`^GSPC`) and individual tickers
- SPX options chains (`^SPX`) for Greeks, GEX, and net premium
- News headlines for sentiment analysis (parsed from current yfinance format)
- Crude oil prices (`BZ=F`, `CL=F`) for correlation

## Known Behaviors

- **News Sentiment**: Uses live yfinance headline format. Keywords scored across 60+ bullish/bearish terms. High-impact events (Fed, CPI, earnings) automatically flag confidence as LOW regardless of signal direction.
- **Net Premium AUTO values**: Approximation only — cannot distinguish bought vs. sold options from public data. Use Unusual Whales manual override for accuracy.
- **5-Day Trend accuracy**: Test set accuracy varies with market regime. The hybrid approach (ML + rules) is more stable across different market conditions than pure ML alone.
- **macOS file permissions**: `data_fetcher.py` automatically clears `com.apple.provenance` extended attributes before writing cache files to prevent permission errors.

## Disclaimer

This tool is for **research and educational purposes only**. It is not financial advice. Past performance of signals and predictions does not guarantee future results. Always do your own analysis and manage risk appropriately.
