# Swing Trade Bot

A real-time S&P 500 / Nasdaq-100 trading research dashboard combining dual machine learning signals, a 12-indicator context-aware confluence engine, options analytics, gamma exposure mapping, chart pattern detection, and portfolio tracking — all in a single browser-based interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

| Tab | Description |
|-----|-------------|
| **SPX** | Dual-panel system: **Trend System** (12-indicator context-aware engine — fires ENTER LONG/SHORT at 8+/12, STAY at 6–7, LEAN at 5) and **Reversal Entry** (6 contrarian indicators that catch bottoms/tops the trend system misses). Includes GEX Flip ⚡ and Flow Flip ⚡ event badges, Fast Pullback/Breakout Alert, trend context label (UPTREND/DOWNTREND/RANGE), and 6 leading confidence indicators. |
| **NDX** | Same 12-indicator confluence engine for Nasdaq-100. Sub-tabs: Confluence, QQQ Options (with Contract Analyzer), ML Signal, GEX, Backtest. |
| **Options** | Full SPX/NDX options chain analysis — Greeks (delta, gamma, theta, vega), IV surface, 0DTE-30DTE premium decay, implied move, key strike levels. Includes Scaled Entry Checklist and Trade Card Enforcer. |
| **Portfolio** | Track open positions with real-time P&L, exit signals, stop/target alerts, position sizing calculator, and trade history. |
| **Scanner** | Batch confluence scan across 38–570 tickers. Filter by signal type (ENTER LONG, ENTER SHORT, STAY LONG, STAY SHORT, LEAN). |
| **Patterns** | AskLivermore-style chart pattern detection across 15 patterns (5 per phase) with A+ to B quality grading, entry zone, stop, target, and R:R. Scans 38–570 tickers. |
| **ML Signal** | **Dual ML system**: tomorrow's candle direction (55%+ accuracy) + 5-day swing trend (hybrid ML + 8-factor rules score). Both signals shown side-by-side with alignment indicator. |
| **GEX** | Dealer gamma exposure by strike with full-price labels, gridlines, and hover tooltips. Shows GEX flip level, key gamma concentrations, and dealer positioning. |
| **Backtest** | Last 30 days of ML predictions vs. actual outcomes with cumulative accuracy tracking. |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/bemoneytalks/Swing-Trading-Bot.git
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

### Confluence System (12 Indicators, Context-Aware)

The core signal engine scores 12 technical indicators on a -1 to +1 scale. Critically, **indicators are scored differently depending on the detected market regime** (confirmed uptrend, confirmed downtrend, or range-bound) — preventing mean-reversion signals from firing bearish during a strong uptrend.

#### Indicators 1–10 (Context-Aware Technical)

| # | Indicator | Uptrend Mode | Downtrend Mode | Range Mode |
|---|-----------|-------------|----------------|------------|
| 1 | **RSI** | >50 = bullish momentum; <40 = pullback re-entry | ≤50 = bearish; >60 = dead cat | <30 oversold / >70 overbought |
| 2 | **VWAP** | Price above = bullish | Price below = bearish | Standard |
| 3 | **EMA Stack** | 9>21>50 = bullish | 9<21<50 = bearish | Standard |
| 4 | **MACD** | Crossover + expanding histogram | — | Standard crossover |
| 5 | **Bollinger Bands** | >75% = riding upper band (momentum); <25% = pullback entry | <25% = riding lower band | Standard mean-reversion |
| 6 | **Volume** | Above-average on up day | Above-average on down day | — |
| 7 | **Key Levels** | New 20-day high = breakout confirmation (not resistance) | New 20-day low = breakdown | Standard support/resistance |
| 8 | **ADX** | >25 with +DI > -DI | >25 with -DI > +DI | <25 = no trend |
| 9 | **Stochastic** | K<30 = pullback entry; K>80 = neutral momentum | K>70 = dead cat short entry; K<20 = neutral momentum | Standard <20/>80 crossover |
| 10 | **200 SMA** | Price above = long-term uptrend | Price below = long-term downtrend | — |

#### Indicator 11 — GEX Regime
- **Long Gamma** (+1): Dealers are hedged long gamma — they buy dips and sell rips, creating a structural bid. Calmer, mean-reverting tape.
- **Short Gamma** (-1): Dealers are short gamma — they sell weakness and buy strength, amplifying moves. Dangerous, volatile environment.
- **⚡ GEX Flip**: Fires immediately the day the regime changes direction (long→short or short→long).

#### Indicator 12 — Net Premium Flow
- **Positive net premium** (+1): Call flow dominant — institutional money positioned long.
- **Negative net premium** (-1): Put flow dominant — institutional hedging/short positioning.
- **⚡ Day-1 Flip**: Signal fires immediately when net premium changes sign (no multi-day wait).
- Streak tiers add context: `flip` (immediate) → `early` (1–2d) → `sustained` (3–6d) → `conviction` (7+d).

#### Signal States

| Score | Signal | Meaning |
|-------|--------|---------|
| 8+/12 | **ENTER LONG / ENTER SHORT** | High-conviction directional trade |
| 6–7/12 | **STAY LONG / STAY SHORT** | Trend intact, not ideal for fresh entry |
| 5/12 | **LEAN LONG / LEAN SHORT** | Informational lean only |
| <5/12 | **NO SIGNAL** | No edge detected |

#### Trend Context Detection

The engine detects the current regime before scoring:
- **UPTREND**: Price > 50 SMA > 200 SMA + ADX > 25 + +DI > -DI
- **DOWNTREND**: Price < 50 SMA < 200 SMA + ADX > 25 + -DI > +DI
- **RANGE**: Neither confirmed — uses traditional mean-reversion logic

This prevents false bearish signals during strong uptrends (e.g., RSI 70 in a confirmed uptrend = continued momentum, not overbought).

### Fast Pullback / Breakout Alert

Fires at **3+ triggers** before the 8/12 confluence threshold is hit — catching fast selloffs and breakouts that develop faster than the daily indicators can confirm.

**6 Triggers (symmetric — bearish and bullish):**
1. VIX Spike ≥15% (or VIX Collapse ≥15%)
2. EMA 9 crossed below/above EMA 21
3. Price broke below/above 50 SMA on above-average volume
4. GEX Flip event (regime change)
5. Net Premium Flip event (flow changed sign)
6. Volume Climax (1.5x+ volume with directional close)

### Confidence Overlay (6 Leading Indicators)

Grades confluence signals as **HIGH / MEDIUM / LOW** confidence based on conditions that exist *before* price moves:

1. **News Sentiment** — Live headline scanning for high-impact events (Fed, CPI, earnings, geopolitical). 60+ bullish/bearish keyword scoring with high-impact event flag.
2. **Crude Oil Correlation** — Brent/WTI trend vs. SPX direction (risk-on/risk-off macro signal).
3. **Dealer Positioning** — Options flow put/call ratio + IV rank (distinct from GEX regime).
4. **Multi-Timeframe Heikin-Ashi** — Weekly (3x weight), Daily (2x), 4-Hour (1x), 90-Min (1x). Dominant trend by weighted score.
5. **Net Premium Flow** — Streak direction and flip events as a leading flow indicator.
6. **GEX Regime** — Long/short gamma regime as a structural market condition indicator.

**Grade logic**: 0 conflicts = HIGH, 1 conflict = MEDIUM, 2+ conflicts = LOW. High-impact news event always forces LOW regardless.

### Dual ML Signal

The ML Signal tab shows **two independent predictions** side-by-side:

#### Tomorrow (Daily Candle Model)
- **Target**: Will tomorrow's close be above its open?
- **Algorithm**: VotingClassifier (Random Forest 500 trees + Gradient Boosting 300 trees)
- **Features**: 79 engineered indicators (momentum, volatility, trend, volume, calendar)
- **Training**: 5 years daily data, 80/20 time-series split (no data leakage)
- **Accuracy**: ~55–58% on out-of-sample data

#### 5-Day Trend (Swing Model)
- **Target**: Hybrid — 40% ML probability + 60% rules-based 8-factor trend score
- **8 Trend Factors**: Price above 50 SMA, Price above 200 SMA, Golden Cross (50/200), ADX > 25, +DI > -DI, 20-day return positive, Higher highs (20-day), MACD histogram positive

#### Signal Alignment Indicator
When both signals agree → **"✓ Both signals agree — higher conviction"**

When they diverge → **"⚠ Signals diverge — daily pullback likely within larger trend"**

### Pattern Scanner

Detects 15 chart patterns with quality grades (A+ to B). Each detection includes entry zone, stop loss, target price, and risk/reward ratio.

#### Phase 1 — Core Long Setups
| Pattern | Description |
|---------|-------------|
| **VCP** | Volatility Contraction Pattern — successive tighter ranges with declining volume (Minervini) |
| **Bull Flag** | Strong pole (8%+) followed by tight consolidation channel, volume dry-up |
| **New Uptrend** | Price > rising 50 SMA > rising 200 SMA, golden cross or price reclaim |
| **Golden Pocket** | Price at 61.8%–65% Fibonacci retracement of recent swing with bounce confirmation |
| **Livermore Breakout** | Breakout from tight consolidation to new highs on volume surge |

#### Phase 2 — Expanded Coverage
| Pattern | Direction | Description |
|---------|-----------|-------------|
| **Stage 1 Base** | LONG | Weinstein: 150d MA flattening, price consolidating above, quiet volume |
| **Close to Bottom** | LONG | Within 15% of 52-week low, RSI oversold, volume pickup (accumulation) |
| **Earnings Gap** | LONG | Gap >4% on 1.5x+ volume, price still holding above gap open |
| **BX Momentum** | LONG/SHORT | EMA 8/21/50 stack aligned, ADX >20, RSI in trend zone |
| **Parabolic Short** | SHORT | >2σ above 50 SMA, RSI >70, volume spike / exhaustion candle (blow-off top) |

#### Phase 3 — Complex Geometric
| Pattern | Direction | Description |
|---------|-----------|-------------|
| **Cup & Handle** | LONG | U-shaped base (15–50% depth) + tight handle consolidation, volume dry-up |
| **Inverse H&S** | LONG | Three troughs (middle lowest), neckline break — bullish reversal |
| **Head & Shoulders** | SHORT | Three peaks (middle highest), neckline break — distribution top |
| **Munger 200W** | LONG | Within 8% of 200-week MA (~1000d) from above — long-term value entry |
| **Stage 3 Top** | SHORT | Weinstein: 150d MA rolling over + price below MA, volume expanding |

### Net Premium Tracker

Tracks net dollar flow into SPX/NDX options as a leading indicator:

- **Auto-calculation**: Approximates net premium from yfinance options chains (volume × mid-price × 100, weighted by DTE: 0–7DTE = 2x, 7–30DTE = 1x, 30+ = 0.5x)
- **Manual override**: Enter real values from Unusual Whales for higher accuracy. Source badge shows AUTO or UW per row.
- **Day-1 Flip Signal**: Net premium changing sign fires an immediate signal — no multi-day wait.
- **Display**: 20-day rolling table with streak counter and flip event detection matching Unusual Whales format.

### Reversal Entry System

A parallel 6-indicator panel in the Confluence tab that fires independently of the 12-indicator trend system. Designed to catch capitulation bottoms and blow-off tops that trend-following indicators structurally miss.

| Indicator | LONG signal | SHORT signal |
|-----------|-------------|--------------|
| **RSI Extreme** | RSI < 35 | RSI > 65 |
| **Stochastic Extreme** | K < 25 | K > 75 |
| **Bollinger Extreme** | BB% < 15% | BB% > 85% |
| **Net Premium** | Positive flow | Negative flow |
| **VIX Regime** | VIX > 28 (panic/elevated) | VIX < 14 (complacency) |
| **Volume Reversal Candle** | Vol > 1.1x avg, up day, close in upper 40% of range | Vol > 1.1x avg, down day, close in lower 40% of range |

**Thresholds**: 4+/6 = ENTER, 3/6 = WATCH, <3 = NO SIGNAL. Regime badge (NORMAL / ELEVATED / EXTREME) adjusts signal context based on VIX level and distance from 200 SMA.

### Scaled Entry Checklist

A 5-point scoring system that auto-populates from live market data and determines position size tier before entry.

| Score | Tier | Action |
|-------|------|--------|
| < 2.5 | **NO TRADE** | Cash is correct. Neither direction confirmed. |
| 2.5–3.4 | **STARTER** | 25% of max outlay. Defined small risk. |
| 3.5–4.4 | **ADD** | 50% total. Only after starter is live and in your favor. |
| 4.5–5.0 | **FULL** | 100% outlay. All checks confirmed. |

### Trade Card Enforcer

Forces a written plan before every entry. Unlocked automatically when checklist score ≥ 2.5 (STARTER). Required fields: contract details, trade thesis, exit targets (T1/T2/T3 SPX levels + stop), and max risk auto-validated against the 2% account rule.

### GEX Chart

Net GEX by Strike chart with full strike price labels, dashed vertical gridlines, and hover tooltips. Shows GEX flip level and dealer positioning regime.

## Project Structure

```
.
├── app.py                 # Flask server + API routes (localhost:5050)
├── config.py              # All tunable parameters
├── model.py               # Dual ML model: daily candle + 5-day trend
├── data_fetcher.py        # Yahoo Finance data + caching + macOS xattr fix
├── indicators.py          # 79 technical indicators including swing-specific features
├── confluence.py          # 12-indicator context-aware signal engine + exit scoring
├── confidence.py          # 6 leading indicators confidence overlay
├── options_analyzer.py    # SPX/NDX options Greeks + analysis
├── gex.py                 # Gamma exposure calculation + regime signal
├── net_premium.py         # Net premium tracking + Day-1 flip detection + manual UW override
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
| `/api/confluence` | GET | 12-indicator confluence signal with trend_context, GEX, net premium, fast pullback |
| `/api/confidence` | GET | 6 leading indicators confidence grade |
| `/api/predict` | GET | Dual ML signal: daily + 5-day trend |
| `/api/backtest` | GET | Last 30 days prediction vs. actual |
| `/api/train` | GET | Retrain both models with latest data |
| `/api/options` | GET | Full SPX/NDX options analysis |
| `/api/options/contract` | GET | Single contract Greeks + P&L |
| `/api/gex` | GET | Gamma exposure by strike + regime signal |
| `/api/scan` | GET | Batch confluence scan |
| `/api/patterns` | GET | Chart pattern scan |
| `/api/net-premium` | GET | Net premium table + Day-1 flip signal |
| `/api/net-premium/update` | POST | Manual premium data entry (Unusual Whales) |
| `/api/portfolio` | GET | Portfolio status + positions |
| `/api/portfolio/add` | POST | Add position |
| `/api/portfolio/remove` | POST | Close position |
| `/api/portfolio/account` | POST | Update account size |
| `/api/risk-calc` | GET | Position sizing calculator |
| `/api/live` | GET | Current SPX price + day stats |
| `/api/exit` | GET | Exit analysis for open position |
| `/api/checklist` | GET | Scaled entry checklist score + tier |
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

- Daily/intraday OHLCV for SPX (`^GSPC`), NDX (`^NDX`), and individual tickers
- SPX options chains (`^SPX`) and NDX proxy via QQQ for Greeks, GEX, and net premium
- News headlines for sentiment analysis
- Crude oil prices (`BZ=F`, `CL=F`) for macro correlation

## Known Behaviors

- **Context-aware scoring**: RSI, Bollinger Bands, Stochastic, and Key Levels all behave differently in uptrend vs. downtrend vs. range. RSI 70 during an SPX uptrend is a continuation signal, not an overbought exit.
- **Net Premium AUTO values**: Approximation only — cannot distinguish bought vs. sold options from public data. Use Unusual Whales manual override for real institutional flow accuracy.
- **GEX Regime vs. Dealer Positioning**: GEX Regime (indicator 11) measures long/short gamma regime from the GEX chart. Dealer Positioning (leading indicator 3) uses put/call ratios and IV rank. They are complementary, not duplicate.
- **5-Day Trend accuracy**: Test set accuracy varies with market regime. The hybrid approach (ML + rules) is more stable across different conditions than pure ML alone.
- **macOS file permissions**: `data_fetcher.py` and `model.py` delete existing cache/model files before rewriting them. macOS `com.apple.provenance` cannot be stripped — deleting and recreating avoids in-place overwrite failures on the Refresh & Retrain path.
- **Account balance**: Never stored in source code. Enter your own balance in the Scaled Entry Checklist — it saves to `localStorage` only.

## Disclaimer

This tool is for **research and educational purposes only**. It is not financial advice. Past performance of signals and predictions does not guarantee future results. Always do your own analysis and manage risk appropriately.
