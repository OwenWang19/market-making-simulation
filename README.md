# DeFi Market Making Strategy Simulator

Quantitative simulation of an inventory-based market making strategy for ETH/USDT on Uniswap V3, implementing the Avellaneda-Stoikov framework with realistic DeFi execution cost modelling.

## Overview

Market makers provide liquidity by continuously quoting bid and ask prices around a reservation price. This project simulates optimal quote placement under:
- **Inventory risk**: long/short ETH exposure from asymmetric fills
- **Adverse selection**: toxic flow from informed traders (OFI-based detection)
- **Execution costs**: L1 Gas, MEV, and AMM slippage

## Methodology

### 1. Avellaneda-Stoikov Model
Reservation price adjusts dynamically based on inventory:
```
r(t,q) = S - q * γ * σ² * (T - t)
```
Optimal spread widens with volatility and risk aversion:
```
δ = γ * σ² * (T-t) + (2/γ) * ln(1 + γ/κ)
```

### 2. Execution Cost Model
| Cost Component | Value |
|---------------|-------|
| L1 Gas        | $6.50 / tx |
| MEV           | 5 bps of trade size |
| Pool Slippage | 10 bps base |
| **Breakeven Spread** | **168.7 bps** |

### 3. Data Pipeline
- 43,000+ hourly ETH/USDT candles from Binance REST API
- Parkinson volatility estimator (σ = 6.1% annualised)
- Synthetic CEX-DEX spread series for cointegration analysis

## Results

| Metric | Value |
|--------|-------|
| Sharpe Ratio (annual) | 36.65 |
| Net P&L | $86,611 |
| Max Drawdown | $137 |
| Fill Rate | 14.1% |
| Avg Spread | 300 bps |
| N Trades | 12,139 |

## Project Structure

```
market_making/
├── run.py             # Main backtest entry point
├── market_maker.py    # Avellaneda-Stoikov model + cointegration signals
├── backtest.py        # Backtesting engine + execution cost model
├── data_pipeline.py   # Binance API + volatility estimation
├── data/              # Cached parquet files (auto-created)
└── output/            # Plots (auto-created)
    └── market_making_analysis.png
```

## Setup & Usage

```bash
pip install numpy pandas scipy matplotlib requests websocket-client web3
python run.py
```

## Key Concepts

- Avellaneda-Stoikov optimal market making
- Inventory risk management
- Order Flow Imbalance (OFI) for toxic flow detection
- DeFi execution costs (Gas, MEV, slippage)
- Engle-Granger cointegration for CEX-DEX spread analysis
