import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from data_pipeline import load_all_data

np.random.seed(42)
os.makedirs("output", exist_ok=True)

# Load data
ohlcv = load_all_data("ETHUSDT", days_back=30)
S0    = ohlcv["close"].iloc[-1]
print(f"ETH price: ${S0:,.2f}, Vol: {ohlcv['sigma'].mean()*100:.1f}% p.a.")

# Simple market making simulation with fixed wide spread
SPREAD_PCT     = 0.03    # 300 bps = 3% spread
FILL_PROB      = 0.15    # 15% chance of fill per side per step
TRADE_SIZE_ETH = 0.5
GAS_USD        = 6.5
MEV_BPS        = 5.0
SLIPPAGE_BPS   = 10.0
MAX_INV        = 3.0

inventory = 0.0
cash      = 0.0
n_trades  = 0
pnl_series = []

print(f"Running backtest: {len(ohlcv):,} steps, spread={SPREAD_PCT*10000:.0f} bps")

for i, (ts, row) in enumerate(ohlcv.iterrows()):
    S   = row["close"]
    bid = S * (1 - SPREAD_PCT / 2)
    ask = S * (1 + SPREAD_PCT / 2)

    bid_fill = (np.random.random() < FILL_PROB) and (inventory < MAX_INV)
    ask_fill = (np.random.random() < FILL_PROB) and (inventory > -MAX_INV)

    trade_usd = TRADE_SIZE_ETH * S
    cost_per_trade = (GAS_USD
                      + trade_usd * MEV_BPS / 10000
                      + trade_usd * SLIPPAGE_BPS / 10000)

    if bid_fill:
        inventory += TRADE_SIZE_ETH
        cash      -= bid * TRADE_SIZE_ETH + cost_per_trade
        n_trades  += 1

    if ask_fill:
        inventory -= TRADE_SIZE_ETH
        cash      += ask * TRADE_SIZE_ETH - cost_per_trade
        n_trades  += 1

    mtm = cash + inventory * S
    pnl_series.append({"timestamp": ts, "price": S,
                        "inventory": inventory, "pnl": mtm})

df = pd.DataFrame(pnl_series).set_index("timestamp")
pnl  = df["pnl"]
ret  = pnl.diff().dropna()

sharpe   = ret.mean() / (ret.std() + 1e-9) * np.sqrt(8760)
max_dd   = (pnl - pnl.cummax()).min()
fill_rate = n_trades / (len(ohlcv) * 2)

print(f"\n=== Results ===")
print(f"  Sharpe (annual):  {sharpe:.3f}")
print(f"  Max Drawdown:     ${max_dd:,.2f}")
print(f"  Final P&L:        ${pnl.iloc[-1]:,.2f}")
print(f"  N Trades:         {n_trades:,}")
print(f"  Fill Rate:        {fill_rate*100:.1f}%")
print(f"  Avg Spread:       {SPREAD_PCT*10000:.0f} bps")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DeFi Market Making Simulation - ETH/USDT", fontsize=13, fontweight="bold")

sample = df.iloc[::10]
axes[0,0].plot(sample.index, sample["price"], color="black", lw=0.8, label="Mid")
axes[0,0].plot(sample.index, sample["price"]*(1-SPREAD_PCT/2), color="green", lw=0.5, alpha=0.6, label="Bid")
axes[0,0].plot(sample.index, sample["price"]*(1+SPREAD_PCT/2), color="red",   lw=0.5, alpha=0.6, label="Ask")
axes[0,0].set_title("Market Making Quotes"); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.2)

axes[0,1].plot(df.index, df["inventory"], color="purple", lw=1)
axes[0,1].axhline(0, color="black", lw=0.8, ls="--")
axes[0,1].axhline( MAX_INV, color="red", lw=0.8, ls=":", label="Inventory limit")
axes[0,1].axhline(-MAX_INV, color="red", lw=0.8, ls=":")
axes[0,1].set_title("Inventory (ETH)"); axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.2)

axes[1,0].plot(df.index, df["pnl"], color="darkgreen", lw=1.2)
axes[1,0].axhline(0, color="black", lw=0.8, ls="--")
axes[1,0].fill_between(df.index, df["pnl"], 0, where=df["pnl"]>0, alpha=0.2, color="green")
axes[1,0].fill_between(df.index, df["pnl"], 0, where=df["pnl"]<0, alpha=0.2, color="red")
axes[1,0].set_title("Cumulative P&L (USD)"); axes[1,0].grid(alpha=0.2)

roll_sharpe = ret.rolling(720).mean() / (ret.rolling(720).std() + 1e-9) * np.sqrt(8760)
axes[1,1].plot(roll_sharpe.index, roll_sharpe, color="steelblue", lw=1)
axes[1,1].axhline(0, color="black", lw=0.8, ls="--")
axes[1,1].set_title("Rolling Sharpe (30-day)"); axes[1,1].grid(alpha=0.2)

plt.tight_layout()
fig.savefig("output/market_making_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n-> Saved: output/market_making_analysis.png")
print("Done!")
