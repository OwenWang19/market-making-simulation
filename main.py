import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from data_pipeline import load_all_data
from market_maker import MMParams, engle_granger_cointegration, compute_zscore, generate_signals
from backtest import ExecutionCosts, run_backtest, run_cointeg_backtest

np.random.seed(42)
os.makedirs("output", exist_ok=True)

# ── CONFIG ────────────────────────────────────
SYMBOL         = "ETHUSDT"
DAYS_BACK      = 30
TRADE_SIZE_ETH = 0.5

# ── STEP 1: LOAD DATA ─────────────────────────
print("=" * 55)
print("Step 1: Load Market Data")
print("=" * 55)

ohlcv = load_all_data(SYMBOL, days_back=DAYS_BACK)
S0    = ohlcv["close"].iloc[-1]
print(f"ETH price: ${S0:,.2f}")
print(f"Avg realised vol: {ohlcv['sigma'].mean()*100:.1f}% p.a.")
print(f"Avg CEX-DEX spread: {ohlcv['spread_bps'].mean():.1f} bps")

# ── STEP 2: EXECUTION COST ANALYSIS ──────────
print("\n" + "=" * 55)
print("Step 2: Execution Cost Model")
print("=" * 55)

exec_costs = ExecutionCosts(
    gas_price_gwei = 20,
    gas_units      = 150_000,
    eth_price      = S0,
    mev_bps        = 5.0,
    slippage_bps   = 10.0,
    pool_liquidity = 1e7,
)

trade_usd = TRADE_SIZE_ETH * S0
print(f"\nTrade size: {TRADE_SIZE_ETH} ETH (${trade_usd:,.0f})")
print(f"Gas cost:   ${exec_costs.gas_cost_usd():.2f}")
print(f"MEV cost:   ${exec_costs.mev_cost(trade_usd):.2f}")
print(f"Slippage:   ${exec_costs.slippage_cost(trade_usd):.2f}")
print(f"Total cost: ${exec_costs.total_cost(trade_usd):.2f}")
print(f"Breakeven spread: {exec_costs.breakeven_spread_bps(trade_usd):.1f} bps")

# ── STEP 3: COINTEGRATION TEST ────────────────
print("\n" + "=" * 55)
print("Step 3: CEX-DEX Cointegration Analysis")
print("=" * 55)

coint = engle_granger_cointegration(ohlcv["close"], ohlcv["dex_price"])
zscore = compute_zscore(coint["residuals"], window=60)
signals = generate_signals(zscore)

print(f"Hedge ratio:      {coint['hedge_ratio']:.4f}")
print(f"ADF statistic:    {coint['adf_stat']:.3f}")
print(f"Cointegrated:     {coint['is_cointegrated']}")
print(f"Arb signals:      {(signals != 0).sum()} / {len(signals)} bars")

# ── STEP 4: MARKET MAKING BACKTEST ───────────
print("\n" + "=" * 55)
print("Step 4: Avellaneda-Stoikov Market Making Backtest")
print("=" * 55)

mm_params = MMParams(
    gamma         = 0.1,
    kappa         = 1.5,
    sigma         = ohlcv["sigma"].mean() / np.sqrt(8760),
    max_inventory = 3.0,
    fee_tier      = 0.003,
)

bt = run_backtest(ohlcv, mm_params, exec_costs, TRADE_SIZE_ETH)
results = bt["results"]
metrics = bt["metrics"]

# ── STEP 5: COINTEGRATION BACKTEST ───────────
print("\n" + "=" * 55)
print("Step 5: Cointegration Strategy Backtest")
print("=" * 55)

coint_bt = run_cointeg_backtest(ohlcv, exec_costs)

# ── STEP 6: PLOTS ─────────────────────────────
print("\n" + "=" * 55)
print("Step 6: Generate Plots")
print("=" * 55)

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# Plot 1: Price + bid/ask quotes
ax1 = fig.add_subplot(gs[0, :])
sample = results.iloc[::10]  # downsample for clarity
ax1.plot(sample.index, sample["price"], color="black",  lw=0.8, label="Mid price")
ax1.plot(sample.index, sample["bid"],   color="green",  lw=0.5, alpha=0.6, label="Bid quote")
ax1.plot(sample.index, sample["ask"],   color="red",    lw=0.5, alpha=0.6, label="Ask quote")
ax1.fill_between(sample.index, sample["bid"], sample["ask"],
                 alpha=0.1, color="blue", label="Quoted spread")
ax1.set_title("Avellaneda-Stoikov Market Making Quotes", fontsize=12)
ax1.set_ylabel("ETH/USDT ($)")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(alpha=0.2)

# Plot 2: Inventory over time
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(results.index, results["inventory"], color="purple", lw=1)
ax2.axhline(0,  color="black", lw=0.8, ls="--")
ax2.axhline( mm_params.max_inventory, color="red", lw=0.8, ls=":", label="Max inventory")
ax2.axhline(-mm_params.max_inventory, color="red", lw=0.8, ls=":")
ax2.fill_between(results.index, results["inventory"], 0,
                 where=results["inventory"] > 0, alpha=0.2, color="green")
ax2.fill_between(results.index, results["inventory"], 0,
                 where=results["inventory"] < 0, alpha=0.2, color="red")
ax2.set_title("Inventory (ETH)")
ax2.set_ylabel("ETH")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

# Plot 3: Cumulative P&L
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(results.index, results["total_pnl"], color="darkgreen", lw=1.2, label="Total P&L")
ax3.plot(results.index, results["cash_pnl"],  color="steelblue", lw=0.8,
         alpha=0.7, label="Cash P&L")
ax3.axhline(0, color="black", lw=0.8, ls="--")
ax3.fill_between(results.index, results["total_pnl"], 0,
                 where=results["total_pnl"] > 0, alpha=0.15, color="green")
ax3.fill_between(results.index, results["total_pnl"], 0,
                 where=results["total_pnl"] < 0, alpha=0.15, color="red")
ax3.set_title("Cumulative P&L (USD)")
ax3.set_ylabel("USD")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.2)

# Plot 4: Z-score and signals
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(zscore.index, zscore, color="steelblue", lw=0.8, label="Z-score")
ax4.axhline( 2.0, color="red",   ls="--", lw=1, label="Entry ±2σ")
ax4.axhline(-2.0, color="red",   ls="--", lw=1)
ax4.axhline( 0.5, color="green", ls=":",  lw=1, label="Exit ±0.5σ")
ax4.axhline(-0.5, color="green", ls=":",  lw=1)
long_mask  = signals == 1
short_mask = signals == -1
ax4.fill_between(signals.index, -3, 3,
                 where=long_mask,  alpha=0.1, color="green")
ax4.fill_between(signals.index, -3, 3,
                 where=short_mask, alpha=0.1, color="red")
ax4.set_ylim(-4, 4)
ax4.set_title("CEX-DEX Spread Z-score & Signals")
ax4.set_ylabel("Z-score")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.2)

# Plot 5: Cointegration strategy cumulative P&L
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(coint_bt["cum_pnl"].index,
         coint_bt["cum_pnl"] * 100, color="darkorange", lw=1.2)
ax5.axhline(0, color="black", lw=0.8, ls="--")
ax5.fill_between(coint_bt["cum_pnl"].index,
                 coint_bt["cum_pnl"] * 100, 0,
                 where=coint_bt["cum_pnl"] > 0, alpha=0.2, color="green")
ax5.fill_between(coint_bt["cum_pnl"].index,
                 coint_bt["cum_pnl"] * 100, 0,
                 where=coint_bt["cum_pnl"] < 0, alpha=0.2, color="red")
ax5.set_title("Cointegration Strategy Cumulative P&L")
ax5.set_ylabel("Return (%)")
ax5.grid(alpha=0.2)

plt.suptitle("DeFi Market Making & CEX-DEX Arbitrage Analysis — ETH/USDT",
             fontsize=13, fontweight="bold")

fig.savefig("output/market_making_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ── SUMMARY ───────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Market Making:")
print(f"  Sharpe (annual):    {metrics['sharpe_annual']:.3f}")
print(f"  Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
print(f"  Avg Spread:         {metrics['avg_spread_bps']:.1f} bps")
print(f"  Fill Rate:          {metrics['fill_rate']*100:.1f}%")
print(f"  Net P&L after costs:${metrics['net_pnl_after_costs']:.2f}")
print(f"\nCointegration Strategy:")
print(f"  Sharpe (annual):    {coint_bt['sharpe']:.3f}")
print(f"  N Trades:           {coint_bt['n_trades']}")
print(f"  Final PnL:          {coint_bt['cum_pnl'].iloc[-1]*100:.2f}%")
print(f"\n-> Saved: output/market_making_analysis.png")
print("Done!")
