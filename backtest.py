import numpy as np
import pandas as pd
from dataclasses import dataclass
from market_maker import AvellanedaStoikov, MMParams, MMState


# ─────────────────────────────────────────────
# EXECUTION COST MODEL
# ─────────────────────────────────────────────

@dataclass
class ExecutionCosts:
    """
    Models all execution costs for DeFi market making.

    L1 Gas:   Each on-chain tx costs gas. ETH gas price is volatile.
    MEV:      Maximal Extractable Value - sandwich bots front-run large orders.
              Estimated as a % of trade size.
    Slippage: Price impact of trading against AMM liquidity.
              For Uniswap V3: slippage ∝ trade_size / pool_liquidity
    """
    gas_price_gwei:   float = 20.0    # ETH gas price in Gwei
    gas_units:        int   = 150_000  # Gas units per swap transaction
    eth_price:        float = 2000.0   # ETH price for gas cost conversion
    mev_bps:          float = 5.0      # MEV cost in basis points of trade size
    slippage_bps:     float = 10.0     # Base slippage in bps
    pool_liquidity:   float = 1e7      # Pool liquidity in USD (affects slippage)

    def gas_cost_usd(self) -> float:
        """Gas cost in USD per transaction."""
        gas_eth = self.gas_price_gwei * self.gas_units * 1e-9
        return gas_eth * self.eth_price

    def mev_cost(self, trade_size_usd: float) -> float:
        """MEV cost scales with trade size."""
        return trade_size_usd * self.mev_bps / 10000

    def slippage_cost(self, trade_size_usd: float) -> float:
        """
        Price impact model: slippage ∝ sqrt(trade_size / liquidity)
        Square root model is standard for AMMs.
        """
        impact = np.sqrt(trade_size_usd / self.pool_liquidity)
        return trade_size_usd * max(impact, self.slippage_bps / 10000)

    def total_cost(self, trade_size_usd: float) -> float:
        """Total execution cost for one trade."""
        return (self.gas_cost_usd()
                + self.mev_cost(trade_size_usd)
                + self.slippage_cost(trade_size_usd))

    def breakeven_spread_bps(self, trade_size_usd: float) -> float:
        """Minimum spread needed to cover all execution costs."""
        return self.total_cost(trade_size_usd) / trade_size_usd * 10000


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(ohlcv: pd.DataFrame,
                 mm_params: MMParams = None,
                 exec_costs: ExecutionCosts = None,
                 trade_size_eth: float = 0.5) -> dict:
    """
    Full market making backtest over historical OHLCV data.

    At each timestep:
    1. AS model quotes bid/ask
    2. Stochastic fill simulation
    3. Deduct execution costs (gas + MEV + slippage)
    4. Track inventory, P&L, Sharpe

    Args:
        ohlcv:           DataFrame with close, sigma, dex_price, spread_bps
        mm_params:       AS model parameters
        exec_costs:      Execution cost model
        trade_size_eth:  ETH per trade

    Returns:
        dict with full results and metrics
    """
    if mm_params  is None: mm_params  = MMParams()
    if exec_costs is None: exec_costs = ExecutionCosts()

    n     = len(ohlcv)
    state = MMState()

    # Result arrays
    timestamps      = []
    prices          = []
    bids            = []
    asks            = []
    spreads         = []
    inventories     = []
    cash_pnl        = []
    total_pnl       = []
    exec_costs_arr  = []
    ofis            = []

    mm = AvellanedaStoikov(mm_params)

    # Update ETH price for gas cost calculation
    exec_costs.eth_price = ohlcv["close"].mean()

    print(f"Running backtest over {n:,} timesteps...")
    print(f"Gas cost per tx: ${exec_costs.gas_cost_usd():.2f}")
    print(f"Breakeven spread at ${trade_size_eth * exec_costs.eth_price:.0f} trade: "
          f"{exec_costs.breakeven_spread_bps(trade_size_eth * exec_costs.eth_price):.1f} bps")

    for i, (ts, row) in enumerate(ohlcv.iterrows()):
        t   = i / n
        S   = row["close"]
        sig = row["sigma"] / np.sqrt(8760)  # annualised → per-hour

        # OFI proxy: use price change direction as simplified OFI
        if i > 0:
            prev_close = ohlcv["close"].iloc[i - 1]
            ofi = np.clip((S - prev_close) / (prev_close * sig + 1e-9), -1, 1)
        else:
            ofi = 0.0

        mm.p.sigma = sig

        result = mm.step(S, state.inventory, t, ofi)

        # Apply execution costs if a fill occurred
        cost = 0.0
        if result["bid_fill"] or result["ask_fill"]:
            trade_usd = trade_size_eth * S
            breakeven = exec_costs.breakeven_spread_bps(trade_usd) / 10000 * S
            if result["spread"] < breakeven:
                result["bid_fill"] = False
                result["ask_fill"] = False
            else:
                cost = exec_costs.total_cost(trade_usd)
                state.n_trades    += 1
                state.total_fees  += result["fee_paid"] + cost
                if result["bid_fill"]: state.n_bid_fills += 1
                if result["ask_fill"]: state.n_ask_fills += 1
        state.inventory += result["inventory_delta"] * trade_size_eth
        state.cash      += result["cash_delta"] * trade_size_eth - cost

        # Mark-to-market total P&L
        mtm_pnl = state.cash + state.inventory * S

        timestamps.append(ts)
        prices.append(S)
        bids.append(result["bid"])
        asks.append(result["ask"])
        spreads.append(result["spread"])
        inventories.append(state.inventory)
        cash_pnl.append(state.cash)
        total_pnl.append(mtm_pnl)
        exec_costs_arr.append(cost)
        ofis.append(ofi)

    results_df = pd.DataFrame({
        "timestamp":   timestamps,
        "price":       prices,
        "bid":         bids,
        "ask":         asks,
        "spread":      spreads,
        "inventory":   inventories,
        "cash_pnl":    cash_pnl,
        "total_pnl":   total_pnl,
        "exec_cost":   exec_costs_arr,
        "ofi":         ofis,
    }).set_index("timestamp")

    metrics = compute_metrics(results_df, exec_costs)
    return {"results": results_df, "metrics": metrics, "state": state}


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame,
                    exec_costs: ExecutionCosts) -> dict:
    """
    Compute comprehensive performance metrics for the MM strategy.
    """
    pnl    = df["total_pnl"]
    ret    = pnl.diff().dropna()

    # Sharpe ratio (annualised, hourly data)
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(8760)

    # Max drawdown
    rolling_max = pnl.cummax()
    drawdown    = (pnl - rolling_max) / (rolling_max.abs() + 1e-9)
    max_dd      = drawdown.min()

    # Calmar ratio
    calmar = (pnl.iloc[-1] / len(pnl)) * 8760 / (abs(max_dd) + 1e-9)

    # Inventory stats
    inv          = df["inventory"]
    avg_inv      = inv.abs().mean()
    max_inv      = inv.abs().max()
    inv_turnover = df["exec_cost"].sum() / (avg_inv + 1e-9)

    # Total execution costs
    total_gas_mev = df["exec_cost"].sum()

    # Fill rate
    n_steps    = len(df)
    n_nonzero  = (df["exec_cost"] > 0).sum()
    fill_rate  = n_nonzero / n_steps

    metrics = {
        "final_pnl_usd":    pnl.iloc[-1],
        "sharpe_annual":    sharpe,
        "max_drawdown":     max_dd,
        "calmar_ratio":     calmar,
        "avg_spread_bps":   (df["spread"] / df["price"] * 10000).mean(),
        "fill_rate":        fill_rate,
        "avg_inventory":    avg_inv,
        "max_inventory":    max_inv,
        "total_exec_costs": total_gas_mev,
        "net_pnl_after_costs": pnl.iloc[-1] - total_gas_mev,
        "pnl_positive":     pnl.iloc[-1] > 0,
    }

    print("\n=== Backtest Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    return metrics


# ─────────────────────────────────────────────
# COINTEGRATION BACKTEST
# ─────────────────────────────────────────────

def run_cointeg_backtest(ohlcv: pd.DataFrame,
                          exec_costs: ExecutionCosts = None) -> dict:
    """
    Backtest the CEX-DEX cointegration strategy.

    Logic:
    1. Compute Z-score of CEX vs DEX spread
    2. Enter when |Z| > 2, exit when |Z| < 0.5
    3. Deduct gas + slippage on each trade
    """
    from market_maker import (engle_granger_cointegration,
                               compute_zscore, generate_signals)
    if exec_costs is None:
        exec_costs = ExecutionCosts()

    cex = ohlcv["close"]
    dex = ohlcv["dex_price"]

    # Cointegration test
    coint = engle_granger_cointegration(cex, dex)
    print(f"\nCointegration ADF stat: {coint['adf_stat']:.3f}")

    # Z-score and signals
    zscore  = compute_zscore(coint["residuals"], window=60)
    signals = generate_signals(zscore, entry_z=2.0, exit_z=0.5)

    # P&L calculation
    returns   = cex.pct_change().fillna(0)
    strat_ret = signals.shift(1).fillna(0) * returns

    # Deduct execution costs on signal changes
    signal_changes = signals.diff().fillna(0) != 0
    trade_costs    = signal_changes * exec_costs.total_cost(
        ohlcv["close"].mean() * 0.5
    ) / ohlcv["close"].mean()

    net_returns = strat_ret - trade_costs
    cum_pnl     = (1 + net_returns).cumprod() - 1

    sharpe = net_returns.mean() / (net_returns.std() + 1e-9) * np.sqrt(8760)
    n_trades = signal_changes.sum()

    print(f"\n=== Cointegration Strategy Results ===")
    print(f"  N trades:        {n_trades}")
    print(f"  Sharpe (annual): {sharpe:.3f}")
    print(f"  Final PnL:       {cum_pnl.iloc[-1]*100:.2f}%")
    print(f"  Max Drawdown:    {(cum_pnl - cum_pnl.cummax()).min()*100:.2f}%")

    return {
        "zscore":    zscore,
        "signals":   signals,
        "cum_pnl":   cum_pnl,
        "net_ret":   net_returns,
        "sharpe":    sharpe,
        "n_trades":  n_trades,
    }
