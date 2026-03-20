import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MMParams:
    gamma:         float = 0.1
    kappa:         float = 1.5
    sigma:         float = 0.02
    T:             float = 1.0
    max_inventory: float = 3.0
    fee_tier:      float = 0.003
    min_spread:    float = 0.025


@dataclass
class MMState:
    inventory:   float = 0.0
    cash:        float = 0.0
    n_trades:    int   = 0
    n_bid_fills: int   = 0
    n_ask_fills: int   = 0
    total_fees:  float = 0.0


class AvellanedaStoikov:
    def __init__(self, params: MMParams):
        self.p = params

    def reservation_price(self, S, q, t):
        tau = max(self.p.T - t, 1e-6)
        return S - q * self.p.gamma * self.p.sigma**2 * tau

    def optimal_spread(self, t):
        tau = max(self.p.T - t, 1e-6)
        spread = (self.p.gamma * self.p.sigma**2 * tau
                  + (2 / self.p.gamma) * np.log(1 + self.p.gamma / self.p.kappa))
        return max(spread, self.p.min_spread)

    def get_quotes(self, S, q, t):
        r     = self.reservation_price(S, q, t)
        delta = self.optimal_spread(t)
        bid   = r - delta / 2
        ask   = r + delta / 2
        if q >= self.p.max_inventory:
            bid = 0.0
        if q <= -self.p.max_inventory:
            ask = 1e9
        return bid, ask

    def fill_probability(self, delta_half):
        return np.exp(-self.p.kappa * delta_half)

    def step(self, S, q, t, ofi=0.0):
        self.p.sigma = self.p.sigma * (1 + abs(ofi) * 0.5)
        bid, ask = self.get_quotes(S, q, t)
        delta_bid = (S - bid) / S if bid > 0 else 1.0
        delta_ask = (ask - S) / S if ask < 1e9 else 1.0
        bid_fill = np.random.random() < self.fill_probability(delta_bid)
        ask_fill = np.random.random() < self.fill_probability(delta_ask)
        if q >= self.p.max_inventory:  bid_fill = False
        if q <= -self.p.max_inventory: ask_fill = False
        inventory_delta = cash_delta = fee_paid = 0.0
        if bid_fill:
            inventory_delta += 1.0
            cash_delta      -= bid
            fee_paid        += bid * self.p.fee_tier
        if ask_fill:
            inventory_delta -= 1.0
            cash_delta      += ask
            fee_paid        += ask * self.p.fee_tier
        return {
            "bid": bid, "ask": ask, "spread": ask - bid,
            "reservation": self.reservation_price(S, q, t),
            "bid_fill": bid_fill, "ask_fill": ask_fill,
            "inventory_delta": inventory_delta,
            "cash_delta": cash_delta - fee_paid,
            "fee_paid": fee_paid, "ofi": ofi,
        }


def engle_granger_cointegration(x, y):
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(x, y)
    residuals  = y - (slope * x + intercept)
    resid_lag  = residuals.shift(1).dropna()
    resid_diff = residuals.diff().dropna()
    if len(resid_lag) < 10:
        return {"hedge_ratio": slope, "residuals": residuals,
                "adf_stat": 0, "is_cointegrated": False}
    slope_adf, _, _, _, _ = stats.linregress(resid_lag, resid_diff)
    t_stat = slope_adf / (residuals.std() / np.sqrt(len(residuals)))
    print(f"Cointegration test: ADF stat = {t_stat:.3f}, cointegrated = {t_stat < -2.86}")
    return {"hedge_ratio": slope, "intercept": intercept,
            "residuals": residuals, "adf_stat": t_stat,
            "is_cointegrated": t_stat < -2.86}


def compute_zscore(residuals, window=60):
    mean = residuals.rolling(window).mean()
    std  = residuals.rolling(window).std()
    return (residuals - mean) / (std + 1e-9)


def generate_signals(zscore, entry_z=2.0, exit_z=0.5):
    signal = pd.Series(0, index=zscore.index)
    position = 0
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z): continue
        if position == 0:
            if z > entry_z:   position = -1
            elif z < -entry_z: position = 1
        elif position == 1 and z > -exit_z:  position = 0
        elif position == -1 and z < exit_z:  position = 0
        signal.iloc[i] = position
    return signal
