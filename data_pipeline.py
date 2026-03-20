import requests
import websocket
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from web3 import Web3


# ─────────────────────────────────────────────
# 1. BINANCE REST API - Historical tick data
# ─────────────────────────────────────────────

def fetch_binance_trades(symbol="ETHUSDT", days_back=30):
    """
    Fetch historical aggregate trades from Binance REST API.
    Each row = one aggTrade (price, quantity, timestamp, buyer_maker).
    buyer_maker=True means the trade was a sell (maker was buyer).
    """
    url = "https://api.binance.com/api/v3/aggTrades"
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)

    all_trades = []
    current    = start_ms

    print(f"Fetching {symbol} trades for last {days_back} days...")

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": current,
            "endTime":   min(current + 3600000, end_ms),  # 1hr chunks
            "limit":     1000
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if not data:
            current += 3600000
            continue

        all_trades.extend(data)
        current = data[-1]["T"] + 1
        time.sleep(0.05)

    df = pd.DataFrame(all_trades)
    df = df.rename(columns={
        "T": "timestamp", "p": "price", "q": "quantity", "m": "buyer_maker"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"]     = df["price"].astype(float)
    df["quantity"]  = df["quantity"].astype(float)
    df = df.set_index("timestamp")[["price", "quantity", "buyer_maker"]]
    df = df.sort_index()

    print(f"Fetched {len(df):,} trades from {df.index[0]} to {df.index[-1]}")
    return df


def fetch_ohlcv(symbol="ETHUSDT", interval="1m", days_back=30):
    """Fetch OHLCV candles - used for volatility and spread calibration."""
    url = "https://api.binance.com/api/v3/klines"
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)

    all_data = []
    current  = start_ms

    print(f"Fetching {symbol} {interval} OHLCV for last {days_back} days...")

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     1000
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1][0] + 1
        time.sleep(0.05)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "buy_base", "buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    df = df.sort_index().drop_duplicates()

    print(f"Fetched {len(df):,} candles")
    return df


# ─────────────────────────────────────────────
# 2. UNISWAP V3 ON-CHAIN DATA via Infura RPC
# ─────────────────────────────────────────────

UNISWAP_V3_POOL_ABI = [
    {
        "inputs":  [],
        "name":    "slot0",
        "outputs": [
            {"name": "sqrtPriceX96",   "type": "uint160"},
            {"name": "tick",           "type": "int24"},
            {"name": "observationIndex","type": "uint16"},
            {"name": "observationCardinality","type": "uint16"},
            {"name": "observationCardinalityNext","type": "uint16"},
            {"name": "feeProtocol",    "type": "uint8"},
            {"name": "unlocked",       "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs":  [],
        "name":    "liquidity",
        "outputs": [{"name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ETH/USDC 0.3% pool on Ethereum mainnet
UNISWAP_ETH_USDC_POOL = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"


def get_uniswap_price(infura_url: str) -> dict:
    """
    Fetch current ETH price from Uniswap V3 on-chain via Infura RPC.

    sqrtPriceX96 is the sqrt of price scaled by 2^96.
    Price = (sqrtPriceX96 / 2^96)^2 * 10^(decimal_diff)
    For ETH/USDC: decimal_diff = 10^6 / 10^18 = 10^-12

    Args:
        infura_url: Your Infura endpoint, e.g.
                    "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
    """
    w3   = Web3(Web3.HTTPProvider(infura_url))
    pool = w3.eth.contract(
        address=Web3.to_checksum_address(UNISWAP_ETH_USDC_POOL),
        abi=UNISWAP_V3_POOL_ABI
    )
    slot0     = pool.functions.slot0().call()
    liquidity = pool.functions.liquidity().call()

    sqrt_price_x96 = slot0[0]
    price_raw      = (sqrt_price_x96 / (2**96)) ** 2
    # Adjust for decimals: USDC=6, ETH=18 → multiply by 1e12
    price_usdc = price_raw * 1e12

    return {
        "price":     price_usdc,
        "tick":      slot0[1],
        "liquidity": liquidity,
        "timestamp": datetime.utcnow(),
    }


def build_price_series_from_ohlcv(df_cex: pd.DataFrame,
                                   spread_bps: float = 30.0) -> pd.DataFrame:
    """
    Simulate DEX mid-price from CEX OHLCV with a synthetic spread.
    Used as a proxy when Infura is not available.

    spread_bps: typical CEX-DEX spread in basis points (30bps = 0.3%)
    """
    df = df_cex[["close"]].copy()
    df.columns = ["cex_price"]

    # Simulate DEX price with small lag and noise (realistic for AMM)
    lag_noise  = np.random.normal(0, spread_bps / 10000 * df["cex_price"].mean(),
                                   len(df))
    df["dex_price"] = df["cex_price"].shift(1).bfill() + lag_noise
    df["spread"]    = df["cex_price"] - df["dex_price"]
    df["spread_bps"] = df["spread"] / df["cex_price"] * 10000

    return df


# ─────────────────────────────────────────────
# 3. ORDER FLOW IMBALANCE (Toxic Flow Detection)
# ─────────────────────────────────────────────

def compute_order_flow_imbalance(trades_df: pd.DataFrame,
                                  window: str = "5min") -> pd.DataFrame:
    """
    Compute Order Flow Imbalance (OFI) - key metric for toxic flow detection.

    OFI = (buy_volume - sell_volume) / total_volume
    OFI ∈ [-1, 1]:
        +1 = all buys  → price pressure upward
        -1 = all sells → price pressure downward

    High |OFI| indicates informed trading (toxic flow for market makers).
    """
    df = trades_df.copy()
    df["buy_vol"]  = np.where(~df["buyer_maker"], df["quantity"], 0)
    df["sell_vol"] = np.where( df["buyer_maker"], df["quantity"], 0)

    ofi = df.resample(window).agg(
        buy_vol  = ("buy_vol",  "sum"),
        sell_vol = ("sell_vol", "sum"),
        price    = ("price",    "last"),
        n_trades = ("price",    "count"),
    )
    ofi["total_vol"] = ofi["buy_vol"] + ofi["sell_vol"]
    ofi["ofi"]       = (ofi["buy_vol"] - ofi["sell_vol"]) / (ofi["total_vol"] + 1e-9)

    # VWAP over the window
    df["pq"] = df["price"] * df["quantity"]
    vwap = df.resample(window).agg(pq=("pq", "sum"), q=("quantity", "sum"))
    ofi["vwap"] = vwap["pq"] / (vwap["q"] + 1e-9)

    return ofi.dropna()


# ─────────────────────────────────────────────
# 4. VOLATILITY ESTIMATION
# ─────────────────────────────────────────────

def estimate_realised_vol(ohlcv: pd.DataFrame,
                           window: int = 24,
                           freq_per_year: int = 8760) -> pd.Series:
    """
    Parkinson volatility estimator using High-Low range.
    More efficient than close-to-close for intraday data.

    σ_park = sqrt(1/(4*ln2) * E[(ln(H/L))^2])
    """
    log_hl = np.log(ohlcv["high"] / ohlcv["low"])
    park   = np.sqrt(1 / (4 * np.log(2)) * log_hl**2)
    return park.rolling(window).mean() * np.sqrt(freq_per_year)


# ─────────────────────────────────────────────
# 5. CONVENIENCE LOADER
# ─────────────────────────────────────────────

def load_all_data(symbol="ETHUSDT", days_back=30, use_cache=True):
    """
    Load OHLCV + compute OFI proxy + realised vol.
    Returns dict with all data needed for backtester.
    """
    os.makedirs("data", exist_ok=True)
    cache = f"data/{symbol}_{days_back}d_mm.parquet"

    if use_cache and os.path.exists(cache):
        print(f"Loading from cache: {cache}")
        ohlcv = pd.read_parquet(cache)
    else:
        ohlcv = fetch_ohlcv(symbol, "1m", days_back)
        ohlcv.to_parquet(cache)

    # Realised volatility
    ohlcv["sigma"] = estimate_realised_vol(ohlcv, window=60)

    # Synthetic CEX-DEX spread series
    price_series = build_price_series_from_ohlcv(ohlcv)
    ohlcv["dex_price"]   = price_series["dex_price"]
    ohlcv["spread_bps"]  = price_series["spread_bps"]

    ohlcv = ohlcv.dropna()
    print(f"Data ready: {len(ohlcv):,} rows, "
          f"avg σ = {ohlcv['sigma'].mean()*100:.1f}% p.a.")

    return ohlcv
