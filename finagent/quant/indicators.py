"""技术指标库（Pandas/NumPy 向量化实现，口径对齐国内行情软件）。

- 均线 MA / EMA：简单/指数移动平均
- MACD：DIF、DEA、MACD柱 = 2×(DIF-DEA)（国内惯例）
- RSI：Wilder 平滑
- KDJ：RSV + K/D/J（EMA 近似平滑）
- BOLL：中轨 ± 2×标准差（总体标准差 ddof=0）
- ATR：Wilder 平滑的真实波幅
- OBV：能量潮
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "macd", "rsi", "kdj", "boll", "atr",
    "obv", "momentum", "volatility", "add_all_indicators",
]


def sma(s: pd.Series, n: int) -> pd.Series:
    """简单移动平均。"""
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """指数移动平均。"""
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD，返回 (dif, dea, hist)。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = 2.0 * (dif - dea)  # 国内行情软件惯例：柱 = 2×(DIF-DEA)
    return dif, dea, hist


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI（Wilder 平滑）。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 无下跌 → RSI = 100
    return out.where(avg_loss != 0.0, other=100.0)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3):
    """KDJ，返回 (k, d, j)。"""
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    denom = (high_n - low_n).replace(0.0, np.nan)
    rsv = (close - low_n) / denom * 100.0
    k = rsv.ewm(alpha=1.0 / m1, adjust=False).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def boll(close: pd.Series, n: int = 20, k: float = 2.0):
    """布林带，返回 (mid, upper, lower)。"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR（Wilder 平滑的真实波幅）。"""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV 能量潮。"""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def momentum(s: pd.Series, n: int) -> pd.Series:
    """N 日动量（收益率）。"""
    return s / s.shift(n) - 1.0


def volatility(s: pd.Series, n: int = 20) -> pd.Series:
    """年化波动率（N 日收益率标准差 × √252）。"""
    return s.pct_change().rolling(n).std() * np.sqrt(252)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """给 OHLCV DataFrame 追加全部技术指标列。"""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = df.copy()
    out["ma5"] = sma(close, 5)
    out["ma20"] = sma(close, 20)
    out["ma60"] = sma(close, 60)
    out["dif"], out["dea"], out["macd_hist"] = macd(close)
    out["rsi14"] = rsi(close, 14)
    out["k"], out["d"], out["j"] = kdj(high, low, close)
    out["boll_mid"], out["boll_up"], out["boll_dn"] = boll(close)
    out["atr14"] = atr(high, low, close, 14)
    out["obv"] = obv(close, volume)
    out["mom20"] = momentum(close, 20)
    out["mom60"] = momentum(close, 60)
    out["vol20"] = volatility(close, 20)
    out["ret_1d"] = close.pct_change()
    return out
