"""特征与因子构建。

因子体系（多因子趋势增强策略，权重见 config.StrategyParams）：
- trend   趋势因子：均线多头排列程度，[-0.5, 0.5]
- momentum 动量因子：20 日动量的滚动 z-score，[-1, 1]
- rsi     RSI 因子：强势加分，超买(>80)扣分、超卖(<20)加分
- volume  量能因子：量比（当日量/20日均量），[-1, 1]
- composite 综合分：加权求和，[-1, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_STRATEGY_PARAMS, StrategyParams
from . import indicators as ind

__all__ = ["build_features", "build_factors", "FACTOR_COLUMNS"]

FACTOR_COLUMNS = ["trend", "momentum", "rsi", "volume"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """基础特征：收益率、动量、波动率、量比、布林位置、区间回撤。"""
    close, volume = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = close.pct_change()
    f["ret_5d"] = ind.momentum(close, 5)
    f["ret_20d"] = ind.momentum(close, 20)
    f["ret_60d"] = ind.momentum(close, 60)
    f["vol20"] = ind.volatility(close, 20)
    f["vol_ratio"] = volume / volume.rolling(20).mean()
    _, boll_up, boll_dn = ind.boll(close)
    width = (boll_up - boll_dn).replace(0.0, np.nan)
    f["boll_pos"] = (close - boll_dn) / width
    f["high_60d"] = close.rolling(60).max()
    f["low_60d"] = close.rolling(60).min()
    f["drawdown_60d"] = close / close.rolling(60).max() - 1.0
    return f


def _rolling_z(s: pd.Series, window: int = 250, clip: float = 2.0) -> pd.Series:
    """滚动 z-score（相对自身近 window 日分布），截断到 [-1, 1]。"""
    min_p = max(window // 2, 30)
    mean = s.rolling(window, min_periods=min_p).mean()
    std = s.rolling(window, min_periods=min_p).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return (z.clip(-clip, clip) / clip).fillna(0.0)


def build_factors(df: pd.DataFrame, params: StrategyParams | None = None) -> pd.DataFrame:
    """构建四大因子与综合分。返回含 trend/momentum/rsi/volume/composite 的 DataFrame。"""
    p = params or DEFAULT_STRATEGY_PARAMS
    close, volume = df["close"], df["volume"]

    ma5, ma20, ma60 = ind.sma(close, 5), ind.sma(close, 20), ind.sma(close, 60)
    rsi14 = ind.rsi(close, 14)
    vol_ratio = volume / volume.rolling(20).mean()

    # 趋势因子：5 项均线关系（价格在均线上方 + 均线多头排列）
    trend = (
        (close > ma5).astype(float)
        + (close > ma20).astype(float)
        + (close > ma60).astype(float)
        + (ma5 > ma20).astype(float)
        + (ma20 > ma60).astype(float)
    ) / 5.0 - 0.5

    # 动量因子：20 日动量滚动 z 分数
    momentum = _rolling_z(ind.momentum(close, 20))

    # RSI 因子：强势加分；超买扣分、超卖加分（风险感知的趋势因子）
    rsi_factor = ((rsi14 - 50.0) / 50.0).clip(-1.0, 1.0)
    rsi_factor = rsi_factor.where(~(rsi14 > 80.0), rsi_factor - 0.4)
    rsi_factor = rsi_factor.where(~(rsi14 < 20.0), rsi_factor + 0.4)

    # 量能因子：量比，放量代表资金关注度提升
    volume_factor = (vol_ratio - 1.0).clip(-1.0, 1.0)

    composite = (
        p.w_trend * trend
        + p.w_momentum * momentum
        + p.w_rsi * rsi_factor
        + p.w_volume * volume_factor
    )

    return pd.DataFrame(
        {
            "trend": trend,
            "momentum": momentum,
            "rsi": rsi_factor,
            "volume": volume_factor,
            "composite": composite,
        }
    )
