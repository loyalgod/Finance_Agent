"""买卖信号生成：多因子趋势增强策略。

信号逻辑：
    composite ≥ buy_threshold  → 目标仓位 1（建仓/持有）
    composite ≤ sell_threshold → 目标仓位 0（清仓）
    其余                        → 维持上一目标仓位（滞回，避免反复打脸）
预热期（指标未就绪）目标仓位恒为 0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_STRATEGY_PARAMS, StrategyParams
from .factors import build_factors

__all__ = ["SignalSet", "MultiFactorTrendStrategy"]


@dataclass
class SignalSet:
    """单标的信号集合。"""

    symbol: str
    composite: pd.Series      # 综合分 [-1, 1]
    target: pd.Series         # 目标仓位 {0, 1}
    factors: pd.DataFrame     # 全部因子明细


class MultiFactorTrendStrategy:
    """多因子趋势增强策略。"""

    name = "多因子趋势增强策略"

    def __init__(self, params: StrategyParams | None = None):
        self.params = params or DEFAULT_STRATEGY_PARAMS

    # ------------------------------------------------------------------ #
    def generate(self, symbol: str, df: pd.DataFrame) -> SignalSet:
        """对单标的 OHLCV 生成信号。"""
        factors = build_factors(df, self.params)
        composite = factors["composite"]

        target = pd.Series(np.nan, index=df.index, dtype="float64")
        target[composite >= self.params.buy_threshold] = 1.0
        target[composite <= self.params.sell_threshold] = 0.0
        # 滞回：信号之间的日期维持上一仓位；预热期（首个有效信号之前）为 0
        target = target.ffill().fillna(0.0)

        return SignalSet(symbol=symbol, composite=composite, target=target, factors=factors)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        p = self.params
        return (
            f"{self.name}：综合分 = {p.w_trend:.0%}×趋势 + {p.w_momentum:.0%}×动量 "
            f"+ {p.w_rsi:.0%}×RSI + {p.w_volume:.0%}×量能；"
            f"综合分 ≥ {p.buy_threshold} 开仓，≤ {p.sell_threshold} 清仓；"
            f"配套 {p.warmup} 日指标预热。"
        )
