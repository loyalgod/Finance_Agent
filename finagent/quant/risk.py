"""风险控制：固定止损、ATR 移动止损（吊灯止损）、固定止盈、账户回撤熔断。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DEFAULT_RISK_PARAMS, RiskParams

__all__ = ["RiskController"]


@dataclass
class ExitSignal:
    """持仓退出信号。"""

    triggered: bool
    reason: str = ""


class RiskController:
    """日内风控检查器（按收盘价评估，T+1 开盘执行）。"""

    def __init__(self, params: RiskParams | None = None):
        self.params = params or DEFAULT_RISK_PARAMS

    # ------------------------------------------------------------------ #
    def check_exit(self, position: dict, row: pd.Series) -> ExitSignal:
        """检查持仓是否触发退出条件。

        position: {"shares", "entry_price", "highest", "atr"}
        优先级：固定止损 > 移动止损 > 固定止盈。
        """
        p = self.params
        close = float(row["close"])
        atr = float(row.get("atr14", 0.0) or 0.0)
        entry = float(position["entry_price"])

        # 1) 固定止损：跌破入场价 (1 - fixed_stop_pct)
        if close <= entry * (1.0 - p.fixed_stop_pct):
            return ExitSignal(True, f"固定止损（跌破入场价 {p.fixed_stop_pct:.0%}）")

        # 2) ATR 移动止损（吊灯止损）：跌破持仓最高价 - ATR × 倍数
        if atr > 0:
            trail_stop = position.get("highest", close) - atr * p.atr_mult
            if close <= trail_stop:
                return ExitSignal(True, f"移动止损（吊灯止损 {p.atr_mult}×ATR）")

        # 3) 固定止盈：涨过入场价 (1 + take_profit_pct)
        if close >= entry * (1.0 + p.take_profit_pct):
            return ExitSignal(True, f"止盈（涨幅达 {p.take_profit_pct:.0%}）")

        return ExitSignal(False)

    # ------------------------------------------------------------------ #
    def check_drawdown(self, equity: float, peak: float) -> bool:
        """账户回撤是否触发熔断（清仓线）。"""
        if peak <= 0:
            return False
        return (equity / peak - 1.0) <= -self.params.dd_breaker

    def check_resume(self, equity: float, peak: float) -> bool:
        """回撤恢复至 dd_resume 以内，解除熔断。"""
        if peak <= 0:
            return True
        return (equity / peak - 1.0) >= -self.params.dd_resume
