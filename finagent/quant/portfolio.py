"""仓位管理：ATR 风险预算法 + 仓位上限约束。

核心思路（ATR 风险预算法）：
    单笔风险预算 = 账户权益 × risk_per_trade
    止损距离   = ATR × atr_mult
    名义仓位   = 风险预算 ÷ (止损距离 ÷ 价格) = 风险预算 × 价格 ÷ 止损距离
    实际仓位   = min(名义仓位, 单票上限)
"""
from __future__ import annotations

from math import floor

from ..config import DEFAULT_RISK_PARAMS, RiskParams, DEFAULT_TRADE_PARAMS, TradeParams

__all__ = ["atr_risk_position_size", "affordable_shares", "min_trade_value"]


def atr_risk_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_params: RiskParams | None = None,
    trade_params: TradeParams | None = None,
) -> int:
    """按 ATR 风险预算计算买入股数（A 股整手 100 股取整）。

    equity: 当前账户权益；price: 参考价；atr: ATR(14) 值。
    """
    rp = risk_params or DEFAULT_RISK_PARAMS
    tp = trade_params or DEFAULT_TRADE_PARAMS
    if price <= 0 or atr <= 0:
        return 0
    risk_budget = equity * rp.risk_per_trade          # 单笔可承受亏损
    stop_dist = atr * rp.atr_mult                      # 每股止损距离
    nominal_value = risk_budget * price / stop_dist    # 名义仓位市值
    capped_value = min(nominal_value, equity * rp.max_position_pct, equity * rp.max_total_pct)
    if capped_value <= 0:
        return 0
    shares = floor(capped_value / price / tp.lot_size) * tp.lot_size
    return max(shares, 0)


def affordable_shares(
    cash: float,
    price: float,
    shares: int,
    trade_params: TradeParams | None = None,
) -> int:
    """按可用现金（含预估费用）向下调整买入股数至可成交。"""
    tp = trade_params or DEFAULT_TRADE_PARAMS
    if shares <= 0 or price <= 0:
        return 0
    while shares > 0:
        value = price * shares
        fee = max(value * tp.commission_rate, tp.commission_min)
        if value + fee <= cash:
            return shares
        shares -= tp.lot_size
    return 0


def min_trade_value(equity: float, threshold: float = 0.01) -> float:
    """最小交易金额（默认权益的 1%），过滤无效调仓。"""
    return equity * threshold
