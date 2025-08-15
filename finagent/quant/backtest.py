"""事件驱动回测引擎。

交易规则（贴近 A 股真实约束）：
- T+1：收盘生成信号，次日开盘成交
- 费用：佣金万2.5（最低5元）、印花税千0.5（仅卖出）、滑点 0.05%（双边）
- 整手 100 股
- 风控：固定止损 8% / ATR 吊灯止损 2.5× / 止盈 25% / 账户回撤 15% 熔断
- 仓位：ATR 风险预算（单笔 2%）+ 单票上限 30% + 总仓位上限 95%

无未来函数：第 t 日信号只使用 ≤ t 日的数据，在 t+1 开盘执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_RISK_PARAMS,
    DEFAULT_STRATEGY_PARAMS,
    DEFAULT_TRADE_PARAMS,
    INITIAL_CASH,
    TRADING_DAYS_PER_YEAR,
    RiskParams,
    StrategyParams,
    TradeParams,
)
from .data import MarketData
from .portfolio import affordable_shares, atr_risk_position_size, min_trade_value
from .risk import RiskController
from .signals import MultiFactorTrendStrategy, SignalSet

__all__ = ["Trade", "BacktestResult", "Backtester", "compute_metrics"]


@dataclass
class Trade:
    """单笔成交记录。"""

    date: pd.Timestamp
    symbol: str
    side: str          # BUY / SELL
    price: float       # 成交价（含滑点）
    shares: int
    value: float       # 成交金额（不含费用）
    fee: float
    reason: str        # 建仓/加仓/减仓/清仓/止损/止盈/熔断
    pnl: float = 0.0   # 回合盈亏（仅清仓时填充）
    pnl_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "symbol": self.symbol,
            "side": self.side,
            "price": round(self.price, 3),
            "shares": self.shares,
            "value": round(self.value, 2),
            "fee": round(self.fee, 2),
            "reason": self.reason,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
        }


@dataclass
class BacktestResult:
    """回测结果。"""

    equity_curve: pd.Series          # 账户权益曲线
    benchmark_curve: pd.Series       # 基准（等权买入持有）曲线
    drawdown_curve: pd.Series        # 回撤曲线
    trades: list[Trade]
    metrics: dict
    signals: dict[str, SignalSet]
    strategy_desc: str
    params: dict
    date_range: tuple[str, str]

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=list(Trade.__dataclass_fields__))
        return pd.DataFrame([t.to_dict() for t in self.trades])


@dataclass
class _Position:
    shares: int = 0
    entry_price: float = 0.0     # 加权成本价（不含费用）
    invested: float = 0.0        # 累计投入（含费用），用于回合盈亏
    highest: float = 0.0         # 持仓期间最高收盘价
    atr: float = 0.0             # 建仓时 ATR


class Backtester:
    """多标的组合回测器（共享资金池）。"""

    def __init__(
        self,
        market: MarketData,
        strategy: MultiFactorTrendStrategy | None = None,
        trade_params: TradeParams | None = None,
        risk_params: RiskParams | None = None,
        initial_cash: float = INITIAL_CASH,
    ):
        self.market = market
        self.strategy = strategy or MultiFactorTrendStrategy()
        self.trade_params = trade_params or DEFAULT_TRADE_PARAMS
        self.risk_params = risk_params or DEFAULT_RISK_PARAMS
        self.risk = RiskController(self.risk_params)
        self.initial_cash = float(initial_cash)

    # ------------------------------------------------------------------ #
    def run(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        """运行回测。start/end 为 'YYYY-MM-DD' 字符串或 None。"""
        symbols = symbols or self.market.symbols()
        symbols = [s for s in symbols if s in self.market]
        if not symbols:
            raise ValueError("样本池为空，无法回测")

        # 预计算信号（向量化，避免循环内重复计算）
        frames, sigs = {}, {}
        for sym in symbols:
            df = self.market.get(sym)
            if start:
                df = df.loc[df.index >= pd.Timestamp(start)]
            if end:
                df = df.loc[df.index <= pd.Timestamp(end)]
            frames[sym] = df
            # 信号需在全量历史上计算（预热），再截取到回测区间
            full = self.market.get(sym)
            if start:
                full = full.loc[full.index >= pd.Timestamp(start) - pd.Timedelta(days=250)]
            sigs[sym] = self.strategy.generate(sym, full)

        dates = self.market.common_dates(symbols)
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        if len(dates) < self.strategy.params.warmup + 10:
            raise ValueError(f"回测区间交易日不足（{len(dates)} 天）")

        cash = self.initial_cash
        positions: dict[str, _Position] = {}
        trades: list[Trade] = []
        pending_orders: list[dict] = []   # 次日开盘执行
        equity_records: list[tuple[pd.Timestamp, float]] = []
        peak = self.initial_cash
        dd_breaker_active = False
        cooldown_left = 0

        for i, date in enumerate(dates):
            # —— 1) 开盘执行昨日订单 ——
            for order in pending_orders:
                cash = self._execute(order, date, frames, positions, trades, cash)
            pending_orders = []

            # —— 2) 盘中/收盘风控评估（用当日收盘价触发，次日开盘执行）——
            equity = self._mark_equity(cash, positions, frames, date)
            peak = max(peak, equity)

            # 回撤熔断：清仓 + 冷静期，期满后重置水位线（否则全现金账户永远无法恢复）
            if not dd_breaker_active and self.risk.check_drawdown(equity, peak):
                dd_breaker_active = True
                cooldown_left = self.risk_params.cooldown_days
                pending_orders = [
                    {"symbol": sym, "shares": pos.shares, "side": "SELL", "reason": "回撤熔断清仓"}
                    for sym, pos in positions.items() if pos.shares > 0
                ]
            elif dd_breaker_active:
                cooldown_left -= 1
                if cooldown_left <= 0:
                    dd_breaker_active = False
                    peak = equity  # 重置高水位，避免全现金状态下无法恢复的死锁

            if not dd_breaker_active:
                for sym, pos in positions.items():
                    if pos.shares <= 0:
                        continue
                    row = frames[sym].loc[date]
                    exit_sig = self.risk.check_exit(
                        {"shares": pos.shares, "entry_price": pos.entry_price,
                         "highest": pos.highest, "atr": pos.atr},
                        row,
                    )
                    if exit_sig.triggered:
                        pending_orders.append(
                            {"symbol": sym, "shares": pos.shares,
                             "side": "SELL", "reason": exit_sig.reason}
                        )

            # —— 3) 标记当日权益 ——
            equity = self._mark_equity(cash, positions, frames, date)
            equity_records.append((date, equity))

            # —— 4) 收盘调仓信号（T+1：次日开盘执行）——
            if i < len(dates) - 1 and not dd_breaker_active:
                for sym in symbols:
                    if sym not in sigs:
                        continue
                    target = float(sigs[sym].target.loc[date]) if date in sigs[sym].target.index else 0.0
                    pos = positions.setdefault(sym, _Position())
                    cur_shares = pos.shares
                    row = frames[sym].loc[date]
                    close = float(row["close"])

                    # 风控平仓订单优先，跳过目标调仓
                    if any(o["symbol"] == sym and o["side"] == "SELL" for o in pending_orders):
                        continue

                    if target >= 0.5 and cur_shares == 0:
                        atr_val = float(row.get("atr14", 0.0)) if "atr14" in row else float(row.get("atr", 0.0))
                        if atr_val <= 0:
                            atr_val = close * 0.02  # ATR 缺失时用 2% 波动兜底
                        shares = atr_risk_position_size(equity, close, atr_val, self.risk_params, self.trade_params)
                        if shares > 0 and close * shares >= min_trade_value(equity):
                            pending_orders.append(
                                {"symbol": sym, "shares": shares, "side": "BUY",
                                 "reason": "信号建仓", "atr": atr_val}
                            )
                    elif target < 0.5 and cur_shares > 0:
                        pending_orders.append(
                            {"symbol": sym, "shares": cur_shares,
                             "side": "SELL", "reason": "信号清仓"}
                        )

        # —— 收尾：最后一日收盘强制平仓（便于统计回合盈亏）——
        last_date = dates[-1]
        for sym, pos in positions.items():
            if pos.shares > 0:
                cash = self._execute(
                    {"symbol": sym, "shares": pos.shares, "side": "SELL", "reason": "期末平仓"},
                    last_date, frames, positions, trades, cash,
                )
        equity = self._mark_equity(cash, positions, frames, last_date)
        if not equity_records or equity_records[-1][0] != last_date:
            equity_records.append((last_date, equity))
        else:
            equity_records[-1] = (last_date, equity)

        # —— 汇总 ——
        equity_curve = pd.Series(
            [e for _, e in equity_records],
            index=pd.DatetimeIndex([d for d, _ in equity_records]),
            name="equity",
        )
        peak_curve = equity_curve.cummax()
        drawdown_curve = equity_curve / peak_curve - 1.0

        benchmark_curve = self._build_benchmark(symbols, dates)
        metrics = compute_metrics(equity_curve, benchmark_curve, trades, self.initial_cash)

        return BacktestResult(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            metrics=metrics,
            signals=sigs,
            strategy_desc=self.strategy.describe(),
            params={
                "strategy": {
                    "buy_threshold": self.strategy.params.buy_threshold,
                    "sell_threshold": self.strategy.params.sell_threshold,
                    "w_trend": self.strategy.params.w_trend,
                    "w_momentum": self.strategy.params.w_momentum,
                    "w_rsi": self.strategy.params.w_rsi,
                    "w_volume": self.strategy.params.w_volume,
                },
                "risk": {
                    "risk_per_trade": self.risk_params.risk_per_trade,
                    "fixed_stop_pct": self.risk_params.fixed_stop_pct,
                    "take_profit_pct": self.risk_params.take_profit_pct,
                    "dd_breaker": self.risk_params.dd_breaker,
                },
                "trade": {
                    "commission_rate": self.trade_params.commission_rate,
                    "stamp_tax_rate": self.trade_params.stamp_tax_rate,
                    "slippage": self.trade_params.slippage,
                },
                "initial_cash": self.initial_cash,
                "symbols": symbols,
            },
            date_range=(str(dates[0].date()), str(dates[-1].date())),
        )

    # ------------------------------------------------------------------ #
    def _execute(
        self,
        order: dict,
        date: pd.Timestamp,
        frames: dict[str, pd.DataFrame],
        positions: dict[str, _Position],
        trades: list[Trade],
        cash: float,
    ) -> float:
        """在开盘价执行一笔订单，返回更新后的现金。"""
        tp = self.trade_params
        sym = order["symbol"]
        row = frames[sym].loc[date]
        open_px = float(row["open"])
        shares = int(order["shares"])
        side = order["side"]

        if shares <= 0 or open_px <= 0:
            return cash

        if side == "BUY":
            px = open_px * (1.0 + tp.slippage)
            shares = affordable_shares(cash, px, shares, tp)
            if shares <= 0:
                return cash
            value = px * shares
            fee = max(value * tp.commission_rate, tp.commission_min)
            cash -= value + fee
            pos = positions.setdefault(sym, _Position())
            total = pos.shares + shares
            if total > 0:
                pos.entry_price = (pos.entry_price * pos.shares + px * shares) / total
                pos.invested += value + fee
            pos.shares = total
            pos.highest = max(pos.highest, px)
            pos.atr = float(order.get("atr", pos.atr))
        else:
            px = open_px * (1.0 - tp.slippage)
            shares = min(shares, positions[sym].shares)
            if shares <= 0:
                return cash
            value = px * shares
            fee = max(value * tp.commission_rate, tp.commission_min) + value * tp.stamp_tax_rate
            cash += value - fee
            pos = positions[sym]
            # 回合盈亏：卖出所得净额 - 对应投入
            if pos.shares > 0:
                avg_invested = pos.invested * (shares / pos.shares)
                pnl = (value - fee) - avg_invested
                pnl_pct = pnl / avg_invested if avg_invested > 0 else 0.0
                pos.invested -= avg_invested
            else:
                pnl, pnl_pct = 0.0, 0.0
            pos.shares -= shares
            if pos.shares == 0:
                pos.entry_price = 0.0
                pos.invested = 0.0
                pos.highest = 0.0
                pos.atr = 0.0

        trades.append(Trade(
            date=date, symbol=sym, side=side, price=px, shares=shares,
            value=value, fee=fee, reason=order.get("reason", ""),
            pnl=pnl if side == "SELL" else 0.0,
            pnl_pct=pnl_pct if side == "SELL" else 0.0,
        ))
        return cash

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mark_equity(
        cash: float, positions: dict[str, _Position], frames: dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> float:
        """按当日收盘价计算账户权益。"""
        equity = cash
        for sym, pos in positions.items():
            if pos.shares > 0 and sym in frames and date in frames[sym].index:
                equity += pos.shares * float(frames[sym].loc[date, "close"])
                pos.highest = max(pos.highest, float(frames[sym].loc[date, "close"]))
        return equity

    # ------------------------------------------------------------------ #
    def _build_benchmark(self, symbols: list[str], dates: pd.DatetimeIndex) -> pd.Series:
        """基准：回测首日等权买入持有（不做调仓）。"""
        values = {}
        for sym in symbols:
            if sym not in self.market:
                continue
            df = self.market.get(sym)
            df = df.loc[df.index.isin(dates)]
            if df.empty:
                continue
            first_close = float(df["close"].iloc[0])
            alloc = self.initial_cash / len(symbols)
            shares = alloc / first_close
            values[sym] = df["close"] * shares
        if not values:
            return pd.Series(index=dates, dtype="float64")
        return pd.DataFrame(values).ffill().sum(axis=1).reindex(dates)


# ---------------------------------------------------------------------- #
def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: list[Trade],
    initial_cash: float,
) -> dict:
    """回测指标：年化收益、最大回撤、夏普、卡玛、胜率、盈亏比等。"""
    n = len(equity)
    if n < 2:
        return {}
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(n / TRADING_DAYS_PER_YEAR, 1e-9)
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0

    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_drawdown = float(drawdown.min())

    std = float(rets.std())
    sharpe = float(rets.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    calmar = annualized / abs(max_drawdown) if max_drawdown < 0 else (math.inf if annualized > 0 else 0.0)

    sells = [t for t in trades if t.side == "SELL"]
    wins = [t for t in sells if t.pnl > 0]
    losses = [t for t in sells if t.pnl <= 0]
    win_rate = len(wins) / len(sells) if sells else 0.0
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    metrics = {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": float(calmar),
        "volatility_annual": float(std * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "win_rate": float(win_rate),
        "trade_count": len(sells),
        "win_count": len(wins),
        "loss_count": len(losses),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "trading_days": n,
    }

    if benchmark is not None and not benchmark.empty and len(benchmark) > 1:
        bench_total = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        bench_years = max(len(benchmark) / TRADING_DAYS_PER_YEAR, 1e-9)
        bench_ann = (1.0 + bench_total) ** (1.0 / bench_years) - 1.0
        aligned = pd.concat([rets, benchmark.pct_change().dropna()], axis=1, join="inner").dropna()
        beta, alpha = 0.0, 0.0
        if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            beta = float(cov[0, 1] / cov[1, 1])
            alpha = float(annualized - beta * bench_ann)
        metrics.update({
            "benchmark_total_return": bench_total,
            "benchmark_annualized_return": float(bench_ann),
            "excess_return": float(annualized - bench_ann),
            "beta": beta,
            "alpha": alpha,
        })
    return metrics
