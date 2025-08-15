"""量化策略回测演示：训练/测试划分 + 绩效指标 + 权益曲线图。

运行：
    python -m demo.demo_backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import create_agent  # noqa: E402
from finagent.config import OUTPUTS_DIR  # noqa: E402
from finagent.quant.backtest import Backtester  # noqa: E402
from finagent.quant.report import plot_backtest_report  # noqa: E402


def main() -> None:
    agent = create_agent()
    market = agent.market
    bt = Backtester(market)

    # 训练 / 测试划分：前 70% 训练，后 30% 样本外检验
    dates = market.common_dates()
    if dates is None or len(dates) < 60:
        print("数据不足，无法回测。请先运行：python -m finagent.data.fetch_data --start 20230101")
        return
    split = dates[int(len(dates) * 0.7)]
    print(f"回测区间：{dates[0].date()} ~ {dates[-1].date()}")
    print(f"训练 / 测试划分点：{split.date()}\n")

    print("── 全样本回测 ──")
    full = bt.run()
    print_metrics(full.metrics, full.date_range)

    print("\n── 训练段（前 70%）──")
    train = bt.run(end=str(split.date()))
    print_metrics(train.metrics, train.date_range)

    print("\n── 样本外测试段（后 30%）──")
    test = bt.run(start=str(split.date()))
    print_metrics(test.metrics, test.date_range)

    # 权益曲线图
    fig_path = OUTPUTS_DIR / "backtest_report.png"
    plot_backtest_report(full, fig_path)
    print(f"\n权益曲线已保存：{fig_path}")

    # 交易明细
    from finagent.quant.report import save_trade_log
    log_path = OUTPUTS_DIR / "trade_log.csv"
    save_trade_log(full, log_path)
    print(f"交易明细已保存：{log_path}（共 {len(full.trades)} 笔）")


def print_metrics(m: dict, date_range: tuple) -> None:
    print(f"  区间 {date_range[0]} ~ {date_range[1]}")
    print(f"  年化收益 {m.get('annualized_return', 0):+.2%} | "
          f"最大回撤 {m.get('max_drawdown', 0):.2%} | "
          f"夏普 {m.get('sharpe', 0):.2f} | 卡玛 {m.get('calmar', 0):.2f}")
    print(f"  胜率 {m.get('win_rate', 0):.1%}（{m.get('win_count', 0)}胜/{m.get('loss_count', 0)}负）| "
          f"盈亏比 {m.get('avg_win', 0) and abs(m.get('avg_win', 0) / m.get('avg_loss', 1)) or 0:.2f} | "
          f"交易 {m.get('trade_count', 0)} 笔")


if __name__ == "__main__":
    main()
