"""回测报告：权益曲线图（matplotlib，自动注册中文字体）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from ..config import OUTPUTS_DIR  # noqa: E402
from .backtest import BacktestResult  # noqa: E402

# 配色（浅色主题；回撤=绿，符合国内「跌绿」习惯）
COLOR_EQUITY = "#12395B"      # 深蓝：策略净值
COLOR_BENCH = "#8a94a6"       # 灰：基准
COLOR_DD = "#1a9e5c"          # 绿：回撤
COLOR_TEXT = "#1f2733"
COLOR_GRID = "#d8dee9"

_FONT_REGISTERED = False


def _ensure_cjk_font() -> None:
    """注册 Windows 中文字体（黑体/微软雅黑），失败则用默认字体。"""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),      # 黑体
        Path("C:/Windows/Fonts/msyh.ttc"),        # 微软雅黑
        Path("C:/Windows/Fonts/simsun.ttc"),      # 宋体
    ]
    for path in candidates:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:  # noqa: BLE001
                continue
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_REGISTERED = True


def plot_backtest_report(result: BacktestResult, out_path: Path | str | None = None) -> Path:
    """绘制权益曲线 + 回撤双子图，返回图片路径。"""
    _ensure_cjk_font()
    out = Path(out_path) if out_path else OUTPUTS_DIR / "backtest_equity.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    equity = result.equity_curve / result.equity_curve.iloc[0]
    bench = result.benchmark_curve
    if bench is not None and not bench.empty:
        bench = bench / bench.iloc[0]
    dd = result.drawdown_curve

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    # 上图：净值曲线
    ax1.set_facecolor("white")
    ax1.plot(equity.index, equity.values, color=COLOR_EQUITY, lw=2.0, label="策略净值")
    if bench is not None and not bench.empty:
        ax1.plot(bench.index, bench.values, color=COLOR_BENCH, lw=1.4,
                 ls="--", label="基准（等权买入持有）")
    ax1.axhline(1.0, color=COLOR_GRID, lw=0.8)
    ax1.set_ylabel("净值（起点=1）", color=COLOR_TEXT)
    ax1.set_title("策略净值曲线与回撤", color=COLOR_TEXT, fontsize=14, pad=12)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(color=COLOR_GRID, lw=0.6, alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_color(COLOR_GRID)
    ax1.tick_params(colors=COLOR_TEXT)

    # 指标注释框
    m = result.metrics
    info = (
        f"年化收益 {m.get('annualized_return', 0):+.1%}    "
        f"最大回撤 {m.get('max_drawdown', 0):.1%}    "
        f"夏普 {m.get('sharpe', 0):.2f}\n"
        f"胜率 {m.get('win_rate', 0):.1%}    "
        f"交易次数 {m.get('trade_count', 0)}    "
        f"卡玛 {m.get('calmar', 0):.2f}"
    )
    ax1.text(
        0.985, 0.04, info, transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=9.5, color=COLOR_TEXT,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f7fb", ec=COLOR_GRID),
    )

    # 下图：回撤
    ax2.set_facecolor("white")
    ax2.fill_between(dd.index, dd.values, 0.0, color=COLOR_DD, alpha=0.35)
    ax2.plot(dd.index, dd.values, color=COLOR_DD, lw=1.0)
    ax2.set_ylabel("回撤", color=COLOR_TEXT)
    ax2.grid(color=COLOR_GRID, lw=0.6, alpha=0.7)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_GRID)
    ax2.tick_params(colors=COLOR_TEXT)

    fig.autofmt_xdate()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_trade_log(result: BacktestResult, out_path: Path | str | None = None) -> Path:
    """导出成交明细 CSV。"""
    out = Path(out_path) if out_path else OUTPUTS_DIR / "backtest_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.trades_df().to_csv(out, index=False, encoding="utf-8-sig")
    return out
