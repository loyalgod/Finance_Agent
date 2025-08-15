"""FinAgent 全局配置：路径、交易参数、风控参数、策略参数、LLM 配置。

所有默认值集中在此，便于调参与复现实验。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# 股票池与基准
# ---------------------------------------------------------------------------
# 默认样本池：消费/新能源/银行/汽车/保险，跨行业以体现组合分散思路
DEFAULT_UNIVERSE = ["600519", "300750", "600036", "002594", "601318"]

# 兜底名称映射（stock_list.csv 缺失时使用，保证离线可运行）
FALLBACK_STOCK_NAMES = {
    "600519": "贵州茅台",
    "300750": "宁德时代",
    "600036": "招商银行",
    "002594": "比亚迪",
    "601318": "中国平安",
}

BENCHMARK_SYMBOL = "000300"  # 沪深300

INITIAL_CASH = 1_000_000.0   # 回测初始资金（元）

TRADING_DAYS_PER_YEAR = 252  # A股年交易日数

# ---------------------------------------------------------------------------
# 交易参数（A股规则）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TradeParams:
    commission_rate: float = 0.00025  # 佣金：万2.5（双边）
    commission_min: float = 5.0       # 单笔最低佣金 5 元
    stamp_tax_rate: float = 0.0005    # 印花税：千0.5（仅卖出，2023-08-28 减半后）
    slippage: float = 0.0005          # 滑点：0.05%（双边）
    lot_size: int = 100               # A股整手 100 股


# ---------------------------------------------------------------------------
# 风控参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskParams:
    risk_per_trade: float = 0.02     # 单笔风险预算 = 账户权益 * 2%
    atr_mult: float = 2.5            # ATR 移动止损倍数（吊灯止损）
    fixed_stop_pct: float = 0.08     # 固定止损：入场价 -8%
    take_profit_pct: float = 0.25    # 固定止盈：入场价 +25%
    max_position_pct: float = 0.30   # 单票仓位上限 30%
    max_total_pct: float = 0.95      # 总仓位上限 95%
    dd_breaker: float = 0.15         # 账户回撤熔断：回撤超 15% 清仓
    dd_resume: float = 0.10          # 回撤恢复至 10% 以内解除熔断
    cooldown_days: int = 5           # 熔断清仓后的冷静期（交易日），期满且信号转多才恢复交易


# ---------------------------------------------------------------------------
# 策略参数：多因子趋势增强策略
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    buy_threshold: float = 0.30      # 综合分 ≥ 0.30 买入/持有
    sell_threshold: float = -0.20    # 综合分 ≤ -0.20 清仓
    w_trend: float = 0.40            # 因子权重：趋势（均线排列）
    w_momentum: float = 0.30         # 因子权重：动量（20日动量滚动z分数）
    w_rsi: float = 0.15              # 因子权重：RSI（超买扣分/超卖加分）
    w_volume: float = 0.15           # 因子权重：量能（量比）
    warmup: int = 60                 # 指标预热期（交易日）


# ---------------------------------------------------------------------------
# LLM 配置（OpenAI 兼容接口，可通过环境变量指向自托管 vLLM）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    base_url: str = ""
    api_key: str = "EMPTY"
    model: str = ""
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("FINAGENT_LLM_BASE_URL", "").strip(),
            api_key=os.environ.get("FINAGENT_LLM_API_KEY", "EMPTY").strip(),
            model=os.environ.get("FINAGENT_LLM_MODEL", "").strip(),
            timeout=int(os.environ.get("FINAGENT_LLM_TIMEOUT", "60")),
        )

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)


DEFAULT_TRADE_PARAMS = TradeParams()
DEFAULT_RISK_PARAMS = RiskParams()
DEFAULT_STRATEGY_PARAMS = StrategyParams()


def ensure_dirs() -> None:
    """确保数据/输出目录存在。"""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, KNOWLEDGE_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
