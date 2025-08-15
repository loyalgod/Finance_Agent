"""FinAgent 装配工厂：一行构建完整 Agent。

用法：
    from finagent import create_agent
    agent = create_agent()
    resp = agent.ask("分析一下贵州茅台")
"""
from __future__ import annotations

from .agent import AgentResponse, FinAgent
from .config import LLMConfig
from .quant.backtest import Backtester
from .quant.data import MarketData
from .rag.retriever import Retriever

__all__ = ["create_agent"]


def create_agent(
    raw_dir=None,
    knowledge_dir=None,
    llm: LLMConfig | None = None,
    with_backtest: bool = True,
) -> FinAgent:
    """装配 FinAgent：行情库 + 知识库 + 回测器 + 工具集 + 校验器。

    数据缺失时仍可运行（知识检索可用，行情相关工具报错降级），
    保证「离线可演示」。
    """
    market = MarketData(raw_dir) if raw_dir else MarketData()
    retriever = Retriever(knowledge_dir) if knowledge_dir else Retriever()
    backtester = Backtester(market) if (with_backtest and len(market) > 0) else None

    # 用样本池真实名称增强股票实体识别（stock_list.csv 兜底）
    name_map = {sym: market.name_of(sym) for sym in market.symbols()} if len(market) else None

    return FinAgent(
        market=market,
        retriever=retriever,
        backtester=backtester,
        llm=llm,
        name_map=name_map,
    )
