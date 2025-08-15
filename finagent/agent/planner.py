"""复杂任务拆解：按意图将问题分解为可执行的任务计划（步骤序列）。

每个计划 = 有序步骤列表，步骤为「检索」或「工具调用」，
执行轨迹即简历中的「复杂任务拆解 → 专业工具调用」环节。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .intent import Entities

__all__ = ["PlanStep", "TaskPlanner"]

_STAGE_RETRIEVE = "retrieve"   # 知识/数据检索
_STAGE_TOOL = "tool"           # 专业工具调用


@dataclass
class PlanStep:
    """单个执行步骤。"""

    index: int
    stage: str        # retrieve / tool
    tool: str         # 工具名（retrieve 阶段为 search_knowledge）
    args: dict = field(default_factory=dict)
    description: str = ""
    status: str = "pending"   # pending / ok / failed / skipped
    summary: str = ""         # 执行结果摘要（进入轨迹）

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "stage": self.stage,
            "tool": self.tool,
            "args": self.args,
            "description": self.description,
            "status": self.status,
            "summary": self.summary,
        }


class TaskPlanner:
    """按意图模板生成任务计划。"""

    def decompose(
        self,
        intent: str,
        rewritten: str,
        entities: Entities,
        context=None,
    ) -> list[PlanStep]:
        if intent == "stock_analysis":
            return self._plan_stock_analysis(rewritten, entities)
        if intent == "stock_screen":
            return self._plan_stock_screen(rewritten, entities)
        if intent == "strategy_explain":
            return self._plan_strategy(rewritten)
        if intent == "wealth_advice":
            return self._plan_wealth(rewritten, entities)
        return self._plan_knowledge(rewritten)

    # ------------------------------------------------------------------ #
    def _plan_knowledge(self, rewritten: str) -> list[PlanStep]:
        return [
            PlanStep(1, _STAGE_RETRIEVE, "search_knowledge",
                     {"query": rewritten},
                     "检索金融知识库，定位相关概念条目"),
        ]

    def _plan_wealth(self, rewritten: str, entities: Entities) -> list[PlanStep]:
        steps = [
            PlanStep(1, _STAGE_RETRIEVE, "search_knowledge",
                     {"query": "资产配置 定投 理财 应急备用金"},
                     "检索理财基础：资产配置框架与定投知识"),
        ]
        if entities.monthly_amount and entities.horizon_years:
            steps.append(PlanStep(
                2, _STAGE_TOOL, "financial_calc",
                {"kind": "dca", "monthly": entities.monthly_amount,
                 "annual_rate": 0.08, "years": entities.horizon_years},
                f"定投测算：每月 {entities.monthly_amount:.0f} 元、"
                f"{entities.horizon_years} 年、假设年化 8%",
            ))
        steps.append(PlanStep(
            len(steps) + 1, _STAGE_RETRIEVE, "search_knowledge",
            {"query": "风险承受能力 风险管理"},
            "检索风险管理知识，输出风险边界提示",
        ))
        return steps

    def _plan_stock_screen(self, rewritten: str, entities: Entities) -> list[PlanStep]:
        conds = entities.screen_conditions or [{"factor": "composite", "dir": "desc", "label": "综合评分高"}]
        return [
            PlanStep(1, _STAGE_TOOL, "screen_stocks",
                     {"conditions": conds, "top_n": entities.screen_top_n},
                     "在样本股票池上计算全部因子，按条件打分排序"),
            PlanStep(2, _STAGE_RETRIEVE, "search_knowledge",
                     {"query": "动量 趋势 多因子 选股"},
                     "检索选股方法论知识，解释筛选逻辑"),
        ]

    def _plan_stock_analysis(self, rewritten: str, entities: Entities) -> list[PlanStep]:
        stock = entities.stocks[0] if entities.stocks else None
        steps = [
            PlanStep(1, _STAGE_TOOL, "analyze_stock",
                     {"symbol": stock.code if stock else ""},
                     f"拉取 {stock.name if stock else '目标股票'} 行情并计算技术指标与量化因子"),
        ]
        if entities.time_range_days:
            steps.append(PlanStep(
                2, _STAGE_TOOL, "get_price_history",
                {"symbol": stock.code if stock else "",
                 "days": min(entities.time_range_days, 120)},
                f"拉取近 {entities.time_range_days} 日行情走势",
            ))
        steps.append(PlanStep(
            len(steps) + 1, _STAGE_RETRIEVE, "search_knowledge",
            {"query": "均线 MACD RSI KDJ 技术指标 估值"},
            "检索技术指标知识，解释信号含义",
        ))
        return steps

    def _plan_strategy(self, rewritten: str) -> list[PlanStep]:
        return [
            PlanStep(1, _STAGE_TOOL, "run_backtest",
                     {"symbols": None, "start": None, "end": None},
                     "运行多因子趋势策略回测（全样本池）"),
            PlanStep(2, _STAGE_RETRIEVE, "search_knowledge",
                     {"query": "回测 年化收益 最大回撤 夏普 胜率 指标"},
                     "检索回测指标知识，解释各项指标含义"),
        ]
