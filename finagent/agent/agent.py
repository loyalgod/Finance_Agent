"""FinAgent 主类：金融投研 Agent 核心链路全串联。

完整链路（对应简历「Agent 核心链路」）：

    用户问题
      │
      ▼
 ┌────────────────────────────────────────────────────┐
 │ ① 意图识别与问题改写   IntentRecognizer               │
 │    （五类意图关键词加权 + 实体抽取 + 口语改写为结构化任务）│
 │ ② 复杂任务拆解         TaskPlanner                    │
 │    （按意图模板生成 检索/工具 两阶段步骤计划）            │
 │ ③ 金融知识/行情数据检索  Retriever(RAG) / MarketData   │
 │ ④ 专业工具调用         ToolRegistry（六大工具）         │
 │ ⑤ 结果校验与风险边界控制 ResultValidator               │
 │    （数值合理性校验 / 禁词拦截 / 免责声明注入 / 数据溯源）  │
 │ ⑥ 答案生成             模板引擎（离线） / LLM 增强（可选） │
 └────────────────────────────────────────────────────┘
      │
      ▼
    结构化答案（含风险提示 + 数据来源）

多轮上下文：ConversationContext（滑动窗口 + 实体承接「那招商银行呢」
+ 意图延续「它的胜率呢」）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import LLMConfig
from ..quant.data import MarketData
from .context import ConversationContext, Turn
from .intent import Entities, IntentRecognizer, StockEntity
from .planner import PlanStep, TaskPlanner
from .tools import ToolRegistry, ToolResult, build_default_tools, _json_safe
from .validator import ResultValidator, ValidationReport

__all__ = ["FinAgent", "AgentResponse"]


# ---------------------------------------------------------------------- #
# LLM 系统提示词（答案生成 LLM 模式）
# ---------------------------------------------------------------------- #
_LLM_SYSTEM_PROMPT = """你是 FinAgent，鲸兆量化实验室研发的面向个人投资者的金融投研助手。

回答必须遵守以下红线：
1. 只能基于下方「工具执行结果」作答，禁止编造任何数据；结果中没有的信息不要臆测。
2. 数字必须与工具结果完全一致；引用知识库内容时注明条目名称。
3. 涉及个股、策略、投资建议时必须客观呈现风险，禁止出现"保证收益""稳赚""内幕"等表述。
4. 结构清晰，使用 Markdown（标题 / 列表 / 表格），先给核心结论再展开细节。
5. 不要自行添加免责声明（系统会自动附加）。
6. 如果工具结果中有失败项，如实说明该项数据不可用，不要掩盖。"""

_LLM_USER_TEMPLATE = """【对话历史】
{history}

【用户问题】{question}

【改写后的投研任务】{rewritten}

【工具执行结果（JSON）】
{tool_payload}

请基于以上工具执行结果，生成专业、可读的投研回答。"""


@dataclass
class AgentResponse:
    """Agent 单次问答的完整响应（含链路轨迹，供演示与审计）。"""

    question: str
    session_id: str
    intent: str
    intent_label: str
    intent_confidence: float
    intent_reason: str
    rewritten: str
    plan: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    mode: str = "template"          # template / llm
    tool_results: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "session_id": self.session_id,
            "intent": self.intent,
            "intent_label": self.intent_label,
            "intent_confidence": self.intent_confidence,
            "intent_reason": self.intent_reason,
            "rewritten": self.rewritten,
            "plan": self.plan,
            "trace": self.trace,
            "answer": self.answer,
            "citations": self.citations,
            "validation": self.validation,
            "mode": self.mode,
            "tool_results": self.tool_results,
            "elapsed_ms": self.elapsed_ms,
        }


class FinAgent:
    """金融投研 Agent 主类。

    用法：
        agent = FinAgent(market=market, retriever=retriever, backtester=bt)
        resp = agent.ask("分析一下贵州茅台最近走势")
        print(resp.answer)          # 结构化答案（含风险提示）
        print(resp.trace)           # 链路执行轨迹（意图→拆解→工具→校验）
    """

    def __init__(
        self,
        market: MarketData,
        retriever=None,
        backtester=None,
        llm: LLMConfig | None = None,
        name_map: dict[str, str] | None = None,
    ):
        self.market = market
        self.retriever = retriever
        self.backtester = backtester
        self.llm = llm if llm is not None else LLMConfig.from_env()
        # 链路组件
        self.recognizer = IntentRecognizer(name_map=name_map)
        self.planner = TaskPlanner()
        self.context = ConversationContext()
        self.validator = ResultValidator()
        self.tools: ToolRegistry = build_default_tools(market, retriever, backtester)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def ask(self, question: str, session_id: str = "default") -> AgentResponse:
        """问答主入口：执行完整 Agent 链路。"""
        t0 = time.time()

        # ① 意图识别 + 实体抽取（含多轮承接）
        intent_res = self.recognizer.classify(question)
        entities = self.recognizer.extract_entities(question)
        intent, entities = self._apply_context(session_id, question, intent_res, entities)

        # ② 问题改写：口语 → 结构化投研任务
        rewritten = self.recognizer.rewrite(question, intent, entities)

        # ③ 复杂任务拆解
        plan = self.planner.decompose(intent, rewritten, entities, self.context)

        # ④ 执行计划（检索 + 工具调用，失败步骤降级继续）
        trace, tool_data, retrievals = self._execute_plan(plan)

        # ⑤ 结果校验（数值合理性 / 禁词 / 风险边界）
        report = self._validate(tool_data)

        # ⑥ 答案生成（LLM 优先，失败降级模板）
        answer, mode = self._generate_answer(
            question, rewritten, intent, entities, tool_data, retrievals, session_id
        )
        # 合规闸门：禁词拦截 + 免责声明 + 数据溯源
        answer = self.validator.check_answer(answer, report)
        answer = self.validator.attach_compliance_footer(answer, report)

        # 记录多轮上下文
        self.context.add_turn(session_id, Turn(
            question=question,
            intent=intent,
            rewritten=rewritten,
            answer=answer,
            stocks=[s.code for s in entities.stocks],
            summary=f"{intent_res.label} → {len(plan)} 步计划，{len(trace)} 步执行",
        ))

        citations = sorted({c for r in retrievals for c in (r.get("citations") or [])})
        return AgentResponse(
            question=question,
            session_id=session_id,
            intent=intent,
            intent_label=intent_res.label,
            intent_confidence=intent_res.confidence,
            intent_reason=intent_res.reason,
            rewritten=rewritten,
            plan=[s.to_dict() for s in plan],
            trace=trace,
            answer=answer,
            citations=citations,
            validation=report.to_dict(),
            mode=mode,
            tool_results=_json_safe(tool_data),
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    def reset(self, session_id: str = "default") -> None:
        """清空会话上下文。"""
        self.context.clear(session_id)

    def describe_tools(self) -> list[dict]:
        """列出已注册工具（供 UI / 调试）。"""
        return self.tools.describe()

    # ------------------------------------------------------------------ #
    # ① 多轮上下文承接
    # ------------------------------------------------------------------ #
    def _apply_context(self, session_id, question, intent_res, entities):
        """实体承接 + 意图延续（多轮上下文管理）。"""
        intent = intent_res.intent
        reason = intent_res.reason

        # 实体承接 A：分析类问题未提到股票 → 继承上一轮股票
        if intent == "stock_analysis" and not entities.stocks:
            last = self.context.last_stocks(session_id)
            if last:
                code = last[0]
                name = self.market.name_of(code) if code in self.market else code
                entities.stocks.append(StockEntity(code=code, name=name))
                reason += "；多轮实体承接（沿用上轮股票）"

        # 意图延续 A：兜底知识问答 + 上轮个股分析 + 指示代词（它/这只/那这）→ 延续个股分析
        if (intent == "knowledge_qa"
                and not entities.stocks
                and self.context.last_intent(session_id) == "stock_analysis"
                and any(k in question for k in ("它", "这只", "那这"))):
            last = self.context.last_stocks(session_id)
            if last:
                code = last[0]
                name = self.market.name_of(code) if code in self.market else code
                entities.stocks.append(StockEntity(code=code, name=name))
                intent = "stock_analysis"
                reason = "多轮意图延续（承接上轮个股分析）+ 实体承接"

        # 意图升级 B：纯兜底（无任何知识模式命中）+ 本轮已抽到股票实体 → 升级为个股分析
        if intent == "knowledge_qa" and entities.stocks and intent_res.scores["knowledge_qa"] == 0:
            intent = "stock_analysis"
            reason += "；问题含股票实体，升级为个股分析"

        # 意图延续 B：知识问答兜底 + 上一轮是策略解释 + 命中指标词 → 延续策略解释
        if intent == "knowledge_qa" and self.context.last_intent(session_id) == "strategy_explain":
            if any(k in question for k in ("胜率", "回撤", "夏普", "收益", "盈亏", "净值", "年化")):
                intent = "strategy_explain"
                reason = "多轮意图延续（承接上轮策略解释）"

        intent_res.intent = intent
        intent_res.reason = reason
        return intent, entities

    # ------------------------------------------------------------------ #
    # ③ 计划执行（检索 + 工具调用）
    # ------------------------------------------------------------------ #
    def _execute_plan(self, plan: list[PlanStep]):
        trace: list[dict] = []
        tool_data: dict = {}
        retrievals: list[dict] = []

        for step in plan:
            result: ToolResult = self.tools.call(step.tool, **step.args)
            if result.ok:
                step.status = "ok"
                step.summary = result.summary
                if step.tool == "search_knowledge":
                    retrievals.append(result.data)
                    tool_data.setdefault("_retrievals", []).append(result.data)
                else:
                    tool_data[step.tool] = result.data
            else:
                step.status = "failed"
                step.summary = f"执行失败：{result.error}"
                # 失败不中断：后续步骤继续（降级容错）
            trace.append(step.to_dict())

        return trace, tool_data, retrievals

    # ------------------------------------------------------------------ #
    # ⑤ 结果校验
    # ------------------------------------------------------------------ #
    def _validate(self, tool_data: dict) -> ValidationReport:
        report = ValidationReport()
        for tool_name, data in tool_data.items():
            if tool_name == "_retrievals":
                continue
            self.validator.check_tool_result(tool_name, data, report)
        # 知识检索溯源：凡经过 RAG 的回答都标注知识库来源
        if tool_data.get("_retrievals"):
            report.data_sources.append("金融知识库（BM25 检索）")
        return report

    # ------------------------------------------------------------------ #
    # ⑥ 答案生成：LLM / 模板双模
    # ------------------------------------------------------------------ #
    def _generate_answer(self, question, rewritten, intent, entities,
                         tool_data, retrievals, session_id):
        if self.llm.available:
            answer = self._generate_llm(
                question, rewritten, tool_data, retrievals, session_id
            )
            if answer:
                return answer, "llm"
        return (
            self._generate_template(intent, entities, tool_data, retrievals),
            "template",
        )

    def _generate_llm(self, question, rewritten, tool_data, retrievals, session_id):
        """调用 OpenAI 兼容接口（可指向自托管 vLLM）。失败返回 None 降级模板。"""
        try:
            import requests  # 延迟导入，离线环境也能运行模板模式

            history = self.context.history_text(session_id)
            user_prompt = _LLM_USER_TEMPLATE.format(
                history=history,
                question=question,
                rewritten=rewritten,
                tool_payload=_json_safe(tool_data),
            )
            url = self.llm.base_url.rstrip("/") + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.llm.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.llm.model,
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            }
            resp = requests.post(url, headers=headers, json=body, timeout=self.llm.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001 - LLM 不可用时静默降级，保证链路可用
            return None

    # ------------------------------------------------------------------ #
    # 模板引擎（离线答案生成）
    # ------------------------------------------------------------------ #
    def _generate_template(self, intent, entities, tool_data, retrievals):
        if intent == "stock_analysis":
            return self._tpl_stock_analysis(tool_data)
        if intent == "stock_screen":
            return self._tpl_stock_screen(tool_data)
        if intent == "strategy_explain":
            return self._tpl_strategy(tool_data, retrievals)
        if intent == "wealth_advice":
            return self._tpl_wealth(tool_data, retrievals)
        return self._tpl_knowledge(retrievals)

    # ---- 个股分析 ----
    def _tpl_stock_analysis(self, tool_data: dict) -> str:
        d = tool_data.get("analyze_stock")
        if d is None:
            return "抱歉，该个股数据拉取失败（可能不在样本池内）。样本池：" + \
                   "、".join(self.market.symbols())
        hist = tool_data.get("get_price_history")
        ma, macd, kdj, boll = d["ma"], d["macd"], d["kdj"], d["boll"]
        fr = d.get("factor_rank") or {}

        lines = [
            f"## {d['name']}（{d['symbol']}）个股分析",
            f"\n> 数据截至 {d['as_of']}，基于本地行情库（前复权日线）计算\n",
            "### 一、行情概览\n",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 最新收盘 | **{d['close']:.2f} 元**（{d['pct_change_1d']:+.2f}%） |",
            f"| 52 周区间 | {d['low_52w']:.2f} ~ {d['high_52w']:.2f} 元 |",
            f"| 52 周价格分位 | {d['position_52w_pct']:.1f}%（0%=区间最低，100%=区间最高） |",
            f"| 20 日年化波动率 | {d['vol20']:.1f}% |",
            f"| 20/60 日动量 | {d['mom20']:+.2f}% / {d['mom60']:+.2f}% |",
            "",
            "### 二、技术面信号\n",
            f"- **均线**：{ma['state']}（MA5={ma['ma5']:.2f}，MA20={ma['ma20']:.2f}，MA60={ma['ma60']:.2f}）",
            f"- **MACD**：{macd['state']}（DIF={macd['dif']:.3f}，DEA={macd['dea']:.3f}，柱={macd['hist']:+.3f}）",
            f"- **RSI(14)**：{d['rsi14']:.1f}",
            f"- **KDJ**：K={kdj['k']:.1f}，D={kdj['d']:.1f}，J={kdj['j']:.1f}（{kdj['state']}）",
            f"- **BOLL**：上轨 {boll['up']:.2f} / 中轨 {boll['mid']:.2f} / 下轨 {boll['dn']:.2f}",
            f"- **ATR(14)**：{d['atr14']:.2f} 元（约占收盘价 {d['atr_pct']:.2f}%）",
            "",
            "### 三、量化因子评分\n",
            "| 因子 | 得分 | 权重 | 说明 |",
            "| --- | --- | --- | --- |",
            f"| 趋势 | {d['factors']['trend']:+.3f} | 40% | 均线排列关系 |",
            f"| 动量 | {d['factors']['momentum']:+.3f} | 30% | 20 日动量滚动 z 分数 |",
            f"| RSI | {d['factors']['rsi']:+.3f} | 15% | 超买扣分 / 超卖加分 |",
            f"| 量能 | {d['factors']['volume']:+.3f} | 15% | 量比 |",
            f"| **综合** | **{d['factors']['composite']:+.3f}** | — | 加权合成 |",
            "",
        ]
        if fr:
            lines.append(
                f"样本池排名：**{fr.get('composite_rank')} / {fr.get('pool_size')}**"
                f"（综合分 {fr.get('composite_score')}）\n"
            )
        if d.get("risk_notes"):
            lines.append("### 四、风险提示\n")
            lines += [f"- {n}" for n in d["risk_notes"]]
            lines.append("")
        if hist:
            lines.append(
                f"**区间走势**：近 {hist['days']} 个交易日"
                f"（{hist['start']} ~ {hist['end']}）累计 {hist['change_pct']:+.2f}%"
            )
        return "\n".join(lines)

    # ---- 自然语言选股 ----
    def _tpl_stock_screen(self, tool_data: dict) -> str:
        d = tool_data.get("screen_stocks")
        if d is None or not d.get("rows"):
            return "选股执行失败：样本池数据不足或条件解析异常。"
        rows = d["rows"]
        lines = [
            "## 自然语言选股结果",
            f"\n> 筛选条件：{d['conditions_desc']}；数据截至 {d['as_of']}；样本池 {len(rows)} 只入选\n",
            "| 排名 | 代码 | 名称 | 收盘价(元) | 20日涨幅 | 年化波动 | 量比 | 综合分 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['rank']} | {r['symbol']} | {r['name']} | {r['close']:.2f} "
                f"| {r['ret_20d']:+.2f}% | {r['vol20']:.1f}% | {r['vol_ratio']:.2f} "
                f"| {r['composite']:+.3f} |"
            )
        lines += [
            "",
            "**筛选逻辑**：在样本股票池内计算趋势 / 动量 / RSI / 量能等因子，"
            "每个条件取池内百分位（方向可逆）后打分，综合排序取前 N。",
            "**注意事项**：选股结果仅反映历史量化特征，不构成买卖推荐；"
            "样本池规模有限，结果仅供演示与方法验证。",
        ]
        return "\n".join(lines)

    # ---- 策略解释 ----
    def _tpl_strategy(self, tool_data: dict, retrievals: list) -> str:
        d = tool_data.get("run_backtest")
        if d is None:
            return "回测执行失败：请检查行情数据完整性。"
        m = d.get("metrics") or {}
        lines = [
            "## 多因子趋势策略回测",
            f"\n> 回测区间：{d['date_range'][0]} ~ {d['date_range'][1]}；"
            f"初始资金 100 万元；已计入佣金 / 印花税 / 滑点\n",
            "### 核心绩效指标\n",
            "| 指标 | 数值 | 含义 |",
            "| --- | --- | --- |",
            f"| 年化收益 | **{m.get('annualized_return', 0):+.2%}** | 策略净值年化增速 |",
            f"| 最大回撤 | {m.get('max_drawdown', 0):.2%} | 净值从高点的最大跌幅 |",
            f"| 夏普比率 | {m.get('sharpe', 0):.2f} | 每单位总风险的超额收益 |",
            f"| 卡玛比率 | {m.get('calmar', 0):.2f} | 年化收益 / 最大回撤 |",
            f"| 年化波动率 | {m.get('volatility_annual', 0):.2%} | 净值波动水平 |",
            f"| 胜率 | {m.get('win_rate', 0):.1%} | 盈利卖出交易占比 |",
            f"| 盈亏比 | {m.get('avg_win', 0) and m.get('avg_loss', 0) and abs(m['avg_win'] / m['avg_loss']) or 0:.2f} | 平均盈利 / 平均亏损 |",
            f"| 交易次数 | {m.get('trade_count', 0)} | 完整卖出次数 |",
        ]
        if "beta" in m:
            lines += [
                f"| Beta | {m['beta']:.2f} | 相对基准的系统性风险 |",
                f"| 超额收益(年化) | {m.get('excess_return', 0):+.2%} | 相对沪深300基准 |",
            ]
        lines += [
            "",
            f"**策略逻辑**：{d.get('strategy_desc', '')}",
            "",
            "**风控机制**：单笔风险预算 2%（ATR 仓位管理）；固定止损 8% / "
            "ATR 吊灯止损 2.5×；止盈 25%；账户回撤 15% 熔断清仓 + 5 日冷静期。",
        ]
        # 指标知识补充（RAG）
        kb_notes = self._retrieval_notes(retrievals, max_chunks=2, max_len=120)
        if kb_notes:
            lines += ["", "**指标解读（知识库）**", kb_notes]
        return "\n".join(lines)

    # ---- 理财咨询 ----
    def _tpl_wealth(self, tool_data: dict, retrievals: list) -> str:
        calc = tool_data.get("financial_calc")
        lines = ["## 理财规划建议\n"]
        if calc:
            lines += [
                "### 定投测算（假设年化 8%）\n",
                f"- 每月投入：**{calc['monthly']:.0f} 元**，期限 {calc['years']:.0f} 年",
                f"- 累计投入：{calc['invested']:.0f} 元",
                f"- 期末市值（测算）：约 **{calc['future_value']:.0f} 元**",
                f"- 累计收益（测算）：约 {calc['gain']:.0f} 元\n",
                "> 注：以上为数学测算，假设年化收益 8% 恒定不变；"
                "实际投资收益存在波动，可能低于预期甚至亏损。\n",
            ]
        lines.append("### 理财知识要点\n")
        kb_notes = self._retrieval_notes(retrievals, max_chunks=3, max_len=150)
        lines.append(kb_notes or "（知识库暂未检索到相关条目）")
        lines += [
            "",
            "### 风险边界提示\n",
            "- **应急先行**：先备足 3~6 个月生活费作为应急金（货币基金 / 活期），再考虑投资；",
            "- **匹配风险承受能力**：根据收入稳定性与风险测评结果确定权益类资产比例；",
            "- **分散配置**：不要把全部资金投入单一资产；",
            "- **不借钱投资**：杠杆会放大亏损，普通投资者应避免。",
        ]
        return "\n".join(lines)

    # ---- 金融知识问答 ----
    def _tpl_knowledge(self, retrievals: list) -> str:
        chunks = []
        for r in retrievals:
            for c in r.get("chunks", []):
                if not any(c["title"] == x["title"] for x in chunks):
                    chunks.append(c)
        if not chunks:
            return ("抱歉，当前知识库中没有与该问题直接匹配的条目。"
                    "你可以换个问法，例如「什么是市盈率」「均线怎么用」「什么是夏普比率」等。")
        lines = ["## 金融知识问答\n"]
        for i, c in enumerate(chunks[:3], 1):
            lines += [f"### {i}. {c['title']}", c["text"], ""]
        refs = "、".join(f"《{c['category']}》" for c in chunks[:3])
        lines.append(f"**参考资料**：{refs}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _retrieval_notes(retrievals: list, max_chunks: int = 2, max_len: int = 120) -> str:
        """从检索结果提炼要点（知识库补充说明）。"""
        seen = set()
        parts = []
        for r in retrievals:
            for c in r.get("chunks", []):
                if c["title"] in seen:
                    continue
                seen.add(c["title"])
                text = c["text"].replace("\n", " ")
                if len(text) > max_len:
                    text = text[:max_len] + "…"
                parts.append(f"- **{c['title']}**：{text}")
                if len(parts) >= max_chunks:
                    return "\n".join(parts)
        return "\n".join(parts)
