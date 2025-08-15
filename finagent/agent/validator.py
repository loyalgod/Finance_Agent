"""结果校验与风险边界控制（Agent 核心链路第 5 步）。

三道闸门（对应简历「结果校验与答案生成」中的校验环节）：

1. 数值合理性校验：对工具返回的价格、涨跌幅、技术指标、因子得分做区间检查，
   异常数据必须被拦截或降级标注，防止脏数据流入最终答案。
2. 合规校验（禁词拦截）：扫描生成内容中的违规/诱导性表述
   （保证收益、稳赚、内幕消息等），命中即记录并阻断。
3. 风险边界注入：所有涉及投资建议的回答强制附加免责声明模板，
   并统一标注数据来源与数据截止日期，保证可溯源。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ValidationReport", "ResultValidator", "DISCLAIMER"]

# ---------------------------------------------------------------------- #
# 合规红线：违禁/诱导性表述
# ---------------------------------------------------------------------- #
BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"保证[收盈]", "承诺收益类违规表述"),
    (r"稳赚", "承诺收益类违规表述"),
    (r"包赚", "承诺收益类违规表述"),
    (r"零风险", "风险误导表述"),
    (r"无风险[收盈]", "风险误导表述"),
    (r"保本保息", "风险误导表述"),
    (r"内幕", "内幕信息诱导"),
    (r"必涨|必跌", "确定性误导表述"),
    (r"承诺收益", "承诺收益类违规表述"),
]

# ---------------------------------------------------------------------- #
# 免责声明模板（强制注入）
# ---------------------------------------------------------------------- #
DISCLAIMER = (
    "⚠️ 风险提示：以上内容由量化模型基于公开市场数据自动生成，"
    "仅供学习与研究参考，不构成任何投资建议或买卖推荐。"
    "市场有风险，投资需谨慎，历史业绩不代表未来表现。"
)


@dataclass
class ValidationReport:
    """单次校验报告。"""

    passed: bool = True
    issues: list[str] = field(default_factory=list)     # 阻断级问题（不通过）
    warnings: list[str] = field(default_factory=list)   # 提示级问题（降级放行）
    data_sources: list[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)
        self.warnings.extend(other.warnings)
        self.data_sources.extend(other.data_sources)
        if not other.passed:
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "data_sources": self.data_sources,
        }


class ResultValidator:
    """结果校验器：数值合理性 + 禁词拦截 + 风险边界注入。"""

    # ------------------------------------------------------------------ #
    # 1. 数值合理性校验（工具结果 → 答案前拦截）
    # ------------------------------------------------------------------ #
    def check_tool_result(self, tool: str, data: dict, report: ValidationReport) -> None:
        """对单个工具结果做数值区间检查，异常写入 report。"""
        if not isinstance(data, dict) or not data:
            return
        try:
            if tool == "analyze_stock":
                self._check_analysis(data, report)
            elif tool == "screen_stocks":
                self._check_screen(data, report)
            elif tool == "get_price_history":
                self._check_history(data, report)
            elif tool == "run_backtest":
                self._check_backtest(data, report)
            elif tool == "financial_calc":
                self._check_calc(data, report)
        except (TypeError, ValueError, KeyError) as err:  # noqa: PERF203
            report.add_warning(f"工具 {tool} 结果校验跳过：{err}")

    def _check_analysis(self, d: dict, r: ValidationReport) -> None:
        close = float(d.get("close", 0))
        if close <= 0:
            r.add_issue(f"个股分析：收盘价 {close} 非正数，数据异常")
        pct = abs(float(d.get("pct_change_1d", 0)))
        if pct > 30:
            r.add_issue(f"个股分析：单日涨跌幅 {pct}% 超出 A 股正常波动区间")
        rsi = float(d.get("rsi14", 50))
        if not 0 <= rsi <= 100:
            r.add_issue(f"个股分析：RSI(14)={rsi} 超出 [0,100] 合理区间")
        factors = d.get("factors") or {}
        for k, v in factors.items():
            if not -1.0 <= float(v) <= 1.0:
                r.add_warning(f"个股分析：因子 {k}={v} 超出 [-1,1] 常规范围，已按原始值展示")
        if d.get("as_of"):
            r.data_sources.append(f"行情数据截至 {d['as_of']}（腾讯财经/东方财富）")

    def _check_screen(self, d: dict, r: ValidationReport) -> None:
        rows = d.get("rows") or []
        if not rows:
            r.add_issue("选股结果为空：样本池不足或数据缺失")
        for row in rows:
            if float(row.get("close", 0)) <= 0:
                r.add_issue(f"选股结果：{row.get('name', row.get('symbol'))} 收盘价非正数")
        if d.get("as_of"):
            r.data_sources.append(f"选股数据截至 {d['as_of']}（腾讯财经/东方财富）")

    def _check_history(self, d: dict, r: ValidationReport) -> None:
        if float(d.get("start_close", 0)) <= 0 or float(d.get("end_close", 0)) <= 0:
            r.add_issue("区间行情：起止收盘价非正数，数据异常")
        if d.get("end"):
            r.data_sources.append(f"行情区间 {d.get('start')} ~ {d['end']}（腾讯财经/东方财富）")

    def _check_backtest(self, d: dict, r: ValidationReport) -> None:
        m = d.get("metrics") or {}
        ann = float(m.get("annualized_return", 0))
        mdd = float(m.get("max_drawdown", 0))
        if abs(ann) > 3:
            r.add_warning(f"回测年化收益 {ann:.0%} 偏离常见区间，请人工复核回测设置")
        if mdd > -0.01:
            r.add_warning(f"回测最大回撤 {mdd:.2%} 异常偏小，请复核回测设置")
        r.data_sources.append("回测数据为本地样本池历史行情（前复权）")

    def _check_calc(self, d: dict, r: ValidationReport) -> None:
        fv = float(d.get("future_value", 0))
        if fv < 0:
            r.add_issue(f"金融计算：终值 {fv} 为负，参数异常")
        r.data_sources.append("金融计算结果为数学测算，不构成收益预期")

    # ------------------------------------------------------------------ #
    # 2. 合规校验（禁词拦截）
    # ------------------------------------------------------------------ #
    def check_answer(self, answer: str, report: ValidationReport) -> str:
        """扫描答案中的违禁表述；命中即记录（模板答案本身由构造保证安全，
        LLM 生成内容必须经过本闸门）。返回处理后的答案。"""
        cleaned = answer
        for pattern, label in BANNED_PATTERNS:
            m = re.search(pattern, cleaned)
            if m:
                hit = m.group(0)
                report.add_issue(f"合规校验拦截：答案包含「{hit}」（{label}）")
                cleaned = cleaned.replace(hit, "[已屏蔽的违规表述]")
        return cleaned

    # ------------------------------------------------------------------ #
    # 3. 风险边界注入（免责声明 + 数据溯源）
    # ------------------------------------------------------------------ #
    def attach_compliance_footer(self, answer: str, report: ValidationReport) -> str:
        """附加免责声明与数据来源，形成合规收尾。"""
        parts = [answer.rstrip()]
        sources = list(dict.fromkeys(report.data_sources))  # 去重保序
        if sources:
            parts.append("\n---\n📎 数据来源：" + "；".join(sources))
        parts.append("\n" + DISCLAIMER)
        return "\n".join(parts)
