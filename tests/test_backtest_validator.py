"""回测引擎与校验器单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import create_agent
from finagent.agent.validator import ResultValidator, ValidationReport, DISCLAIMER


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = create_agent()
        if len(cls.agent.market) == 0:
            raise unittest.SkipTest("无行情数据，跳过回测测试")
        cls.result = cls.agent.backtester.run()

    def test_metrics_sane(self):
        m = self.result.metrics
        self.assertIn("annualized_return", m)
        self.assertIn("max_drawdown", m)
        self.assertGreaterEqual(m["sharpe"], -5)
        self.assertLessEqual(m["max_drawdown"], 0.001, "最大回撤应为非正值")
        self.assertGreater(m["trade_count"], 0, "应有交易")
        self.assertGreaterEqual(m["win_rate"], 0)
        self.assertLessEqual(m["win_rate"], 1)

    def test_equity_curve_monotonic_dates(self):
        eq = self.result.equity_curve
        self.assertTrue(eq.index.is_monotonic_increasing)
        self.assertTrue((eq > 0).all(), "权益应为正值")

    def test_no_future_function(self):
        """交易日期不得晚于回测区间末日（无未来函数的基本约束）。"""
        import pandas as pd
        end = pd.Timestamp(self.result.date_range[1])
        for t in self.result.trades[:20]:
            self.assertLessEqual(pd.Timestamp(t.date), end)

    def test_trade_log_columns(self):
        df = self.result.trades_df()
        for col in ("date", "side", "price", "shares", "value"):
            self.assertIn(col, df.columns)


class TestValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = ResultValidator()

    def test_disclaimer_present(self):
        self.assertIn("不构成任何投资建议", DISCLAIMER)
        self.assertIn("风险", DISCLAIMER)

    def test_banned_words_intercepted(self):
        report = ValidationReport()
        out = self.v.check_answer("这只股票保证收益，稳赚不赔", report)
        self.assertFalse(report.passed, "违禁词应被拦截")
        self.assertIn("已屏蔽", out)
        self.assertNotIn("保证收益", out)

    def test_clean_answer_passes(self):
        report = ValidationReport()
        out = self.v.check_answer("该股当前均线多头排列，但波动率偏高，请注意风险。", report)
        self.assertTrue(report.passed)
        self.assertEqual(out, "该股当前均线多头排列，但波动率偏高，请注意风险。")

    def test_numeric_anomaly_detected(self):
        report = ValidationReport()
        self.v.check_tool_result("analyze_stock", {
            "close": -5, "pct_change_1d": 0, "rsi14": 50, "factors": {},
        }, report)
        self.assertFalse(report.passed, "收盘价非正数应被拦截")

    def test_rsi_out_of_range(self):
        report = ValidationReport()
        self.v.check_tool_result("analyze_stock", {
            "close": 100, "pct_change_1d": 1, "rsi14": 150, "factors": {},
        }, report)
        self.assertFalse(report.passed)

    def test_footer_attached(self):
        report = ValidationReport()
        report.data_sources.append("行情数据截至 2026-09-11（腾讯财经）")
        out = self.v.attach_compliance_footer("正文内容", report)
        self.assertIn("数据来源", out)
        self.assertIn(DISCLAIMER, out)

    def test_backtest_metrics_check(self):
        report = ValidationReport()
        self.v.check_tool_result("run_backtest", {
            "metrics": {"annualized_return": 0.06, "max_drawdown": -0.16}
        }, report)
        self.assertTrue(report.passed)


class TestAgentEndToEnd(unittest.TestCase):
    """Agent 全链路冒烟测试（模板模式）。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = create_agent()

    def test_knowledge_qa(self):
        resp = self.agent.ask("什么是夏普比率？", session_id="t-kb")
        self.assertEqual(resp.intent, "knowledge_qa")
        self.assertIn("夏普", resp.answer)
        self.assertTrue(resp.validation["passed"])
        self.assertIn("不构成任何投资建议", resp.answer)

    def test_stock_analysis_flow(self):
        resp = self.agent.ask("分析一下贵州茅台", session_id="t-stock")
        self.assertEqual(resp.intent, "stock_analysis")
        self.assertIn("600519", resp.rewritten)
        self.assertGreaterEqual(len(resp.trace), 1)
        self.assertIn("不构成任何投资建议", resp.answer)

    def test_multi_turn_entity_inheritance(self):
        sid = "t-multi"
        self.agent.ask("分析一下招商银行", session_id=sid)
        resp = self.agent.ask("那它呢", session_id=sid)
        self.assertEqual(resp.intent, "stock_analysis")
        self.assertIn("600036", resp.rewritten, "应继承上轮股票实体")

    def test_validation_report_in_response(self):
        resp = self.agent.ask("帮我筛选 2 只股票", session_id="t-screen")
        self.assertIn(resp.intent, ("stock_screen", "knowledge_qa"))
        self.assertIn("passed", resp.validation)

    def test_disclaimer_always_attached(self):
        resp = self.agent.ask("理财怎么开始", session_id="t-wealth")
        self.assertIn("风险提示", resp.answer)
        self.assertIn("数据来源", resp.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
