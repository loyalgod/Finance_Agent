"""意图识别与实体抽取单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.agent.intent import IntentRecognizer


class TestIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = IntentRecognizer()

    def assert_intent(self, question: str, expected: str):
        res = self.rec.classify(question)
        self.assertEqual(res.intent, expected,
                         f"「{question}」期望 {expected}，实际 {res.intent}（{res.reason}）")

    # ---- 五类意图 ----
    def test_knowledge_definitional(self):
        self.assert_intent("什么是夏普比率？", "knowledge_qa")
        self.assert_intent("MACD 金叉是什么意思", "knowledge_qa")
        self.assert_intent("夏普比率怎么算", "knowledge_qa")
        self.assert_intent("市盈率和市净率有什么区别", "knowledge_qa")

    def test_knowledge_vs_strategy(self):
        # 定义式提问优先于策略执行
        self.assert_intent("什么是最大回撤？", "knowledge_qa")
        # 执行式提问走策略解释
        self.assert_intent("你们的量化策略回测表现怎么样？胜率多少？", "strategy_explain")

    def test_stock_screen(self):
        self.assert_intent("帮我筛选 3 只趋势强的股票", "stock_screen")
        self.assert_intent("哪些股票动量强？", "stock_screen")

    def test_stock_analysis(self):
        self.assert_intent("分析一下贵州茅台", "stock_analysis")
        self.assert_intent("300750 怎么样", "stock_analysis")
        self.assert_intent("宁德时代能买吗", "stock_analysis")

    def test_strategy_explain(self):
        self.assert_intent("量化策略的胜率是多少", "strategy_explain")
        self.assert_intent("回测的年化收益和夏普怎么样", "strategy_explain")

    def test_wealth_advice(self):
        self.assert_intent("我每月能存 5000 元，该怎么理财", "wealth_advice")
        self.assert_intent("资产配置怎么分配比例", "wealth_advice")


class TestEntities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = IntentRecognizer()

    def test_stock_by_name(self):
        ent = self.rec.extract_entities("分析一下贵州茅台")
        self.assertEqual(ent.stocks[0].code, "600519")
        self.assertEqual(ent.stocks[0].name, "贵州茅台")

    def test_stock_by_code(self):
        ent = self.rec.extract_entities("300750 最近走势")
        self.assertEqual(ent.stocks[0].code, "300750")

    def test_long_name_first(self):
        # 「贵州茅台」不应被截成「茅台」匹配错误
        ent = self.rec.extract_entities("贵州茅台和宁德时代")
        codes = {s.code for s in ent.stocks}
        self.assertEqual(codes, {"600519", "300750"})

    def test_time_range(self):
        ent = self.rec.extract_entities("最近 60 天走势")
        self.assertEqual(ent.time_range_days, 60)
        ent = self.rec.extract_entities("最近 3 个月")
        self.assertEqual(ent.time_range_days, 90)

    def test_screen_conditions(self):
        ent = self.rec.extract_entities("筛选趋势强、超跌的股票")
        factors = {c["factor"] for c in ent.screen_conditions}
        self.assertIn("trend", factors)
        self.assertIn("drawdown_60d", factors)

    def test_top_n(self):
        ent = self.rec.extract_entities("帮我挑 4 只股票")
        self.assertEqual(ent.screen_top_n, 4)

    def test_wealth_amounts(self):
        ent = self.rec.extract_entities("每月能存 5000 元，存 10 年")
        self.assertEqual(ent.monthly_amount, 5000)
        self.assertEqual(ent.horizon_years, 10)
        ent = self.rec.extract_entities("每月定投 1 万元")
        self.assertEqual(ent.monthly_amount, 10000)


class TestRewrite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = IntentRecognizer()

    def test_rewrite_analysis(self):
        ent = self.rec.extract_entities("分析一下贵州茅台最近60天")
        text = self.rec.rewrite("分析一下贵州茅台最近60天", "stock_analysis", ent)
        self.assertIn("600519", text)
        self.assertIn("60", text)

    def test_rewrite_screen(self):
        ent = self.rec.extract_entities("帮我筛选 3 只趋势强的股票")
        text = self.rec.rewrite("帮我筛选 3 只趋势强的股票", "stock_screen", ent)
        self.assertIn("趋势强", text)
        self.assertIn("3", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
