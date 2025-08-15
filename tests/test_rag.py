"""RAG 检索单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.rag.retriever import Retriever


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()

    def test_kb_loaded(self):
        self.assertGreater(len(self.retriever.kb), 0, "知识库应有条目")

    def test_hit_expected_concept(self):
        res = self.retriever.retrieve("什么是夏普比率")
        self.assertTrue(res.hit, "「什么是夏普比率」应检索到知识")
        titles = [c.title for c in res.chunks]
        self.assertIn("夏普比率", titles)

    def test_hit_max_drawdown(self):
        res = self.retriever.retrieve("最大回撤是什么意思")
        titles = [c.title for c in res.chunks]
        self.assertIn("最大回撤", titles)

    def test_hit_ma(self):
        res = self.retriever.retrieve("均线怎么用")
        titles = [c.title for c in res.chunks]
        self.assertTrue(any("MA" in t or "均线" in t for t in titles))

    def test_citations_format(self):
        res = self.retriever.retrieve("止损")
        cites = res.citations()
        self.assertTrue(cites, "应有引用")
        self.assertTrue(all("《" in c for c in cites))

    def test_low_score_filtered(self):
        # 与知识库无关的查询应被 min_score 过滤
        res = self.retriever.retrieve("量子引力波在天文台的数据同步问题")
        self.assertFalse(res.hit, "无关查询不应命中")


if __name__ == "__main__":
    unittest.main(verbosity=2)
