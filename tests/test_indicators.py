"""指标库单元测试：与手工计算值交叉验证。"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent.quant import indicators as IND


def make_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="close")


class TestSMA(unittest.TestCase):
    def test_basic(self):
        s = make_series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma5 = IND.sma(s, 5)
        self.assertAlmostEqual(ma5.iloc[4], 3.0)    # (1+2+3+4+5)/5
        self.assertAlmostEqual(ma5.iloc[-1], 8.0)   # (6+...+10)/5
        self.assertTrue(math.isnan(ma5.iloc[3]))    # 预热期

    def test_length_preserved(self):
        s = make_series(list(range(50)))
        self.assertEqual(len(IND.sma(s, 10)), 50)


class TestRSI(unittest.TestCase):
    def test_wilder_basic(self):
        # 单调上涨序列 RSI 应接近 100
        s = make_series([100 + i for i in range(60)])
        rsi = IND.rsi(s, 14)
        self.assertGreater(rsi.iloc[-1], 99)
        # 单调下跌 RSI 应接近 0
        s2 = make_series([200 - i for i in range(60)])
        self.assertLess(IND.rsi(s2, 14).iloc[-1], 1)

    def test_range(self):
        rng = np.random.default_rng(42)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)),
                      index=pd.date_range("2024-01-01", periods=300, freq="B"))
        rsi = IND.rsi(s, 14).dropna()
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())


class TestMACD(unittest.TestCase):
    def test_hist_is_2x_diff(self):
        rng = np.random.default_rng(7)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 2, 200)),
                      index=pd.date_range("2024-01-01", periods=200, freq="B"))
        dif, dea, hist = IND.macd(s)
        # hist = 2 × (DIF - DEA)
        self.assertAlmostEqual(
            float(hist.iloc[-1]), 2 * (float(dif.iloc[-1]) - float(dea.iloc[-1])),
            places=6,
        )


class TestATR(unittest.TestCase):
    def test_positive(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "high": 100 + rng.normal(0, 1, 100).cumsum() + 1,
            "low": 100 + rng.normal(0, 1, 100).cumsum() - 1,
            "close": 100 + rng.normal(0, 1, 100).cumsum(),
        }, index=pd.date_range("2024-01-01", periods=100, freq="B"))
        atr = IND.atr(df["high"], df["low"], df["close"], 14).dropna()
        self.assertTrue((atr > 0).all())


class TestKDJ(unittest.TestCase):
    def test_range(self):
        # 构造一致 OHLC：low ≤ close ≤ high（真实 K 线约束，RSV 才能落在 [0,100]）
        rng = np.random.default_rng(9)
        base = 50 + rng.normal(0, 1, 120).cumsum()
        df = pd.DataFrame({
            "high": base + np.abs(rng.normal(0, 0.5, 120)),
            "low": base - np.abs(rng.normal(0, 0.5, 120)),
            "close": base,
        }, index=pd.date_range("2024-01-01", periods=120, freq="B"))
        kdj = IND.kdj(df["high"], df["low"], df["close"])
        k, d, j = kdj[0].dropna(), kdj[1].dropna(), kdj[2].dropna()
        self.assertTrue(((k >= 0) & (k <= 100)).all())
        self.assertTrue(((d >= 0) & (d <= 100)).all())
        # J 可超出 [0,100]
        self.assertTrue(j.notna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
