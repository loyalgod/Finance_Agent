"""专业工具调用：工具注册表 + 默认工具集。

工具集（对应简历「专业工具调用」）：
- search_knowledge   金融知识检索（RAG）
- screen_stocks      自然语言选股（因子池排序）
- analyze_stock      个股分析（行情 + 技术指标 + 因子评分）
- get_price_history  区间行情走势
- run_backtest       策略回测（年化/回撤/胜率等指标）
- financial_calc     金融计算器（复利/定投/年化/回撤/仓位）
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT_RISK_PARAMS, DEFAULT_TRADE_PARAMS, TRADING_DAYS_PER_YEAR
from ..quant.data import MarketData
from ..quant import factors as F
from ..quant import indicators as IND
from ..quant.backtest import Backtester
from ..quant.portfolio import atr_risk_position_size

__all__ = ["ToolResult", "ToolRegistry", "build_default_tools"]


@dataclass
class ToolResult:
    """工具执行结果。"""

    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    summary: str = ""
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool, "ok": self.ok, "summary": self.summary,
            "error": self.error, "elapsed_ms": self.elapsed_ms,
        }


class ToolRegistry:
    """工具注册表：统一注册、调用、异常兜底。"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, func, description: str, params: str = "") -> None:
        self._tools[name] = {"func": func, "description": description, "params": params}

    def call(self, name: str, **kwargs) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(tool=name, ok=False, error=f"工具不存在：{name}")
        t0 = time.time()
        try:
            data = entry["func"](**kwargs)
            elapsed = int((time.time() - t0) * 1000)
            summary = str(data.get("summary", "完成")) if isinstance(data, dict) else "完成"
            return ToolResult(tool=name, ok=True, data=data or {},
                              summary=summary, elapsed_ms=elapsed)
        except Exception as err:  # noqa: BLE001 - 工具异常不中断 Agent 主链路
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(tool=name, ok=False, error=str(err), elapsed_ms=elapsed)

    def describe(self) -> list[dict]:
        return [
            {"name": name, "description": e["description"], "params": e["params"]}
            for name, e in self._tools.items()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------- #
# 默认工具集
# ---------------------------------------------------------------------- #
def build_default_tools(
    market: MarketData,
    retriever=None,
    backtester: Backtester | None = None,
) -> ToolRegistry:
    """装配默认工具集。"""
    reg = ToolRegistry()

    # ---- 知识检索（RAG）----
    if retriever is not None:
        def search_knowledge(query: str, top_k: int = 3):
            res = retriever.retrieve(query, top_k=top_k)
            chunks = [c.to_dict() for c in res.chunks]
            return {
                "query": query,
                "chunks": chunks,
                "citations": res.citations(),
                "context_text": res.context_text(),
                "summary": f"检索到 {len(chunks)} 条相关知识"
                            + (f"（{chunks[0]['title']} 等）" if chunks else ""),
            }
        reg.register("search_knowledge", search_knowledge,
                     "检索金融知识库（理财/概念/指标/策略/风控）", "query: 检索词; top_k: 条数")

    # ---- 个股分析 ----
    def analyze_stock(symbol: str):
        if symbol not in market:
            raise KeyError(f"股票 {symbol} 不在样本池，可用：{', '.join(market.symbols())}")
        df = market.get(symbol)
        ind = IND.add_all_indicators(df)
        fac = F.build_factors(df)
        latest = ind.iloc[-1]
        fl = fac.iloc[-1]
        as_of = str(ind.index[-1].date())

        # 52 周高低
        year_df = ind.tail(250)
        high_52w, low_52w = float(year_df["high"].max()), float(year_df["low"].min())
        pos_52w = ((latest["close"] - low_52w) / (high_52w - low_52w)
                   if high_52w > low_52w else 0.5)

        # 均线排列状态
        ma5, ma20, ma60 = float(latest["ma5"]), float(latest["ma20"]), float(latest["ma60"])
        if ma5 > ma20 > ma60:
            ma_state = "多头排列（MA5>MA20>MA60，趋势向上）"
        elif ma5 < ma20 < ma60:
            ma_state = "空头排列（MA5<MA20<MA60，趋势向下）"
        else:
            ma_state = "均线纠缠（方向不明）"

        # MACD 状态
        hist = float(latest["macd_hist"])
        macd_state = "多头（DIF>DEA，红柱）" if hist > 0 else "空头（DIF<DEA，绿柱）"

        # KDJ 状态
        k, d, j = float(latest["k"]), float(latest["d"]), float(latest["j"])
        kdj_state = ("超买区（J>100）" if j > 100 else
                     "超卖区（J<0）" if j < 0 else "中性区间")

        # 因子排名（样本池内）
        ranks = _factor_ranks(market)
        rank_info = ranks.get(symbol, {})

        risk_notes = []
        rsi14 = float(latest["rsi14"])
        if rsi14 > 80:
            risk_notes.append(f"RSI(14)={rsi14:.0f} 进入超买区，短期回调风险积聚")
        elif rsi14 < 20:
            risk_notes.append(f"RSI(14)={rsi14:.0f} 进入超卖区，关注反弹机会")
        if pos_52w > 0.9:
            risk_notes.append("股价接近 52 周高点，追高风险较大")
        elif pos_52w < 0.1:
            risk_notes.append("股价接近 52 周低点，需警惕趋势性下跌")
        vol20 = float(latest["vol20"])
        if vol20 > 0.4:
            risk_notes.append(f"年化波动率 {vol20:.0%} 偏高，注意仓位控制")

        return {
            "symbol": symbol,
            "name": market.name_of(symbol),
            "as_of": as_of,
            "close": round(float(latest["close"]), 2),
            "pct_change_1d": round(float(latest["ret_1d"]) * 100, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "position_52w_pct": round(pos_52w * 100, 1),
            "ma": {"ma5": round(ma5, 2), "ma20": round(ma20, 2),
                   "ma60": round(ma60, 2), "state": ma_state},
            "macd": {"dif": round(float(latest["dif"]), 3),
                     "dea": round(float(latest["dea"]), 3),
                     "hist": round(hist, 3), "state": macd_state},
            "rsi14": round(rsi14, 1),
            "kdj": {"k": round(k, 1), "d": round(d, 1), "j": round(j, 1),
                    "state": kdj_state},
            "boll": {"mid": round(float(latest["boll_mid"]), 2),
                     "up": round(float(latest["boll_up"]), 2),
                     "dn": round(float(latest["boll_dn"]), 2)},
            "atr14": round(float(latest["atr14"]), 2),
            "atr_pct": round(float(latest["atr14"]) / float(latest["close"]) * 100, 2),
            "mom20": round(float(latest["mom20"]) * 100, 2),
            "mom60": round(float(latest["mom60"]) * 100, 2),
            "vol20": round(vol20 * 100, 1),
            "factors": {k: round(float(fl[k]), 3) for k in
                        ["trend", "momentum", "rsi", "volume", "composite"]},
            "factor_rank": rank_info,
            "risk_notes": risk_notes,
            "summary": f"{market.name_of(symbol)}（{symbol}）截至 {as_of}："
                       f"收盘 {float(latest['close']):.2f}，{ma_state}",
        }
    reg.register("analyze_stock", analyze_stock,
                 "个股分析：行情概览 + 技术指标（MA/MACD/RSI/KDJ/BOLL/ATR）+ 量化因子评分 + 风险提示",
                 "symbol: 6 位股票代码")

    # ---- 自然语言选股 ----
    def screen_stocks(conditions: list[dict], top_n: int = 5):
        if not conditions:
            conditions = [{"factor": "composite", "dir": "desc", "label": "综合评分高"}]
        latest_date = market.latest_date()
        rows = []
        for sym in market.symbols():
            df = market.get(sym)
            fac = F.build_factors(df).loc[df.index <= latest_date] if latest_date else F.build_factors(df)
            fl = fac.iloc[-1]
            feats = F.build_features(df).loc[fac.index[-1]]
            rows.append({
                "symbol": sym,
                "name": market.name_of(sym),
                "close": round(float(df["close"].iloc[-1]), 2),
                "trend": float(fl["trend"]),
                "momentum": float(fl["momentum"]),
                "rsi": float(fl["rsi"]),
                "volume": float(fl["volume"]),
                "composite": float(fl["composite"]),
                "ret_20d": round(float(feats["ret_20d"]) * 100, 2),
                "vol20": round(float(feats["vol20"]) * 100, 1),
                "vol_ratio": round(float(feats["vol_ratio"]), 2),
                "drawdown_60d": round(float(feats["drawdown_60d"]) * 100, 1),
            })
        frame = pd.DataFrame(rows)

        # 多条件打分：每个条件取池内百分位（方向调整），取均值
        scores = []
        for _, row in frame.iterrows():
            parts = []
            for cond in conditions:
                factor = cond.get("factor", "composite")
                direction = cond.get("dir", "desc")
                if factor not in frame.columns:
                    continue
                col = frame[factor]
                pct = col.rank(pct=True).loc[row.name] if len(col) > 1 else 0.5
                parts.append(pct if direction == "desc" else 1.0 - pct)
            scores.append(sum(parts) / len(parts) if parts else 0.0)
        frame["score"] = scores
        frame = frame.sort_values("score", ascending=False).head(top_n)
        frame.insert(0, "rank", range(1, len(frame) + 1))

        cond_desc = "、".join(c.get("label", c.get("factor", "")) for c in conditions)
        return {
            "as_of": str(latest_date.date()) if latest_date is not None else "",
            "conditions": conditions,
            "conditions_desc": cond_desc,
            "rows": frame.to_dict(orient="records"),
            "summary": f"按「{cond_desc}」筛选出 {len(frame)} 只标的（截至 "
                       f"{frame['as_of'].iloc[0] if 'as_of' in frame.columns else ''}）"
                       if False else f"按「{cond_desc}」筛选出 {len(frame)} 只标的",
        }
    reg.register("screen_stocks", screen_stocks,
                 "自然语言选股：在样本池上按多条件（趋势/动量/波动/量能/超跌）打分排序",
                 "conditions: 条件列表; top_n: 返回数量")

    # ---- 区间行情 ----
    def get_price_history(symbol: str, days: int = 60):
        if symbol not in market:
            raise KeyError(f"股票 {symbol} 不在样本池")
        df = market.get(symbol).tail(days)
        first, last = df["close"].iloc[0], df["close"].iloc[-1]
        chg = (last / first - 1) * 100
        series = [
            {"date": str(d.date()), "close": round(float(c), 2)}
            for d, c in df["close"].items()
        ]
        return {
            "symbol": symbol, "name": market.name_of(symbol),
            "days": len(df), "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "start_close": round(float(first), 2), "end_close": round(float(last), 2),
            "change_pct": round(float(chg), 2),
            "series": series,
            "summary": f"{market.name_of(symbol)} 近 {len(df)} 个交易日涨跌 {chg:+.2f}%",
        }
    reg.register("get_price_history", get_price_history,
                 "区间行情走势（近 N 日收盘价序列与区间涨跌幅）",
                 "symbol: 代码; days: 天数")

    # ---- 策略回测 ----
    if backtester is not None:
        def run_backtest(symbols: list[str] | None = None,
                         start: str | None = None, end: str | None = None):
            result = backtester.run(symbols=symbols, start=start, end=end)
            m = result.metrics
            return {
                "date_range": list(result.date_range),
                "metrics": _json_safe(m),
                "strategy_desc": result.strategy_desc,
                "trade_count": m.get("trade_count", 0),
                "summary": (f"回测完成（{result.date_range[0]} ~ {result.date_range[1]}）："
                            f"年化 {m.get('annualized_return', 0):+.2%}，"
                            f"最大回撤 {m.get('max_drawdown', 0):.2%}，"
                            f"夏普 {m.get('sharpe', 0):.2f}，胜率 {m.get('win_rate', 0):.1%}"),
            }
        reg.register("run_backtest", run_backtest,
                     "运行多因子趋势策略回测，返回年化收益/最大回撤/夏普/胜率等指标",
                     "symbols: 股票池; start/end: 区间")

    # ---- 金融计算器 ----
    def financial_calc(kind: str, **kw):
        if kind == "compound":          # 复利终值
            p, r, years = float(kw["principal"]), float(kw["annual_rate"]), float(kw["years"])
            fv = p * (1 + r) ** years
            return {"kind": kind, "principal": p, "annual_rate": r, "years": years,
                    "future_value": round(fv, 2),
                    "gain": round(fv - p, 2),
                    "summary": f"本金 {p:.0f} 元按年化 {r:.0%} 复利 {years:.0f} 年，终值约 {fv:.0f} 元"}
        if kind == "dca":               # 定投终值（期初投入）
            monthly = float(kw["monthly"])
            r = float(kw.get("annual_rate", 0.08)) / 12
            years = float(kw.get("years", 10))
            n = int(years * 12)
            if r == 0:
                fv = monthly * n
            else:
                fv = monthly * ((1 + r) ** n - 1) / r * (1 + r)
            invested = monthly * n
            return {"kind": kind, "monthly": monthly, "annual_rate": r * 12,
                    "years": years, "invested": round(invested, 2),
                    "future_value": round(fv, 2),
                    "gain": round(fv - invested, 2),
                    "summary": f"每月定投 {monthly:.0f} 元、{years:.0f} 年、假设年化 {r*12:.0%}，"
                               f"累计投入 {invested:.0f} 元，期末市值约 {fv:.0f} 元（收益 {fv-invested:.0f} 元）"}
        if kind == "annualize":         # 年化
            total = float(kw["total_return"])
            years = float(kw.get("years", 1))
            ann = (1 + total) ** (1 / years) - 1
            return {"kind": kind, "total_return": total, "years": years,
                    "annualized": round(ann, 4),
                    "summary": f"总收益 {total:.2%}（{years:.0f} 年）对应年化 {ann:.2%}"}
        if kind == "drawdown":          # 回撤
            peak, trough = float(kw["peak"]), float(kw["trough"])
            dd = trough / peak - 1
            recover = (peak / trough - 1) if dd < 0 else 0
            return {"kind": kind, "peak": peak, "trough": trough,
                    "drawdown": round(dd, 4),
                    "recovery_needed": round(recover, 4),
                    "summary": f"从 {peak:.2f} 跌至 {trough:.2f}，回撤 {dd:.2%}，"
                               f"需上涨 {recover:.2%} 回本"}
        if kind == "position_size":     # ATR 仓位
            equity = float(kw["equity"])
            price = float(kw["price"])
            atr_val = float(kw["atr"])
            shares = atr_risk_position_size(equity, price, atr_val)
            stop_dist = atr_val * DEFAULT_RISK_PARAMS.atr_mult
            risk_amount = equity * DEFAULT_RISK_PARAMS.risk_per_trade
            return {"kind": kind, "equity": equity, "price": price, "atr": atr_val,
                    "shares": shares, "position_value": round(shares * price, 2),
                    "stop_distance": round(stop_dist, 2),
                    "risk_amount": round(risk_amount, 2),
                    "summary": f"账户 {equity:.0f} 元、价格 {price:.2f}、ATR {atr_val:.2f}："
                               f"可买 {shares} 股（市值 {shares*price:.0f} 元），"
                               f"止损距离 {stop_dist:.2f} 元，单笔风险 {risk_amount:.0f} 元"}
        raise ValueError(f"不支持的计算类型：{kind}")
    reg.register("financial_calc", financial_calc,
                 "金融计算器：复利终值/定投终值/年化换算/回撤测算/ATR仓位",
                 "kind: compound|dca|annualize|drawdown|position_size")

    return reg


# ---------------------------------------------------------------------- #
# 内部辅助
# ---------------------------------------------------------------------- #
def _factor_ranks(market: MarketData) -> dict[str, dict]:
    """样本池内各股因子综合分排名（1 = 最强）。"""
    scores = []
    for sym in market.symbols():
        df = market.get(sym)
        fl = F.build_factors(df).iloc[-1]
        scores.append((sym, float(fl["composite"])))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranks = {}
    n = len(scores)
    for i, (sym, sc) in enumerate(scores):
        ranks[sym] = {"composite_rank": i + 1, "composite_score": round(sc, 3),
                      "pool_size": n}
    return ranks


def _json_safe(obj):
    """将 numpy/inf/NaN 转换为可 JSON 序列化的值。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
