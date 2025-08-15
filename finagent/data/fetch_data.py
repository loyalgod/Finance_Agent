"""行情数据抓取脚本：拉取样本股票池与基准指数日K线，落盘 data/raw/。

多源策略：腾讯财经接口为主，东方财富接口为备（东财 CDN 对 TLS 指纹有限流）。

用法：
    python -m finagent.data.fetch_data                                  # 默认股票池，2023 年至今
    python -m finagent.data.fetch_data --universe 600519,300750 --start 20230101
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..config import (
    BENCHMARK_SYMBOL,
    DEFAULT_UNIVERSE,
    FALLBACK_STOCK_NAMES,
    RAW_DATA_DIR,
    ensure_dirs,
)
from . import eastmoney, tencent

# 输出统一列
OUTPUT_COLUMNS = ["date", "open", "close", "high", "low", "volume", "symbol", "name"]

INDEX_NAMES = {
    "000300": "沪深300",
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000905": "中证500",
    "000016": "上证50",
}


def _fetch_with_fallback(symbol: str, start: str) -> pd.DataFrame:
    """腾讯 → 东财 依次尝试，任一成功即返回统一列 DataFrame。"""
    errors = []
    for source_name, fetcher in (
        ("腾讯财经", tencent.fetch_daily),
        ("东方财富", eastmoney.fetch_daily),
    ):
        try:
            df = fetcher(symbol, beg=start)
            if df.empty:
                raise RuntimeError("空数据")
            # 统一列与顺序（东财返回列更多，只保留 OHLCV 核心）
            for col in OUTPUT_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[OUTPUT_COLUMNS].copy()
            if not df["name"].astype(str).str.len().gt(0).any():
                df["name"] = FALLBACK_STOCK_NAMES.get(symbol, INDEX_NAMES.get(symbol, ""))
            return df
        except Exception as err:  # noqa: BLE001 - 两源都失败才抛出
            errors.append(f"{source_name}: {err}")
    raise RuntimeError("；".join(errors))


def _save(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if "date" in df.columns and not df.empty:
        span = f"（{df['date'].min().date()} ~ {df['date'].max().date()}）"
    else:
        span = ""
    print(f"  已保存 {path.name}：{len(df)} 行{span}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="抓取 A 股样本池行情数据（腾讯主源/东财备源）")
    parser.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE),
                        help="逗号分隔的股票代码列表")
    parser.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    args = parser.parse_args(argv)

    ensure_dirs()
    universe = [s.strip() for s in args.universe.split(",") if s.strip()]
    ok, failed = 0, []

    for symbol in universe + [BENCHMARK_SYMBOL]:
        is_bench = symbol == BENCHMARK_SYMBOL
        out_name = f"benchmark_{BENCHMARK_SYMBOL}.csv" if is_bench else f"{symbol}.csv"
        try:
            df = _fetch_with_fallback(symbol, args.start)
            _save(df, RAW_DATA_DIR / out_name)
            ok += 1
        except Exception as err:  # noqa: BLE001
            tag = "基准" if is_bench else "股票"
            print(f"  [错误] {tag} {symbol} 抓取失败：{err}")
            failed.append(symbol)

    print(f"\n完成：成功 {ok} 项，失败 {len(failed)} 项"
          + (f"（{', '.join(failed)}）" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
