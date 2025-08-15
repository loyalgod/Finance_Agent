"""腾讯财经行情接口客户端（仅依赖 requests）。

数据源（公开行情接口）：
- 日K线：https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
  param 格式：{市场}{代码},day,{起始},{截止},{条数},{复权}
  返回 data[{key}]['qfqday'|'day'] = [[date, open, close, high, low, volume], ...]
  股票名称取自 data[{key}]['qt'][{key}][1]

注意：接口单次 count 上限约 640 条，长区间需按年分块抓取后拼接。
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests

KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

# 常见指数 → 腾讯市场前缀（指数与股票规则不同，单独维护）
INDEX_MARKET = {
    "000300": "sh",  # 沪深300
    "000001": "sh",  # 上证指数
    "000905": "sh",  # 中证500
    "000016": "sh",  # 上证50
    "399001": "sz",  # 深证成指
    "399006": "sz",  # 创业板指
}

KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume"]


class TencentQuoteError(RuntimeError):
    """腾讯行情接口异常。"""


def market_of(symbol: str) -> str:
    """代码 → 腾讯市场前缀。沪市 6/9 开头为 sh，深市其余为 sz（指数按表）。"""
    if symbol in INDEX_MARKET:
        return INDEX_MARKET[symbol]
    return "sh" if symbol.startswith(("6", "9")) else "sz"


def _get_json(params: dict, retries: int = 3, timeout: int = 20) -> dict:
    last_err: Exception | None = None
    with requests.Session() as session:
        for attempt in range(retries):
            try:
                resp = session.get(
                    KLINE_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as err:  # noqa: BLE001 - 统一转成 TencentQuoteError
                last_err = err
                time.sleep(1.5 * (attempt + 1))
    raise TencentQuoteError(f"请求失败 {KLINE_URL}: {last_err}")


def _fetch_chunk(mkt_code: str, beg: str, end: str, count: int, fq: str) -> tuple[list, str]:
    """抓取单个时间块，返回 (kline 列表, 名称)。"""
    payload = _get_json({"param": f"{mkt_code},day,{beg},{end},{count},{fq}"})
    data = payload.get("data") or {}
    # 返回键可能是带市场前缀的代码（如 sh600519）
    key = next((k for k in data if k not in ("qt", "mx_price", "prec", "version")), None)
    if key is None:
        raise TencentQuoteError(f"{mkt_code} 无行情数据返回")
    node = data[key]
    days = node.get("qfqday") or node.get("day") or []
    name = ""
    qt = node.get("qt") or {}
    for qk, qv in qt.items():
        if isinstance(qv, list) and len(qv) > 1:
            name = str(qv[1])
            break
    return days, name


def fetch_daily(
    symbol: str,
    beg: str = "20200101",
    end: str = "20500101",
    fq: str = "qfq",
) -> pd.DataFrame:
    """抓取单标的日K线（qfq 前复权）。

    按自然年分块抓取后拼接去重（单次 count 上限约 640）。
    返回 DataFrame，列：date, open, close, high, low, volume, symbol, name。
    """
    mkt_code = f"{market_of(symbol)}{symbol}"
    beg_d = datetime.strptime(beg, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()

    frames, name = [], ""
    year = beg_d.year
    while year <= end_d.year:
        chunk_beg = max(beg_d, datetime(year, 1, 1).date())
        chunk_end = min(end_d, datetime(year, 12, 31).date())
        days, chunk_name = _fetch_chunk(
            mkt_code, chunk_beg.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), 640, fq
        )
        # 含除权除息的区间行会多带一个复权因子字段，统一截取前 6 列
        rows = [row[: len(KLINE_COLUMNS)] for row in days if len(row) >= len(KLINE_COLUMNS)]
        if rows:
            frames.append(pd.DataFrame(rows, columns=KLINE_COLUMNS))
        if chunk_name:
            name = chunk_name
        year += 1

    if not frames:
        raise TencentQuoteError(f"{symbol} 在 {beg}~{end} 区间无行情数据")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    df["name"] = name
    return df
