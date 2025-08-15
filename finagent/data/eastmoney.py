"""东方财富行情接口客户端（依赖 requests）。

数据源（公开行情接口）：
- 日K线：https://push2his.eastmoney.com/api/qt/stock/kline/get
- 全市场股票列表：https://push2.eastmoney.com/api/qt/clist/get

说明：东方财富 CDN 对 TLS 指纹有校验，标准库 urllib 会被直接断连，
故使用 requests（OpenSSL 指纹与浏览器一致）。
"""
from __future__ import annotations

import time

import pandas as pd
import requests

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

# 常见指数 secid 映射（指数与股票的市场归属规则不同，单独维护）
INDEX_SECID = {
    "000300": "1.000300",  # 沪深300
    "000001": "1.000001",  # 上证指数
    "399001": "0.399001",  # 深证成指
    "399006": "0.399006",  # 创业板指
    "000905": "1.000905",  # 中证500
    "000016": "1.000016",  # 上证50
}

# push2his kline 字段顺序（fields2 的 f51..f61）
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume", "amount",
    "amplitude", "pct_change", "change", "turnover",
]

NUMERIC_KLINE_COLUMNS = [
    "open", "close", "high", "low", "volume", "amount",
    "amplitude", "pct_change", "change", "turnover",
]


class EastMoneyError(RuntimeError):
    """行情接口异常。"""


def _get_json(url: str, params: dict, retries: int = 3, timeout: int = 15) -> dict:
    """带重试的 JSON GET。"""
    last_err: Exception | None = None
    with requests.Session() as session:
        for attempt in range(retries):
            try:
                resp = session.get(
                    url, params=params, headers=DEFAULT_HEADERS, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as err:  # noqa: BLE001 - 统一转成 EastMoneyError
                last_err = err
                time.sleep(1.5 * (attempt + 1))
    raise EastMoneyError(f"请求失败 {url}: {last_err}")


def to_secid(symbol: str) -> str:
    """股票/指数代码 → 东方财富 secid。沪市（6/9 开头）与指数为 1.，深市为 0.。"""
    if symbol in INDEX_SECID:
        return INDEX_SECID[symbol]
    return f"1.{symbol}" if symbol.startswith(("6", "9")) else f"0.{symbol}"


def fetch_daily(
    symbol: str,
    beg: str = "20200101",
    end: str = "20500101",
    fqt: int = 1,
) -> pd.DataFrame:
    """抓取单标的日K线数据（fqt=1 为前复权）。

    返回 DataFrame，列：date, open, close, high, low, volume, amount,
    amplitude, pct_change, change, turnover, symbol, name。按日期升序。
    """
    payload = _get_json(
        KLINE_URL,
        {
            "secid": to_secid(symbol),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",       # 日线
            "fqt": str(fqt),    # 1=前复权
            "beg": beg,
            "end": end,
            "lmt": "100000",
        },
    )
    data = payload.get("data")
    if not data or not data.get("klines"):
        raise EastMoneyError(f"{symbol} 无行情数据返回（secid={to_secid(symbol)}）")
    rows = [line.split(",") for line in data["klines"]]
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["symbol"] = symbol
    df["name"] = str(data.get("name", ""))
    df["date"] = pd.to_datetime(df["date"])
    df[NUMERIC_KLINE_COLUMNS] = df[NUMERIC_KLINE_COLUMNS].astype(float)
    df["volume"] = df["volume"].astype("int64")
    return df.sort_values("date").reset_index(drop=True)


def fetch_stock_list() -> pd.DataFrame:
    """抓取沪深 A 股全市场列表，返回列：symbol, name。"""
    payload = _get_json(
        LIST_URL,
        {
            "pn": 1,
            "pz": 60000,
            "po": 0,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f12",
            # 深主板 + 创业板 + 沪主板 + 科创板
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
            "fields": "f12,f14",
        },
    )
    diff = payload.get("data", {}).get("diff") or []
    records = [
        {"symbol": str(item.get("f12", "")).zfill(6), "name": str(item.get("f14", ""))}
        for item in diff
        if item.get("f12")
    ]
    return pd.DataFrame(records, columns=["symbol", "name"])
