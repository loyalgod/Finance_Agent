"""行情数据处理：加载、清洗、对齐与缓存。

输入 data/raw/*.csv（腾讯/东财统一格式：date,open,close,high,low,volume,symbol,name）
输出以日期为索引的 OHLCV DataFrame，供指标/因子/回测消费。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import (
    BENCHMARK_SYMBOL,
    FALLBACK_STOCK_NAMES,
    RAW_DATA_DIR,
)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketData:
    """样本股票池行情数据访问层。

    - 加载 data/raw/ 下全部个股 CSV 与基准指数 CSV
    - 清洗：数值类型转换、日期去重、升序排列、缺失剔除
    - 提供 get / benchmark / common_dates / name_of / latest_date
    """

    def __init__(self, raw_dir: Path | str = RAW_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self._frames: dict[str, pd.DataFrame] = {}
        self._names: dict[str, str] = {}
        self._bench: pd.DataFrame | None = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.raw_dir.exists():
            return
        for path in sorted(self.raw_dir.glob("*.csv")):
            if path.stem.startswith(("stock_list", "benchmark_")):
                continue
            df, name = self._read_csv(path)
            if df.empty:
                continue
            symbol = path.stem
            self._frames[symbol] = df
            self._names[symbol] = name or FALLBACK_STOCK_NAMES.get(symbol, symbol)

        bench_path = self.raw_dir / f"benchmark_{BENCHMARK_SYMBOL}.csv"
        if bench_path.exists():
            df, name = self._read_csv(bench_path)
            if not df.empty:
                self._bench = df
                self._names[BENCHMARK_SYMBOL] = name or FALLBACK_STOCK_NAMES.get(BENCHMARK_SYMBOL, BENCHMARK_SYMBOL)

    @staticmethod
    def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
        """读取单个 CSV，返回 (OHLCV DataFrame, 名称)。"""
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty or "date" not in df.columns:
            return pd.DataFrame(columns=OHLCV_COLUMNS), ""
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset="date").sort_values("date")
        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=OHLCV_COLUMNS)

        name = ""
        if "name" in df.columns:
            names = df["name"].dropna().astype(str)
            names = names[names.str.len() > 0]
            if not names.empty:
                name = str(names.iloc[0])

        df = df.set_index("date")
        return df[OHLCV_COLUMNS].copy(), name

    # ------------------------------------------------------------------ #
    def symbols(self) -> list[str]:
        """样本池股票代码列表（不含基准）。"""
        return sorted(self._frames.keys())

    def name_of(self, symbol: str) -> str:
        return self._names.get(symbol, symbol)

    def get(self, symbol: str) -> pd.DataFrame:
        """取单只股票 OHLCV（日期索引）。不存在则抛 KeyError。"""
        if symbol not in self._frames:
            raise KeyError(f"股票 {symbol} 不在样本池中，可用：{', '.join(self.symbols())}")
        return self._frames[symbol].copy()

    def benchmark(self) -> pd.DataFrame | None:
        """基准指数 OHLCV（日期索引），未加载时为 None。"""
        return self._bench.copy() if self._bench is not None else None

    def common_dates(self, symbols: list[str] | None = None) -> pd.DatetimeIndex:
        """多标的交易日交集（用于组合回测对齐）。"""
        symbols = symbols or self.symbols()
        if not symbols:
            return pd.DatetimeIndex([])
        idx = None
        for sym in symbols:
            cur = self._frames[sym].index
            idx = cur if idx is None else idx.intersection(cur)
        return idx if idx is not None else pd.DatetimeIndex([])

    def latest_date(self, symbols: list[str] | None = None) -> pd.Timestamp | None:
        idx = self.common_dates(symbols)
        return idx[-1] if len(idx) else None

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._frames
