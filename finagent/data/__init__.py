"""行情数据模块：多源接口客户端与数据抓取脚本。

主源：腾讯财经 ifzq（沙箱内可达，按年分块抓取）
备源：东方财富 push2his（CDN 对 urllib 指纹有限流，requests 可过）
"""
from . import eastmoney, tencent
from .eastmoney import EastMoneyError, fetch_daily as eastmoney_fetch_daily
from .tencent import TencentQuoteError, fetch_daily as tencent_fetch_daily

__all__ = [
    "eastmoney",
    "tencent",
    "EastMoneyError",
    "TencentQuoteError",
    "eastmoney_fetch_daily",
    "tencent_fetch_daily",
]
