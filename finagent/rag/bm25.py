"""纯 Python BM25 检索器（中文 n-gram 分词，零第三方依赖）。

分词策略：
- 连续中文 → 生成单字 + 二字组（bigram）token，兼顾召回与精度
- 连续英文/数字 → 小写词 token（含点号，支持 600519、PE、MACD 等混合）
"""
from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["tokenize", "BM25Index"]

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.]*")


def tokenize(text: str) -> list[str]:
    """中英文混合分词：中文单字+bigram，英文数字整词。"""
    tokens: list[str] = []
    for seg in _CJK_RUN.findall(text):
        tokens.extend(seg)                       # 单字
        tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))  # 二字组
    tokens.extend(m.group(0).lower() for m in _LATIN_RUN.finditer(text))
    return tokens


class BM25Index:
    """BM25 倒排索引（k1=1.5, b=0.75）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: Counter[str] = Counter()
        self._tf: list[Counter[str]] = []
        self._n = 0

    # ------------------------------------------------------------------ #
    def fit(self, docs: list[str]) -> "BM25Index":
        self._docs = []
        self._df = Counter()
        self._tf = []
        for doc in docs:
            tokens = tokenize(doc)
            tf = Counter(tokens)
            self._docs.append(tokens)
            self._doc_len.append(len(tokens))
            self._tf.append(tf)
            self._df.update(tf.keys())
        self._n = len(self._docs)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0
        return self

    # ------------------------------------------------------------------ #
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson 平滑 idf，保证非负
        return max(math.log(1.0 + (self._n - df + 0.5) / (df + 0.5)), 1e-6)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(doc_index, score)]，按分数降序。"""
        if not self._n:
            return []
        q_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for idx in range(self._n):
            score = 0.0
            dl = self._doc_len[idx]
            tf = self._tf[idx]
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * f * (self.k1 + 1.0) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
