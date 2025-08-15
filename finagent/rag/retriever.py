"""RAG 检索封装：对知识库检索结果做格式化，供 Agent 生成答案使用。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import KNOWLEDGE_DIR
from .bm25 import tokenize
from .knowledge_base import KnowledgeBase, KnowledgeChunk

__all__ = ["RetrievalResult", "Retriever"]


@dataclass
class RetrievalResult:
    """一次检索的结果。"""

    query: str
    chunks: list[KnowledgeChunk]

    @property
    def hit(self) -> bool:
        return bool(self.chunks)

    def context_text(self, max_chunks: int = 3) -> str:
        """拼装供 LLM 使用的上下文字符串。"""
        if not self.chunks:
            return "（知识库未检索到相关内容）"
        parts = []
        for c in self.chunks[:max_chunks]:
            parts.append(f"【{c.cite()}】\n{c.text}")
        return "\n\n".join(parts)

    def citations(self) -> list[str]:
        """溯源引用列表。"""
        return [c.cite() for c in self.chunks]


class Retriever:
    """知识检索器（RAG 的 R）。

    双信号相关性过滤（解决 BM25 绝对分数不可比的问题）：
    - 信号 1（分数归一化）：原始分数 ÷ 查询字符数 ≥ min_score。
      长查询（如 17 字无关长句）会靠常用汉字（的/在/问题）累积原始分数，
      按字符数归一后可区分真命中与噪音；
    - 信号 2（标题命中）：查询的 2 字以上 token 出现在条目标题中。
      概念词（止损/夏普/最大回撤）即使查询很长也能可靠命中。
    """

    def __init__(self, knowledge_dir=KNOWLEDGE_DIR):
        self.kb = KnowledgeBase(knowledge_dir)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 1.5) -> RetrievalResult:
        """检索相关知识条目（双信号过滤后取前 top_k）。"""
        raw = self.kb.search(query, top_k=top_k * 3, min_score=0.0)
        if not raw:
            return RetrievalResult(query=query, chunks=[])
        q_len = max(1, len(re.sub(r"\s+", "", query)))
        accepted: list[KnowledgeChunk] = []
        for c in raw:
            if len(accepted) >= top_k:
                break
            norm = c.score / q_len
            title_hit = any(len(t) >= 2 and t in c.title for t in tokenize(query))
            if norm >= min_score or title_hit:
                accepted.append(c)
        return RetrievalResult(query=query, chunks=accepted)

    def reload(self) -> None:
        self.kb.reload()

    def __len__(self) -> int:
        return len(self.kb)
