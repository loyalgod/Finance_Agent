"""金融知识库：加载 knowledge/*.md，按 ## 标题切分为条目，建立 BM25 索引。

条目结构：每个 ## 标题为一个知识条目，含来源文件与类别，支持检索溯源。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import KNOWLEDGE_DIR
from .bm25 import BM25Index

__all__ = ["KnowledgeChunk", "KnowledgeBase"]


@dataclass
class KnowledgeChunk:
    """知识条目（检索返回单元）。"""

    chunk_id: int
    source: str        # 来源文件名（如 量化策略.md）
    category: str      # 类别（文件名去后缀）
    title: str         # 条目标题
    text: str          # 正文（不含标题行）
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 4),
        }

    def cite(self) -> str:
        """溯源引用串。"""
        return f"《{self.category}》{self.title}"


class KnowledgeBase:
    """金融知识库（文件扫描 + 条目切分 + BM25 索引）。"""

    def __init__(self, knowledge_dir: Path | str = KNOWLEDGE_DIR):
        self.knowledge_dir = Path(knowledge_dir)
        self.chunks: list[KnowledgeChunk] = []
        self._index: BM25Index | None = None
        self.reload()

    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """重新扫描知识目录并重建索引。"""
        self.chunks = []
        if not self.knowledge_dir.exists():
            return
        for path in sorted(self.knowledge_dir.glob("*.md")):
            self._load_file(path)
        self._index = BM25Index()
        self._index.fit([f"{c.title}\n{c.text}" for c in self.chunks])

    def _load_file(self, path: Path) -> None:
        category = path.stem
        text = path.read_text(encoding="utf-8")
        sections = text.split("\n## ")
        for section in sections[1:]:  # 跳过文件首段（大标题与说明）
            lines = section.strip().splitlines()
            if not lines:
                continue
            title = lines[0].strip().lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            self.chunks.append(KnowledgeChunk(
                chunk_id=len(self.chunks),
                source=path.name,
                category=category,
                title=title,
                text=body,
            ))

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[KnowledgeChunk]:
        """BM25 检索，返回带分数的条目副本（不修改库内对象）。"""
        if not self.chunks or self._index is None:
            return []
        hits = self._index.search(query, top_k=top_k)
        results = []
        for idx, score in hits:
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            results.append(KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                category=chunk.category,
                title=chunk.title,
                text=chunk.text,
                score=score,
            ))
        return results

    def __len__(self) -> int:
        return len(self.chunks)

    def categories(self) -> list[str]:
        return sorted({c.category for c in self.chunks})
