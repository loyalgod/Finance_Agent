"""多轮上下文管理：会话存储、实体承接、历史窗口。

支撑两类多轮能力：
- 实体承接：「那招商银行呢」继承上一轮股票上下文
- 意图延续：「它的胜率呢」延续上一轮的策略解释上下文
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Turn", "ConversationContext"]


@dataclass
class Turn:
    """一轮对话记录。"""

    question: str
    intent: str
    rewritten: str
    answer: str
    stocks: list[str] = field(default_factory=list)   # 本轮涉及股票代码
    summary: str = ""                                  # 轨迹摘要


class ConversationContext:
    """会话上下文（滑动窗口 + 实体记忆）。"""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._sessions: dict[str, list[Turn]] = {}

    # ------------------------------------------------------------------ #
    def add_turn(self, session_id: str, turn: Turn) -> None:
        turns = self._sessions.setdefault(session_id, [])
        turns.append(turn)
        if len(turns) > self.max_turns:
            del turns[: len(turns) - self.max_turns]

    def history(self, session_id: str) -> list[Turn]:
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ #
    def last_stocks(self, session_id: str) -> list[str]:
        """最近一轮涉及的股票（实体承接用）。"""
        for turn in reversed(self.history(session_id)):
            if turn.stocks:
                return list(turn.stocks)
        return []

    def last_intent(self, session_id: str) -> str | None:
        turns = self.history(session_id)
        return turns[-1].intent if turns else None

    # ------------------------------------------------------------------ #
    def history_text(self, session_id: str, max_chars: int = 1500) -> str:
        """拼装历史对话文本（供 LLM 提示词）。"""
        turns = self.history(session_id)
        if not turns:
            return "（首轮对话）"
        parts = [f"用户：{t.question}\n助手：{t.answer[:200]}" for t in turns]
        text = "\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text
