"""FinAgent —— 金融投研与量化分析智能体。

模块划分：
- finagent.data    行情数据获取与处理（东方财富公开接口）
- finagent.quant   行情处理、指标因子、策略信号、仓位与风控、回测引擎
- finagent.rag     金融知识库与 BM25 检索
- finagent.agent   Agent 核心链路（意图识别 → 任务拆解 → 检索 → 工具调用 → 校验 → 生成）
- finagent.web     轻量 Web 演示界面
"""

from .agent import AgentResponse, FinAgent
from .factory import create_agent

__version__ = "1.0.0"
__all__ = ["data", "quant", "rag", "agent", "web", "FinAgent", "AgentResponse", "create_agent"]
