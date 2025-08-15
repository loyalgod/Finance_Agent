"""Agent 核心链路 CLI 演示：五大场景 + 多轮上下文。

运行：
    python -m demo.demo_agent                 # 五大场景
    python -m demo.demo_agent --interactive   # 交互模式
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 python demo/demo_agent.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finagent import create_agent  # noqa: E402
from finagent.agent.intent import INTENT_LABELS  # noqa: E402

SCENARIOS = [
    ("金融知识问答", "什么是夏普比率？它和最大回撤有什么区别？"),
    ("理财咨询", "我每月能存 5000 元，想 10 年后退休，该怎么配置？"),
    ("自然语言选股", "帮我筛选 3 只趋势强、动量强的股票"),
    ("个股分析", "分析一下贵州茅台最近 60 天的走势"),
    ("策略解释", "你们的量化策略回测表现怎么样？胜率和年化收益多少？"),
]

MULTI_TURN = [
    "分析一下招商银行",
    "那它最近一年的走势呢",   # 多轮实体承接（沿用上轮股票）
    "它的胜率呢",             # 多轮意图延续（承接策略解释）
]


def print_divider(title: str = "", char: str = "─", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{char * pad} {title} {char * pad}")
    else:
        print(char * width)


def print_trace(resp) -> None:
    """打印链路执行轨迹（简历核心链路的可视化）。"""
    print(f"\n【链路轨迹】意图：{resp.intent_label}（置信度 {resp.intent_confidence:.0%}，{resp.intent_reason}）")
    print(f"  改写：{resp.rewritten}")
    print(f"  计划：{len(resp.plan)} 步 | 执行：{len(resp.trace)} 步 | 生成：{resp.mode} | 耗时 {resp.elapsed_ms}ms")
    for step in resp.trace:
        icon = "✓" if step["status"] == "ok" else "✗"
        print(f"    {icon} [{step['stage']}] {step['tool']}: {step['summary'][:60]}")
    v = resp.validation
    if v["issues"]:
        print(f"  ⚠ 校验问题：{v['issues']}")
    elif v["warnings"]:
        print(f"  ⚠ 校验提示：{v['warnings']}")
    else:
        print("  校验：全部通过 ✓")


def run_scenarios(agent) -> None:
    for i, (name, q) in enumerate(SCENARIOS, 1):
        print_divider(f"场景 {i}：{name}")
        print(f"\n用户：{q}")
        resp = agent.ask(q, session_id=f"scenario-{i}")
        print_trace(resp)
        print(f"\n{resp.answer}")
        if resp.citations:
            print(f"\n引用：{'、'.join(resp.citations[:3])}")


def run_multi_turn(agent) -> None:
    print_divider("多轮上下文演示（实体承接 + 意图延续）")
    sid = "multi-turn"
    for q in MULTI_TURN:
        print(f"\n用户：{q}")
        resp = agent.ask(q, session_id=sid)
        print(f"  意图：{resp.intent_label}（{resp.intent_reason}）")
        print(f"  改写：{resp.rewritten}")
        print(f"  → {resp.answer.splitlines()[0]}")
    print("\n（前两轮围绕招商银行延续；第三轮「它的胜率呢」触发策略回测上下文延续）")


def interactive(agent) -> None:
    print("FinAgent 交互模式（输入 exit 退出，clear 清空会话）")
    sid = "interactive"
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        if q == "clear":
            agent.reset(sid)
            print("（会话已清空）")
            continue
        resp = agent.ask(q, session_id=sid)
        print_trace(resp)
        print(f"\nFinAgent：\n{resp.answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinAgent 核心链路演示")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    agent = create_agent()
    print(f"FinAgent v1.0 已装配：行情 {len(agent.market)} 只标的 | "
          f"知识库 {len(agent.retriever.kb) if agent.retriever else 0} 条 | "
          f"工具 {len(agent.describe_tools())} 个 | LLM：{'已启用' if agent.llm.available else '未启用（模板模式）'}")

    if args.interactive:
        interactive(agent)
    else:
        run_scenarios(agent)
        run_multi_turn(agent)
    print("\n演示完成。")


if __name__ == "__main__":
    main()
