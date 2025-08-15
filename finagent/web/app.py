"""FinAgent Web 层：标准库 HTTP 服务（零第三方依赖）。

路由：
- GET  /                聊天 + 回测面板（单页应用）
- GET  /api/info        系统信息（标的池 / 知识库 / 工具清单）
- POST /api/chat        {question, session_id} → 完整链路 JSON（答案已渲染为 HTML）
- POST /api/backtest    策略回测（指标 + 权益曲线 + 回撤曲线）
- POST /api/analyze     {symbol} → 个股分析数据

启动：
    python -m finagent.web.app --port 8600
"""
from __future__ import annotations

import html as html_mod
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..factory import create_agent
from ..quant.backtest import Backtester

STATIC_DIR = Path(__file__).parent / "static"

# 全局单例（演示服务，单进程足够）
_AGENT = None
_BACKTESTER = None
_BACKTEST_CACHE = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = create_agent()
    return _AGENT


def get_backtester():
    global _BACKTESTER
    if _BACKTESTER is None:
        _BACKTESTER = Backtester(get_agent().market)
    return _BACKTESTER


# ---------------------------------------------------------------------- #
# Markdown → HTML（答案渲染，服务端完成，前端零依赖）
# ---------------------------------------------------------------------- #
def _inline(text: str) -> str:
    text = html_mod.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def md_to_html(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        # 表格：| h | h |  紧跟 | --- | --- |
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            head = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<div class='table-wrap'><table><thead><tr>"
                       + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue

        if not stripped:
            i += 1
            continue

        # 合规横幅特殊处理
        if stripped.startswith("⚠️"):
            out.append(f"<div class='compliance-warning'>{_inline(stripped)}</div>")
            i += 1
            continue
        if stripped.startswith("📎"):
            out.append(f"<div class='data-source'>{_inline(stripped)}</div>")
            i += 1
            continue
        if set(stripped) == {"-"} and len(stripped) >= 3:
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("> "):
            out.append(f"<blockquote>{_inline(stripped[2:])}</blockquote>")
            i += 1
            continue

        if stripped.startswith("- "):
            out.append(f"<div class='li'>{_inline(stripped[2:])}</div>")
            i += 1
            continue

        out.append(f"<p>{_inline(stripped)}</p>")
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------- #
# 回测数据序列（供前端 SVG 绘图，降采样）
# ---------------------------------------------------------------------- #
def build_backtest_payload(result, max_points: int = 400) -> dict:
    eq = result.equity_curve
    bench = result.benchmark_curve
    dd = result.drawdown_curve

    step = max(1, len(eq) // max_points)
    dates = [str(d.date()) for d in eq.index][::step]
    strategy = [round(float(v), 2) for v in eq.values][::step]
    benchmark = [round(float(v), 2) for v in bench.values][::step] if bench is not None and len(bench) else []
    drawdown = [round(float(v) * 100, 3) for v in dd.values][::step]

    trades_df = result.trades_df() if hasattr(result, "trades_df") else None
    sell_count = int((trades_df["side"] == "SELL").sum()) if trades_df is not None and "side" in trades_df else 0

    return {
        "date_range": list(result.date_range),
        "metrics": result.metrics,
        "strategy_desc": result.strategy_desc,
        "series": {
            "dates": dates,
            "strategy": strategy,
            "benchmark": benchmark,
            "drawdown": drawdown,
        },
        "trade_count": sell_count,
    }


# ---------------------------------------------------------------------- #
# HTTP 处理器
# ---------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "FinAgent/1.0"

    # 静默日志（演示服务）
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # ------------------------------------------------------------------ #
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/api/info":
            self._handle_info()
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/api/chat":
            self._handle_chat(body)
        elif path == "/api/backtest":
            self._handle_backtest(body)
        elif path == "/api/analyze":
            self._handle_analyze(body)
        else:
            self.send_error(404)

    # ------------------------------------------------------------------ #
    def _handle_info(self) -> None:
        agent = get_agent()
        self._send_json({
            "universe": [
                {"symbol": s, "name": agent.market.name_of(s)}
                for s in agent.market.symbols()
            ],
            "knowledge_chunks": len(agent.retriever.kb) if agent.retriever else 0,
            "tools": agent.describe_tools(),
            "llm_available": agent.llm.available,
            "disclaimer": (
                "本平台所有内容由量化模型基于公开数据自动生成，仅供学习研究参考，"
                "不构成投资建议。市场有风险，投资需谨慎。"
            ),
        })

    def _handle_chat(self, body: dict) -> None:
        question = (body.get("question") or "").strip()
        session_id = (body.get("session_id") or "web").strip()
        if not question:
            self._send_json({"error": "问题不能为空"}, status=400)
            return
        agent = get_agent()
        try:
            resp = agent.ask(question, session_id=session_id)
        except Exception as err:  # noqa: BLE001
            self._send_json({"error": f"链路执行异常：{err}"}, status=500)
            return
        self._send_json({
            "question": resp.question,
            "intent": resp.intent,
            "intent_label": resp.intent_label,
            "intent_confidence": resp.intent_confidence,
            "intent_reason": resp.intent_reason,
            "rewritten": resp.rewritten,
            "answer_html": md_to_html(resp.answer),
            "plan": resp.plan,
            "trace": resp.trace,
            "citations": resp.citations,
            "validation": resp.validation,
            "mode": resp.mode,
            "elapsed_ms": resp.elapsed_ms,
        })

    def _handle_backtest(self, body: dict) -> None:
        global _BACKTEST_CACHE
        try:
            if _BACKTEST_CACHE is None:
                result = get_backtester().run()
                _BACKTEST_CACHE = build_backtest_payload(result)
            self._send_json(_BACKTEST_CACHE)
        except Exception as err:  # noqa: BLE001
            self._send_json({"error": f"回测执行失败：{err}"}, status=500)

    def _handle_analyze(self, body: dict) -> None:
        symbol = (body.get("symbol") or "").strip()
        if not symbol:
            self._send_json({"error": "缺少股票代码"}, status=400)
            return
        agent = get_agent()
        result = agent.tools.call("analyze_stock", symbol=symbol)
        if not result.ok:
            self._send_json({"error": result.error}, status=400)
            return
        self._send_json(result.data)


def run(host: str = "127.0.0.1", port: int = 8600) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"FinAgent Web 界面已启动：http://{host}:{port}")
    print("（Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinAgent Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    run(args.host, args.port)
