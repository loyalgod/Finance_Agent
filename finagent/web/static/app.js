/* FinAgent Web 前端逻辑 */
(function () {
  "use strict";

  var messagesEl = document.getElementById("chat-messages");
  var formEl = document.getElementById("chat-form");
  var inputEl = document.getElementById("chat-input");
  var sendBtn = document.getElementById("send-btn");
  var suggestionsEl = document.getElementById("suggestions");
  var sessionId = "web-" + Date.now();

  /* ---------------- 聊天 ---------------- */
  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMessage(role, cardHtml) {
    var msg = document.createElement("div");
    msg.className = "msg " + (role === "user" ? "msg-user" : "msg-assistant");
    var avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "user" ? "你" : "FA";
    var body = document.createElement("div");
    body.className = "msg-body";
    var card = document.createElement("div");
    card.className = "msg-card";
    card.innerHTML = cardHtml;
    body.appendChild(card);
    msg.appendChild(avatar);
    msg.appendChild(body);
    messagesEl.appendChild(msg);
    scrollBottom();
    return card;
  }

  function buildTraceHtml(resp) {
    var lines = resp.trace.map(function (s) {
      var icon = s.status === "ok" ? "✓" : "✗";
      return '<div class="trace-line">' +
        '<span class="trace-icon ' + s.status + '">' + icon + '</span>' +
        '<span><b>' + s.tool + '</b>（' + s.stage + "）" + escapeHtml(s.summary.slice(0, 70)) + "</span>" +
        '<span class="trace-meta">' + resp.plan.length + " 步计划 · " + resp.elapsed_ms + "ms · " +
        (resp.mode === "llm" ? "LLM 生成" : "模板生成") + "</span></div>";
    }).join("");
    var v = resp.validation || {};
    var vline = (v.issues && v.issues.length)
      ? '<div class="trace-line"><span class="trace-icon failed">⚠</span><span>' + escapeHtml(v.issues.join("；")) + "</span></div>"
      : '<div class="trace-line"><span class="trace-icon ok">✓</span><span>结果校验全部通过</span></div>';
    return '<details class="trace"><summary>查看 Agent 链路轨迹（意图 → 拆解 → 工具 → 校验）</summary>' +
      '<div class="trace-body">' +
      '<div class="trace-line"><span><b>意图</b>：' + escapeHtml(resp.intent_label) +
      "（置信度 " + Math.round(resp.intent_confidence * 100) + "%）</span></div>" +
      '<div class="trace-line"><span><b>改写</b>：' + escapeHtml(resp.rewritten) + "</span></div>" +
      lines + vline + "</div></details>";
  }

  function buildCitationsHtml(resp) {
    if (!resp.citations || !resp.citations.length) return "";
    return '<div class="citations">' + resp.citations.slice(0, 5).map(function (c) {
      return '<span class="citation-tag">📚 ' + escapeHtml(c) + "</span>";
    }).join("") + "</div>";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function send(question) {
    if (!question.trim()) return;
    addMessage("user", "<p>" + escapeHtml(question) + "</p>");
    var thinking = addMessage("assistant",
      '<span class="thinking"><i></i><i></i><i></i></span><p style="color:#8a94a3;font-size:13px">正在执行 Agent 链路…</p>');
    sendBtn.disabled = true;

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, session_id: sessionId })
    })
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (resp.error) {
          thinking.innerHTML = "<p class='down'>" + escapeHtml(resp.error) + "</p>";
          return;
        }
        thinking.innerHTML =
          '<span class="intent-badge">' + escapeHtml(resp.intent_label) + " · " +
          Math.round(resp.intent_confidence * 100) + "%</span>" +
          resp.answer_html +
          buildCitationsHtml(resp) +
          buildTraceHtml(resp);
        scrollBottom();
      })
      .catch(function (err) {
        thinking.innerHTML = "<p class='down'>请求失败：" + escapeHtml(String(err)) + "</p>";
      })
      .finally(function () {
        sendBtn.disabled = false;
        inputEl.focus();
      });
  }

  formEl.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = inputEl.value;
    inputEl.value = "";
    autoResize();
    send(q);
  });

  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formEl.dispatchEvent(new Event("submit"));
    }
  });

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
  }
  inputEl.addEventListener("input", autoResize);

  suggestionsEl.addEventListener("click", function (e) {
    var btn = e.target.closest(".suggestion-chip");
    if (btn) send(btn.dataset.q);
  });

  /* ---------------- 系统信息 ---------------- */
  fetch("/api/info")
    .then(function (r) { return r.json(); })
    .then(function (info) {
      var chips = document.getElementById("status-chips");
      chips.innerHTML =
        '<span class="chip">样本池 <b>' + info.universe.length + "</b> 只</span>" +
        '<span class="chip">知识库 <b>' + info.knowledge_chunks + "</b> 条</span>" +
        '<span class="chip">工具 <b>' + info.tools.length + "</b> 个</span>" +
        '<span class="chip">' + (info.llm_available ? "LLM 已启用" : "模板模式（离线可用）") + "</span>";
      var list = document.getElementById("stock-list");
      list.innerHTML = info.universe.map(function (s) {
        return '<div class="stock-row" data-symbol="' + s.symbol + '">' +
          '<span><span class="stock-name">' + escapeHtml(s.name) + "</span>" +
          '<span class="stock-code">' + s.symbol + "</span></span>" +
          '<span class="stock-action">分析 →</span></div>';
      }).join("");
      list.addEventListener("click", function (e) {
        var row = e.target.closest(".stock-row");
        if (row) send("分析一下 " + row.dataset.symbol);
      });
    });

  /* ---------------- 回测面板 ---------------- */
  var btBody = document.getElementById("backtest-body");
  var btBtn = document.getElementById("run-backtest");

  function pct(v, digits) {
    var d = digits === undefined ? 2 : digits;
    var cls = v >= 0 ? "up" : "down";  // 涨红跌绿
    return '<span class="' + cls + '">' + (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%</span>";
  }

  function metricCard(label, valueHtml) {
    return '<div class="metric"><div class="metric-label">' + label +
      '</div><div class="metric-value">' + valueHtml + "</div></div>";
  }

  function equityChart(series) {
    var dates = series.dates, strat = series.strategy, bench = series.benchmark;
    if (!strat || strat.length < 2) return "<p class='placeholder'>数据不足</p>";
    var W = 360, H = 170, P = 6;
    var all = strat.concat(bench && bench.length ? bench : []);
    var min = Math.min.apply(null, all), max = Math.max.apply(null, all);
    if (max - min < 1) { max = min + 1; }
    function x(i) { return P + (i / (strat.length - 1)) * (W - 2 * P); }
    function y(v) { return H - P - ((v - min) / (max - min)) * (H - 2 * P); }
    var pathStrat = strat.map(function (v, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); }).join("");
    var pathBench = "";
    if (bench && bench.length === strat.length) {
      pathBench = bench.map(function (v, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); }).join("");
    }
    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">' +
      '<line x1="' + P + '" y1="' + (H - P) + '" x2="' + (W - P) + '" y2="' + (H - P) + '" stroke="#e4e7ec"/>' +
      '<line x1="' + P + '" y1="' + P + '" x2="' + (W - P) + '" y2="' + P + '" stroke="#e4e7ec" stroke-dasharray="3 3"/>' +
      '<path d="' + pathBench + '" fill="none" stroke="#9aa3af" stroke-width="1.4" stroke-dasharray="4 3"/>' +
      '<path d="' + pathStrat + '" fill="none" stroke="#d64541" stroke-width="2"/>' +
      '<text x="' + P + '" y="' + (H - 1) + '" font-size="9" fill="#8a94a3">' + dates[0] + "</text>" +
      '<text x="' + (W - P - 60) + '" y="' + (H - 1) + '" font-size="9" fill="#8a94a3">' + dates[dates.length - 1] + "</text>" +
      "</svg>";
    return svg +
      '<div class="chart-legend">' +
      '<span><span class="legend-dot" style="background:#d64541"></span>策略净值（红）</span>' +
      '<span><span class="legend-dot" style="background:#9aa3af"></span>基准（沪深300 等权，灰）</span>' +
      "</div>";
  }

  function drawdownChart(series) {
    var dd = series.drawdown;
    if (!dd || dd.length < 2) return "";
    var W = 360, H = 90, P = 6;
    var min = Math.min.apply(null, dd);
    var max = 0;
    function x(i) { return P + (i / (dd.length - 1)) * (W - 2 * P); }
    function y(v) { return H - P - ((v - min) / (max - min || 1)) * (H - 2 * P); }
    var path = dd.map(function (v, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); }).join("");
    var area = path + "L" + x(dd.length - 1).toFixed(1) + " " + (H - P) + "L" + P + " " + (H - P) + "Z";
    return '<div class="chart-title">回撤曲线（绿=下跌）</div>' +
      '<svg viewBox="0 0 ' + W + " " + H + '" style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">' +
      '<path d="' + area + '" fill="rgba(46,158,107,.15)" stroke="none"/>' +
      '<path d="' + path + '" fill="none" stroke="#2e9e6b" stroke-width="1.6"/>' +
      '<text x="' + P + '" y="12" font-size="9" fill="#8a94a3">最大回撤 ' + min.toFixed(2) + "%</text>" +
      "</svg>";
  }

  btBtn.addEventListener("click", function () {
    btBtn.disabled = true;
    btBtn.textContent = "回测中…";
    btBody.innerHTML = '<span class="thinking"><i></i><i></i><i></i></span>';
    fetch("/api/backtest", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) { btBody.innerHTML = "<p class='down'>" + escapeHtml(d.error) + "</p>"; return; }
        var m = d.metrics || {};
        btBody.innerHTML =
          '<div style="font-size:12px;color:#8a94a3;margin-bottom:10px">' +
          d.date_range[0] + " ~ " + d.date_range[1] + " · " + d.trade_count + " 笔完整交易</div>" +
          '<div class="metric-grid">' +
          metricCard("年化收益", pct(m.annualized_return || 0)) +
          metricCard("最大回撤", '<span class="down">' + ((m.max_drawdown || 0) * 100).toFixed(2) + "%</span>") +
          metricCard("夏普比率", (m.sharpe || 0).toFixed(2)) +
          metricCard("卡玛比率", (m.calmar || 0).toFixed(2)) +
          metricCard("胜率", ((m.win_rate || 0) * 100).toFixed(1) + "%") +
          metricCard("盈亏比", (m.avg_win && m.avg_loss ? Math.abs(m.avg_win / m.avg_loss).toFixed(2) : "—")) +
          "</div>" +
          '<div class="chart-title">策略净值 vs 基准</div>' +
          equityChart(d.series) +
          drawdownChart(d.series) +
          '<div class="chart-note">策略：多因子趋势增强（趋势40%+动量30%+RSI15%+量能15%）；' +
          "含佣金万2.5/印花税千0.5/滑点0.05%；回撤15%熔断+5日冷静期。历史业绩不代表未来。</div>";
      })
      .catch(function (err) {
        btBody.innerHTML = "<p class='down'>请求失败：" + escapeHtml(String(err)) + "</p>";
      })
      .finally(function () {
        btBtn.disabled = false;
        btBtn.textContent = "重新回测";
      });
  });
})();
