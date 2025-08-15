```text
用户问题
   │
   ▼
① 意图识别与问题改写      关键词加权评分 + 实体抽取（股票/时间/选股条件/金额）
   │                      五类意图：理财咨询 · 知识问答 · 选股 · 个股分析 · 策略解释
   ▼
② 复杂任务拆解            按意图模板生成「检索 / 工具」两阶段步骤计划
   │                      失败步骤降级容错，不中断主链路
   ▼
③ 金融知识/行情数据检索    RAG：BM25 检索（35 条金融知识，纯 Python 零依赖）
   │                      行情：MarketData 本地 CSV 访问层
   ▼
④ 专业工具调用            六大工具：知识检索 / 个股分析 / 自然语言选股 /
   │                      区间行情 / 策略回测 / 金融计算器
   ▼
⑤ 结果校验与风险边界控制   数值合理性校验（价格/涨跌幅/RSI 区间）·
   │                      禁词拦截（保证收益/稳赚/内幕）·
   │                      免责声明强制注入 · 数据溯源标注
   ▼
⑥ 答案生成                模板引擎（离线可用） / LLM 增强（OpenAI 兼容接口，可选）
   │
   ▼
结构化答案（含链路轨迹、引用、风险提示与数据来源）
```

**多轮上下文**：实体承接（「那它呢」沿用上轮股票）与意图延续（「它的胜率呢」触发策略回测），滑动窗口 8 轮。

## 🚀 快速开始

```bash
# 1. 安装依赖（仅 pandas / numpy / matplotlib）
pip install pandas numpy matplotlib

# 2. CLI 演示：五大场景 + 多轮上下文（含链路轨迹打印）
python -m demo.demo_agent

# 3. 回测演示：训练/测试划分 + 绩效指标 + 权益曲线图（输出到 outputs/）
python -m demo.demo_backtest

# 4. Web 界面（聊天 + 回测面板），打开 http://127.0.0.1:8600
python -m finagent.web.app --port 8600

# 5. 单元测试（44 个用例）
python -m unittest discover -s tests -v

# 可选：交互模式
python -m demo.demo_agent --interactive
```

<details>
<summary>可选：接入 LLM 增强答案生成（OpenAI 兼容接口，如自托管 vLLM）</summary>

```bash
export FINAGENT_LLM_BASE_URL=http://127.0.0.1:10001/v1   # 你的 OpenAI 兼容端点
export FINAGENT_LLM_API_KEY=EMPTY                          # 按端点要求填写
export FINAGENT_LLM_MODEL=Qwen3-27B                        # 模型名
```

未配置时自动使用模板引擎（离线可用）；LLM 调用失败自动降级模板，不影响链路可用性。LLM 模式下工具结果以 JSON 注入提示词，红线：禁止编造数据、数字必须与工具结果一致、违规表述由校验器拦截。

</details>

<details>
<summary>可选：抓取最新行情数据（腾讯财经主源）</summary>

```bash
# 抓取样本池最新行情（增量会覆盖 data/raw/*.csv）
python -m finagent.data.fetch_data --start 20230101
```

数据源说明：腾讯财经 `ifzq` 接口为主源（沙箱/内网可达），东方财富 `push2his` 为备源，两源交叉验证一致（以茅台为例）。网络受限环境下仓库自带的 373 KB 样本数据可直接运行全部演示。

</details>

---

## 📁 项目结构

```text
Finance_agent/
├── finagent/
│   ├── config.py              # 全局配置：股票池/交易参数/风控参数/策略参数/LLM 配置
│   ├── factory.py             # 装配工厂：create_agent() 一行构建
│   ├── data/                  # 行情数据层
│   │   ├── tencent.py         #   腾讯财经客户端（主源，按年分块抓取）
│   │   ├── eastmoney.py       #   东方财富客户端（备源）
│   │   └── fetch_data.py      #   抓取脚本（多源 fallback）
│   ├── quant/                 # 量化引擎
│   │   ├── data.py            #   MarketData 行情访问层
│   │   ├── indicators.py      #   技术指标库（MA/MACD/RSI/KDJ/BOLL/ATR/OBV）
│   │   ├── factors.py         #   特征与因子（趋势/动量/RSI/量能 + 综合加权）
│   │   ├── signals.py         #   多因子趋势策略信号（滞回处理）
│   │   ├── portfolio.py       #   ATR 风险预算仓位管理
│   │   ├── risk.py            #   风控（止损/止盈/熔断/冷静期）
│   │   ├── backtest.py        #   事件驱动回测引擎（无未来函数）
│   │   └── report.py          #   权益曲线/回撤图 + 交易明细导出
│   ├── rag/                   # RAG 检索
│   │   ├── bm25.py            #   BM25 索引（中文单字+bigram 分词，零依赖）
│   │   ├── knowledge_base.py  #   知识库（knowledge/*.md 按 ## 切分）
│   │   └── retriever.py       #   检索器（双信号相关性过滤）
│   ├── agent/                 # Agent 核心链路
│   │   ├── intent.py          #   意图识别 + 实体抽取 + 问题改写
│   │   ├── planner.py         #   复杂任务拆解
│   │   ├── tools.py           #   工具注册表 + 六大工具
│   │   ├── validator.py       #   结果校验与风险边界控制
│   │   ├── context.py         #   多轮上下文（实体承接/意图延续）
│   │   └── agent.py           #   FinAgent 主类（全链路串联 + 双模答案生成）
│   └── web/                   # Web 界面
│       ├── app.py             #   标准库 HTTP 服务（/api/chat /api/backtest /api/analyze）
│       └── static/            #   单页应用（原生 JS，SVG 图表，无 CDN 依赖）
├── knowledge/                 # 金融知识库（5 领域 35 条）
│   ├── 理财基础.md            #   复利/定投/资产配置/应急金/通胀/风险测评
│   ├── 金融概念.md            #   PE/PB/夏普/最大回撤/年化/波动率/Beta/净值
│   ├── 技术指标.md            #   MA/MACD/RSI/KDJ/BOLL/ATR/OBV
│   ├── 量化策略.md            #   双均线/海龟/多因子/动量/均值回归/回测/网格
│   └── 风险管理.md            #   止损/止盈/仓位/熔断/风险预算/分散化/合规边界
├── demo/
│   ├── demo_agent.py          # 五大场景 + 多轮上下文 CLI 演示
│   └── demo_backtest.py       # 训练/测试划分 + 绩效 + 权益曲线
├── tests/                     # 44 个单元测试
│   ├── test_indicators.py     #   指标库（与手工计算交叉验证）
│   ├── test_rag.py            #   RAG 检索质量
│   ├── test_intent.py         #   意图识别 + 实体抽取 + 问题改写
│   └── test_backtest_validator.py  # 回测引擎 + 校验器 + Agent 全链路冒烟
├── data/raw/                  # 样本池行情（5 股 + 基准，373 KB，前复权）
└── outputs/                   # 运行产物（git 忽略，可重新生成）
```

---

## 📈 量化策略与风控

**多因子趋势增强策略**：综合分 = 40%×趋势 + 30%×动量 + 15%×RSI + 15%×量能；综合分 ≥ 0.30 开仓，≤ -0.20 清仓；60 日指标预热。

| 环节     | 实现                                                                |
| -------- | ------------------------------------------------------------------- |
| 仓位管理 | ATR 风险预算法：单笔风险预算 = 账户权益 × 2%                        |
| 止损     | 固定止损 8%（入场价）优先于 ATR 吊灯止损 2.5×                       |
| 止盈     | 固定止盈 +25%                                                       |
| 熔断     | 账户回撤 > 15% 清仓，5 日冷静期（期满重置峰值水位，防止熔断死锁）   |
| 交易成本 | 佣金万 2.5（双边）、印花税千 0.5（仅卖出）、滑点 0.05%、整手 100 股 |

**回测结果**（2023-01-03 ~ 2026-09-11，初始资金 100 万元）：

| 区间                 | 年化收益 | 最大回撤 | 夏普 | 胜率  | 盈亏比 | 交易  |
| -------------------- | -------- | -------- | ---- | ----- | ------ | ----- |
| 全样本               | +6.62%   | -16.71%  | 0.46 | 34.7% | 2.55   | 72 笔 |
| 训练段（前 70%）     | +5.91%   | -16.71%  | 0.41 | 33.3% | 2.59   | 51 笔 |
| 样本外测试（后 30%） | +8.16%   | -9.09%   | 0.60 | 33.3% | 2.96   | 24 笔 |

> 样本外表现优于训练段，策略未见明显过拟合；但胜率偏低依赖盈亏比（2.5+）的结构特征，实盘需关注交易成本敏感性与容量约束。

---

## 🧪 测试

44 个单元测试全部通过（`python -m unittest discover -s tests`）：

| 测试文件                     | 覆盖                                                                 |
| ---------------------------- | -------------------------------------------------------------------- |
| `test_indicators.py`         | MA/RSI/MACD/ATR/KDJ 与手工计算交叉验证、值域约束                     |
| `test_rag.py`                | 知识命中（夏普/最大回撤/均线）、引用格式、垃圾查询过滤               |
| `test_intent.py`             | 五类意图分类、实体抽取（股票/时间/条件/金额）、问题改写              |
| `test_backtest_validator.py` | 回测指标合理性、无未来函数约束、禁词拦截、数值校验、Agent 全链路冒烟 |
