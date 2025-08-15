"""意图识别与问题改写（Agent 核心链路第一步）。

五类业务意图（对应简历场景）：
- wealth_advice    理财咨询
- knowledge_qa     金融知识问答
- stock_screen     自然语言选股
- stock_analysis   个股分析
- strategy_explain 策略解释

规则版识别：关键词命中 + 优先级裁决，离线可用；
实体抽取：股票代码/名称、时间范围、选股条件、数量。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import FALLBACK_STOCK_NAMES

INTENT_LABELS = {
    "wealth_advice": "理财咨询",
    "knowledge_qa": "金融知识问答",
    "stock_screen": "自然语言选股",
    "stock_analysis": "个股分析",
    "strategy_explain": "策略解释",
}

# 常见股票名称 → 代码（扩大 NLU 覆盖；数据目录有 stock_list.csv 时以它为准）
COMMON_STOCK_NAMES = {
    "贵州茅台": "600519", "茅台": "600519",
    "宁德时代": "300750", "宁德": "300750",
    "招商银行": "600036", "招行": "600036",
    "比亚迪": "002594",
    "中国平安": "601318", "平安": "601318",
    "五粮液": "000858", "泸州老窖": "000568",
    "隆基绿能": "601012", "阳光电源": "300274",
    "东方财富": "300059", "同花顺": "300033",
    "中信证券": "600030", "东方证券": "600958",
    "中国移动": "600941", "中国电信": "601728", "中国联通": "600050",
    "工商银行": "601398", "建设银行": "601939", "农业银行": "601288",
    "中国银行": "601988", "交通银行": "601328", "兴业银行": "601166",
    "民生银行": "600016", "浦发银行": "600000", "平安银行": "000001",
    "万科a": "000002", "保利发展": "600048",
    "长江电力": "600900", "中国神华": "601088", "陕西煤业": "601225",
    "万华化学": "600309", "恒瑞医药": "600276", "药明康德": "603259",
    "迈瑞医疗": "300760", "爱尔眼科": "300015", "智飞生物": "300122",
    "海天味业": "603288", "伊利股份": "600887", "牧原股份": "002714",
    "海康威视": "002415", "立讯精密": "002475", "工业富联": "601138",
    "京东方a": "000725", "tcl科技": "000100", "中芯国际": "688981",
    "寒武纪": "688256", "海光信息": "688041", "中际旭创": "300308",
    "新易盛": "300502", "天孚通信": "300394",
    "三一重工": "600031", "潍柴动力": "000338", "汇川技术": "300124",
    "格力电器": "000651", "美的集团": "000333", "海尔智家": "600690",
    "上汽集团": "600104", "长城汽车": "601633", "赛力斯": "601127",
    "中国中免": "601888", "上海机场": "600009",
    "中国石油": "601857", "中国石化": "600028",
    "紫金矿业": "601899", "山东黄金": "600547",
    "韦尔股份": "603501", "兆易创新": "603986", "澜起科技": "688008",
}

# 各意图关键词（含权重，多命中累加）
INTENT_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "stock_analysis": [
        ("分析", 2.0), ("怎么样", 2.0), ("走势", 1.5), ("能买", 1.5), ("技术面", 2.0),
        ("基本面", 2.0), ("估值", 1.0), ("支撑", 1.0), ("压力", 1.0), ("目标价", 1.5),
        ("行情", 1.0), ("涨得", 1.0), ("跌得", 1.0), ("评论", 0.5),
    ],
    "stock_screen": [
        ("选股", 3.0), ("筛选", 2.5), ("推荐几只", 3.0), ("帮我找", 2.0), ("哪些股票", 2.0),
        ("强势", 1.0), ("超跌", 1.5), ("低波动", 1.5), ("放量", 1.0), ("动量", 1.0),
        ("龙头", 1.0), ("选一", 1.0), ("挑几", 2.0),
    ],
    "strategy_explain": [
        ("策略", 2.0), ("回测", 3.0), ("胜率", 2.5), ("年化", 1.5), ("回撤", 2.0),
        ("夏普", 2.0), ("盈亏比", 2.0), ("净值", 1.0), ("卡玛", 2.0), ("收益曲线", 2.0),
        ("绩效", 1.0), ("超额", 1.0), ("跑赢", 1.0),
    ],
    "knowledge_qa": [
        ("什么是", 3.0), ("是什么", 3.0), ("是什么意思", 3.0), ("啥是", 3.0), ("啥意思", 3.0),
        ("介绍一下", 2.0), ("解释", 1.5),
        ("原理", 1.5), ("怎么算", 2.0), ("如何计算", 2.0), ("区别", 1.5), ("定义", 1.5),
        ("为什么", 1.0), ("指标", 0.5),
    ],
    "wealth_advice": [
        ("理财", 2.5), ("定投", 2.5), ("资产配置", 3.0), ("怎么投资", 2.0),
        ("如何理财", 2.5), ("存款", 1.5), ("基金", 1.0), ("保险", 1.0), ("养老", 1.0),
        ("攒钱", 1.5), ("配置比例", 2.0), ("每月", 1.0), ("工资", 1.0), ("结余", 1.5),
        ("闲钱", 1.5), ("风险偏好", 2.0), ("组合", 0.5),
    ],
}

# 选股条件 → 因子映射（自然语言选股 NLU）
SCREEN_RULES: list[tuple[tuple[str, ...], dict]] = [
    (("趋势强", "均线多头", "强势", "趋势好", "多头排列", "趋势向上"),
     {"factor": "trend", "dir": "desc", "label": "趋势强"}),
    (("动量", "涨得多", "领涨", "涨幅大", "近期涨幅", "冲得猛"),
     {"factor": "momentum", "dir": "desc", "label": "动量强"}),
    (("超跌", "回调", "跌得多", "跌幅大", "超卖", "洼地"),
     {"factor": "drawdown_60d", "dir": "desc", "label": "超跌"}),
    (("低波动", "稳健", "波动小", "波动低"),
     {"factor": "vol20", "dir": "asc", "label": "低波动"}),
    (("放量", "量能", "成交活跃", "量比大"),
     {"factor": "vol_ratio", "dir": "desc", "label": "放量"}),
    (("综合", "综合评分", "量化评分", "因子评分"),
     {"factor": "composite", "dir": "desc", "label": "综合评分高"}),
]

_TIME_RE = re.compile(r"(?:最近|(?<!接)近)\s*(\d+)?\s*个?\s*(天|日|周|月|年)")
_TOPN_RE = re.compile(r"(\d+)\s*(?:只|个|支)")
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:万|千)?\s*元?")
_YEAR_RE = re.compile(r"(\d+)\s*年")
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass
class StockEntity:
    code: str
    name: str


@dataclass
class Entities:
    stocks: list[StockEntity] = field(default_factory=list)
    time_range_days: int | None = None
    screen_conditions: list[dict] = field(default_factory=list)
    screen_top_n: int = 5
    monthly_amount: float | None = None   # 理财场景：每月可投金额
    horizon_years: int | None = None      # 理财场景：投资年限


@dataclass
class IntentResult:
    intent: str
    label: str
    confidence: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


class IntentRecognizer:
    """规则版意图识别 + 实体抽取 + 问题改写。"""

    def __init__(self, name_map: dict[str, str] | None = None):
        # name → code
        self.name_map = dict(COMMON_STOCK_NAMES)
        for code, name in (name_map or FALLBACK_STOCK_NAMES).items():
            self.name_map[name] = code
            self.name_map[name.rstrip("Aa").rstrip("Aa")] = code

    # ------------------------------------------------------------------ #
    def classify(self, text: str) -> IntentResult:
        """关键词加权评分 + 优先级裁决。"""
        scores = {intent: 0.0 for intent in INTENT_KEYWORDS}
        lowered = text.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            for kw, weight in keywords:
                if kw in lowered:
                    scores[intent] += weight

        # 优先级裁决：个股实体 + 分析词 > 选股动词 > 策略词 > 知识模式 > 理财词
        entities = self.extract_entities(text)
        intent, reason = "knowledge_qa", "未命中强特征，按知识问答处理"

        has_stock = bool(entities.stocks)
        if has_stock and scores["stock_analysis"] >= 1.5:
            intent = "stock_analysis"
            reason = f"识别到股票实体（{entities.stocks[0].name}）+ 分析意图词"
        elif scores["stock_screen"] >= 2.0:
            intent = "stock_screen"
            reason = "命中选股动词/条件词"
        elif scores["knowledge_qa"] >= 2.0 and scores["knowledge_qa"] >= scores["strategy_explain"]:
            # 定义式提问（什么是/怎么算）优先于策略执行：用户要解释，不是要跑回测
            intent = "knowledge_qa"
            reason = "命中定义式提问模式（什么是/怎么算/区别等）"
        elif scores["strategy_explain"] >= 2.0:
            intent = "strategy_explain"
            reason = "命中策略/回测/指标词"
        elif scores["knowledge_qa"] >= 2.0:
            intent = "knowledge_qa"
            reason = "命中知识问答模式（什么是/怎么算等）"
        elif scores["wealth_advice"] >= 2.0:
            intent = "wealth_advice"
            reason = "命中理财咨询关键词"
        elif max(scores.values()) > 0:
            intent = max(scores, key=scores.get)
            reason = f"关键词加权最高：{INTENT_LABELS[intent]}"

        # 置信度：命中强度 + 领先优势
        top = scores[intent]
        second = max((v for k, v in scores.items() if k != intent), default=0.0)
        margin = top - second
        confidence = min(0.55 + 0.08 * top + 0.1 * margin, 0.97)

        return IntentResult(
            intent=intent, label=INTENT_LABELS[intent],
            confidence=round(confidence, 2), reason=reason, scores=scores,
        )

    # ------------------------------------------------------------------ #
    def extract_entities(self, text: str) -> Entities:
        """抽取股票实体、时间范围、选股条件、数量、理财金额。"""
        ent = Entities()

        # 股票：代码优先，其次名称（长名优先匹配，避免「茅台」抢在「贵州茅台」前）
        for m in _CODE_RE.finditer(text):
            code = m.group(1)
            ent.stocks.append(StockEntity(code=code, name=self._code_to_name(code)))
        for name in sorted(self.name_map, key=len, reverse=True):
            code = self.name_map[name]
            if name in text and not any(s.code == code for s in ent.stocks):
                ent.stocks.append(StockEntity(code=code, name=name))

        # 时间范围
        m = _TIME_RE.search(text)
        if m:
            num = int(m.group(1)) if m.group(1) else 1
            unit = m.group(2)
            ent.time_range_days = num * {"天": 1, "日": 1, "周": 7, "月": 30, "年": 365}[unit]
        elif "今年以来" in text or "今年" in text:
            ent.time_range_days = None  # 由工具层按数据区间处理

        # 选股条件
        for keywords, spec in SCREEN_RULES:
            if any(kw in text for kw in keywords):
                ent.screen_conditions.append(dict(spec))

        # 数量
        m = _TOPN_RE.search(text)
        if m:
            ent.screen_top_n = min(int(m.group(1)), 10)

        # 理财金额与年限（金额必须锚定「每月」语义，避免把总资产误当月投）
        m = re.search(r"每月(?:定投|存|能存|拿出|投)?\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元?", text)
        if m:
            v = float(m.group(1))
            unit = m.group(2)
            if unit == "万":
                v *= 10000
            elif unit == "千":
                v *= 1000
            ent.monthly_amount = v
        m = _YEAR_RE.search(text)
        if m:
            ent.horizon_years = int(m.group(1))
        return ent

    # ------------------------------------------------------------------ #
    def rewrite(self, text: str, intent: str, entities: Entities) -> str:
        """问题改写：将口语化提问改写为结构化投研任务描述。"""
        if intent == "stock_analysis" and entities.stocks:
            s = entities.stocks[0]
            period = f"近 {entities.time_range_days} 日" if entities.time_range_days else "近一年"
            return f"分析{s.name}（{s.code}）{period}的行情表现、技术面信号（均线/MACD/RSI/KDJ）与量化因子评分"
        if intent == "stock_screen":
            conds = "、".join(c["label"] for c in entities.screen_conditions) or "综合因子评分"
            return f"在样本股票池中按「{conds}」筛选，返回排名前 {entities.screen_top_n} 的标的"
        if intent == "strategy_explain":
            return "运行多因子趋势策略回测，解释年化收益、最大回撤、夏普比率、胜率等关键指标的含义与策略逻辑"
        if intent == "wealth_advice":
            profile = ""
            if entities.monthly_amount:
                profile += f"每月可投入约 {entities.monthly_amount:.0f} 元"
            if entities.horizon_years:
                profile += f"、投资期限 {entities.horizon_years} 年"
            return f"基于公开理财知识，为{profile or '普通投资者'}提供资产配置框架与定投测算建议"
        # knowledge_qa：去口语化填充
        cleaned = text.strip()
        for filler in ("请问", "我想问一下", "我想知道", "帮我", "麻烦", "一下", "呢"):
            cleaned = cleaned.replace(filler, "")
        return cleaned or text

    # ------------------------------------------------------------------ #
    @staticmethod
    def _code_to_name(code: str) -> str:
        return FALLBACK_STOCK_NAMES.get(code, code)
