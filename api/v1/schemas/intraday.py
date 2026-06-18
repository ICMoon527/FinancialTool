# -*- coding: utf-8 -*-
"""分时做T API 响应模型"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntradayKlinePoint(BaseModel):
    """分时K线数据点"""

    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Amount: Optional[float] = None
    AvgPrice: Optional[float] = None  # 累计分时均价（成交额/成交量）
    timestamp: str  # ISO格式时间字符串


class WeightContribution(BaseModel):
    """单项权重贡献明细"""

    key: str  # 权重键名
    label: str  # 中文标签
    weight: float  # 满分数
    triggered: bool = False  # 是否触发
    score: float = 0.0  # 实际得分


class IntradaySignal(BaseModel):
    """做T信号"""

    stock_code: str
    signal_type: str  # "buy" | "sell"
    trigger_time: str
    price: float
    score: float
    max_score: int = 10
    confidence: float
    position_advice: str
    reasoning: str = ""
    gravity_adjustment: float = 0.0  # 引力场修正分
    support_force: float = 0.0
    pressure_force: float = 0.0
    buy_weight_details: List[WeightContribution] = Field(default_factory=list)
    sell_weight_details: List[WeightContribution] = Field(default_factory=list)


class ReferenceLine(BaseModel):
    """支撑/压力参考线"""

    id: str
    label: str
    price: float
    category: str
    color: str
    style: str = "dashed"
    base_weight: float = 1.0


# ── 指标子图模型 ──


class IndicatorLinePoint(BaseModel):
    """指标线数据点"""

    time: str  # 时间标签 "HH:MM"
    value: float


class IndicatorLine(BaseModel):
    """指标图中的一条线"""

    name: str
    label: str
    color: str
    data: List[IndicatorLinePoint] = Field(default_factory=list)


class IndicatorSubChart(BaseModel):
    """单个指标子图"""

    id: str  # "absorption" / "main_in_out" / "cyw"
    label: str  # 中文标题 "主力吸筹"
    height: int = 120
    lines: List[IndicatorLine] = Field(default_factory=list)
    signal_text: str = ""  # 最新信号文本，如 "正T买入 ↑"、"卖出 ↓" 等
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="指标元数据，如MACD的bar_sum/bar_diff，与后端策略算法一致",
    )


class IntradayDataResponse(BaseModel):
    """分时数据完整响应"""

    stock_code: str
    stock_name: str = ""
    date: str
    latest_price: float = Field(default=0.0, description="最新价（用于搜索历史快照回填）")
    change_pct: float = Field(default=0.0, description="涨跌幅（用于搜索历史快照回填）")
    kline_data: List[IntradayKlinePoint] = Field(default_factory=list)
    signals: List[IntradaySignal] = Field(default_factory=list)
    reference_lines: List[ReferenceLine] = Field(default_factory=list)
    indicator_sub_charts: List[IndicatorSubChart] = Field(default_factory=list)
    signal_summary: Dict[str, Any] = Field(default_factory=dict)
    rsi_overbought: float = Field(default=65, description="RSI超买阈值（来自策略配置）")
    rsi_oversold: float = Field(default=20, description="RSI超卖阈值（来自策略配置）")
    mfi_overbought: float = Field(default=80, description="MFI超买阈值（来自策略配置）")
    mfi_oversold: float = Field(default=20, description="MFI超卖阈值（来自策略配置）")
    buy_weights: Dict[str, float] = Field(default_factory=dict, description="买入信号权重配置（来自策略配置）")
    sell_weights: Dict[str, float] = Field(default_factory=dict, description="卖出信号权重配置（来自策略配置）")
    warm_up_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="前一个交易日的分时终值快照，用于开盘时指标预热（含价格、成交量、技术指标终值）",
    )
    warmup_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="预热状态信息: {last_klines_count, prev_date, enabled}",
    )


class SearchHistoryItem(BaseModel):
    """搜索历史条目"""

    id: int
    stock_code: str
    stock_name: str = ""
    date: str = ""
    search_time: str = ""


class SearchHistoryResponse(BaseModel):
    """搜索历史响应"""

    items: List[SearchHistoryItem] = Field(default_factory=list)
    total: int = 0


class SearchHistoryRequest(BaseModel):
    """保存搜索历史请求"""

    stock_code: str
    stock_name: str = ""
    date: str = ""


class DeleteHistoryResponse(BaseModel):
    """删除历史响应"""

    success: bool = True
    message: str = ""


# ── 批量状态查询 ──


class StockSnapshot(BaseModel):
    """单只股票实时快照（腾讯 qt.gtimg.cn 数据源，含五档盘口）"""

    stock_code: str
    stock_name: str = ""
    latest_price: float = 0.0
    change_pct: float = 0.0  # 涨跌幅百分比
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    pre_close: float = 0.0  # 昨收价
    volume: int = 0  # 成交量(股)
    timestamp: str = ""  # "HH:MM:SS"

    # 五档盘口（卖1→卖5价/量，买1→买5价/量，量单位为手）
    ask_prices: List[float] = Field(default_factory=list)  # [卖一价, 卖二价, 卖三价, 卖四价, 卖五价]
    ask_volumes: List[int] = Field(default_factory=list)  # [卖一量, 卖二量, 卖三量, 卖四量, 卖五量] (手)
    bid_prices: List[float] = Field(default_factory=list)  # [买一价, 买二价, 买三价, 买四价, 买五价]
    bid_volumes: List[int] = Field(default_factory=list)  # [买一量, 买二量, 买三量, 买四量, 买五量] (手)

    # 估值指标（可选，腾讯接口附带）
    volume_ratio: Optional[float] = None  # 量比
    turnover_rate: Optional[float] = None  # 换手率(%)
    pe_ratio: Optional[float] = None  # 市盈率
    pb_ratio: Optional[float] = None  # 市净率


class BatchStatusRequest(BaseModel):
    """批量状态查询请求"""

    stock_codes: List[str] = Field(..., description="需要查询的股票代码列表")
    current_code: str = Field("", description="当前展示的股票代码")
    include_signals: bool = Field(False, description="是否对全部股票进行信号检测（用于铃铛通知）")
    skip_kline_fetch: bool = Field(False, description="是否跳过K线拉取（仅返回快照，可视化页面使用）")
    existing_kline_count: int = Field(0, description="前端已有的K线数量，用于增量返回新K线（0表示返回全部）")


class SignalAlert(BaseModel):
    """单只股票的信号告警"""

    stock_code: str
    signal_type: str  # "buy" | "sell"
    trigger_time: str  # 信号触发时间
    price: float  # 信号触发价


class BatchStatusResponse(BaseModel):
    """批量状态查询响应"""

    snapshots: Dict[str, StockSnapshot] = Field(default_factory=dict)
    current_updated: bool = False  # 当前展示股票是否有新数据
    current_full_data: Optional[IntradayDataResponse] = None  # 仅 current_updated=True 时有值
    signal_alerts: Optional[Dict[str, Optional[SignalAlert]]] = Field(
        default=None,
        description="当 include_signals=True 时返回每只股票的最新信号告警",
    )


class SimulatedTradeItem(BaseModel):
    """单笔模拟交易记录（摘要）"""

    buy_time: str = ""
    buy_price: float = 0.0
    sell_time: str = ""
    sell_price: float = 0.0
    return_pct: float = 0.0


class SimulationReportResponse(BaseModel):
    """模拟交易统计报告"""

    stock_code: str = ""
    total_klines: int = 0
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    total_trades: int = 0
    win_trades: int = 0
    lose_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    max_return_pct: float = 0.0
    min_return_pct: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    trades: List[SimulatedTradeItem] = Field(default_factory=list)


class BatchDownloadStatus(BaseModel):
    """批量下载分时数据进度"""

    task_id: str = ""
    status: str = "idle"  # idle | running | completed | cancelled | failed
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    current_code: str = ""
    current_name: str = ""
    elapsed_seconds: float = 0.0
    errors: List[dict] = Field(default_factory=list)  # [{"code": ..., "error": ...}], 最多 20 条
    date: str = ""
    paused: bool = False
    waiting_retry: bool = False
    retry_countdown: int = 0


class FailedListItem(BaseModel):
    """批量下载失败标的项"""

    code: str = ""
    error_msg: str = ""
    retry_count: int = 0


class FailedListResponse(BaseModel):
    """批量下载失败列表响应"""

    date: str = ""
    failed_list: List[FailedListItem] = Field(default_factory=list)
    count: int = 0
