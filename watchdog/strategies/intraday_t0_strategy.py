# -*- coding: utf-8 -*-
"""
分时做T策略模块

整合"主力吸筹、主力进出、龙虎动力、CYW主力控盘"四大指标，
基于评分制生成分时级别的高抛低吸买卖信号。

使用示例:
    from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy, T0Signal

    strategy = IntradayT0Strategy(stock_code="000001")
    strategy.register_signal_callback(lambda sig: print(sig))

    for kline in minute_klines:
        signal = strategy.feed_kline(kline)
        if signal:
            print(f"信号: {signal.signal_type} @ {signal.price}")
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- 指标计算临时别名，避免循环导入 ----------
# 策略内部直接实现分时版本计算，不依赖 indicators 包的类导入，确保独立运行


# ============================================================
# Task 1: 数据结构定义
# ============================================================


@dataclass
class IndicatorSnapshot:
    """四指标在某一时刻的状态快照"""

    # 主力吸筹
    absorption_value: float = 0.0
    absorption_active: bool = False

    # 主力进出
    main_in_value: float = 50.0
    main_out_value: float = 50.0
    main_in_cross_up: bool = False
    main_in_cross_down: bool = False

    # 龙虎动力
    dominant_power: float = 0.0
    power_growth: bool = False
    power_decay: bool = False
    power_depression: bool = False
    power_recovery: bool = False
    volume_growth: bool = False

    # CYW 主力控盘
    cyw_value: float = 0.0
    cyw_ma: float = 0.0
    cyw_positive: bool = False
    cyw_rising: bool = False
    cyw_cross_ma_up: bool = False
    cyw_cross_ma_down: bool = False

    # 量能
    volume_surge: bool = False
    volume_shrink: bool = False


@dataclass
class T0Signal:
    """做T交易信号"""

    stock_code: str
    signal_type: str  # "buy" / "sell" / "hold"
    trigger_time: datetime = field(default_factory=datetime.now)
    price: float = 0.0
    score: int = 0
    max_score: int = 10
    confidence: float = 0.0
    position_advice: str = ""  # "全仓" / "半仓" / "1/3仓"
    indicator_status: IndicatorSnapshot = field(default_factory=IndicatorSnapshot)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "signal_type": self.signal_type,
            "trigger_time": self.trigger_time.isoformat(),
            "price": self.price,
            "score": self.score,
            "max_score": self.max_score,
            "confidence": self.confidence,
            "position_advice": self.position_advice,
            "reasoning": self.reasoning,
        }

    def __repr__(self) -> str:
        return (
            f"T0Signal({self.signal_type}, code={self.stock_code}, "
            f"price={self.price:.2f}, score={self.score}/{self.max_score}, "
            f"conf={self.confidence:.2f}, pos={self.position_advice})"
        )


# ============================================================
# Task 1: IntradayDataBuffer 滚动窗口数据管理
# ============================================================


class IntradayDataBuffer:
    """分时K线数据滚动窗口缓冲区"""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, max_window: int = 200, kline_freq: str = "1min"):
        """
        Args:
            max_window: 最大保留K线数量
            kline_freq: K线频率，'1min' 或 '5min'
        """
        self.max_window = max_window
        self.kline_freq = kline_freq
        self._data: pd.DataFrame = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        self._timestamps: List[datetime] = []

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def length(self) -> int:
        return len(self._data)

    @property
    def is_ready(self) -> bool:
        return len(self._data) >= 15

    def validate_kline(self, kline: Dict[str, Any]) -> Tuple[bool, str]:
        """校验单根K线数据有效性"""
        for col in self.REQUIRED_COLUMNS:
            if col not in kline:
                return False, f"缺少字段: {col}"
        close_val = kline.get("Close", 0)
        if close_val <= 0:
            return False, f"异常价格 Close={close_val}"
        high_val = kline.get("High", 0)
        low_val = kline.get("Low", 0)
        if high_val < low_val:
            return False, f"High({high_val}) < Low({low_val})"
        volume_val = kline.get("Volume", -1)
        if volume_val < 0:
            return False, f"异常成交量 Volume={volume_val}"
        return True, ""

    def append(self, kline: Dict[str, Any]) -> bool:
        """追加一根K线到缓冲区"""
        valid, error_msg = self.validate_kline(kline)
        if not valid:
            logger.warning(f"K线数据校验失败: {error_msg}, 数据: {kline}")
            return False

        row = {col: kline[col] for col in self.REQUIRED_COLUMNS}
        timestamp = kline.get("timestamp", kline.get("datetime", datetime.now()))

        new_row = pd.DataFrame([row], columns=self.REQUIRED_COLUMNS)
        if self._data.empty:
            self._data = new_row
        else:
            self._data = pd.concat([self._data, new_row], ignore_index=True)

        self._timestamps.append(timestamp)
        self._trim()
        return True

    def _trim(self) -> None:
        """裁剪超出窗口的数据"""
        if len(self._data) > self.max_window:
            excess = len(self._data) - self.max_window
            self._data = self._data.iloc[excess:].reset_index(drop=True)
            self._timestamps = self._timestamps[excess:]

    def get_latest_price(self) -> Optional[float]:
        if self._data.empty:
            return None
        return float(self._data["Close"].iloc[-1])

    def get_latest_time(self) -> Optional[datetime]:
        if not self._timestamps:
            return None
        return self._timestamps[-1]


# ============================================================
# Task 2: 四指标分时计算引擎
# ============================================================


class IntradayIndicatorEngine:
    """分时版本的四指标计算引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        indicator_cfg = cfg.get("indicators", {})

        # 主力吸筹参数
        abs_cfg = indicator_cfg.get("absorption", {})
        self.abs_llv_period = abs_cfg.get("llv_period", 10)
        self.abs_hhv_period = abs_cfg.get("hhv_period", 10)
        self.abs_var7_period = abs_cfg.get("var7_period", 20)
        self.abs_filter_threshold = abs_cfg.get("filter_threshold", 0.5)

        # 主力进出参数
        mio_cfg = indicator_cfg.get("main_in_out", {})
        self.mio_hhv_llv_period = mio_cfg.get("hhv_llv_period", 10)
        self.mio_ema_period1 = mio_cfg.get("ema_period1", 2)
        self.mio_ema_period2 = mio_cfg.get("ema_period2", 3)

        # 龙虎动力参数
        dtp_cfg = indicator_cfg.get("dragon_tiger_power", {})
        self.dtp_ema_period = dtp_cfg.get("ema_period", 4)

        # CYW参数
        cyw_cfg = indicator_cfg.get("cyw", {})
        self.cyw_period = cyw_cfg.get("period", 5)
        self.cyw_ma_period = cyw_cfg.get("ma_period", 5)

        # 量能参数
        volume_cfg = indicator_cfg.get("volume", {})
        self.vol_ma_period = volume_cfg.get("ma_period", 5)
        self.vol_surge_ratio = volume_cfg.get("surge_ratio", 1.5)

    # ---------- 工具函数 ----------

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _llv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).min()

    @staticmethod
    def _hhv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).max()

    # ---------- 主力吸筹 ----------

    def calc_absorption(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算主力吸筹分时版本"""
        result = data.copy()
        low = data["Low"]
        close = data["Close"]

        var2 = low - low.shift(1)
        abs_var2 = var2.abs()
        sma_abs_var2 = self._sma(abs_var2, 3)
        max_var2 = var2.clip(lower=0)
        sma_max_var2 = self._sma(max_var2, 3)

        denominator = sma_max_var2.replace(0, np.nan)
        var3 = (sma_abs_var2 / denominator) * 100
        var3 = var3.fillna(0)

        var4_condition = close * 1.2
        var4_arr = np.where(var4_condition > 0, var3 * 10, var3 / 10)
        var4 = self._ema(pd.Series(var4_arr, index=data.index), 3)

        var5 = self._llv(low, self.abs_llv_period)
        var6 = self._hhv(var4, self.abs_hhv_period)

        var7_arr = np.where(self._llv(low, self.abs_var7_period) > 0, 1, 0)

        var8_condition = low <= var5
        var8_value = np.where(var8_condition, (var4 + var6 * 2) / 2, 0)
        var8 = self._ema(pd.Series(var8_value, index=data.index), 3) / 618 * var7_arr

        threshold = self.abs_filter_threshold
        var8 = np.where(np.abs(var8) < threshold, 0, var8)

        result["absorption"] = var8
        return result

    # ---------- 主力进出 ----------

    def calc_main_in_out(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算主力进出分时版本"""
        result = data.copy()
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        llv_low = self._llv(low, self.mio_hhv_llv_period)
        hhv_high = self._hhv(high, self.mio_hhv_llv_period)
        denom = hhv_high - llv_low
        denom = denom.replace(0, np.nan)

        main_in = (close - llv_low) / denom * 100
        main_out = self._ema(main_in, self.mio_ema_period1)
        in_out_line = self._ema(main_in, self.mio_ema_period2)

        result["main_in"] = main_in
        result["main_out"] = main_out
        result["in_out_line"] = in_out_line
        result["main_in_signal"] = (main_in > main_out) & (main_in.shift(1) <= main_out.shift(1))
        result["main_out_signal"] = (main_in < main_out) & (main_in.shift(1) >= main_out.shift(1))
        return result

    # ---------- 龙虎动力 ----------

    def calc_dragon_tiger_power(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算龙虎动力分时版本"""
        result = data.copy()

        tt = 2 * data["Close"] + data["Open"] + data["High"] + data["Low"]
        ema_tt = self._ema(tt, self.dtp_ema_period)
        ff = 100 * (tt / (ema_tt + 1e-10) - 1)

        result["dominant_power"] = ff
        result["power_growth"] = (ff > 0) & (ff > ff.shift(1))
        result["power_decay"] = (ff > 0) & (ff <= ff.shift(1))
        result["power_depression"] = (ff <= 0) & (ff <= ff.shift(1))
        result["power_recovery"] = (ff <= 0) & (ff > ff.shift(1))

        if "Amount" in data.columns:
            turnover_billion = data["Amount"] / 100000000
        else:
            turnover_billion = data["Volume"] * data["Close"] / 100000000
        ma3 = self._sma(turnover_billion, 2)
        ma20 = self._sma(turnover_billion, 20)
        volume_ratio = self._ema(ma3 / (ma20 + 1e-10), 2)
        result["volume_ratio"] = volume_ratio
        result["volume_growth"] = result["power_growth"] & (volume_ratio >= 1.15)
        return result

    # ---------- CYW ----------

    def calc_cyw(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算CYW主力控盘分时版本"""
        result = data.copy()

        high_low = data["High"] - data["Low"]
        mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / high_low.replace(0, np.nan)
        mfm = mfm.fillna(0)
        mfv = mfm * data["Volume"]
        cyw = mfv.rolling(window=self.cyw_period, min_periods=1).sum() / data["Volume"].rolling(
            window=self.cyw_period, min_periods=1
        ).sum().replace(0, np.nan)
        cyw = cyw.fillna(0)
        cyw_ma = self._sma(cyw, self.cyw_ma_period)

        result["CYW"] = cyw
        result["CYW_MA"] = cyw_ma
        result["CYW_positive"] = cyw > 0
        result["CYW_rising"] = cyw > cyw.shift(1)
        result["CYW_cross_ma_up"] = (cyw > cyw_ma) & (cyw.shift(1) <= cyw_ma.shift(1))
        result["CYW_cross_ma_down"] = (cyw < cyw_ma) & (cyw.shift(1) >= cyw_ma.shift(1))
        return result

    # ---------- 量能 ----------

    @staticmethod
    def calc_volume_surge(data: pd.DataFrame, ma_period: int = 5, surge_ratio: float = 1.5) -> pd.DataFrame:
        """检测量能放大"""
        result = data.copy()
        vol_ma = result["Volume"].rolling(window=ma_period, min_periods=1).mean()
        result["volume_surge"] = result["Volume"] > vol_ma * surge_ratio
        result["volume_shrink"] = result["Volume"] < vol_ma * 0.5
        return result

    # ---------- 综合计算 ----------

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算全部四个指标，返回包含所有指标列的 DataFrame"""
        df = data.copy()
        df = self.calc_absorption(df)
        df = self.calc_main_in_out(df)
        df = self.calc_dragon_tiger_power(df)
        df = self.calc_cyw(df)
        df = IntradayIndicatorEngine.calc_volume_surge(df, self.vol_ma_period, self.vol_surge_ratio)
        return df


# ============================================================
# Task 3: 买卖信号评分与生成
# ============================================================


class SignalEvaluator:
    """基于四指标评分的买卖信号评估器"""

    # 买入评分权重
    BUY_WEIGHTS = {
        "absorption_active": 3,
        "cyw_cross_ma_up": 2,
        "power_recovery": 2,
        "main_in_signal": 2,
        "volume_surge": 1,
    }

    # 卖出评分权重
    SELL_WEIGHTS = {
        "main_out_signal": 2,
        "cyw_cross_ma_down": 2,
        "power_decay_or_depression": 2,
        "absorption_inactive": 1,
        "volume_stagnation": 1,
    }

    # 买入信号分级
    BUY_THRESHOLDS = {"strong": 7, "medium": 5, "weak": 3}
    BUY_POSITIONS = {"strong": "全仓", "medium": "半仓", "weak": "1/3仓"}
    BUY_CONFIDENCE = {"strong": 0.85, "medium": 0.65, "weak": 0.40}

    # 卖出信号分级
    SELL_THRESHOLDS = {"strong": 7, "medium": 5, "weak": 3}
    SELL_POSITIONS = {"strong": "全仓卖出", "medium": "半仓卖出", "weak": "1/3仓卖出"}
    SELL_CONFIDENCE = {"strong": 0.85, "medium": 0.65, "weak": 0.40}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        signal_cfg = cfg.get("signals", {})

        buy_cfg = signal_cfg.get("buy", {})
        self.BUY_THRESHOLDS = buy_cfg.get("thresholds", self.BUY_THRESHOLDS)
        self.BUY_POSITIONS = buy_cfg.get("positions", self.BUY_POSITIONS)
        self.BUY_CONFIDENCE = buy_cfg.get("confidence", self.BUY_CONFIDENCE)
        self.BUY_WEIGHTS = buy_cfg.get("weights", self.BUY_WEIGHTS)

        sell_cfg = signal_cfg.get("sell", {})
        self.SELL_THRESHOLDS = sell_cfg.get("thresholds", self.SELL_THRESHOLDS)
        self.SELL_POSITIONS = sell_cfg.get("positions", self.SELL_POSITIONS)
        self.SELL_CONFIDENCE = sell_cfg.get("confidence", self.SELL_CONFIDENCE)
        self.SELL_WEIGHTS = sell_cfg.get("weights", self.SELL_WEIGHTS)

        operation_cfg = cfg.get("operation", {})
        self.signal_cooldown_bars = operation_cfg.get("signal_cooldown_bars", 5)

        self._last_buy_bar: int = -999
        self._last_sell_bar: int = -999
        self._current_bar: int = 0

    def _get_level(self, score: int, thresholds: Dict[str, int]) -> str:
        if score >= thresholds["strong"]:
            return "strong"
        elif score >= thresholds["medium"]:
            return "medium"
        elif score >= thresholds["weak"]:
            return "weak"
        return "none"

    def evaluate_buy(self, status: IndicatorSnapshot) -> Tuple[int, str, str, float]:
        """评估买入信号

        Returns:
            (score, level, position_advice, confidence)
        """
        score = 0
        details: List[str] = []

        if status.absorption_active:
            score += self.BUY_WEIGHTS["absorption_active"]
            details.append(f"主力吸筹活跃(+{self.BUY_WEIGHTS['absorption_active']})")
        if status.cyw_cross_ma_up:
            score += self.BUY_WEIGHTS["cyw_cross_ma_up"]
            details.append(f"CYW上穿MA(+{self.BUY_WEIGHTS['cyw_cross_ma_up']})")
        if status.power_recovery:
            score += self.BUY_WEIGHTS["power_recovery"]
            details.append(f"龙虎动力复苏(+{self.BUY_WEIGHTS['power_recovery']})")
        if status.main_in_cross_up:
            score += self.BUY_WEIGHTS["main_in_signal"]
            details.append(f"主力进出金叉(+{self.BUY_WEIGHTS['main_in_signal']})")
        if status.volume_surge:
            score += self.BUY_WEIGHTS["volume_surge"]
            details.append(f"量能放大(+{self.BUY_WEIGHTS['volume_surge']})")

        level = self._get_level(score, self.BUY_THRESHOLDS)
        position_advice = self.BUY_POSITIONS.get(level, "")
        confidence = self.BUY_CONFIDENCE.get(level, 0.0)

        if level == "none":
            return score, level, "", 0.0

        return score, level, position_advice, confidence

    def evaluate_sell(self, status: IndicatorSnapshot) -> Tuple[int, str, str, float]:
        """评估卖出信号

        Returns:
            (score, level, position_advice, confidence)
        """
        score = 0
        details: List[str] = []

        if status.main_in_cross_down:
            score += self.SELL_WEIGHTS["main_out_signal"]
            details.append(f"主力进出死叉(+{self.SELL_WEIGHTS['main_out_signal']})")
        if status.cyw_cross_ma_down:
            score += self.SELL_WEIGHTS["cyw_cross_ma_down"]
            details.append(f"CYW下穿MA(+{self.SELL_WEIGHTS['cyw_cross_ma_down']})")
        if status.power_decay or status.power_depression:
            score += self.SELL_WEIGHTS["power_decay_or_depression"]
            details.append(f"龙虎动力衰减/萧条(+{self.SELL_WEIGHTS['power_decay_or_depression']})")
        if not status.absorption_active:
            score += self.SELL_WEIGHTS["absorption_inactive"]
            details.append(f"主力吸筹归零(+{self.SELL_WEIGHTS['absorption_inactive']})")
        if status.volume_surge and not status.power_growth:
            score += self.SELL_WEIGHTS["volume_stagnation"]
            details.append(f"放量滞涨(+{self.SELL_WEIGHTS['volume_stagnation']})")

        level = self._get_level(score, self.SELL_THRESHOLDS)
        position_advice = self.SELL_POSITIONS.get(level, "")
        confidence = self.SELL_CONFIDENCE.get(level, 0.0)

        if level == "none":
            return score, level, "", 0.0

        return score, level, position_advice, confidence

    def check_cooldown(self, signal_type: str) -> bool:
        """检查冷却时间"""
        if signal_type == "buy":
            return (self._current_bar - self._last_buy_bar) >= self.signal_cooldown_bars
        elif signal_type == "sell":
            return (self._current_bar - self._last_sell_bar) >= self.signal_cooldown_bars
        return True

    def record_signal(self, signal_type: str) -> None:
        """记录信号触发时间"""
        if signal_type == "buy":
            self._last_buy_bar = self._current_bar
        elif signal_type == "sell":
            self._last_sell_bar = self._current_bar


# ============================================================
# Task 4: IntradayT0Strategy 主类
# ============================================================


class IntradayT0Strategy:
    """
    分时做T策略主类

    整合数据缓冲、指标计算和信号评分，提供简洁的 feed_kline 接口。
    支持信号回调注册和日志两种输出模式。

    使用示例:
        strategy = IntradayT0Strategy(stock_code="000001")

        @strategy.on_signal
        def handle(signal):
            print(f"收到信号: {signal}")

        for kline in minute_data:
            strategy.feed_kline(kline)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "data": {"max_window": 200, "kline_freq": "1min"},
        "indicators": {
            "absorption": {"llv_period": 10, "hhv_period": 10, "var7_period": 20, "filter_threshold": 0.5},
            "main_in_out": {"hhv_llv_period": 10, "ema_period1": 2, "ema_period2": 3},
            "dragon_tiger_power": {"ema_period": 4},
            "cyw": {"period": 5, "ma_period": 5},
            "volume": {"ma_period": 5, "surge_ratio": 1.5},
        },
        "signals": {
            "buy": {
                "thresholds": {"strong": 7, "medium": 5, "weak": 3},
                "positions": {"strong": "全仓", "medium": "半仓", "weak": "1/3仓"},
                "confidence": {"strong": 0.85, "medium": 0.65, "weak": 0.40},
                "weights": {"absorption_active": 3, "cyw_cross_ma_up": 2, "power_recovery": 2, "main_in_signal": 2, "volume_surge": 1},
            },
            "sell": {
                "thresholds": {"strong": 7, "medium": 5, "weak": 3},
                "positions": {"strong": "全仓卖出", "medium": "半仓卖出", "weak": "1/3仓卖出"},
                "confidence": {"strong": 0.85, "medium": 0.65, "weak": 0.40},
                "weights": {"main_out_signal": 2, "cyw_cross_ma_down": 2, "power_decay_or_depression": 2, "absorption_inactive": 1, "volume_stagnation": 1},
            },
        },
        "operation": {"signal_cooldown_bars": 5, "log_signals": True},
    }

    def __init__(
        self,
        stock_code: str,
        stock_name: str = "",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            config: 策略参数字典（优先级高于配置文件）
            config_path: YAML配置文件路径
        """
        self.stock_code = stock_code
        self.stock_name = stock_name

        # 加载配置
        self.config = self._load_config(config, config_path)

        # 初始化组件
        data_cfg = self.config.get("data", {})
        self.buffer = IntradayDataBuffer(
            max_window=data_cfg.get("max_window", 200),
            kline_freq=data_cfg.get("kline_freq", "1min"),
        )
        self.engine = IntradayIndicatorEngine(self.config)
        self.evaluator = SignalEvaluator(self.config)

        # 信号回调
        self._signal_callbacks: List[Callable[[T0Signal], None]] = []
        self._log_signals = self.config.get("operation", {}).get("log_signals", True)

        # 历史信号记录
        self.signals: List[T0Signal] = []

        # 模拟持仓管理
        self._position: Optional[Dict[str, Any]] = None

    def _load_config(
        self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """加载配置：优先使用传入dict，其次文件，最后默认"""
        if config is not None:
            return config
        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        logger.info("使用默认策略参数配置")
        return self.DEFAULT_CONFIG.copy()

    def on_signal(self, callback: Callable[[T0Signal], None]) -> Callable[[T0Signal], None]:
        """注册信号回调函数（支持装饰器语法）"""
        self._signal_callbacks.append(callback)
        return callback

    def _emit_signal(self, signal: T0Signal) -> None:
        """发送信号"""
        self.signals.append(signal)
        if self._log_signals:
            logger.info(f"[做T信号] {signal}")
        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"信号回调执行失败: {e}")

    def _build_snapshot(self, row: pd.Series) -> IndicatorSnapshot:
        """从 DataFrame 行构建指标快照"""
        return IndicatorSnapshot(
            absorption_value=float(row.get("absorption", 0)),
            absorption_active=bool(row.get("absorption", 0)) != 0,
            main_in_value=float(row.get("main_in", 50)),
            main_out_value=float(row.get("main_out", 50)),
            main_in_cross_up=bool(row.get("main_in_signal", False)),
            main_in_cross_down=bool(row.get("main_out_signal", False)),
            dominant_power=float(row.get("dominant_power", 0)),
            power_growth=bool(row.get("power_growth", False)),
            power_decay=bool(row.get("power_decay", False)),
            power_depression=bool(row.get("power_depression", False)),
            power_recovery=bool(row.get("power_recovery", False)),
            volume_growth=bool(row.get("volume_growth", False)),
            cyw_value=float(row.get("CYW", 0)),
            cyw_ma=float(row.get("CYW_MA", 0)),
            cyw_positive=bool(row.get("CYW_positive", False)),
            cyw_rising=bool(row.get("CYW_rising", False)),
            cyw_cross_ma_up=bool(row.get("CYW_cross_ma_up", False)),
            cyw_cross_ma_down=bool(row.get("CYW_cross_ma_down", False)),
            volume_surge=bool(row.get("volume_surge", False)),
            volume_shrink=bool(row.get("volume_shrink", False)),
        )

    def feed_kline(self, kline: Dict[str, Any]) -> Optional[T0Signal]:
        """
        推送一根K线数据，返回可能触发的信号

        Args:
            kline: 单根K线字典，需包含 Open, High, Low, Close, Volume

        Returns:
            触发的 T0Signal 或 None
        """
        success = self.buffer.append(kline)
        if not success:
            return None

        if not self.buffer.is_ready:
            return None

        # 计算指标
        df = self.engine.calculate_all(self.buffer.data)
        if df.empty:
            return None

        latest = df.iloc[-1]
        snapshot = self._build_snapshot(latest)
        price = self.buffer.get_latest_price() or 0.0

        self.evaluator._current_bar = self.buffer.length

        # 先检查卖出信号（优先止盈）
        sell_score, sell_level, sell_pos, sell_conf = self.evaluator.evaluate_sell(snapshot)
        if sell_level != "none" and self.evaluator.check_cooldown("sell"):
            self.evaluator.record_signal("sell")
            signal = T0Signal(
                stock_code=self.stock_code,
                signal_type="sell",
                trigger_time=self.buffer.get_latest_time() or datetime.now(),
                price=price,
                score=sell_score,
                max_score=9,
                confidence=sell_conf,
                position_advice=sell_pos,
                indicator_status=snapshot,
                reasoning=f"卖出信号({sell_level}级)，得分{sell_score}/9",
            )
            self._emit_signal(signal)

            # 模拟清仓
            if self._position is not None:
                buy_price = self._position["price"]
                pnl_pct = (price - buy_price) / buy_price * 100
                logger.info(f"[模拟交易] 卖出 {self.stock_code} @ {price:.2f}, 收益率 {pnl_pct:+.2f}%")
                self._position = None

            return signal

        # 检查买入信号
        buy_score, buy_level, buy_pos, buy_conf = self.evaluator.evaluate_buy(snapshot)
        if buy_level != "none" and self.evaluator.check_cooldown("buy"):
            self.evaluator.record_signal("buy")
            signal = T0Signal(
                stock_code=self.stock_code,
                signal_type="buy",
                trigger_time=self.buffer.get_latest_time() or datetime.now(),
                price=price,
                score=buy_score,
                max_score=10,
                confidence=buy_conf,
                position_advice=buy_pos,
                indicator_status=snapshot,
                reasoning=f"买入信号({buy_level}级)，得分{buy_score}/10",
            )
            self._emit_signal(signal)

            # 模拟建仓
            if self._position is None:
                self._position = {"price": price, "time": self.buffer.get_latest_time()}
                logger.info(f"[模拟交易] 买入 {self.stock_code} @ {price:.2f}")

            return signal

        return None

    def reset(self) -> None:
        """重置策略状态"""
        self.buffer = IntradayDataBuffer(
            max_window=self.config.get("data", {}).get("max_window", 200),
            kline_freq=self.config.get("data", {}).get("kline_freq", "1min"),
        )
        self.evaluator = SignalEvaluator(self.config)
        self.signals.clear()
        self._position = None
