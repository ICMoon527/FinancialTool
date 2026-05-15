# -*- coding: utf-8 -*-
"""
分时做T策略模块

整合"主力吸筹、主力进出、CYW主力控盘"三大指标，
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
import math
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
    """多指标在某一时刻的状态快照"""

    # 主力吸筹
    absorption_value: float = 0.0
    absorption_active: bool = False

    # 主力出货
    distribution_active: bool = False

    # 主力进出
    main_in_value: float = 50.0
    main_out_value: float = 50.0
    main_in_cross_up: bool = False
    main_in_cross_down: bool = False

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

    # 价格均线关系
    ma5: float = 0.0
    ma20: float = 0.0
    price_above_ma5: bool = False
    price_above_ma20: bool = False
    price_cross_ma5_up: bool = False
    price_cross_ma5_down: bool = False

    # 均价偏离度
    avg_price: float = 0.0
    deviation_pct: float = 0.0
    deviation_oversold: bool = False
    deviation_narrowing: bool = False
    deviation_overbought: bool = False
    deviation_peaking: bool = False

    # MACD
    dif: float = 0.0
    dea: float = 0.0
    macd_bar: float = 0.0
    macd_golden_cross: bool = False
    macd_death_cross: bool = False
    macd_bullish_weakening: bool = False    # MACD多头动能衰减(卖)
    macd_bearish_recovering: bool = False   # MACD空头动能衰竭(买)

    # RSI
    rsi_value: float = 50.0
    rsi_oversold: bool = False
    rsi_overbought: bool = False


@dataclass
class T0Signal:
    """做T交易信号"""

    stock_code: str
    signal_type: str  # "buy" / "sell" / "hold"
    trigger_time: datetime = field(default_factory=datetime.now)
    price: float = 0.0
    score: float = 0.0
    max_score: int = 10
    confidence: float = 0.0
    position_advice: str = ""  # "全仓" / "半仓" / "1/3仓"
    indicator_status: IndicatorSnapshot = field(default_factory=IndicatorSnapshot)
    reasoning: str = ""
    buy_weight_details: List[Dict[str, Any]] = field(default_factory=list)
    sell_weight_details: List[Dict[str, Any]] = field(default_factory=list)
    support_force: float = 0.0
    pressure_force: float = 0.0

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
            "buy_weight_details": self.buy_weight_details,
            "sell_weight_details": self.sell_weight_details,
            "support_force": self.support_force,
            "pressure_force": self.pressure_force,
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

        # 主力出货参数
        dist_cfg = indicator_cfg.get("distribution", {})
        self.dist_llv_period = dist_cfg.get("llv_period", self.abs_llv_period)
        self.dist_hhv_period = dist_cfg.get("hhv_period", self.abs_hhv_period)
        self.dist_var7_period = dist_cfg.get("var7_period", self.abs_var7_period)
        self.dist_filter_threshold = dist_cfg.get("filter_threshold", self.abs_filter_threshold)

        # 主力进出参数
        mio_cfg = indicator_cfg.get("main_in_out", {})
        self.mio_hhv_llv_period = mio_cfg.get("hhv_llv_period", 10)
        self.mio_ema_period1 = mio_cfg.get("ema_period1", 2)
        self.mio_ema_period2 = mio_cfg.get("ema_period2", 3)

        # CYW参数
        cyw_cfg = indicator_cfg.get("cyw", {})
        self.cyw_period = cyw_cfg.get("period", 5)
        self.cyw_ma_period = cyw_cfg.get("ma_period", 5)

        # 量能参数
        volume_cfg = indicator_cfg.get("volume", {})
        self.vol_ma_period = volume_cfg.get("ma_period", 5)
        self.vol_surge_ratio = volume_cfg.get("surge_ratio", 1.5)

        # 价格均线参数
        price_ma_cfg = indicator_cfg.get("price_ma", {})
        self.price_ma5_period = price_ma_cfg.get("ma5_period", 5)
        self.price_ma20_period = price_ma_cfg.get("ma20_period", 20)

        # 均价偏离度参数
        dev_cfg = indicator_cfg.get("avg_price_deviation", {})
        self.dev_oversold_threshold = dev_cfg.get("oversold_threshold", -2.5)
        self.dev_overbought_threshold = dev_cfg.get("overbought_threshold", 2.5)

        # MACD参数
        macd_cfg = indicator_cfg.get("macd", {})
        self.macd_fast = macd_cfg.get("fast_period", 12)
        self.macd_slow = macd_cfg.get("slow_period", 26)
        self.macd_signal = macd_cfg.get("signal_period", 9)

        # RSI参数
        rsi_cfg = indicator_cfg.get("rsi", {})
        self.rsi_period = rsi_cfg.get("period", 14)
        self.rsi_overbought = rsi_cfg.get("overbought", 65)
        self.rsi_oversold = rsi_cfg.get("oversold", 20)

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

    # ---------- 主力出货（镜像公式）----------

    def calc_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算主力出货分时版本（主力吸筹公式的镜像）

        与吸筹公式对称，使用最高价替代最低价，HHV替代LLV，
        检测"创新高时的下跌动能爆发"即主力高位派发行为。

        计算结果存入 "distribution" 列（负值，用于与吸筹共用渲染通道）。
        """
        result = data.copy()
        high = data["High"]
        close = data["Close"]

        var2 = high - high.shift(1)
        abs_var2 = var2.abs()
        sma_abs_var2 = self._sma(abs_var2, 3)
        min_var2_abs = var2.clip(upper=0).abs()
        sma_min_var2 = self._sma(min_var2_abs, 3)

        denominator = sma_min_var2.replace(0, np.nan)
        var3 = (sma_abs_var2 / denominator) * 100
        var3 = var3.fillna(0)

        var4_condition = close * 0.8
        var4_arr = np.where(var4_condition > 0, var3 * 10, var3 / 10)
        var4 = self._ema(pd.Series(var4_arr, index=data.index), 3)

        var5 = self._hhv(high, self.dist_llv_period)
        var6 = self._llv(var4, self.dist_hhv_period)

        var7_arr = np.where(self._hhv(high, self.dist_var7_period) > 0, 1, 0)

        var8_condition = high >= var5
        var8_value = np.where(var8_condition, (var4 + var6 * 2) / 2, 0)
        var8 = self._ema(pd.Series(var8_value, index=data.index), 3) / 618 * var7_arr

        threshold = self.dist_filter_threshold
        var8 = np.where(np.abs(var8) < threshold, 0, var8)

        result["distribution"] = -var8
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



    # ---------- CYW ----------

    def calc_cyw(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算CYW主力控盘指标（自动适配tick数据）

        当检测到High=Low=Close（腾讯1分钟tick数据特征，spread=0占比>80%），
        自动切换为tick版资金流向计算：使用价格涨跌方向×成交量，替代原生MFM公式。
        """
        result = data.copy()

        high_low = data["High"] - data["Low"]
        tick_ratio = (high_low == 0).sum() / max(len(data), 1)
        is_tick_data = tick_ratio > 0.8

        if is_tick_data:
            price_change = data["Close"].diff()
            tick_mfm = np.sign(price_change)
            tick_mfm = tick_mfm.fillna(0)
            tick_mfv = tick_mfm * data["Volume"]
            sum_mfv = tick_mfv.rolling(window=self.cyw_period, min_periods=1).sum()
            sum_vol = data["Volume"].rolling(window=self.cyw_period, min_periods=1).sum().replace(0, np.nan)
            cyw = sum_mfv / sum_vol
            cyw = cyw.fillna(0)
        else:
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

    # ---------- 价格均线关系 ----------

    def calc_price_ma_relation(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算分时价格与均线关系"""
        result = data.copy()
        close = data["Close"]

        ma5 = close.rolling(window=self.price_ma5_period, min_periods=1).mean()
        ma20 = close.rolling(window=self.price_ma20_period, min_periods=1).mean()

        result["ma5"] = ma5
        result["ma20"] = ma20
        result["price_above_ma5"] = close > ma5
        result["price_above_ma20"] = close > ma20
        result["price_cross_ma5_up"] = (close > ma5) & (close.shift(1) <= ma5.shift(1))
        result["price_cross_ma5_down"] = (close < ma5) & (close.shift(1) >= ma5.shift(1))
        return result

    # ---------- 均价偏离度 ----------

    def calc_avg_price_deviation(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算均价偏离度（基于累计成交额/成交量）"""
        result = data.copy()
        close = data["Close"]
        volume = data["Volume"].fillna(0)

        cum_amount = (close * volume).cumsum()
        cum_vol = volume.cumsum().replace(0, np.nan)
        avg_price = cum_amount / cum_vol
        avg_price = avg_price.fillna(close)

        deviation_pct = (close - avg_price) / avg_price * 100

        result["avg_price"] = avg_price
        result["deviation_pct"] = deviation_pct
        result["deviation_oversold"] = deviation_pct < self.dev_oversold_threshold
        result["deviation_overbought"] = deviation_pct > self.dev_overbought_threshold
        result["deviation_narrowing"] = (
            result["deviation_oversold"] & (deviation_pct > deviation_pct.shift(1))
        )
        result["deviation_peaking"] = (
            result["deviation_overbought"] & (deviation_pct < deviation_pct.shift(1))
        )
        return result

    # ---------- MACD ----------

    def calc_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标（复用 indicators.indicators.macd.MACD 类）"""
        from indicators.indicators.macd import MACD
        macd = MACD(fast_period=self.macd_fast, slow_period=self.macd_slow, signal_period=self.macd_signal)
        result = macd.calculate(data)
        result["macd_golden_cross"] = result["golden_cross"]
        result["macd_death_cross"] = result["death_cross"]
        return result

    # ---------- RSI ----------

    def calc_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标（复用 indicators.indicators.rsi.RSI 类）"""
        from indicators.indicators.rsi import RSI
        rsi = RSI(period=self.rsi_period, overbought=self.rsi_overbought, oversold=self.rsi_oversold)
        result = rsi.calculate(data)
        result["rsi_overbought"] = result["overbought"]
        result["rsi_oversold"] = result["oversold"]
        return result

    # ---------- 综合计算 ----------

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算全部指标，返回包含所有指标列的 DataFrame"""
        df = data.copy()
        df = self.calc_absorption(df)
        df = self.calc_distribution(df)
        df = self.calc_macd(df)
        df = self.calc_rsi(df)
        df = self.calc_main_in_out(df)
        df = self.calc_cyw(df)
        df = IntradayIndicatorEngine.calc_volume_surge(df, self.vol_ma_period, self.vol_surge_ratio)
        df = self.calc_price_ma_relation(df)
        df = self.calc_avg_price_deviation(df)

        df["absorption"] = df["absorption"].fillna(0) + df["distribution"].fillna(0)
        return df


# ============================================================
# Task 3: 买卖信号评分与生成
# ============================================================


class SignalEvaluator:
    """基于多指标评分的买卖信号评估器，含引力场模型"""

    BUY_WEIGHTS = {
        "absorption_active": 5,
        "cyw_cross_ma_up": 1,
        "main_in_signal": 1,
        "price_cross_ma5_up": 1,
        "avg_price_oversold_fix": 2,
        "price_above_ma20": 1,
        "volume_surge": 1,
        "macd_golden_cross": 2,
        "macd_bearish_recovering": 5,   # MACD空头动能衰竭(买)
        "rsi_oversold": 5,
    }
    BUY_LABELS = {
        "absorption_active": "主力吸筹活跃",
        "cyw_cross_ma_up": "CYW上穿MA",
        "main_in_signal": "主力进出金叉",
        "price_cross_ma5_up": "价格上穿MA5",
        "avg_price_oversold_fix": "均价超卖修复",
        "price_above_ma20": "价格>MA20趋势",
        "volume_surge": "量能放大",
        "macd_golden_cross": "MACD金叉",
        "macd_bearish_recovering": "MACD空头动能衰竭",
        "rsi_oversold": "RSI超卖",
    }

    SELL_WEIGHTS = {
        "distribution_active": 0,
        "main_out_signal": 0,
        "cyw_cross_ma_down": 0,
        "volume_stagnation": 3,
        "price_cross_ma5_down": 2,
        "avg_price_overbought_fix": 2,
        "macd_death_cross": 2,
        "macd_bullish_weakening": 5,    # MACD多头动能衰减(卖)
        "rsi_overbought": 5,
    }
    SELL_LABELS = {
        "distribution_active": "主力出货活跃",
        "main_out_signal": "主力进出死叉",
        "cyw_cross_ma_down": "CYW下穿MA",
        "volume_stagnation": "放量滞涨",
        "price_cross_ma5_down": "价格下穿MA5",
        "avg_price_overbought_fix": "均价超买回落",
        "macd_death_cross": "MACD死叉",
        "macd_bullish_weakening": "MACD多头动能衰减",
        "rsi_overbought": "RSI超买",
    }

    BUY_THRESHOLDS = {"strong": 6, "medium": 5, "weak": 4}
    BUY_POSITIONS = {"strong": "全仓", "medium": "半仓", "weak": "1/3仓"}
    BUY_CONFIDENCE = {"strong": 0.85, "medium": 0.65, "weak": 0.40}

    SELL_THRESHOLDS = {"strong": 11, "medium": 7, "weak": 4}
    SELL_POSITIONS = {"strong": "全仓卖出", "medium": "半仓卖出", "weak": "1/3仓卖出"}
    SELL_CONFIDENCE = {"strong": 0.85, "medium": 0.65, "weak": 0.40}

    # 引力场参数（连续化版本）
    GRAVITY_ENABLED = True
    GRAVITY_DECAY_SIGMA = 0.75
    GRAVITY_SMOOTH_WIDTH = 0.20
    GRAVITY_MAX_ADJUSTMENT = 2.0
    GRAVITY_TANH_SCALE = 1.5
    GRAVITY_LINE_WEIGHTS = {"main_trading": 3, "ma_line": 2, "extreme": 2, "chip_zone": 2, "prev_close": 1.5}
    MAIN_TRADING_IDS = {"attack_line", "operation_line", "defense_line"}
    MA_LINE_IDS = {"ma5", "ma10", "ma20"}
    EXTREME_IDS = {"hhv_30", "llv_30"}
    CHIP_IDS = {"chip_upper", "chip_lower"}
    PREV_CLOSE_IDS = {"prev_close"}

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

        gravity_cfg = signal_cfg.get("gravity", {})
        self.GRAVITY_ENABLED = gravity_cfg.get("enabled", self.GRAVITY_ENABLED)
        self.GRAVITY_DECAY_SIGMA = gravity_cfg.get("decay_sigma", self.GRAVITY_DECAY_SIGMA)
        self.GRAVITY_SMOOTH_WIDTH = gravity_cfg.get("smooth_width", self.GRAVITY_SMOOTH_WIDTH)
        self.GRAVITY_MAX_ADJUSTMENT = gravity_cfg.get("max_adjustment", self.GRAVITY_MAX_ADJUSTMENT)
        self.GRAVITY_TANH_SCALE = gravity_cfg.get("tanh_scale", self.GRAVITY_TANH_SCALE)
        self.GRAVITY_LINE_WEIGHTS = gravity_cfg.get("weights", self.GRAVITY_LINE_WEIGHTS)

        operation_cfg = cfg.get("operation", {})
        self.signal_cooldown_bars = operation_cfg.get("signal_cooldown_bars", 5)

        self._last_buy_bar: int = -999
        self._last_sell_bar: int = -999
        self._current_bar: int = 0

    # ── 引力场模型 ──

    def _classify_line_weight(self, line_id: str) -> float:
        """根据参考线 id 获取引力权重"""
        if line_id in self.MAIN_TRADING_IDS:
            return self.GRAVITY_LINE_WEIGHTS["main_trading"]
        if line_id in self.MA_LINE_IDS:
            return self.GRAVITY_LINE_WEIGHTS["ma_line"]
        if line_id in self.EXTREME_IDS:
            return self.GRAVITY_LINE_WEIGHTS["extreme"]
        if line_id in self.CHIP_IDS:
            return self.GRAVITY_LINE_WEIGHTS["chip_zone"]
        if line_id in self.PREV_CLOSE_IDS:
            return self.GRAVITY_LINE_WEIGHTS["prev_close"]
        return 1.0

    def compute_gravity(self, price: float,
                        reference_lines: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """计算引力场合力（连续化版本）

        每条参考线的贡献通过两个连续函数叠加：
        1. 高斯衰减：影响力随距离平滑衰减，无硬截断
        2. sigmoid 软穿越：支撑/压力角色平滑过渡，无二值翻转

        Args:
            price: 当前价格
            reference_lines: 参考线列表，每条含 price/id

        Returns:
            (net_force, support_force, pressure_force)
        """
        support_force = 0.0
        pressure_force = 0.0

        sigma = self.GRAVITY_DECAY_SIGMA
        smooth_width = self.GRAVITY_SMOOTH_WIDTH

        for rl in reference_lines:
            line_price = rl.get("price", 0)
            line_id = rl.get("id", "")
            if line_price <= 0:
                continue

            d_pct = abs(price - line_price) / price * 100

            raw_strength = math.exp(-(d_pct ** 2) / (2 * sigma ** 2))
            line_w = self._classify_line_weight(line_id)
            strength = raw_strength * line_w

            rel_diff = (line_price - price) / price * 100
            support_ratio = 1.0 / (1.0 + math.exp(-rel_diff / smooth_width))
            pressure_ratio = 1.0 - support_ratio

            support_force += strength * support_ratio
            pressure_force += strength * pressure_ratio

        net_force = support_force - pressure_force
        return round(net_force, 4), round(support_force, 4), round(pressure_force, 4)

    def _apply_gravity_adjustment(self, net_force: float, signal_type: str) -> Tuple[float, str]:
        """根据净力连续调整评分（tanh 平滑版本）

        Returns:
            (adjustment, description)
        """
        if abs(net_force) < 0.01:
            return 0.0, ""

        raw_adj = self.GRAVITY_MAX_ADJUSTMENT * math.tanh(net_force / self.GRAVITY_TANH_SCALE)

        if signal_type == "buy":
            adj = round(raw_adj, 2)
            if adj > 0:
                return adj, f"多头引力(净力{net_force:.2f})+{adj:.1f}分"
            else:
                return adj, f"空头引力(净力{net_force:.2f}){adj:.1f}分"
        else:  # sell
            adj = round(-raw_adj, 2)
            if adj > 0:
                return adj, f"空头引力(净力{net_force:.2f})+{adj:.1f}分"
            else:
                return adj, f"多头引力(净力{net_force:.2f}){adj:.1f}分"

    # ── 信号评估 ──

    def _get_level(self, score: float, thresholds: Dict[str, int]) -> str:
        if score >= thresholds["strong"]:
            return "strong"
        elif score >= thresholds["medium"]:
            return "medium"
        elif score >= thresholds["weak"]:
            return "weak"
        return "none"

    def evaluate_buy(self, status: IndicatorSnapshot,
                     reference_lines: Optional[List[Dict[str, Any]]] = None,
                     price: float = 0.0) -> Tuple[float, str, str, float, List[Dict[str, Any]]]:
        """评估买入信号

        主力吸筹活跃为买入必备条件，吸筹不活跃时直接返回无信号。

        Returns:
            (score, level, position_advice, confidence, weight_details)
        """
        score = 0.0
        weight_details: List[Dict[str, Any]] = []

        if not status.absorption_active:
            weight_details.append({
                "key": "absorption_required",
                "label": "主力吸筹(必备条件)",
                "weight": 0,
                "triggered": False,
                "score": 0,
            })
            return 0.0, "none", "", 0.0, weight_details

        for key, weight in self.BUY_WEIGHTS.items():
            triggered = False
            if key == "absorption_active":
                triggered = status.absorption_active
            elif key == "cyw_cross_ma_up":
                triggered = status.cyw_cross_ma_up
            elif key == "main_in_signal":
                triggered = status.main_in_cross_up
            elif key == "price_cross_ma5_up":
                triggered = status.price_cross_ma5_up
            elif key == "avg_price_oversold_fix":
                triggered = status.deviation_oversold and status.deviation_narrowing
            elif key == "price_above_ma20":
                triggered = status.price_above_ma20
            elif key == "volume_surge":
                triggered = status.volume_surge
            elif key == "macd_golden_cross":
                triggered = status.macd_golden_cross
            elif key == "macd_bearish_recovering":
                triggered = status.macd_bearish_recovering
            elif key == "rsi_oversold":
                triggered = status.rsi_oversold

            score += weight if triggered else 0
            weight_details.append({
                "key": key,
                "label": self.BUY_LABELS.get(key, key),
                "weight": weight,
                "triggered": triggered,
                "score": weight if triggered else 0,
            })

        gravity_adj = 0.0
        gravity_desc = ""
        support_f = pressure_f = net_f = 0.0
        if self.GRAVITY_ENABLED and reference_lines and price > 0:
            net_f, support_f, pressure_f = self.compute_gravity(price, reference_lines)
            gravity_adj, gravity_desc = self._apply_gravity_adjustment(net_f, "buy")

        score += gravity_adj
        weight_details.append({
            "key": "gravity",
            "label": f"引力场({gravity_desc})" if gravity_desc else "引力场",
            "weight": gravity_adj if gravity_adj > 0 else 0,
            "triggered": abs(gravity_adj) > 0.001,
            "score": gravity_adj,
            "support_force": support_f,
            "pressure_force": pressure_f,
            "net_force": net_f,
        })

        level = self._get_level(score, self.BUY_THRESHOLDS)
        position_advice = self.BUY_POSITIONS.get(level, "")
        confidence = self.BUY_CONFIDENCE.get(level, 0.0)

        if level == "none":
            return score, level, "", 0.0, weight_details

        return score, level, position_advice, confidence, weight_details

    def evaluate_sell(self, status: IndicatorSnapshot,
                      reference_lines: Optional[List[Dict[str, Any]]] = None,
                      price: float = 0.0) -> Tuple[float, str, str, float, List[Dict[str, Any]]]:
        """评估卖出信号

        Returns:
            (score, level, position_advice, confidence, weight_details)
        """
        score = 0.0
        weight_details: List[Dict[str, Any]] = []

        if not status.distribution_active:
            weight_details.append({
                "key": "distribution_required",
                "label": "主力出货(必备条件)",
                "weight": 0,
                "triggered": False,
                "score": 0,
            })
            return 0.0, "none", "", 0.0, weight_details

        for key, weight in self.SELL_WEIGHTS.items():
            triggered = False
            if key == "distribution_active":
                triggered = status.distribution_active
            elif key == "main_out_signal":
                triggered = status.main_in_cross_down
            elif key == "cyw_cross_ma_down":
                triggered = status.cyw_cross_ma_down
            elif key == "volume_stagnation":
                triggered = status.volume_surge
            elif key == "price_cross_ma5_down":
                triggered = status.price_cross_ma5_down
            elif key == "avg_price_overbought_fix":
                triggered = status.deviation_overbought and status.deviation_peaking
            elif key == "macd_death_cross":
                triggered = status.macd_death_cross
            elif key == "macd_bullish_weakening":
                triggered = status.macd_bullish_weakening
            elif key == "rsi_overbought":
                triggered = status.rsi_overbought

            score += weight if triggered else 0
            weight_details.append({
                "key": key,
                "label": self.SELL_LABELS.get(key, key),
                "weight": weight,
                "triggered": triggered,
                "score": weight if triggered else 0,
            })

        gravity_adj = 0.0
        gravity_desc = ""
        support_f = pressure_f = net_f = 0.0
        if self.GRAVITY_ENABLED and reference_lines and price > 0:
            net_f, support_f, pressure_f = self.compute_gravity(price, reference_lines)
            gravity_adj, gravity_desc = self._apply_gravity_adjustment(net_f, "sell")

        score += gravity_adj
        weight_details.append({
            "key": "gravity",
            "label": f"引力场({gravity_desc})" if gravity_desc else "引力场",
            "weight": abs(gravity_adj) if gravity_adj > 0 else 0,
            "triggered": abs(gravity_adj) > 0.001,
            "score": gravity_adj,
            "support_force": support_f,
            "pressure_force": pressure_f,
            "net_force": net_f,
        })

        level = self._get_level(score, self.SELL_THRESHOLDS)
        position_advice = self.SELL_POSITIONS.get(level, "")
        confidence = self.SELL_CONFIDENCE.get(level, 0.0)

        if level == "none":
            return score, level, "", 0.0, weight_details

        return score, level, position_advice, confidence, weight_details

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
            "cyw": {"period": 5, "ma_period": 5},
            "volume": {"ma_period": 5, "surge_ratio": 1.5},
            "price_ma": {"ma5_period": 5, "ma20_period": 20},
            "avg_price_deviation": {"oversold_threshold": -2.5, "overbought_threshold": 2.5},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "rsi": {"period": 14, "overbought": 65, "oversold": 20},
        },
        "signals": {
            "buy": {
                "thresholds": {"strong": 9, "medium": 7, "weak": 5},
                "positions": {"strong": "全仓", "medium": "半仓", "weak": "1/3仓"},
                "confidence": {"strong": 0.85, "medium": 0.65, "weak": 0.40},
                "weights": {
                    "absorption_active": 5,
                    "cyw_cross_ma_up": 1,
                    "main_in_signal": 1,
                    "price_cross_ma5_up": 1,
                    "avg_price_oversold_fix": 2,
                    "price_above_ma20": 1,
                    "volume_surge": 1,
                    "macd_golden_cross": 2,
                    "macd_bearish_recovering": 5,
                    "rsi_oversold": 5,
                },
            },
            "sell": {
                "thresholds": {"strong": 11, "medium": 7, "weak": 5},
                "positions": {"strong": "全仓卖出", "medium": "半仓卖出", "weak": "1/3仓卖出"},
                "confidence": {"strong": 0.85, "medium": 0.65, "weak": 0.40},
                "weights": {
                    "distribution_active": 0,
                    "main_out_signal": 0,
                    "cyw_cross_ma_down": 0,
                    "volume_stagnation": 3,
                    "price_cross_ma5_down": 2,
                    "avg_price_overbought_fix": 2,
                    "macd_death_cross": 2,
                    "macd_bullish_weakening": 5,
                    "rsi_overbought": 5,
                },
            },
            "gravity": {
                "enabled": True,
                "decay_sigma": 0.75,
                "smooth_width": 0.20,
                "max_adjustment": 2.0,
                "tanh_scale": 1.5,
                "weights": {"main_trading": 3, "ma_line": 2, "extreme": 2, "chip_zone": 2, "prev_close": 1.5},
            },
        },
        "operation": {"signal_cooldown_bars": 0, "log_signals": True},
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
            absorption_active=float(row.get("absorption", 0)) > 0,
            distribution_active=float(row.get("absorption", 0)) < 0,
            main_in_value=float(row.get("main_in", 50)),
            main_out_value=float(row.get("main_out", 50)),
            main_in_cross_up=bool(row.get("main_in_signal", False)),
            main_in_cross_down=bool(row.get("main_out_signal", False)),
            cyw_value=float(row.get("CYW", 0)),
            cyw_ma=float(row.get("CYW_MA", 0)),
            cyw_positive=bool(row.get("CYW_positive", False)),
            cyw_rising=bool(row.get("CYW_rising", False)),
            cyw_cross_ma_up=bool(row.get("CYW_cross_ma_up", False)),
            cyw_cross_ma_down=bool(row.get("CYW_cross_ma_down", False)),
            volume_surge=bool(row.get("volume_surge", False)),
            volume_shrink=bool(row.get("volume_shrink", False)),
            ma5=float(row.get("ma5", 0)),
            ma20=float(row.get("ma20", 0)),
            price_above_ma5=bool(row.get("price_above_ma5", False)),
            price_above_ma20=bool(row.get("price_above_ma20", False)),
            price_cross_ma5_up=bool(row.get("price_cross_ma5_up", False)),
            price_cross_ma5_down=bool(row.get("price_cross_ma5_down", False)),
            avg_price=float(row.get("avg_price", 0)),
            deviation_pct=float(row.get("deviation_pct", 0)),
            deviation_oversold=bool(row.get("deviation_oversold", False)),
            deviation_narrowing=bool(row.get("deviation_narrowing", False)),
            deviation_overbought=bool(row.get("deviation_overbought", False)),
            deviation_peaking=bool(row.get("deviation_peaking", False)),
            dif=float(row.get("DIF", 0)),
            dea=float(row.get("DEA", 0)),
            macd_bar=float(row.get("MACD_Bar", 0)),
            macd_golden_cross=bool(row.get("macd_golden_cross", False)),
            macd_death_cross=bool(row.get("macd_death_cross", False)),
            macd_bullish_weakening=(
                float(row.get("DIF", 0)) > float(row.get("DEA", 0))
                and float(row.get("MACD_Bar_Sum", 0)) >= 0
                and (float(row.get("MACD_Bar_Diff", 0)) if not pd.isna(row.get("MACD_Bar_Diff")) else 0) >= 0
            ),
            macd_bearish_recovering=(
                float(row.get("DIF", 0)) < float(row.get("DEA", 0))
                and float(row.get("MACD_Bar_Sum", 0)) <= 0
                and (float(row.get("MACD_Bar_Diff", 0)) if not pd.isna(row.get("MACD_Bar_Diff")) else 0) <= 0
            ),
            rsi_value=float(row.get("RSI", 50)),
            rsi_oversold=bool(row.get("rsi_oversold", False)),
            rsi_overbought=bool(row.get("rsi_overbought", False)),
        )

    def feed_kline(self, kline: Dict[str, Any],
                   reference_lines: Optional[List[Dict[str, Any]]] = None) -> Optional[T0Signal]:
        """
        推送一根K线数据，返回可能触发的信号

        Args:
            kline: 单根K线字典，需包含 Open, High, Low, Close, Volume
            reference_lines: 日线级参考线列表（用于引力场模型）

        Returns:
            触发的 T0Signal 或 None
        """
        success = self.buffer.append(kline)
        if not success:
            return None

        df = self.engine.calculate_all(self.buffer.data)
        if df.empty:
            return None

        latest = df.iloc[-1]
        snapshot = self._build_snapshot(latest)
        price = self.buffer.get_latest_price() or 0.0

        self.evaluator._current_bar = self.buffer.length

        ref_lines = reference_lines or []

        # 先检查卖出信号（优先止盈）
        sell_score, sell_level, sell_pos, sell_conf, sell_details = self.evaluator.evaluate_sell(
            snapshot, ref_lines, price
        )
        if sell_level != "none" and self.evaluator.check_cooldown("sell"):
            self.evaluator.record_signal("sell")
            # 计算引力详情
            net_f, sup_f, pre_f = self.evaluator.compute_gravity(price, ref_lines) if ref_lines else (0, 0, 0)
            signal = T0Signal(
                stock_code=self.stock_code,
                signal_type="sell",
                trigger_time=self.buffer.get_latest_time() or datetime.now(),
                price=price,
                score=sell_score,
                max_score=sum(self.evaluator.SELL_WEIGHTS.values()) + 2,
                confidence=sell_conf,
                position_advice=sell_pos,
                indicator_status=snapshot,
                reasoning=f"卖出信号({sell_level}级)，得分{sell_score}/{sum(self.evaluator.SELL_WEIGHTS.values())+2}",
                sell_weight_details=sell_details,
                support_force=sup_f,
                pressure_force=pre_f,
            )

            # 同时计算买入权重明细（用于前端展示，即使未触发买入）
            _, _, _, _, buy_details = self.evaluator.evaluate_buy(snapshot, ref_lines, price)
            signal.buy_weight_details = buy_details

            self._emit_signal(signal)

            if self._position is not None:
                buy_price = self._position["price"]
                pnl_pct = (price - buy_price) / buy_price * 100
                logger.debug(f"[模拟交易] 卖出 {self.stock_code} @ {price:.2f}, 收益率 {pnl_pct:+.2f}%")
                self._position = None

            return signal

        # 检查买入信号
        buy_score, buy_level, buy_pos, buy_conf, buy_details = self.evaluator.evaluate_buy(
            snapshot, ref_lines, price
        )
        if buy_level != "none" and self.evaluator.check_cooldown("buy"):
            self.evaluator.record_signal("buy")
            net_f, sup_f, pre_f = self.evaluator.compute_gravity(price, ref_lines) if ref_lines else (0, 0, 0)
            signal = T0Signal(
                stock_code=self.stock_code,
                signal_type="buy",
                trigger_time=self.buffer.get_latest_time() or datetime.now(),
                price=price,
                score=buy_score,
                max_score=sum(self.evaluator.BUY_WEIGHTS.values()) + 2,
                confidence=buy_conf,
                position_advice=buy_pos,
                indicator_status=snapshot,
                reasoning=f"买入信号({buy_level}级)，得分{buy_score}/{sum(self.evaluator.BUY_WEIGHTS.values())+2}",
                buy_weight_details=buy_details,
                support_force=sup_f,
                pressure_force=pre_f,
            )

            _, _, _, _, sell_details = self.evaluator.evaluate_sell(snapshot, ref_lines, price)
            signal.sell_weight_details = sell_details

            if self._position is None:
                logger.debug(f"[模拟交易] 买入 {self.stock_code} @ {price:.2f}")
                self._position = {"price": price, "time": self.buffer.get_latest_time()}

            self._emit_signal(signal)
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
