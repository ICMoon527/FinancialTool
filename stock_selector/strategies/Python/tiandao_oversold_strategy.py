# -*- coding: utf-8 -*-
"""
天道超跌反弹策略 - Python 实现。

基于天道指标内置的 td_xg（▲买入）信号选股，
叠加缩量企稳、BBI多空、DDX资金流等辅助条件进行评分排序。
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from indicators.indicators.tiandao import Tiandao
from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class TiandaoOversoldStrategy(StockSelectorStrategy):
    """
    天道超跌反弹策略。

    使用天道指标内置的 td_xg（▲买入）信号作为核心选股条件：
    - td_xg = 金钻趋势 > HIGH（全天价格在支撑位下方） AND 回调买（VAR23动量反转）
    - 叠加缩量企稳、BBI多空、DDX资金流、反弹力度、金钻起涨等辅助条件评分。
    """

    # 子类可通过覆盖此标志切换核心条件：True=td_xg==1, False=K线触及金钻趋势线
    _use_xg_core_condition: bool = False

    def __init__(self):
        metadata = StrategyMetadata(
            id="tiandao_oversold",
            name="tiandao_oversold",
            display_name="天道超跌反弹策略",
            description="基于天道指标识别上升趋势中部分超跌的反弹机会，综合金钻趋势线、金牛2趋势线等条件",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
        super().__init__(metadata)
        self.indicator = Tiandao()

    def select(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        daily_data: Optional[pd.DataFrame] = None,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
        require_jinzuan_above_jinniu2: bool = False,
    ) -> StrategyMatch:
        """
        对单只股票执行天道超跌反弹策略。

        Args:
            stock_code: 要分析的股票代码
            stock_name: 可选的股票名称
            daily_data: 可选的预加载日线数据
            precomputed_metrics: 可选的预计算指标
            require_jinzuan_above_jinniu2: 是否要求金钻趋势线在金牛2线上方（回测中启用，选股页面不启用）

        Returns:
            StrategyMatch 结果对象
        """
        match_details = {}
        conditions_met = []
        conditions_failed = []
        total_score = 0.0
        max_score = 100.0

        # 核心条件标志，初始化为 False
        core_xg_signal = False

        try:
            if self._data_provider:
                realtime_quote = self._data_provider.get_realtime_quote(stock_code)
                if daily_data is None or not isinstance(daily_data, pd.DataFrame) or daily_data.empty:
                    raw_result = self._data_provider.get_daily_data(stock_code, days=120)
                    if isinstance(raw_result, tuple) and len(raw_result) == 2:
                        daily_data, data_source = raw_result
                    else:
                        daily_data = raw_result  # type: ignore[assignment]
                        data_source = "unknown"
                else:
                    data_source = "precomputed"

                match_details["realtime_quote"] = {}
                match_details["indicator_results"] = {}
                match_details["conditions"] = {}
                match_details["data_source"] = data_source

                if realtime_quote:
                    price = getattr(realtime_quote, "price", None)
                    match_details["realtime_quote"] = {"price": price}

                if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
                    daily_data = daily_data.copy()
                    if "close" in daily_data.columns:
                        daily_data = daily_data.rename(
                            columns={
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume",
                            }
                        )

                    if "amount" in daily_data.columns:
                        daily_data = daily_data.rename(columns={"amount": "Amount"})

                    try:
                        indicator_result = self.indicator.calculate(daily_data)
                        match_details["indicator_results"] = (
                            indicator_result.iloc[-1].to_dict() if len(indicator_result) > 0 else {}
                        )

                        if len(indicator_result) > 0:
                            latest = indicator_result.iloc[-1]
                            latest_raw = daily_data.iloc[-1]

                            # 获取当日关键数据
                            low_val = float(latest_raw.get("Low", 0))
                            close_val = float(latest_raw.get("Close", 0))
                            open_val = float(latest_raw.get("Open", 0))
                            volume_val = float(latest_raw.get("Volume", 0))

                            # 获取天道指标值
                            td_jinzuan = float(latest.get("td_jinzuan", 0))
                            td_jinniu2 = float(latest.get("td_jinniu2", 999999))
                            td_bbi = float(latest.get("td_bbi", 0))
                            td_ddx = float(latest.get("td_ddx", 0))
                            td_xg = int(latest.get("td_xg", 0))
                            td_xg2 = int(latest.get("td_xg2", 0))

                            # 计算5日均量
                            vol_col = daily_data["Volume"]
                            vol_ma5 = float(vol_col.rolling(5).mean().iloc[-1])  # type: ignore[union-attr]

                            # ========== 除权跳空检测 ==========
                            # 检测当日开盘价是否相比前一日收盘价大幅跳空（除权除息导致）。
                            # 主板跌停板为 -10%，因此阈值设为 -10.5% 以排除正常跌停，仅捕获除权除息事件。
                            prev_close = 0.0
                            if len(daily_data) >= 2:
                                prev_close = float(daily_data.iloc[-2].get("Close", 0))
                                if prev_close > 0:
                                    gap_ratio = (open_val - prev_close) / prev_close
                                    gap_threshold = -0.105
                                    is_gap_event = gap_ratio < gap_threshold
                                else:
                                    gap_ratio = 0.0
                                    is_gap_event = False
                            else:
                                gap_ratio = 0.0
                                is_gap_event = False

                            match_details["conditions"]["gap_detection"] = {
                                "passed": not is_gap_event,
                                "open": round(open_val, 3),
                                "prev_close": round(prev_close, 3) if len(daily_data) >= 2 else None,
                                "gap_ratio": round(gap_ratio, 4),
                            }

                            # ========== 核心条件判断 ==========
                            if is_gap_event:
                                # 记录跳空事件
                                gap_msg = (
                                    f"除权跳空(Open={open_val:.2f} PrevClose={prev_close:.2f} " f"Gap={gap_ratio:.1%})"
                                )
                                conditions_failed.append(gap_msg)

                                # 尝试从 API 获取前复权数据并更新数据库
                                refresh_ok = self._refresh_adjusted_data(stock_code)
                                if refresh_ok:
                                    # 数据修复成功，重新获取前复权数据并重新评估
                                    raw_result = self._data_provider.get_daily_data(stock_code, days=120)
                                    if isinstance(raw_result, tuple) and len(raw_result) == 2:
                                        daily_data, _ = raw_result
                                    else:
                                        daily_data = raw_result  # type: ignore[assignment]

                                    if (
                                        daily_data is not None
                                        and isinstance(daily_data, pd.DataFrame)
                                        and not daily_data.empty
                                    ):
                                        daily_data = daily_data.copy()
                                        if "close" in daily_data.columns:
                                            daily_data = daily_data.rename(
                                                columns={
                                                    "open": "Open",
                                                    "high": "High",
                                                    "low": "Low",
                                                    "close": "Close",
                                                    "volume": "Volume",
                                                }
                                            )
                                        if "amount" in daily_data.columns:
                                            daily_data = daily_data.rename(columns={"amount": "Amount"})

                                        # 使用前复权数据重新计算指标
                                        indicator_result = self.indicator.calculate(daily_data)
                                        if len(indicator_result) > 0:
                                            latest = indicator_result.iloc[-1]
                                            latest_raw = daily_data.iloc[-1]

                                            low_val = float(latest_raw.get("Low", 0))
                                            close_val = float(latest_raw.get("Close", 0))
                                            open_val = float(latest_raw.get("Open", 0))
                                            volume_val = float(latest_raw.get("Volume", 0))

                                            td_jinzuan = float(latest.get("td_jinzuan", 0))
                                            td_jinniu2 = float(latest.get("td_jinniu2", 999999))
                                            td_bbi = float(latest.get("td_bbi", 0))
                                            td_ddx = float(latest.get("td_ddx", 0))
                                            td_xg = int(latest.get("td_xg", 0))
                                            td_xg2 = int(latest.get("td_xg2", 0))

                                            vol_col = daily_data["Volume"]
                                            vol_ma5 = float(vol_col.rolling(5).mean().iloc[-1])  # type: ignore[union-attr]

                                            data_source = "akshare_qfq"
                                            conditions_met.append("除权数据已刷新为前复权")
                                            is_gap_event = False  # 使用修正数据正常评估

                                            # 更新 match_details 中的指标结果
                                            match_details["indicator_results"] = latest.to_dict()
                                            match_details["data_source"] = data_source
                                            match_details["conditions"]["gap_detection"] = {
                                                "passed": True,
                                                "open": round(open_val, 3),
                                                "prev_close": round(prev_close, 3),
                                                "gap_ratio": round(gap_ratio, 4),
                                                "data_refreshed": True,
                                            }
                                        else:
                                            conditions_failed.append("前复权数据刷新后无法计算指标，排除")
                                    else:
                                        conditions_failed.append("前复权数据刷新后仍无可用数据，排除")
                                else:
                                    conditions_failed.append("前复权数据刷新失败，排除")

                            if not is_gap_event:
                                if require_jinzuan_above_jinniu2:
                                    core_xg_signal = (td_xg == 1) and (td_jinzuan > td_jinniu2)
                                else:
                                    if self._use_xg_core_condition:
                                        core_xg_signal = td_xg == 1
                                    else:
                                        # 选股标签页(默认)：当日K线有部分在金钻趋势线下（实体或下影线等）
                                        core_xg_signal = low_val < td_jinzuan

                            match_details["conditions"]["core_xg"] = {
                                "passed": core_xg_signal,
                                "td_xg": td_xg,
                                "td_jinzuan": round(td_jinzuan, 3),
                                "td_jinniu2": round(td_jinniu2, 3),
                                "low": round(low_val, 3),
                            }

                            if core_xg_signal:
                                if require_jinzuan_above_jinniu2:
                                    core_msg = "td_xg买入信号+金钻>金牛2(基础60分)"
                                elif self._use_xg_core_condition:
                                    core_msg = "td_xg买入信号(基础60分)"
                                else:
                                    core_msg = "K线触及金钻趋势线(基础60分)"
                                conditions_met.append(core_msg)
                                total_score += 60

                                # ========== 辅助加分条件 ==========

                                # 条件3：缩量企稳
                                cond_shrink = (volume_val < vol_ma5) and (close_val >= open_val)
                                match_details["conditions"]["shrink_stabilize"] = {
                                    "passed": cond_shrink,
                                    "volume": volume_val,
                                    "vol_ma5": round(vol_ma5, 1),
                                    "close": round(close_val, 3),
                                    "open": round(open_val, 3),
                                }
                                if cond_shrink:
                                    conditions_met.append("缩量企稳(+10)")
                                    total_score += 10

                                # 条件4：BBI多空确认
                                cond_bbi = close_val > td_bbi
                                match_details["conditions"]["bbi_confirm"] = {
                                    "passed": cond_bbi,
                                    "close": round(close_val, 3),
                                    "td_bbi": round(td_bbi, 3),
                                }
                                if cond_bbi:
                                    conditions_met.append("BBI多空确认(+10)")
                                    total_score += 10

                                # 条件5：DDX资金流确认
                                cond_ddx = td_ddx > 0
                                match_details["conditions"]["ddx_inflow"] = {
                                    "passed": cond_ddx,
                                    "td_ddx": round(td_ddx, 4),
                                }
                                if cond_ddx:
                                    conditions_met.append("DDX流入(+10)")
                                    total_score += 10

                                # 条件6：收盘反弹力度
                                if low_val > 0:
                                    rebound_ratio = (close_val - low_val) / low_val
                                else:
                                    rebound_ratio = 0.0
                                cond_rebound = rebound_ratio > 0.02
                                match_details["conditions"]["rebound_strength"] = {
                                    "passed": cond_rebound,
                                    "rebound_ratio": round(rebound_ratio, 4),
                                    "close": round(close_val, 3),
                                    "low": round(low_val, 3),
                                }
                                if cond_rebound:
                                    conditions_met.append("反弹力度(+5)")
                                    total_score += 5

                                # 条件7：金钻起涨共振
                                cond_xg2 = td_xg2 == 1
                                match_details["conditions"]["jinzuan_xg2"] = {
                                    "passed": cond_xg2,
                                    "td_xg2": td_xg2,
                                }
                                if cond_xg2:
                                    conditions_met.append("金钻起涨(+5)")
                                    total_score += 5
                            else:
                                # 核心条件不满足，记录失败原因
                                # 若是除权跳空导致，不重复追加核心条件失败消息（已通过 gap_detection 说明）
                                if not is_gap_event:
                                    if not core_xg_signal:
                                        conditions_failed.append(
                                            f"td_xg信号未触发(xg={td_xg}, 金钻={td_jinzuan:.3f}, Low={low_val:.3f})"
                                        )
                    except Exception as e:
                        logger.warning(f"天道指标计算失败: {e}")
                        conditions_failed.append(f"指标计算错误: {str(e)[:50]}")
        except Exception as e:
            logger.warning(f"执行天道超跌反弹策略出错 {stock_code}: {e}")
            conditions_failed.append(f"策略执行错误: {str(e)[:50]}")

        raw_score = min(total_score, max_score)
        # 直接使用 td_xg 核心信号判断匹配
        matched = core_xg_signal

        # 计算主力控盘度
        control_degree = None
        if daily_data is not None and isinstance(daily_data, pd.DataFrame) and not daily_data.empty:
            try:
                close_col = "Close" if "Close" in daily_data.columns else "close"
                if close_col in daily_data.columns:
                    ma12 = daily_data[close_col].rolling(window=12).mean()
                    ma36 = daily_data[close_col].rolling(window=36).mean()
                    ma36_prev = ma36.shift(1)
                    cd = (ma12 - ma36_prev) / ma36_prev * 100 + 50
                    latest_cd = cd.iloc[-1]
                    if pd.notna(latest_cd):
                        control_degree = float(latest_cd)
            except Exception:
                pass

        if conditions_met:
            reason_parts = []
            for cond in conditions_met:
                reason_parts.append(cond)
            reason = "、".join(reason_parts) + f" = {raw_score:.0f}分"
        else:
            if conditions_failed:
                reason = f"未匹配({raw_score:.0f}分): " + "; ".join(conditions_failed)
            else:
                reason = f"未匹配({raw_score:.0f}分): 无可用数据"

        match_details["conditions_met"] = conditions_met
        match_details["conditions_failed"] = conditions_failed
        if control_degree is not None:
            match_details["control_degree"] = control_degree

        return self.create_strategy_match(
            raw_score=raw_score,
            matched=matched,
            reason=reason,
            match_details=match_details,
            control_degree=control_degree,
        )

    @staticmethod
    def _get_market_prefix(stock_code: str) -> str:
        """根据股票代码判断市场前缀（腾讯接口需要）。"""
        if stock_code.startswith("6"):
            return "sh"
        elif stock_code.startswith(("0", "3")):
            return "sz"
        elif stock_code.startswith(("4", "8")):
            return "bj"
        return "sh"  # fallback

    @staticmethod
    def _refresh_adjusted_data(stock_code: str) -> bool:
        """检测到除权跳空后，从 AKShare 腾讯接口获取前复权数据并更新数据库。

        腾讯接口比东方财富更稳定，支持 qfq（前复权）参数。

        返回 True 表示刷新成功，False 表示刷新失败（网络问题等）。
        """
        logger = logging.getLogger(__name__)
        try:
            import akshare as ak
        except ImportError:
            logger.warning(f"[数据修复] akshare 未安装，无法刷新 {stock_code} 的除权数据")
            return False

        try:
            from datetime import date, timedelta

            end_date = date.today().strftime("%Y%m%d")
            start_date = (date.today() - timedelta(days=400)).strftime("%Y%m%d")
            tx_symbol = f"{TiandaoOversoldStrategy._get_market_prefix(stock_code)}{stock_code}"

            # 带指数退避的重试机制
            max_retries = 3
            df = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"[数据修复] 正在从腾讯接口获取 {stock_code} 前复权数据(第{attempt}次尝试)...")
                    df = ak.stock_zh_a_hist_tx(
                        symbol=tx_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    if attempt < max_retries:
                        wait = attempt * 2  # 2s, 4s
                        logger.warning(f"[数据修复] {stock_code} 第{attempt}次请求失败: {e}，{wait}s后重试")
                        time.sleep(wait)
                    else:
                        raise

            if df is None or df.empty:
                logger.warning(f"[数据修复] 腾讯接口返回空数据 {stock_code}")
                return False

            # 腾讯接口返回英文列名，只需重命名换手率列
            # 列名: date, open, close, high, low, volume, turnover, amount
            df = df.rename(columns={"turnover": "turnover_rate"})

            # 单位转换：成交量统一为股（科创板原生为股，其余为手）
            if not tx_symbol.startswith("sh68"):
                df["volume"] = df["volume"] * 100  # 手 → 股
            # 成交额：万元 → 元
            df["amount"] = df["amount"] * 10000

            # 写入数据库（覆盖已有记录）
            from src.storage import get_db

            db = get_db()
            count = db.save_daily_data_bulk(df, stock_code, data_source="akshare_qfq")
            logger.info(f"[数据修复] {stock_code} 前复权数据已更新(腾讯)，共 {count} 条")
            return True

        except Exception as e:
            logger.warning(f"[数据修复] 刷新 {stock_code} 前复权数据失败: {e}")
            return False


@register_strategy
class TiandaoXgBuyStrategy(TiandaoOversoldStrategy):
    """
    天道超跌反弹买入策略。

    与天道超跌反弹策略的区别：仅使用 td_xg（▲买入）信号作为核心选股条件，
    不依赖 K 线是否触及金钻趋势线。
    """

    _use_xg_core_condition: bool = True

    def __init__(self):
        # 先走父类 __init__ 注册 indicator
        super().__init__()
        # 覆盖为子类独有的元数据
        self.metadata = StrategyMetadata(
            id="tiandao_xg_buy",
            name="tiandao_xg_buy",
            display_name="天道超跌反弹买入策略",
            description="仅筛选具有 td_xg（▲买入）信号的标的，不依赖K线是否触及金钻趋势线",
            strategy_type=StrategyType.PYTHON,
            category="trend",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=100.0,
        )
