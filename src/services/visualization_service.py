# -*- coding: utf-8 -*-
"""
===================================
可视化数据服务
===================================

职责：
1. 批量检索数据库中缺失的指标
2. 批量计算指标
3. 批量存入数据库
4. 提供完整的可视化数据
"""

import logging
from datetime import date, timedelta
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from data_provider import DataFetcherManager
from indicators.indicators.banker_control import BankerControl
from indicators.indicators.main_capital_absorption import MainCapitalAbsorption
from indicators.indicators.main_cost import MainCost
from indicators.indicators.main_trading import MainTrading
from src.storage import DatabaseManager, get_db

logger = logging.getLogger(__name__)

# 定义所有支持的指标类型
AVAILABLE_INDICATORS = [
    'banker_control',
    'main_capital_absorption',
    'main_cost',
    'main_trading'
]

# 指标计算器映射
INDICATOR_CALCULATORS = {
    'banker_control': BankerControl,
    'main_capital_absorption': MainCapitalAbsorption,
    'main_cost': MainCost,
    'main_trading': MainTrading
}


class VisualizationService:
    """
    可视化数据服务
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db()

    def get_visualization_data(
        self,
        stock_code: str,
        days: int = 150,
        indicator_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        获取股票可视化数据

        Args:
            stock_code: 股票代码
            days: 获取天数
            indicator_types: 指标类型列表，不填则使用所有可用指标

        Returns:
            包含K线数据和指标数据的字典
        """
        if indicator_types is None:
            indicator_types = AVAILABLE_INDICATORS

        # 验证指标类型
        invalid_indicators = [t for t in indicator_types if t not in AVAILABLE_INDICATORS]
        if invalid_indicators:
            logger.warning(f"无效的指标类型: {invalid_indicators}")
            indicator_types = [t for t in indicator_types if t in AVAILABLE_INDICATORS]

        # 自动更新最新数据
        try:
            self._refresh_stock_data(stock_code, days)
        except Exception as e:
            logger.warning(f"自动更新数据失败: {e}", exc_info=False)
            # 数据更新失败不影响返回现有数据

        # 获取K线数据
        kline_data, stock_name = self._get_kline_data(stock_code, days)

        if not kline_data:
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'kline_data': [],
                'indicators': []
            }

        # 确保指标数据已计算并保存
        self._ensure_indicators_calculated(stock_code, kline_data, indicator_types)

        # 获取指标数据
        indicators_data = self._get_indicators_data(stock_code, days, indicator_types)

        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'kline_data': kline_data,
            'indicators': indicators_data
        }

    def _refresh_stock_data(self, stock_code: str, days: int = 3650):
        """
        刷新股票数据，确保有足够的历史数据

        Args:
            stock_code: 股票代码
            days: 需要的天数（默认十年）
        """
        try:
            fetcher_manager = DataFetcherManager()
            
            # 检查数据库中现有数据的范围
            latest_date = self._get_latest_kline_date(stock_code)
            earliest_date = self._get_earliest_kline_date(stock_code)
            today = date.today()
            target_start_date = today - timedelta(days=days)
            
            # 获取股票上市日期
            list_date = fetcher_manager.get_list_date(stock_code)
            
            # 如果能获取到上市日期，调整目标开始日期
            if list_date:
                if target_start_date < list_date:
                    logger.info(f"{stock_code} 上市日期为 {list_date}，调整数据获取起始日期")
                    target_start_date = list_date
            
            # 判断市场和交易日
            from src.core.trading_calendar import get_market_for_stock, is_market_open
            market = get_market_for_stock(stock_code)
            is_today_open = is_market_open(market, today) if market else True
            
            need_update = False
            start_date = None
            end_date = None
            
            # 情况1：完全没有数据
            if not latest_date or not earliest_date:
                logger.info(f"{stock_code} 数据库中没有数据，正在获取最近 {days} 天数据...")
                start_date = target_start_date
                end_date = today
                need_update = True
            
            # 情况2：数据不够十年
            elif earliest_date > target_start_date:
                # 如果有上市日期，检查是否还需要补充数据
                if list_date:
                    if earliest_date <= list_date:
                        logger.info(f"{stock_code} 数据已覆盖上市日期，无需补充历史数据")
                    else:
                        missing_days = (earliest_date - target_start_date).days
                        logger.info(f"{stock_code} 历史数据不足，缺少 {missing_days} 天，正在补充历史数据...")
                        start_date = target_start_date
                        end_date = earliest_date - timedelta(days=1)
                        need_update = True
                else:
                    # 无法获取上市日期，保守策略：不补充历史数据，避免尝试获取上市前不存在的数据
                    logger.info(f"{stock_code} 无法获取上市日期，跳过补充历史数据（避免尝试获取上市前数据）")
            
            # 情况3：只需要更新最新数据
            elif latest_date < today:
                if is_today_open:
                    # 今天是交易日，只获取从 latest_date + 1 天到 today 的数据
                    start_date = latest_date + timedelta(days=1)
                    end_date = today
                    if start_date <= end_date:
                        logger.info(f"{stock_code} 今天是交易日，正在更新最新数据...")
                        need_update = True
                    else:
                        logger.info(f"{stock_code} 数据已是最新，无需更新")
                else:
                    # 今天不是交易日，检查是否需要补充之前的交易日数据
                    # 获取最近5个交易日的数据以确保完整性
                    start_date = latest_date - timedelta(days=5)
                    end_date = latest_date
                    logger.info(f"{stock_code} 今天不是交易日，检查最近5天数据完整性...")
                    need_update = True
            
            if not need_update:
                logger.info(f"{stock_code} 数据已是最新且足够，无需更新")
                return
            
            # 转换为字符串格式
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # 获取股票名称
            stock_name = fetcher_manager.get_stock_name(stock_code)
            
            # 获取日线数据
            logger.info(f"正在获取 {stock_code} 数据: {start_date_str} ~ {end_date_str}")
            daily_data, source_name = fetcher_manager.get_daily_data(
                stock_code,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if daily_data is not None and not daily_data.empty:
                saved_count = self.db.save_daily_data(daily_data, stock_code, 'VisualizationService')
                logger.info(f"{stock_code} 更新了 {saved_count} 条日线数据 (来源: {source_name})")
            else:
                logger.warning(f"{stock_code} 未获取到新数据")
                
        except Exception as e:
            logger.error(f"刷新数据失败: {e}", exc_info=True)
            raise

    def _get_earliest_kline_date(self, stock_code: str) -> Optional[date]:
        """
        获取最早K线日期

        Args:
            stock_code: 股票代码

        Returns:
            最早日期或None
        """
        try:
            with self.db.session_scope() as session:
                from src.storage import StockDaily
                from sqlalchemy import func
                
                result = session.query(
                    func.min(StockDaily.date).label('min_date')
                ).filter(StockDaily.code == stock_code).first()
                
                return result.min_date if result else None
        except Exception as e:
            logger.error(f"获取最早日期失败: {e}", exc_info=True)
            return None

    def _get_latest_kline_date(self, stock_code: str) -> Optional[date]:
        """
        获取最新K线日期

        Args:
            stock_code: 股票代码

        Returns:
            最新日期或None
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            stock_dailies = self.db.get_data_range(stock_code, start_date, end_date)
            
            if not stock_dailies:
                return None
            
            return max(sd.date for sd in stock_dailies)
        except Exception as e:
            logger.error(f"获取最新日期失败: {e}", exc_info=True)
            return None

    def _get_kline_data(
        self,
        stock_code: str,
        days: int
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        获取K线数据

        Args:
            stock_code: 股票代码
            days: 获取天数

        Returns:
            (K线数据列表, 股票名称)
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        try:
            stock_dailies = self.db.get_data_range(stock_code, start_date, end_date)

            if not stock_dailies:
                logger.warning(f"未找到 {stock_code} 的数据")
                return [], None

            kline_data = []
            for sd in stock_dailies:
                kline_data.append({
                    'date': sd.date.isoformat(),
                    'open': sd.open,
                    'high': sd.high,
                    'low': sd.low,
                    'close': sd.close,
                    'volume': sd.volume,
                    'amount': sd.amount,
                    'pct_chg': sd.pct_chg
                })

            # 从 STOCK_NAME_MAP 获取股票名称，如果没有则尝试从数据管理器获取
            from src.analyzer import STOCK_NAME_MAP
            stock_name = STOCK_NAME_MAP.get(stock_code)
            
            # 如果 STOCK_NAME_MAP 中没有，尝试从数据管理器获取
            if not stock_name:
                try:
                    from data_provider import DataFetcherManager
                    fetcher_manager = DataFetcherManager()
                    stock_name = fetcher_manager.get_stock_name(stock_code)
                except Exception as e:
                    logger.warning(f"获取股票名称失败：{e}")
            
            return kline_data, stock_name
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}", exc_info=True)
            return [], None

    def _ensure_indicators_calculated(
        self,
        stock_code: str,
        kline_data: List[Dict[str, Any]],
        indicator_types: List[str]
    ):
        """
        确保指标已计算并保存

        Args:
            stock_code: 股票代码
            kline_data: K线数据
            indicator_types: 指标类型列表
        """
        if not kline_data:
            return

        # 转换为DataFrame
        df = pd.DataFrame(kline_data)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)
        df.index.name = 'date'

        # 为每个指标类型检查和计算
        for indicator_type in indicator_types:
            try:
                self._calculate_and_save_indicator(stock_code, df, indicator_type)
            except Exception as e:
                logger.error(f"计算指标 {indicator_type} 失败: {e}", exc_info=True)

    def _calculate_and_save_indicator(
        self,
        stock_code: str,
        df: pd.DataFrame,
        indicator_type: str
    ):
        """
        计算并保存单个指标（每次都重新计算，确保使用最新策略）

        Args:
            stock_code: 股票代码
            df: K线数据DataFrame
            indicator_type: 指标类型
        """
        end_date = df.index.max()
        start_date = df.index.min()

        logger.info(f"{stock_code} {indicator_type} 正在重新计算指标 (策略可能已更新)...")

        # 计算指标
        calculator_class = INDICATOR_CALCULATORS.get(indicator_type)
        if not calculator_class:
            logger.warning(f"未知的指标类型: {indicator_type}")
            return

        calculator = calculator_class()
        
        # 转换列名为大写以符合指标计算器要求
        df_for_calc = df.reset_index()
        df_for_calc = df_for_calc.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'amount': 'Amount',
            'pct_chg': 'Pct_chg'
        })
        
        result_df = calculator.calculate(df_for_calc)

        # 保存指标数据
        saved_count = self.db.save_stock_indicators(
            stock_code,
            indicator_type,
            result_df,
            'VisualizationService'
        )

        logger.info(f"{stock_code} {indicator_type} 重新计算并保存了 {saved_count} 条指标数据")

    def _get_indicators_data(
        self,
        stock_code: str,
        days: int,
        indicator_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        获取指标数据

        Args:
            stock_code: 股票代码
            days: 获取天数
            indicator_types: 指标类型列表

        Returns:
            指标数据列表
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        indicators_data = []

        for indicator_type in indicator_types:
            try:
                df = self.db.get_stock_indicators_as_df(
                    stock_code,
                    indicator_type,
                    start_date,
                    end_date
                )

                if not df.empty:
                    # 转换为列表格式
                    data_list = df.to_dict('records')
                    indicators_data.append({
                        'indicator_type': indicator_type,
                        'data': data_list
                    })
            except Exception as e:
                logger.error(f"获取指标数据 {indicator_type} 失败: {e}", exc_info=True)

        return indicators_data

    def save_search_history(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        selected_indicators: Optional[List[str]] = None
    ) -> int:
        """
        保存搜索历史

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            selected_indicators: 选中的指标列表

        Returns:
            记录ID
        """
        # 删除重复记录
        self.db.delete_duplicate_visualization_history(stock_code)
        # 保存新记录
        return self.db.save_visualization_search_history(
            stock_code,
            stock_name,
            selected_indicators
        )

    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取搜索历史

        Args:
            limit: 返回数量限制

        Returns:
            搜索历史列表
        """
        return self.db.get_visualization_search_history(limit)
