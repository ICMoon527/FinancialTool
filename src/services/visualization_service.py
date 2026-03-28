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
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from data_provider import DataFetcherManager
from data_provider.realtime_types import UnifiedRealtimeQuote
from indicators.indicators.banker_control import BankerControl
from indicators.indicators.main_capital_absorption import MainCapitalAbsorption
from indicators.indicators.main_cost import MainCost
from indicators.indicators.main_trading import MainTrading
from indicators.indicators.momentum_2 import Momentum2
from indicators.indicators.strong_detonation import StrongDetonation
from indicators.indicators.resonance_chase import ResonanceChase
from src.storage import DatabaseManager, get_db

logger = logging.getLogger(__name__)


def _is_trading_time() -> bool:
    """
    判断当前是否为交易时间（包含集合竞价）
    
    Returns:
        True 如果是交易时间，否则返回 False
    """
    now = datetime.now()
    
    # 检查是否为周末（周六、周日）
    if now.weekday() >= 5:
        return False
    
    # 检查当前时间是否在 9:15-15:00 之间（包含集合竞价时间）
    current_time = now.time()
    trading_start = time(9, 15)
    trading_end = time(15, 0)
    
    return trading_start <= current_time <= trading_end


def _should_add_realtime_quote(kline_data: List[Dict[str, Any]]) -> bool:
    """
    判断是否应该添加实时行情
    
    Args:
        kline_data: 现有的 K线数据列表，每个元素是包含 'date' 字段的字典
        
    Returns:
        True 如果应该添加实时行情，否则返回 False
    """
    # 首先检查是否为交易时间
    if not _is_trading_time():
        return False
    
    # 获取今天的日期字符串
    today_str = date.today().strftime('%Y-%m-%d')
    
    # 检查 kline_data 中是否已经有今天的数据
    for kline in kline_data:
        if kline.get('date') == today_str:
            return False
    
    # 没有今天的数据，应该添加实时行情
    return True


def _convert_realtime_quote_to_kline(quote: UnifiedRealtimeQuote) -> Dict[str, Any]:
    """
    将 UnifiedRealtimeQuote 对象转换为 K线数据格式字典
    
    Args:
        quote: 实时行情对象
        
    Returns:
        符合 K线数据格式的字典
    """
    today = date.today()
    return {
        'date': today.strftime('%Y-%m-%d'),
        'open': float(quote.open_price) if quote.open_price is not None else 0.0,
        'high': float(quote.high) if quote.high is not None else 0.0,
        'low': float(quote.low) if quote.low is not None else 0.0,
        'close': float(quote.price) if quote.price is not None else 0.0,
        'volume': float(quote.volume) if quote.volume is not None else 0.0,
        'amount': float(quote.amount) if quote.amount is not None else 0.0,
        'pct_chg': float(quote.change_pct) if quote.change_pct is not None else 0.0
    }


# 定义所有支持的指标类型
AVAILABLE_INDICATORS = [
    'volume',
    'banker_control',
    'main_capital_absorption',
    'main_cost',
    'main_trading',
    'momentum_2',
    'strong_detonation',
    'resonance_chase'
]

# 指标计算器映射
INDICATOR_CALCULATORS = {
    'banker_control': BankerControl,
    'main_capital_absorption': MainCapitalAbsorption,
    'main_cost': MainCost,
    'main_trading': MainTrading,
    'momentum_2': Momentum2,
    'strong_detonation': StrongDetonation,
    'resonance_chase': ResonanceChase
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
        days: int = 365,
        indicator_types: Optional[List[str]] = None,
        start_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        获取股票可视化数据（每次都重新获取所有数据）

        Args:
            stock_code: 股票代码
            days: 获取天数（如果提供了 start_date，则此参数被忽略）
            indicator_types: 指标类型列表，不填则使用所有可用指标
            start_date: 起始日期（可选）

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

        # 每次都重新获取数据，不依赖数据库
        logger.info(f"正在直接获取 {stock_code} 的数据...")
        
        fetcher_manager = DataFetcherManager()
        today = date.today()
        
        # 确定查询范围
        if start_date:
            query_start_date = start_date
        else:
            query_start_date = today - timedelta(days=days)
        
        # 直接获取数据
        daily_data, source_name = fetcher_manager.get_daily_data(
            stock_code,
            start_date=query_start_date.strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d')
        )
        
        if daily_data is None or daily_data.empty:
            logger.warning(f"未获取到 {stock_code} 的数据")
            return {
                'stock_code': stock_code,
                'stock_name': None,
                'kline_data': [],
                'indicators': []
            }
        
        logger.info(f"成功获取 {stock_code} 数据: {len(daily_data)} 条 (来源: {source_name})")
        
        # 转换数据格式
        kline_data = []
        for _, row in daily_data.iterrows():
            # 确保日期格式为 YYYY-MM-DD
            if hasattr(row['date'], 'strftime'):
                date_str = row['date'].strftime('%Y-%m-%d')
            elif hasattr(row['date'], 'date'):
                date_str = row['date'].date().strftime('%Y-%m-%d')
            else:
                date_str = str(row['date']).split('T')[0]
            
            kline_data.append({
                'date': date_str,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if pd.notna(row['volume']) else 0,
                'amount': float(row['amount']) if pd.notna(row['amount']) else 0,
                'pct_chg': float(row['pct_chg']) if pd.notna(row['pct_chg']) else 0
            })
        
        # 整合实时行情数据
        try:
            logger.info(f"正在获取 {stock_code} 实时行情数据...")
            quote = fetcher_manager.get_realtime_quote(stock_code)
            
            # 判断是否需要添加实时行情
            if _should_add_realtime_quote(kline_data) and quote is not None:
                logger.info(f"将 {stock_code} 实时行情整合到 K线数据中...")
                realtime_kline = _convert_realtime_quote_to_kline(quote)
                kline_data.append(realtime_kline)
                logger.info(f"成功整合 {stock_code} 实时行情数据")
            else:
                logger.info(f"{stock_code} 不需要整合实时行情或获取失败，使用历史数据")
        except Exception as e:
            logger.warning(f"获取或整合 {stock_code} 实时行情失败: {e}，继续使用历史数据")
        
        # 获取股票名称
        stock_name = fetcher_manager.get_stock_name(stock_code)
        
        # 计算指标（不保存到数据库，直接计算）
        indicators_data = []
        
        # 获取资金流向数据（如果需要）
        fund_flow_data = None
        if 'main_cost' in indicator_types or 'main_capital_absorption' in indicator_types:
            try:
                logger.info(f"正在获取 {stock_code} 的资金流向数据...")
                # 从 K线数据获取日期范围
                start_date = None
                end_date = None
                if kline_data:
                    start_date = kline_data[0].get('date')
                    end_date = kline_data[-1].get('date')
                fund_flow_data = fetcher_manager.get_fund_flow_data(stock_code, start_date=start_date, end_date=end_date)
                if fund_flow_data is not None and not fund_flow_data.empty:
                    logger.info(f"成功获取 {stock_code} 资金流向数据: {len(fund_flow_data)} 条")
                else:
                    logger.warning(f"未获取到 {stock_code} 资金流向数据，将使用模拟数据")
            except Exception as e:
                logger.warning(f"获取 {stock_code} 资金流向数据失败: {e}，将使用模拟数据")
        
        # 获取大盘指数数据（如果需要）
        index_data = None
        if 'strong_detonation' in indicator_types:
            try:
                # 根据股票代码判断对应的大盘指数
                index_symbol = None
                if stock_code.startswith('60') or stock_code.startswith('688'):
                    # 上海股票对应上证指数
                    index_symbol = 'sh000001'
                elif stock_code.startswith('00') or stock_code.startswith('30'):
                    # 深圳股票对应深证成指
                    index_symbol = 'sz399001'
                
                if index_symbol:
                    logger.info(f"正在获取 {stock_code} 对应的大盘指数数据 ({index_symbol})...")
                    # 获取与K线数据相同日期范围的大盘指数
                    if kline_data:
                        start_date = kline_data[0].get('date')
                        end_date = kline_data[-1].get('date')
                        index_data = fetcher_manager.get_index_daily_data(index_symbol, start_date, end_date)
                    else:
                        index_data = fetcher_manager.get_index_daily_data(index_symbol)
                    
                    if index_data is not None and not index_data.empty:
                        logger.info(f"成功获取 {index_symbol} 大盘指数数据: {len(index_data)} 条")
                    else:
                        logger.warning(f"未获取到 {index_symbol} 大盘指数数据，将使用代理模式")
            except Exception as e:
                logger.warning(f"获取大盘指数数据失败: {e}，将使用代理模式")
        
        if kline_data:
            # 转换为 DataFrame 用于指标计算
            df = pd.DataFrame(kline_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            # 指标计算器需要首字母大写的列名
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            for indicator_type in indicator_types:
                calculator_class = INDICATOR_CALCULATORS.get(indicator_type)
                if not calculator_class:
                    continue
                
                try:
                    calculator = calculator_class()
                    # 如果是主力成本指标且有资金流向数据，传递资金流向数据
                    # 如果是强势起爆指标且有大盘数据，传递大盘数据
                    if (indicator_type == 'main_cost') and fund_flow_data is not None:
                        result_df = calculator.calculate(df.copy(), fund_flow_data=fund_flow_data)
                    elif (indicator_type == 'strong_detonation') and index_data is not None:
                        result_df = calculator.calculate(df.copy(), index_data=index_data)
                    else:
                        result_df = calculator.calculate(df.copy())
                    
                    # 转换为指标数据格式
                    indicator_data_list = []
                    for _, row in result_df.iterrows():
                        # 确保日期格式为 YYYY-MM-DD
                        if hasattr(row['date'], 'strftime'):
                            date_str = row['date'].strftime('%Y-%m-%d')
                        elif hasattr(row['date'], 'date'):
                            date_str = row['date'].date().strftime('%Y-%m-%d')
                        else:
                            date_str = str(row['date']).split('T')[0]
                        
                        item = {
                            'date': date_str
                        }
                        # 添加指标特定字段
                        for col in result_df.columns:
                            if col != 'date':
                                item[col] = float(row[col]) if pd.notna(row[col]) else None
                        indicator_data_list.append(item)
                    
                    # 如果是主力成本指标且有资金流向数据，添加资金流向信息
                    indicator_metadata = {}
                    if indicator_type == 'main_cost' and fund_flow_data is not None and not fund_flow_data.empty:
                        try:
                            # 获取最新一天的资金流向数据
                            latest_fund_flow = fund_flow_data.iloc[-1]
                            indicator_metadata = {
                                'main_net_inflow': float(latest_fund_flow.get('main_net_inflow', 0)),
                                'super_net_inflow': float(latest_fund_flow.get('super_net_inflow', 0)),
                                'big_net_inflow': float(latest_fund_flow.get('big_net_inflow', 0)),
                                'medium_net_inflow': float(latest_fund_flow.get('medium_net_inflow', 0)),
                                'small_net_inflow': float(latest_fund_flow.get('small_net_inflow', 0)),
                            }
                        except Exception as e:
                            logger.warning(f"获取最新资金流向数据失败: {e}")
                    
                    indicators_data.append({
                        'indicator_type': indicator_type,
                        'data': indicator_data_list,
                        'metadata': indicator_metadata
                    })
                    logger.info(f"计算完成 {stock_code} {indicator_type} 指标: {len(indicator_data_list)} 条")
                    
                except Exception as e:
                    logger.warning(f"计算 {stock_code} {indicator_type} 指标失败: {e}")
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'kline_data': kline_data,
            'indicators': indicators_data
        }

    def _refresh_stock_data(self, stock_code: str, days: int = 3650, start_date: Optional[date] = None):
        """
        刷新股票数据，确保有足够的历史数据

        Args:
            stock_code: 股票代码
            days: 需要的天数（如果提供了 start_date，则此参数被忽略）
            start_date: 起始日期（可选）
        """
        try:
            fetcher_manager = DataFetcherManager()
            
            # 检查数据库中现有数据的范围
            latest_date = self._get_latest_kline_date(stock_code)
            today = date.today()
            
            # 确定数据获取范围
            if start_date:
                target_start_date = start_date
            else:
                target_start_date = today - timedelta(days=days)
            
            # 获取股票名称（只获取一次）
            stock_name = fetcher_manager.get_stock_name(stock_code)
            
            # 情况1：完全没有数据
            if not latest_date:
                logger.info(f"{stock_code} 数据库中没有数据，正在获取数据...")
                data_start_date = target_start_date
                # 从今天开始，往前找能获取到数据的日期
                data_end_date = today
                max_attempts = 30
                found = False
                
                for attempt in range(max_attempts):
                    # 转换为字符串格式
                    start_date_str = data_start_date.strftime('%Y-%m-%d')
                    end_date_str = data_end_date.strftime('%Y-%m-%d')
                    
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
                        found = True
                        break
                    
                    # 如果没获取到数据，往前推一天
                    logger.info(f"{stock_code} 未获取到数据，尝试往前推一天...")
                    data_end_date = data_end_date - timedelta(days=1)
                
                if not found:
                    logger.warning(f"{stock_code} 在 {max_attempts} 天内未获取到数据，请缩小查询范围")
            
            # 情况2：只需要更新最新数据
            else:
                # 从今天开始，往前找能获取到数据的日期
                data_end_date = today
                data_start_date = latest_date + timedelta(days=1)
                max_attempts = 30
                found = False
                query_window_expanded = False
                
                for attempt in range(max_attempts):
                    if data_start_date > data_end_date:
                        logger.info(f"{stock_code} 数据已是最新，无需更新")
                        break
                    
                    # 转换为字符串格式
                    start_date_str = data_start_date.strftime('%Y-%m-%d')
                    end_date_str = data_end_date.strftime('%Y-%m-%d')
                    
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
                        found = True
                        break
                    
                    # 如果没获取到数据，根据情况调整查询策略
                    if not query_window_expanded and (data_end_date - data_start_date).days < 14:
                        # 策略1：如果查询窗口小于14天，先扩展窗口到14天，避免只查周末
                        logger.info(f"{stock_code} 窄范围查询无数据，扩展查询窗口至14天...")
                        data_start_date = data_end_date - timedelta(days=14)
                        query_window_expanded = True
                    else:
                        # 策略2：窗口已足够大，往前推一天继续尝试
                        logger.info(f"{stock_code} 未获取到数据，尝试往前推一天...")
                        data_end_date = data_end_date - timedelta(days=1)
                
                if not found and max_attempts > 0:
                    logger.info(f"{stock_code} 数据已是最新（最后交易日 {latest_date}）")
                
        except Exception as e:
            logger.warning(f"刷新数据失败: {e}", exc_info=False)
            # 数据更新失败不影响返回现有数据，降级处理

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
        days: int,
        start_date: Optional[date] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        获取K线数据

        Args:
            stock_code: 股票代码
            days: 获取天数（如果提供了 start_date，则此参数被忽略）
            start_date: 起始日期（可选）

        Returns:
            (K线数据列表, 股票名称)
        """
        end_date = date.today()
        if start_date:
            start_date = start_date
        else:
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
            if indicator_type == 'volume':
                continue
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
            if indicator_type != 'volume':
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
        indicator_types: List[str],
        start_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指标数据

        Args:
            stock_code: 股票代码
            days: 获取天数（如果提供了 start_date，则此参数被忽略）
            indicator_types: 指标类型列表
            start_date: 起始日期（可选）

        Returns:
            指标数据列表
        """
        end_date = date.today()
        if start_date:
            start_date = start_date
        else:
            start_date = end_date - timedelta(days=days)

        indicators_data = []

        for indicator_type in indicator_types:
            if indicator_type == 'volume':
                continue
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
