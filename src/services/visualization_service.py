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
from indicators.indicators.chip_distribution import ChipDistribution
from src.storage import DatabaseManager, get_db
from src.core.trading_calendar import get_start_date_by_trading_days, get_market_for_stock
from src.services.turnover_service import TurnoverService

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


def _should_add_realtime_quote(kline_data: List[Dict[str, Any]], force_update: bool = False) -> bool:
    """
    判断是否应该添加实时行情
    
    Args:
        kline_data: 现有的 K线数据列表，每个元素是包含 'date' 字段的字典
        force_update: 是否强制更新，跳过交易时间和已有数据的检查
        
    Returns:
        True 如果应该添加实时行情，否则返回 False
    """
    # 如果强制更新，直接返回 True
    if force_update:
        return True
    
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
        start_date: Optional[date] = None,
        force_update: bool = False,
        update_circulating_shares: bool = True
    ) -> Dict[str, Any]:
        """
        获取股票可视化数据（每次都重新获取所有数据）

        Args:
            stock_code: 股票代码
            days: 获取天数（如果提供了 start_date，则此参数被忽略）
            indicator_types: 指标类型列表，不填则使用所有可用指标
            start_date: 起始日期（可选）
            force_update: 是否强制更新，跳过交易时间和已有数据的检查
            update_circulating_shares: 是否更新流通股本和换手率

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
            # 根据股票代码确定市场
            market = get_market_for_stock(stock_code) or "cn"
            # 使用交易日数计算起始日期
            query_start_date = get_start_date_by_trading_days(today, days, market)
        
        # 直接获取数据 - 结束日期加一天，确保包含当天的数据
        end_date_for_query = today + timedelta(days=1)
        daily_data, source_name = fetcher_manager.get_daily_data(
            stock_code,
            start_date=query_start_date.strftime('%Y-%m-%d'),
            end_date=end_date_for_query.strftime('%Y-%m-%d')
        )
        
        if daily_data is None or daily_data.empty:
            logger.warning(f"未获取到 {stock_code} 的数据")
            return {
                'stock_code': stock_code,
                'stock_name': None,
                'kline_data': [],
                'indicators': [],
                'circulating_shares_updated': False,
                'circulating_shares': None,
                'turnover_filled_count': 0
            }
        
        logger.info(f"成功获取 {stock_code} 数据: {len(daily_data)} 条 (来源: {source_name})")
        
        # 打印数据日期范围，便于调试
        if not daily_data.empty:
            if 'date' in daily_data.columns:
                first_date = daily_data['date'].iloc[0]
                last_date = daily_data['date'].iloc[-1]
                logger.info(f"{stock_code} 数据日期范围: {first_date} ~ {last_date}")
        
        # 更新流通股本（不更新历史换手率，后续实时计算）
        circulating_shares_updated = False
        circulating_shares = None
        if update_circulating_shares:
            try:
                logger.info(f"正在更新 {stock_code} 的流通股本...")
                turnover_service = TurnoverService()
                stock_name_for_update = fetcher_manager.get_stock_name(stock_code)
                # fill_historical=False：不更新数据库中的历史换手率，只保存流通股本
                success, _ = turnover_service.process_akshare_data(
                    daily_data,
                    stock_code,
                    stock_name_for_update,
                    fill_historical=False
                )
                if success:
                    circulating_shares_updated = True
                    # 获取更新后的流通股本信息
                    basic_info = turnover_service.get_stock_basic(stock_code)
                    if basic_info:
                        circulating_shares = basic_info.circulating_shares
                    logger.info(f"{stock_code} 流通股本更新成功: {circulating_shares:.0f} 股，后续将实时计算换手率")
                else:
                    logger.warning(f"{stock_code} 流通股本更新失败，数据可能不足")
            except Exception as e:
                logger.error(f"更新 {stock_code} 流通股本时出错: {e}", exc_info=True)
        
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
            
            kline_item = {
                'date': date_str,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if pd.notna(row['volume']) else 0,
                'amount': float(row['amount']) if pd.notna(row['amount']) else 0,
                'pct_chg': float(row['pct_chg']) if pd.notna(row['pct_chg']) else 0
            }
            
            # 添加换手率信息（如果有）
            if 'turnover_rate' in row and pd.notna(row['turnover_rate']):
                kline_item['turnover_rate'] = float(row['turnover_rate'])
            elif '换手率' in row and pd.notna(row['换手率']):
                # 如果是百分比格式，转换为小数
                kline_item['turnover_rate'] = float(row['换手率']) / 100.0
            
            kline_data.append(kline_item)
        
        # 打印转换后 kline_data 的日期范围，便于调试
        if kline_data:
            first_kline_date = kline_data[0]['date']
            last_kline_date = kline_data[-1]['date']
            logger.info(f"{stock_code} 转换后 K线数据日期范围: {first_kline_date} ~ {last_kline_date}")
        
        # 整合实时行情数据
        try:
            # 首先判断是否需要整合实时行情
            should_add = _should_add_realtime_quote(kline_data, force_update=force_update)
            
            if not should_add:
                logger.info(f"{stock_code} 不需要整合实时行情（非交易时间或已有今日数据），使用历史数据")
            else:
                logger.info(f"正在获取 {stock_code} 实时行情数据...")
                quote = fetcher_manager.get_realtime_quote(stock_code)
                
                if quote is not None:
                    logger.info(f"将 {stock_code} 实时行情整合到 K线数据中...")
                    realtime_kline = _convert_realtime_quote_to_kline(quote)
                    kline_data.append(realtime_kline)
                    logger.info(f"成功整合 {stock_code} 实时行情数据")
                else:
                    logger.warning(f"{stock_code} 获取实时行情失败，尝试其他方法...")
                    # 这里可以添加其他获取实时行情的方法
        except Exception as e:
            logger.warning(f"获取或整合 {stock_code} 实时行情失败: {e}，继续使用历史数据")
        
        # 获取股票名称 - 如果不需要整合实时行情，则跳过实时行情获取
        should_add = _should_add_realtime_quote(kline_data, force_update=force_update)
        stock_name = fetcher_manager.get_stock_name(stock_code, skip_realtime=not should_add)
        
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
        
        # 预计算最新日期的筹码分布数据
        chip_distribution_data = None
        try:
            # 检查是否有换手率数据（必须要有真实换手率才能计算）
            has_turnover = False
            if daily_data is not None and 'turnover_rate' in daily_data.columns:
                has_turnover = daily_data['turnover_rate'].notna().any()
            elif kline_data:
                # 检查kline_data中是否有turnover_rate字段
                has_turnover = any('turnover_rate' in item for item in kline_data)
            
            if has_turnover:
                logger.info(f"开始预计算 {stock_code} 的筹码分布数据...")
                chip_distribution_data = self._precompute_chip_distribution(
                    daily_data,
                    kline_data,
                    stock_code
                )
                logger.info(f"{stock_code} 筹码分布预计算完成")
            else:
                logger.warning(f"{stock_code} 没有真实换手率数据，跳过筹码分布预计算")
        except Exception as e:
            logger.warning(f"预计算 {stock_code} 筹码分布失败: {e}")
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'kline_data': kline_data,
            'indicators': indicators_data,
            'chip_distribution': chip_distribution_data,
            'circulating_shares_updated': circulating_shares_updated,
            'circulating_shares': circulating_shares
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
                # 根据股票代码确定市场
                market = get_market_for_stock(stock_code) or "cn"
                # 使用交易日数计算起始日期
                target_start_date = get_start_date_by_trading_days(today, days, market)
            
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
                    
                    # 获取日线数据 - 结束日期加一天，确保包含当天的数据
                    end_date_for_query = data_end_date + timedelta(days=1)
                    logger.info(f"正在获取 {stock_code} 数据: {start_date_str} ~ {end_date_for_query.strftime('%Y-%m-%d')}")
                    daily_data, source_name = fetcher_manager.get_daily_data(
                        stock_code,
                        start_date=start_date_str,
                        end_date=end_date_for_query.strftime('%Y-%m-%d')
                    )
                    
                    if daily_data is not None and not daily_data.empty:
                        saved_count = self.db.save_daily_data(daily_data, stock_code, 'VisualizationService')
                        logger.info(f"{stock_code} 更新了 {saved_count} 条日线数据 (来源: {source_name})")
                        
                        # 处理流通股本和换手率
                        try:
                            turnover_service = TurnoverService()
                            stock_name = fetcher_manager.get_stock_name(stock_code)
                            success, fill_count = turnover_service.process_akshare_data(
                                daily_data, 
                                stock_code, 
                                stock_name
                            )
                            if success:
                                logger.info(f"{stock_code} 成功处理流通股本和换手率，填充了 {fill_count} 条历史换手率")
                        except Exception as e:
                            logger.warning(f"{stock_code} 处理流通股本和换手率失败: {e}")
                        
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
                    
                    # 获取日线数据 - 结束日期加一天，确保包含当天的数据
                    end_date_for_query = data_end_date + timedelta(days=1)
                    logger.info(f"正在获取 {stock_code} 数据: {start_date_str} ~ {end_date_for_query.strftime('%Y-%m-%d')}")
                    daily_data, source_name = fetcher_manager.get_daily_data(
                        stock_code,
                        start_date=start_date_str,
                        end_date=end_date_for_query.strftime('%Y-%m-%d')
                    )
                    
                    if daily_data is not None and not daily_data.empty:
                        saved_count = self.db.save_daily_data(daily_data, stock_code, 'VisualizationService')
                        logger.info(f"{stock_code} 更新了 {saved_count} 条日线数据 (来源: {source_name})")
                        
                        # 处理流通股本和换手率
                        try:
                            turnover_service = TurnoverService()
                            stock_name = fetcher_manager.get_stock_name(stock_code)
                            success, fill_count = turnover_service.process_akshare_data(
                                daily_data, 
                                stock_code, 
                                stock_name
                            )
                            if success:
                                logger.info(f"{stock_code} 成功处理流通股本和换手率，填充了 {fill_count} 条历史换手率")
                        except Exception as e:
                            logger.warning(f"{stock_code} 处理流通股本和换手率失败: {e}")
                        
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
            # 根据股票代码确定市场
            market = get_market_for_stock(stock_code) or "cn"
            # 使用交易日数计算起始日期
            start_date = get_start_date_by_trading_days(end_date, days, market)

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

        # 保存指标数据（使用批量保存优化）
        saved_count = self.db.save_stock_indicators_bulk(
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
            # 根据股票代码确定市场
            market = get_market_for_stock(stock_code) or "cn"
            # 使用交易日数计算起始日期
            start_date = get_start_date_by_trading_days(end_date, days, market)

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
        # 先保存新记录
        record_id = self.db.save_visualization_search_history(
            stock_code,
            stock_name,
            selected_indicators
        )
        # 然后删除重复记录，只保留最新的那条（刚保存的）
        self.db.delete_duplicate_visualization_history(stock_code)
        return record_id

    def get_search_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取搜索历史

        Args:
            limit: 返回数量限制

        Returns:
            搜索历史列表
        """
        return self.db.get_visualization_search_history(limit)

    def update_search_history_timestamp(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        更新搜索历史记录的时间戳为当前时间

        Args:
            record_id: 记录ID

        Returns:
            更新后的记录信息，失败返回 None
        """
        return self.db.update_visualization_search_history_timestamp(record_id)

    def delete_search_history(self, record_id: int) -> bool:
        """
        删除搜索历史记录

        Args:
            record_id: 记录ID

        Returns:
            是否成功删除
        """
        return self.db.delete_visualization_search_history(record_id)
    
    def _precompute_chip_distribution(
        self,
        daily_data: Optional[pd.DataFrame],
        kline_data: List[Dict[str, Any]],
        stock_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        预计算最新日期的筹码分布数据

        Args:
            daily_data: 从API获取的原始数据DataFrame
            kline_data: 转换后的K线数据列表
            stock_code: 股票代码

        Returns:
            预计算的筹码分布数据，失败返回None
        """
        try:
            # 准备数据
            df = None
            if daily_data is not None and not daily_data.empty:
                df = daily_data.copy()
            elif kline_data:
                df = pd.DataFrame(kline_data)
            
            if df is None or df.empty:
                logger.warning("没有数据用于预计算筹码分布")
                return None
            
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            
            # 指标计算器需要首字母大写的列名
            rename_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            # 只重命名存在的列
            for old_col, new_col in rename_map.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # 只计算最新日期的筹码分布
            calculator = ChipDistribution()
            result = calculator.calculate(df)
            # 确保结果包含 stock_code 字段
            if result:
                result['stock_code'] = stock_code
            
            logger.info(f"仅预计算了最新日期的筹码分布")
            
            # 返回结果 - 只返回最新数据，不预计算历史
            return {
                'latest': result,
                'history': []  # 历史数据留空，十字线移动时直接调用API
            }
        except Exception as e:
            logger.warning(f"预计算筹码分布失败: {e}")
            return None

    def get_chip_distribution(
        self,
        stock_code: str,
        days: int = 365,
        start_date: Optional[date] = None,
        end_date_idx: Optional[int] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        获取股票筹码分布数据（仅从数据库获取，禁止使用API和模拟换手率）

        Args:
            stock_code: 股票代码
            days: 获取天数（如果提供了 start_date，则此参数被忽略）
            start_date: 起始日期（可选）
            end_date_idx: 截止日期索引（用于移动成本分布，可选）
            end_date: 截止日期（可选，如果提供会根据此日期计算end_date_idx）

        Returns:
            包含筹码分布数据的字典
        """
        today = date.today()
        logger.info(f"开始获取 {stock_code} 的筹码分布数据，日期: {end_date if end_date else '最新'}")

        # 确定查询范围
        if start_date:
            query_start_date = start_date
        else:
            market = get_market_for_stock(stock_code) or "cn"
            query_start_date = get_start_date_by_trading_days(today, days, market)

        # 如果提供了end_date，查询数据时需要包含到end_date
        if end_date:
            end_date_for_query = end_date + timedelta(days=1)
        else:
            end_date_for_query = today + timedelta(days=1)
        
        # 仅从数据库获取数据
        daily_data = None
        
        # 获取流通股本用于计算换手率
        from src.services.turnover_service import TurnoverService
        turnover_service = TurnoverService()
        stock_basic = turnover_service.get_stock_basic(stock_code)
        circulating_shares = None
        
        if stock_basic and stock_basic.circulating_shares and stock_basic.circulating_shares > 0:
            circulating_shares = stock_basic.circulating_shares
            logger.info(f"获取到 {stock_code} 的流通股本: {circulating_shares:.0f} 股")
        else:
            logger.warning(f"{stock_code} 没有有效的流通股本数据，将尝试使用数据库中的换手率")
        
        try:
            from src.repositories.stock_repo import StockRepository
            repo = StockRepository()
            
            # 从数据库获取数据
            db_records = repo.get_range(
                stock_code, 
                query_start_date, 
                end_date_for_query
            )
            
            if db_records and len(db_records) > 0:
                # 将数据库记录转换为 DataFrame，安全获取字段
                kline_data = []
                for record in db_records:
                    kline_item = {
                        'date': record.date,
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                        'amount': record.amount,
                        'pct_chg': record.pct_chg
                    }
                    
                    # 优先用流通股本实时计算换手率（更准确）
                    if circulating_shares and circulating_shares > 0 and record.volume and record.volume > 0:
                        # 实时计算：换手率 = 成交量 / 流通股本
                        turnover_rate = record.volume / circulating_shares
                        turnover_rate = min(turnover_rate, 1.0)  # 限制在100%以内
                        kline_item['turnover_rate'] = turnover_rate
                    else:
                        # 回退到数据库中存储的换手率
                        try:
                            kline_item['turnover_rate'] = getattr(record, 'turnover_rate', None)
                        except Exception:
                            kline_item['turnover_rate'] = None
                    
                    kline_data.append(kline_item)
                
                daily_data = pd.DataFrame(kline_data)
                if circulating_shares:
                    logger.info(f"{stock_code} 从数据库获取了 {len(daily_data)} 条数据，使用流通股本实时计算换手率")
                else:
                    logger.info(f"{stock_code} 从数据库获取了 {len(daily_data)} 条数据，使用数据库中的换手率")
            else:
                logger.warning(f"{stock_code} 数据库中没有找到数据")
        except Exception as e:
            logger.error(f"{stock_code} 从数据库获取数据失败: {e}")

        if daily_data is None or daily_data.empty:
            logger.warning(f"未获取到 {stock_code} 的数据")
            return {
                'stock_code': stock_code,
                'price_bins': [],
                'chip_volumes': [],
                'profit_volumes': [],
                'loss_volumes': [],
                'avg_cost': None,
                'max_chip_price': None,
                'current_price': None
            }

        # 检查是否有真实换手率数据
        has_real_turnover = False
        if 'turnover_rate' in daily_data.columns:
            has_real_turnover = daily_data['turnover_rate'].notna().any()
        
        if not has_real_turnover:
            logger.warning(f"{stock_code} 数据库中没有真实换手率数据，无法计算筹码分布")
            return {
                'stock_code': stock_code,
                'price_bins': [],
                'chip_volumes': [],
                'profit_volumes': [],
                'loss_volumes': [],
                'avg_cost': None,
                'max_chip_price': None,
                'current_price': None
            }

        # 转换为筹码分布计算需要的格式
        kline_data = []
        for _, row in daily_data.iterrows():
            if hasattr(row['date'], 'strftime'):
                date_str = row['date'].strftime('%Y-%m-%d')
            elif hasattr(row['date'], 'date'):
                date_str = row['date'].date().strftime('%Y-%m-%d')
            elif pd.notna(row['date']):
                date_str = str(row['date']).split('T')[0]
            else:
                continue

            kline_data.append({
                'date': date_str,
                'Open': float(row['open']),
                'High': float(row['high']),
                'Low': float(row['low']),
                'Close': float(row['close']),
                'Volume': float(row['volume']) if pd.notna(row['volume']) else 0
            })
            
            # 添加换手率（如果有）
            if 'turnover_rate' in row and pd.notna(row['turnover_rate']):
                kline_data[-1]['turnover_rate'] = float(row['turnover_rate'])

        df = pd.DataFrame(kline_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 如果提供了end_date但没有提供end_date_idx，根据end_date计算idx
        calculated_end_date_idx = end_date_idx
        if end_date is not None and end_date_idx is None:
            # 找到与end_date最接近的数据点的索引
            end_date_ts = pd.Timestamp(end_date)
            date_diffs = (df['date'] - end_date_ts).abs()
            calculated_end_date_idx = date_diffs.idxmin()
            logger.info(f"根据end_date={end_date}计算出end_date_idx={calculated_end_date_idx}")

        # 使用真实换手率计算筹码分布，禁止使用模拟换手率
        calculator = ChipDistribution()
        result = calculator.calculate(df, calculated_end_date_idx)
        result['stock_code'] = stock_code

        logger.info(f"{stock_code} 筹码分布计算完成")
        return result
