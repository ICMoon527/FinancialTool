# -*- coding: utf-8 -*-
"""
Mock classes for testing and caching scenarios.

这些类用于模拟数据提供者和实时行情对象，主要用于：
1. 使用数据库缓存数据执行策略时
2. 单元测试中模拟外部依赖
3. 其他需要隔离数据获取逻辑的场景
"""

from datetime import date, timedelta
from typing import Optional, Dict, Any
import pandas as pd


class MockQuote:
    """
    模拟实时行情对象
    
    用于在策略执行时提供缓存的实时行情数据，避免重复调用外部 API。
    """
    
    def __init__(self, context: Dict[str, Any], stock_name: Optional[str] = None):
        """
        初始化 MockQuote
        
        Args:
            context: 数据库返回的分析上下文，包含今日数据和价格变化信息
            stock_name: 股票名称（可选）
        """
        self.price = context.get('today', {}).get('close')
        self.change_pct = context.get('price_change_ratio')
        self.volume = context.get('today', {}).get('volume')
        self.volume_ratio = context.get('volume_change_ratio')
        self.turnover_rate = None
        self.name = stock_name
        
    def has_basic_data(self) -> bool:
        """检查是否有基本的行情数据"""
        return self.price is not None


class MockDataProvider:
    """
    模拟数据提供者
    
    用于在策略执行时从数据库获取历史数据，避免重复调用外部 API。
    """
    
    def __init__(self, db_manager, stock_code: str, mock_quote: MockQuote, original_data_provider=None):
        """
        初始化 MockDataProvider
        
        Args:
            db_manager: 数据库管理器实例
            stock_code: 股票代码
            mock_quote: MockQuote 实例
            original_data_provider: 原始数据提供者（用于获取大盘数据等）
        """
        self.db_manager = db_manager
        self.stock_code = stock_code
        self.mock_quote = mock_quote
        self.original_data_provider = original_data_provider
        
    def get_realtime_quote(self, code: str) -> MockQuote:
        """
        获取实时行情（返回 MockQuote 对象）
        
        Args:
            code: 股票代码
            
        Returns:
            MockQuote 对象
        """
        return self.mock_quote
        
    def get_daily_data(self, code: str, days: int = 60):
        """
        获取历史日线数据（从数据库）
        
        Args:
            code: 股票代码
            days: 获取天数
            
        Returns:
            Tuple[DataFrame, str]: (历史数据 DataFrame, 数据源标识 "database")
        """
        # 获取足够的数据，days*2 以确保有足够的交易日
        end_date = date.today()
        start_date = end_date - timedelta(days=max(days * 2, 365))
        data = self.db_manager.get_data_range(code, start_date, end_date)
        
        if data:
            df = pd.DataFrame([item.to_dict() for item in data])
            return df, "database"
        
        return None, "database"
    
    def get_index_daily_data(self, symbol: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取大盘指数历史数据（代理到原始数据提供者）
        
        Args:
            symbol: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指数历史数据 DataFrame
        """
        if self.original_data_provider and hasattr(self.original_data_provider, 'get_index_daily_data'):
            return self.original_data_provider.get_index_daily_data(symbol, start_date, end_date)
        return None
