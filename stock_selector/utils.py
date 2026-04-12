# -*- coding: utf-8 -*-
"""
工具模块 - 集中存放公共函数，消除代码重复
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


def filter_and_prepare_stock_codes(
    stock_codes: Optional[List[str]],
    db_manager: Optional[any] = None
) -> Tuple[List[str], Dict[str, str]]:
    """
    统一的股票代码过滤和准备函数
    
    Args:
        stock_codes: 股票代码列表（可选）
        db_manager: 数据库管理器（可选）
    
    Returns:
        Tuple: (过滤后的股票代码列表, 股票代码到名称的映射)
    """
    from stock_selector.stock_pool import (
        get_all_stock_code_name_pairs,
        filter_special_stock_codes,
        filter_st_stocks
    )
    
    code_to_name_map = {}
    
    if stock_codes is None:
        # 使用完整股票池
        stock_code_name_pairs = get_all_stock_code_name_pairs()
        # 过滤ST股票
        stock_code_name_pairs = filter_st_stocks(stock_code_name_pairs)
        # 提取股票代码
        stock_codes = [code for code, name in stock_code_name_pairs]
        # 过滤特定板块的股票代码
        stock_codes = filter_special_stock_codes(stock_codes)
        # 构建名称映射
        code_to_name_map = {code: name for code, name in stock_code_name_pairs if code in stock_codes}
        logger.info("使用完整股票池: %d 只股票", len(stock_codes))
    else:
        # 用户提供了股票代码，也需要过滤
        try:
            all_pairs = get_all_stock_code_name_pairs()
            code_to_name_full = {code: name for code, name in all_pairs}
            # 过滤ST股票
            filtered_codes = []
            for code in stock_codes:
                name = code_to_name_full.get(code, "")
                if not any(keyword in name.upper() for keyword in ['ST', '*ST', 'SST', 'S*ST']):
                    filtered_codes.append(code)
                    code_to_name_map[code] = name
            stock_codes = filtered_codes
        except Exception as e:
            logger.warning("过滤股票代码时出错: %s", e)
    
    # 尝试从数据库批量获取股票名称
    if db_manager and stock_codes:
        try:
            from stock_selector.stock_pool import StockPoolItem
            with db_manager.get_session() as session:
                from sqlalchemy import select
                results = session.execute(
                    select(StockPoolItem.code, StockPoolItem.name)
                    .where(StockPoolItem.code.in_(stock_codes))
                ).all()
                db_code_to_name = {code: name for code, name in results}
                # 更新名称映射（优先使用数据库中的）
                code_to_name_map.update(db_code_to_name)
        except Exception as e:
            logger.debug("批量获取股票名称失败: %s", e)
    
    return stock_codes, code_to_name_map


def batch_fetch_daily_data(
    db_manager: any,
    stock_codes: List[str],
    days: int = 200,
    min_required_days: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    批量获取股票日线数据
    
    Args:
        db_manager: 数据库管理器
        stock_codes: 股票代码列表
        days: 获取天数
        min_required_days: 最小要求天数
    
    Returns:
        股票代码到DataFrame的映射
    """
    code_to_daily_data = {}
    
    if not db_manager or not stock_codes:
        return code_to_daily_data
    
    try:
        from src.storage import StockDaily
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        with db_manager.get_session() as session:
            from sqlalchemy import select
            records = session.execute(
                select(StockDaily)
                .where(
                    StockDaily.code.in_(stock_codes),
                    StockDaily.date >= start_date,
                    StockDaily.date <= end_date
                )
                .order_by(StockDaily.code, StockDaily.date)
            ).scalars().all()
            
            # 按股票代码分组
            current_code = None
            current_records = []
            
            for record in records:
                if record.code != current_code:
                    if current_code is not None:
                        df = _convert_records_to_df(current_records, min_required_days)
                        if df is not None:
                            code_to_daily_data[current_code] = df
                    current_code = record.code
                    current_records = []
                current_records.append(record)
            
            # 处理最后一组
            if current_code is not None:
                df = _convert_records_to_df(current_records, min_required_days)
                if df is not None:
                    code_to_daily_data[current_code] = df
        
        logger.info("从数据库批量读取了 %d 只股票的日线数据", len(code_to_daily_data))
    except Exception as e:
        logger.warning("从数据库批量读取日线数据失败: %s", e)
    
    return code_to_daily_data


def _convert_records_to_df(
    records: List[any],
    min_required_days: int
) -> Optional[pd.DataFrame]:
    """
    将数据库记录转换为DataFrame并进行清洗
    
    Args:
        records: 数据库记录列表
        min_required_days: 最小要求天数
    
    Returns:
        清洗后的DataFrame，或None
    """
    if len(records) < min_required_days:
        return None
    
    df = pd.DataFrame([
        {
            'date': r.date,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'amount': r.amount,
            'pct_chg': r.pct_chg
        }
        for r in records
    ])
    
    # 数据清洗
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[df['low'] <= df['high']]
    
    if len(df) < min_required_days:
        return None
    
    return df


def get_market_index_code(stock_code: str) -> str:
    """
    根据股票代码获取对应的大盘指数代码
    
    Args:
        stock_code: 股票代码
    
    Returns:
        大盘指数代码
    """
    if stock_code.startswith(('60', '688')):
        return 'sh000001'
    elif stock_code.startswith(('00', '30')):
        return 'sz399001'
    else:
        return 'sh000001'
