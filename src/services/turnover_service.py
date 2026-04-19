# -*- coding: utf-8 -*-
"""
===================================
流通股本和换手率服务
===================================

职责：
1. 从AKShare数据中反推流通股本
2. 使用流通股本计算历史换手率并保存到数据库
3. 支持流通股本数据的存储和查询
"""

import logging
from datetime import date, datetime
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from sqlalchemy import and_, select

from src.storage import DatabaseManager, StockDaily, StockBasic

logger = logging.getLogger(__name__)


class TurnoverService:
    """
    流通股本和换手率服务
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        初始化服务
        
        Args:
            db_manager: 数据库管理器（可选，默认使用单例）
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    def estimate_circulating_shares_from_akshare(
        self, 
        df: pd.DataFrame, 
        code: str,
        name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        从AKShare数据中反推流通股本
        
        使用公式：流通股本 = 成交量 / 换手率（当换手率 > 0 时）
        
        Args:
            df: 包含AKShare数据的DataFrame
            code: 股票代码
            name: 股票名称（可选）
            
        Returns:
            包含流通股本信息的字典，失败返回 None
        """
        if df is None or df.empty:
            logger.warning(f"数据为空，无法估算 {code} 的流通股本")
            return None
        
        try:
            # 确保有必要的列
            if 'volume' not in df.columns:
                logger.warning(f"数据缺少 volume 列，无法估算 {code} 的流通股本")
                return None
            
            # 尝试使用 turnover_rate 列（已经转换为小数）
            has_valid_turnover = False
            if 'turnover_rate' in df.columns:
                # 过滤掉换手率为0或异常小的值
                valid_data = df[
                    (df['turnover_rate'] > 0.0001) &  # 至少 0.01%
                    (df['volume'] > 0)
                ]
                
                if len(valid_data) >= 3:  # 至少有3个有效数据点
                    # 计算流通股本 = 成交量 / 换手率
                    shares_estimates = valid_data['volume'] / valid_data['turnover_rate']
                    
                    # 去除异常值（使用中位数，避免极端值影响）
                    circulating_shares = shares_estimates.median()
                    
                    if circulating_shares > 0:
                        has_valid_turnover = True
                        logger.info(
                            f"从 {len(valid_data)} 个有效数据点估算 {code} 流通股本: {circulating_shares:.0f} 股"
                        )
            
            # 如果没有有效的换手率数据，暂时返回None
            if not has_valid_turnover:
                logger.warning(f"没有足够的有效换手率数据，无法估算 {code} 的流通股本")
                return None
            
            # 确定日期范围
            if 'date' in df.columns:
                df_sorted = df.sort_values('date')
                start_date = pd.to_datetime(df_sorted['date'].iloc[0]).date()
                end_date = pd.to_datetime(df_sorted['date'].iloc[-1]).date()
            else:
                start_date = date(1990, 1, 1)
                end_date = date.today()
            
            return {
                'code': code,
                'name': name or code,
                'circulating_shares': circulating_shares,
                'total_shares': None,  # 暂时留空
                'start_date': start_date,
                'end_date': end_date,
                'data_source': 'EstimatedFromAKShare'
            }
            
        except Exception as e:
            logger.error(f"估算 {code} 流通股本时出错: {e}")
            return None
    
    def save_stock_basic(self, basic_info: Dict[str, Any]) -> bool:
        """
        保存股票基础信息（包括流通股本）
        
        Args:
            basic_info: 股票基础信息字典
            
        Returns:
            是否保存成功
        """
        try:
            with self.db.get_session() as session:
                # 检查是否已存在
                existing = session.execute(
                    select(StockBasic).where(
                        and_(
                            StockBasic.code == basic_info['code'],
                            StockBasic.start_date == basic_info['start_date'],
                            StockBasic.end_date == basic_info['end_date']
                        )
                    )
                ).scalar_one_or_none()
                
                if existing:
                    # 更新现有记录
                    existing.name = basic_info.get('name')
                    existing.circulating_shares = basic_info.get('circulating_shares')
                    existing.total_shares = basic_info.get('total_shares')
                    existing.data_source = basic_info.get('data_source')
                    existing.updated_at = datetime.now()
                    logger.info(f"更新 {basic_info['code']} 的流通股本信息")
                else:
                    # 创建新记录
                    record = StockBasic(
                        code=basic_info['code'],
                        name=basic_info.get('name'),
                        circulating_shares=basic_info.get('circulating_shares'),
                        total_shares=basic_info.get('total_shares'),
                        start_date=basic_info['start_date'],
                        end_date=basic_info['end_date'],
                        data_source=basic_info.get('data_source')
                    )
                    session.add(record)
                    logger.info(f"保存 {basic_info['code']} 的流通股本信息")
                
                session.commit()
                return True
                
        except Exception as e:
            logger.error(f"保存股票基础信息失败: {e}")
            return False
    
    def get_stock_basic(
        self, 
        code: str, 
        target_date: Optional[date] = None
    ) -> Optional[StockBasic]:
        """
        获取股票基础信息
        
        Args:
            code: 股票代码
            target_date: 目标日期（可选，默认今天）
            
        Returns:
            StockBasic 对象或 None
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            with self.db.get_session() as session:
                # 查找包含目标日期的记录
                result = session.execute(
                    select(StockBasic).where(
                        and_(
                            StockBasic.code == code,
                            StockBasic.start_date <= target_date,
                            StockBasic.end_date >= target_date
                        )
                    ).order_by(StockBasic.updated_at.desc()).limit(1)
                ).scalar_one_or_none()
                
                # 如果没找到，返回最新的记录
                if result is None:
                    result = session.execute(
                        select(StockBasic).where(StockBasic.code == code)
                        .order_by(StockBasic.updated_at.desc()).limit(1)
                    ).scalar_one_or_none()
                
                return result
                
        except Exception as e:
            logger.error(f"获取 {code} 的股票基础信息失败: {e}")
            return None
    
    def fill_historical_turnover(
        self, 
        code: str, 
        circulating_shares: float,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> int:
        """
        填充历史换手率
        
        使用公式：换手率 = 成交量 / 流通股本
        
        Args:
            code: 股票代码
            circulating_shares: 流通股本
            start_date: 开始日期（可选，默认获取所有数据）
            end_date: 结束日期（可选，默认今天）
            
        Returns:
            更新的记录数
        """
        if circulating_shares <= 0:
            logger.warning(f"流通股本无效，无法填充 {code} 的历史换手率")
            return 0
        
        try:
            with self.db.get_session() as session:
                # 构建查询条件
                query = select(StockDaily).where(
                    and_(
                        StockDaily.code == code,
                        StockDaily.turnover_rate.is_(None),  # 只填充没有换手率的记录
                        StockDaily.volume > 0  # 只更新有成交量的记录
                    )
                )
                
                if start_date:
                    query = query.where(StockDaily.date >= start_date)
                if end_date:
                    query = query.where(StockDaily.date <= end_date)
                
                # 查询需要更新的记录
                records = session.execute(query).scalars().all()
                
                if not records:
                    logger.debug(f"{code} 没有需要填充换手率的记录")
                    return 0
                
                # 更新换手率
                update_count = 0
                for record in records:
                    if record.volume:
                        turnover = record.volume / circulating_shares
                        record.turnover_rate = min(turnover, 1.0)  # 限制在100%以内
                        record.updated_at = datetime.now()
                        update_count += 1
                
                session.commit()
                
                logger.info(f"填充 {code} 的历史换手率: {update_count} 条")
                return update_count
                
        except Exception as e:
            logger.error(f"填充 {code} 历史换手率失败: {e}")
            return 0
    
    def process_akshare_data(
        self, 
        df: pd.DataFrame, 
        code: str,
        name: Optional[str] = None
    ) -> Tuple[bool, int]:
        """
        处理AKShare数据的完整流程
        
        1. 反推流通股本
        2. 保存流通股本信息
        3. 填充历史换手率
        
        Args:
            df: 包含AKShare数据的DataFrame
            code: 股票代码
            name: 股票名称（可选）
            
        Returns:
            (是否成功, 填充的换手率记录数)
        """
        # 首先反推流通股本
        basic_info = self.estimate_circulating_shares_from_akshare(df, code, name)
        
        if not basic_info or basic_info['circulating_shares'] <= 0:
            return False, 0
        
        # 保存流通股本信息
        saved = self.save_stock_basic(basic_info)
        
        if not saved:
            return False, 0
        
        # 填充历史换手率
        fill_count = self.fill_historical_turnover(
            code,
            basic_info['circulating_shares'],
            basic_info['start_date'],
            basic_info['end_date']
        )
        
        return True, fill_count


def get_turnover_service() -> TurnoverService:
    """
    获取流通股本和换手率服务单例
    
    Returns:
        TurnoverService 实例
    """
    return TurnoverService()
