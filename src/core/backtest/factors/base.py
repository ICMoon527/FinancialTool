# -*- coding: utf-8 -*-
"""
===================================
因子定义与因子组合模块
===================================

提供因子定义、因子计算和因子组合功能。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """因子类型"""
    TECHNICAL = "technical"  # 技术因子
    FUNDAMENTAL = "fundamental"  # 基本面因子
    SENTIMENT = "sentiment"  # 情绪因子
    MARKET = "market"  # 市场因子
    CUSTOM = "custom"  # 自定义因子


class FactorDirection(Enum):
    """因子方向"""
    LONG = "long"  # 看多（因子值越大越好）
    SHORT = "short"  # 看空（因子值越小越好）
    NEUTRAL = "neutral"  # 中性（方向不确定）


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    factor_type: FactorType
    direction: FactorDirection = FactorDirection.NEUTRAL
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True


class Factor(ABC):
    """因子基类"""
    
    def __init__(self, config: FactorConfig):
        self.config = config
        self.name = config.name
        self.factor_type = config.factor_type
        self.direction = config.direction
        self.weight = config.weight
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 股票数据 DataFrame
            
        Returns:
            因子值 Series
        """
        pass
    
    def normalize(self, factor_values: pd.Series) -> pd.Series:
        """
        标准化因子值（Z-score）
        
        Args:
            factor_values: 因子值 Series
            
        Returns:
            标准化后的因子值
        """
        mean = factor_values.mean()
        std = factor_values.std()
        if std == 0:
            return pd.Series(0.0, index=factor_values.index)
        return (factor_values - mean) / std
    
    def neutralize(self, factor_values: pd.Series, market_cap: Optional[pd.Series] = None) -> pd.Series:
        """
        市值中性化（可选）
        
        Args:
            factor_values: 因子值 Series
            market_cap: 市值 Series（可选）
            
        Returns:
            中性化后的因子值
        """
        if market_cap is None:
            return factor_values
        
        # 简单的市值分位数中性化
        market_cap_quantiles = pd.qcut(market_cap, q=5, labels=False, duplicates='drop')
        
        # 按市值分位数去均值
        neutralized = factor_values.copy()
        for q in market_cap_quantiles.unique():
            mask = market_cap_quantiles == q
            neutralized[mask] = factor_values[mask] - factor_values[mask].mean()
        
        return neutralized


class MomentumFactor(Factor):
    """动量因子"""
    
    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self.lookback_period = config.parameters.get("lookback_period", 20)
        self.skip_period = config.parameters.get("skip_period", 1)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算动量因子（过去 N 天收益率，跳过最近 M 天）"""
        if 'close' not in data.columns:
            raise ValueError("数据缺少 'close' 列")
        
        returns = data['close'].pct_change(self.lookback_period)
        # 跳过最近 skip_period 天
        returns = returns.shift(self.skip_period)
        return returns


class ValueFactor(Factor):
    """估值因子（基于价格和均线的比值）"""
    
    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self.ma_period = config.parameters.get("ma_period", 20)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算估值因子（价格 / MA）"""
        if 'close' not in data.columns:
            raise ValueError("数据缺少 'close' 列")
        
        ma = data['close'].rolling(self.ma_period).mean()
        value_factor = data['close'] / ma
        return value_factor


class VolatilityFactor(Factor):
    """波动率因子"""
    
    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self.lookback_period = config.parameters.get("lookback_period", 20)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率因子（日收益率标准差）"""
        if 'close' not in data.columns:
            raise ValueError("数据缺少 'close' 列")
        
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.lookback_period).std()
        return volatility


class VolumeFactor(Factor):
    """成交量因子"""
    
    def __init__(self, config: FactorConfig):
        super().__init__(config)
        self.lookback_period = config.parameters.get("lookback_period", 20)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量因子（当前成交量 / 平均成交量）"""
        if 'volume' not in data.columns:
            raise ValueError("数据缺少 'volume' 列")
        
        avg_volume = data['volume'].rolling(self.lookback_period).mean()
        volume_factor = data['volume'] / avg_volume
        return volume_factor


class FactorCombination:
    """
    因子组合器
    
    支持多种因子组合方式：
    - 等权平均
    - 加权平均
    - 分位数合成
    - 机器学习合成（预留接口）
    """
    
    class CombinationMethod(Enum):
        """组合方法"""
        EQUAL_WEIGHT = "equal_weight"  # 等权
        WEIGHTED = "weighted"  # 加权
        QUANTILE = "quantile"  # 分位数
        RANK = "rank"  # 排名
    
    def __init__(
        self,
        factors: List[Factor],
        method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
        normalize: bool = True,
    ):
        self.factors = factors
        self.method = method
        self.normalize = normalize
    
    def combine(self, data: pd.DataFrame) -> pd.Series:
        """
        组合多个因子
        
        Args:
            data: 股票数据 DataFrame
            
        Returns:
            组合后的因子值
        """
        factor_values = {}
        
        # 计算每个因子
        for factor in self.factors:
            if not factor.config.enabled:
                continue
            
            values = factor.calculate(data)
            
            # 标准化
            if self.normalize:
                values = factor.normalize(values)
            
            # 调整方向（统一为 LONG 方向）
            if factor.direction == FactorDirection.SHORT:
                values = -values
            
            factor_values[factor.name] = values
        
        if not factor_values:
            return pd.Series(0.0, index=data.index)
        
        # 组合因子
        df = pd.DataFrame(factor_values)
        
        if self.method == self.CombinationMethod.EQUAL_WEIGHT:
            return df.mean(axis=1)
        elif self.method == self.CombinationMethod.WEIGHTED:
            weights = {f.name: f.weight for f in self.factors if f.config.enabled}
            return df.apply(lambda x: sum(x[col] * weights[col] for col in df.columns), axis=1)
        elif self.method == self.CombinationMethod.QUANTILE:
            # 分位数合成：每个因子转换为分位数，然后平均
            quantile_df = df.rank(pct=True)
            return quantile_df.mean(axis=1)
        elif self.method == self.CombinationMethod.RANK:
            # 排名合成：每个因子转换为排名，然后平均
            rank_df = df.rank()
            return rank_df.mean(axis=1)
        else:
            raise ValueError(f"未知的组合方法: {self.method}")


def create_factor(name: str, factor_type: FactorType, **kwargs) -> Factor:
    """
    工厂函数：创建因子实例
    
    Args:
        name: 因子名称
        factor_type: 因子类型
        **kwargs: 其他参数
        
    Returns:
        Factor 实例
    """
    config = FactorConfig(
        name=name,
        factor_type=factor_type,
        direction=kwargs.get("direction", FactorDirection.NEUTRAL),
        weight=kwargs.get("weight", 1.0),
        parameters=kwargs.get("parameters", {}),
        description=kwargs.get("description", ""),
    )
    
    # 根据名称创建对应因子
    if "momentum" in name.lower():
        return MomentumFactor(config)
    elif "value" in name.lower():
        return ValueFactor(config)
    elif "volatility" in name.lower() or "vol" in name.lower():
        return VolatilityFactor(config)
    elif "volume" in name.lower():
        return VolumeFactor(config)
    else:
        raise ValueError(f"未知的因子类型: {name}")
