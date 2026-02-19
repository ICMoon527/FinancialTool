"""特征工程模块"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .config import default_config
from strategy.factors import factor_library


class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, config=default_config):
        """初始化特征提取器"""
        self.config = config
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从股票历史数据中提取特征
        
        Args:
            df: 股票历史数据
        
        Returns:
            包含提取特征的DataFrame
        """
        if df.empty:
            return df
        
        # 创建特征副本
        features = df.copy()
        
        # 1. 计算技术指标
        features = self._extract_technical_indicators(features)
        
        # 2. 计算量能指标
        features = self._extract_volume_indicators(features)
        
        # 3. 计算动量指标
        features = self._extract_momentum_indicators(features)
        
        # 4. 计算波动率指标
        features = self._extract_volatility_indicators(features)
        
        # 5. 计算市场情绪指标
        features = self._extract_sentiment_indicators(features)
        
        # 6. 计算时间相关特征
        features = self._extract_time_features(features)
        
        # 7. 计算交叉特征
        features = self._extract_cross_features(features)
        
        return features
    
    def _extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
        # 使用现有的factor_library计算技术指标
        df = factor_library.calculate_all_factors(df)
        
        # 计算额外的技术指标
        if 'close' in df.columns:
            # 计算价格变化率
            df['price_change'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(5)
            df['price_change_20d'] = df['close'].pct_change(20)
        
        return df
    
    def _extract_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取量能指标特征"""
        if 'volume' in df.columns:
            # 成交量变化率
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change_5d'] = df['volume'].pct_change(5)
            df['volume_change_20d'] = df['volume'].pct_change(20)
            
            # 成交量均值
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            df['volume_ma20'] = df['volume'].rolling(20).mean()
            
            # 量价配合指标
            if 'price_change' in df.columns:
                df['volume_price_corr'] = df['volume'].rolling(20).corr(df['price_change'])
        
        return df
    
    def _extract_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取动量指标特征"""
        if 'close' in df.columns:
            # 动量指标
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            
            # ROC指标
            df['roc_5'] = ((df['close'] / df['close'].shift(5)) - 1) * 100
            df['roc_20'] = ((df['close'] / df['close'].shift(20)) - 1) * 100
        
        return df
    
    def _extract_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取波动率指标特征"""
        if 'close' in df.columns:
            # 历史波动率
            df['volatility_5d'] = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
            df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # 真实波动幅度均值（ATR）已由factor_library计算
        
        return df
    
    def _extract_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取市场情绪指标特征"""
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # 涨跌比
            df['price_range'] = df['high'] - df['low']
            df['price_range_ratio'] = df['price_range'] / df['close'].shift(1)
            
            # 收盘位置
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间相关特征"""
        if 'trade_date' in df.columns:
            # 转换为日期时间类型
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                try:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                except:
                    pass
            
            if pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                # 提取年、月、日、星期等时间特征
                df['year'] = df['trade_date'].dt.year
                df['month'] = df['trade_date'].dt.month
                df['day'] = df['trade_date'].dt.day
                df['weekday'] = df['trade_date'].dt.weekday
                df['is_month_end'] = df['trade_date'].dt.is_month_end.astype(int)
                df['is_quarter_end'] = df['trade_date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def _extract_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取交叉特征"""
        # 计算均线交叉特征
        if all(col in df.columns for col in ['MA5', 'MA10']):
            df['ma5_ma10_cross'] = (df['MA5'] > df['MA10']).astype(int)
            df['ma5_ma10_diff'] = (df['MA5'] - df['MA10']) / df['MA10']
        
        if all(col in df.columns for col in ['MA10', 'MA20']):
            df['ma10_ma20_cross'] = (df['MA10'] > df['MA20']).astype(int)
            df['ma10_ma20_diff'] = (df['MA10'] - df['MA20']) / df['MA20']
        
        # 计算量价配合特征
        if all(col in df.columns for col in ['volume_change', 'price_change']):
            df['volume_price_confirm'] = ((df['volume_change'] > 0) & (df['price_change'] > 0)).astype(int)
            df['volume_price_divergence'] = ((df['volume_change'] < 0) & (df['price_change'] > 0)).astype(int)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        获取所有特征名称
        
        Args:
            df: 包含特征的DataFrame
        
        Returns:
            特征名称列表
        """
        # 排除非特征列
        exclude_cols = ['ts_code', 'name', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        feature_names = [col for col in df.columns if col not in exclude_cols]
        return feature_names
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'correlation', threshold: float = 0.1) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            X: 特征矩阵
            y: 目标变量
            method: 特征选择方法 ('correlation', 'importance')
            threshold: 选择阈值
        
        Returns:
            选择后的特征矩阵
        """
        if method == 'correlation':
            # 基于与目标变量的相关性进行选择
            corr_matrix = X.corrwith(y)
            selected_features = corr_matrix[abs(corr_matrix) > threshold].index.tolist()
        else:
            # 默认返回所有特征
            selected_features = X.columns.tolist()
        
        return X[selected_features]
    
    def _reduce_dimensionality(self, X: pd.DataFrame, method: str = 'pca', n_components: int = 20) -> pd.DataFrame:
        """
        特征降维
        
        Args:
            X: 特征矩阵
            method: 降维方法 ('pca')
            n_components: 降维后的特征数量
        
        Returns:
            降维后的特征矩阵
        """
        from sklearn.decomposition import PCA
        
        if method == 'pca' and X.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)
            X_reduced = pd.DataFrame(X_reduced, columns=[f'pca_{i}' for i in range(n_components)])
            return X_reduced
        
        return X
    
    def prepare_training_data(self, df: pd.DataFrame, target_days: int = 1, feature_selection: bool = True, dimensionality_reduction: bool = False) -> tuple:
        """
        准备训练数据
        
        Args:
            df: 股票历史数据
            target_days: 预测目标天数
            feature_selection: 是否进行特征选择
            dimensionality_reduction: 是否进行降维
        
        Returns:
            (X, y) 特征矩阵和目标向量
        """
        # 提取特征
        features = self.extract_features(df)
        
        # 生成目标变量：未来target_days的收益率
        if 'close' in features.columns:
            features['target'] = features['close'].shift(-target_days) / features['close'] - 1
        else:
            features['target'] = 0
        
        # 获取特征名称
        feature_names = self.get_feature_names(features)
        
        # 移除NaN值
        features = features.dropna()
        
        if len(features) == 0:
            return None, None
        
        # 分离特征和目标
        X = features[feature_names]
        y = features['target']
        
        # 特征选择
        if feature_selection:
            X = self._select_features(X, y)
        
        # 特征降维
        if dimensionality_reduction:
            X = self._reduce_dimensionality(X)
        
        return X, y
