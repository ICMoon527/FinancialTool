"""模型配置管理"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """模型配置类"""
    # 特征提取配置
    feature_window: int = 60  # 特征提取窗口大小
    short_term_window: int = 5  # 短期窗口大小
    medium_term_window: int = 20  # 中期窗口大小
    long_term_window: int = 60  # 长期窗口大小
    sequence_length: int = 10  # 深度学习模型序列长度
    
    # 模型训练配置
    test_size: float = 0.2  # 测试集比例
    random_state: int = 42  # 随机种子
    cv_folds: int = 5  # 交叉验证折数
    
    # XGBoost配置
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    })
    
    # LightGBM配置
    lgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'regression',
        'metric': 'rmse'
    })
    
    # 模型集成配置
    model_weights: Dict = field(default_factory=lambda: {
        'xgboost': 0.5,
        'lightgbm': 0.5
    })
    
    # 技术分析与机器学习权重
    traditional_weight: float = 0.5  # 传统技术分析权重
    ml_weight: float = 0.5  # 机器学习预测权重
    
    # 模型保存路径
    model_save_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保模型保存目录存在
        os.makedirs(self.model_save_dir, exist_ok=True)


# 创建默认配置实例
default_config = ModelConfig()
