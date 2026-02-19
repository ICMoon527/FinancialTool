"""模型工具函数"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .models import XGBoostModel, LightGBMModel, EnsembleModel
from .config import default_config


def load_model(model_path: str) -> object:
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        加载的模型对象
    """
    try:
        # 尝试加载模型
        if model_path.endswith('.json'):
            # XGBoost模型
            model = XGBoostModel()
            model.load(model_path)
        elif model_path.endswith('.pkl'):
            # LightGBM或集成模型
            model = joblib.load(model_path)
        else:
            # 尝试直接加载
            model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def save_model(model: object, model_path: str):
    """
    保存模型
    
    Args:
        model: 模型对象
        model_path: 模型保存路径
    """
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        if isinstance(model, XGBoostModel):
            model.save(model_path)
        else:
            joblib.dump(model, model_path)
        print(f"模型保存成功: {model_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")


def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标向量
    
    Returns:
        评估结果
    """
    try:
        return model.evaluate(X, y)
    except Exception as e:
        print(f"评估模型失败: {e}")
        return {}


def generate_model_filename(model_type: str, target_days: int) -> str:
    """
    生成模型文件名
    
    Args:
        model_type: 模型类型
        target_days: 预测目标天数
    
    Returns:
        模型文件名
    """
    import datetime
    today = datetime.datetime.now().strftime('%Y%m%d')
    return f"{model_type}_model_{target_days}d_{today}.pkl"


def get_model_path(model_type: str, target_days: int, config=default_config) -> str:
    """
    获取模型保存路径
    
    Args:
        model_type: 模型类型
        target_days: 预测目标天数
        config: 模型配置
    
    Returns:
        模型保存路径
    """
    filename = generate_model_filename(model_type, target_days)
    return os.path.join(config.model_save_dir, filename)


def prepare_stock_data(stock_data: Dict[str, pd.DataFrame], target_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备多支股票的训练数据
    
    Args:
        stock_data: 股票数据字典
        target_days: 预测目标天数
    
    Returns:
        (X, y) 特征矩阵和目标向量
    """
    from ml_models.features import FeatureExtractor
    
    extractor = FeatureExtractor()
    all_X = []
    all_y = []
    
    for ts_code, df in stock_data.items():
        if df.empty:
            continue
        
        # 准备训练数据
        X, y = extractor.prepare_training_data(df, target_days)
        if X is not None and y is not None:
            # 添加股票代码作为特征
            X['ts_code'] = ts_code
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        return None, None
    
    # 合并所有股票的数据
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    # 处理分类特征
    if 'ts_code' in X_combined.columns:
        X_combined = pd.get_dummies(X_combined, columns=['ts_code'], drop_first=True)
    
    return X_combined, y_combined


def calculate_feature_importance(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征重要性
    
    Args:
        model: 模型对象
        X: 特征矩阵
    
    Returns:
        特征重要性DataFrame
    """
    try:
        if hasattr(model, 'get_feature_importance'):
            importance_dict = model.get_feature_importance(X)
            
            # 处理特征重要性结果
            importances = []
            for model_name, importance_values in importance_dict.items():
                for i, (feature, importance) in enumerate(zip(X.columns, importance_values)):
                    importances.append({
                        'model': model_name,
                        'feature': feature,
                        'importance': importance
                    })
            
            return pd.DataFrame(importances).sort_values('importance', ascending=False)
        elif hasattr(model.model, 'feature_importances_'):
            # 单个模型的特征重要性
            importances = []
            for feature, importance in zip(X.columns, model.model.feature_importances_):
                importances.append({
                    'model': 'single',
                    'feature': feature,
                    'importance': importance
                })
            return pd.DataFrame(importances).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"计算特征重要性失败: {e}")
        return pd.DataFrame()


def optimize_model_parameters(model: object, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    优化模型参数
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标向量
    
    Returns:
        优化后的参数
    """
    try:
        from sklearn.model_selection import GridSearchCV
        
        # 定义参数网格
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            params = model.model.get_params()
            
            # 简单的参数网格
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200]
            }
            
            # 执行网格搜索
            grid_search = GridSearchCV(
                model.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_
            }
        else:
            return {}
    except Exception as e:
        print(f"优化模型参数失败: {e}")
        return {}


def create_rolling_window_splits(X: pd.DataFrame, y: pd.Series, window_size: int, step_size: int) -> List[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    创建滚动窗口分割
    
    Args:
        X: 特征矩阵
        y: 目标向量
        window_size: 窗口大小
        step_size: 步长
    
    Returns:
        滚动窗口分割列表
    """
    splits = []
    n_samples = len(X)
    
    for i in range(0, n_samples - window_size, step_size):
        # 训练窗口
        train_start = i
        train_end = i + window_size
        
        # 测试窗口
        test_start = train_end
        test_end = test_start + step_size
        
        if test_end > n_samples:
            break
        
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        splits.append((X_train, y_train, X_test, y_test))
    
    return splits


def run_rolling_backtest(model: object, X: pd.DataFrame, y: pd.Series, window_size: int, step_size: int) -> Dict:
    """
    运行滚动回测
    
    Args:
        model: 模型对象
        X: 特征矩阵
        y: 目标向量
        window_size: 窗口大小
        step_size: 步长
    
    Returns:
        回测结果
    """
    splits = create_rolling_window_splits(X, y, window_size, step_size)
    results = []
    
    for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
        # 训练模型
        model.train(X_train, y_train)
        
        # 评估模型
        eval_result = model.evaluate(X_test, y_test)
        results.append(eval_result)
    
    # 计算平均结果
    if results:
        avg_results = {
            'mean_rmse': np.mean([r['rmse'] for r in results]),
            'mean_direction_accuracy': np.mean([r['direction_accuracy'] for r in results]),
            'mean_r2': np.mean([r['r2'] for r in results]),
            'n_windows': len(results)
        }
        return avg_results
    else:
        return {}
