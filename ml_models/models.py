"""机器学习模型实现"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# 尝试导入深度学习库
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    keras_available = True
except ImportError:
    keras_available = False

from ml_models.config import default_config
from ml_models.features import FeatureExtractor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class BaseModel:
    """基础模型类"""
    
    def __init__(self, config=default_config):
        """初始化基础模型"""
        self.config = config
        self.model = None
        self.feature_extractor = FeatureExtractor(config)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        raise NotImplementedError
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        # 计算预测准确率（方向正确的比例）
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(y))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """交叉验证"""
        raise NotImplementedError
    
    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError
    
    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        模型超参数调优
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            调优结果
        """
        raise NotImplementedError


class XGBoostModel(BaseModel):
    """XGBoost模型"""
    
    def __init__(self, config=default_config):
        """初始化XGBoost模型"""
        super().__init__(config)
        if xgb is None:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
        self.model = xgb.XGBRegressor(**config.xgb_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练XGBoost模型"""
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # 训练模型
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # 评估模型
        eval_result = self.evaluate(X_test, y_test)
        return eval_result['rmse']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """交叉验证"""
        scores = cross_val_score(
            self.model, X, y, cv=self.config.cv_folds, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(-scores)
        
        return {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'min_rmse': np.min(rmse_scores),
            'max_rmse': np.max(rmse_scores)
        }
    
    def save(self, path: str):
        """保存模型"""
        self.model.save_model(path)
    
    def load(self, path: str):
        """加载模型"""
        self.model.load_model(path)
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        XGBoost模型超参数调优
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            调优结果
        """
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }
        
        # 创建基础模型
        base_model = xgb.XGBRegressor(random_state=self.config.random_state)
        
        # 使用GridSearchCV进行参数搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # 执行搜索
        grid_search.fit(X, y)
        
        # 更新模型为最佳参数模型
        self.model = grid_search.best_estimator_
        
        # 返回调优结果
        return {
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }


class LightGBMModel(BaseModel):
    """LightGBM模型"""
    
    def __init__(self, config=default_config):
        """初始化LightGBM模型"""
        super().__init__(config)
        if lgb is None:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'")
        self.model = lgb.LGBMRegressor(**config.lgb_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练LightGBM模型"""
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # 训练模型
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # 评估模型
        eval_result = self.evaluate(X_test, y_test)
        return eval_result['rmse']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """交叉验证"""
        scores = cross_val_score(
            self.model, X, y, cv=self.config.cv_folds, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(-scores)
        
        return {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'min_rmse': np.min(rmse_scores),
            'max_rmse': np.max(rmse_scores)
        }
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """加载模型"""
        import joblib
        self.model = joblib.load(path)
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        LightGBM模型超参数调优
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            调优结果
        """
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.2],
            'reg_lambda': [0.1, 1, 10]
        }
        
        # 创建基础模型
        base_model = lgb.LGBMRegressor(random_state=self.config.random_state)
        
        # 使用GridSearchCV进行参数搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # 执行搜索
        grid_search.fit(X, y)
        
        # 更新模型为最佳参数模型
        self.model = grid_search.best_estimator_
        
        # 返回调优结果
        return {
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }


class DeepLearningModel(BaseModel):
    """深度学习基础模型"""
    
    def __init__(self, config=default_config):
        """初始化深度学习模型"""
        super().__init__(config)
        if not keras_available:
            raise ImportError("TensorFlow/Keras is not installed. Please install it with 'pip install tensorflow'")
        self.model = None
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        Args:
            X: 特征矩阵
            y: 目标变量
            sequence_length: 序列长度
        
        Returns:
            (X_seq, y_seq) 序列特征和目标
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X.iloc[i:i+sequence_length].values)
            y_seq.append(y.iloc[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """交叉验证"""
        # 简单的时间序列交叉验证
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        rmse_scores = []
        direction_accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 创建序列数据
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)
            
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                continue
            
            # 训练模型
            self.model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
            
            # 评估模型
            predictions = self.model.predict(X_test_seq, verbose=0)
            mse = mean_squared_error(y_test_seq, predictions)
            rmse = np.sqrt(mse)
            direction_accuracy = np.mean(np.sign(predictions) == np.sign(y_test_seq))
            
            rmse_scores.append(rmse)
            direction_accuracies.append(direction_accuracy)
        
        return {
            'mean_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'mean_direction_accuracy': np.mean(direction_accuracies)
        }
    
    def save(self, path: str):
        """保存模型"""
        if self.model:
            self.model.save(path)
    
    def load(self, path: str):
        """加载模型"""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)


class LSTMModel(DeepLearningModel):
    """LSTM模型"""
    
    def __init__(self, config=default_config):
        """初始化LSTM模型"""
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 10)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练LSTM模型"""
        # 创建序列数据
        X_seq, y_seq = self._create_sequences(X, y, self.sequence_length)
        
        if len(X_seq) == 0:
            return float('inf')
        
        # 构建LSTM模型
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        
        # 编译模型
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # 训练模型
        self.model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)
        
        # 评估模型
        predictions = self.model.predict(X_seq, verbose=0)
        mse = mean_squared_error(y_seq, predictions)
        rmse = np.sqrt(mse)
        
        return rmse
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        # 创建序列数据
        X_seq = []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X.iloc[i:i+self.sequence_length].values)
        
        if len(X_seq) == 0:
            return np.zeros(len(X))
        
        X_seq = np.array(X_seq)
        predictions = self.model.predict(X_seq, verbose=0)
        
        # 填充预测结果
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length-1:] = predictions.flatten()
        
        return full_predictions


class GRUModel(DeepLearningModel):
    """GRU模型"""
    
    def __init__(self, config=default_config):
        """初始化GRU模型"""
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 10)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练GRU模型"""
        # 创建序列数据
        X_seq, y_seq = self._create_sequences(X, y, self.sequence_length)
        
        if len(X_seq) == 0:
            return float('inf')
        
        # 构建GRU模型
        self.model = Sequential()
        self.model.add(GRU(units=50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(GRU(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        
        # 编译模型
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # 训练模型
        self.model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)
        
        # 评估模型
        predictions = self.model.predict(X_seq, verbose=0)
        mse = mean_squared_error(y_seq, predictions)
        rmse = np.sqrt(mse)
        
        return rmse
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        # 创建序列数据
        X_seq = []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X.iloc[i:i+self.sequence_length].values)
        
        if len(X_seq) == 0:
            return np.zeros(len(X))
        
        X_seq = np.array(X_seq)
        predictions = self.model.predict(X_seq, verbose=0)
        
        # 填充预测结果
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length-1:] = predictions.flatten()
        
        return full_predictions


class EnsembleModel(BaseModel):
    """模型集成"""
    
    def __init__(self, config=default_config):
        """初始化集成模型"""
        super().__init__(config)
        self.models = {
            'xgboost': XGBoostModel(config),
            'lightgbm': LightGBMModel(config)
        }
        # 如果Keras可用，添加深度学习模型
        if keras_available:
            self.models['lstm'] = LSTMModel(config)
            self.models['gru'] = GRUModel(config)
        self.weights = config.model_weights
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """训练集成模型"""
        # 训练每个模型
        rmse_values = {}
        for model_name, model in self.models.items():
            rmse = model.train(X, y)
            rmse_values[model_name] = rmse
        
        # 计算加权平均RMSE
        weighted_rmse = sum(rmse_values[model_name] * self.weights[model_name] for model_name in self.models)
        return weighted_rmse
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        # 获取每个模型的预测结果
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        # 加权平均预测结果
        weighted_predictions = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weighted_predictions += pred * self.weights[model_name]
        
        return weighted_predictions
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """交叉验证"""
        # 对每个模型进行交叉验证
        cv_results = {}
        for model_name, model in self.models.items():
            cv_results[model_name] = model.cross_validate(X, y)
        
        # 计算集成模型的交叉验证结果
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # 训练模型
        self.train(X_train, y_train)
        
        # 评估集成模型
        eval_result = self.evaluate(X_test, y_test)
        
        return {
            'ensemble_rmse': eval_result['rmse'],
            'ensemble_direction_accuracy': eval_result['direction_accuracy'],
            'individual_results': cv_results
        }
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump(self, path)
    
    def load(self, path: str):
        """加载模型"""
        import joblib
        loaded_model = joblib.load(path)
        self.models = loaded_model.models
        self.weights = loaded_model.weights
        self.config = loaded_model.config
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新模型权重"""
        self.weights = new_weights
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """获取特征重要性"""
        importance = {}
        for model_name, model in self.models.items():
            if hasattr(model.model, 'feature_importances_'):
                importance[model_name] = model.model.feature_importances_
        return importance
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        集成模型超参数调优
        
        Args:
            X: 特征矩阵
            y: 目标变量
        
        Returns:
            调优结果
        """
        tuning_results = {}
        
        # 对每个子模型进行调优
        for model_name, model in self.models.items():
            print(f"Tuning {model_name}...")
            tuning_results[model_name] = model.hyperparameter_tuning(X, y)
        
        # 重新训练集成模型
        print("Retraining ensemble model...")
        ensemble_score = self.train(X, y)
        
        # 返回调优结果
        return {
            'individual_models': tuning_results,
            'ensemble_score': ensemble_score
        }
