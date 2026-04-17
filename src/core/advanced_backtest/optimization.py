# -*- coding: utf-8 -*-
"""
===================================
参数优化模块
===================================

提供网格搜索、贝叶斯优化、过拟合检测等功能。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import itertools
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"  # 网格搜索
    RANDOM_SEARCH = "random_search"  # 随机搜索
    # BAYESIAN = "bayesian"  # 贝叶斯优化（预留）


class OverfitCheckMethod(Enum):
    """过拟合检测方法"""
    WALK_FORWARD = "walk_forward"  # 滚动窗口验证
    TRAIN_TEST_SPLIT = "train_test_split"  # 训练测试分割


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    param_type: str  # "int", "float", "categorical"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    values: Optional[List[Any]] = None
    description: str = ""
    
    def __post_init__(self):
        if self.param_type == "categorical" and self.values is None:
            raise ValueError("Categorical parameter must have values")
        if self.param_type in ["int", "float"] and (self.min_value is None or self.max_value is None):
            raise ValueError("Numeric parameter must have min_value and max_value")


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    optimization_method: OptimizationMethod
    total_trials: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    overfit_check_passed: bool = False
    overfit_score: float = 0.0


@dataclass
class WalkForwardResult:
    """滚动窗口验证结果"""
    train_metrics: List[Dict[str, float]]
    test_metrics: List[Dict[str, float]]
    average_train_metrics: Dict[str, float]
    average_test_metrics: Dict[str, float]
    overfit_score: float  # (测试 - 训练) / |训练|


class ParameterOptimizer(ABC):
    """参数优化器基类"""
    
    def __init__(
        self,
        objective_function: Callable,
        param_ranges: List[ParameterRange],
        metric_name: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        """
        初始化参数优化器
        
        Args:
            objective_function: 目标函数（接受参数字典，返回指标字典）
            param_ranges: 参数范围列表
            metric_name: 优化目标指标名称
            maximize: 是否最大化指标
        """
        self.objective_function = objective_function
        self.param_ranges = param_ranges
        self.param_name_to_range = {p.name: p for p in param_ranges}
        self.metric_name = metric_name
        self.maximize = maximize
        
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
    
    @abstractmethod
    def optimize(self, max_trials: Optional[int] = None) -> OptimizationResult:
        """
        执行优化
        
        Args:
            max_trials: 最大试验次数
            
        Returns:
            优化结果
        """
        pass
    
    def _evaluate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        评估一组参数
        
        Args:
            params: 参数字典
            
        Returns:
            指标字典
        """
        try:
            metrics = self.objective_function(params)
            return metrics
        except Exception as e:
            logger.warning(f"参数评估失败: {params}, 错误: {e}")
            return {self.metric_name: -np.inf if self.maximize else np.inf}
    
    def _update_best(self, params: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        更新最佳参数
        
        Args:
            params: 参数字典
            metrics: 指标字典
        """
        current_value = metrics.get(self.metric_name)
        if current_value is None:
            return
        
        if self.best_metric_value is None:
            self.best_metric_value = current_value
            self.best_params = params.copy()
        else:
            if self.maximize:
                if current_value > self.best_metric_value:
                    self.best_metric_value = current_value
                    self.best_params = params.copy()
            else:
                if current_value < self.best_metric_value:
                    self.best_metric_value = current_value
                    self.best_params = params.copy()


class GridSearchOptimizer(ParameterOptimizer):
    """网格搜索优化器"""
    
    def __init__(
        self,
        objective_function: Callable,
        param_ranges: List[ParameterRange],
        metric_name: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        super().__init__(objective_function, param_ranges, metric_name, maximize)
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """
        生成所有参数组合
        
        Returns:
            参数字典列表
        """
        param_values = {}
        for param_range in self.param_ranges:
            if param_range.param_type == "categorical":
                param_values[param_range.name] = param_range.values
            elif param_range.param_type == "int":
                start = int(param_range.min_value)
                end = int(param_range.max_value)
                step = int(param_range.step) if param_range.step else 1
                param_values[param_range.name] = list(range(start, end + 1, step))
            elif param_range.param_type == "float":
                start = param_range.min_value
                end = param_range.max_value
                step = param_range.step if param_range.step else (end - start) / 10
                param_values[param_range.name] = np.arange(start, end + step, step).tolist()
        
        # 生成所有组合
        keys = list(param_values.keys())
        values_list = list(param_values.values())
        combinations = []
        for combination in itertools.product(*values_list):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def optimize(self, max_trials: Optional[int] = None) -> OptimizationResult:
        """
        执行网格搜索
        
        Args:
            max_trials: 最大试验次数
            
        Returns:
            优化结果
        """
        start_time = datetime.now()
        logger.info(f"开始网格搜索优化，目标指标: {self.metric_name}")
        
        combinations = self._generate_param_combinations()
        total_combinations = len(combinations)
        logger.info(f"参数组合总数: {total_combinations}")
        
        if max_trials and max_trials < total_combinations:
            combinations = combinations[:max_trials]
            logger.info(f"限制为 {max_trials} 次试验")
        
        for i, params in enumerate(combinations):
            logger.debug(f"试验 {i+1}/{len(combinations)}: {params}")
            
            metrics = self._evaluate(params)
            self._update_best(params, metrics)
            
            result_entry = {
                "trial": i + 1,
                "params": params.copy(),
                "metrics": metrics,
            }
            self.results.append(result_entry)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"网格搜索完成，耗时: {duration:.2f} 秒")
        logger.info(f"最佳参数: {self.best_params}")
        logger.info(f"最佳指标值: {self.best_metric_value}")
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_metrics=self.results[-1]["metrics"] if self.results else {},
            all_results=self.results,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            total_trials=len(self.results),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )


class RandomSearchOptimizer(ParameterOptimizer):
    """随机搜索优化器"""
    
    def __init__(
        self,
        objective_function: Callable,
        param_ranges: List[ParameterRange],
        metric_name: str = "sharpe_ratio",
        maximize: bool = True,
        random_seed: int = 42,
    ):
        super().__init__(objective_function, param_ranges, metric_name, maximize)
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """
        随机采样参数
        
        Returns:
            参数字典
        """
        params = {}
        for param_range in self.param_ranges:
            if param_range.param_type == "categorical":
                params[param_range.name] = np.random.choice(param_range.values)
            elif param_range.param_type == "int":
                low = int(param_range.min_value)
                high = int(param_range.max_value)
                params[param_range.name] = np.random.randint(low, high + 1)
            elif param_range.param_type == "float":
                low = param_range.min_value
                high = param_range.max_value
                params[param_range.name] = np.random.uniform(low, high)
        return params
    
    def optimize(self, max_trials: int = 50) -> OptimizationResult:
        """
        执行随机搜索
        
        Args:
            max_trials: 最大试验次数
            
        Returns:
            优化结果
        """
        start_time = datetime.now()
        logger.info(f"开始随机搜索优化，目标指标: {self.metric_name}")
        logger.info(f"试验次数: {max_trials}")
        
        for i in range(max_trials):
            params = self._sample_random_params()
            logger.debug(f"试验 {i+1}/{max_trials}: {params}")
            
            metrics = self._evaluate(params)
            self._update_best(params, metrics)
            
            result_entry = {
                "trial": i + 1,
                "params": params.copy(),
                "metrics": metrics,
            }
            self.results.append(result_entry)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"随机搜索完成，耗时: {duration:.2f} 秒")
        logger.info(f"最佳参数: {self.best_params}")
        logger.info(f"最佳指标值: {self.best_metric_value}")
        
        return OptimizationResult(
            best_params=self.best_params or {},
            best_metrics=self.results[-1]["metrics"] if self.results else {},
            all_results=self.results,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            total_trials=len(self.results),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )


class OverfitDetector:
    """过拟合检测器"""
    
    def __init__(
        self,
        metric_name: str = "sharpe_ratio",
        threshold: float = 0.3,  # 过拟合阈值
    ):
        self.metric_name = metric_name
        self.threshold = threshold
    
    def check_train_test(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ) -> Tuple[bool, float]:
        """
        检查训练测试过拟合
        
        Args:
            train_metrics: 训练集指标
            test_metrics: 测试集指标
            
        Returns:
            (是否过拟合, 过拟合分数)
        """
        train_value = train_metrics.get(self.metric_name, 0)
        test_value = test_metrics.get(self.metric_name, 0)
        
        if abs(train_value) < 1e-6:
            return False, 0.0
        
        # 计算过拟合分数: (测试 - 训练) / |训练|
        overfit_score = (test_value - train_value) / abs(train_value)
        
        # 如果测试指标显著低于训练，认为过拟合
        is_overfit = overfit_score < -self.threshold
        
        return is_overfit, overfit_score
    
    def walk_forward_validation(
        self,
        objective_function: Callable,
        params: Dict[str, Any],
        windows: int = 5,
    ) -> WalkForwardResult:
        """
        滚动窗口验证（Walk-Forward Validation）
        
        Args:
            objective_function: 目标函数（接受窗口参数）
            params: 参数字典
            windows: 窗口数量
            
        Returns:
            滚动窗口验证结果
        """
        train_metrics_list = []
        test_metrics_list = []
        
        for i in range(windows):
            # 这里应该实现真正的滚动窗口逻辑
            # 简化实现：模拟训练和测试
            # 实际应用中需要传入数据分割逻辑
            train_metrics = objective_function({**params, "window": i, "phase": "train"})
            test_metrics = objective_function({**params, "window": i, "phase": "test"})
            
            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)
        
        # 计算平均指标
        def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {key: np.mean([m.get(key, 0) for m in metrics_list]) for key in keys}
        
        avg_train = average_metrics(train_metrics_list)
        avg_test = average_metrics(test_metrics_list)
        
        # 计算过拟合分数
        train_value = avg_train.get(self.metric_name, 0)
        test_value = avg_test.get(self.metric_name, 0)
        
        if abs(train_value) < 1e-6:
            overfit_score = 0.0
        else:
            overfit_score = (test_value - train_value) / abs(train_value)
        
        return WalkForwardResult(
            train_metrics=train_metrics_list,
            test_metrics=test_metrics_list,
            average_train_metrics=avg_train,
            average_test_metrics=avg_test,
            overfit_score=overfit_score,
        )


def create_optimizer(
    method: OptimizationMethod,
    objective_function: Callable,
    param_ranges: List[ParameterRange],
    metric_name: str = "sharpe_ratio",
    maximize: bool = True,
    **kwargs,
) -> ParameterOptimizer:
    """
    工厂函数：创建优化器
    
    Args:
        method: 优化方法
        objective_function: 目标函数
        param_ranges: 参数范围
        metric_name: 优化指标
        maximize: 是否最大化
        **kwargs: 其他参数
    
    Returns:
        参数优化器实例
    """
    if method == OptimizationMethod.GRID_SEARCH:
        return GridSearchOptimizer(
            objective_function, param_ranges, metric_name, maximize)
    elif method == OptimizationMethod.RANDOM_SEARCH:
        return RandomSearchOptimizer(
            objective_function, param_ranges, metric_name, maximize,
            random_seed=kwargs.get("random_seed", 42))
    else:
        raise ValueError(f"未知的优化方法: {method}")
