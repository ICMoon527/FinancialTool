# -*- coding: utf-8 -*-
"""
===================================
增强的熔断器实现
===================================

设计目标：
1. 统一管理所有数据源的熔断状态
2. 支持灵活的配置
3. 提供详细的状态追踪
4. 线程安全

状态机：
CLOSED -> OPEN (failure count > threshold)
OPEN -> HALF_OPEN (recovery timeout)
HALF_OPEN -> CLOSED (success)
HALF_OPEN -> OPEN (failure)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, Callable, Type, Tuple
from functools import wraps

from .config import get_config, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitMetrics:
    """熔断器指标收集"""
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_transitions: int = 0

    def record_request(self) -> None:
        self.total_requests += 1

    def record_success(self) -> None:
        self.success_count += 1
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()

    def reset_failures(self) -> None:
        self.failure_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'state_transitions': self.state_transitions,
        }


class CircuitBreaker:
    """
    增强的熔断器实现

    特点：
    - 线程安全
    - 支持配置化
    - 详细的指标收集
    - 支持半开状态试探
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or get_config().circuit_breaker
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._lock = threading.RLock()
        self._open_time: Optional[float] = None
        self._failure_count = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        with self._lock:
            return CircuitMetrics(**self._metrics.__dict__)

    def is_available(self) -> bool:
        """检查是否可以请求"""
        with self._lock:
            self._check_state_transition()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        """记录成功"""
        with self._lock:
            self._metrics.record_success()
            self._metrics.record_request()

            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"[{self.name}] 熔断器从 HALF_OPEN 切换到 CLOSED")
                self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """记录失败"""
        with self._lock:
            self._metrics.record_failure()
            self._metrics.record_request()
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"[{self.name}] 熔断器从 HALF_OPEN 切换到 OPEN (试探失败)")
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"[{self.name}] 熔断器触发: 失败次数 {self._failure_count} >= "
                        f"阈值 {self.config.failure_threshold}"
                    )
                    self._transition_to(CircuitState.OPEN)

    def _check_state_transition(self) -> None:
        """检查是否需要状态转换"""
        if self._state == CircuitState.OPEN:
            if self._open_time is not None:
                elapsed = time.time() - self._open_time
                if elapsed >= self.config.recovery_timeout_seconds:
                    logger.info(f"[{self.name}] 熔断器从 OPEN 切换到 HALF_OPEN (恢复超时)")
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """转换状态"""
        old_state = self._state
        self._state = new_state
        self._metrics.state_transitions += 1

        if new_state == CircuitState.OPEN:
            self._open_time = time.time()
        elif new_state == CircuitState.CLOSED:
            self._open_time = None
            self._failure_count = 0

        logger.debug(
            f"[{self.name}] 熔断器状态变更: {old_state.value} -> {new_state.value}"
        )

    def reset(self) -> None:
        """重置熔断器状态"""
        with self._lock:
            logger.info(f"[{self.name}] 熔断器重置")
            self._state = CircuitState.CLOSED
            self._metrics = CircuitMetrics()
            self._open_time = None
            self._failure_count = 0

    def get_state_info(self) -> Dict[str, Any]:
        """获取完整状态信息"""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'open_time': self._open_time,
                'metrics': self._metrics.to_dict(),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout_seconds': self.config.recovery_timeout_seconds,
                },
            }


class CircuitBreakerRegistry:
    """
    熔断器注册表

    统一管理所有熔断器实例
    """

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config)
            return self._circuit_breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """获取已存在的熔断器"""
        with self._lock:
            return self._circuit_breakers.get(name)

    def reset_all(self) -> None:
        """重置所有熔断器"""
        with self._lock:
            for cb in self._circuit_breakers.values():
                cb.reset()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有熔断器状态"""
        with self._lock:
            return {
                name: cb.get_state_info()
                for name, cb in self._circuit_breakers.items()
            }


_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """获取全局熔断器注册表单例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """获取指定名称的熔断器"""
    return get_circuit_breaker_registry().get_or_create(name)


def with_circuit_breaker(
    name: str,
    fallback: Optional[Callable] = None,
    expected_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    熔断器装饰器

    Args:
        name: 熔断器名称
        fallback: 降级回调函数
        expected_exceptions: 期望捕获的异常类型
    """
    if expected_exceptions is None:
        expected_exceptions = (Exception,)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = get_circuit_breaker(name)

            if not cb.is_available():
                logger.warning(f"[{name}] 熔断器已打开，执行降级")
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise RuntimeError(f"Circuit breaker {name} is open")

            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except expected_exceptions as e:
                cb.record_failure(e)
                logger.warning(f"[{name}] 请求失败: {e}")
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise

        return wrapper

    return decorator
