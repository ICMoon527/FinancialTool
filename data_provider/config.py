# -*- coding: utf-8 -*-
"""
===================================
数据获取模块统一配置
===================================

职责：
1. 统一管理所有数据获取相关配置
2. 提供默认配置值
3. 支持环境变量覆盖
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetryConfig:
    """重试策略配置"""
    max_attempts: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 10.0
    multiplier: float = 2.0


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    expected_exception_types: List[type] = field(default_factory=list)


@dataclass
class CacheConfig:
    """缓存配置"""
    realtime_ttl_seconds: int = 600
    etf_realtime_ttl_seconds: int = 600
    stock_name_ttl_seconds: int = 3600


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    sleep_min_seconds: float = 1.0
    sleep_max_seconds: float = 3.0
    enable_random_jitter: bool = True


@dataclass
class TimeoutConfig:
    """超时配置"""
    request_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0


@dataclass
class DataProviderConfig:
    """数据获取模块完整配置"""
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    
    user_agents: List[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ])
    
    @classmethod
    def from_env(cls) -> 'DataProviderConfig':
        """从环境变量加载配置"""
        config = cls()
        
        if 'RETRY_MAX_ATTEMPTS' in os.environ:
            config.retry.max_attempts = int(os.environ['RETRY_MAX_ATTEMPTS'])
        if 'RETRY_MIN_WAIT' in os.environ:
            config.retry.min_wait_seconds = float(os.environ['RETRY_MIN_WAIT'])
        if 'RETRY_MAX_WAIT' in os.environ:
            config.retry.max_wait_seconds = float(os.environ['RETRY_MAX_WAIT'])
        
        if 'CB_FAILURE_THRESHOLD' in os.environ:
            config.circuit_breaker.failure_threshold = int(os.environ['CB_FAILURE_THRESHOLD'])
        if 'CB_RECOVERY_TIMEOUT' in os.environ:
            config.circuit_breaker.recovery_timeout_seconds = int(os.environ['CB_RECOVERY_TIMEOUT'])
        
        if 'CACHE_REALTIME_TTL' in os.environ:
            config.cache.realtime_ttl_seconds = int(os.environ['CACHE_REALTIME_TTL'])
        if 'CACHE_ETF_REALTIME_TTL' in os.environ:
            config.cache.etf_realtime_ttl_seconds = int(os.environ['CACHE_ETF_REALTIME_TTL'])
        if 'CACHE_STOCK_NAME_TTL' in os.environ:
            config.cache.stock_name_ttl_seconds = int(os.environ['CACHE_STOCK_NAME_TTL'])
        
        if 'RATE_LIMIT_SLEEP_MIN' in os.environ:
            config.rate_limit.sleep_min_seconds = float(os.environ['RATE_LIMIT_SLEEP_MIN'])
        if 'RATE_LIMIT_SLEEP_MAX' in os.environ:
            config.rate_limit.sleep_max_seconds = float(os.environ['RATE_LIMIT_SLEEP_MAX'])
        
        if 'TIMEOUT_REQUEST' in os.environ:
            config.timeout.request_timeout_seconds = float(os.environ['TIMEOUT_REQUEST'])
        if 'TIMEOUT_CONNECT' in os.environ:
            config.timeout.connect_timeout_seconds = float(os.environ['TIMEOUT_CONNECT'])
        
        return config


_global_config: Optional[DataProviderConfig] = None


def get_config() -> DataProviderConfig:
    """获取全局配置单例"""
    global _global_config
    if _global_config is None:
        _global_config = DataProviderConfig.from_env()
    return _global_config


def set_config(config: DataProviderConfig) -> None:
    """设置全局配置（用于测试）"""
    global _global_config
    _global_config = config
