# -*- coding: utf-8 -*-
"""
===================================
数据获取模块诊断与监控
===================================

职责：
1. 提供数据获取模块的状态诊断
2. 收集和报告指标
3. 提供健康检查接口
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import get_config
from .circuit_breaker import get_circuit_breaker_registry

logger = logging.getLogger(__name__)


@dataclass
class DataProviderHealth:
    """数据获取模块健康状态"""
    status: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    circuit_breakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'circuit_breakers': self.circuit_breakers,
            'config_summary': self.config_summary,
            'errors': self.errors,
            'warnings': self.warnings,
        }


def check_health() -> DataProviderHealth:
    """
    检查数据获取模块的健康状态

    Returns:
        DataProviderHealth 对象
    """
    health = DataProviderHealth()
    
    try:
        config = get_config()
        health.config_summary = {
            'retry': {
                'max_attempts': config.retry.max_attempts,
                'min_wait': config.retry.min_wait_seconds,
                'max_wait': config.retry.max_wait_seconds,
            },
            'circuit_breaker': {
                'failure_threshold': config.circuit_breaker.failure_threshold,
                'recovery_timeout': config.circuit_breaker.recovery_timeout_seconds,
            },
            'cache': {
                'realtime_ttl': config.cache.realtime_ttl_seconds,
                'etf_realtime_ttl': config.cache.etf_realtime_ttl_seconds,
                'stock_name_ttl': config.cache.stock_name_ttl_seconds,
            },
            'rate_limit': {
                'sleep_min': config.rate_limit.sleep_min_seconds,
                'sleep_max': config.rate_limit.sleep_max_seconds,
            },
            'timeout': {
                'request': config.timeout.request_timeout_seconds,
                'connect': config.timeout.connect_timeout_seconds,
            },
        }
        
        registry = get_circuit_breaker_registry()
        health.circuit_breakers = registry.get_all_states()
        
        open_circuits = [
            name for name, state in health.circuit_breakers.items()
            if state.get('state') == 'open'
        ]
        
        half_open_circuits = [
            name for name, state in health.circuit_breakers.items()
            if state.get('state') == 'half_open'
        ]
        
        if open_circuits:
            health.warnings.append(f"Open circuit breakers: {', '.join(open_circuits)}")
        
        if half_open_circuits:
            health.warnings.append(f"Half-open circuit breakers: {', '.join(half_open_circuits)}")
        
        if health.errors:
            health.status = "degraded"
        elif health.warnings:
            health.status = "warning"
        else:
            health.status = "healthy"
            
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        health.status = "error"
        health.errors.append(f"Health check exception: {str(e)}")
    
    return health


def get_diagnostics_report() -> Dict[str, Any]:
    """
    获取完整的诊断报告

    Returns:
        诊断报告字典
    """
    health = check_health()
    
    report = {
        'health': health.to_dict(),
        'generated_at': datetime.now().isoformat(),
        'version': '2.0',
    }
    
    return report
