# -*- coding: utf-8 -*-
"""
===================================
数据获取模块通用工具
===================================

职责：
1. 提供通用的工具函数，消除代码重复
2. 统一的 User-Agent 管理
3. 统一的随机休眠机制
4. 数据验证工具
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List

from .config import get_config

logger = logging.getLogger(__name__)


def get_random_user_agent() -> str:
    """
    获取随机 User-Agent

    Returns:
        随机选择的 User-Agent 字符串
    """
    config = get_config()
    return random.choice(config.user_agents)


def smart_sleep(
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
    enable_jitter: Optional[bool] = None,
) -> None:
    """
    智能随机休眠（防封禁策略）

    Args:
        min_seconds: 最小休眠时间（秒），默认使用配置
        max_seconds: 最大休眠时间（秒），默认使用配置
        enable_jitter: 是否启用随机抖动，默认使用配置
    """
    config = get_config()
    
    if min_seconds is None:
        min_seconds = config.rate_limit.sleep_min_seconds
    if max_seconds is None:
        max_seconds = config.rate_limit.sleep_max_seconds
    if enable_jitter is None:
        enable_jitter = config.rate_limit.enable_random_jitter
    
    if enable_jitter:
        sleep_time = random.uniform(min_seconds, max_seconds)
    else:
        sleep_time = (min_seconds + max_seconds) / 2
    
    logger.debug(f"智能休眠 {sleep_time:.2f} 秒...")
    time.sleep(sleep_time)


def validate_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    default_days: int = 30,
) -> tuple[str, str]:
    """
    验证并补全日期范围

    Args:
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        default_days: 默认天数

    Returns:
        (start_date, end_date) 元组
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=default_days * 2)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    return start_date, end_date


def is_valid_stock_code(code: str) -> bool:
    """
    简单验证股票代码格式

    Args:
        code: 股票代码

    Returns:
        是否有效
    """
    if not code or not isinstance(code, str):
        return False
    
    code = code.strip()
    if len(code) < 1:
        return False
    
    # 允许字母、数字、点、下划线
    import re
    return bool(re.match(r'^[A-Za-z0-9._]+$', code))


def safe_truncate_dict(
    data: Dict[str, Any],
    max_length: int = 100,
    keys_to_truncate: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    安全截断字典中的长字符串值（用于日志记录）

    Args:
        data: 输入字典
        max_length: 最大长度
        keys_to_truncate: 需要截断的键列表，None 表示所有字符串值

    Returns:
        截断后的字典
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            if keys_to_truncate is None or key in keys_to_truncate:
                if len(value) > max_length:
                    result[key] = value[:max_length] + '...'
                    continue
        result[key] = value
    return result


class RequestRateLimiter:
    """
    请求速率限制器

    用于控制请求频率，防止被封禁
    """

    def __init__(
        self,
        min_interval_seconds: float = 1.0,
        max_jitter_seconds: float = 2.0,
    ):
        self.min_interval = min_interval_seconds
        self.max_jitter = max_jitter_seconds
        self._last_request_time: Optional[float] = None
        self._lock = None

    def wait_if_needed(self) -> None:
        """
        如果需要，等待足够的时间
        """
        now = time.time()
        
        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                if self.max_jitter > 0:
                    wait_time += random.uniform(0, self.max_jitter)
                logger.debug(f"速率限制: 等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        self._last_request_time = time.time()

    def reset(self) -> None:
        """重置计时器"""
        self._last_request_time = None


# 全局速率限制器实例
_global_rate_limiter: Optional[RequestRateLimiter] = None


def get_rate_limiter() -> RequestRateLimiter:
    """获取全局速率限制器"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        config = get_config()
        _global_rate_limiter = RequestRateLimiter(
            min_interval_seconds=config.rate_limit.sleep_min_seconds,
            max_jitter_seconds=config.rate_limit.sleep_max_seconds - config.rate_limit.sleep_min_seconds,
        )
    return _global_rate_limiter
