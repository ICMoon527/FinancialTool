# -*- coding: utf-8 -*-
"""
异常处理模块 - 统一的异常类和处理机制
"""

import logging
import time
from typing import Any, Callable, Optional, Tuple, Type, Union
from functools import wraps

logger = logging.getLogger(__name__)


class StockSelectorError(Exception):
    """选股系统基础异常类"""
    pass


class DatabaseError(StockSelectorError):
    """数据库操作异常"""
    pass


class DataFetchError(StockSelectorError):
    """数据获取异常"""
    pass


class RateLimitError(StockSelectorError):
    """速率限制异常"""
    pass


class ValidationError(StockSelectorError):
    """数据验证异常"""
    pass


class ConfigurationError(StockSelectorError):
    """配置错误异常"""
    pass


class CacheError(StockSelectorError):
    """缓存操作异常"""
    pass


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    指数退避重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(f"达到最大重试次数 {max_attempts}，放弃重试")
                        raise
                    
                    # 计算延迟时间（指数退避）
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    
                    # 添加随机抖动避免雪崩
                    jitter = delay * 0.1 * (2 * (time.time() % 1) - 1)
                    total_delay = delay + jitter
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    logger.warning(
                        f"第 {attempt} 次尝试失败，{total_delay:.2f} 秒后重试: {type(e).__name__}: {e}"
                    )
                    
                    time.sleep(total_delay)
            
            # 理论上不会到达这里，只是为了类型安全
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_error: bool = True,
    reraise: bool = False,
    **kwargs
) -> Tuple[Any, Optional[Exception]]:
    """
    安全执行函数，捕获并处理异常
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        default: 异常时返回的默认值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        **kwargs: 函数关键字参数
    
    Returns:
        Tuple: (执行结果, 异常对象)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        if log_error:
            logger.error(f"执行 {func.__name__} 失败: {type(e).__name__}: {e}", exc_info=True)
        
        if reraise:
            raise
        
        return default, e


def handle_exception(
    exception: Exception,
    context: str = "",
    fallback_action: Optional[Callable] = None,
    fallback_args: tuple = (),
    fallback_kwargs: dict = None
) -> Any:
    """
    统一异常处理函数
    
    Args:
        exception: 异常对象
        context: 异常上下文信息
        fallback_action: 回退操作函数
        fallback_args: 回退操作参数
        fallback_kwargs: 回退操作关键字参数
    
    Returns:
        回退操作的结果（如果有）
    """
    if fallback_kwargs is None:
        fallback_kwargs = {}
    
    error_type = type(exception).__name__
    error_msg = str(exception)
    
    logger.error(
        f"[{error_type}] {context}: {error_msg}",
        exc_info=True
    )
    
    # 根据异常类型进行不同处理
    if isinstance(exception, RateLimitError):
        logger.warning("检测到速率限制，建议稍后重试")
    elif isinstance(exception, DatabaseError):
        logger.error("数据库操作异常，请检查数据库连接和状态")
    elif isinstance(exception, ValidationError):
        logger.error("数据验证失败，请检查输入数据")
    elif isinstance(exception, ConfigurationError):
        logger.error("配置错误，请检查配置文件")
    
    # 执行回退操作
    if fallback_action:
        try:
            return fallback_action(*fallback_args, **fallback_kwargs)
        except Exception as fallback_error:
            logger.error(
                f"回退操作也失败: {type(fallback_error).__name__}: {fallback_error}",
                exc_info=True
            )
    
    return None
