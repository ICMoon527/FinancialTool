# -*- coding: utf-8 -*-
"""
===================================
量化交易系统 - 缓存管理器
===================================

职责：
1. 提供 Redis + LRU 内存双层缓存
2. 统一的缓存访问接口
3. 缓存失效和更新机制
"""

import json
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Optional, Callable, TypeVar
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """缓存条目数据结构"""
    data: Any
    timestamp: datetime
    ttl_seconds: int


class CacheManager:
    """
    缓存管理器 - 双层缓存架构
    
    第一层：内存 LRU 缓存（快速访问）
    第二层：Redis 分布式缓存（持久化、跨进程共享）
    """
    
    _instance: Optional['CacheManager'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化缓存管理器"""
        if getattr(self, "_initialized", False):
            return
        
        from src.config import get_config
        self.config = get_config()
        
        # 内存 LRU 缓存
        self._lru_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_lru_entries = self.config.redis_max_cache_entries
        
        # Redis 客户端（可选）
        self._redis_client = None
        if self.config.redis_enabled:
            self._init_redis()
        
        self._initialized = True
        logger.info(f"缓存管理器初始化完成（Redis: {'启用' if self._redis_client else '禁用'}）")
    
    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _init_redis(self):
        """初始化 Redis 客户端"""
        try:
            import redis
            
            redis_config = {
                'host': self.config.redis_host,
                'port': self.config.redis_port,
                'db': self.config.redis_db,
                'decode_responses': False,  # 保持二进制以支持复杂对象
            }
            
            if self.config.redis_password:
                redis_config['password'] = self.config.redis_password
            
            self._redis_client = redis.Redis(**redis_config)
            
            # 测试连接
            self._redis_client.ping()
            logger.info(f"Redis 连接成功: {self.config.redis_host}:{self.config.redis_port}")
            
        except ImportError:
            logger.warning("Redis 库未安装，请运行: pip install redis")
            self._redis_client = None
        except Exception as e:
            logger.warning(f"Redis 连接失败，将仅使用内存缓存: {e}")
            self._redis_client = None
    
    def _serialize(self, data: Any) -> bytes:
        """序列化数据"""
        try:
            return json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        except Exception as e:
            logger.warning(f"数据序列化失败: {e}")
            return str(data).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.debug(f"数据反序列化失败，返回原始数据: {e}")
            return data.decode('utf-8', errors='ignore')
    
    def _is_entry_expired(self, entry: CacheEntry) -> bool:
        """检查缓存条目是否过期"""
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age > entry.ttl_seconds
    
    def _lru_get(self, key: str) -> Optional[Any]:
        """从 LRU 缓存获取数据"""
        if key not in self._lru_cache:
            return None
        
        entry = self._lru_cache[key]
        
        if self._is_entry_expired(entry):
            del self._lru_cache[key]
            return None
        
        # 移到末尾表示最近使用
        self._lru_cache.move_to_end(key)
        return entry.data
    
    def _lru_set(self, key: str, data: Any, ttl_seconds: Optional[int] = None):
        """设置 LRU 缓存"""
        if ttl_seconds is None:
            ttl_seconds = self.config.redis_ttl_seconds
        
        # 如果已满，删除最旧的条目
        if len(self._lru_cache) >= self._max_lru_entries:
            self._lru_cache.popitem(last=False)
        
        self._lru_cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    def _lru_delete(self, key: str):
        """删除 LRU 缓存条目"""
        if key in self._lru_cache:
            del self._lru_cache[key]
    
    def _lru_clear(self):
        """清空 LRU 缓存"""
        self._lru_cache.clear()
    
    def _redis_get(self, key: str) -> Optional[Any]:
        """从 Redis 获取数据"""
        if not self._redis_client:
            return None
        
        try:
            data = self._redis_client.get(key)
            if data is None:
                return None
            
            return self._deserialize(data)
        except Exception as e:
            logger.warning(f"Redis 获取失败: {e}")
            return None
    
    def _redis_set(self, key: str, data: Any, ttl_seconds: Optional[int] = None):
        """设置 Redis 缓存"""
        if not self._redis_client:
            return
        
        if ttl_seconds is None:
            ttl_seconds = self.config.redis_ttl_seconds
        
        try:
            serialized_data = self._serialize(data)
            self._redis_client.setex(key, ttl_seconds, serialized_data)
        except Exception as e:
            logger.warning(f"Redis 设置失败: {e}")
    
    def _redis_delete(self, key: str):
        """删除 Redis 缓存条目"""
        if not self._redis_client:
            return
        
        try:
            self._redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Redis 删除失败: {e}")
    
    def _redis_clear_pattern(self, pattern: str):
        """按模式清空 Redis 缓存"""
        if not self._redis_client:
            return
        
        try:
            keys = self._redis_client.keys(pattern)
            if keys:
                self._redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis 按模式清空失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据，不存在返回 None
        """
        # 先查 LRU 缓存
        data = self._lru_get(key)
        if data is not None:
            logger.debug(f"LRU 缓存命中: {key}")
            return data
        
        # 再查 Redis 缓存
        data = self._redis_get(key)
        if data is not None:
            logger.debug(f"Redis 缓存命中: {key}")
            # 回填到 LRU 缓存
            self._lru_set(key, data)
            return data
        
        logger.debug(f"缓存未命中: {key}")
        return None
    
    def set(self, key: str, data: Any, ttl_seconds: Optional[int] = None):
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 缓存数据
            ttl_seconds: 过期时间（秒），默认使用配置值
        """
        self._lru_set(key, data, ttl_seconds)
        self._redis_set(key, data, ttl_seconds)
        logger.debug(f"缓存已设置: {key}")
    
    def delete(self, key: str):
        """
        删除缓存条目
        
        Args:
            key: 缓存键
        """
        self._lru_delete(key)
        self._redis_delete(key)
        logger.debug(f"缓存已删除: {key}")
    
    def delete_pattern(self, pattern: str):
        """
        按模式删除缓存
        
        Args:
            pattern: 缓存键模式（支持通配符 *）
        """
        # 删除 LRU 缓存中匹配的键
        keys_to_delete = [k for k in self._lru_cache.keys() if self._match_pattern(k, pattern)]
        for key in keys_to_delete:
            del self._lru_cache[key]
        
        # 删除 Redis 缓存
        self._redis_clear_pattern(pattern)
        
        logger.debug(f"按模式删除缓存: {pattern}, 删除 {len(keys_to_delete)} 个 LRU 条目")
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """简单的模式匹配，支持 * 通配符"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def clear(self):
        """清空所有缓存"""
        self._lru_clear()
        if self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis 清空失败: {e}")
        logger.info("所有缓存已清空")
    
    def get_or_set(
        self,
        key: str,
        loader: Callable[[], T],
        ttl_seconds: Optional[int] = None
    ) -> T:
        """
        获取缓存，如果不存在则加载并设置
        
        Args:
            key: 缓存键
            loader: 数据加载函数
            ttl_seconds: 过期时间（秒）
            
        Returns:
            缓存数据
        """
        data = self.get(key)
        if data is not None:
            return data
        
        data = loader()
        self.set(key, data, ttl_seconds)
        return data
    
    def invalidate_stock_data(self, code: str):
        """
        使指定股票的所有缓存失效
        
        Args:
            code: 股票代码
        """
        pattern = f"stock:{code}:*"
        self.delete_pattern(pattern)
        logger.debug(f"股票缓存已失效: {code}")
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'lru_cache_size': len(self._lru_cache),
            'lru_max_entries': self._max_lru_entries,
            'redis_enabled': self._redis_client is not None,
        }
        
        if self._redis_client:
            try:
                stats['redis_info'] = self._redis_client.info()
            except Exception:
                pass
        
        return stats


# 便捷函数
def get_cache() -> CacheManager:
    """获取缓存管理器实例的快捷方式"""
    return CacheManager.get_instance()


def cached(
    key_prefix: str,
    ttl_seconds: Optional[int] = None,
    key_builder: Optional[Callable[..., str]] = None
):
    """
    缓存装饰器
    
    Args:
        key_prefix: 缓存键前缀
        ttl_seconds: 过期时间（秒）
        key_builder: 自定义缓存键构建函数
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache = get_cache()
            
            # 构建缓存键
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # 默认键构建：prefix + 位置参数 + 关键字参数
                import hashlib
                key_parts = [key_prefix]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_str = "|".join(key_parts)
                cache_key = f"{key_prefix}:{hashlib.md5(key_str.encode()).hexdigest()[:16]}"
            
            # 尝试从缓存获取
            data = cache.get(cache_key)
            if data is not None:
                return data
            
            # 执行原函数
            data = func(*args, **kwargs)
            
            # 设置缓存
            cache.set(cache_key, data, ttl_seconds)
            
            return data
        
        return wrapper
    
    return decorator
