# -*- coding: utf-8 -*-
"""
缓存模块测试
"""
import pytest
import time
from src.cache import CacheManager, cached


def test_cache_singleton():
    """测试缓存管理器单例模式"""
    cache1 = CacheManager()
    cache2 = CacheManager()
    assert cache1 is cache2


def test_lru_cache_operations():
    """测试 LRU 缓存操作"""
    cache = CacheManager()
    
    # 设置缓存
    cache.set("test_key", {"data": "test_value"})
    
    # 获取缓存
    result = cache.get("test_key")
    assert result == {"data": "test_value"}


def test_lru_cache_expiration():
    """测试 LRU 缓存过期"""
    cache = CacheManager()
    
    # 设置短期缓存（1秒）
    cache.set("expire_key", "expire_value", ttl_seconds=1)
    
    # 立即获取应该能获取到
    assert cache.get("expire_key") == "expire_value"
    
    # 等待过期
    time.sleep(1.1)
    
    # 过期后应该返回 None
    assert cache.get("expire_key") is None


def test_cache_delete():
    """测试缓存删除"""
    cache = CacheManager()
    
    cache.set("delete_key", "delete_value")
    assert cache.get("delete_key") == "delete_value"
    
    cache.delete("delete_key")
    assert cache.get("delete_key") is None


def test_cache_clear():
    """测试缓存清空"""
    cache = CacheManager()
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    
    cache.clear()
    
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_get_or_set():
    """测试 get_or_set 方法"""
    cache = CacheManager()
    
    call_count = 0
    
    def loader():
        nonlocal call_count
        call_count += 1
        return f"loaded_value_{call_count}"
    
    # 第一次调用，应该执行 loader
    result1 = cache.get_or_set("loader_key", loader)
    assert result1 == "loaded_value_1"
    assert call_count == 1
    
    # 第二次调用，应该从缓存获取，不执行 loader
    result2 = cache.get_or_set("loader_key", loader)
    assert result2 == "loaded_value_1"
    assert call_count == 1


def test_cached_decorator():
    """测试 cached 装饰器"""
    cache = CacheManager()
    
    call_count = 0
    
    @cached(key_prefix="test_decorator")
    def add(a, b):
        nonlocal call_count
        call_count += 1
        return a + b
    
    # 第一次调用
    result1 = add(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # 第二次调用相同参数，应该从缓存
    result2 = add(1, 2)
    assert result2 == 3
    assert call_count == 1
    
    # 不同参数，应该重新计算
    result3 = add(3, 4)
    assert result3 == 7
    assert call_count == 2


def test_cache_stats():
    """测试缓存统计"""
    cache = CacheManager()
    
    cache.set("stats_key", "stats_value")
    
    stats = cache.get_stats()
    assert "lru_cache_size" in stats
    assert "lru_max_entries" in stats
    assert "redis_enabled" in stats
