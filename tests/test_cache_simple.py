# -*- coding: utf-8 -*-
"""
缓存模块简单测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import time
from src.cache import CacheManager


class TestCache(unittest.TestCase):
    """缓存模块测试"""
    
    def test_cache_singleton(self):
        """测试缓存管理器单例模式"""
        cache1 = CacheManager()
        cache2 = CacheManager()
        self.assertIs(cache1, cache2)
    
    def test_lru_cache_operations(self):
        """测试 LRU 缓存操作"""
        cache = CacheManager()
        
        # 设置缓存
        cache.set("test_key", {"data": "test_value"})
        
        # 获取缓存
        result = cache.get("test_key")
        self.assertEqual(result, {"data": "test_value"})
    
    def test_cache_delete(self):
        """测试缓存删除"""
        cache = CacheManager()
        
        cache.set("delete_key", "delete_value")
        self.assertEqual(cache.get("delete_key"), "delete_value")
        
        cache.delete("delete_key")
        self.assertIsNone(cache.get("delete_key"))
    
    def test_cache_clear(self):
        """测试缓存清空"""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        
        cache.clear()
        
        self.assertIsNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
    
    def test_get_or_set(self):
        """测试 get_or_set 方法"""
        cache = CacheManager()
        
        call_count = [0]
        
        def loader():
            call_count[0] += 1
            return f"loaded_value_{call_count[0]}"
        
        # 第一次调用，应该执行 loader
        result1 = cache.get_or_set("loader_key", loader)
        self.assertEqual(result1, "loaded_value_1")
        self.assertEqual(call_count[0], 1)
        
        # 第二次调用，应该从缓存获取，不执行 loader
        result2 = cache.get_or_set("loader_key", loader)
        self.assertEqual(result2, "loaded_value_1")
        self.assertEqual(call_count[0], 1)
    
    def test_cache_stats(self):
        """测试缓存统计"""
        cache = CacheManager()
        
        cache.set("stats_key", "stats_value")
        
        stats = cache.get_stats()
        self.assertIn("lru_cache_size", stats)
        self.assertIn("lru_max_entries", stats)
        self.assertIn("redis_enabled", stats)


if __name__ == "__main__":
    unittest.main()
