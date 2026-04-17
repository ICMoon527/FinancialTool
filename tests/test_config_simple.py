# -*- coding: utf-8 -*-
"""
配置模块简单测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from src.config import Config, get_config


class TestConfig(unittest.TestCase):
    """配置模块测试"""
    
    def test_config_singleton(self):
        """测试配置单例模式"""
        config1 = get_config()
        config2 = get_config()
        self.assertIs(config1, config2)
    
    def test_config_default_values(self):
        """测试配置默认值"""
        config = Config()
        self.assertEqual(config.database_path, "./data/stock_analysis.db")
        self.assertFalse(config.redis_enabled)
        self.assertEqual(config.redis_ttl_seconds, 300)


if __name__ == "__main__":
    unittest.main()
