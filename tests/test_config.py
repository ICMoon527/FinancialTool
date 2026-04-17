# -*- coding: utf-8 -*-
"""
配置模块测试
"""
import pytest
from src.config import Config, get_config


def test_config_singleton():
    """测试配置单例模式"""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_default_values():
    """测试配置默认值"""
    config = Config()
    assert config.database_path == "./data/stock_analysis.db"
    assert config.redis_enabled is False
    assert config.redis_ttl_seconds == 300


def test_config_env_override(monkeypatch):
    """测试环境变量覆盖配置"""
    monkeypatch.setenv("REDIS_ENABLED", "true")
    monkeypatch.setenv("REDIS_HOST", "test-host")
    monkeypatch.setenv("REDIS_PORT", "6380")
    
    config = Config()
    config._load_from_env()
    
    assert config.redis_enabled is True
    assert config.redis_host == "test-host"
    assert config.redis_port == 6380
