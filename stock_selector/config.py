# -*- coding: utf-8 -*-
"""
Stock Selector Configuration Module

Configuration management for the stock selector system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Callable, Any, Tuple
from dotenv import load_dotenv
from .dynamic_config_manager import DynamicConfigManager


@dataclass
class StockSelectorConfig:
    """
    Stock Selector Configuration.

    All configuration options for the stock selector system, with sensible defaults.
    """

    nl_strategy_dir: Optional[str] = None
    python_strategy_dir: Optional[str] = None
    default_top_n: int = 10
    min_match_score: float = 50.0
    
    auto_activate_all: bool = False
    default_active_strategies: List[str] = field(default_factory=list)
    excluded_strategies: List[str] = field(default_factory=list)
    preferred_strategy_type: Optional[str] = None
    price_condition_weight: float = 20.0
    volume_condition_weight: float = 20.0
    technical_indicator_weight: float = 25.0
    market_condition_weight: float = 15.0
    trend_condition_weight: float = 20.0
    short_term_gain_min: float = 1.0
    short_term_gain_max: float = 5.0
    short_term_volume_ratio_threshold: float = 1.5
    short_term_volume_avg_threshold: float = 1.5
    ma_golden_cross_fast_period: int = 5
    ma_golden_cross_slow_period: int = 20
    volume_breakout_lookback_days: int = 20
    volume_breakout_multiplier: float = 2.0
    log_level: str = "DEBUG"
    debug_strategy_execution: bool = False
    strategy_multipliers: Dict[str, float] = field(default_factory=dict)
    
    # Sector analysis configuration
    enable_sector_analysis: bool = True
    sector_hot_threshold: float = 2.0
    sector_cache_ttl: int = 1800
    sector_bonus_low: float = 3.0
    sector_bonus_mid: float = 5.0
    sector_bonus_high: float = 10.0
    sector_bonus_threshold_low: float = 2.0
    sector_bonus_threshold_high: float = 5.0
    update_sector_data: bool = False
    
    # 多线程配置
    enable_multithreading: bool = True
    multithreading_workers: int = 15
    
    # 数据下载批次配置
    efinance_batch_size: int = 10
    tushare_batch_size: int = 20
    # 默认数据更新天数
    update_data_default_days: int = 150
    
    # 六维选股配置
    six_dimension_main_trading_weight: float = 1.0
    six_dimension_bank_control_weight: float = 1.0
    six_dimension_momentum_v2_weight: float = 1.0
    six_dimension_resonance_weight: float = 1.0
    six_dimension_strong_blast_weight: float = 1.0
    six_dimension_sector_weight: float = 1.0
    six_dimension_min_matched_dimensions: int = 4

    @classmethod
    def _parse_strategy_multipliers(cls, env_str: str) -> Dict[str, float]:
        """Parse strategy multipliers from environment variable string."""
        multipliers = {}
        if not env_str:
            return multipliers
        parts = env_str.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                strategy_id, multiplier = part.split(":", 1)
                try:
                    multipliers[strategy_id.strip()] = float(multiplier.strip())
                except ValueError:
                    pass
        return multipliers

    @classmethod
    def from_env(cls) -> "StockSelectorConfig":
        # Load environment variables from .env file
        env_file = os.getenv("ENV_FILE")
        if env_file:
            env_path = Path(env_file)
        else:
            env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path, override=True)
        
        project_root = Path(__file__).parent.parent
        default_nl_dir = project_root / "stock_selector" / "strategies"
        default_python_dir = project_root / "stock_selector" / "strategies"

        # 先创建默认配置（使用 dataclass 字段默认值）
        config = cls()
        
        # 用环境变量覆盖
        config.nl_strategy_dir = os.getenv(
            "STOCK_SELECTOR_NL_STRATEGY_DIR",
            str(default_nl_dir) if default_nl_dir.exists() else None
        )
        config.python_strategy_dir = os.getenv(
            "STOCK_SELECTOR_PYTHON_STRATEGY_DIR",
            str(default_python_dir) if default_python_dir.exists() else None
        )
        
        # 只有设置了环境变量才覆盖
        if os.getenv("STOCK_SELECTOR_DEFAULT_TOP_N"):
            config.default_top_n = int(os.getenv("STOCK_SELECTOR_DEFAULT_TOP_N"))
        
        if os.getenv("STOCK_SELECTOR_MIN_MATCH_SCORE"):
            config.min_match_score = float(os.getenv("STOCK_SELECTOR_MIN_MATCH_SCORE"))
        
        if os.getenv("STOCK_SELECTOR_AUTO_ACTIVATE_ALL"):
            config.auto_activate_all = os.getenv("STOCK_SELECTOR_AUTO_ACTIVATE_ALL").lower() == "true"
        
        if os.getenv("STOCK_SELECTOR_DEFAULT_ACTIVE_STRATEGIES"):
            config.default_active_strategies = [
                s.strip() for s in os.getenv("STOCK_SELECTOR_DEFAULT_ACTIVE_STRATEGIES").split(",")
                if s.strip()
            ]
        
        if os.getenv("STOCK_SELECTOR_EXCLUDED_STRATEGIES"):
            config.excluded_strategies = [
                s.strip() for s in os.getenv("STOCK_SELECTOR_EXCLUDED_STRATEGIES").split(",")
                if s.strip()
            ]
        
        config.preferred_strategy_type = os.getenv("STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE") or None
        
        if os.getenv("STOCK_SELECTOR_PRICE_WEIGHT"):
            config.price_condition_weight = float(os.getenv("STOCK_SELECTOR_PRICE_WEIGHT"))
        
        if os.getenv("STOCK_SELECTOR_VOLUME_WEIGHT"):
            config.volume_condition_weight = float(os.getenv("STOCK_SELECTOR_VOLUME_WEIGHT"))
        
        if os.getenv("STOCK_SELECTOR_TECHNICAL_WEIGHT"):
            config.technical_indicator_weight = float(os.getenv("STOCK_SELECTOR_TECHNICAL_WEIGHT"))
        
        if os.getenv("STOCK_SELECTOR_MARKET_WEIGHT"):
            config.market_condition_weight = float(os.getenv("STOCK_SELECTOR_MARKET_WEIGHT"))
        
        if os.getenv("STOCK_SELECTOR_TREND_WEIGHT"):
            config.trend_condition_weight = float(os.getenv("STOCK_SELECTOR_TREND_WEIGHT"))
        
        if os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MIN"):
            config.short_term_gain_min = float(os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MIN"))
        
        if os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MAX"):
            config.short_term_gain_max = float(os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MAX"))
        
        if os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_RATIO"):
            config.short_term_volume_ratio_threshold = float(os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_RATIO"))
        
        if os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_AVG"):
            config.short_term_volume_avg_threshold = float(os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_AVG"))
        
        if os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_FAST"):
            config.ma_golden_cross_fast_period = int(os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_FAST"))
        
        if os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_SLOW"):
            config.ma_golden_cross_slow_period = int(os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_SLOW"))
        
        if os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_LOOKBACK"):
            config.volume_breakout_lookback_days = int(os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_LOOKBACK"))
        
        if os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_MULTIPLIER"):
            config.volume_breakout_multiplier = float(os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_MULTIPLIER"))
        
        if os.getenv("STOCK_SELECTOR_LOG_LEVEL"):
            config.log_level = os.getenv("STOCK_SELECTOR_LOG_LEVEL")
        
        if os.getenv("STOCK_SELECTOR_DEBUG_EXECUTION"):
            config.debug_strategy_execution = os.getenv("STOCK_SELECTOR_DEBUG_EXECUTION").lower() == "true"
        
        if os.getenv("STOCK_SELECTOR_STRATEGY_MULTIPLIERS"):
            config.strategy_multipliers = cls._parse_strategy_multipliers(
                os.getenv("STOCK_SELECTOR_STRATEGY_MULTIPLIERS")
            )
        
        if os.getenv("STOCK_SELECTOR_ENABLE_SECTOR_ANALYSIS"):
            config.enable_sector_analysis = os.getenv("STOCK_SELECTOR_ENABLE_SECTOR_ANALYSIS").lower() == "true"
        
        if os.getenv("STOCK_SELECTOR_SECTOR_THRESHOLD"):
            config.sector_hot_threshold = float(os.getenv("STOCK_SELECTOR_SECTOR_THRESHOLD"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_CACHE_TTL"):
            config.sector_cache_ttl = int(os.getenv("STOCK_SELECTOR_SECTOR_CACHE_TTL"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_BONUS_LOW"):
            config.sector_bonus_low = float(os.getenv("STOCK_SELECTOR_SECTOR_BONUS_LOW"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_BONUS_MID"):
            config.sector_bonus_mid = float(os.getenv("STOCK_SELECTOR_SECTOR_BONUS_MID"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_BONUS_HIGH"):
            config.sector_bonus_high = float(os.getenv("STOCK_SELECTOR_SECTOR_BONUS_HIGH"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_BONUS_THRESHOLD_LOW"):
            config.sector_bonus_threshold_low = float(os.getenv("STOCK_SELECTOR_SECTOR_BONUS_THRESHOLD_LOW"))
        
        if os.getenv("STOCK_SELECTOR_SECTOR_BONUS_THRESHOLD_HIGH"):
            config.sector_bonus_threshold_high = float(os.getenv("STOCK_SELECTOR_SECTOR_BONUS_THRESHOLD_HIGH"))
        
        if os.getenv("UPDATE_SECTOR_DATA"):
            config.update_sector_data = os.getenv("UPDATE_SECTOR_DATA").lower() == "true"
        
        if os.getenv("STOCK_SELECTOR_ENABLE_MULTITHREADING"):
            config.enable_multithreading = os.getenv("STOCK_SELECTOR_ENABLE_MULTITHREADING").lower() == "true"
        
        if os.getenv("STOCK_SELECTOR_MULTITHREADING_WORKERS"):
            config.multithreading_workers = int(os.getenv("STOCK_SELECTOR_MULTITHREADING_WORKERS"))
        
        # 六维选股配置
        if os.getenv("SIX_DIMENSION_MAIN_TRADING_WEIGHT"):
            config.six_dimension_main_trading_weight = float(os.getenv("SIX_DIMENSION_MAIN_TRADING_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_BANK_CONTROL_WEIGHT"):
            config.six_dimension_bank_control_weight = float(os.getenv("SIX_DIMENSION_BANK_CONTROL_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_MOMENTUM_V2_WEIGHT"):
            config.six_dimension_momentum_v2_weight = float(os.getenv("SIX_DIMENSION_MOMENTUM_V2_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_RESONANCE_WEIGHT"):
            config.six_dimension_resonance_weight = float(os.getenv("SIX_DIMENSION_RESONANCE_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_STRONG_BLAST_WEIGHT"):
            config.six_dimension_strong_blast_weight = float(os.getenv("SIX_DIMENSION_STRONG_BLAST_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_SECTOR_WEIGHT"):
            config.six_dimension_sector_weight = float(os.getenv("SIX_DIMENSION_SECTOR_WEIGHT"))
        
        if os.getenv("SIX_DIMENSION_MIN_MATCHED_DIMENSIONS"):
            config.six_dimension_min_matched_dimensions = int(os.getenv("SIX_DIMENSION_MIN_MATCHED_DIMENSIONS"))
        
        # 数据下载批次配置
        if os.getenv("STOCK_SELECTOR_EFINANCE_BATCH_SIZE"):
            config.efinance_batch_size = int(os.getenv("STOCK_SELECTOR_EFINANCE_BATCH_SIZE"))
        
        if os.getenv("STOCK_SELECTOR_TUSHARE_BATCH_SIZE"):
            config.tushare_batch_size = int(os.getenv("STOCK_SELECTOR_TUSHARE_BATCH_SIZE"))
        
        if os.getenv("STOCK_SELECTOR_UPDATE_DATA_DEFAULT_DAYS"):
            config.update_data_default_days = int(os.getenv("STOCK_SELECTOR_UPDATE_DATA_DEFAULT_DAYS"))
        
        return config


_config_instance: Optional[StockSelectorConfig] = None
_dynamic_config_manager: Optional[DynamicConfigManager] = None
_dynamic_config_enabled: bool = False


def get_config() -> StockSelectorConfig:
    """
    获取当前配置
    
    如果启用了动态配置，则从动态配置管理器获取配置；
    否则，使用传统的全局配置实例。
    
    Returns:
        StockSelectorConfig: 当前生效的配置对象
    """
    global _config_instance, _dynamic_config_manager, _dynamic_config_enabled
    
    if _dynamic_config_enabled:
        if _dynamic_config_manager is None:
            _dynamic_config_manager = DynamicConfigManager()
        return _dynamic_config_manager.get_config()
    else:
        if _config_instance is None:
            _config_instance = StockSelectorConfig.from_env()
        return _config_instance


def set_config(config: StockSelectorConfig, 
               validator: Optional[Callable[[StockSelectorConfig], bool]] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    设置新配置
    
    如果启用了动态配置，则使用动态配置管理器原子性地设置配置；
    否则，直接设置全局配置实例。
    
    Args:
        config: 新的配置对象
        validator: 可选的配置验证函数，仅在启用动态配置时有效
    
    Returns:
        Tuple[bool, Dict[str, Any]]: 
            - 第一个元素：配置设置成功返回 True
            - 第二个元素：包含变更信息的字典（仅在启用动态配置时有值）
    """
    global _config_instance, _dynamic_config_manager, _dynamic_config_enabled
    
    if _dynamic_config_enabled:
        if _dynamic_config_manager is None:
            _dynamic_config_manager = DynamicConfigManager()
        return _dynamic_config_manager.set_config(config, validator)
    else:
        _config_instance = config
        return True, {}


def reload_config(config_path: Optional[str] = None,
                  validator: Optional[Callable[[StockSelectorConfig], bool]] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    重新加载配置
    
    如果启用了动态配置，则使用动态配置管理器重新加载配置；
    否则，重新从环境变量加载配置并设置为全局实例。
    
    Args:
        config_path: 可选的配置文件路径，仅在启用动态配置时有效
        validator: 可选的配置验证函数，仅在启用动态配置时有效
    
    Returns:
        Tuple[bool, Dict[str, Any]]: 
            - 第一个元素：配置重新加载成功返回 True
            - 第二个元素：包含变更信息的字典（仅在启用动态配置时有值）
    """
    global _config_instance, _dynamic_config_manager, _dynamic_config_enabled
    
    if _dynamic_config_enabled:
        if _dynamic_config_manager is None:
            _dynamic_config_manager = DynamicConfigManager()
        return _dynamic_config_manager.reload_config(config_path, validator)
    else:
        _config_instance = StockSelectorConfig.from_env()
        return True, {}


def enable_dynamic_config(enable: bool) -> None:
    """
    启用或禁用动态配置功能
    
    当从禁用切换到启用时，会自动创建动态配置管理器并将当前配置迁移过去；
    当从启用切换到禁用时，会将当前配置保存到全局配置实例中。
    
    Args:
        enable: True 表示启用动态配置，False 表示禁用动态配置
    """
    global _config_instance, _dynamic_config_manager, _dynamic_config_enabled
    
    if enable == _dynamic_config_enabled:
        return
    
    if enable:
        if _dynamic_config_manager is None:
            if _config_instance is not None:
                _dynamic_config_manager = DynamicConfigManager(_config_instance)
            else:
                _dynamic_config_manager = DynamicConfigManager()
        _dynamic_config_enabled = True
    else:
        if _dynamic_config_manager is not None:
            _config_instance = _dynamic_config_manager.get_config()
        _dynamic_config_enabled = False


def rollback() -> Tuple[bool, Dict[str, Any]]:
    """
    回滚到上一个有效配置（仅在启用动态配置时有效）
    
    Returns:
        Tuple[bool, Dict[str, Any]]: 
            - 第一个元素：回滚成功返回 True
            - 第二个元素：包含变更信息的字典
    """
    global _dynamic_config_manager, _dynamic_config_enabled
    
    if not _dynamic_config_enabled or _dynamic_config_manager is None:
        return False, {}
    
    return _dynamic_config_manager.rollback()


def register_callback(callback: Callable[[StockSelectorConfig, StockSelectorConfig], None]) -> str:
    """
    注册配置变更回调函数（仅在启用动态配置时有效）
    
    Args:
        callback: 回调函数，签名为 (old_config: StockSelectorConfig, new_config: StockSelectorConfig) -> None
    
    Returns:
        str: 回调函数的唯一ID，可用于后续取消注册
    """
    global _dynamic_config_manager, _dynamic_config_enabled
    
    if not _dynamic_config_enabled:
        raise RuntimeError("动态配置未启用，无法注册回调函数")
    
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager()
    
    return _dynamic_config_manager.register_callback(callback)


def unregister_callback(callback_id: str) -> bool:
    """
    取消注册配置变更回调函数（仅在启用动态配置时有效）
    
    Args:
        callback_id: 回调函数的唯一ID
    
    Returns:
        bool: 成功取消返回 True
    """
    global _dynamic_config_manager, _dynamic_config_enabled
    
    if not _dynamic_config_enabled or _dynamic_config_manager is None:
        return False
    
    return _dynamic_config_manager.unregister_callback(callback_id)


def start_monitoring(env_file_path: Optional[str] = None, 
                     monitor_interval: float = 1.0,
                     debounce_time: float = 0.5) -> None:
    """
    启动 .env 文件监控（仅在启用动态配置时有效）
    
    Args:
        env_file_path: .env 文件路径
        monitor_interval: 监控间隔（秒）
        debounce_time: 防抖时间（秒）
    """
    global _dynamic_config_manager, _dynamic_config_enabled
    
    if not _dynamic_config_enabled:
        raise RuntimeError("动态配置未启用，无法启动文件监控")
    
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager()
    
    _dynamic_config_manager.start_monitoring(env_file_path, monitor_interval, debounce_time)


def stop_monitoring() -> None:
    """
    停止 .env 文件监控（仅在启用动态配置时有效）
    """
    global _dynamic_config_manager, _dynamic_config_enabled
    
    if not _dynamic_config_enabled or _dynamic_config_manager is None:
        return
    
    _dynamic_config_manager.stop_monitoring()


# 模块初始化时从环境变量读取 ENABLE_DYNAMIC_CONFIG 配置
# 如果 ENABLE_DYNAMIC_CONFIG=true，则自动启用动态配置功能并启动文件监控
_env_enable = os.getenv("ENABLE_DYNAMIC_CONFIG", "").lower() == "true"
if _env_enable:
    enable_dynamic_config(True)
    # 自动启动 .env 文件监控
    try:
        start_monitoring()
    except Exception as e:
        pass
