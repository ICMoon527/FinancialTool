# -*- coding: utf-8 -*-
"""
Stock Selector Configuration Module

Configuration management for the stock selector system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv


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
    
    # 六维选股配置
    six_dimension_main_trading_weight: float = 1.0
    six_dimension_bank_control_weight: float = 1.0
    six_dimension_momentum_v2_weight: float = 1.0
    six_dimension_resonance_weight: float = 1.0
    six_dimension_strong_blast_weight: float = 1.0
    six_dimension_sector_weight: float = 1.0
    six_dimension_min_matched_dimensions: int = 2

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
        load_dotenv(dotenv_path=env_path, override=False)
        
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
        
        return config


_config_instance: Optional[StockSelectorConfig] = None


def get_config() -> StockSelectorConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = StockSelectorConfig.from_env()
    return _config_instance


def set_config(config: StockSelectorConfig) -> None:
    global _config_instance
    _config_instance = config
