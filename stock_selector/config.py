# -*- coding: utf-8 -*-
"""
Stock Selector Configuration Module

Configuration management for the stock selector system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv


@dataclass
class StockSelectorConfig:
    """
    Stock Selector Configuration.

    All configuration options for the stock selector system, with sensible defaults.
    """

    nl_strategy_dir: Optional[str] = None
    python_strategy_dir: Optional[str] = None
    default_top_n: int = 5
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
    log_level: str = "INFO"
    debug_strategy_execution: bool = False

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

        return cls(
            nl_strategy_dir=os.getenv(
                "STOCK_SELECTOR_NL_STRATEGY_DIR",
                str(default_nl_dir) if default_nl_dir.exists() else None
            ),
            python_strategy_dir=os.getenv(
                "STOCK_SELECTOR_PYTHON_STRATEGY_DIR",
                str(default_python_dir) if default_python_dir.exists() else None
            ),
            default_top_n=int(os.getenv("STOCK_SELECTOR_DEFAULT_TOP_N", "5")),
            min_match_score=float(os.getenv("STOCK_SELECTOR_MIN_MATCH_SCORE", "50.0")),
            auto_activate_all=os.getenv("STOCK_SELECTOR_AUTO_ACTIVATE_ALL", "true").lower() == "true",
            default_active_strategies=[
                s.strip() for s in os.getenv("STOCK_SELECTOR_DEFAULT_ACTIVE_STRATEGIES", "").split(",")
                if s.strip()
            ],
            excluded_strategies=[
                s.strip() for s in os.getenv("STOCK_SELECTOR_EXCLUDED_STRATEGIES", "").split(",")
                if s.strip()
            ],
            preferred_strategy_type=os.getenv("STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE") or None,
            price_condition_weight=float(os.getenv("STOCK_SELECTOR_PRICE_WEIGHT", "20.0")),
            volume_condition_weight=float(os.getenv("STOCK_SELECTOR_VOLUME_WEIGHT", "20.0")),
            technical_indicator_weight=float(os.getenv("STOCK_SELECTOR_TECHNICAL_WEIGHT", "25.0")),
            market_condition_weight=float(os.getenv("STOCK_SELECTOR_MARKET_WEIGHT", "15.0")),
            trend_condition_weight=float(os.getenv("STOCK_SELECTOR_TREND_WEIGHT", "20.0")),
            short_term_gain_min=float(os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MIN", "1.0")),
            short_term_gain_max=float(os.getenv("STOCK_SELECTOR_SHORT_TERM_GAIN_MAX", "5.0")),
            short_term_volume_ratio_threshold=float(os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_RATIO", "1.5")),
            short_term_volume_avg_threshold=float(os.getenv("STOCK_SELECTOR_SHORT_TERM_VOLUME_AVG", "1.5")),
            ma_golden_cross_fast_period=int(os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_FAST", "5")),
            ma_golden_cross_slow_period=int(os.getenv("STOCK_SELECTOR_MA_GOLDEN_CROSS_SLOW", "20")),
            volume_breakout_lookback_days=int(os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_LOOKBACK", "20")),
            volume_breakout_multiplier=float(os.getenv("STOCK_SELECTOR_VOLUME_BREAKOUT_MULTIPLIER", "2.0")),
            log_level=os.getenv("STOCK_SELECTOR_LOG_LEVEL", "INFO"),
            debug_strategy_execution=os.getenv("STOCK_SELECTOR_DEBUG_EXECUTION", "false").lower() == "true",
        )


_config_instance: Optional[StockSelectorConfig] = None


def get_config() -> StockSelectorConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = StockSelectorConfig.from_env()
    return _config_instance


def set_config(config: StockSelectorConfig) -> None:
    global _config_instance
    _config_instance = config
