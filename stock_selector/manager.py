# -*- coding: utf-8 -*-
"""
Strategy Manager - Loads and manages stock selector strategies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from stock_selector.base import (
    StockCandidate,
    StockSelectorStrategy,
    StrategyMatch,
)
from stock_selector.config import StockSelectorConfig
from stock_selector.sector_manager import SectorManager
from stock_selector.strategies.nl_strategy_loader import load_nl_strategies_from_dir
from stock_selector.strategies.python_strategy_loader import load_python_strategies_from_dir

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages stock selector strategies.

    Responsibilities:
    - Load strategies from multiple sources (NL and Python)
    - Activate/deactivate strategies
    - Execute strategies on stocks
    - Combine results from multiple strategies
    - Apply configuration (exclusions, preferences)
    """

    def __init__(
        self,
        nl_strategy_dir: Optional[Path] = None,
        python_strategy_dir: Optional[Path] = None,
        data_provider: Optional[any] = None,
        config: Optional[StockSelectorConfig] = None,
    ):
        """
        Initialize the strategy manager.

        Args:
            nl_strategy_dir: Directory for natural language YAML strategies
            python_strategy_dir: Directory for Python strategy modules
            data_provider: Optional data provider for strategies
            config: Optional StockSelectorConfig instance
        """
        self._strategies: Dict[str, StockSelectorStrategy] = {}
        self._active_strategy_ids: List[str] = []
        self._data_provider = data_provider
        self._config = config
        self._db_manager = None

        # Initialize database manager
        try:
            from src.storage import get_db
            self._db_manager = get_db()
        except Exception as e:
            import logging
            logging.warning(f"Failed to initialize database manager: {e}")
        
        # Initialize sector manager
        self._sector_manager = SectorManager(
            data_manager=data_provider, 
            db_manager=self._db_manager,
            config=self._config
        )

        if nl_strategy_dir:
            # Load NLP strategies from NLP subdirectory
            nlp_dir = nl_strategy_dir / "NLP"
            if nlp_dir.exists():
                self.load_nl_strategies(nlp_dir)
            else:
                self.load_nl_strategies(nl_strategy_dir)
        if python_strategy_dir:
            # Load Python strategies from Python subdirectory
            python_dir = python_strategy_dir / "Python"
            if python_dir.exists():
                self.load_python_strategies(python_dir)
            else:
                self.load_python_strategies(python_strategy_dir)

        self._apply_config()
        self._apply_config_multipliers()
        
        if self._config and self._config.auto_activate_all:
            self.activate_all_strategies()
        elif self._config and self._config.default_active_strategies:
            self.activate_strategies(self._config.default_active_strategies)
        else:
            self.activate_all_strategies()

    def _apply_config(self) -> None:
        """
        Apply configuration to filter strategies.
        """
        if not self._config:
            return

        excluded = self._config.excluded_strategies
        preferred_type = self._config.preferred_strategy_type

        if excluded:
            excluded_set = set(excluded)
            excluded_ids = [sid for sid in self._strategies.keys() if sid in excluded_set]
            for sid in excluded_ids:
                del self._strategies[sid]
            if excluded_ids:
                logger.info("Excluded %d strategies: %s", len(excluded_ids), ", ".join(excluded_ids))

        if preferred_type:
            preferred_type = preferred_type.upper()
            filtered = {}
            for sid, strategy in self._strategies.items():
                if strategy.metadata.strategy_type.name == preferred_type:
                    filtered[sid] = strategy
            removed_count = len(self._strategies) - len(filtered)
            if removed_count > 0:
                self._strategies = filtered
                logger.info("Filtered to %s strategies only, removed %d strategies", preferred_type, removed_count)

    def set_data_provider(self, data_provider: any) -> None:
        self._data_provider = data_provider
        for strategy in self._strategies.values():
            strategy.set_data_provider(data_provider)
        # Update sector manager data provider
        if self._sector_manager:
            self._sector_manager._data_manager = data_provider
    
    def get_sector_manager(self) -> Optional[SectorManager]:
        """
        Get the sector manager instance.
        
        Returns:
            SectorManager instance or None
        """
        return self._sector_manager

    def load_nl_strategies(self, directory: Path) -> int:
        strategies = load_nl_strategies_from_dir(directory)
        count = 0
        for strategy in strategies:
            if strategy.id not in self._strategies:
                if self._data_provider:
                    strategy.set_data_provider(self._data_provider)
                if self._sector_manager:
                    strategy.set_sector_manager(self._sector_manager)
                self._strategies[strategy.id] = strategy
                count += 1
        logger.info("Loaded %d new NL strategies from %s", count, directory)
        return count

    def load_python_strategies(self, directory: Path) -> int:
        strategies = load_python_strategies_from_dir(directory)
        count = 0
        for strategy in strategies:
            if strategy.id not in self._strategies:
                if self._data_provider:
                    strategy.set_data_provider(self._data_provider)
                if self._sector_manager:
                    strategy.set_sector_manager(self._sector_manager)
                self._strategies[strategy.id] = strategy
                count += 1
        logger.info("Loaded %d new Python strategies from %s", count, directory)
        return count

    def get_all_strategies(self) -> List[StockSelectorStrategy]:
        return list(self._strategies.values())

    def get_strategy(self, strategy_id: str) -> Optional[StockSelectorStrategy]:
        return self._strategies.get(strategy_id)

    def get_active_strategies(self) -> List[StockSelectorStrategy]:
        return [self._strategies[sid] for sid in self._active_strategy_ids if sid in self._strategies]

    def activate_strategy(self, strategy_id: str) -> bool:
        if strategy_id not in self._strategies:
            logger.warning("Cannot activate unknown strategy: %s", strategy_id)
            return False
        if strategy_id not in self._active_strategy_ids:
            self._active_strategy_ids.append(strategy_id)
            logger.info("Activated strategy: %s", strategy_id)
        
        if strategy_id == "dragon_leader":
            sub_strategies = [
                "dragon_leader_start",
                "dragon_leader_main_rise",
                "dragon_leader_second_wave"
            ]
            for sub_id in sub_strategies:
                if sub_id in self._strategies and sub_id not in self._active_strategy_ids:
                    self._active_strategy_ids.append(sub_id)
                    logger.info("Activated sub-strategy for dragon_leader: %s", sub_id)
        
        return True

    def deactivate_strategy(self, strategy_id: str) -> None:
        if strategy_id in self._active_strategy_ids:
            self._active_strategy_ids.remove(strategy_id)
            logger.info("Deactivated strategy: %s", strategy_id)
        
        if strategy_id == "dragon_leader":
            sub_strategies = [
                "dragon_leader_start",
                "dragon_leader_main_rise",
                "dragon_leader_second_wave"
            ]
            for sub_id in sub_strategies:
                if sub_id in self._active_strategy_ids:
                    self._active_strategy_ids.remove(sub_id)
                    logger.info("Deactivated sub-strategy for dragon_leader: %s", sub_id)

    def activate_strategies(self, strategy_ids: List[str]) -> None:
        for sid in strategy_ids:
            self.activate_strategy(sid)

    def deactivate_strategies(self, strategy_ids: List[str]) -> None:
        for sid in strategy_ids:
            self.deactivate_strategy(sid)

    def activate_all_strategies(self) -> None:
        self._active_strategy_ids = list(self._strategies.keys())
        logger.info("Activated all %d strategies", len(self._active_strategy_ids))

    def deactivate_all_strategies(self) -> None:
        self._active_strategy_ids = []
        logger.info("Deactivated all strategies")

    def execute_strategies(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        strategy_ids: Optional[List[str]] = None,
    ) -> List[StrategyMatch]:
        strategies_to_use = []
        if strategy_ids:
            for sid in strategy_ids:
                strategy = self._strategies.get(sid)
                if strategy:
                    strategies_to_use.append(strategy)
                else:
                    logger.warning("Unknown strategy ID: %s", sid)
        else:
            strategies_to_use = self.get_active_strategies()

        results = []
        use_cached_data = False
        
        # Check if we have cached data in database for all strategies
        if self._db_manager:
            from datetime import date
            today = date.today()
            if self._db_manager.has_today_data(stock_code, today):
                logger.debug(f"Using cached data for {stock_code} from database")
                # Get cached data from database
                context = self._db_manager.get_analysis_context(stock_code, today)
                if context:
                    use_cached_data = True

        for strategy in strategies_to_use:
            try:
                if use_cached_data and self._db_manager:
                    from datetime import date
                    today = date.today()
                    context = self._db_manager.get_analysis_context(stock_code, today)
                    if context:
                        # Create a mock quote object with cached data
                        class MockQuote:
                            def __init__(self, context):
                                self.price = context.get('today', {}).get('close')
                                self.change_pct = context.get('price_change_ratio')
                                self.volume = context.get('today', {}).get('volume')
                                self.volume_ratio = context.get('volume_change_ratio')
                                self.turnover_rate = None
                                self.name = stock_name
                                
                            def has_basic_data(self):
                                return self.price is not None
                        
                        # Set the mock quote for the strategy
                        original_data_provider = strategy._data_provider
                        
                        # Create a mock data provider that returns cached data
                        class MockDataProvider:
                            def __init__(self, db_manager, stock_code, mock_quote):
                                self.db_manager = db_manager
                                self.stock_code = stock_code
                                self.mock_quote = mock_quote
                                
                            def get_realtime_quote(self, code):
                                return self.mock_quote
                                
                            def get_daily_data(self, code, days=60):
                                # Get historical data from database - use 120 days to ensure enough trading days
                                from datetime import timedelta
                                end_date = date.today()
                                start_date = end_date - timedelta(days=120)
                                data = self.db_manager.get_data_range(code, start_date, end_date)
                                if data:
                                    import pandas as pd
                                    df = pd.DataFrame([item.to_dict() for item in data])
                                    return df, "database"
                                return None, "database"
                        
                        mock_quote = MockQuote(context)
                        mock_data_provider = MockDataProvider(self._db_manager, stock_code, mock_quote)
                        strategy._data_provider = mock_data_provider
                        
                        # Execute strategy with cached data
                        match = strategy.select(stock_code, stock_name)
                        results.append(match)
                        
                        # Restore original data provider
                        strategy._data_provider = original_data_provider
                        continue
                
                # No cached data, use original data provider
                match = strategy.select(stock_code, stock_name)
                results.append(match)
            except Exception as e:
                logger.error("Strategy %s failed for %s: %s", strategy.id, stock_code, e)
                results.append(
                    StrategyMatch(
                        strategy_id=strategy.id,
                        strategy_name=strategy.display_name,
                        matched=False,
                        reason=f"Strategy execution error: {str(e)}",
                    )
                )

        return results

    def _apply_config_multipliers(self) -> None:
        """Apply strategy multipliers from configuration."""
        if not self._config or not self._config.strategy_multipliers:
            return
        
        for strategy_id, multiplier in self._config.strategy_multipliers.items():
            strategy = self._strategies.get(strategy_id)
            if strategy:
                strategy.score_multiplier = multiplier
                logger.info("Applied multiplier %.2f to strategy %s", multiplier, strategy_id)

    def set_strategy_multiplier(self, strategy_id: str, multiplier: float) -> bool:
        """
        Set the score multiplier for a specific strategy.
        
        Args:
            strategy_id: ID of the strategy to update
            multiplier: New multiplier value (must be >= 0)
            
        Returns:
            True if strategy was found and updated, False otherwise
        """
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            logger.warning("Cannot set multiplier for unknown strategy: %s", strategy_id)
            return False
        
        old_multiplier = strategy.score_multiplier
        strategy.score_multiplier = max(0.0, multiplier)
        logger.info("Updated strategy %s multiplier: %.2f -> %.2f", 
                   strategy_id, old_multiplier, strategy.score_multiplier)
        return True

    def get_strategy_multiplier(self, strategy_id: str) -> Optional[float]:
        """
        Get the current score multiplier for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Current multiplier or None if strategy not found
        """
        strategy = self._strategies.get(strategy_id)
        return strategy.score_multiplier if strategy else None

    def get_all_strategy_multipliers(self) -> Dict[str, float]:
        """
        Get multipliers for all strategies.
        
        Returns:
            Dictionary of {strategy_id: multiplier}
        """
        return {sid: strategy.score_multiplier for sid, strategy in self._strategies.items()}

    def reset_strategy_multipliers(self) -> None:
        """Reset all strategies to their default multipliers (1.0)."""
        for strategy in self._strategies.values():
            strategy.score_multiplier = 1.0
        logger.info("Reset all strategy multipliers to default (1.0)")

    def rank_candidates(
        self,
        candidates: List[StockCandidate],
        top_n: int = 5,
    ) -> List[StockCandidate]:
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.match_score,
            reverse=True,
        )
        return sorted_candidates[:top_n]
