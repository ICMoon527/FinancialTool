# -*- coding: utf-8 -*-
"""
Stock Selector Service - High-level service for stock screening.
"""

import logging
import concurrent.futures
from pathlib import Path
from typing import List, Optional

from stock_selector.base import StockCandidate, StrategyMetadata
from stock_selector.config import StockSelectorConfig, get_config
from stock_selector.manager import StrategyManager
from stock_selector.stock_pool import get_all_stock_codes

logger = logging.getLogger(__name__)


class StockSelectorService:
    """
    High-level service for stock screening.

    Responsibilities:
    - Orchestrate stock selection workflow
    - Manage strategy manager
    - Provide high-level screening API
    """

    def __init__(
        self,
        nl_strategy_dir: Optional[Path] = None,
        python_strategy_dir: Optional[Path] = None,
        data_provider: Optional[any] = None,
        config: Optional[StockSelectorConfig] = None,
    ):
        self.config = config or get_config()
        self._db_manager = None

        # Initialize database manager
        try:
            from src.storage import get_db
            self._db_manager = get_db()
        except Exception as e:
            logger.warning(f"Failed to initialize database manager: {e}")

        if nl_strategy_dir is None and self.config.nl_strategy_dir:
            nl_strategy_dir = Path(self.config.nl_strategy_dir)

        if python_strategy_dir is None and self.config.python_strategy_dir:
            python_strategy_dir = Path(self.config.python_strategy_dir)

        self.strategy_manager = StrategyManager(
            nl_strategy_dir=nl_strategy_dir,
            python_strategy_dir=python_strategy_dir,
            data_provider=data_provider,
            config=self.config,
        )

    def set_data_provider(self, data_provider: any) -> None:
        self.strategy_manager.set_data_provider(data_provider)

    def screen_stocks(
        self,
        stock_codes: Optional[List[str]],
        strategy_ids: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ) -> List[StockCandidate]:
        if top_n is None:
            top_n = self.config.default_top_n

        # Use complete stock pool if none provided
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
            logger.info("No stock codes provided, using complete stock pool: %d stocks", len(stock_codes))

        candidates: List[StockCandidate] = []

        # Use thread pool for parallel processing
        max_workers = min(10, len(stock_codes))  # Limit concurrent workers to avoid rate limiting
        logger.info("Using thread pool with %d workers", max_workers)

        def process_stock(code):
            try:
                return self._screen_single_stock(code, strategy_ids)
            except Exception as e:
                logger.error("Failed to screen stock %s: %s", code, e)
                return None

        # Process stocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_stock, stock_codes))

        # Collect valid candidates
        for result in results:
            if result:
                candidates.append(result)

        ranked_candidates = self.strategy_manager.rank_candidates(candidates, top_n=top_n)
        logger.info("Screened %d stocks, returned top %d", len(stock_codes), len(ranked_candidates))
        return ranked_candidates

    def _screen_single_stock(
        self,
        stock_code: str,
        strategy_ids: Optional[List[str]] = None,
    ) -> Optional[StockCandidate]:
        matches = self.strategy_manager.execute_strategies(
            stock_code=stock_code,
            strategy_ids=strategy_ids,
        )

        if not matches:
            return None

        current_price = 0.0
        stock_name = stock_code

        # Try to get data from database cache first
        if self._db_manager:
            from datetime import date
            today = date.today()
            if self._db_manager.has_today_data(stock_code, today):
                logger.info(f"Using cached data for {stock_code} from database in service")
                context = self._db_manager.get_analysis_context(stock_code, today)
                if context:
                    current_price = context.get('today', {}).get('close', 0.0)
                    stock_name = context.get('stock_name', stock_code)
                    # Create a candidate with cached data
                    candidate = StockCandidate(
                        code=stock_code,
                        name=stock_name,
                        current_price=current_price,
                    )
                    for match in matches:
                        candidate.add_strategy_match(match)
                    return candidate

        # Fallback to realtime data if no cache
        try:
            if self.strategy_manager._data_provider:
                quote = self.strategy_manager._data_provider.get_realtime_quote(stock_code)
                if quote:
                    current_price = getattr(quote, "price", 0.0)
                    if getattr(quote, "name", None):
                        stock_name = quote.name
        except Exception as e:
            logger.warning("Failed to get realtime quote for %s: %s", stock_code, e)

        candidate = StockCandidate(
            code=stock_code,
            name=stock_name,
            current_price=current_price,
        )

        for match in matches:
            candidate.add_strategy_match(match)

        return candidate

    def get_available_strategies(self) -> List[StrategyMetadata]:
        strategies = self.strategy_manager.get_all_strategies()
        return [s.metadata for s in strategies]

    def activate_strategies(self, strategy_ids: List[str]) -> None:
        self.strategy_manager.deactivate_all_strategies()
        for sid in strategy_ids:
            self.strategy_manager.activate_strategy(sid)

    def get_active_strategy_ids(self) -> List[str]:
        return [s.id for s in self.strategy_manager.get_active_strategies()]

    def deactivate_strategies(self, strategy_ids: List[str]) -> None:
        for sid in strategy_ids:
            self.strategy_manager.deactivate_strategy(sid)
