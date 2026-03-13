# -*- coding: utf-8 -*-
"""
Base classes and data structures for stock selector.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enum."""
    NATURAL_LANGUAGE = "nl"
    PYTHON = "python"


@dataclass
class StrategyMetadata:
    """Metadata for a strategy."""
    id: str
    name: str
    display_name: str
    description: str
    strategy_type: StrategyType
    category: str = "trend"
    source: str = "builtin"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class StrategyMatch:
    """Single strategy match result."""
    strategy_id: str
    strategy_name: str
    matched: bool
    score: float = 0.0
    reason: Optional[str] = None
    match_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StockCandidate:
    """Stock candidate data structure."""
    code: str
    name: str
    current_price: float
    match_score: float = 0.0
    strategy_matches: List[StrategyMatch] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def get_matched_strategies(self) -> List[StrategyMatch]:
        """Get list of matched strategies."""
        return [m for m in self.strategy_matches if m.matched]

    def get_unmatched_strategies(self) -> List[StrategyMatch]:
        """Get list of unmatched strategies."""
        return [m for m in self.strategy_matches if not m.matched]

    def add_strategy_match(self, match: StrategyMatch) -> None:
        """Add a strategy match result."""
        self.strategy_matches.append(match)
        self._recalculate_match_score()

    def _recalculate_match_score(self) -> None:
        """Recalculate overall match score based on strategy matches."""
        if not self.strategy_matches:
            self.match_score = 0.0
            return
        matched = [m for m in self.strategy_matches if m.matched]
        if not matched:
            self.match_score = 0.0
            return
        total = sum(m.score for m in matched)
        self.match_score = total / len(self.strategy_matches)


class StockSelectorStrategy(ABC):
    """Abstract base class for stock selector strategies."""

    def __init__(self, metadata: StrategyMetadata):
        self.metadata = metadata
        self._data_provider = None

    @property
    def id(self) -> str:
        """Get strategy ID."""
        return self.metadata.id

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.metadata.name

    @property
    def display_name(self) -> str:
        """Get strategy display name."""
        return self.metadata.display_name

    def set_data_provider(self, data_provider: Any) -> None:
        """Set data provider for the strategy."""
        self._data_provider = data_provider

    @abstractmethod
    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute strategy for a single stock.

        Args:
            stock_code: Stock code to analyze
            stock_name: Optional stock name

        Returns:
            StrategyMatch result
        """
        pass

    def select_batch(self, stock_codes: List[str]) -> List[Tuple[str, StrategyMatch]]:
        """
        Execute strategy for multiple stocks.

        Args:
            stock_codes: List of stock codes

        Returns:
            List of (stock_code, StrategyMatch) tuples
        """
        results = []
        for code in stock_codes:
            try:
                match = self.select(code)
                results.append((code, match))
            except Exception as e:
                logger.error(f"Strategy {self.id} failed for {code}: {e}")
                match = StrategyMatch(
                    strategy_id=self.id,
                    strategy_name=self.display_name,
                    matched=False,
                    reason=f"Strategy execution error: {str(e)}",
                )
                results.append((code, match))
        return results
