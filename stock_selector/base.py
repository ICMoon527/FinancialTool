# -*- coding: utf-8 -*-
"""
Base classes and data structures for stock selector.
"""

import logging
import threading
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
    score_multiplier: float = 1.0
    max_raw_score: float = 100.0


@dataclass
class StrategyMatch:
    """Single strategy match result."""
    strategy_id: str
    strategy_name: str
    matched: bool
    score: float = 0.0
    raw_score: float = 0.0
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
        self.match_score = total / len(matched)


class StockSelectorStrategy(ABC):
    """Abstract base class for stock selector strategies."""

    def __init__(self, metadata: StrategyMetadata):
        self.metadata = metadata
        self._sector_manager = None
        self._score_multiplier = metadata.score_multiplier
        self._max_raw_score = metadata.max_raw_score
        
        # 线程局部存储容器
        self._local = threading.local()
        # 初始化默认值
        self._local._data_provider = None

    @property
    def score_multiplier(self) -> float:
        """Get the score multiplier."""
        return self._score_multiplier

    @score_multiplier.setter
    def score_multiplier(self, value: float) -> None:
        """Set the score multiplier."""
        self._score_multiplier = max(0.0, value)

    @property
    def max_raw_score(self) -> float:
        """Get the maximum raw score."""
        return self._max_raw_score

    @max_raw_score.setter
    def max_raw_score(self, value: float) -> None:
        """Set the maximum raw score."""
        self._max_raw_score = max(1.0, value)
    
    def set_data_provider(self, data_provider: Any) -> None:
        """设置默认的数据提供者（向后兼容）"""
        self._local._data_provider = data_provider
    
    def set_data_provider_for_thread(self, data_provider: Any) -> None:
        """为当前线程设置数据提供者"""
        self._local._data_provider = data_provider
    
    def get_data_provider(self) -> Any:
        """获取当前线程的数据提供者"""
        return getattr(self._local, '_data_provider', None)
    
    # 保持向后兼容：通过 property 让 self._data_provider 仍然能工作
    @property
    def _data_provider(self):
        return self.get_data_provider()
    
    @_data_provider.setter
    def _data_provider(self, value):
        self.set_data_provider_for_thread(value)

    def normalize_score(self, raw_score: float) -> float:
        """
        Normalize a raw score to the 0-100 range using linear mapping.
        
        Args:
            raw_score: The raw score from the strategy
            
        Returns:
            Normalized score in 0-100 range
        """
        # Apply multiplier to raw score
        multiplied_score = raw_score * self._score_multiplier
        
        # Normalize to 0-100 range
        normalized_score = (multiplied_score / self._max_raw_score) * 100.0
        
        # Clamp to 0-100 range
        return max(0.0, min(100.0, normalized_score))

    def create_strategy_match(
        self, 
        raw_score: float, 
        matched: bool, 
        reason: Optional[str] = None,
        match_details: Optional[Dict[str, Any]] = None
    ) -> StrategyMatch:
        """
        Create a StrategyMatch with normalized score.
        
        Args:
            raw_score: The raw score from the strategy
            matched: Whether the strategy matched
            reason: Optional reason for the match
            match_details: Optional match details
            
        Returns:
            StrategyMatch with normalized score
        """
        normalized_score = self.normalize_score(raw_score)
        
        return StrategyMatch(
            strategy_id=self.id,
            strategy_name=self.display_name,
            matched=matched,
            score=normalized_score,
            raw_score=raw_score,
            reason=reason,
            match_details=match_details or {},
        )

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
    
    def set_sector_manager(self, sector_manager: Any) -> None:
        """Set sector manager for the strategy."""
        self._sector_manager = sector_manager
    
    @property
    def sector_manager(self) -> Any:
        """Get sector manager."""
        return self._sector_manager

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
