# -*- coding: utf-8 -*-
"""
Natural Language Strategy Loader.

Loads strategies from YAML files (compatible with existing strategies/ directory).
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)

logger = logging.getLogger(__name__)


class NaturalLanguageStrategy(StockSelectorStrategy):
    """
    Natural language strategy wrapper.

    This class adapts YAML-based strategies from the existing strategies/ directory
    to the stock selector interface.
    """

    def __init__(self, yaml_path: Path):
        """
        Initialize from a YAML file.

        Args:
            yaml_path: Path to the YAML strategy file
        """
        self.yaml_path = yaml_path
        self._load_from_yaml(yaml_path)
        metadata = StrategyMetadata(
            id=self._yaml_data.get("name", yaml_path.stem),
            name=self._yaml_data.get("name", yaml_path.stem),
            display_name=self._yaml_data.get("display_name", yaml_path.stem),
            description=self._yaml_data.get("description", ""),
            strategy_type=StrategyType.NATURAL_LANGUAGE,
            category=self._yaml_data.get("category", "trend"),
            source=str(yaml_path),
            version="1.0.0",
        )
        super().__init__(metadata)

    def _load_from_yaml(self, yaml_path: Path) -> None:
        """Load strategy data from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._yaml_data = yaml.safe_load(f)

    @property
    def instructions(self) -> str:
        """Get strategy instructions from YAML."""
        return self._yaml_data.get("instructions", "")

    def select(self, stock_code: str, stock_name: Optional[str] = None) -> StrategyMatch:
        """
        Execute natural language strategy for a single stock.

        Note: For natural language strategies, actual execution would require
        an LLM to parse and apply the instructions. This method currently
        returns a placeholder result and is intended to be extended with LLM integration.

        Args:
            stock_code: Stock code to analyze
            stock_name: Optional stock name

        Returns:
            StrategyMatch result (placeholder)
        """
        logger.warning(
            f"Natural language strategy {self.id} requires LLM integration "
            f"for actual execution. Returning placeholder result."
        )
        return StrategyMatch(
            strategy_id=self.id,
            strategy_name=self.display_name,
            matched=False,
            score=0.0,
            reason="Natural language strategy requires LLM integration",
            match_details={"yaml_instructions": self.instructions},
        )


def load_nl_strategies_from_dir(directory: Path) -> List[NaturalLanguageStrategy]:
    """
    Load all natural language strategies from a directory.

    Args:
        directory: Directory containing YAML strategy files

    Returns:
        List of NaturalLanguageStrategy instances
    """
    strategies = []
    if not directory.is_dir():
        logger.warning(f"Strategy directory not found: {directory}")
        return strategies

    yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    for yaml_path in yaml_files:
        try:
            strategy = NaturalLanguageStrategy(yaml_path)
            strategies.append(strategy)
            logger.debug(f"Loaded NL strategy: {strategy.id} from {yaml_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load NL strategy from {yaml_path}: {e}")

    logger.info(f"Loaded {len(strategies)} natural language strategies from {directory}")
    return strategies
