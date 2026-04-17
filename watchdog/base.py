# -*- coding: utf-8 -*-
"""
Base classes and data structures for watchdog module.
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


class ConditionType(Enum):
    """Condition type enum."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_PCT_ABOVE = "change_pct_above"
    CHANGE_PCT_BELOW = "change_pct_below"
    MA_CROSS_UP = "ma_cross_up"
    MA_CROSS_DOWN = "ma_cross_down"
    VOLUME_SURGE = "volume_surge"
    VOLUME_DROP = "volume_drop"
    BIAS_RATIO_ABOVE = "bias_ratio_above"
    BIAS_RATIO_BELOW = "bias_ratio_below"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity level."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class StrategyMetadata:
    """Metadata for a watchdog strategy."""
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
class WatchdogCondition:
    """Single trigger condition for a strategy."""
    condition_type: ConditionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_type": self.condition_type.value,
            "parameters": self.parameters,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchdogCondition":
        """Create from dictionary."""
        return cls(
            condition_type=ConditionType(data["condition_type"]),
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )


@dataclass
class WatchdogAlert:
    """Alert notification data structure."""
    stock_code: str
    stock_name: str
    strategy_id: str
    strategy_name: str
    alert_level: AlertLevel
    message: str
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    trigger_time: datetime = field(default_factory=datetime.now)
    triggered_conditions: List[WatchdogCondition] = field(default_factory=list)
    current_data: Dict[str, Any] = field(default_factory=dict)
    acked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "alert_level": self.alert_level.value,
            "message": self.message,
            "trigger_time": self.trigger_time.isoformat(),
            "triggered_conditions": [c.to_dict() for c in self.triggered_conditions],
            "current_data": self.current_data,
            "acked": self.acked,
        }


@dataclass
class WatchdogDecision:
    """Decision recommendation data structure."""
    stock_code: str
    stock_name: str
    strategy_id: str
    decision_type: str  # "buy", "sell", "hold", "watch"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    suggested_action: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "strategy_id": self.strategy_id,
            "decision_type": self.decision_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggested_action": self.suggested_action,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WatchlistItem:
    """Single stock in watchlist with strategy configuration."""
    stock_code: str
    stock_name: str
    strategy_ids: List[str] = field(default_factory=list)
    custom_conditions: List[WatchdogCondition] = field(default_factory=list)
    enabled: bool = True
    added_at: datetime = field(default_factory=datetime.now)
    last_alert_time: Optional[datetime] = None
    alert_cooldown_minutes: int = 30  # Minimum minutes between alerts for same stock

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "strategy_ids": self.strategy_ids,
            "custom_conditions": [c.to_dict() for c in self.custom_conditions],
            "enabled": self.enabled,
            "added_at": self.added_at.isoformat(),
            "last_alert_time": self.last_alert_time.isoformat() if self.last_alert_time else None,
            "alert_cooldown_minutes": self.alert_cooldown_minutes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchlistItem":
        """Create from dictionary."""
        last_alert_time = None
        if data.get("last_alert_time"):
            try:
                last_alert_time = datetime.fromisoformat(data["last_alert_time"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            stock_code=data["stock_code"],
            stock_name=data["stock_name"],
            strategy_ids=data.get("strategy_ids", []),
            custom_conditions=[WatchdogCondition.from_dict(c) for c in data.get("custom_conditions", [])],
            enabled=data.get("enabled", True),
            added_at=datetime.fromisoformat(data.get("added_at", datetime.now().isoformat())),
            last_alert_time=last_alert_time,
            alert_cooldown_minutes=data.get("alert_cooldown_minutes", 30),
        )


@dataclass
class WatchdogWatchlist:
    """Watchlist management data structure."""
    id: str = "default"
    name: str = "我的自选股"
    items: List[WatchlistItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_item(self, stock_code: str) -> Optional[WatchlistItem]:
        """Get watchlist item by stock code."""
        for item in self.items:
            if item.stock_code == stock_code:
                return item
        return None

    def add_item(self, item: WatchlistItem) -> bool:
        """Add item to watchlist."""
        if self.get_item(item.stock_code):
            logger.warning(f"Stock {item.stock_code} already in watchlist")
            return False
        self.items.append(item)
        self.updated_at = datetime.now()
        return True

    def remove_item(self, stock_code: str) -> bool:
        """Remove item from watchlist."""
        original_len = len(self.items)
        self.items = [item for item in self.items if item.stock_code != stock_code]
        if len(self.items) != original_len:
            self.updated_at = datetime.now()
            return True
        return False

    def update_item(self, item: WatchlistItem) -> bool:
        """Update existing item."""
        for i, existing in enumerate(self.items):
            if existing.stock_code == item.stock_code:
                self.items[i] = item
                self.updated_at = datetime.now()
                return True
        return False

    def get_enabled_items(self) -> List[WatchlistItem]:
        """Get all enabled items."""
        return [item for item in self.items if item.enabled]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchdogWatchlist":
        """Create from dictionary."""
        return cls(
            id=data.get("id", "default"),
            name=data.get("name", "我的自选股"),
            items=[WatchlistItem.from_dict(item) for item in data.get("items", [])],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
        )


class WatchdogStrategy(ABC):
    """Abstract base class for watchdog strategies."""

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
    def get_conditions(self) -> List[WatchdogCondition]:
        """
        Get the trigger conditions for this strategy.
        
        Returns:
            List of WatchdogCondition objects
        """
        pass

    @abstractmethod
    def check_conditions(
        self,
        stock_code: str,
        stock_name: str,
        current_data: Dict[str, Any]
    ) -> Tuple[bool, List[WatchdogCondition], WatchdogDecision]:
        """
        Check if strategy conditions are met.
        
        Args:
            stock_code: Stock code
            stock_name: Stock name
            current_data: Current market data for the stock
            
        Returns:
            Tuple of (triggered, triggered_conditions, decision)
        """
        pass

    def format_alert_message(
        self,
        stock_code: str,
        stock_name: str,
        triggered_conditions: List[WatchdogCondition],
        decision: WatchdogDecision
    ) -> str:
        """
        Format alert message.
        
        Args:
            stock_code: Stock code
            stock_name: Stock name
            triggered_conditions: List of triggered conditions
            decision: Decision recommendation
            
        Returns:
            Formatted alert message
        """
        conditions_str = "\n".join([f"- {c.description}" for c in triggered_conditions])
        
        return f"""
【盯盘预警】{stock_name}({stock_code})

触发策略: {self.display_name}

触发条件:
{conditions_str}

决策建议: {decision.suggested_action}
置信度: {int(decision.confidence * 100)}%

{decision.reasoning}
""".strip()
