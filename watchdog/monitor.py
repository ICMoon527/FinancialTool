# -*- coding: utf-8 -*-
"""
Watchdog Monitor Module

Real-time market data monitoring engine.
"""

import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple

from watchdog.base import (
    AlertLevel,
    ConditionType,
    StrategyType,
    WatchdogAlert,
    WatchdogCondition,
    WatchdogDecision,
    WatchdogStrategy,
    WatchdogWatchlist,
)
from watchdog.config import WatchdogConfig

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Market data provider wrapper for watchdog module.
    """

    def __init__(self):
        self._data_manager = None
        self._initialize_data_manager()

    def _initialize_data_manager(self) -> None:
        """Initialize the project's data fetcher manager."""
        try:
            from data_provider.base import DataFetcherManager
            self._data_manager = DataFetcherManager()
            logger.info("Market data provider initialized successfully")
        except ImportError as e:
            logger.warning(f"Data provider not available: {e}")
            self._data_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize data provider: {e}")
            self._data_manager = None

    def is_available(self) -> bool:
        """Check if market data provider is available."""
        return self._data_manager is not None

    def get_realtime_quote(self, stock_code: str) -> Optional[Dict]:
        """
        Get real-time quote for a stock.

        Args:
            stock_code: Stock code

        Returns:
            Quote data as dict, or None if failed
        """
        if not self.is_available():
            return None

        try:
            quote = self._data_manager.get_realtime_quote(stock_code)
            if quote:
                return quote.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {stock_code}: {e}")
            return None

    def get_stock_name(self, stock_code: str) -> str:
        """
        Get stock name.

        Args:
            stock_code: Stock code

        Returns:
            Stock name
        """
        if not self.is_available():
            return stock_code

        try:
            name = self._data_manager.get_stock_name(stock_code)
            return name or stock_code
        except Exception as e:
            logger.error(f"Failed to get stock name for {stock_code}: {e}")
            return stock_code


class WatchdogMonitor:
    """
    Watchdog Monitor - monitors market data and checks strategy conditions.
    """

    def __init__(self, config: WatchdogConfig):
        """
        Initialize the monitor.

        Args:
            config: Watchdog configuration
        """
        self.config = config
        self.data_provider = MarketDataProvider()
        self._strategies: Dict[str, WatchdogStrategy] = {}
        self._last_check_times: Dict[str, datetime.datetime] = {}
        self._alert_cooldowns: Dict[str, datetime.datetime] = {}

    def register_strategy(self, strategy: WatchdogStrategy) -> None:
        """
        Register a strategy.

        Args:
            strategy: WatchdogStrategy to register
        """
        self._strategies[strategy.id] = strategy
        logger.info(f"Registered strategy: {strategy.name} ({strategy.id})")

    def unregister_strategy(self, strategy_id: str) -> None:
        """
        Unregister a strategy.

        Args:
            strategy_id: Strategy ID to unregister
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger.info(f"Unregistered strategy: {strategy_id}")

    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open
        """
        if not self.config.monitor_only_during_market_hours:
            return True

        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")

        market_open = self.config.market_open_start <= current_time <= self.config.market_open_end

        weekday = now.weekday()
        is_weekday = 0 <= weekday <= 4

        return market_open and is_weekday

    def check_cooldown(self, stock_code: str, strategy_id: str) -> bool:
        """
        Check if alert cooldown period has passed.

        Args:
            stock_code: Stock code
            strategy_id: Strategy ID

        Returns:
            True if cooldown has passed
        """
        key = f"{stock_code}:{strategy_id}"
        if key not in self._alert_cooldowns:
            return True

        last_alert = self._alert_cooldowns[key]
        cooldown = datetime.timedelta(minutes=self.config.alert_cooldown_minutes)
        return datetime.datetime.now() - last_alert > cooldown

    def update_cooldown(self, stock_code: str, strategy_id: str) -> None:
        """
        Update alert cooldown timestamp.

        Args:
            stock_code: Stock code
            strategy_id: Strategy ID
        """
        key = f"{stock_code}:{strategy_id}"
        self._alert_cooldowns[key] = datetime.datetime.now()

    def evaluate_condition(
        self, condition: WatchdogCondition, current_data: Dict
    ) -> Tuple[bool, str]:
        """
        Evaluate a single condition against current market data.

        Args:
            condition: Condition to evaluate
            current_data: Current market data

        Returns:
            Tuple of (is_triggered, description)
        """
        price = current_data.get("price")
        change_pct = current_data.get("change_pct")
        volume_ratio = current_data.get("volume_ratio")
        pre_close = current_data.get("pre_close")

        if condition.condition_type == ConditionType.PRICE_ABOVE:
            threshold = condition.parameters.get("threshold", 0)
            if price is not None and price > threshold:
                return True, f"价格 {price} 超过阈值 {threshold}"
        elif condition.condition_type == ConditionType.PRICE_BELOW:
            threshold = condition.parameters.get("threshold", 0)
            if price is not None and price < threshold:
                return True, f"价格 {price} 低于阈值 {threshold}"
        elif condition.condition_type == ConditionType.CHANGE_ABOVE:
            threshold = condition.parameters.get("threshold", 0)
            if change_pct is not None and change_pct > threshold:
                return True, f"涨幅 {change_pct}% 超过阈值 {threshold}%"
        elif condition.condition_type == ConditionType.CHANGE_BELOW:
            threshold = condition.parameters.get("threshold", 0)
            if change_pct is not None and change_pct < threshold:
                return True, f"跌幅 {abs(change_pct)}% 超过阈值 {abs(threshold)}%"
        elif condition.condition_type == ConditionType.VOLUME_RATIO_ABOVE:
            threshold = condition.parameters.get("threshold", 1)
            if volume_ratio is not None and volume_ratio > threshold:
                return True, f"量比 {volume_ratio} 超过阈值 {threshold}"

        return False, ""

    def check_watchlist(
        self, watchlist: WatchdogWatchlist
    ) -> List[WatchdogAlert]:
        """
        Check all items in watchlist for triggered conditions.

        Args:
            watchlist: WatchdogWatchlist to check

        Returns:
            List of triggered WatchdogAlert
        """
        alerts = []

        if not self.is_market_open():
            logger.debug("Market closed, skipping watchlist check")
            return alerts

        for item in watchlist.items:
            if not item.enabled:
                continue

            stock_code = item.stock_code

            current_data = self.data_provider.get_realtime_quote(stock_code)
            if not current_data:
                logger.warning(f"No data available for {stock_code}")
                continue

            stock_name = self.data_provider.get_stock_name(stock_code)

            for strategy_id in item.strategy_ids:
                if strategy_id not in self._strategies:
                    continue

                strategy = self._strategies[strategy_id]

                if not self.check_cooldown(stock_code, strategy_id):
                    logger.debug(f"Cooldown active for {stock_code}:{strategy_id}")
                    continue

                triggered_conditions = []
                for condition in strategy.conditions:
                    is_triggered, desc = self.evaluate_condition(condition, current_data)
                    if is_triggered:
                        triggered_condition = WatchdogCondition(
                            condition_type=condition.condition_type,
                            parameters=condition.parameters,
                            description=desc,
                        )
                        triggered_conditions.append(triggered_condition)

                should_trigger = False
                if strategy.strategy_type == StrategyType.ALL_CONDITIONS:
                    should_trigger = len(triggered_conditions) == len(strategy.conditions)
                elif strategy.strategy_type == StrategyType.ANY_CONDITION:
                    should_trigger = len(triggered_conditions) > 0

                if should_trigger:
                    alert = self._create_alert(
                        stock_code=stock_code,
                        stock_name=stock_name,
                        strategy=strategy,
                        triggered_conditions=triggered_conditions,
                        current_data=current_data,
                    )
                    alerts.append(alert)
                    self.update_cooldown(stock_code, strategy_id)

        return alerts

    def _create_alert(
        self,
        stock_code: str,
        stock_name: str,
        strategy: WatchdogStrategy,
        triggered_conditions: List[WatchdogCondition],
        current_data: Dict,
    ) -> WatchdogAlert:
        """
        Create a WatchdogAlert from triggered conditions.

        Args:
            stock_code: Stock code
            stock_name: Stock name
            strategy: Triggered strategy
            triggered_conditions: List of triggered conditions
            current_data: Current market data

        Returns:
            WatchdogAlert instance
        """
        alert_id = f"{stock_code}_{strategy.id}_{int(time.time())}"

        decision = WatchdogDecision(
            action=strategy.default_action,
            reason=strategy.default_reason or "策略条件触发",
            target_price=strategy.target_price,
            stop_loss=strategy.stop_loss,
        )

        alert = WatchdogAlert(
            id=alert_id,
            stock_code=stock_code,
            stock_name=stock_name,
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            alert_level=strategy.alert_level or AlertLevel.WARNING,
            message=f"{strategy.name} 策略触发",
            trigger_time=datetime.datetime.now(),
            triggered_conditions=triggered_conditions,
            current_data=current_data,
            decision=decision,
            acked=False,
        )

        return alert
