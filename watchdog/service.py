# -*- coding: utf-8 -*-
"""
Watchdog Service Module

Main service class that orchestrates the entire watchdog system.
"""

import logging
import threading
import time
from typing import List, Optional

from watchdog.base import WatchdogAlert, WatchdogStrategy, WatchdogWatchlist
from watchdog.config import WatchdogConfig, WatchdogPersistence, get_config
from watchdog.monitor import WatchdogMonitor
from watchdog.notifier import WatchdogNotifier

logger = logging.getLogger(__name__)


class WatchdogService:
    """
    Watchdog Service - main orchestrator for the watchdog system.

    This class integrates:
    - Watchlist management
    - Strategy management
    - Market monitoring
    - Alert notification
    - Data persistence
    """

    def __init__(self, config: Optional[WatchdogConfig] = None):
        """
        Initialize the watchdog service.

        Args:
            config: Watchdog configuration (uses default if not provided)
        """
        self.config = config or get_config()
        self.persistence = WatchdogPersistence(self.config)
        self.monitor = WatchdogMonitor(self.config)
        self.notifier = WatchdogNotifier(enable_notifications=self.config.enable_notifications)

        self._watchlist: Optional[WatchdogWatchlist] = None
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load_initial_data()
        self._register_default_strategies()

    def _load_initial_data(self) -> None:
        """Load initial watchlist from persistence."""
        self._watchlist = self.persistence.load_watchlist()
        logger.info(f"Loaded watchlist with {len(self._watchlist.items)} items")

    def _register_default_strategies(self) -> None:
        """Register default built-in strategies."""
        try:
            from watchdog.strategies import get_builtin_strategies

            strategies = get_builtin_strategies()
            for strategy in strategies:
                self.monitor.register_strategy(strategy)
            logger.info(f"Registered {len(strategies)} built-in strategies")
        except Exception as e:
            logger.warning(f"Failed to load built-in strategies: {e}")

    @property
    def watchlist(self) -> WatchdogWatchlist:
        """Get the current watchlist."""
        return self._watchlist

    def save_watchlist(self) -> bool:
        """
        Save the current watchlist to persistence.

        Returns:
            True if successful
        """
        return self.persistence.save_watchlist(self._watchlist)

    def add_stock_to_watchlist(
        self, stock_code: str, strategy_ids: Optional[List[str]] = None, notes: str = ""
    ) -> bool:
        """
        Add a stock to the watchlist.

        Args:
            stock_code: Stock code
            strategy_ids: List of strategy IDs to apply
            notes: Optional notes

        Returns:
            True if successful
        """
        success = self._watchlist.add_stock(
            stock_code=stock_code,
            strategy_ids=strategy_ids or [],
            notes=notes,
        )
        if success:
            self.save_watchlist()
        return success

    def remove_stock_from_watchlist(self, stock_code: str) -> bool:
        """
        Remove a stock from the watchlist.

        Args:
            stock_code: Stock code

        Returns:
            True if successful
        """
        success = self._watchlist.remove_stock(stock_code)
        if success:
            self.save_watchlist()
        return success

    def register_strategy(self, strategy: WatchdogStrategy) -> None:
        """
        Register a custom strategy.

        Args:
            strategy: WatchdogStrategy to register
        """
        self.monitor.register_strategy(strategy)

    def unregister_strategy(self, strategy_id: str) -> None:
        """
        Unregister a strategy.

        Args:
            strategy_id: Strategy ID to unregister
        """
        self.monitor.unregister_strategy(strategy_id)

    def check_once(self) -> List[WatchdogAlert]:
        """
        Perform a single check of the watchlist.

        Returns:
            List of triggered alerts
        """
        alerts = self.monitor.check_watchlist(self._watchlist)

        for alert in alerts:
            self.persistence.save_alert(alert)

        if alerts and self.notifier.is_available():
            self.notifier.send_alerts(alerts)

        return alerts

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in a separate thread)."""
        logger.info("Watchdog monitor loop started")

        while self._running and not self._stop_event.is_set():
            try:
                alerts = self.check_once()
                if alerts:
                    logger.info(f"Triggered {len(alerts)} alerts")

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            self._stop_event.wait(self.config.check_interval_seconds)

        logger.info("Watchdog monitor loop stopped")

    def start(self) -> None:
        """Start the watchdog monitoring service."""
        if self._running:
            logger.warning("Watchdog service is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Watchdog service started")

    def stop(self) -> None:
        """Stop the watchdog monitoring service."""
        if not self._running:
            logger.warning("Watchdog service is not running")
            return

        self._running = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("Watchdog service stopped")

    def is_running(self) -> bool:
        """Check if the service is currently running."""
        return self._running

    def get_recent_alerts(self, max_count: int = 50) -> List[WatchdogAlert]:
        """
        Get recent alerts from persistence.

        Args:
            max_count: Maximum number of alerts to return

        Returns:
            List of WatchdogAlert
        """
        return self.persistence.load_alerts(max_count=max_count)
