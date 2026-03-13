# -*- coding: utf-8 -*-
"""
Watchdog Configuration Module

Configuration management for the watchdog module.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from watchdog.base import AlertLevel, WatchdogAlert, WatchdogWatchlist

logger = logging.getLogger(__name__)


@dataclass
class WatchdogConfig:
    """
    Watchdog Configuration.

    All configuration options for the watchdog module, with sensible defaults.
    """

    # Watchlist persistence
    watchlist_file: Optional[str] = None
    alerts_file: Optional[str] = None

    # Monitoring settings
    check_interval_seconds: int = 60
    max_alerts_per_stock_per_hour: int = 3
    alert_cooldown_minutes: int = 30

    # Market hours (China A-share)
    market_open_start: str = "09:30"
    market_open_end: str = "15:00"
    monitor_only_during_market_hours: bool = True

    # Strategies
    strategy_dir: Optional[str] = None
    default_strategies: List[str] = field(default_factory=list)

    # Notifications
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=list)

    # Logging
    log_level: str = "INFO"
    debug_monitoring: bool = False

    @classmethod
    def from_env(cls) -> "WatchdogConfig":
        """
        Load configuration from environment variables.
        """
        import os
        from dotenv import load_dotenv

        # Load environment variables from .env file
        env_file = os.getenv("ENV_FILE")
        if env_file:
            env_path = Path(env_file)
        else:
            env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

        project_root = Path(__file__).parent.parent
        default_watchlist_file = project_root / "data" / "watchdog_watchlist.json"
        default_alerts_file = project_root / "data" / "watchdog_alerts.json"
        default_strategy_dir = project_root / "watchdog" / "strategies"

        return cls(
            watchlist_file=os.getenv(
                "WATCHDOG_WATCHLIST_FILE",
                str(default_watchlist_file) if default_watchlist_file else None
            ),
            alerts_file=os.getenv(
                "WATCHDOG_ALERTS_FILE",
                str(default_alerts_file) if default_alerts_file else None
            ),
            check_interval_seconds=int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60")),
            max_alerts_per_stock_per_hour=int(os.getenv("WATCHDOG_MAX_ALERTS_PER_HOUR", "3")),
            alert_cooldown_minutes=int(os.getenv("WATCHDOG_ALERT_COOLDOWN", "30")),
            market_open_start=os.getenv("WATCHDOG_MARKET_OPEN_START", "09:30"),
            market_open_end=os.getenv("WATCHDOG_MARKET_OPEN_END", "15:00"),
            monitor_only_during_market_hours=os.getenv(
                "WATCHDOG_MONITOR_MARKET_HOURS_ONLY", "true"
            ).lower() == "true",
            strategy_dir=os.getenv(
                "WATCHDOG_STRATEGY_DIR",
                str(default_strategy_dir) if default_strategy_dir.exists() else None
            ),
            default_strategies=[
                s.strip() for s in os.getenv("WATCHDOG_DEFAULT_STRATEGIES", "").split(",")
                if s.strip()
            ],
            enable_notifications=os.getenv("WATCHDOG_ENABLE_NOTIFICATIONS", "true").lower() == "true",
            notification_channels=[
                s.strip() for s in os.getenv("WATCHDOG_NOTIFICATION_CHANNELS", "").split(",")
                if s.strip()
            ],
            log_level=os.getenv("WATCHDOG_LOG_LEVEL", "INFO"),
            debug_monitoring=os.getenv("WATCHDOG_DEBUG_MONITORING", "false").lower() == "true",
        )


_config_instance: Optional[WatchdogConfig] = None


def get_config() -> WatchdogConfig:
    """Get the global watchdog config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = WatchdogConfig.from_env()
    return _config_instance


def set_config(config: WatchdogConfig) -> None:
    """Set the global watchdog config instance."""
    global _config_instance
    _config_instance = config


class WatchdogPersistence:
    """
    Watchlist and alerts persistence manager.
    """

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self._watchlist_cache: Optional[WatchdogWatchlist] = None
        self._alerts_cache: List[WatchdogAlert] = []

    def _ensure_data_dir(self, file_path: str) -> None:
        """Ensure the directory for a file exists."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def load_watchlist(self) -> WatchdogWatchlist:
        """
        Load watchlist from file.
        
        Returns:
            WatchdogWatchlist instance
        """
        if self._watchlist_cache is not None:
            return self._watchlist_cache

        if not self.config.watchlist_file:
            logger.warning("No watchlist file configured, using empty watchlist")
            self._watchlist_cache = WatchdogWatchlist()
            return self._watchlist_cache

        try:
            file_path = Path(self.config.watchlist_file)
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._watchlist_cache = WatchdogWatchlist.from_dict(data)
                logger.info(f"Loaded watchlist with {len(self._watchlist_cache.items)} items")
            else:
                logger.info("Watchlist file not found, creating empty watchlist")
                self._watchlist_cache = WatchdogWatchlist()
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
            self._watchlist_cache = WatchdogWatchlist()

        return self._watchlist_cache

    def save_watchlist(self, watchlist: WatchdogWatchlist) -> bool:
        """
        Save watchlist to file.
        
        Args:
            watchlist: WatchdogWatchlist to save
            
        Returns:
            True if successful
        """
        if not self.config.watchlist_file:
            logger.warning("No watchlist file configured, not saving")
            return False

        try:
            self._ensure_data_dir(self.config.watchlist_file)
            with open(self.config.watchlist_file, "w", encoding="utf-8") as f:
                json.dump(watchlist.to_dict(), f, ensure_ascii=False, indent=2)
            self._watchlist_cache = watchlist
            logger.info(f"Saved watchlist with {len(watchlist.items)} items")
            return True
        except Exception as e:
            logger.error(f"Error saving watchlist: {e}")
            return False

    def load_alerts(self, max_count: int = 100) -> List[WatchdogAlert]:
        """
        Load alerts from file.
        
        Args:
            max_count: Maximum number of alerts to load (most recent first)
            
        Returns:
            List of WatchdogAlert instances
        """
        if not self.config.alerts_file:
            logger.warning("No alerts file configured")
            return []

        try:
            file_path = Path(self.config.alerts_file)
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
                # Convert to Alert objects
                alerts = []
                for data in data_list[-max_count:]:
                    alert = WatchdogAlert(
                        id=data.get("id", ""),
                        stock_code=data.get("stock_code", ""),
                        stock_name=data.get("stock_name", ""),
                        strategy_id=data.get("strategy_id", ""),
                        strategy_name=data.get("strategy_name", ""),
                        alert_level=AlertLevel(data.get("alert_level", "info")),
                        message=data.get("message", ""),
                        trigger_time=datetime.fromisoformat(data.get("trigger_time", datetime.now().isoformat())),
                        triggered_conditions=[],
                        current_data=data.get("current_data", {}),
                        acked=data.get("acked", False),
                    )
                    alerts.append(alert)
                logger.info(f"Loaded {len(alerts)} alerts")
                return alerts
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
        return []

    def save_alert(self, alert: WatchdogAlert) -> bool:
        """
        Save a single alert to file.
        
        Args:
            alert: WatchdogAlert to save
            
        Returns:
            True if successful
        """
        if not self.config.alerts_file:
            logger.warning("No alerts file configured, not saving")
            return False

        try:
            self._ensure_data_dir(self.config.alerts_file)
            file_path = Path(self.config.alerts_file)
            
            # Load existing alerts
            data_list = []
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
            
            # Add new alert
            data_list.append(alert.to_dict())
            
            # Keep only last 1000 alerts
            if len(data_list) > 1000:
                data_list = data_list[-1000:]
            
            # Save
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved alert for {alert.stock_code}")
            return True
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return False
