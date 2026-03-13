# -*- coding: utf-8 -*-
"""
Watchdog Usage Example

Demonstrates how to use the watchdog module.
"""

import logging
import time

from watchdog import WatchdogService


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_basic_usage():
    """Example: Basic usage of WatchdogService."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    service = WatchdogService()

    print(f"\nWatchdog service initialized")
    print(f"Watchlist has {len(service.watchlist.items)} items")

    print("\nBuilt-in strategies:")
    from watchdog.strategies import get_builtin_strategies
    strategies = get_builtin_strategies()
    for strategy in strategies:
        print(f"  - {strategy.name} ({strategy.id})")


def example_add_stock_to_watchlist():
    """Example: Add stocks to watchlist."""
    print("\n" + "=" * 60)
    print("Example 2: Add Stocks to Watchlist")
    print("=" * 60)

    service = WatchdogService()

    stock_codes = ["600519", "000001", "000002"]
    strategy_ids = ["price_drop_5pct", "price_rise_5pct"]

    for code in stock_codes:
        success = service.add_stock_to_watchlist(
            stock_code=code,
            strategy_ids=strategy_ids,
            notes=f"Added by example"
        )
        print(f"Added {code} to watchlist: {'Success' if success else 'Failed'}")

    print(f"\nWatchlist now has {len(service.watchlist.items)} items")
    for item in service.watchlist.items:
        print(f"  - {item.stock_code}: {item.strategy_ids}")


def example_single_check():
    """Example: Perform a single check."""
    print("\n" + "=" * 60)
    print("Example 3: Single Check")
    print("=" * 60)

    service = WatchdogService()

    print("\nPerforming single check...")
    alerts = service.check_once()

    if alerts:
        print(f"\nTriggered {len(alerts)} alerts:")
        for alert in alerts:
            print(f"  - {alert.stock_name}({alert.stock_code}): {alert.message}")
    else:
        print("\nNo alerts triggered")


def example_manage_alerts():
    """Example: Manage and view alerts."""
    print("\n" + "=" * 60)
    print("Example 4: Manage Alerts")
    print("=" * 60)

    service = WatchdogService()

    recent_alerts = service.get_recent_alerts(max_count=10)
    print(f"\nRecent alerts ({len(recent_alerts)}):")
    for alert in recent_alerts:
        print(f"  - [{alert.trigger_time}] {alert.stock_code}: {alert.message}")


def example_custom_strategy():
    """Example: Create and register a custom strategy."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Strategy")
    print("=" * 60)

    from watchdog import (
        ActionType,
        AlertLevel,
        ConditionType,
        StrategyType,
        WatchdogCondition,
        WatchdogStrategy,
    )

    service = WatchdogService()

    custom_strategy = WatchdogStrategy(
        id="custom_price_target",
        name="自定义价格目标",
        description="当价格达到1800元时触发",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.PRICE_ABOVE,
                parameters={"threshold": 1800.0},
                description="价格超过1800元",
            )
        ],
        alert_level=AlertLevel.WARNING,
        default_action=ActionType.SELL,
        default_reason="达到预设价格目标",
        target_price=1850.0,
        stop_loss=1750.0,
    )

    service.register_strategy(custom_strategy)
    print(f"Registered custom strategy: {custom_strategy.name}")

    service.add_stock_to_watchlist(
        stock_code="600519",
        strategy_ids=["custom_price_target"],
        notes="自定义价格策略"
    )
    print("Added 600519 with custom strategy to watchlist")


def main():
    """Run all examples."""
    setup_logging()

    print("\n" + "=" * 60)
    print("Watchdog Module Usage Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_add_stock_to_watchlist()
        example_single_check()
        example_manage_alerts()
        example_custom_strategy()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
