#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selector Command Line Interface
负责选股
Command line tool to interact with the stock selector system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional



logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_selector import StockSelectorService


def list_strategies(service: StockSelectorService):
    strategies = service.get_available_strategies()
    active_ids = service.get_active_strategy_ids()

    print("\nAvailable Strategies:")
    print("=" * 80)

    for strategy in strategies:
        status = "ACTIVE" if strategy.id in active_ids else "INACTIVE"
        strategy_type = strategy.strategy_type.name

        print(f"\n{strategy.display_name} ({strategy_type})")
        print(f"ID: {strategy.id}")
        print(f"Status: {status}")
        print(f"Description: {strategy.description}")
        print(f"Category: {strategy.category}")
        print(f"Source: {strategy.source}")
        print(f"Version: {strategy.version}")

    print("\n" + "=" * 80)
    print(f"Total: {len(strategies)} strategies")
    print(f"Active: {len(active_ids)} strategies")


def activate_strategy(service: StockSelectorService, strategy_ids: List[str]):
    service.activate_strategies(strategy_ids)
    print(f"Activated strategies: {', '.join(strategy_ids)}")


def deactivate_strategy(service: StockSelectorService, strategy_ids: List[str]):
    service.deactivate_strategies(strategy_ids)
    print(f"Deactivated strategies: {', '.join(strategy_ids)}")


def screen_stocks(service: StockSelectorService, stock_codes: Optional[List[str]], top_n: int, update_data: bool = False, update_days: int = 365):
    print("Screening stocks...")
    print(f"Top N: {top_n}")
    if stock_codes:
        print(f"Stock codes: {', '.join(stock_codes)}")
    else:
        print("Stock codes: All available")
    
    # 如果需要先更新数据
    if update_data:
        from stock_selector.batch_data_updater import get_batch_updater
        from stock_selector.data_update_tracker import get_update_tracker
        from datetime import date, timedelta
        from stock_selector.stock_pool import get_all_stock_codes
        
        print(f"\nChecking and updating stock data (up to last {update_days} days)...")
        
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
        
        end_date = date.today()
        target_start_date = end_date - timedelta(days=update_days - 1)
        
        # 使用更新追踪器来智能判断需要更新的日期范围
        update_tracker = get_update_tracker()
        
        # 统计需要更新的股票和日期
        latest_last_updated = None
        
        for code in stock_codes:
            record = update_tracker.get_update_record(code)
            if record is not None and record.last_updated_date is not None:
                if latest_last_updated is None or record.last_updated_date > latest_last_updated:
                    latest_last_updated = record.last_updated_date
        
        # 确定实际需要更新的开始日期
        if latest_last_updated is None:
            actual_start_date = target_start_date
            print(f"No existing data found, will update full {update_days} days")
        else:
            actual_start_date = latest_last_updated + timedelta(days=1)
            actual_start_date = max(actual_start_date, target_start_date)
            days_to_update = (end_date - actual_start_date).days + 1
            print(f"Existing data found up to {latest_last_updated}, will update {days_to_update} day(s)")
        
        if actual_start_date <= end_date:
            batch_updater = get_batch_updater()
            stats = batch_updater.update_stocks_for_date_range(
                stock_codes=stock_codes,
                start_date=actual_start_date,
                end_date=end_date
            )
            
            print(f"Data update complete: {stats['stocks_updated']} updated, {stats['stocks_failed']} failed\n")
        else:
            print("All data is already up to date!\n")

    candidates = service.screen_stocks(
        stock_codes=stock_codes,
        top_n=top_n
    )

    # Get sector manager
    sector_manager = None
    if service.strategy_manager:
        sector_manager = service.strategy_manager.get_sector_manager()

    print("\nTop Candidates:")
    print("=" * 120)

    for i, candidate in enumerate(candidates, 1):
        print(f"\nRank {i}: {candidate.code} - {candidate.name}")
        print(f"Score: {candidate.match_score:.2f}")
        print(f"Price: {candidate.current_price:.2f}")

        # Show sector information
        if sector_manager:
            try:
                sectors = sector_manager.get_stock_sectors(candidate.code)
                if sectors:
                    print(f"\nSectors: {', '.join(sectors)}")
                    
                    # Check if any sector is hot and show details
                    hot_sector_found = False
                    for sector_name in sectors:
                        is_hot = sector_manager.is_hot_sector(sector_name)
                        change_pct = sector_manager.get_sector_change_pct(sector_name)
                        
                        if is_hot and change_pct:
                            if not hot_sector_found:
                                print("Hot Sectors:")
                                hot_sector_found = True
                            print(f"  🔥 {sector_name}: +{change_pct:.2f}%")
                        elif change_pct:
                            if not hot_sector_found:
                                print("Sector Performance:")
                                hot_sector_found = True
                            if change_pct >= 0:
                                print(f"  • {sector_name}: +{change_pct:.2f}%")
                            else:
                                print(f"  • {sector_name}: {change_pct:.2f}%")
            except Exception as e:
                logger.warning(f"Failed to get sector info for {candidate.code}: {e}")

        print("\nStrategy Matches:")
        for match in candidate.strategy_matches:
            status = "✓" if match.matched else "✗"
            print(f"  {status} {match.strategy_name}: {match.score:.2f} - {match.reason}")

    print("\n" + "=" * 120)
    print(f"Found {len(candidates)} candidates")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Selector Command Line Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all strategies
  python -m stock_selector.main list

  # Activate strategies
  python -m stock_selector.main activate short_term_strategy ma_golden_cross

  # 激活 DragonLeader 组合策略（将同时激活三个子策略）
  python -m stock_selector.main activate dragon_leader

  # Deactivate strategies
  python -m stock_selector.main deactivate volume_breakout

  # Screen stocks (top 5)
  # 推荐用法：智能更新（默认检查365天）
  python -m stock_selector.main screen --update-data

  # Screen specific stocks
  python -m stock_selector.main screen --stocks 600519 000001

  # Screen with custom top N
  python -m stock_selector.main screen --top 10

  # Screen with custom log level (ERROR/WARNING/INFO/DEBUG)
  python -m stock_selector.main screen --log-level ERROR

  # Screen with smart data update (default: check and update up to last 365 days)
  python -m stock_selector.main screen --update-data

  # Screen with custom check window (e.g., only check last 7 days)
  python -m stock_selector.main screen --update-data --update-days 7
        """)

    # Add global log level argument
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO, use ERROR for less output during screening)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('list', help='List all available strategies')

    activate_parser = subparsers.add_parser('activate', help='Activate strategies')
    activate_parser.add_argument('strategy_ids', nargs='+', help='Strategy IDs to activate')

    deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate strategies')
    deactivate_parser.add_argument('strategy_ids', nargs='+', help='Strategy IDs to deactivate')

    screen_parser = subparsers.add_parser('screen', help='Screen stocks using active strategies')
    screen_parser.add_argument('--stocks', nargs='*', help='Stock codes to screen (default: all)')
    screen_parser.add_argument('--top', type=int, default=5, help='Number of top candidates to return (default: 5)')
    screen_parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='ERROR',
        help='Set logging level for screening (default: ERROR to reduce output)'
    )
    screen_parser.add_argument(
        '--update-data',
        action='store_true',
        help='Update stock data before screening (default: False)'
    )
    screen_parser.add_argument(
        '--update-days',
        type=int,
        default=365,
        help='Number of days to check and update data for (default: 365, smart incremental update)'
    )

    args = parser.parse_args()

    # Determine effective log level
    effective_log_level = args.log_level
    
    # Set logging level
    logging.basicConfig(
        level=getattr(logging, effective_log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Also set root logger level
    logging.getLogger().setLevel(getattr(logging, effective_log_level))

    try:
        # Initialize data fetcher manager
        from data_provider import DataFetcherManager
        data_fetcher_manager = DataFetcherManager()
        
        service = StockSelectorService()
        service.set_data_provider(data_fetcher_manager)
        logger.info("Using config: default_top_n=%d", service.config.default_top_n)
    except Exception as e:
        logger.error("Failed to initialize stock selector service: %s", e)
        return 1

    if args.command == 'list':
        list_strategies(service)
    elif args.command == 'activate':
        activate_strategy(service, args.strategy_ids)
    elif args.command == 'deactivate':
        deactivate_strategy(service, args.strategy_ids)
    elif args.command == 'screen':
        screen_stocks(service, args.stocks, args.top, args.update_data, args.update_days)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
