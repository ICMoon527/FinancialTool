#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selector Command Line Interface
负责选股策略
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


def screen_stocks(service: StockSelectorService, stock_codes: Optional[List[str]], top_n: int, update_data: bool = False, update_days: int = 365, use_enhanced: bool = True, max_workers: int = 4, validate_data: bool = False, report_path: Optional[str] = None, use_tushare: bool = True, rate_limit: int = 50):
    print("Screening stocks...")
    print(f"Top N: {top_n}")
    if stock_codes:
        print(f"Stock codes: {', '.join(stock_codes)}")
    else:
        print("Stock codes: All available")
    
    # 如果需要先更新数据
    if update_data:
        from datetime import date, timedelta
        from stock_selector.stock_pool import get_all_stock_codes
        
        print(f"\nUpdating stock data (last {update_days} days) using Tushare...")
        
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
        
        # 使用 Tushare 专用下载器强制更新数据
        if use_tushare:
            try:
                from stock_selector.tushare_data_downloader import get_tushare_downloader
                
                print(f"Using Tushare data downloader (rate limit: {rate_limit} calls/min)\n")
                
                downloader = get_tushare_downloader(rate_limit_per_minute=rate_limit)
                stats = downloader.download_data(
                    stock_codes=stock_codes,
                    days=update_days,
                    batch_size=10
                )
                
                print(f"\nData update complete using Tushare!\n")
                
            except Exception as e:
                print(f"\nTushare downloader failed: {e}")
                print("Falling back to legacy updater...\n")
                use_tushare = False
        
        # 如果 Tushare 不可用，回退到旧版更新器
        if not use_tushare:
            from stock_selector.data_update_tracker import get_update_tracker
            from stock_selector.batch_data_updater import get_batch_updater
            
            end_date = date.today()
            target_start_date = end_date - timedelta(days=update_days - 1)
            
            # --update-data 时强制更新所有数据，不检查已有数据
            actual_start_date = target_start_date
            print(f"强制更新全部 {update_days} 天数据")
            print(f"日期范围：{actual_start_date} 至 {end_date}")
            
            if actual_start_date <= end_date:
                batch_updater = get_batch_updater()
                stats = batch_updater.update_stocks_for_date_range(
                    stock_codes=stock_codes,
                    start_date=actual_start_date,
                    end_date=end_date
                )
                
                print(f"Data update complete: {stats['stocks_updated']} updated, {stats['stocks_failed']} failed\n")
            else:
                print("Invalid date range!\n")

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

  # 推荐用法：智能更新（默认检查365天）
  python -m stock_selector.main screen --update-data

  # Screen specific stocks
  python -m stock_selector.main screen --stocks 600519 000001

  # Screen with custom top N
  python -m stock_selector.main screen --top 10

  # Screen with custom log level (ERROR/WARNING/INFO/DEBUG)
  python -m stock_selector.main screen --log-level ERROR

  # Screen with custom check window (e.g., only check last 7 days)
  python -m stock_selector.main screen --update-data --update-days 7

  # 不使用 Tushare，回退到旧版
  python -m stock_selector.main screen --update-data --no-tushare

  # 自定义 Tushare 速率限制
  python -m stock_selector.main screen --update-data --rate-limit 50
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
    screen_parser.add_argument(
        '--use-enhanced',
        action='store_true',
        default=True,
        help='Use enhanced batch updater with concurrency (default: True)'
    )
    screen_parser.add_argument(
        '--no-enhanced',
        action='store_false',
        dest='use_enhanced',
        help='Use legacy batch updater instead of enhanced'
    )
    screen_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of concurrent workers for enhanced updater (default: 4)'
    )
    screen_parser.add_argument(
        '--validate-data',
        action='store_true',
        help='Validate data integrity after update (default: False)'
    )
    screen_parser.add_argument(
        '--report-path',
        type=str,
        default=None,
        help='Path to save detailed update report (optional)'
    )
    screen_parser.add_argument(
        '--use-tushare',
        action='store_true',
        default=True,
        help='Use Tushare dedicated downloader (default: True)'
    )
    screen_parser.add_argument(
        '--no-tushare',
        action='store_false',
        dest='use_tushare',
        help='Use legacy updater instead of Tushare'
    )
    screen_parser.add_argument(
        '--rate-limit',
        type=int,
        default=50,
        help='Tushare API calls per minute (default: 50)'
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
        screen_stocks(service, args.stocks, args.top, args.update_data, args.update_days, args.use_enhanced, args.max_workers, args.validate_data, args.report_path, args.use_tushare, args.rate_limit)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
