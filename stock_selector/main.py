#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selector Command Line Interface

Command line tool to interact with the stock selector system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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


def screen_stocks(service: StockSelectorService, stock_codes: Optional[List[str]], top_n: int):
    print("Screening stocks...")
    print(f"Top N: {top_n}")
    if stock_codes:
        print(f"Stock codes: {', '.join(stock_codes)}")
    else:
        print("Stock codes: All available")

    candidates = service.screen_stocks(
        stock_codes=stock_codes,
        top_n=top_n
    )

    print("\nTop Candidates:")
    print("=" * 100)

    for i, candidate in enumerate(candidates, 1):
        print(f"\nRank {i}: {candidate.code} - {candidate.name}")
        print(f"Score: {candidate.match_score:.2f}")
        print(f"Price: {candidate.current_price:.2f}")

        print("\nStrategy Matches:")
        for match in candidate.strategy_matches:
            status = "✓" if match.matched else "✗"
            print(f"  {status} {match.strategy_name}: {match.score:.2f} - {match.reason}")

    print("\n" + "=" * 100)
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

  # Deactivate strategies
  python -m stock_selector.main deactivate volume_breakout

  # Screen stocks (top 5)
  python -m stock_selector.main screen

  # Screen specific stocks
  python -m stock_selector.main screen --stocks 600519 000001

  # Screen with custom top N
  python -m stock_selector.main screen --top 10
        """)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('list', help='List all available strategies')

    activate_parser = subparsers.add_parser('activate', help='Activate strategies')
    activate_parser.add_argument('strategy_ids', nargs='+', help='Strategy IDs to activate')

    deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate strategies')
    deactivate_parser.add_argument('strategy_ids', nargs='+', help='Strategy IDs to deactivate')

    screen_parser = subparsers.add_parser('screen', help='Screen stocks using active strategies')
    screen_parser.add_argument('--stocks', nargs='*', help='Stock codes to screen (default: all)')
    screen_parser.add_argument('--top', type=int, default=5, help='Number of top candidates to return (default: 5)')

    args = parser.parse_args()

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
        screen_stocks(service, args.stocks, args.top)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
