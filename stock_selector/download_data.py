#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Data Downloader - 股票数据下载器

专门为 stock_selector 设计的数据下载工具，使用 Tushare 接口
"""

import sys
import logging
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import date, timedelta

from stock_selector.tushare_data_downloader import get_tushare_downloader
from stock_selector.stock_pool import get_all_stock_codes


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stock Data Downloader - 使用 Tushare 下载股票数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 下载所有股票最近 365 天数据（默认）
  python -m stock_selector.download_data

  # 下载指定股票最近 365 天数据
  python -m stock_selector.download_data --stocks 600519 000001

  # 下载最近 30 天数据
  python -m stock_selector.download_data --days 30

  # 自定义每批股票数量（默认 10）
  python -m stock_selector.download_data --batch-size 20

  # 下载前 100 只股票最近 7 天数据
  python -m stock_selector.download_data --days 7 --limit 100

  # 自定义速率限制（高级用户，默认 50 次/分钟）
  python -m stock_selector.download_data --rate-limit 50
        """
    )

    parser.add_argument(
        '--stocks',
        nargs='*',
        help='指定股票代码列表（默认：所有股票）'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='获取多少天的数据（默认：365）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='每批处理多少只股票（默认：10）'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='只下载前 N 只股票（用于测试）'
    )
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=50,
        help='Tushare 每分钟最大请求数（默认：50，Tushare 免费用户配额）'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='日志级别（默认：INFO）'
    )

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 获取股票列表
        if args.stocks:
            stock_codes = args.stocks
        else:
            stock_codes = get_all_stock_codes()

        # 限制股票数量（用于测试）
        if args.limit is not None:
            stock_codes = stock_codes[:args.limit]
            print(f"只下载前 {args.limit} 只股票")

        # 初始化下载器
        print("初始化 Tushare 数据下载器...")
        downloader = get_tushare_downloader(rate_limit_per_minute=args.rate_limit)

        # 开始下载
        stats = downloader.download_data(
            stock_codes=stock_codes,
            days=args.days,
            batch_size=args.batch_size
        )

        # 返回状态码
        if stats['stocks_failed'] == 0:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
