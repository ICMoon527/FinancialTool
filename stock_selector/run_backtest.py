# -*- coding: utf-8 -*-
"""
六维选股策略回测运行脚本。

使用方法:
    python run_backtest.py --config backtest_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="六维选股策略回测")
    parser.add_argument(
        "--config",
        type=str,
        default="backtest_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results",
        help="输出目录",
    )
    args = parser.parse_args()

    try:
        from data_provider.base import DataFetcherManager
        from src.core.strategy_backtest import BacktestOrchestrator
        from stock_selector.strategies.Python.six_dimension_selector import (
            SixDimensionSelectorStrategy,
        )
        from stock_selector.stock_pool import (
            get_all_stock_code_name_pairs,
            filter_special_stock_codes,
            filter_st_stocks,
        )

        logger.info("初始化数据提供器...")
        data_manager = DataFetcherManager()
        data_provider = data_manager.get_default_fetcher()

        logger.info("初始化策略...")
        strategy = SixDimensionSelectorStrategy()

        logger.info("初始化回测编排器...")
        orchestrator = BacktestOrchestrator(
            config_path=args.config,
            output_dir=args.output,
        )

        logger.info("运行完整回测...")
        config = orchestrator.config
        stock_pool = config.get("stock_pool", [])

        if not stock_pool:
            logger.info("股票池为空，自动从stock_pool获取股票列表...")
            all_pairs = get_all_stock_code_name_pairs()
            logger.info(f"从数据库获取到 {len(all_pairs)} 只股票")
            
            filtered_codes = filter_special_stock_codes([code for code, name in all_pairs])
            logger.info(f"过滤特定板块后剩余 {len(filtered_codes)} 只股票")
            
            filtered_pairs = [(code, name) for code, name in all_pairs if code in filtered_codes]
            filtered_pairs = filter_st_stocks(filtered_pairs)
            
            stock_pool = [code for code, name in filtered_pairs]
            logger.info(f"最终回测股票池包含 {len(stock_pool)} 只股票")

        start_date = config.get("start_date")
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        end_date = config.get("end_date")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        results = orchestrator.run_full_backtest(
            data_provider=data_provider,
            strategy=strategy,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
            strategy_name="六维选股策略",
        )

        logger.info("=" * 50)
        logger.info("回测完成！")
        logger.info(f"结果目录: {args.output}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"回测失败: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
