# -*- coding: utf-8 -*-
"""
回测报告输出脚本。
用法：修改 stock_selector/backtest_config.yaml 中 exit_strategy.active 为所需配置，
      然后运行本脚本即可输出指定格式报告。
"""

import logging
import sys
from datetime import datetime, date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.strategy_backtest.orchestrator import BacktestOrchestrator
from data_provider import DataFetcherManager
from stock_selector.manager import StrategyManager
from stock_selector.config import StockSelectorConfig
from stock_selector.stock_pool import get_all_stock_codes, filter_special_stock_codes


def main():
    # 读取 YAML 获取当前 active 配置名
    import yaml
    config_path = project_root / "stock_selector" / "backtest_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    active_name = cfg["exit_strategy"]["active"]
    preset = cfg["exit_strategy"]["presets"].get(active_name, {})
    logger.info("当前 active 配置: %s (%s)", active_name, preset.get("name", ""))

    # 初始化
    data_provider = DataFetcherManager()
    strategies_dir = project_root / "stock_selector" / "strategies"
    scfg = StockSelectorConfig(auto_activate_all=True, default_active_strategies=[], excluded_strategies=[], preferred_strategy_type=None, strategy_multipliers={})
    strategy_manager = StrategyManager(nl_strategy_dir=strategies_dir, python_strategy_dir=strategies_dir, data_provider=data_provider, config=scfg)

    strategy = strategy_manager.get_strategy("tiandao_xg_buy")
    if not strategy:
        logger.error("未找到 tiandao_xg_buy 策略")
        return

    stock_codes = get_all_stock_codes()
    stock_codes = filter_special_stock_codes(stock_codes)
    logger.info("股票池大小: %d", len(stock_codes))

    # 创建编排器
    orchestrator = BacktestOrchestrator(config_path=str(config_path), output_dir="strategy_backtest_results")

    # 运行完整回测
    result = orchestrator.run_full_backtest(
        data_provider=data_provider,
        strategy=strategy,
        stock_pool=stock_codes,
        start_date=date(2021, 7, 31),
        end_date=date(2026, 7, 31),
        max_positions=1,
        strategy_name=f"天道超跌反弹({active_name})",
    )

    # 提取指标
    metrics = orchestrator.metrics
    all_metrics = metrics.get_all_metrics() if metrics else {}

    total_return = all_metrics.get("total_return", 0)
    max_dd = all_metrics.get("max_drawdown", 0)
    cum_dd_ratio = all_metrics.get("cum_max_dd_ratio", 0.0)

    # 从 results.json 读取 equity_history（含 position_ratio）
    import json
    results_json_path = orchestrator.output_dir / "results.json"
    if results_json_path.exists():
        with open(results_json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        equity_history = results_data.get("results", {}).get("equity_history", [])
    else:
        equity_history = []

    # 查找指定日期的仓位比例
    def find_pos_ratio(d: date) -> str:
        for entry in equity_history:
            if entry["date"] == d.isoformat():
                ratio = entry.get("position_ratio", 0.0)
                return f"{ratio*100:.1f}%"
        return "N/A"

    pos_0924 = find_pos_ratio(date(2024, 9, 24))
    pos_1011 = find_pos_ratio(date(2024, 10, 11))

    # 计算 2024Q4 收益率
    ret_24q4 = 0.0
    q4_start_val = None
    q4_end_val = None
    for entry in equity_history:
        ed = date.fromisoformat(entry["date"])
        if ed >= date(2024, 10, 1) and q4_start_val is None:
            q4_start_val = entry["equity"]
        if ed <= date(2024, 12, 31):
            q4_end_val = entry["equity"]
    if q4_start_val and q4_end_val:
        ret_24q4 = (q4_end_val - q4_start_val) / q4_start_val

    # 获取盈利保护触发统计
    exit_strategy = orchestrator.engine.exit_strategy if hasattr(orchestrator.engine, 'exit_strategy') else None
    pp_count = 0
    pp_avg_gain = 0.0
    if exit_strategy and hasattr(exit_strategy, 'profit_protect_trigger_count'):
        pp_count = exit_strategy.profit_protect_trigger_count
        if exit_strategy.profit_protect_trigger_gains:
            pp_avg_gain = sum(exit_strategy.profit_protect_trigger_gains) / len(exit_strategy.profit_protect_trigger_gains)

    # 输出报告
    logger.info("=" * 60)
    logger.info("回测报告: %s", preset.get("name", active_name))
    logger.info("=" * 60)
    logger.info(
        "config=%s | cum=%.2f%% | maxDD=%.2f%% | calmar(cum/DD)=%.2f | "
        "pos_0924=%s | pos_1011=%s | ret_24Q4=%.2f%% | "
        "保护触发次数=%d | 触发时平均盈利=%.2f%%",
        active_name,
        total_return * 100, max_dd * 100, cum_dd_ratio,
        pos_0924, pos_1011, ret_24q4 * 100,
        pp_count, pp_avg_gain * 100,
    )


if __name__ == "__main__":
    main()