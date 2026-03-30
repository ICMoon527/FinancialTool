# -*- coding: utf-8 -*-
"""
回测流程编排器。

实现五阶段回测流程：数据准备、策略执行、结果记录、指标计算、报告生成。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .engine import StrategyBacktestEngine, Portfolio
from .metrics import PerformanceMetrics
from .visualization import BacktestVisualizer
from .report import BacktestReportGenerator

logger = logging.getLogger(__name__)


class BacktestOrchestrator:
    """
    回测流程编排器。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "backtest_results",
    ):
        """
        初始化回测编排器。

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_path)
        self.engine: Optional[StrategyBacktestEngine] = None
        self.portfolio: Optional[Portfolio] = None
        self.metrics: Optional[PerformanceMetrics] = None
        self.chart_paths: Dict[str, str] = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        加载配置文件。

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        default_config = {
            "initial_capital": 1000000.0,
            "commission_rate": 0.0003,
            "slippage_rate": 0.001,
            "risk_free_rate": 0.03,
            "start_date": None,
            "end_date": None,
            "stock_pool": [],
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f)
                default_config.update(file_config)
                logger.info(f"配置已加载: {config_path}")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}")

        return default_config

    def prepare_data(
        self,
        data_provider: Any,
        stock_pool: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[date]:
        """
        阶段1: 数据准备。

        Args:
            data_provider: 数据提供器
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        logger.info("=== 阶段1: 数据准备 ===")

        if stock_pool is None:
            stock_pool = self.config.get("stock_pool", [])

        if start_date is None:
            start_date = self.config.get("start_date")
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        if end_date is None:
            end_date = self.config.get("end_date")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        self.config["stock_pool"] = stock_pool
        self.config["start_date"] = start_date
        self.config["end_date"] = end_date

        logger.info(f"股票池: {stock_pool}")
        logger.info(f"回测期间: {start_date} 至 {end_date}")

        trading_dates = self._generate_trading_dates(start_date, end_date)
        logger.info(f"交易日数: {len(trading_dates)}")

        return trading_dates

    def _generate_trading_dates(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> List[date]:
        """
        生成交易日列表。

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)

        return dates

    def execute_strategy(
        self,
        data_provider: Any,
        strategy: Any,
        trading_dates: List[date],
        stock_pool: Optional[List[str]] = None,
    ) -> Portfolio:
        """
        阶段2: 策略执行。

        Args:
            data_provider: 数据提供器
            strategy: 策略对象
            trading_dates: 交易日列表
            stock_pool: 股票池

        Returns:
            投资组合
        """
        logger.info("=== 阶段2: 策略执行 ===")

        if stock_pool is None:
            stock_pool = self.config.get("stock_pool", [])

        self.engine = StrategyBacktestEngine(
            data_provider=data_provider,
            initial_capital=self.config.get("initial_capital", 1000000.0),
            commission_rate=self.config.get("commission_rate", 0.0003),
            slippage_rate=self.config.get("slippage_rate", 0.001),
            start_date=self.config.get("start_date"),
            end_date=self.config.get("end_date"),
        )

        self.engine.set_strategy(strategy)
        self.engine.set_stock_pool(stock_pool)
        self.engine.set_trading_dates(trading_dates)

        self.portfolio = self.engine.run()

        return self.portfolio

    def record_results(self) -> Dict[str, Any]:
        """
        阶段3: 结果记录。

        Returns:
            结果字典
        """
        logger.info("=== 阶段3: 结果记录 ===")

        if self.portfolio is None:
            raise ValueError("投资组合未初始化，请先执行策略")

        results = {
            "initial_capital": self.portfolio.initial_capital,
            "final_equity": self.portfolio.get_total_equity(),
            "total_trades": len(self.portfolio.trades),
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "stock_code": t.stock_code,
                    "order_type": t.order_type.value,
                    "quantity": t.quantity,
                    "price": t.price,
                    "date": t.date.isoformat() if t.date else None,
                    "commission": t.commission,
                    "slippage": t.slippage,
                }
                for t in self.portfolio.trades
            ],
            "equity_history": [
                {"date": d.isoformat(), "equity": e}
                for d, e in self.portfolio.equity_history
            ],
        }

        results_file = self.output_dir / "backtest_results.json"
        import json
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存至: {results_file}")

        return results

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        阶段4: 指标计算。

        Returns:
            指标字典
        """
        logger.info("=== 阶段4: 指标计算 ===")

        if self.portfolio is None:
            raise ValueError("投资组合未初始化，请先执行策略")

        self.metrics = PerformanceMetrics(
            portfolio=self.portfolio,
            risk_free_rate=self.config.get("risk_free_rate", 0.03),
        )

        all_metrics = self.metrics.get_all_metrics()

        logger.info("绩效指标计算完成:")
        for key, value in all_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        return all_metrics

    def generate_reports(self, strategy_name: str = "六维选股策略") -> Dict[str, str]:
        """
        阶段5: 报告生成。

        Args:
            strategy_name: 策略名称

        Returns:
            报告文件路径字典
        """
        logger.info("=== 阶段5: 报告生成 ===")

        if self.portfolio is None or self.metrics is None:
            raise ValueError("请先执行策略并计算指标")

        visualizer = BacktestVisualizer(
            portfolio=self.portfolio,
            metrics=self.metrics,
            output_dir=str(self.output_dir),
        )
        self.chart_paths = visualizer.plot_all()

        report_generator = BacktestReportGenerator(
            portfolio=self.portfolio,
            metrics=self.metrics,
            strategy_name=strategy_name,
            output_dir=str(self.output_dir),
        )

        backtest_config_for_report = {
            "初始资金": f"{self.config.get('initial_capital', 0):,.2f} 元",
            "手续费率": f"{self.config.get('commission_rate', 0)*100:.4f}%",
            "滑点率": f"{self.config.get('slippage_rate', 0)*100:.4f}%",
            "无风险利率": f"{self.config.get('risk_free_rate', 0)*100:.2f}%",
            "回测开始日期": str(self.config.get('start_date')),
            "回测结束日期": str(self.config.get('end_date')),
            "股票池数量": len(self.config.get('stock_pool', [])),
        }

        report_path = report_generator.generate_markdown_report(
            backtest_config=backtest_config_for_report,
            chart_paths=self.chart_paths,
        )

        logger.info(f"报告已生成: {report_path}")

        return {
            "report": report_path,
            **self.chart_paths,
        }

    def run_full_backtest(
        self,
        data_provider: Any,
        strategy: Any,
        stock_pool: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        strategy_name: str = "六维选股策略",
    ) -> Dict[str, Any]:
        """
        运行完整回测流程。

        Args:
            data_provider: 数据提供器
            strategy: 策略对象
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
            strategy_name: 策略名称

        Returns:
            回测结果字典
        """
        logger.info("=" * 50)
        logger.info("开始完整回测流程")
        logger.info("=" * 50)

        trading_dates = self.prepare_data(
            data_provider=data_provider,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
        )

        self.execute_strategy(
            data_provider=data_provider,
            strategy=strategy,
            trading_dates=trading_dates,
        )

        results = self.record_results()
        metrics = self.calculate_metrics()
        reports = self.generate_reports(strategy_name=strategy_name)

        logger.info("=" * 50)
        logger.info("完整回测流程完成")
        logger.info("=" * 50)

        return {
            "results": results,
            "metrics": metrics,
            "reports": reports,
        }
