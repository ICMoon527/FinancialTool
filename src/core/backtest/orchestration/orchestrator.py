
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

from ..core.engine import StrategyBacktestEngine, Portfolio
from ..metrics.performance import PerformanceMetrics
from ..visualization.core import BacktestVisualizer
from ..reporting.generator import BacktestReportGenerator
from .preloader import SmartDataPreloader

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
        self._should_stop = False
        self._preloader: Optional[SmartDataPreloader] = None
        self._actual_start_date: Optional[date] = None
        self._warmup_days = 365

    def stop(self):
        """
        停止回测。
        """
        logger.info("编排器收到停止回测信号")
        self._should_stop = True
        if self.engine:
            self.engine.stop()
        if self._preloader:
            self._preloader.stop()

    def _load_config(self, config_path: Optional[str]):
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
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "max_positions": None,
            "max_holding_days": 5,
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f)
                default_config.update(file_config)
                logger.info("配置已加载: %s", config_path)
                logger.info("手续费率: %.4f (%.2f%%)", 
                          default_config.get("commission_rate"), 
                          default_config.get("commission_rate", 0) * 100)
                logger.info("滑点率: %.4f (%.2f%%)", 
                          default_config.get("slippage_rate"), 
                          default_config.get("slippage_rate", 0) * 100)
                logger.info("最长持股时间: %d 个交易日", 
                          default_config.get("max_holding_days", 5))
            except Exception as e:
                logger.warning("加载配置文件失败: %s", e)

        return default_config

    def _preload_data_with_batch(
        self,
        data_provider: Any,
        stock_pool: List[str],
        start_date: date,
        end_date: date,
    ):
        """
        使用智能数据预加载器预加载数据。
        检查数据库中已有数据，只下载缺失的部分。

        Args:
            data_provider: 数据提供器
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
        """
        if self._should_stop:
            logger.info("回测已被终止，跳过数据预加载")
            return
            
        try:
            logger.info("使用智能数据预加载器...")
            
            from src.storage import DatabaseManager
            from stock_selector.tushare_data_downloader import get_tushare_downloader
            
            db_manager = DatabaseManager.get_instance()
            tushare_downloader = get_tushare_downloader()
            
            self._preloader = SmartDataPreloader(
                db_manager=db_manager,
                tushare_downloader=tushare_downloader
            )
            
            if self._should_stop:
                logger.info("回测已被终止，跳过数据预加载")
                return
            
            try:
                self._preloader.ensure_data_available(
                    stock_codes=stock_pool,
                    target_start_date=start_date,
                    target_end_date=end_date
                )
            except KeyboardInterrupt:
                logger.info("智能数据预加载被用户中断 (Ctrl+C)")
                self._should_stop = True
            
        except Exception as e:
            logger.warning("智能数据预加载失败: %s, 将使用默认数据获取方式", e, exc_info=True)
        finally:
            self._preloader = None

    def prepare_data(
        self,
        data_provider: Any,
        stock_pool: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ):
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

        self._actual_start_date = start_date
        
        warmup_start_date = None
        if start_date:
            warmup_start_date = start_date - timedelta(days=self._warmup_days)
            logger.info("配置回测起始日期: %s", start_date)
            logger.info("预热期起始日期: %s", warmup_start_date)
        
        effective_start_date = warmup_start_date if warmup_start_date else start_date
        
        self.config["stock_pool"] = stock_pool
        self.config["start_date"] = start_date
        self.config["end_date"] = end_date

        logger.info("股票池: %s", stock_pool)
        logger.info("回测期间: %s 至 %s", start_date, end_date)
        logger.info("数据预加载范围: %s 至 %s", warmup_start_date, end_date)

        if stock_pool and start_date and end_date:
            self._preload_data_with_batch(data_provider, stock_pool, start_date, end_date)

        trading_dates = self._generate_trading_dates(start_date, end_date)
        logger.info("交易日数: %d", len(trading_dates))

        return trading_dates

    def _generate_trading_dates(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
    ):
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

        # 先尝试使用交易日历
        try:
            from stock_selector.trading_calendar import get_trading_days
            
            dates = get_trading_days(start_date, end_date)
            
            total_days = (end_date - start_date).days + 1
            skipped_days = total_days - len(dates)
            
            if skipped_days > 0:
                logger.info("跳过 %d 个非交易日", skipped_days)
            
            return dates
            
        except Exception as e:
            logger.warning("使用交易日历失败，回退到周末判断: %s", e)
        
        # 降级方案：使用周末判断
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date = current_date + timedelta(days=1)
        
        return dates

    def execute_strategy(
        self,
        data_provider: Any,
        strategy: Any,
        trading_dates: List[date],
        stock_pool: Optional[List[str]] = None,
        max_positions: Optional[int] = None,
    ):
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

        final_max_positions = max_positions
        if final_max_positions is None:
            final_max_positions = self.config.get("max_positions")
        
        from .data_access import DatabaseFirstDataProvider
        from src.storage import DatabaseManager
        db_manager = DatabaseManager.get_instance()
        db_first_provider = DatabaseFirstDataProvider(data_provider, db_manager)
        
        self.engine = StrategyBacktestEngine(
            data_provider=db_first_provider,
            initial_capital=self.config.get("initial_capital", 1000000.0),
            commission_rate=self.config.get("commission_rate", 0.0003),
            slippage_rate=self.config.get("slippage_rate", 0.001),
            start_date=self.config.get("start_date"),
            end_date=self.config.get("end_date"),
            stop_loss_pct=self.config.get("stop_loss_pct"),
            take_profit_pct=self.config.get("take_profit_pct"),
            max_positions=final_max_positions,
            max_holding_days=self.config.get("max_holding_days", 5),
        )

        self.engine.set_strategy(strategy)
        self.engine.set_stock_pool(stock_pool)
        self.engine.set_trading_dates(trading_dates)

        self.portfolio = self.engine.run()

        return self.portfolio

    def record_results(self):
        """
        阶段3: 结果记录。

        Returns:
            结果字典
        """
        logger.info("=== 阶段3: 结果记录 ===")

        if self.portfolio is None:
            raise ValueError("投资组合未初始化，请先执行策略")

        actual_start_date = self._actual_start_date
        if actual_start_date:
            logger.info("只记录 %s 之后的交易和权益（预热期数据被过滤）", actual_start_date)

        filtered_trades = []
        for t in self.portfolio.trades:
            if actual_start_date is None or (t.date and t.date >= actual_start_date):
                filtered_trades.append(t)

        filtered_equity_history = []
        for d, e in self.portfolio.equity_history:
            if actual_start_date is None or d >= actual_start_date:
                filtered_equity_history.append((d, e))

        results = {
            "initial_capital": self.portfolio.initial_capital,
            "final_equity": self.portfolio.get_total_equity(),
            "total_trades": len(filtered_trades),
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
                for t in filtered_trades
            ],
            "equity_history": [
                {"date": d.isoformat(), "equity": e}
                for d, e in filtered_equity_history
            ],
        }

        results_file = self.output_dir / "backtest_results.json"
        import json
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("结果已保存至: %s", results_file)

        return results

    def _get_benchmark_returns(self, data_provider: Any):
        """
        获取上证指数收益率作为基准。

        Args:
            data_provider: 数据提供器

        Returns:
            基准日收益率列表，与投资组合收益率序列对应
        """
        if not self.portfolio or not self.portfolio.equity_history:
            return None

        try:
            actual_start_date = self._actual_start_date
            filtered_equity_history = []
            for d, e in self.portfolio.equity_history:
                if actual_start_date is None or d >= actual_start_date:
                    filtered_equity_history.append((d, e))
            
            dates = [date for date, _ in filtered_equity_history]
            if len(dates) < 2:
                return None

            start_date_str = dates[0].strftime("%Y-%m-%d")
            end_date_str = dates[-1].strftime("%Y-%m-%d")

            logger.info("获取上证指数数据: %s 至 %s", start_date_str, end_date_str)

            index_df = data_provider.get_index_daily_data(
                symbol="sh000001",
                start_date=start_date_str,
                end_date=end_date_str,
            )

            if index_df is None or index_df.empty:
                logger.warning("无法获取上证指数数据，基准收益率将不可用")
                return None

            if "date" in index_df.columns:
                index_df["date"] = pd.to_datetime(index_df["date"]).dt.date

            if "close" in index_df.columns:
                index_df["return"] = index_df["close"].pct_change()

                date_to_return = dict(zip(index_df["date"], index_df["return"]))

                benchmark_returns = []
                for i, current_date in enumerate(dates):
                    if i == 0:
                        continue
                    if current_date in date_to_return and pd.notna(date_to_return[current_date]):
                        benchmark_returns.append(float(date_to_return[current_date]))
                    else:
                        benchmark_returns.append(0.0)

                logger.info("成功获取基准收益率，共 %d 个数据点", len(benchmark_returns))
                return benchmark_returns

        except Exception as e:
            logger.warning("获取基准收益率失败: %s", e, exc_info=True)

        return None

    def calculate_metrics(self, data_provider: Optional[Any] = None):
        """
        阶段4: 指标计算。

        Args:
            data_provider: 数据提供器（用于获取基准指数数据）

        Returns:
            指标字典
        """
        logger.info("=== 阶段4: 指标计算 ===")

        if self.portfolio is None:
            raise ValueError("投资组合未初始化，请先执行策略")

        benchmark_returns = None
        if data_provider is not None:
            benchmark_returns = self._get_benchmark_returns(data_provider)

        self.metrics = PerformanceMetrics(
            portfolio=self.portfolio,
            risk_free_rate=self.config.get("risk_free_rate", 0.03),
            benchmark_returns=benchmark_returns,
        )

        all_metrics = self.metrics.get_all_metrics()

        logger.info("绩效指标计算完成:")
        for key, value in all_metrics.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", key, value)
            else:
                logger.info("  %s: %s", key, value)

        return all_metrics

    def generate_reports(self, strategy_name: str = "六维选股策略"):
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
            "初始资金": "%.2f 元" % self.config.get("initial_capital", 0),
            "手续费率": "%.4f%%" % (self.config.get("commission_rate", 0) * 100),
            "滑点率": "%.4f%%" % (self.config.get("slippage_rate", 0) * 100),
            "无风险利率": "%.2f%%" % (self.config.get("risk_free_rate", 0) * 100),
            "回测开始日期": str(self.config.get("start_date")),
            "回测结束日期": str(self.config.get("end_date")),
            "股票池数量": len(self.config.get("stock_pool", [])),
        }

        report_path = report_generator.generate_markdown_report(
            backtest_config=backtest_config_for_report,
            chart_paths=self.chart_paths,
        )

        logger.info("报告已生成: %s", report_path)

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
        max_positions: Optional[int] = None,
        strategy_name: str = "六维选股策略",
    ):
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
            max_positions=max_positions,
        )

        results = self.record_results()
        metrics = self.calculate_metrics(data_provider=data_provider)
        reports = self.generate_reports(strategy_name=strategy_name)

        logger.info("=" * 50)
        logger.info("完整回测流程完成")
        logger.info("=" * 50)

        return {
            "results": results,
            "metrics": metrics,
            "reports": reports,
        }

