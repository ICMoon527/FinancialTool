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
from .smart_data_preloader import SmartDataPreloader

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

    def stop(self) -> None:
        """
        停止回测。
        """
        logger.info("编排器收到停止回测信号")
        self._should_stop = True
        if self.engine:
            self.engine.stop()
        if self._preloader:
            self._preloader.stop()

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
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "max_positions": None,
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

    def _preload_data_with_batch(
        self,
        data_provider: Any,
        stock_pool: List[str],
        start_date: date,
        end_date: date,
    ) -> None:
        """
        使用智能数据预加载器预加载数据。
        检查数据库中已有数据，只下载缺失的部分。

        Args:
            data_provider: 数据提供器
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
        """
        # 检查终止标志
        if self._should_stop:
            logger.info("回测已被终止，跳过数据预加载")
            return
            
        try:
            logger.info("使用智能数据预加载器...")
            
            # 导入必要的模块
            from src.storage import DatabaseManager
            from stock_selector.tushare_data_downloader import get_tushare_downloader
            
            # 获取数据库管理器和Tushare下载器
            db_manager = DatabaseManager.get_instance()
            tushare_downloader = get_tushare_downloader()
            
            # 创建智能数据预加载器并保存引用
            self._preloader = SmartDataPreloader(
                db_manager=db_manager,
                tushare_downloader=tushare_downloader
            )
            
            # 再次检查终止标志
            if self._should_stop:
                logger.info("回测已被终止，跳过数据预加载")
                return
            
            # 确保数据可用
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
            logger.warning(f"智能数据预加载失败: {e}, 将使用默认数据获取方式", exc_info=True)
        finally:
            # 清除 preloader 引用
            self._preloader = None

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

        # 使用批量数据获取策略预加载数据
        if stock_pool and start_date and end_date:
            self._preload_data_with_batch(data_provider, stock_pool, start_date, end_date)

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
        max_positions: Optional[int] = None,
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

        # 使用传入的max_positions参数，如果没有则使用配置文件中的
        final_max_positions = max_positions
        if final_max_positions is None:
            final_max_positions = self.config.get("max_positions")
        
        # 使用数据库优先的数据提供器
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

    def _get_benchmark_returns(self, data_provider: Any) -> Optional[List[float]]:
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
            # 获取投资组合的日期序列
            dates = [date for date, _ in self.portfolio.equity_history]
            if len(dates) < 2:
                return None

            start_date = dates[0].strftime("%Y-%m-%d")
            end_date = dates[-1].strftime("%Y-%m-%d")

            logger.info(f"获取上证指数数据: {start_date} 至 {end_date}")

            # 获取上证指数数据 (sh000001)
            index_df = data_provider.get_index_daily_data(
                symbol="sh000001",
                start_date=start_date,
                end_date=end_date,
            )

            if index_df is None or index_df.empty:
                logger.warning("无法获取上证指数数据，基准收益率将不可用")
                return None

            # 确保日期列格式正确
            if "date" in index_df.columns:
                index_df["date"] = pd.to_datetime(index_df["date"]).dt.date

            # 计算指数日收益率
            if "close" in index_df.columns:
                index_df["return"] = index_df["close"].pct_change()

                # 创建日期到收益率的映射
                date_to_return = dict(zip(index_df["date"], index_df["return"]))

                # 按照投资组合的日期序列提取收益率
                benchmark_returns = []
                for i, date in enumerate(dates):
                    if i == 0:
                        # 第一天没有收益率
                        continue
                    if date in date_to_return and pd.notna(date_to_return[date]):
                        benchmark_returns.append(float(date_to_return[date]))
                    else:
                        # 如果该日期没有指数数据，使用0
                        benchmark_returns.append(0.0)

                logger.info(f"成功获取基准收益率，共 {len(benchmark_returns)} 个数据点")
                return benchmark_returns

        except Exception as e:
            logger.warning(f"获取基准收益率失败: {e}", exc_info=True)

        return None

    def calculate_metrics(self, data_provider: Optional[Any] = None) -> Dict[str, Any]:
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

        # 获取基准收益率（上证指数）
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
        max_positions: Optional[int] = None,
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
