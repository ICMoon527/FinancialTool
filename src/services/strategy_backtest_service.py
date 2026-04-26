# -*- coding: utf-8 -*-
"""
策略回测服务 - 使用策略回测引擎进行回测。
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.strategy_backtest.orchestrator import BacktestOrchestrator
from src.core.strategy_backtest.task_manager import (
    get_task_manager,
    BacktestTaskStatus,
)
from stock_selector.manager import StrategyManager
from stock_selector.stock_pool import get_all_stock_codes, filter_special_stock_codes
from data_provider import DataFetcherManager

logger = logging.getLogger(__name__)


class StrategyBacktestService:
    """
    策略回测服务。
    """

    def __init__(self):
        """
        初始化策略回测服务。
        """
        self.data_provider = DataFetcherManager()
        self.strategy_manager = None
        self._current_orchestrator = None
        self._init_strategy_manager()

    def stop_backtest(self) -> None:
        """
        停止当前运行的回测。
        """
        if self._current_orchestrator:
            logger.info("服务层收到停止回测信号")
            self._current_orchestrator.stop()

    def _init_strategy_manager(self):
        """
        初始化策略管理器。
        """
        try:
            from pathlib import Path
            import sys
            from stock_selector.config import StockSelectorConfig

            # 获取策略目录
            project_root = Path(__file__).parent.parent.parent
            strategies_dir = project_root / "stock_selector" / "strategies"

            config = StockSelectorConfig(
                auto_activate_all=True,
                default_active_strategies=[],
                excluded_strategies=[],
                preferred_strategy_type=None,
                strategy_multipliers={},
            )

            self.strategy_manager = StrategyManager(
                nl_strategy_dir=strategies_dir,
                python_strategy_dir=strategies_dir,
                data_provider=self.data_provider,
                config=config,
            )

            logger.info("策略管理器初始化成功")
        except Exception as e:
            logger.error(f"策略管理器初始化失败: {e}", exc_info=True)

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        获取所有可用策略列表。

        Returns:
            策略信息列表
        """
        if not self.strategy_manager:
            return []

        strategies = self.strategy_manager.get_all_strategies()
        return [
            {
                "id": strategy.id,
                "name": strategy.display_name,
                "description": getattr(strategy.metadata, "description", ""),
                "type": strategy.metadata.strategy_type.name
                if hasattr(strategy.metadata, "strategy_type")
                else "UNKNOWN",
            }
            for strategy in strategies
        ]

    def get_stock_pool(self, use_filter: bool = True) -> List[str]:
        """
        获取股票池（沪A+深A）。

        Args:
            use_filter: 是否使用过滤器过滤特殊股票

        Returns:
            股票代码列表
        """
        try:
            # 使用stock_pool.py中的完整股票池（沪A+深A）
            stock_codes = get_all_stock_codes()
            
            if use_filter:
                # 过滤特殊股票（科创板、创业板、北交所等）
                stock_codes = filter_special_stock_codes(stock_codes)
            
            logger.info(f"获取沪A+深A股票池，共 {len(stock_codes)} 只股票")
            return stock_codes
        except Exception as e:
            logger.error(f"获取股票池失败: {e}", exc_info=True)
            return []

    def run_backtest(
        self,
        strategy_id: Optional[str] = None,
        strategy_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        stock_pool: Optional[List[str]] = None,
        max_positions: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行策略回测。

        Args:
            strategy_id: 策略ID（已弃用，向后兼容）
            strategy_ids: 策略ID列表（多策略支持）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_pool: 股票池，如果为None则使用默认股票池
            config_path: 配置文件路径

        Returns:
            回测结果字典
        """
        if not self.strategy_manager:
            raise ValueError("策略管理器未初始化")

        # 确定要使用的策略列表
        strategies_to_use = []
        
        # 优先使用 strategy_ids
        if strategy_ids:
            for sid in strategy_ids:
                strategy = self.strategy_manager.get_strategy(sid)
                if strategy:
                    strategies_to_use.append(strategy)
                else:
                    raise ValueError(f"策略不存在: {sid}")
        # 其次使用单个 strategy_id（向后兼容）
        elif strategy_id:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy:
                raise ValueError(f"策略不存在: {strategy_id}")
            strategies_to_use.append(strategy)
        # 都没有提供则报错
        else:
            raise ValueError("必须提供 strategy_id 或 strategy_ids")

        # 解析日期
        start_date_obj = None
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            # 默认一年前
            start_date_obj = date.today() - timedelta(days=365)

        end_date_obj = None
        if end_date:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end_date_obj = date.today()

        # 获取股票池
        if stock_pool is None:
            stock_pool = self.get_stock_pool()
            logger.info(f"使用默认股票池，共 {len(stock_pool)} 只股票")

        # 设置配置路径
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "stock_selector" / "backtest_config.yaml")

        # 创建回测编排器
        orchestrator = BacktestOrchestrator(
            config_path=config_path,
            output_dir="strategy_backtest_results",
        )
        
        # 保存当前编排器引用
        self._current_orchestrator = orchestrator

        # 确定策略名称（用于报告）
        strategy_name = "多策略组合" if len(strategies_to_use) > 1 else strategies_to_use[0].display_name

        # 运行完整回测
        try:
            result = orchestrator.run_full_backtest(
                data_provider=self.data_provider,
                strategy=strategies_to_use[0] if len(strategies_to_use) == 1 else None,  # 单个策略保持向后兼容
                strategies=strategies_to_use,  # 多策略支持
                stock_pool=stock_pool,
                start_date=start_date_obj,
                end_date=end_date_obj,
                max_positions=max_positions,
                strategy_name=strategy_name,
            )
            return result
        except Exception as e:
            logger.error(f"回测执行失败: {e}", exc_info=True)
            raise
        finally:
            # 清除当前编排器引用
            self._current_orchestrator = None
    
    def _run_backtest_in_background(
        self,
        task_id: str,
        strategy: Optional[Any],
        strategies: Optional[List[Any]],
        stock_pool: List[str],
        start_date_obj: date,
        end_date_obj: date,
        max_positions: int,
        on_complete: Optional[callable] = None,
        on_error: Optional[callable] = None,
    ) -> None:
        """
        在后台线程中运行回测
        
        Args:
            task_id: 任务ID
            strategy: 单个策略对象（已弃用，向后兼容）
            strategies: 策略对象列表（多策略支持）
            stock_pool: 股票池
            start_date_obj: 开始日期
            end_date_obj: 结束日期
            max_positions: 最大持仓数
            on_complete: 完成回调
            on_error: 错误回调
        """
        task_manager = get_task_manager()
        
        # 确定要使用的策略列表
        strategies_to_use = []
        if strategies:
            strategies_to_use = strategies
        elif strategy:
            strategies_to_use = [strategy]
        
        # 确定策略名称
        strategy_name = "多策略组合" if len(strategies_to_use) > 1 else strategies_to_use[0].display_name
        
        try:
            logger.info(f"后台回测任务开始: {task_id}")
            
            # 更新任务状态为运行中
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.RUNNING
            )
            
            # 创建编排器 - 使用与stock_selector相同的配置文件
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "stock_selector" / "backtest_config.yaml")
            
            # 创建编排器 - 使用配置文件
            orchestrator = BacktestOrchestrator(
                config_path=config_path,
                output_dir="strategy_backtest_results"
            )
            
            # 更新任务状态，保存orchestrator引用用于终止
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.RUNNING,
                orchestrator=orchestrator
            )
            
            # 运行完整回测
            result = orchestrator.run_full_backtest(
                data_provider=self.data_provider,
                strategy=strategies_to_use[0] if len(strategies_to_use) == 1 else None,
                strategies=strategies_to_use,
                stock_pool=stock_pool,
                start_date=start_date_obj,
                end_date=end_date_obj,
                max_positions=max_positions,
                strategy_name=strategy_name,
            )
            
            # 更新任务状态为完成
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.COMPLETED,
                result=result
            )
            
            logger.info(f"后台回测任务完成: {task_id}")
            
            # 调用完成回调
            if on_complete:
                try:
                    on_complete()
                except Exception as e:
                    logger.error(f"完成回调执行失败: {e}")
            
        except Exception as e:
            logger.error(f"后台回测任务失败: {task_id}, 错误: {e}", exc_info=True)
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.FAILED,
                error=str(e)
            )
            
            # 调用错误回调
            if on_error:
                try:
                    on_error()
                except Exception as e_call:
                    logger.error(f"错误回调执行失败: {e_call}")
    
    def run_backtest_async(
        self,
        strategy_id: Optional[str] = None,
        strategy_ids: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        max_positions: int = 3,
        on_complete: Optional[callable] = None,
        on_error: Optional[callable] = None,
    ) -> str:
        """
        异步运行策略回测，立即返回task_id
        
        Args:
            strategy_id: 策略ID（已弃用，向后兼容）
            strategy_ids: 策略ID列表（多策略支持）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_positions: 最大持仓数
            on_complete: 完成回调
            on_error: 错误回调
            
        Returns:
            task_id
        """
        # 创建任务
        task_manager = get_task_manager()
        task_id = task_manager.create_task()
        
        # 解析日期
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"日期格式错误: {e}")
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.FAILED,
                error=f"日期格式错误: {e}"
            )
            if on_error:
                on_error()
            return task_id
        
        # 确定要使用的策略列表
        strategies_to_use = []
        
        # 优先使用 strategy_ids
        if strategy_ids:
            for sid in strategy_ids:
                strategy = self.strategy_manager.get_strategy(sid)
                if strategy:
                    strategies_to_use.append(strategy)
                else:
                    error_msg = f"策略不存在: {sid}"
                    logger.error(error_msg)
                    task_manager.update_task_status(
                        task_id,
                        BacktestTaskStatus.FAILED,
                        error=error_msg
                    )
                    if on_error:
                        on_error()
                    return task_id
        # 其次使用单个 strategy_id（向后兼容）
        elif strategy_id:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if not strategy:
                error_msg = f"策略不存在: {strategy_id}"
                logger.error(error_msg)
                task_manager.update_task_status(
                    task_id,
                    BacktestTaskStatus.FAILED,
                    error=error_msg
                )
                if on_error:
                    on_error()
                return task_id
            strategies_to_use.append(strategy)
        # 都没有提供则报错
        else:
            error_msg = "必须提供 strategy_id 或 strategy_ids"
            logger.error(error_msg)
            task_manager.update_task_status(
                task_id,
                BacktestTaskStatus.FAILED,
                error=error_msg
            )
            if on_error:
                on_error()
            return task_id
        
        # 获取股票池
        stock_pool = self.get_stock_pool()
        
        # 在后台线程中运行回测
        thread = threading.Thread(
            target=self._run_backtest_in_background,
            args=(
                task_id,
                strategies_to_use[0] if len(strategies_to_use) == 1 else None,
                strategies_to_use,
                stock_pool,
                start_date_obj,
                end_date_obj,
                max_positions,
                on_complete,
                on_error,
            ),
            daemon=True
        )
        thread.start()
        
        logger.info(f"已启动异步回测任务: {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典或None
        """
        task_manager = get_task_manager()
        
        # 清理过期任务
        task_manager.cleanup_expired_tasks()
        
        task = task_manager.get_task(task_id)
        if not task:
            return None
        
        return task.to_dict()
    
    def stop_backtest_by_task_id(self, task_id: str) -> bool:
        """
        通过task_id停止回测
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功停止
        """
        task_manager = get_task_manager()
        return task_manager.stop_task(task_id)
