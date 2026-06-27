# -*- coding: utf-8 -*-
"""RL 服务层：管理训练任务、模型存储、评估结果

参考 StrategyBacktestService 的异步任务模式。
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from rl.data.dataset import IntradayDataset
from rl.algorithms.dqn import DQNModel
from rl.training.trainer import RLTrainer
from rl.training.callbacks import ProgressCallback
from rl.evaluation.evaluator import RLEvaluator

if TYPE_CHECKING:
    from rl.config import RLConfig
    from rl.algorithms.base import AbstractRLModel
    from src.storage import DatabaseManager

logger = logging.getLogger(__name__)


class RLService:
    """RL 服务层"""

    def __init__(self, config: "RLConfig", db: "DatabaseManager"):
        self.config = config
        self.db = db
        self._tasks: Dict[str, Dict] = {}       # task_id → 任务状态
        self._models: Dict[str, Dict] = {}      # model_id → 模型元信息
        self._lock = threading.Lock()

    def start_training(self, params: Dict) -> str:
        """启动异步训练任务

        Args:
            params: 训练参数覆盖（可选）

        Returns:
            task_id: 训练任务ID
        """
        task_id = str(uuid.uuid4())
        config = self._build_config(params)

        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "准备中...",
            "created_at": datetime.now().isoformat(),
            "config": config.__dict__,
            "thread": None,
            "trainer": None,
            "progress_store": {},
        }

        with self._lock:
            self._tasks[task_id] = task

        # 在后台线程中启动训练
        thread = threading.Thread(
            target=self._run_training, args=(task_id, config), daemon=True
        )
        task["thread"] = thread
        thread.start()

        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """查询训练任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"],
        }

    def get_training_progress(self, task_id: str) -> Optional[Dict]:
        """获取训练进度数据（供前端轮询）"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        store = task.get("progress_store", {})
        return {
            "current_episode": store.get("current_episode", 0),
            "latest_reward": store.get("latest_reward", 0.0),
            "metrics": store.get("metrics", {}),
            "status": task["status"],
        }

    def stop_training(self, task_id: str) -> bool:
        """停止训练任务"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        task["status"] = "stopped"
        task["message"] = "用户手动停止"
        return True

    def get_models(self) -> List[Dict]:
        """获取已训练的模型列表"""
        return list(self._models.values())

    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    def evaluate(self, model_id: str, stock_codes: Optional[List[str]] = None) -> Optional[Dict]:
        """对指定模型运行评估"""
        model_info = self._models.get(model_id)
        if not model_info:
            return None

        model = model_info["model"]
        dataset = self._build_dataset()
        dataset.load()
        evaluator = RLEvaluator(self.config, model, dataset)
        return evaluator.evaluate()

    def evaluate_daily(self, model_id: str, stock_code: str, date_str: str) -> Optional[Dict]:
        """获取单日逐笔决策明细"""
        model_info = self._models.get(model_id)
        if not model_info:
            return None

        model = model_info["model"]
        dataset = self._build_dataset()
        dataset.load()
        evaluator = RLEvaluator(self.config, model, dataset)
        from datetime import date as date_type
        return evaluator.evaluate_daily(stock_code, date_type.fromisoformat(date_str))

    def _run_training(self, task_id: str, config: "RLConfig") -> None:
        """后台线程中执行训练"""
        task = self._tasks[task_id]
        try:
            task["status"] = "running"
            task["message"] = "加载数据..."

            dataset = self._build_dataset(config)
            dataset.load()

            task["message"] = "初始化模型..."
            model = self._create_model(config)

            progress_store = {}
            task["progress_store"] = progress_store

            trainer = RLTrainer(
                config=config,
                model=model,
                dataset=dataset,
                callbacks=[ProgressCallback(progress_store)],
            )
            task["trainer"] = trainer

            task["message"] = "训练中..."
            metrics = trainer.train()

            # 保存模型信息
            model_id = (
                f"{config.default_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self._models[model_id] = {
                "model_id": model_id,
                "algorithm": config.default_algorithm,
                "model": model,
                "config": config,
                "metrics": metrics.to_dict(),
                "created_at": datetime.now().isoformat(),
            }

            task["status"] = "completed"
            task["message"] = "训练完成"
            task["progress"] = 100
            task["model_id"] = model_id

        except Exception as e:
            logger.exception(f"训练失败: {e}")
            task["status"] = "failed"
            task["message"] = str(e)

    def _build_config(self, params: Dict) -> "RLConfig":
        """根据参数覆盖构建配置"""
        from rl.config import RLConfig
        config = RLConfig.from_env()
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def _build_dataset(self, config: "RLConfig" = None) -> IntradayDataset:
        """构建数据集"""
        cfg = config or self.config
        return IntradayDataset(cfg, self.db)

    def _create_model(self, config: "RLConfig") -> "AbstractRLModel":
        """根据配置创建模型"""
        if config.default_algorithm == "dqn":
            return DQNModel(config)
        elif config.default_algorithm == "ppo":
            # Phase B 实现
            raise NotImplementedError("PPO model not yet implemented")
        else:
            raise ValueError(f"Unknown algorithm: {config.default_algorithm}")