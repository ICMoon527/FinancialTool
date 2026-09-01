# -*- coding: utf-8 -*-
"""RL 服务层：管理训练任务、模型存储、评估结果

参考 StrategyBacktestService 的异步任务模式。
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
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


class EvaluationInterrupted(Exception):
    """用户终止评估任务时由进度回调抛出，中断逐日回放循环"""


class RLService:
    """RL 服务层"""

    def __init__(self, config: "RLConfig", db: "DatabaseManager"):
        self.config = config
        self.db = db
        self._tasks: Dict[str, Dict] = {}       # task_id → 任务状态
        self._models: Dict[str, Dict] = {}      # model_id → 模型元信息
        self._lock = threading.Lock()
        # 启动时扫描磁盘上的历史模型（脚本训练/历史训练产生的 checkpoint）
        self._scan_disk_models()

    def _scan_disk_models(self) -> None:
        """扫描 model_dir 下的 checkpoint 目录，注册为可评估模型（不加载权重）

        目录命名约定: {algorithm}_{tag}，tag 为 latest（最近状态）/ best（历史最优）
        每个目录需包含 model.pt；metrics.json / trainer_state.json 可选
        """
        models_root = Path(self.config.model_dir)
        if not models_root.is_dir():
            return
        count = 0
        for d in sorted(models_root.iterdir()):
            if not d.is_dir() or not (d / "model.pt").exists():
                continue
            parts = d.name.split("_")
            algorithm = parts[0] if parts and parts[0] in ("dqn", "ppo") else "dqn"
            # 目录名含 _prior 段表示训练时开启了先验买卖点（state_dim=52）
            use_signal_scores = len(parts) >= 2 and parts[1] == "prior"

            # 从目录名解析创建时间，解析失败则用目录修改时间
            created_at = datetime.fromtimestamp(d.stat().st_mtime).isoformat()
            if len(parts) >= 3:
                try:
                    created_at = datetime.strptime(
                        f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S"
                    ).isoformat()
                except ValueError:
                    pass

            # 读取指标摘要（可选）
            metrics = None
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                try:
                    import json
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    pass

            self._models[d.name] = {
                "model_id": d.name,
                "algorithm": algorithm,
                "use_signal_scores": use_signal_scores,
                "model": None,  # 惰性加载：评估时才从磁盘读取权重
                # 每个模型携带其专属配置：use_signal_scores 必须与训练时一致，
                # 否则加载权重/构建评估环境时 state_dim 不匹配
                "config": dataclasses.replace(
                    self.config, use_signal_scores=use_signal_scores
                ),
                "metrics": metrics,
                "created_at": created_at,
                "checkpoint_dir": str(d),
            }
            count += 1
        if count:
            logger.info(f"已扫描到 {count} 个磁盘模型 checkpoint")

    def start_training(self, params: Dict) -> str:
        """启动异步训练任务

        Args:
            params: 训练参数覆盖（可选）。
                特殊键 resume_from: 模型 ID 或 checkpoint 目录名，指定后从断点续训

        Returns:
            task_id: 训练任务ID
        """
        task_id = str(uuid.uuid4())
        resume_from = params.pop("resume_from", None)
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
            "pause_event": threading.Event(),   # 置位 = 暂停
            "stop_event": threading.Event(),    # 置位 = 停止
            "resume_from": resume_from,
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
        """获取训练进度数据（供前端轮询）

        返回完整指标历史（episode_rewards/losses/epsilons/val_* 等），
        前端据此增量绘制实时监控图表。
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        store = task.get("progress_store", {})
        total = task["config"].get("training_episodes", 0)
        current = store.get("current_episode", 0)
        return {
            "current_episode": current,
            "latest_reward": store.get("latest_reward", 0.0),
            "metrics": store.get("metrics", {}),
            "status": task["status"],
            "progress": round(current / total * 100, 1) if total else 0,
            "total_episodes": total,
            "message": task["message"],
            "paused": task.get("pause_event").is_set() if task.get("pause_event") else False,
        }

    def pause_training(self, task_id: str) -> bool:
        """暂停训练任务（当前 episode 结束后生效）"""
        task = self._tasks.get(task_id)
        if not task or task["status"] != "running":
            return False
        task["pause_event"].set()
        task["message"] = "已暂停（等待当前 episode 结束）"
        return True

    def resume_training(self, task_id: str) -> bool:
        """恢复已暂停的训练任务"""
        task = self._tasks.get(task_id)
        if not task or not task.get("pause_event"):
            return False
        task["pause_event"].clear()
        task["message"] = "训练中..."
        return True

    def stop_training(self, task_id: str) -> bool:
        """停止训练任务（保存断点 checkpoint，可通过 resume_from 续训）"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.get("stop_event"):
            task["stop_event"].set()
        task["status"] = "stopping"
        task["message"] = "正在停止（保存断点）..."
        return True

    def get_models(self) -> List[Dict]:
        """获取已训练的模型列表"""
        return list(self._models.values())

    def delete_model(self, model_id: str) -> bool:
        """删除模型（内存注册 + 磁盘 checkpoint 目录）"""
        model_info = self._models.get(model_id)
        if not model_info:
            return False

        # 若该模型有运行中/待运行的评估任务，先标记终止，避免后台线程继续使用已删除的模型
        for task in self._tasks.values():
            if (
                task.get("kind") == "evaluate"
                and task.get("model_id") == model_id
                and task.get("status") in ("running", "pending")
            ):
                task["stop_event"].set()
                task["pause_event"].clear()

        # 删除磁盘 checkpoint 目录
        ckpt_dir = model_info.get("checkpoint_dir")
        if ckpt_dir and Path(ckpt_dir).is_dir():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info(f"[模型管理] 已删除磁盘 checkpoint 目录: {ckpt_dir}")

        # 释放已加载的模型权重引用（惰性加载的模型在删除后应可被 GC 回收）
        model_info["model"] = None
        del self._models[model_id]
        logger.info(f"[模型管理] 模型已删除: {model_id}")
        return True

    def _get_loaded_model(self, model_id: str):
        """获取已加载权重的模型（磁盘模型首次使用时惰性加载）"""
        model_info = self._models.get(model_id)
        if not model_info:
            return None
        model = model_info.get("model")
        if model is None:
            ckpt_dir = model_info.get("checkpoint_dir")
            if not ckpt_dir or not Path(ckpt_dir, "model.pt").exists():
                raise FileNotFoundError(f"模型权重文件缺失: {model_id}")
            loaded = self._create_model(model_info["config"])
            loaded.load(str(Path(ckpt_dir) / "model.pt"))
            model_info["model"] = loaded
            model = loaded
            logger.info(f"已从磁盘加载模型权重: {model_id}")
        return model

    def evaluate(self, model_id: str, stock_codes: Optional[List[str]] = None) -> Optional[Dict]:
        """对指定模型同步评估（保留给内部调用，前端走 start_evaluate 异步任务）"""
        model = self._get_loaded_model(model_id)
        if model is None:
            return None
        model_info = self._models.get(model_id, {})
        # 使用模型自身配置，保证 use_signal_scores 与训练时一致（state_dim 匹配）
        model_cfg = model_info.get("config") or self.config
        dataset = self._build_dataset(model_cfg)
        dataset.load()
        evaluator = RLEvaluator(model_cfg, model, dataset)
        return evaluator.evaluate()

    def start_evaluate(
        self, model_id: str, stock_codes: Optional[List[str]] = None, max_days: Optional[int] = 100
    ) -> str:
        """启动异步评估任务，返回 task_id（前端轮询 get_evaluate_progress 获取进度）

        Args:
            model_id: 模型 ID
            stock_codes: 可选股票过滤（当前版本忽略，评估全部验证集）
            max_days: 抽样评估的最大交易日数；None 或 <=0 表示全量评估。
                默认 100（随机抽样，固定种子保证可复现），避免全量评估耗时过长

        Raises:
            KeyError: 模型不存在
        """
        if model_id not in self._models:
            raise KeyError(f"模型不存在: {model_id}")

        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "kind": "evaluate",
            "model_id": model_id,
            "status": "pending",
            "progress": 0.0,
            "done": 0,      # 已完成样本数
            "total": 0,     # 总样本数
            "message": "准备中...",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "thread": None,
            "pause_event": threading.Event(),   # 置位 = 暂停（在逐日回调处阻塞）
            "stop_event": threading.Event(),    # 置位 = 终止（逐日回调处抛异常中断）
        }
        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_evaluate, args=(task_id, model_id, max_days), daemon=True
        )
        task["thread"] = thread
        thread.start()
        return task_id

    def start_evaluate_compare(
        self, model_ids: List[str], max_days: Optional[int] = 100
    ) -> str:
        """启动多模型对比评估任务：所有模型在同一批抽样数据（同基准）上评估

        Args:
            model_ids: 待对比的模型 ID 列表（>=1）
            max_days: 抽样评估的最大交易日数；None 或 <=0 表示全量评估

        Raises:
            KeyError: 任一模型不存在
        """
        for mid in model_ids:
            if mid not in self._models:
                raise KeyError(f"模型不存在: {mid}")

        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "kind": "evaluate_compare",
            "model_ids": model_ids,
            "status": "pending",
            "progress": 0.0,
            "done": 0,      # 已完成样本数（= 样本数 × 已评估模型数 的累计）
            "total": 0,     # 总样本数（= 样本数 × 模型数）
            "message": "准备中...",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "thread": None,
            "pause_event": threading.Event(),   # 置位 = 暂停（在逐日回调处阻塞）
            "stop_event": threading.Event(),    # 置位 = 终止（逐日回调处抛异常中断）
        }
        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_evaluate_compare,
            args=(task_id, model_ids, max_days),
            daemon=True,
        )
        task["thread"] = thread
        thread.start()
        return task_id

    def _run_evaluate_compare(
        self, task_id: str, model_ids: List[str], max_days: Optional[int] = 100
    ) -> None:
        """后台线程：加载数据集并抽样一次，逐个模型在同一批样本上评估"""
        import random
        import time

        task = self._tasks[task_id]
        try:
            task["status"] = "running"
            task["message"] = "加载数据集..."
            t_start = time.time()

            dataset = self._build_dataset()
            dataset.load()
            val_samples = list(getattr(dataset, "val_samples", []) or [])
            # 抽样一次（固定种子），所有模型共用同一批数据（同基准）
            if max_days and max_days > 0 and len(val_samples) > max_days:
                val_samples = random.Random(42).sample(val_samples, max_days)
                logger.info(
                    f"[对比评估] 抽样 {max_days} 个交易日（seed=42），"
                    f"{len(model_ids)} 个模型共用同一基准"
                )
            total = len(val_samples) * len(model_ids)
            task["total"] = total
            done = 0
            models_result = []
            benchmark_returns = None
            samples_meta = [
                {
                    "stock_code": (
                        s.stock_code if hasattr(s, "stock_code") else s.get("stock_code", "")
                    ),
                    "date": (
                        s.date.isoformat() if hasattr(s, "date") else str(s.get("date", ""))
                    ),
                }
                for s in val_samples
            ]

            for idx, model_id in enumerate(model_ids):
                task["message"] = f"加载模型权重 ({idx + 1}/{len(model_ids)})..."
                # 记录当前模型在全部模型中的序号（0-based），供前端按模型拆分进度
                task["current_model_idx"] = idx
                task["model_done"] = 0
                task["model_total"] = len(val_samples)
                model = self._get_loaded_model(model_id)
                if model is None:
                    raise ValueError(f"模型不存在: {model_id}")
                model_info = self._models.get(model_id, {})
                # 各模型使用自身配置（use_signal_scores 与训练时一致），保证 state_dim 匹配
                model_cfg = model_info.get("config") or self.config
                evaluator = RLEvaluator(model_cfg, model, dataset)

                cum_t_pnl = 0.0   # 当前模型累计做T已实现盈亏（%）

                def progress_cb(
                    done_in_model, total_in_model, sample, day_summary, bench_return,
                    mid=model_id, midx=idx,
                ) -> None:
                    nonlocal done, cum_t_pnl
                    # ── 控制检查点：逐日边界处响应暂停/终止 ──
                    if task["stop_event"].is_set():
                        raise EvaluationInterrupted()
                    if task["pause_event"].is_set():
                        task["message"] = f"已暂停（完成 {done}/{total}），等待恢复..."
                        task["pause_event"].wait()
                        if task["stop_event"].is_set():
                            raise EvaluationInterrupted()

                    if hasattr(sample, "stock_code"):
                        stock = sample.stock_code
                        day = sample.date
                    else:
                        stock = sample.get("stock_code", "")
                        day = sample.get("date", "")

                    day_pnl = float(day_summary.get("realized_pnl", 0.0))
                    cum_t_pnl += day_pnl
                    done += 1
                    task["done"] = done
                    task["progress"] = round(done / total * 100, 1) if total else 0.0
                    # 按模型拆分进度：当前模型的序号与已完成/总交易日数
                    task["current_model_idx"] = midx
                    task["model_done"] = done_in_model
                    task["model_total"] = total_in_model
                    task["message"] = (
                        f"[模型 {midx + 1}/{len(model_ids)}] {mid} 回放 {stock} {day} | "
                        f"当日做T {day_pnl:+.2f}% | 累计做T {cum_t_pnl:+.2f}% | "
                        f"本模型 {done_in_model}/{total_in_model} | 总进度 {done}/{total}"
                    )
                    # 终端进度日志节流：每 100 个及最后输出一次，避免刷屏
                    if done % 100 == 0 or done == total or (midx == 0 and done_in_model == 1):
                        elapsed = time.time() - t_start
                        logger.info(
                            f"[对比评估] {task['message']} | "
                            f"耗时 {elapsed / 3600:.2f}h 平均 {elapsed / max(done, 1):.1f}s/样本"
                        )

                result = evaluator.evaluate(progress_cb=progress_cb, samples=val_samples)
                if benchmark_returns is None:
                    benchmark_returns = result.get("benchmark_returns", [])
                models_result.append(
                    {
                        "model_id": model_id,
                        "cumulative_returns": result.get("cumulative_returns", []),
                        "summary_metrics": result.get("summary_metrics", {}),
                    }
                )
                summary = result.get("summary_metrics", {})
                logger.info(
                    f"[对比评估] 模型 {model_id} 完成 | "
                    f"总收益 {summary.get('total_return', 0) * 100:.2f}% "
                    f"夏普 {summary.get('sharpe_ratio', 0):.2f} "
                    f"胜率 {summary.get('win_rate', 0) * 100:.1f}% "
                    f"| 耗时 {(time.time() - t_start) / 3600:.2f}h"
                )

            task["result"] = {
                "samples": samples_meta,
                "benchmark_returns": benchmark_returns or [],
                "models": models_result,
            }
            task["status"] = "completed"
            task["message"] = "对比评估完成"
            task["progress"] = 100.0

        except EvaluationInterrupted:
            logger.info(f"[对比评估] 任务被用户终止")
            task["status"] = "stopped"
            task["message"] = f"对比评估已终止（完成 {task['done']}/{task['total']}）"

        except Exception as e:
            logger.exception(f"对比评估失败: {e}")
            task["status"] = "failed"
            task["message"] = str(e)

    def get_evaluate_progress(self, task_id: str) -> Optional[Dict]:
        """获取评估任务进度（供前端轮询）

        任务完成时附带完整评估结果 result，失败时 message 为错误信息
        """
        task = self._tasks.get(task_id)
        if not task or task.get("kind") not in ("evaluate", "evaluate_compare"):
            return None
        progress: Dict = {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "done": task["done"],
            "total": task["total"],
            "message": task["message"],
            "paused": task["pause_event"].is_set(),
            "result": None,
            # 对比评估：按模型拆分的进度（单模型评估为 None）
            "current_model_idx": task.get("current_model_idx"),
            "model_done": task.get("model_done"),
            "model_total": task.get("model_total"),
        }
        # 仅在终态附带结果，避免轮询期间重复传输大 payload
        if task["status"] in ("completed", "failed", "stopped"):
            progress["result"] = task.get("result")
        return progress

    def pause_evaluate(self, task_id: str) -> bool:
        """暂停评估任务（当前交易日回放完成后生效）"""
        task = self._tasks.get(task_id)
        if not task or task.get("kind") not in ("evaluate", "evaluate_compare") or task["status"] != "running":
            return False
        task["pause_event"].set()
        return True

    def resume_evaluate(self, task_id: str) -> bool:
        """恢复已暂停的评估任务"""
        task = self._tasks.get(task_id)
        if not task or task.get("kind") not in ("evaluate", "evaluate_compare"):
            return False
        if not task["pause_event"].is_set():
            return False
        task["pause_event"].clear()
        return True

    def stop_evaluate(self, task_id: str) -> bool:
        """终止评估任务（在下一个交易日回放前中断）"""
        task = self._tasks.get(task_id)
        if not task or task.get("kind") not in ("evaluate", "evaluate_compare"):
            return False
        if task["status"] not in ("running", "pending"):
            return False
        task["stop_event"].set()
        # 若正处于暂停阻塞中，先解除阻塞让线程走到终止检查点
        task["pause_event"].clear()
        return True

    def _run_evaluate(self, task_id: str, model_id: str, max_days: Optional[int] = 100) -> None:
        """后台线程中执行评估，通过进度回调实时更新任务状态"""
        import random
        import time

        task = self._tasks[task_id]
        try:
            task["status"] = "running"
            task["message"] = "加载模型权重..."
            logger.info(f"[评估] 开始评估模型: {model_id} (max_days={max_days or '全量'})")
            t_start = time.time()

            model = self._get_loaded_model(model_id)
            if model is None:
                raise ValueError(f"模型不存在: {model_id}")
            model_info = self._models.get(model_id, {})
            # 使用模型自身配置（use_signal_scores 与训练时一致），保证 state_dim 匹配
            model_cfg = model_info.get("config") or self.config
            logger.info(
                f"[评估] 模型权重加载完成，耗时 {time.time() - t_start:.1f}s"
            )

            task["message"] = "加载数据集..."
            dataset = self._build_dataset(model_cfg)
            dataset.load()
            val_samples = list(getattr(dataset, "val_samples", []) or [])
            logger.info(
                f"[评估] 数据集加载完成，验证集 {len(val_samples)} 样本，"
                f"耗时 {time.time() - t_start:.1f}s"
            )

            # 抽样评估：超过 max_days 时随机抽样（固定种子保证可复现）
            if max_days and max_days > 0 and len(val_samples) > max_days:
                val_samples = random.Random(42).sample(val_samples, max_days)
                logger.info(
                    f"[评估] 抽样评估: 从验证集随机抽取 {max_days} 个交易日"
                    f"（seed=42 保证可复现）"
                )

            evaluator = RLEvaluator(model_cfg, model, dataset)

            cum_t_pnl = 0.0       # 累计做T已实现盈亏（%，简单加总）
            cum_bench = 0.0       # 累计基准收益（%，简单加总）
            cum_win = 0           # 做T盈利天数

            def progress_cb(done: int, total: int, sample, day_summary: Dict, bench_return: float) -> None:
                nonlocal cum_t_pnl, cum_bench, cum_win
                # ── 控制检查点：逐日边界处响应暂停/终止 ──
                if task["stop_event"].is_set():
                    raise EvaluationInterrupted()
                if task["pause_event"].is_set():
                    task["message"] = f"已暂停（完成 {done}/{total}），等待恢复..."
                    task["pause_event"].wait()
                    # 暂停期间可能收到终止指令
                    if task["stop_event"].is_set():
                        raise EvaluationInterrupted()

                # 逐日回放实时报告：更新进度与当日做T信息
                task["done"] = done
                task["total"] = total
                if hasattr(sample, "stock_code"):
                    stock = sample.stock_code
                    day = sample.date
                else:
                    stock = sample.get("stock_code", "")
                    day = sample.get("date", "")

                day_pnl = float(day_summary.get("realized_pnl", 0.0))
                buy_n = sum(1 for t in day_summary.get("trades", []) if t.get("action") == "BUY")
                sell_n = sum(1 for t in day_summary.get("trades", []) if t.get("action") == "SELL")
                cum_t_pnl += day_pnl
                cum_bench += bench_return * 100
                if day_pnl > 0:
                    cum_win += 1

                task["message"] = (
                    f"回放 {stock} {day} | 当日做T {day_pnl:+.2f}%（买{buy_n}/卖{sell_n}）"
                    f" | 基准 {bench_return * 100:+.2f}% | 累计做T {cum_t_pnl:+.2f}%"
                )
                task["progress"] = round(done / total * 100, 1) if total else 0.0
                # 终端进度日志：首个样本、之后每 50 个及最后一个输出一次，避免刷屏
                if done == 1 or done % 50 == 0 or done == total:
                    elapsed = time.time() - t_start
                    speed = elapsed / max(done, 1)
                    eta = speed * max(total - done, 0)
                    logger.info(
                        f"[评估] 进度 {done}/{total} ({done / max(total, 1) * 100:.1f}%) "
                        f"当前 {stock} {day} 当日做T {day_pnl:+.2f}% | "
                        f"累计做T {cum_t_pnl:+.1f}% 累计基准 {cum_bench:+.1f}% "
                        f"做T胜率(日) {cum_win / max(done, 1) * 100:.0f}% | "
                        f"平均 {speed:.1f}s/样本 预计剩余 {eta / 3600:.1f} 小时"
                    )

            task["message"] = "逐日回放验证集..."
            result = evaluator.evaluate(progress_cb=progress_cb, samples=val_samples)

            summary = result.get("summary_metrics", {}) if result else {}
            logger.info(
                f"[评估] 完成: 总耗时 {(time.time() - t_start) / 3600:.2f} 小时 | "
                f"夏普 {summary.get('sharpe_ratio', 0):.2f} "
                f"总收益 {summary.get('total_return', 0) * 100:.2f}% "
                f"胜率 {summary.get('win_rate', 0) * 100:.1f}%"
            )

            task["result"] = result
            task["status"] = "completed"
            task["message"] = "评估完成"
            task["progress"] = 100.0

        except EvaluationInterrupted:
            logger.info(f"[评估] 任务被用户终止: {model_id}")
            task["status"] = "stopped"
            task["message"] = f"评估已终止（完成 {task['done']}/{task['total']}）"

        except Exception as e:
            logger.exception(f"评估失败: {e}")
            task["status"] = "failed"
            task["message"] = str(e)

    def evaluate_daily(self, model_id: str, stock_code: str, date_str: str) -> Optional[Dict]:
        """获取单日逐笔决策明细"""
        model = self._get_loaded_model(model_id)
        if model is None:
            return None

        dataset = self._build_dataset()
        dataset.load()
        evaluator = RLEvaluator(self.config, model, dataset)
        from datetime import date as date_type
        return evaluator.evaluate_daily(stock_code, date_type.fromisoformat(date_str))

    def _run_training(self, task_id: str, config: "RLConfig") -> None:
        """后台线程中执行训练（支持暂停/停止/断点续训）"""
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

            from rl.training.callbacks import TrainingControlCallback, TrainingInterrupted

            trainer = RLTrainer(
                config=config,
                model=model,
                dataset=dataset,
                callbacks=[
                    ProgressCallback(progress_store),
                    TrainingControlCallback(
                        task["pause_event"], task["stop_event"]
                    ),
                ],
                save_freq=20,
            )
            task["trainer"] = trainer

            # 断点续训：从指定模型/checkpoint 恢复
            start_episode = 0
            resume_from = task.get("resume_from")
            if resume_from:
                ckpt_dir = self._resolve_checkpoint_dir(resume_from)
                if ckpt_dir is None:
                    raise ValueError(f"找不到可恢复的 checkpoint: {resume_from}")
                task["message"] = f"从 {ckpt_dir} 恢复..."
                start_episode = trainer.resume(str(ckpt_dir))
                # 用户输入的 episodes 语义为「续训轮数」：目标轮数 = 当前进度 + 输入轮数
                config.training_episodes = start_episode + config.training_episodes
                task["message"] = (
                    f"从 {ckpt_dir.name} 恢复, 续训 "
                    f"{config.training_episodes - start_episode} 轮, "
                    f"目标 episode {config.training_episodes}"
                )

            task["message"] = "训练中..."
            trainer.train(start_episode=start_episode)

            # 注册模型（含最新 checkpoint 目录，供评估与续训）
            self._register_model(task, config, model, trainer)

            task["status"] = "completed"
            task["message"] = "训练完成"
            task["progress"] = 100
            task["model_id"] = task.get("model_id") or self._latest_model_id()

        except TrainingInterrupted as e:
            # 用户停止：保存断点，注册模型供续训/评估
            logger.info(f"训练被用户停止: {e}")
            trainer = task.get("trainer")
            if trainer is not None:
                try:
                    trainer._save_checkpoint("latest", next_episode=e.next_episode)
                    self._register_model(task, config, trainer.model, trainer)
                except Exception as save_err:
                    logger.exception(f"停止时保存断点失败: {save_err}")
            task["status"] = "stopped"
            task["message"] = f"已停止于 episode {e.next_episode}（可断点续训）"
            task["progress"] = 0

        except Exception as e:
            logger.exception(f"训练失败: {e}")
            task["status"] = "failed"
            task["message"] = str(e)

    def _resolve_checkpoint_dir(self, resume_from: str):
        """解析续训来源：模型 ID 或 checkpoint 目录名 → Path"""
        # 1) 已注册模型 ID → 取其 checkpoint 目录
        model_info = self._models.get(resume_from)
        if model_info and model_info.get("checkpoint_dir"):
            return Path(model_info["checkpoint_dir"])
        # 2) "latest" 或直接目录名 → rl/models/<name>
        from rl.config import RLConfig
        models_root = Path(self.config.model_dir)
        candidate = models_root / resume_from
        if candidate.is_dir() and (candidate / "model.pt").exists():
            return candidate
        # 3) 特殊值 "latest" → dqn_latest / ppo_latest（带先验时为 dqn_prior_latest）
        candidate = models_root / f"{self.config.model_tag}_latest"
        if candidate.is_dir() and (candidate / "model.pt").exists():
            return candidate
        return None

    def _register_model(self, task: Dict, config: "RLConfig", model, trainer) -> None:
        """注册模型到内存列表（供评估/续训），记录 checkpoint 目录"""
        model_id = (
            f"{config.model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._models[model_id] = {
            "model_id": model_id,
            "algorithm": config.default_algorithm,
            "use_signal_scores": config.use_signal_scores,
            "model": model,
            "config": config,
            "metrics": trainer.metrics.to_dict(),
            "created_at": datetime.now().isoformat(),
            "checkpoint_dir": str(Path(config.model_dir) / f"{config.model_tag}_latest"),
        }
        task["model_id"] = model_id

    def _latest_model_id(self) -> Optional[str]:
        """获取最新注册的模型 ID"""
        if not self._models:
            return None
        return max(
            self._models.values(), key=lambda m: m["created_at"]
        )["model_id"]

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