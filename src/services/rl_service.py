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

    def _to_model_id(self, ckpt_dir: Path) -> str:
        """把 checkpoint 目录转换为 model_id

        model_id = 相对服务端 model_dir 的路径，用双下划线 "__" 连接各段。
        这样既避免不同实验下同名 checkpoint（如多个实验都有 dqn_prevf_cnn_best）
        互相覆盖，又保证 id 不含 "/"（可安全放入 URL 路径供删除/回放接口使用）。
        目录不在 model_dir 之下时退化为目录名（保底）。
        """
        try:
            rel = Path(ckpt_dir).resolve().relative_to(Path(self.config.model_dir).resolve())
            if rel.parts:
                return "__".join(rel.parts)
        except ValueError:
            pass
        return Path(ckpt_dir).name

    def _scan_disk_models(self) -> None:
        """扫描 model_dir 下的 checkpoint 目录，注册为可评估模型（不加载权重）

        兼容两种目录布局：
        - 扁平：{model_dir}/{algorithm}_{tag}/model.pt
        - 嵌套：{model_dir}/{实验名}/{algorithm}_{tag}/model.pt
          （训练用 RL_MODEL_DIR 指向实验子目录时会产生嵌套布局）

        每个目录需包含 model.pt；metrics.json / trainer_state.json 可选。
        目录名（末段）用于解析算法、先验标记与时间戳；model_id 由 _to_model_id 生成。
        """
        models_root = Path(self.config.model_dir)
        if not models_root.is_dir():
            return
        count = 0
        for model_file in sorted(models_root.rglob("model.pt")):
            ckpt_dir = model_file.parent
            name = ckpt_dir.name
            model_id = self._to_model_id(ckpt_dir)

            parts = name.split("_")
            algorithm = parts[0] if parts and parts[0] in ("dqn", "ppo") else "dqn"
            # 目录名含 _prior 段表示训练时开启了先验买卖点（state_dim=10）
            use_signal_scores = len(parts) >= 2 and parts[1] == "prior"

            # 从目录名解析创建时间，解析失败则用目录修改时间
            created_at = datetime.fromtimestamp(ckpt_dir.stat().st_mtime).isoformat()
            if len(parts) >= 3:
                try:
                    created_at = datetime.strptime(
                        f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S"
                    ).isoformat()
                except ValueError:
                    pass

            # 读取指标摘要（可选）
            metrics = None
            metrics_file = ckpt_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    import json
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    pass

            # 保留已加载的权重引用：重扫时若已有同名校对注册（训练中已载入内存），
            # 不要用惰性占位覆盖，避免评估前被迫重新从磁盘读取
            existing = self._models.get(model_id, {})
            self._models[model_id] = {
                "model_id": model_id,
                "algorithm": algorithm,
                "use_signal_scores": use_signal_scores,
                # 每个模型携带其专属配置：use_signal_scores 必须与训练时一致，
                # 否则加载权重/构建评估环境时 state_dim 不匹配
                "model": existing.get("model"),
                "config": existing.get("config")
                or dataclasses.replace(
                    self.config, use_signal_scores=use_signal_scores
                ),
                "metrics": metrics,
                "created_at": created_at,
                "checkpoint_dir": str(ckpt_dir),
            }
            count += 1
        if count:
            logger.info(f"已扫描到 {count} 个磁盘模型 checkpoint")

    def reload_config(self) -> None:
        """重新加载 .env 中的 RL 配置（设置页保存后由端点调用）

        就地替换 self.config，不重建 RLService，从而保留运行中的训练/评估任务
        与已加载模型权重；仅当 model_dir 变化时才重新扫描磁盘模型，避免每次
        保存配置都做全量 rglob。
        """
        from rl.config import RLConfig

        old_model_dir = str(Path(self.config.model_dir).resolve())
        self.config = RLConfig.from_env()
        new_model_dir = str(Path(self.config.model_dir).resolve())
        if new_model_dir != old_model_dir:
            self._scan_disk_models()

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
        """获取已训练的模型列表

        每次调用先重新扫描磁盘 checkpoint 目录，保证训练过程中新生成/更新的
        best/latest 模型（即使是服务启动后才创建的目录）也能出现在列表中；
        随后按 model.pt 的实际修改时间刷新 created_at，确保展示时间准确。
        """
        self._scan_disk_models()
        models = []
        for m in self._models.values():
            item = dict(m)
            ckpt = item.get("checkpoint_dir")
            if ckpt:
                model_file = Path(ckpt) / "model.pt"
                if model_file.exists():
                    item["created_at"] = datetime.fromtimestamp(
                        model_file.stat().st_mtime
                    ).isoformat()
            models.append(item)
        return models

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

        # 删除磁盘 checkpoint 目录，并向上清理随之变空的实验子目录
        ckpt_dir = model_info.get("checkpoint_dir")
        if ckpt_dir:
            ckpt_path = Path(ckpt_dir)
            if ckpt_path.is_dir():
                shutil.rmtree(ckpt_path, ignore_errors=True)
                logger.info(f"[模型管理] 已删除磁盘 checkpoint 目录: {ckpt_path}")
            self._cleanup_empty_parents(ckpt_path)

        # 释放已加载的模型权重引用（惰性加载的模型在删除后应可被 GC 回收）
        model_info["model"] = None
        del self._models[model_id]
        logger.info(f"[模型管理] 模型已删除: {model_id}")
        return True

    def _cleanup_empty_parents(self, ckpt_dir: Path) -> None:
        """删除 checkpoint 后，向上清理随之变空的实验子目录

        嵌套布局下 checkpoint 位于 {model_dir}/{实验名}/{checkpoint}/，
        删除 checkpoint 后实验目录可能已空，一并删除避免残留空文件夹；
        清理到 model_dir 边界即停，不会误删 model_dir 自身，
        也不影响扁平布局（checkpoint 的父目录就是 model_dir）。
        """
        models_root = Path(self.config.model_dir).resolve()
        parent = Path(ckpt_dir).resolve().parent
        while parent != models_root and parent.is_relative_to(models_root):
            try:
                # 目录非空（仍有同级 checkpoint）则停止向上清理
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                logger.info(f"[模型管理] 已清理空的实验目录: {parent}")
            except OSError:
                break
            parent = parent.parent

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
            # 回写模型自身配置：load 会按 checkpoint 反推真实 state_dim（先验 1 维/旧版 2 维），
            # 评估环境必须用同一口径构建，否则 state_dim 与权重不匹配
            model_info["config"] = loaded.config
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

            # 空训练集保护：与脚本 train_dqn.py 行为对齐。缺此检查时后续 sample_train
            # 会在 randint(0, 0) 处抛 ValueError，前端只显示「训练失败」而无可操作指引
            if len(dataset.train_samples) == 0:
                raise ValueError(
                    "训练集为空：数据库中没有可用的分时数据，"
                    "请先确认已导入分时K线，或检查股票池与日期范围。"
                )

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
                ckpt_dir = self._resolve_checkpoint_dir(resume_from, config)
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

    def _resolve_checkpoint_dir(self, resume_from: str, config: "RLConfig" = None):
        """解析续训来源：模型 ID / checkpoint 目录名 / latest → Path

        config：本次任务的训练配置。传入后 "latest" 按该配置的 model_tag 匹配
        （面板切换网络结构/先验时，单例 config 未必与本次任务一致）。
        """
        cfg = config or self.config
        # 1) 已注册模型 ID → 取其 checkpoint 目录
        model_info = self._models.get(resume_from)
        if model_info and model_info.get("checkpoint_dir"):
            return Path(model_info["checkpoint_dir"])
        models_root = Path(cfg.model_dir)
        # 2) 嵌套 model_id（实验名__checkpoint名）→ 还原为目录路径
        if resume_from and resume_from != "latest" and "__" in resume_from:
            candidate = models_root.joinpath(*resume_from.split("__"))
            if candidate.is_dir() and (candidate / "model.pt").exists():
                return candidate
        # 3) 直接目录名 → model_dir/<name>（兼容扁平布局）
        if resume_from and resume_from != "latest":
            candidate = models_root / resume_from
            if candidate.is_dir() and (candidate / "model.pt").exists():
                return candidate
        # 4) 特殊值 "latest" → 递归查找最近的 <model_tag>_latest（兼容嵌套布局）
        target = f"{cfg.model_tag}_latest"
        matches = [
            p.parent for p in models_root.rglob("model.pt") if p.parent.name == target
        ]
        if matches:
            return max(matches, key=lambda d: d.stat().st_mtime)
        return None

    def _register_model(self, task: Dict, config: "RLConfig", model, trainer) -> None:
        """注册模型到内存列表（供评估/续训）

        合并到固定目录 {model_tag}_latest，不再生成时间戳 ID 的模型条目
        （模型目录只保留 best/latest，避免列表中出现"带时间戳的新模型"）。
        model_id 由 _to_model_id 生成，与磁盘扫描结果保持一致：训练中注册的内存条目
        与随后扫描到的磁盘条目是同一个 key，不会重复出现。
        """
        ckpt_path = Path(config.model_dir) / f"{config.model_tag}_latest"
        model_id = self._to_model_id(ckpt_path)
        self._models[model_id] = {
            "model_id": model_id,
            "algorithm": config.default_algorithm,
            "use_signal_scores": config.use_signal_scores,
            "model": model,
            "config": config,
            "metrics": trainer.metrics.to_dict(),
            "created_at": datetime.now().isoformat(),
            "checkpoint_dir": str(ckpt_path),
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
        """构建数据集

        max_samples 透传给数据集：0/None 表示不限制（全量），>0 时超出部分随机下采样，
        由训练面板滑块控制，用于在数据集过大时限制抽取的样本规模。
        """
        cfg = config or self.config
        max_samples = cfg.max_samples if cfg.max_samples else None
        return IntradayDataset(cfg, self.db, max_samples=max_samples)

    def get_dataset_info(self) -> Dict:
        """获取分时数据集规模（供训练面板滑块上限展示）

        直接读取元数据缓存 _meta_cache.pkl（毫秒级，不构建样本对象），
        返回股票数与样本总数（股票×交易日）。缓存不存在时 cache_exists=False，
        表示尚未建立索引（可先跑一次训练或脚本 --rebuild 生成）。
        """
        from rl.data.dataset import read_meta_cache

        meta = read_meta_cache()
        if not meta:
            return {"cache_exists": False, "total_stocks": 0, "total_samples": 0}
        return {
            "cache_exists": True,
            "total_stocks": len(meta),
            "total_samples": sum(len(dates) for dates in meta.values()),
        }

    def _create_model(self, config: "RLConfig") -> "AbstractRLModel":
        """根据配置创建模型"""
        if config.default_algorithm == "dqn":
            return DQNModel(config)
        elif config.default_algorithm == "ppo":
            # Phase B 实现
            raise NotImplementedError("PPO model not yet implemented")
        else:
            raise ValueError(f"Unknown algorithm: {config.default_algorithm}")