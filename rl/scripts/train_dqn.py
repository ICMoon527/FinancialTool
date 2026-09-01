# -*- coding: utf-8 -*-
"""使用真实数据库数据训练 DQN 模型

用法：
    cd e:/工作/Code/FinancialTool
    python -m rl.scripts.train_dqn
    python -m rl.scripts.train_dqn --episodes 500 --batch-size 128
    python -m rl.scripts.train_dqn --resume                # 从最近 checkpoint 续训
    python -m rl.scripts.train_dqn --resume --resume-path rl/models/dqn_latest
    python -m rl.scripts.train_dqn --no-gpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("train_dqn")


def setup_logging(log_dir: Path) -> None:
    """配置日志：同时输出到控制台和文件"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # 文件（含完整时间戳，便于事后排查）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)
    logger.info(f"日志文件: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="使用真实数据训练 DQN 模型")
    parser.add_argument("--episodes", type=int, default=None, help="训练轮数（覆盖 .env，续训时为续训轮数）")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--stock", type=str, default=None, help="仅使用指定股票（逗号分隔，如 000001,600519）")
    parser.add_argument("--max-samples", type=int, default=5000, help="样本总数上限（0=不限制），防止验证/训练规模失控")
    parser.add_argument("--rebuild", action="store_true", help="强制重建元数据缓存")
    parser.add_argument("--no-gpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--resume", action="store_true", help="从最近 checkpoint 断点续训")
    parser.add_argument("--resume-path", type=str, default=None, help="指定续训的 checkpoint 目录（默认自动找 dqn_latest）")
    parser.add_argument("--save-freq", type=int, default=50, help="latest checkpoint 保存频率（episode，0=不保存）")
    parser.add_argument("--log-dir", type=str, default=None, help="日志目录（默认 rl/models/logs）")
    args = parser.parse_args()

    # 强制 CPU
    if args.no_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from rl.config import RLConfig
    from rl.algorithms.dqn import DQNModel
    from rl.data.dataset import IntradayDataset
    from rl.training.trainer import RLTrainer
    from rl.training.callbacks import ProgressCallback
    from src.storage import get_db

    # 日志（控制台 + 文件）
    log_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "rl" / "models" / "logs"
    setup_logging(log_dir)

    # 加载配置
    config = RLConfig.from_env()
    if args.episodes:
        config.training_episodes = args.episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    import torch
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"
    logger.info(f"设备: {device}")
    logger.info(f"配置: episodes={config.training_episodes}, batch={config.batch_size}, lr={config.learning_rate}")

    # 加载数据（惰性：仅建立元数据索引）
    db = get_db()
    stock_filter = args.stock.split(",") if args.stock else None
    dataset = IntradayDataset(
        config, db,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        stock_filter=stock_filter,
        rebuild=args.rebuild,
    )
    dataset.load()

    if len(dataset.train_samples) == 0:
        logger.error("训练集为空，请先确保数据库中有分时数据")
        logger.error("可运行 python -m rl.tests.test_dqn_smoke 使用模拟数据验证")
        return 1

    logger.info(f"训练集: {len(dataset.train_samples)}, 验证集: {len(dataset.val_samples)}")

    # 创建模型
    model = DQNModel(config)

    # 训练器（save_freq: latest checkpoint 保存频率）
    progress_store = {}
    trainer = RLTrainer(
        config=config,
        model=model,
        dataset=dataset,
        callbacks=[ProgressCallback(progress_store)],
        save_freq=args.save_freq,
        log_dir=str(log_dir),
    )

    # 断点续训
    start_episode = 0
    if args.resume:
        resume_path = args.resume_path or str(Path(config.model_dir) / "dqn_latest")
        if not Path(resume_path, "model.pt").exists():
            logger.error(f"找不到可恢复的 checkpoint: {resume_path}")
            logger.error("请先完成一次训练，或用 --resume-path 指定 checkpoint 目录")
            return 1
        start_episode = trainer.resume(resume_path)
        # 用户输入的 --episodes 语义为「续训轮数」：目标轮数 = 当前进度 + 输入轮数
        config.training_episodes = start_episode + config.training_episodes
        logger.info(
            f"从 episode {start_episode} 续训, 续训 {config.training_episodes - start_episode} 轮, "
            f"目标 episode {config.training_episodes}"
        )

    # 训练
    metrics = trainer.train(start_episode=start_episode)

    # 打印结果
    print("\n" + "=" * 50)
    print("  训练完成")
    print("=" * 50)
    print(f"  总 episode: {len(metrics.episode_rewards)}")
    if metrics.episode_rewards:
        recent = metrics.episode_rewards[-10:]
        print(f"  最终平均 reward: {sum(recent) / len(recent):.4f}")
    if metrics.val_sharpe_ratios:
        print(f"  最佳验证 Sharpe: {max(metrics.val_sharpe_ratios):.4f}")
        print(f"  最佳验证收益: {max(metrics.val_returns):.4%}")
    print(f"  模型保存至: {config.model_dir}/")
    print(f"  日志目录: {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())