# -*- coding: utf-8 -*-
"""使用真实数据库数据训练 DQN 模型

用法：
    cd e:\工作\Code\FinancialTool
    python -m rl.scripts.train_dqn
    python -m rl.scripts.train_dqn --episodes 500 --batch-size 128
    python -m rl.scripts.train_dqn --no-gpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dqn")


def main():
    parser = argparse.ArgumentParser(description="使用真实数据训练 DQN 模型")
    parser.add_argument("--episodes", type=int, default=None, help="训练轮数（覆盖 .env）")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--no-gpu", action="store_true", help="强制使用 CPU")
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

    # 加载数据
    db = get_db()
    dataset = IntradayDataset(config, db)
    dataset.load()

    if len(dataset.train_samples) == 0:
        logger.error("训练集为空，请先确保数据库中有分时数据")
        logger.error("可运行 python -m rl.tests.test_dqn_smoke 使用模拟数据验证")
        return 1

    logger.info(f"训练集: {len(dataset.train_samples)}, 验证集: {len(dataset.val_samples)}")

    # 创建模型
    model = DQNModel(config)

    # 训练
    progress_store = {}
    trainer = RLTrainer(
        config=config,
        model=model,
        dataset=dataset,
        callbacks=[ProgressCallback(progress_store)],
    )
    metrics = trainer.train()

    # 打印结果
    print("\n" + "=" * 50)
    print("  训练完成")
    print("=" * 50)
    print(f"  总 episode: {len(metrics.episode_rewards)}")
    if metrics.episode_rewards:
        print(f"  最终平均 reward: {sum(metrics.episode_rewards[-10:]) / 10:.4f}")
    if metrics.val_sharpe_ratios:
        print(f"  最佳验证 Sharpe: {max(metrics.val_sharpe_ratios):.4f}")
        print(f"  最佳验证收益: {max(metrics.val_returns):.4%}")
    print(f"  模型保存至: {config.model_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())