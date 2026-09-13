# -*- coding: utf-8 -*-
"""临时脚本：评估已训练的 DQN 模型在验证集上的真实交易绩效"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（与 train_dqn 的 import 链一致，确保 RL_USE_SIGNAL_SCORES 等生效）
from dotenv import load_dotenv

load_dotenv(Path(PROJECT_ROOT / ".env"))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rl/models/dqn_best", help="模型目录（含 model.pt）")
    parser.add_argument("--stock", default="000001,600519,000858,600036")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--prev-day-features", action="store_true", help="时间维度拼接前日全天分时K线（与训练时一致）")
    parser.add_argument("--cnn-encoder", action="store_true", help="启用 1D-CNN 形态编码器（与训练时一致）")
    parser.add_argument("--no-gpu", action="store_true", help="强制 CPU 评估")
    args = parser.parse_args()

    if args.no_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from rl.config import RLConfig
    from rl.algorithms.dqn import DQNModel
    from rl.data.dataset import IntradayDataset
    from rl.evaluation.evaluator import RLEvaluator
    from src.storage import get_db

    config = RLConfig.from_env()
    if args.prev_day_features:
        config.use_prev_day_features = True
    if args.cnn_encoder:
        config.use_cnn_encoder = True
    logger = logging.getLogger("rl_eval")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    # 与训练一致的数据装配
    db = get_db()
    stock_filter = args.stock.split(",") if args.stock else None
    dataset = IntradayDataset(config, db, max_samples=args.max_samples, stock_filter=stock_filter)
    dataset.load()
    logger.info(f"训练集: {len(dataset.train_samples)}, 验证集: {len(dataset.val_samples)}")
    logger.info(f"状态维度: {config.state_dim} (use_signal_scores={config.use_signal_scores})")

    # 加载模型
    import torch
    torch.set_num_threads(1)
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if (cuda_ok and not args.no_gpu) else "cpu"
    logger.info(f"评估设备: {device}")
    model = DQNModel(config)
    model_path = Path(args.model) / "model.pt"
    model.load(str(model_path))
    logger.info(f"已加载模型: {args.model} (epsilon={model.epsilon:.4f})")

    # 评估（全量验证集）
    evaluator = RLEvaluator(config, model, dataset)
    result = evaluator.evaluate()

    # 修正指标口径：真实做T绩效用 daily_summaries 的 realized_pnl（T+1 真实做T已实现盈亏%），
    # 而非错误地把 per-step reward 当收益（reward≈-85/天被复利成 -99.9%，纯指标bug）
    real_returns = [s["realized_pnl"] / 100.0 for s in result["daily_summaries"]]
    if real_returns:
        rp = np.array(real_returns)
        real_sharpe = float(np.mean(rp) / (np.std(rp) + 1e-8) * np.sqrt(252))
        real_total = float(np.prod(1 + rp) - 1)
        real_win = float(np.mean(rp > 0))
        cum = np.cumprod(1 + rp)
        peak = np.maximum.accumulate(cum)
        real_dd = float(np.min((cum - peak) / peak))
    else:
        rp, real_sharpe, real_total, real_win, real_dd = np.array([]), 0.0, 0.0, 0.0, 0.0

    sha = result["summary_metrics"]
    br = result["benchmark_returns"]

    print("\n" + "=" * 60)
    print(f"  模型评估: {args.model}")
    print("=" * 60)
    print(f"  验证集天数: {len(real_returns)}")
    print(f"  [修正] 做T累计收益(realized_pnl): {real_total * 100:.2f}%")
    print(f"  [修正] 做T Sharpe: {real_sharpe:.4f}")
    print(f"  [修正] 做T胜率: {real_win * 100:.2f}%")
    print(f"  [修正] 做T最大回撤: {real_dd * 100:.2f}%")
    print(f"  基准总收益率(买入持有): {(br[-1] if br else float('nan')) * 100:.2f}%")
    print(f"  交易次数(买+卖): {sha.get('total_trades', 0)}")
    print(f"  对比: 做T {real_total * 100:.2f}% vs 基准 {(br[-1] if br else float('nan')) * 100:.2f}%")
    print(f"  [错误口径参照] 旧 total_return(reward误当收益): {sha.get('total_return', float('nan')) * 100:.2f}%")
    print("  每日明细(日期 股票 做T% 交易 决策" )
    for s in result["daily_summaries"]:
        print(f"    {s['date']} {s['stock_code']} {s['realized_pnl']:+.2f}% trades={s['trade_count']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())