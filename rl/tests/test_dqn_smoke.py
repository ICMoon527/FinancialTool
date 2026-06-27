# -*- coding: utf-8 -*-
"""DQN 训练管道冒烟测试：使用模拟数据端到端验证

用法：
    cd e:\工作\Code\FinancialTool
    python -m rl.tests.test_dqn_smoke

生成 1 只模拟股票 × 60 个交易日 × 240 根 1 分钟K线，
运行 50 个 episode 短训练，验证 pipeline 是否跑通。
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_test")

# 设置 RL 环境变量（测试用）
os.environ.setdefault("RL_ENABLED", "true")
os.environ.setdefault("RL_TRAINING_EPISODES", "50")
os.environ.setdefault("RL_BATCH_SIZE", "32")
os.environ.setdefault("RL_LEARNING_RATE", "0.001")
os.environ.setdefault("RL_GAMMA", "0.99")
os.environ.setdefault("RL_VALIDATION_SPLIT", "0.2")
os.environ.setdefault("RL_WARMUP_STEPS", "20")
os.environ.setdefault("RL_DENSE_REWARD_SCALE", "20")
os.environ.setdefault("RL_REPLAY_BUFFER_SIZE", "5000")
os.environ.setdefault("RL_TARGET_UPDATE_FREQ", "50")
os.environ.setdefault("RL_VALIDATION_FREQ", "25")
os.environ.setdefault("RL_EARLY_STOPPING_PATIENCE", "5")
os.environ.setdefault("RL_REWARD_CLIP", "5.0")


# ═══════════════════════════════════════════════
#  Step 1: 生成模拟分时K线数据
# ═══════════════════════════════════════════════

def generate_mock_intraday_klines(
    base_price: float = 10.0,
    n_bars: int = 240,
    volatility: float = 0.003,
    trend: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """生成一天的分时K线（带随机游走 + 趋势）

    Args:
        base_price: 开盘基准价
        n_bars: K线数量（默认 240 = 1分钟线）
        volatility: 波动率
        trend: 趋势强度（正=上涨，负=下跌）
        seed: 随机种子
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend / n_bars, volatility, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    klines = []
    for i in range(n_bars):
        minute = 9 * 60 + 30 + i  # 从 9:30 开始
        hour, minute = divmod(minute, 60)
        time_str = f"{hour:02d}:{minute:02d}"

        close = float(prices[i])
        bar_vol = volatility * 0.5
        high = close * (1 + abs(rng.normal(0, bar_vol)))
        low = close * (1 - abs(rng.normal(0, bar_vol)))
        open_price = close * (1 + rng.normal(0, bar_vol * 0.3))

        klines.append({
            "timestamp": time_str,
            "Open": float(open_price),
            "High": float(high),
            "Low": float(low),
            "Close": float(close),
            "Volume": float(rng.integers(100_000, 5_000_000)),
            "AvgPrice": float(close * (1 + rng.normal(0, 0.0005))),
        })

    return klines


def generate_mock_dataset(
    n_stocks: int = 1,
    n_days: int = 60,
    seed: int = 42,
) -> list[dict]:
    """生成模拟数据集：多个股票 × 多个交易日

    Returns:
        List[DaySample-like dict]: 每个元素包含 stock_code, date, klines, prev_day_klines
    """
    rng = np.random.default_rng(seed)
    today = date.today()
    samples = []

    for stock_idx in range(n_stocks):
        stock_code = f"{stock_idx + 1:06d}"
        base_price = 10.0 + rng.random() * 90  # 10-100 元

        for day_idx in range(n_days):
            trade_date = today - timedelta(days=(n_days - day_idx) * 2)  # 跳过周末模拟
            day_seed = seed + stock_idx * 1000 + day_idx

            # 当日 K 线
            trend = rng.normal(0, 0.5)  # 随机趋势
            klines = generate_mock_intraday_klines(
                base_price=base_price,
                n_bars=240,
                volatility=0.003,
                trend=trend,
                seed=day_seed,
            )

            # 前一日 K 线（预热用）
            prev_klines = generate_mock_intraday_klines(
                base_price=base_price * (1 - trend * 0.5),
                n_bars=240,
                volatility=0.003,
                trend=0.0,
                seed=day_seed - 1,
            )[-30:]  # 只取最后 30 根

            # 更新基准价格
            base_price = klines[-1]["Close"]

            samples.append({
                "stock_code": stock_code,
                "date": trade_date,
                "klines": klines,
                "prev_day_klines": prev_klines,
            })

    return samples


# ═══════════════════════════════════════════════
#  Step 2: 模拟 IntradayDataset（跳过数据库）
# ═══════════════════════════════════════════════

class MockIntradayDataset:
    """模拟数据集，代替从数据库读取"""

    def __init__(self, samples: list[dict], validation_split: float = 0.2):
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - validation_split))
        self._train_samples = samples[:split_idx]
        self._val_samples = samples[split_idx:]

    @property
    def train_samples(self):
        return self._train_samples

    @property
    def val_samples(self):
        return self._val_samples

    def sample_train(self):
        idx = np.random.randint(0, len(self._train_samples))
        return self._train_samples[idx]

    def sample_val(self):
        idx = np.random.randint(0, len(self._val_samples))
        return self._val_samples[idx]


# ═══════════════════════════════════════════════
#  Step 3: 冒烟测试
# ═══════════════════════════════════════════════

def test_config_loading():
    """测试配置加载"""
    from rl.config import RLConfig

    config = RLConfig.from_env()
    assert config.state_dim == 50, f"state_dim={config.state_dim}, expected 50"
    assert config.action_dim == 3, f"action_dim={config.action_dim}, expected 3"
    assert config.transaction_cost == 0.004, f"transaction_cost={config.transaction_cost}, expected 0.004"
    logger.info("  [OK] RLConfig 加载成功")
    return config


def test_environment(config):
    """测试环境 reset + step"""
    from rl.environment import T0Environment

    env = T0Environment(config)

    # 生成一天模拟数据
    klines = generate_mock_intraday_klines(base_price=20.0, n_bars=240, seed=123)
    prev_klines = generate_mock_intraday_klines(base_price=20.0, n_bars=240, seed=122)[-30:]

    sample = {
        "klines": klines,
        "stock_code": "000001",
        "date": date.today(),
    }

    # 测试 reset
    state = env.reset(sample, prev_day_klines=prev_klines)
    assert state.shape == (50,), f"state shape={state.shape}, expected (50,)"
    assert np.all(np.isfinite(state)), "state contains NaN or Inf"
    logger.info(f"  [OK] env.reset() 返回 state shape={state.shape}")

    # 测试 step（HOLD）
    next_state, reward, done, info = env.step(0)
    assert next_state.shape == (50,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    logger.info(f"  [OK] env.step(HOLD) reward={reward:.4f}, done={done}")

    # 测试 step（BUY - 预热期后应该是有效的）
    # 快速推进到预热期结束
    for _ in range(config.warmup_steps):
        _, _, done, _ = env.step(0)
        if done:
            break

    if not done:
        next_state, reward, done, info = env.step(1)  # BUY
        logger.info(f"  [OK] env.step(BUY) reward={reward:.4f}, action_valid={info['action_valid']}")

    # 测试 step（SELL）
    if not done:
        next_state, reward, done, info = env.step(2)  # SELL
        logger.info(f"  [OK] env.step(SELL) reward={reward:.4f}, action_valid={info['action_valid']}")

    return True


def test_networks(config):
    """测试网络模型创建和前向传播"""
    import torch
    from rl.networks import DQNNetwork, DuelingDQNNetwork, create_dqn_network

    batch_size = 4
    x = torch.randn(batch_size, config.state_dim)

    # DQN
    dqn = DQNNetwork(config.state_dim, config.action_dim)
    out = dqn(x)
    assert out.shape == (batch_size, config.action_dim), f"shape={out.shape}"
    logger.info(f"  [OK] DQNNetwork forward shape={out.shape}")

    # Dueling DQN
    dueling = DuelingDQNNetwork(config.state_dim, config.action_dim)
    out = dueling(x)
    assert out.shape == (batch_size, config.action_dim), f"shape={out.shape}"
    logger.info(f"  [OK] DuelingDQNNetwork forward shape={out.shape}")

    # 工厂函数
    net = create_dqn_network(config)
    out = net(x)
    assert out.shape == (batch_size, config.action_dim)
    logger.info(f"  [OK] create_dqn_network forward shape={out.shape}")

    return True


def test_dqn_model(config):
    """测试 DQNModel predict + train_step"""
    from rl.algorithms.dqn import DQNModel, ReplayBuffer

    model = DQNModel(config)

    # 测试 predict
    state = np.random.randn(config.state_dim).astype(np.float32)
    action = model.predict(state, deterministic=False)
    assert 0 <= action < config.action_dim, f"action={action} out of range"
    logger.info(f"  [OK] DQNModel.predict() returned action={action}")

    # 填充 ReplayBuffer
    for _ in range(config.batch_size + 10):
        model.replay_buffer.push(
            np.random.randn(config.state_dim).astype(np.float32),
            np.random.randint(0, config.action_dim),
            float(np.random.randn()),
            np.random.randn(config.state_dim).astype(np.float32),
            0.0,
        )

    # 测试 train_step
    metrics = model.train_step()
    assert "loss" in metrics
    assert "td_error" in metrics
    logger.info(f"  [OK] DQNModel.train_step() loss={metrics['loss']:.4f}, td_error={metrics['td_error']:.4f}")

    # 测试 save / load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        model2 = DQNModel(config)
        model2.load(tmp_path)
        action2 = model2.predict(state, deterministic=True)
        logger.info(f"  [OK] DQNModel save/load, action={action2}")
    finally:
        os.unlink(tmp_path)

    return model


def test_full_training_loop(config):
    """测试完整训练循环（50 episodes）"""
    from rl.algorithms.dqn import DQNModel
    from rl.training.trainer import RLTrainer
    from rl.training.callbacks import ProgressCallback

    # 生成数据集
    logger.info("  生成模拟数据...")
    samples = generate_mock_dataset(n_stocks=1, n_days=60, seed=42)
    dataset = MockIntradayDataset(samples, validation_split=0.2)
    logger.info(f"  训练集: {len(dataset.train_samples)}, 验证集: {len(dataset.val_samples)}")

    # 创建模型
    model = DQNModel(config)

    # 创建训练器
    progress_store = {}
    trainer = RLTrainer(
        config=config,
        model=model,
        dataset=dataset,
        callbacks=[ProgressCallback(progress_store)],
    )

    # 运行训练
    logger.info("  开始训练 (50 episodes)...")
    metrics = trainer.train()

    # 验证结果
    assert len(metrics.episode_rewards) > 0, "No episode rewards recorded"
    assert len(metrics.losses) > 0, "No losses recorded"
    logger.info(f"  [OK] 训练完成: {len(metrics.episode_rewards)} episodes")
    logger.info(f"       最终 epsilon: {metrics.epsilons[-1]:.4f}" if metrics.epsilons else "")
    logger.info(f"       验证 Sharpe: {max(metrics.val_sharpe_ratios):.4f}" if metrics.val_sharpe_ratios else "        (无验证数据)")

    return metrics


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DQN 训练管道冒烟测试")
    print("=" * 60)

    try:
        # 1. 配置加载
        logger.info("--- 1. 配置加载 ---")
        config = test_config_loading()

        # 2. 环境测试
        logger.info("--- 2. 环境测试 ---")
        test_environment(config)

        # 3. 网络测试
        logger.info("--- 3. 网络测试 ---")
        test_networks(config)

        # 4. DQN 模型测试
        logger.info("--- 4. DQN 模型测试 ---")
        test_dqn_model(config)

        # 5. 完整训练循环
        logger.info("--- 5. 完整训练循环 ---")
        test_full_training_loop(config)

        print("\n" + "=" * 60)
        print("  全部测试通过!")
        print("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"测试失败: {e}")
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())