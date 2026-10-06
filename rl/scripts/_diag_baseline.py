# -*- coding: utf-8 -*-
"""临时诊断：基准模型 dqn_cnn_best 的日度奖励构成（用完即删）

拆解 episode reward = 已实现收益增量 - 开仓惩罚 + 终态约束，
统计收盘未平仓手数（leftover），判断负 reward 究竟来自交易亏损还是终态惩罚。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
import torch
from rl.algorithms.dqn import DQNModel
from rl.config import RLConfig
from rl.data.dataset import IntradayDataset
from rl.environment import T0Environment
from src.storage import get_db

MODEL_DIR = PROJECT_ROOT / "rl/models/dqn_cnn_best"
N_DAYS = 100

config = RLConfig.from_env()
config.use_cnn_encoder = True
db = get_db()
dataset = IntradayDataset(config, db, max_samples=config.max_samples or None)
dataset.load()

model = DQNModel(config)
model.load(str(MODEL_DIR / "model.pt"))
print(
    f"CUDA={torch.cuda.is_available()}  state_dim={config.state_dim}  "
    f"val_pool={len(dataset.val_samples)}  reward_clip={config.reward_clip}"
)

env = T0Environment(config)
val = dataset.val_samples
rng = np.random.default_rng(42)
idx = rng.choice(len(val), size=min(N_DAYS, len(val)), replace=False)


def run(sample, policy) -> dict:
    klines = dataset.get_klines(sample)
    prev = dataset.get_prev_klines(sample)
    prev_full = dataset.get_prev_day_full_klines(sample)
    init = {"klines": klines, "stock_code": sample.stock_code, "date": sample.date}

    cap: dict = {}
    orig_fc = env._force_close

    def fc():
        cap["leftover"] = len(env._today_bought)
        return orig_fc()

    env._force_close = fc
    state = env.reset(init, prev, prev_full)
    window = env.kline_window
    rewards = []
    done = False
    while not done:
        action = policy(state, window)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
        window = env.kline_window
    env._force_close = orig_fc

    return {
        "total": env._total_reward,
        "term": rewards[-1] if rewards else 0.0,
        "rest": float(np.sum(rewards[:-1])) if len(rewards) > 1 else 0.0,
        "realized": env._realized_pnl,
        "leftover": cap.get("leftover", 0),
        "trades": len(env._trades),
    }


def report(tag: str, rows) -> None:
    tot = np.array([r["total"] for r in rows])
    term = np.array([r["term"] for r in rows])
    rest = np.array([r["rest"] for r in rows])
    real = np.array([r["realized"] for r in rows])
    lo = np.array([r["leftover"] for r in rows], dtype=float)
    tr = np.array([r["trades"] for r in rows], dtype=float)
    print("-" * 74)
    print(f"[{tag}]  日数={len(rows)}")
    print(f"  episode reward  均值={tot.mean():+8.3f}  中位={np.median(tot):+8.3f}  "
          f"范围={tot.min():+.2f}~{tot.max():+.2f}")
    print(f"  末步 reward     均值={term.mean():+8.3f}  占总量={term.mean() / tot.mean() * 100 if tot.mean() else 0:.1f}%")
    print(f"  其余步合计      均值={rest.mean():+8.3f}")
    print(f"  已实现收益 %    均值={real.mean():+8.4f}  中位={np.median(real):+8.4f}")
    print(f"  理论终态约束    均值={(-0.5 * lo**2).mean():+8.3f}  (-0.5*leftover^2)")
    print(f"  leftover        均值={lo.mean():6.3f}  最大={lo.max():.0f}  "
          f"分布={dict(zip(*np.unique(lo, return_counts=True)))}")
    print(f"  日均交易笔数    均值={tr.mean():6.2f}  零交易日占比={np.mean(tr == 0):.1%}")
    if tot.std() > 0 and lo.std() > 0:
        print(f"  corr(leftover, reward) = {np.corrcoef(lo, tot)[0, 1]:+.3f}")


report("基准模型 dqn_cnn_best", [run(val[j], lambda s, w: int(model.predict(s, w, deterministic=True))) for j in idx])
report("对照：全程 HOLD 不交易", [run(val[j], lambda s, w: int(env.HOLD)) for j in idx])