# -*- coding: utf-8 -*-
"""临时诊断：累计 env 实际返回的 reward，对比 R1 重标定前后"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(Path(PROJECT_ROOT / ".env"))

import torch

torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

from rl.config import RLConfig
from rl.algorithms.dqn import DQNModel
from rl.data.dataset import IntradayDataset
from rl.environment import T0Environment
from src.storage import get_db

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="rl/models/dqn_prior_best")
args = parser.parse_args()

config = RLConfig.from_env()
db = get_db()
dataset = IntradayDataset(config, db, max_samples=5000, stock_filter=["000001", "600519", "000858", "600036"])
dataset.load()
model = DQNModel(config)
model.load(str(Path(args.model) / "model.pt"))
env = T0Environment(config)

sum_env_reward = 0.0
n_days = 0
n_trades = 0
sum_realized = 0.0
max_daily_reward = -1e9
min_daily_reward = 1e9

for samp in dataset.val_samples:
    klines = dataset.get_klines(samp) if hasattr(dataset, "get_klines") else samp.klines
    env.reset({"klines": klines, "stock_code": samp.stock_code, "date": samp.date}, None)
    done = False
    while not done:
        state = env._get_state()
        action = int(model.predict(state, deterministic=True))
        _, reward, done, _ = env.step(action)
    sum_env_reward += env._total_reward
    max_daily_reward = max(max_daily_reward, env._total_reward)
    min_daily_reward = min(min_daily_reward, env._total_reward)
    n_days += 1
    n_trades += len(env._trades)
    sum_realized += env._realized_pnl

print("=" * 46)
print(f"  env 实际 reward 汇总: {args.model} (state_dim={config.state_dim})")
print("=" * 46)
print(f"日数={n_days}  总交易={n_trades}")
print(f"env reward 累计: {sum_env_reward:+.1f}  每天平均 {sum_env_reward/n_days:+.2f}")
print(f"每日 reward 范围: {min_daily_reward:+.1f} ~ {max_daily_reward:+.1f}")
print(f"做T已实现累计: {sum_realized:+.2f}%")
print(f"（R1前累计=-344、每天约-10.4；若已明显抬升则重标定有效）")