# -*- coding: utf-8 -*-
"""临时诊断：同一进程内对比 trainer 验证路径 与 RLEvaluator 路径 的逐日 realized_pnl"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(Path(PROJECT_ROOT / ".env"))

import os
os.environ["RL_USE_SIGNAL_SCORES"] = "false"
import torch
torch.set_num_threads(1)

from rl.config import RLConfig
from rl.algorithms.dqn import DQNModel
from rl.data.dataset import IntradayDataset
from rl.environment import T0Environment
from rl.evaluation.evaluator import RLEvaluator
from src.storage import get_db

config = RLConfig.from_env()
print("use_signal_scores:", config.use_signal_scores, "state_dim:", config.state_dim)

db = get_db()
dataset = IntradayDataset(config, db, max_samples=5000, stock_filter=["000001", "600519", "000858", "600036"])
dataset.load()
model = DQNModel(config)
model.load("rl/models/dqn_best/model.pt")

# ── 路径 A：独立评估（RLEvaluator）──
evaluator = RLEvaluator(config, model, dataset)
res = evaluator.evaluate()
a_by_date = {s["date"]: s["realized_pnl"] for s in res["daily_summaries"]}

# ── 路径 B：trainer 风格（直接 dataset.get_klines/get_prev_klines + env）──
env = T0Environment(config)
b_by_date = {}
for sample in dataset.val_samples:
    klines = dataset.get_klines(sample)
    prev_klines = dataset.get_prev_klines(sample)
    state = env.reset({"klines": klines, "stock_code": sample.stock_code, "date": sample.date}, prev_klines)
    done = False
    while not done:
        action = int(model.predict(state, deterministic=True))
        state, reward, done, info = env.step(action)
    b_by_date[sample.date.isoformat()] = env._realized_pnl

# ── 逐日对比 ──
all_dates = sorted(set(a_by_date) | set(b_by_date))
print(f"\n{'日期':<12}{'A独立评估':>12}{'B-trainer路径':>14}{'差':>10}")
diff = 0.0
for d in all_dates:
    a = a_by_date.get(d, float('nan'))
    b = b_by_date.get(d, float('nan'))
    dd = b - a
    diff += dd
    if abs(dd) > 1e-6:
        print(f"{d:<12}{a:>10.3f}%{b:>12.3f}%{dd:>10.3f}")
sa = sum(a_by_date.values())
sb = sum(b_by_date.values())
print(f"\nA 累计 realized: {sa:+.2f}%  ({len(a_by_date)}天)")
print(f"B 累计 realized: {sb:+.2f}%  ({len(b_by_date)}天)")
print(f"累计差(B-A): {diff:+.2f}%")