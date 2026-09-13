# -*- coding: utf-8 -*-
"""临时诊断：回放 600519/600036 验证日，定位做T亏损模式（追高/逆势/强平/做空止损）

口径与 A 评估完全一致：dqn_zero_cost/dqn_best + 真实 0.2% 成本 + 18 维状态。
输出每个交易日的动作时间线（_trades）与首笔买入相对开盘/前收的偏移，
汇总追高比例、当日涨跌分布，定位 600519 稳定亏损的根因（对照 600036 找差异）。
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(Path(PROJECT_ROOT / ".env"))

# 与 A 评估一致：真实 0.2% 成本 + 18 维（use_signal_scores=False）
os.environ["RL_COMMISSION_RATE"] = "0.0005"
os.environ["RL_SLIPPAGE_RATE"] = "0.0005"
os.environ["RL_USE_SIGNAL_SCORES"] = "false"

import numpy as np

from rl.config import RLConfig
from rl.algorithms.dqn import DQNModel
from rl.data.dataset import IntradayDataset
from rl.environment import T0Environment
from src.storage import get_db

MODEL_DIR = "rl/models/dqn_zero_cost/dqn_best"
STOCKS = ["600519", "600036"]


def kline_price(kline: dict, key: str) -> float:
    """取K线字段数值（键名大写，如 Time/Open/Close）"""
    return float(kline.get(key, 0.0) or 0.0)


def parse_minute(time_str: str) -> int:
    """"HH:MM" → 开盘后第几分钟（9:30=0）"""
    try:
        hh, mm = time_str.split(":")
        return int(hh) * 60 + int(mm) - 9 * 60 - 30
    except (ValueError, IndexError):
        return -1


config = RLConfig.from_env()
db = get_db()
dataset = IntradayDataset(config, db, max_samples=5000)
dataset.load()

model = DQNModel(config)
model.load(str(Path(MODEL_DIR) / "model.pt"))
env = T0Environment(config)

for stock in STOCKS:
    print(f"\n{'=' * 72}\n  {stock} 验证日做T诊断\n{'=' * 72}")
    chase_offsets = []    # 首买相对开盘偏移（>0 追高）
    vs_prev_offsets = []  # 首买相对前收偏移
    buy_minutes = []      # 首买时间（开盘后第几分钟）
    day_rets = []         # 当日涨跌幅
    t0_rets = []          # 当日做T收益

    for samp in dataset.val_samples:
        if getattr(samp, "stock_code", None) != stock:
            continue
        klines = dataset.get_klines(samp)
        prev_klines = dataset.get_prev_klines(samp)
        if not klines:
            continue
        prev_close = kline_price(prev_klines[-1], "Close") if prev_klines else 0.0
        day_open = kline_price(klines[0], "Open")
        day_close = kline_price(klines[-1], "Close")
        day_ret = (day_close / prev_close - 1.0) * 100 if prev_close > 0 else 0.0
        day_rets.append(day_ret)

        state = env.reset({"klines": klines, "stock_code": samp.stock_code, "date": samp.date}, prev_klines)
        done = False
        while not done:
            action = int(model.predict(state, deterministic=True))
            state, reward, done, _ = env.step(action)

        realized = env._realized_pnl
        t0_rets.append(realized)
        trades = list(env._trades)
        print(
            f"\n  {samp.date} | 做T {realized:+.2f}% | "
            f"前收 {prev_close:.2f} 开 {day_open:.2f} 收 {day_close:.2f} | 当日 {day_ret:+.2f}%"
        )
        for t in trades:
            t_pnl = t.get("pnl", 0.0)
            print(f"    {str(t.get('time', '?')):>5} {t['action']:<12} @ {t.get('price', 0.0):.2f}  pnl={t_pnl:+.2f}%")

        # 首笔买入诊断
        buys = [t for t in trades if t["action"].startswith("BUY")]
        if buys and day_open > 0:
            bp = buys[0].get("price", 0.0)
            chase = (bp / day_open - 1.0) * 100
            vs_prev = (bp / prev_close - 1.0) * 100 if prev_close > 0 else 0.0
            minute = parse_minute(str(buys[0].get("time", "")))
            chase_offsets.append(chase)
            vs_prev_offsets.append(vs_prev)
            buy_minutes.append(minute)
            print(
                f"    [诊断] 首买@{bp:.2f} 相对开盘 {chase:+.2f}% | "
                f"相对前收 {vs_prev:+.2f}% | 开盘后第{minute}分钟"
            )

    n = len(t0_rets)
    if n == 0:
        continue
    print(f"\n  ---- {stock} 汇总 ({n} 天) ----")
    print(
        f"  做T收益: 合计 {sum(t0_rets):+.2f}% | 平均 {np.mean(t0_rets):+.2f}% | "
        f"胜率 {(np.array(t0_rets) > 0).mean() * 100:.0f}%"
    )
    print(f"  当日涨跌: 平均 {np.mean(day_rets):+.2f}% | 上涨 {(np.array(day_rets) > 0).sum()}/{n} 天")
    if chase_offsets:
        ca = np.array(chase_offsets)
        print(f"  首买相对开盘: 平均 {ca.mean():+.2f}% | 追高天数 {(ca > 0).sum()}/{len(ca)}")
    if vs_prev_offsets:
        vp = np.array(vs_prev_offsets)
        print(f"  首买相对前收: 平均 {vp.mean():+.2f}%")
    if buy_minutes:
        bm = np.array([m for m in buy_minutes if m >= 0])
        print(f"  首买时间: 平均开盘后第 {bm.mean():.0f} 分钟")
