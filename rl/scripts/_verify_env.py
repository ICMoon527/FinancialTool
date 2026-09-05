# -*- coding: utf-8 -*-
"""临时验证脚本：验证多份买入/卖出（T+1：当天买入不可卖出，SELL 只卖底仓）、尾盘强平与底仓恢复

校验点（对应需求）：
1. 一次可买入多份（BUY1/2/3），当日累计买入 ≤ 3 份，超额度动作无效
2. T+1：SELL1/2/3 只卖底仓（先卖后买），当天买入份额不可当天卖出、不参与配对
3. 卖出份数 ≤ 底仓，超过底仓的动作无效
4. 尾盘强平所有当日买入（lock 到尾盘），仅保留 3 份底仓；卖出的底仓按收盘价买回补齐
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("RL_WARMUP_STEPS", "5")
os.environ.setdefault("RL_DENSE_REWARD_SCALE", "20")
os.environ.setdefault("RL_TRADE_ACT_BONUS", "0.05")

from rl.config import RLConfig  # noqa: E402
from rl.environment import T0Environment  # noqa: E402

cfg = RLConfig.from_env()
env = T0Environment(cfg)

# 构造 30 根K线，先涨后跌便于测试先卖后买
prices = [10.0 + 0.05 * i for i in range(15)] + [10.75 - 0.05 * i for i in range(15)]
klines = []
for i, p in enumerate(prices):
    klines.append({
        "timestamp": f"{9 + (i // 60)}:{30 + i % 60:02d}",
        "Open": p - 0.01, "High": p + 0.03, "Low": p - 0.03,
        "Close": p, "Volume": 1000, "AvgPrice": p,
    })

sample = {"klines": klines, "stock_code": "TEST", "date": date(2026, 9, 4)}
env.reset(sample)


def skip_warmup():
    """推进越过预热期（强制 HOLD），使后续动作真实生效"""
    for _ in range(env.warmup_steps):
        s, r, d, info = env.step(0)
        if d or not env._is_warmup:
            break


def print_pos(tag):
    print(f"  {tag}: 底仓={env._base_position}, 当日买入={len(env._today_bought)}, "
          f"pending={len(env._pending_buys)}, sold_short={len(env._sold_short)}")


# 场景1：BUY2 + BUY1 满额，超额度无效
print("--- 场景1: BUY2(动作2) + BUY1，超额度无效 ---")
skip_warmup()
s, r, d, info = env.step(2)  # BUY2
print(f"  BUY2 reward={r:.3f}, valid={info['action_valid']}")
s, r, d, info = env.step(1)  # BUY1 -> 当日买入3份(满)
print(f"  BUY1 reward={r:.3f}, valid={info['action_valid']}")
s, r, d, info = env.step(1)  # BUY1 -> 超出额度，无效
print(f"  BUY1(超额度) reward={r:.3f}, valid={info['action_valid']}")
print_pos("满额后")
assert len(env._today_bought) == 3 and len(env._pending_buys) == 3 and env._base_position == 3
assert info["action_valid"] is False
print("  [OK] 一次可买多份，累计 ≤ 3，超额度无效")

# 场景2：T+1 —— SELL2（动作5）卖出2份，只卖底仓，不改动当日买入
print("--- 场景2: SELL2 只卖底仓（T+1，不配对当日买入） ---")
s, r, d, info = env.step(5)  # SELL2
print(f"  SELL2 reward={r:.3f}, valid={info['action_valid']}")
print_pos("SELL2后")
assert info["action_valid"] is True
assert env._base_position == 1, f"应卖2份底仓: {env._base_position}"
assert len(env._sold_short) == 2, f"sold_short应=2: {len(env._sold_short)}"
assert len(env._today_bought) == 3 and len(env._pending_buys) == 3, "T+1：当日买入不可卖出，应保持3份"
assert env._realized_pnl == 0.0, "卖出底仓的盈亏应延后到尾盘买回时结算"
print("  [OK] SELL 只卖底仓，当日买入保持锁定，盈亏延后到尾盘")

# 场景3：SELL3（动作6）超底仓失效
print("--- 场景3: SELL3 超底仓（底仓=1）无效 ---")
s, r, d, info = env.step(6)  # SELL3 -> 3>1，无效
print(f"  SELL3 reward={r:.3f}, valid={info['action_valid']}")
assert info["action_valid"] is False
assert env._base_position == 1 and len(env._sold_short) == 2
print("  [OK] 卖出份数超过底仓时动作无效")

# 场景4：推进到尾盘，当日买入全部强平 + 底仓恢复为3
print("--- 场景4: 尾盘强平当日买入，仅保留3份底仓 ---")
steps = 0
while not d:
    s, r, d, info = env.step(0)
    steps += 1
    if steps > 60:
        break
print(f"  done={d}, realized_pnl={env._realized_pnl:.3f}")
print_pos("尾盘")
assert env._base_position == 3, f"底仓未恢复到3: {env._base_position}"
assert len(env._today_bought) == 0 and len(env._pending_buys) == 0, "当日买入未强平"
assert len(env._sold_short) == 0, "sold_short未清空"
assert env.total_position == 3
print("  [OK] 尾盘仅保留3份底仓，当日买入全部强平")

# 场景5：SELL3（动作6）直接卖光3份底仓（无当日买入），尾盘恢复3份
print("--- 场景5: SELL3 一次卖光底仓，尾盘恢复3份 ---")
env.reset(sample)
skip_warmup()
s, r, d, info = env.step(6)  # SELL3
print(f"  SELL3 reward={r:.3f}, valid={info['action_valid']}")
print_pos("SELL3后")
assert info["action_valid"] is True
assert env._base_position == 0 and len(env._sold_short) == 3
s, r, d, info = env.step(4)  # SELL1 -> 底仓=0，无效
assert info["action_valid"] is False and env._base_position == 0
steps = 0
while not d:
    s, r, d, info = env.step(0)
    steps += 1
    if steps > 60:
        break
print(f"  尾盘底仓={env._base_position}, sold_short={len(env._sold_short)}")
assert env._base_position == 3, f"卖光后尾盘未恢复3份: {env._base_position}"
print("  [OK] SELL3 一次卖光底仓，尾盘仍恢复3份")

print("\n全部验证通过")