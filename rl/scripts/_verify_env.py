# -*- coding: utf-8 -*-
"""临时验证脚本：验证多份买卖 + T+1 仓位约束 + 配对奖励（残余结算保证全天总收益精确）

校验点（对应需求）：
1. 一次可买入多份（BUY1/2/3），当日累计买入 ≤ 3 份，超额度动作无效
2. T+1：SELL 只卖底仓（先卖后买），当天买入份额保持锁定（不可卖出）
3. 配对奖励：SELL 卖出底仓时，若有未配对当日买入，立即按（卖出价-买入成本）计入做T收益（日内即时信号）
4. 残余结算：已配对买入尾盘只结算（收盘价-配对卖出价），未配对按（收盘价-买入价）结算
   —— 全天总收益与真实 T+1 账目（买入收盘强平 + 底仓收盘买回）完全一致
5. 尾盘仅保留 3 份底仓
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
s, r, d, info = env.step(1)  # BUY1 -> 当日买入3份(满)
s, r, d, info = env.step(1)  # BUY1 -> 超出额度，无效
print(f"  BUY1(超额度) valid={info['action_valid']}")
print_pos("满额后")
assert len(env._today_bought) == 3 and len(env._pending_buys) == 3 and env._base_position == 3
assert info["action_valid"] is False
print("  [OK] 一次可买多份，累计 ≤ 3，超额度无效")

# 记录买入价
buys = [b["price"] for b in env._pending_buys]
print(f"  买入价: {buys}")

# 场景2：SELL2 配对2份当日买入（先卖后买做T），立即产生配对收益
print("--- 场景2: SELL2(动作5) 配对2份，即时做T收益 ---")
s, r, d, info = env.step(5)  # SELL2 at kline[8]=10.40
print(f"  SELL2 reward={r:.6f}, valid={info['action_valid']}")
print_pos("SELL2后")
sell_price = 10.40
expected_pair = 2 * ((sell_price - 10.25) / 10.25 * 100 - cfg.transaction_cost) + cfg.trade_act_bonus  # 配对收益 + 有效SELL行为激励
print(f"  期望配对收益(含SELL激励)={expected_pair:.6f}")
assert abs(r - expected_pair) < 1e-4, f"SELL2配对收益不符: {r:.6f} vs {expected_pair:.6f}"
assert env._base_position == 1, f"T+1：应卖2份底仓: {env._base_position}"
assert len(env._sold_short) == 2
assert len(env._today_bought) == 3, "T+1：当日买入应保持锁定3份"
assert len(env._pending_buys) == 3, "当日买入仍持有（配对仅影响奖励账目）"
assert sum(1 for b in env._pending_buys if b["paired_at"] is not None) == 2, "应有2份已配对"
assert abs(env._realized_pnl - (expected_pair - cfg.trade_act_bonus)) < 1e-4, "realized_pnl 只含配对收益（不含行为激励）"
print("  [OK] SELL 配对当日买入即时记收益，底仓-2、当日买入仍锁定")

# 场景3：SELL1 配对最后1份
print("--- 场景3: SELL1(动作4) 配对最后1份 ---")
s, r, d, info = env.step(4)  # SELL1 at kline[9]=10.45
sell_price2 = 10.45
expected_pair2 = (sell_price2 - 10.30) / 10.30 * 100 - cfg.transaction_cost + cfg.trade_act_bonus
print(f"  SELL1 reward={r:.6f}, valid={info['action_valid']} (期望 {expected_pair2:.6f})")
assert abs(r - expected_pair2) < 1e-4
assert env._base_position == 0 and len(env._sold_short) == 3
assert len(env._today_bought) == 3, "当日买入3份全部锁定（T+1）"
print("  [OK] 3份买入全部配对，底仓卖光，当日买入仍锁定")

# 场景3b：SELL1 无底仓 -> 无效
print("--- 场景3b: SELL1(动作4) 无底仓无效 ---")
s, r, d, info = env.step(4)
print(f"  SELL1 reward={r:.3f}, valid={info['action_valid']}")
assert info["action_valid"] is False
print("  [OK] 无底仓时 SELL 无效")

# 场景4：推进到尾盘，验证残余结算 + 全天总收益与真实T+1账目一致
print("--- 场景4: 尾盘强平，全天总收益精确验证 ---")
steps = 0
while not d:
    s, r, d, info = env.step(0)
    steps += 1
    if steps > 60:
        break
close_price = 10.05
print_pos("尾盘")
print(f"  realized_pnl={env._realized_pnl:.6f}")
assert env._base_position == 3, f"底仓未恢复到3: {env._base_position}"
assert len(env._today_bought) == 0 and len(env._pending_buys) == 0 and len(env._sold_short) == 0
# 独立重算真实 T+1 账目：买入按收盘价强平 + 底仓按收盘价买回
expected_total = 0.0
for b in buys:
    expected_total += (close_price - b) / b * 100 - cfg.transaction_cost
for sp in (10.40, 10.40, 10.45):
    expected_total += (sp - close_price) / close_price * 100 - cfg.transaction_cost
print(f"  独立重算真实T+1总收益={expected_total:.6f}")
assert abs(env._realized_pnl - expected_total) < 1e-6, f"总收益不符: {env._realized_pnl:.6f} vs {expected_total:.6f}"
print("  [OK] 配对奖励+残余结算后，全天总收益与真实T+1账目精确一致")

# 场景5：SELL3 卖光底仓（无当日买入）-> 纯先卖后买，reward=0；尾盘恢复3份
print("--- 场景5: SELL3 卖光底仓（无买入），尾盘恢复3份 ---")
env.reset(sample)
skip_warmup()
s, r, d, info = env.step(6)  # SELL3
print(f"  SELL3 reward={r:.6f}, valid={info['action_valid']}")
print_pos("SELL3后")
assert info["action_valid"] is True
assert abs(r - cfg.trade_act_bonus) < 1e-9, "无当日买入可配对时，SELL 只产生行为激励（盈亏尾盘买回结算）"
assert env._base_position == 0 and len(env._sold_short) == 3
sold_prices = list(env._sold_short)  # 在尾盘强平前记录卖出价
steps = 0
while not d:
    s, r, d, info = env.step(0)
    steps += 1
    if steps > 60:
        break
# 纯先卖后买：卖底仓3份，尾盘按收盘价买回 -> Σ(卖出价-收盘价)/收盘价*100 - 3*cost
expected_s5 = sum((sp - close_price) / close_price * 100 - cfg.transaction_cost for sp in sold_prices)
print(f"  sold={sold_prices}, 收盘={close_price}")
print(f"  realized_pnl={env._realized_pnl:.6f}, 期望={expected_s5:.6f}")
assert abs(env._realized_pnl - expected_s5) < 1e-6
assert env._base_position == 3, f"卖光后尾盘未恢复3份: {env._base_position}"
print("  [OK] SELL3 一次卖光底仓，尾盘恢复3份，盈亏正确结算")

print("\n全部验证通过")
