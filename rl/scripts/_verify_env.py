# -*- coding: utf-8 -*-
"""临时验证脚本：验证多份买卖 + T+1 仓位约束 + 配对奖励（残余结算保证全天总收益精确）

校验点（对应需求）：
1. 一次可买入多份（BUY1/2/3），当日累计买入 ≤ 3 份，超额度动作无效
2. T+1：SELL 只卖底仓（先卖后买），当天买入份额保持锁定（不可卖出）
3. 配对奖励：SELL 卖出底仓时，若有未配对当日买入，立即按（卖出价-买入成本）计入做T收益（日内即时信号）
4. 残余结算：已配对买入尾盘只结算（收盘价-配对卖出价），未配对按（收盘价-买入价）结算
   —— 全天总收益与真实 T+1 账目（买入收盘强平 + 底仓收盘买回）完全一致
5. 尾盘仅保留 3 份底仓
6. reward 口径：r = Δ已实现收益 − λ·Δ敞口² + 稠密终态惩罚（收盘前 window 根内每根
   -κ·leftover²/window，leftover 只数「未配对」当日买入），逐场景独立重算比对
7. 做空守卫只拦纯开空：能配对的卖出（正T 平仓）不受守卫约束，配对不上的纯开空才拦
8. 先验买卖点默认压成 1 维有符号净信号（买/卖分结构性互斥），signal_state_dims=2 仍兼容 10 维
"""
import dataclasses
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("RL_WARMUP_STEPS", "5")
# 本脚本只验证配对/T+1 账目，需关闭做空弱势约束与做空硬止损，避免中途强平干扰账目核对
os.environ.setdefault("RL_SHORT_GUARD_ENABLED", "false")
os.environ.setdefault("RL_SHORT_STOP", "0")

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


def exposure():
    """当前日内敞口 = 当日买入份数 + 先卖后买做空份数（reward 开仓惩罚的基数）"""
    return len(env._today_bought) + len(env._sold_short)


def terminal_penalty():
    """本步的稠密终态惩罚：收盘前 window 根K线内每根计提 -κ·leftover²/window

    须在 env.step() 之后调用（读取本步推进后的 _step 与 _pending_buys）。
    leftover 只数「未配对」的当日买入（已配对=已平仓，不再计入惩罚）。
    注意：本脚本的合成日仅 30 根K线（≤ window=30），故整段 episode 都落在收盘窗口内、
    每一步剩余敞口都被计提；真实交易日约 240 根时只有最后 window 根受影响。
    """
    episode_len = min(len(klines), env.MAX_STEPS)
    bars_left = episode_len - env.current_step
    leftover = env._unpaired_buy_count()
    window = max(1, cfg.reward_terminal_window)
    if leftover > 0 and 1 <= bars_left <= window:
        return -cfg.reward_terminal_coef * leftover**2 / window
    return 0.0


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
exp_before = exposure()
s, r, d, info = env.step(5)  # SELL2 at kline[8]=10.40
print(f"  SELL2 reward={r:.6f}, valid={info['action_valid']}")
print_pos("SELL2后")
sell_price = 10.40
# 配对收益：2 份按（卖出价 - 买入价）即时结算，扣双边成本
pair_gain = 2 * ((sell_price - 10.25) / 10.25 * 100 - cfg.transaction_cost)
# 开仓惩罚：本步新增敞口（先卖后买做空 2 份）一次性收 λ·Δexposure²
inv_pen = cfg.reward_lambda * max(0, exposure() - exp_before) ** 2
# 稠密终态惩罚：本步仍有 3 份未平当日买入且处在收盘窗口内
term_pen = terminal_penalty()
expected_r = pair_gain - inv_pen + term_pen
print(f"  期望 reward = 配对收益 {pair_gain:.6f} - 开仓惩罚 {inv_pen:.6f} "
      f"+ 终态惩罚 {term_pen:.6f} = {expected_r:.6f}")
assert abs(r - expected_r) < 1e-4, f"SELL2 奖励不符: {r:.6f} vs {expected_r:.6f}"
assert env._base_position == 1, f"T+1：应卖2份底仓: {env._base_position}"
assert len(env._sold_short) == 2
assert len(env._today_bought) == 3, "T+1：当日买入应保持锁定3份"
assert len(env._pending_buys) == 3, "当日买入仍持有（配对仅影响奖励账目）"
assert sum(1 for b in env._pending_buys if b["paired_at"] is not None) == 2, "应有2份已配对"
assert abs(env._realized_pnl - pair_gain) < 1e-4, "realized_pnl 只含配对收益（不含开仓惩罚）"
print("  [OK] SELL 配对当日买入即时记收益，底仓-2、当日买入仍锁定")

# 场景3：SELL1 配对最后1份
print("--- 场景3: SELL1(动作4) 配对最后1份 ---")
exp_before = exposure()
s, r, d, info = env.step(4)  # SELL1 at kline[9]=10.45
sell_price2 = 10.45
pair_gain2 = (sell_price2 - 10.30) / 10.30 * 100 - cfg.transaction_cost
inv_pen2 = cfg.reward_lambda * max(0, exposure() - exp_before) ** 2
expected_r2 = pair_gain2 - inv_pen2 + terminal_penalty()
print(f"  SELL1 reward={r:.6f}, valid={info['action_valid']} (期望 {expected_r2:.6f})")
assert abs(r - expected_r2) < 1e-4
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

# 场景5：SELL3 卖光底仓（无当日买入）-> 纯先卖后买，reward 仅含开仓惩罚；尾盘恢复3份
print("--- 场景5: SELL3 卖光底仓（无买入），尾盘恢复3份 ---")
env.reset(sample)
skip_warmup()
exp_before = exposure()
s, r, d, info = env.step(6)  # SELL3
print(f"  SELL3 reward={r:.6f}, valid={info['action_valid']}")
print_pos("SELL3后")
assert info["action_valid"] is True
# 无当日买入可配对 → 无已实现收益，reward 仅含本步新增敞口的开仓惩罚与终态项（后者此时为 0）
inv_pen5 = cfg.reward_lambda * max(0, exposure() - exp_before) ** 2
expected_r5 = -inv_pen5 + terminal_penalty()
assert abs(r - expected_r5) < 1e-9, f"SELL3 应只产生开仓惩罚 -{inv_pen5:.6f}，实际 {r:.6f}"
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

# 场景6：做空守卫只拦「纯开空」——正T 平仓不受守卫约束
# 本脚本全局关闭了守卫（RL_SHORT_GUARD_ENABLED=false）以便核对账目，
# 此处单独构造一个开启守卫的 env 来覆盖守卫分支。
print("--- 场景6: 做空守卫只拦纯开空，配对平仓直接放行 ---")
cfg_guard = dataclasses.replace(cfg, short_guard_enabled=True, short_down_margin=0.0)
env = T0Environment(cfg_guard)
env.reset(sample)
skip_warmup()
# 合成日开盘 9.99、价格场上行，越过预热后必然处于「价格 >= 开盘价」的强势状态，
# 即旧实现下一律禁空的状态
up_pct = (env._current_kline["Close"] / env._day_open - 1.0) * 100.0
print(f"  当前 up_pct={up_pct:.3f}%（>=0 即旧实现下一律禁空）")
assert up_pct >= 0.0, f"合成日应处于强势状态，实际 up_pct={up_pct:.3f}%"

s, r, d, info = env.step(1)  # BUY1 -> 产生 1 份未配对当日买入
assert env._unpaired_buy_count() == 1, f"应有1份未配对买入: {env._unpaired_buy_count()}"
s, r, d, info = env.step(4)  # SELL1 -> 可配对，属正T 平仓，守卫应放行
print(f"  SELL1(可与当日买入配对) valid={info['action_valid']}（期望 True）")
assert info["action_valid"] is True, "正T 平仓不应被做空守卫拦截"
assert env._unpaired_buy_count() == 0, "卖出后该买入应已配对平仓"
s, r, d, info = env.step(4)  # SELL1 -> 无可配对份额，属纯开空，守卫应拦截
print(f"  SELL1(纯开空) valid={info['action_valid']}（期望 False）")
assert info["action_valid"] is False, "纯开空在强势状态下应被做空守卫拦截"
print("  [OK] 守卫只拦纯开空份额，做T 的高卖平仓路径不再被封死")

# 场景7：先验买卖点默认 1 维有符号净信号（买/卖分结构性互斥，无需买/卖分列 2 维）
# 买点必备「主力吸筹活跃」(absorption>0)、卖点必备「主力出货活跃」(absorption<0)，
# 二者互斥 → 买卖分不可能同时为正，1 维「买分−卖分」即可无损表达。
print("--- 场景7: 先验净信号 1 维编码 + 买卖分互斥 ---")
cfg_sig = dataclasses.replace(cfg, use_signal_scores=True, signal_state_dims=1)
cfg_legacy = dataclasses.replace(cfg, use_signal_scores=True, signal_state_dims=2)
assert cfg_sig.state_dim == 9, f"开启先验应为 9 维，实际 {cfg_sig.state_dim}"
assert cfg_legacy.state_dim == 10, f"旧版先验应为 10 维，实际 {cfg_legacy.state_dim}"

env_sig = T0Environment(cfg_sig)
env_sig.reset(sample)
while env_sig._is_warmup:
    env_sig.step(0)
st = env_sig._get_state()
assert st.shape == (9,), f"开启先验的状态维度应为 9，实际 {st.shape}"

exp_net = 0.0
ind = env_sig._current_indicator
if ind is not None:
    buy_s, _, _, _, _ = env_sig._signal_evaluator.evaluate_buy(ind)
    sell_s, _, _, _, _ = env_sig._signal_evaluator.evaluate_sell(ind)
    buy_s, sell_s = float(buy_s), float(sell_s)
    exp_net = buy_s / env_sig._signal_buy_max - sell_s / env_sig._signal_sell_max
    assert not (buy_s > 0 and sell_s > 0), "买/卖分结构性互斥，不应同时为正"
print(f"  买分/卖分净信号特征={float(st[8]):+.6f}（期望 {exp_net:+.6f}），且 |特征| ≤ 1")
assert abs(float(st[8]) - exp_net) < 1e-6, f"净信号特征不符: {st[8]} vs {exp_net}"
assert abs(float(st[8])) <= 1.0 + 1e-6, "净信号应落在 [-1,1]"

# 旧版 2 维路径仍可用（兼容 10 维历史 checkpoint 的加载回放）
env_old = T0Environment(cfg_legacy)
env_old.reset(sample)
while env_old._is_warmup:
    env_old.step(0)
assert env_old._get_state().shape == (10,), "signal_state_dims=2 应保持 10 维"
print("  [OK] 先验默认 1 维有符号净信号；signal_state_dims=2 仍兼容旧 10 维")

print("\n全部验证通过")
