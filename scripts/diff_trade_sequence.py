# -*- coding: utf-8 -*-
"""
对比两份回测的事件级交易序列，定位第一个分叉点（买入不一致 or 卖出不一致）。

策略：把每份 backtest_results.json 的 trades 展开为按 (date, seq) 排序的事件，
由于 max_positions=1，事件大体呈 买入->卖出->买入->卖出 交错。逐条对齐两份的事件，
输出第一个不匹配的编号及详情。

用法：
    python scripts/diff_trade_sequence.py \
        "strategy_backtest_results/<目录A>/backtest_results.json" \
        "strategy_backtest_results/<目录B>/backtest_results.json"
"""
from __future__ import annotations

import json
import sys


def load_trades(path: str):
    """加载 trades 并展开为有序事件列表 [(date, order_type, stock_code, price, quantity)]。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    trades = data["trades"]
    events = []
    for t in trades:
        events.append((
            t["date"],
            t["order_type"],      # buy / sell
            t["stock_code"],
            t["price"],
            t["quantity"],
        ))
    # 按日期稳定排序（同一日期内保持原始顺序）
    events.sort(key=lambda e: e[0])
    return events, len(trades)


def fmt(e):
    d, ot, code, price, qty = e
    return f"{d} {ot:4s} {code} @{price:.3f} x{qty}"


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    ea, na = load_trades(pa)
    eb, nb = load_trades(pb)
    print(f"A({pa.split('/')[-2]}) -> {na} events")
    print(f"B({pb.split('/')[-2]}) -> {nb} events")

    n = max(len(ea), len(eb))
    first_diff = None
    for i in range(n):
        a = ea[i] if i < len(ea) else None
        b = eb[i] if i < len(eb) else None
        if a != b:
            first_diff = i
            print("\n=== 第一个分叉点：事件 #%d ===" % i)
            print("A:", fmt(a) if a else "(无)")
            print("B:", fmt(b) if b else "(无)")
            break

    if first_diff is None:
        print("\n两个序列的事件完全一致。")
        return

    print("\n[分叉前 5 条] A 侧：")
    for i in range(max(0, first_diff - 5), first_diff):
        print("  A%2d: %s" % (i, fmt(ea[i])))
    print("[分叉前 5 条] B 侧：")
    for i in range(max(0, first_diff - 5), first_diff):
        print("  B%2d: %s" % (i, fmt(eb[i])))


if __name__ == "__main__":
    main()