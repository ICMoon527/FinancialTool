# -*- coding: utf-8 -*-
"""临时脚本：分析最新训练日志，检查 reward 与验证指标趋势"""
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if path is None:
    logs = sorted(Path(r"e:\工作\Code\FinancialTool\rl\models\logs").glob("train_log_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    path = logs[0]
print(f"文件: {path}")

rows = list(csv.DictReader(open(path, encoding="utf-8")))
print(f"总 episodes: {len(rows)}")

print("\n--- 每 50 轮采样 ---")
for i in range(0, len(rows), 50):
    r = rows[i]
    print(f"E{int(r['episode']):>4} reward={float(r['reward']):8.3f} len={r['length']:>3} "
          f"eps={float(r['epsilon']):.3f} loss={float(r['loss']):.4f} "
          f"val_sharpe={r['val_sharpe']:>8} val_ret={r['val_return']:>8} val_win={r['val_win_rate']:>7}")

print("\n--- 最后 15 轮 ---")
for r in rows[-15:]:
    print(f"E{int(r['episode']):>4} reward={float(r['reward']):8.3f} len={r['length']:>3} "
          f"eps={float(r['epsilon']):.3f} val_sharpe={r['val_sharpe']:>8} val_ret={r['val_return']:>8}")

# 统计 reward 分布与有效动作占比
import statistics
rewards = [float(r["reward"]) for r in rows]
print(f"\nreward: mean={statistics.mean(rewards):.3f} min={min(rewards):.3f} max={max(rewards):.3f} "
      f"std={statistics.stdev(rewards):.3f}")
pos = sum(1 for x in rewards if x > 0)
print(f"reward>0 轮数: {pos}/{len(rewards)} ({pos/len(rewards):.1%})")

# 无验证行的轮次统计（验证每50轮一次，抽样100天）
val_rows = [r for r in rows if r["val_sharpe"]]
print(f"验证次数: {len(val_rows)}")
for r in val_rows:
    print(f"  E{int(r['episode']):>4} sharpe={float(r['val_sharpe']):.3f} "
          f"ret={float(r['val_return']):.3f}% win={float(r['val_win_rate']):.1%}")
