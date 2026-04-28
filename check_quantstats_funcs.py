
# -*- coding: utf-8 -*-
"""检查 QuantStats 统计函数"""

import quantstats as qs
from quantstats import stats

print("QuantStats 统计模块中的函数:")
print("="*80)

available_funcs = []
for name in dir(stats):
    if not name.startswith('_'):
        obj = getattr(stats, name)
        if callable(obj):
            available_funcs.append(name)

print(f"共 {len(available_funcs)} 个函数:")
print(sorted(available_funcs))
print("\n" + "="*80)

# 测试一下我们使用的函数
print("\n测试一些关键函数的签名:")

test_funcs = ['cagr', 'volatility', 'sharpe', 'sortino', 'max_drawdown', 
             'calmar', 'win_rate', 'skew', 'kurtosis', 'tail_ratio',
             'profit_factor', 'omega', 'risk_of_ruin', 'value_at_risk',
             'expected_shortfall', 'avg_drawdown', 'avg_drawdown_days',
             'common_sense_ratio', 'gain_to_pain_ratio', 'beta', 'alpha',
             'treynor_ratio', 'capture_ratio', 'upside_capture', 
             'downside_capture', 'information_ratio', 'outperformance']

for func_name in test_funcs:
    try:
        func = getattr(stats, func_name)
        print(f"\n✓ {func_name}: 存在")
        import inspect
        sig = inspect.signature(func)
        print(f"  签名: {sig}")
    except Exception as e:
        print(f"\n✗ {func_name}: 不存在 - {e}")
