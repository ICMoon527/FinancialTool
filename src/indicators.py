# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 技术指标库
===================================

职责：
1. 提供各类技术指标计算函数
2. 封装 MACDFS 主力成本计算指标
"""

import pandas as pd
import numpy as np


def calculate_macdfs(df):
    """
    Calculate the MACDFS indicator (main cost calculation).
    Input: df must contain ['open', 'high', 'low', 'close', 'vol']
    Returns: DataFrame with XA_1 to XA_16 columns added
    """
    cols = ['open', 'high', 'low', 'close', 'vol']
    df = df.copy()
    df[cols] = df[cols].astype(float)
    C, H, L, O, V = df['close'], df['high'], df['low'], df['open'], df['vol']

    # 1. XA_1: Volatility factor (min 1 as period)
    wp1 = (3.48 * C + H + L + O) / 5.0
    ema30 = C.ewm(span=30, adjust=False).mean()
    ema20 = C.ewm(span=20, adjust=False).mean()
    df['XA_1'] = ((wp1 - ema30) / ema20).abs().clip(lower=1.0)

    # 2. XA_2: Dynamic period EMA
    wp2 = (2.1 * C + H + L + O) / 5.0
    xa2_vals = np.zeros(len(df))
    dyn_span = df['XA_1'].ewm(span=5, adjust=False).mean().clip(lower=1.0).values
    
    for i in range(len(df)):
        n = dyn_span[i]
        alpha = 2.0 / (n + 1.0)
        if i == 0:
            xa2_vals[i] = wp2.iloc[i]
        else:
            xa2_vals[i] = alpha * wp2.iloc[i] + (1.0 - alpha) * xa2_vals[i-1]
    df['XA_2'] = xa2_vals

    # 3. XA_5: 5-day typical price MA (calculate first as XA_3 depends on it)
    tp = (L + H + C) / 3.0
    df['XA_5'] = tp.rolling(window=5).mean()

    # 4. XA_3, XA_4: Long-term cost orbit
    df['XA_3'] = df['XA_5'].ewm(span=300, adjust=False).mean() * 1.26
    df['XA_4'] = df['XA_2'].ewm(span=200, adjust=False).mean() * 1.18

    # 5. XA_6: 120-day highest cost
    df['XA_6'] = df['XA_5'].rolling(window=120).max()

    # 6. XA_7: Conditional filter
    cond = df['XA_1'] > df['XA_6'].shift(1)
    df['XA_7'] = np.where(cond, df['XA_6'], np.nan)

    # 7. XA_8: Turnover amount
    df['XA_8'] = C * V

    # 8. XA_9: Core comprehensive cost (multi-period volume-price weighted)
    def expma_div(num, den, n1, n2):
        return num.ewm(span=n1, adjust=False).mean() / den.ewm(span=n2, adjust=False).mean()

    t1 = expma_div(df['XA_8'], V, 3, 3)
    t2 = expma_div(df['XA_8'], V, 9, 6)
    t3 = expma_div(df['XA_3'], V, 12, 12)
    t4 = expma_div(df['XA_8'], V, 24, 24)
    
    avg_cost = (t1 + t2 + t3 + t4) / 4.0
    df['XA_9'] = avg_cost.ewm(span=13, adjust=False).mean()

    # 9. XA_10, XA_11: Resistance levels
    df['XA_10'] = df['XA_9'] * 1.03
    df['XA_11'] = df['XA_9'] * 1.13

    # 10. XA_12 ~ XA_15: Multiple minimum value filtering
    df['XA_12'] = np.minimum(df['XA_7'], df['XA_3'])
    df['XA_13'] = np.minimum(df['XA_10'], df['XA_3'])
    df['XA_14'] = np.minimum(df['XA_9'], df['XA_4'])
    df['XA_15'] = np.minimum(df['XA_10'], df['XA_4'])

    # 11. XA_16: Strong breakout signal
    pct = C / C.shift(1)
    is_high = (C == H)
    df['XA_16'] = ((pct > 1.03) & is_high).astype(int)

    return df


def get_macdfs_summary(df_with_macdfs):
    """
    Get a summary of MACDFS indicators from the latest data point.
    Returns a dictionary with key MACDFS metrics.
    """
    if df_with_macdfs is None or df_with_macdfs.empty:
        return None
    
    latest = df_with_macdfs.iloc[-1]
    
    return {
        'core_cost': float(latest.get('XA_9', 0)) if pd.notna(latest.get('XA_9')) else None,
        'resistance_1': float(latest.get('XA_10', 0)) if pd.notna(latest.get('XA_10')) else None,
        'resistance_2': float(latest.get('XA_11', 0)) if pd.notna(latest.get('XA_11')) else None,
        'long_term_orbit_1': float(latest.get('XA_3', 0)) if pd.notna(latest.get('XA_3')) else None,
        'long_term_orbit_2': float(latest.get('XA_4', 0)) if pd.notna(latest.get('XA_4')) else None,
        'strong_breakout_signal': int(latest.get('XA_16', 0)) if pd.notna(latest.get('XA_16')) else 0,
        'volatility_factor': float(latest.get('XA_1', 0)) if pd.notna(latest.get('XA_1')) else None,
        'support_level_1': float(latest.get('XA_12', 0)) if pd.notna(latest.get('XA_12')) else None,
        'support_level_2': float(latest.get('XA_13', 0)) if pd.notna(latest.get('XA_13')) else None,
    }
