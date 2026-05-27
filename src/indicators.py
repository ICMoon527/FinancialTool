# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 技术指标库
===================================

职责：
1. 提供各类技术指标计算函数
2. 封装 MACDFS 主力成本计算指标（委托到 indicators.indicators.macdfs.MACDFS 类）
"""

from indicators.indicators.macdfs import MACDFS


def calculate_macdfs(df):
    """
    计算 MACDFS 主力成本指标（向后兼容委托接口）。

    实际计算逻辑位于 indicators.indicators.macdfs.MACDFS 类中，
    本函数保留以兼容现有调用方（如 src/core/pipeline.py）。

    Args:
        df: DataFrame，必须包含 ['open', 'high', 'low', 'close', 'vol'] 列（小写）。

    Returns:
        添加了 XA_1 ~ XA_16 列的 DataFrame。
    """
    # 列名映射：现有调用方使用小写列名，MACDFS 类要求大写首字母
    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume'}
    df_upper = df.rename(columns=col_map)

    indicator = MACDFS()
    result = indicator.calculate(df_upper)

    # 将列名恢复为小写以保持向后兼容
    reverse_map = {v: k for k, v in col_map.items()}
    return result.rename(columns=reverse_map)


def get_macdfs_summary(df_with_macdfs):
    """
    从已计算的 MACDFS 数据中提取最新一个交易日的关键指标摘要（向后兼容委托接口）。

    Args:
        df_with_macdfs: 已调用 calculate_macdfs() 计算过的 DataFrame。

    Returns:
        包含 core_cost, resistance_1 等字段的字典，若无有效数据则返回 None。
    """
    indicator = MACDFS()
    return indicator.get_summary(df_with_macdfs)