# -*- coding: utf-8 -*-
"""MACDFS 主力成本分析指标

该指标基于量价关系和多周期 EMA 融合技术，估算主力资金的综合持仓成本，
并据此推导压力位、支撑位和强势突破信号。

理论依据：
- 量价加权成本：某价格区间成交额越大，说明该区间换手越充分，持仓成本越集中。
- 自适应 EMA：波动率因子 XA_1 动态调整 EMA 周期，高波动时敏感跟踪，低波动时平滑去噪。
- 多周期融合：3/9/12/24 四个时间窗口覆盖短、中、长线成本，避免单一窗口偏差。
"""

from typing import cast

import numpy as np
import pandas as pd

from ..base import BaseIndicator


class MACDFS(BaseIndicator):
    """
    MACDFS（Multi-period Adaptive Cost Dynamic Fusion System）主力成本指标。

    通过多周期量价加权 EMA 融合计算主力综合持仓成本，并基于此推导：
    - 核心综合成本（XA_9）：主力平均持仓成本估计
    - 第一/第二压力位（XA_10/XA_11）：核心成本上浮 3% / 13%
    - 长期成本轨道（XA_3/XA_4）：长期趋势支撑线
    - 强势突破信号（XA_16）：单日涨幅 >3% 且收于最高点
    - 波动率因子（XA_1）：自适应 EMA 的动态周期参数

    计算公式（逐步）：
    1. XA_1 = |(3.48*C + H + L + O)/5 - EMA30(C)| / EMA20(C)，截断下限为 1.0
    2. XA_2 = 自适应 EMA，周期 n 由 XA_1 的 5 日 EMA 决定，alpha = 2/(n+1)
    3. XA_5 = (L + H + C)/3 的 5 日 SMA
    4. XA_3 = XA_5 的 300 日 EMA × 1.26（长期成本轨道1）
    5. XA_4 = XA_2 的 200 日 EMA × 1.18（长期成本轨道2）
    6. XA_6 = XA_5 的 120 日滚动最大值
    7. XA_7 = 条件过滤器：XA_1 > 前一日 XA_6 时取 XA_6，否则 NaN
    8. XA_8 = C × V（成交额）
    9. XA_9 = 多周期量价加权 EMA 的等权平均，再经 13 日 EMA 平滑
    10. XA_10/XA_11 = XA_9 × 1.03 / 1.13
    11. XA_12~XA_15 = 多层 min 过滤的支撑位
    12. XA_16 = 涨幅 >3% 且收盘价 = 最高价 → 1，否则 0

    输入参数：
    - data: DataFrame，必须包含 Open, High, Low, Close, Volume 列
    - 该指标无额外可调参数（所有系数均为固定值）

    输出参数（新增列）：
    - XA_1: 波动率因子（自适应 EMA 的动态周期）
    - XA_2: 动态周期 EMA（自适应均价线）
    - XA_3: 长期成本轨道 1（基于典型价格的长期趋势支撑线）
    - XA_4: 长期成本轨道 2（基于自适应 EMA 的中期趋势支撑线）
    - XA_5: 5 日典型价格 SMA
    - XA_6: 120 日最高成本
    - XA_7: 条件过滤器
    - XA_8: 成交额（C × V）
    - XA_9: 核心综合成本（多周期量价加权）
    - XA_10: 第一压力位（核心成本 × 1.03）
    - XA_11: 第二压力位（核心成本 × 1.13）
    - XA_12: 支撑位1（min(XA_7, XA_3)）
    - XA_13: 支撑位2（min(XA_10, XA_3)）
    - XA_14: 支撑位3（min(XA_9, XA_4)）
    - XA_15: 支撑位4（min(XA_10, XA_4)）
    - XA_16: 强势突破信号（0/1）

    注意事项：
    - 需要至少 100 条历史数据才能得到有意义的计算结果（EMA 收敛需要时间）
    - 所有系数（3.48, 2.1, 1.26, 1.18, 1.03, 1.13）为固定经验值
    - 成本线本质是数学模型拟合，不代表真实的主力持仓成本
    - 在低换手率股票上，量价加权成本可能失真
    - XA_1 截断下限 1.0 意味着低波动盘整期自适应 EMA 退化为接近当前价格的快速线
    - 建议结合成交量、均线系统、筹码分布等指标交叉验证

    使用示例：
        >>> macdfs = MACDFS()
        >>> result = macdfs.calculate(df)          # 计算完整指标
        >>> summary = macdfs.get_summary(result)   # 提取最新摘要
    """

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算 MACDFS 指标。

        Args:
            data: 包含 OHLCV 数据的 DataFrame，必须包含 Open, High, Low, Close, Volume 列。

        Returns:
            添加了 XA_1 ~ XA_16 列的 DataFrame。
        """
        self.validate_input(data)

        # 列名兼容：BaseIndicator.validate_input 要求大写首字母，
        # 但内部计算逻辑使用小写列名，此处做映射。
        df = data.copy()
        # 保留原始列，同时创建小写别名供计算使用
        C = df['Close'].astype(float)
        H = df['High'].astype(float)
        L = df['Low'].astype(float)
        O = df['Open'].astype(float)
        V = df['Volume'].astype(float)

        # ---- XA_1: 波动率因子 ----
        wp1 = (3.48 * C + H + L + O) / 5.0
        ema30 = C.ewm(span=30, adjust=False).mean()
        ema20 = C.ewm(span=20, adjust=False).mean()
        df['XA_1'] = ((wp1 - ema30) / ema20).abs().clip(lower=1.0)

        # ---- XA_2: 动态周期自适应 EMA ----
        wp2 = (2.1 * C + H + L + O) / 5.0
        xa2_vals = np.zeros(len(df))
        dyn_span = df['XA_1'].ewm(span=5, adjust=False).mean().clip(lower=1.0).values

        for i in range(len(df)):
            n = dyn_span[i]
            alpha = 2.0 / (n + 1.0)
            if i == 0:
                xa2_vals[i] = wp2.iloc[i]
            else:
                xa2_vals[i] = alpha * wp2.iloc[i] + (1.0 - alpha) * xa2_vals[i - 1]
        df['XA_2'] = xa2_vals

        # ---- XA_5: 5 日典型价格 SMA（在 XA_3 之前计算，因为 XA_3 依赖它）----
        tp = (L + H + C) / 3.0
        df['XA_5'] = tp.rolling(window=5).mean()

        # ---- XA_3, XA_4: 长期成本轨道 ----
        df['XA_3'] = df['XA_5'].ewm(span=300, adjust=False).mean() * 1.26
        df['XA_4'] = df['XA_2'].ewm(span=200, adjust=False).mean() * 1.18

        # ---- XA_6: 120 日最高成本 ----
        df['XA_6'] = df['XA_5'].rolling(window=120).max()

        # ---- XA_7: 条件过滤器 ----
        cond = df['XA_1'] > df['XA_6'].shift(1)
        df['XA_7'] = np.where(cond, df['XA_6'], np.nan)

        # ---- XA_8: 成交额 ----
        df['XA_8'] = C * V

        # ---- XA_9: 核心综合成本（多周期量价加权）----
        def _expma_div(num: pd.Series, den: pd.Series, n1: int, n2: int) -> pd.Series:
            """num 的 n1 日 EMA 除以 den 的 n2 日 EMA。"""
            return num.ewm(span=n1, adjust=False).mean() / den.ewm(span=n2, adjust=False).mean()

        # 使用 cast 缩窄类型以通过 pyright 检查（实际运行时均为 Series）
        xa8 = cast(pd.Series, df['XA_8'])
        xa3 = cast(pd.Series, df['XA_3'])
        vol = cast(pd.Series, V)

        t1 = _expma_div(xa8, vol, 3, 3)
        t2 = _expma_div(xa8, vol, 9, 6)
        t3 = _expma_div(xa3, vol, 12, 12)
        t4 = _expma_div(xa8, vol, 24, 24)

        avg_cost = (t1 + t2 + t3 + t4) / 4.0
        df['XA_9'] = avg_cost.ewm(span=13, adjust=False).mean()

        # ---- XA_10, XA_11: 压力位 ----
        df['XA_10'] = df['XA_9'] * 1.03
        df['XA_11'] = df['XA_9'] * 1.13

        # ---- XA_12 ~ XA_15: 多层最小值过滤（支撑位）----
        df['XA_12'] = np.minimum(df['XA_7'], df['XA_3'])
        df['XA_13'] = np.minimum(df['XA_10'], df['XA_3'])
        df['XA_14'] = np.minimum(df['XA_9'], df['XA_4'])
        df['XA_15'] = np.minimum(df['XA_10'], df['XA_4'])

        # ---- XA_16: 强势突破信号 ----
        pct = C / C.shift(1)
        is_high = (C == H)
        df['XA_16'] = ((pct > 1.03) & is_high).astype(int)

        return df

    def get_summary(self, data: pd.DataFrame) -> dict | None:
        """
        从已计算的 MACDFS 数据中提取最新一个交易日的关键指标摘要。

        Args:
            data: 已调用 calculate() 计算过的 DataFrame，包含 XA_1 ~ XA_16 列。

        Returns:
            包含以下字段的字典，若无有效数据则返回 None：
            - core_cost: 核心综合成本（XA_9）
            - resistance_1: 第一压力位（XA_10）
            - resistance_2: 第二压力位（XA_11）
            - long_term_orbit_1: 长期成本轨道 1（XA_3）
            - long_term_orbit_2: 长期成本轨道 2（XA_4）
            - strong_breakout_signal: 强势突破信号（XA_16，0/1）
            - volatility_factor: 波动率因子（XA_1）
            - support_level_1: 支撑位 1（XA_12）
            - support_level_2: 支撑位 2（XA_13）
        """
        if data is None or data.empty:
            return None

        latest = data.iloc[-1]

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