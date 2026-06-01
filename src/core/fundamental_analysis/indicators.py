# -*- coding: utf-8 -*-
"""
===================================
财务指标计算
===================================

职责：
1. 从财报数据中提取并计算各项财务指标
2. 支持盈利能力、成长能力、偿债能力、运营能力、现金流五大维度
3. 支持同比/环比计算
4. 返回结构化的指标字典

依赖的财报数据：
- 财务摘要（akshare stock_financial_abstract_ths）
- 利润表（akshare stock_profit_sheet_by_report_em）
- 资产负债表（akshare stock_balance_sheet_by_report_em）
- 现金流量表（akshare stock_cash_flow_sheet_by_report_em）
"""

import logging
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FinancialIndicators:
    """
    财务指标计算器

    从原始的财报 DataFrame 中提取并计算结构化的财务指标，
    涵盖盈利能力、成长能力、偿债能力、运营能力、现金流五个维度。
    """

    # ============================================================
    # 盈利能力指标
    # ============================================================

    @staticmethod
    def calc_profitability(
        abstract_df: Optional[pd.DataFrame],
        income_df: Optional[pd.DataFrame],
        balance_df: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        """
        计算盈利能力指标

        包含：ROE、ROA、毛利率、净利率、营业利润率

        Args:
            abstract_df: 财务摘要 DataFrame
            income_df: 利润表 DataFrame
            balance_df: 资产负债表 DataFrame

        Returns:
            盈利能力指标字典
        """
        result: Dict[str, Optional[float]] = {
            "roe": None,
            "roa": None,
            "gross_profit_margin": None,
            "net_profit_margin": None,
            "operating_profit_margin": None,
        }

        # 从财务摘要中提取 ROE
        if abstract_df is not None and not abstract_df.empty:
            latest = abstract_df.iloc[0]
            result["roe"] = FinancialIndicators._safe_float(latest, "净资产收益率")
            result["gross_profit_margin"] = FinancialIndicators._safe_float(latest, "销售毛利率")
            result["net_profit_margin"] = FinancialIndicators._safe_float(latest, "销售净利率")

        # 从利润表中提取营业利润率
        if income_df is not None and not income_df.empty:
            latest = income_df.iloc[0]
            total_revenue = FinancialIndicators._safe_float(latest, "营业总收入")
            operate_profit = FinancialIndicators._safe_float(latest, "营业利润")
            if total_revenue and operate_profit and total_revenue > 0:
                result["operating_profit_margin"] = round(operate_profit / total_revenue * 100, 2)

        # 计算 ROA（总资产收益率）
        if income_df is not None and not income_df.empty and balance_df is not None and not balance_df.empty:
            income_latest = income_df.iloc[0]
            balance_latest = balance_df.iloc[0]
            net_income = FinancialIndicators._safe_float(income_latest, "净利润")
            total_assets = FinancialIndicators._safe_float(balance_latest, "*资产合计")
            if net_income and total_assets and total_assets > 0:
                result["roa"] = round(net_income / total_assets * 100, 2)

        return result

    # ============================================================
    # 成长能力指标
    # ============================================================

    @staticmethod
    def calc_growth(
        abstract_df: Optional[pd.DataFrame],
        income_df: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        """
        计算成长能力指标

        包含：营收增长率（同比）、净利润增长率（同比）、每股收益增长率

        Args:
            abstract_df: 财务摘要 DataFrame
            income_df: 利润表 DataFrame

        Returns:
            成长能力指标字典
        """
        result: Dict[str, Optional[float]] = {
            "revenue_growth_yoy": None,
            "net_profit_growth_yoy": None,
            "eps_growth_yoy": None,
        }

        # 从财务摘要中提取同比增长率
        if abstract_df is not None and not abstract_df.empty:
            latest = abstract_df.iloc[0]
            result["revenue_growth_yoy"] = FinancialIndicators._safe_float(latest, "营业总收入同比增长率")
            result["net_profit_growth_yoy"] = FinancialIndicators._safe_float(latest, "净利润同比增长率")

        # 从利润表中计算 EPS 增长率
        if income_df is not None and len(income_df) >= 2:
            current_eps = FinancialIndicators._safe_float(income_df.iloc[0], "基本每股收益")
            prev_eps = FinancialIndicators._safe_float(income_df.iloc[1], "基本每股收益")
            if current_eps and prev_eps and prev_eps != 0:
                result["eps_growth_yoy"] = round((current_eps - prev_eps) / abs(prev_eps) * 100, 2)

        return result

    # ============================================================
    # 偿债能力指标
    # ============================================================

    @staticmethod
    def calc_solvency(
        abstract_df: Optional[pd.DataFrame],
        balance_df: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        """
        计算偿债能力指标

        包含：资产负债率、流动比率、速动比率

        Args:
            abstract_df: 财务摘要 DataFrame
            balance_df: 资产负债表 DataFrame

        Returns:
            偿债能力指标字典
        """
        result: Dict[str, Optional[float]] = {
            "debt_to_asset_ratio": None,
            "current_ratio": None,
            "quick_ratio": None,
        }

        # 从财务摘要中提取资产负债率
        if abstract_df is not None and not abstract_df.empty:
            latest = abstract_df.iloc[0]
            result["debt_to_asset_ratio"] = FinancialIndicators._safe_float(latest, "资产负债率")

        # 从资产负债表中计算流动比率和速动比率
        if balance_df is not None and not balance_df.empty:
            latest = balance_df.iloc[0]
            total_cur_assets = FinancialIndicators._safe_float(latest, "流动资产合计")
            total_cur_liab = FinancialIndicators._safe_float(latest, "流动负债合计")
            inventories = FinancialIndicators._safe_float(latest, "存货")

            if total_cur_assets and total_cur_liab and total_cur_liab > 0:
                result["current_ratio"] = round(total_cur_assets / total_cur_liab, 2)
                if inventories is not None:
                    quick_assets = total_cur_assets - inventories
                    result["quick_ratio"] = round(quick_assets / total_cur_liab, 2)

        return result

    # ============================================================
    # 运营能力指标
    # ============================================================

    @staticmethod
    def calc_operational_efficiency(
        income_df: Optional[pd.DataFrame],
        balance_df: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        """
        计算运营能力指标

        包含：总资产周转率、存货周转率、应收账款周转率

        Args:
            income_df: 利润表 DataFrame
            balance_df: 资产负债表 DataFrame

        Returns:
            运营能力指标字典
        """
        result: Dict[str, Optional[float]] = {
            "total_asset_turnover": None,
            "inventory_turnover": None,
            "receivables_turnover": None,
        }

        if income_df is None or income_df.empty or balance_df is None or balance_df.empty:
            return result

        income_latest = income_df.iloc[0]
        balance_latest = balance_df.iloc[0]

        total_revenue = FinancialIndicators._safe_float(income_latest, "营业总收入")
        total_assets = FinancialIndicators._safe_float(balance_latest, "*资产合计")
        inventories = FinancialIndicators._safe_float(balance_latest, "存货")
        accounts_receiv = FinancialIndicators._safe_float(balance_latest, "应收账款")

        if total_revenue and total_assets and total_assets > 0:
            result["total_asset_turnover"] = round(total_revenue / total_assets, 2)

        if total_revenue and inventories and inventories > 0:
            result["inventory_turnover"] = round(total_revenue / inventories, 2)

        if total_revenue and accounts_receiv and accounts_receiv > 0:
            result["receivables_turnover"] = round(total_revenue / accounts_receiv, 2)

        return result

    # ============================================================
    # 现金流指标
    # ============================================================

    @staticmethod
    def calc_cashflow_quality(
        income_df: Optional[pd.DataFrame],
        cashflow_df: Optional[pd.DataFrame],
    ) -> Dict[str, Optional[float]]:
        """
        计算现金流质量指标

        包含：经营现金流/营业收入、自由现金流

        Args:
            income_df: 利润表 DataFrame
            cashflow_df: 现金流量表 DataFrame

        Returns:
            现金流指标字典
        """
        result: Dict[str, Optional[float]] = {
            "operating_cashflow_to_revenue": None,
            "free_cashflow": None,
        }

        if income_df is None or income_df.empty or cashflow_df is None or cashflow_df.empty:
            return result

        income_latest = income_df.iloc[0]
        cashflow_latest = cashflow_df.iloc[0]

        total_revenue = FinancialIndicators._safe_float(income_latest, "营业总收入")
        n_cashflow_act = FinancialIndicators._safe_float(cashflow_latest, "经营活动产生的现金流量净额")

        if n_cashflow_act and total_revenue and total_revenue > 0:
            result["operating_cashflow_to_revenue"] = round(n_cashflow_act / total_revenue * 100, 2)

        # 自由现金流 ≈ 经营活动现金流净额 - 资本支出
        # 简化处理：如果数据中有自由现金流字段则直接使用
        free_cf = FinancialIndicators._safe_float(cashflow_latest, "企业自由现金流量")
        if free_cf is not None:
            result["free_cashflow"] = free_cf
        elif n_cashflow_act is not None:
            # 简化估算：自由现金流 = 经营活动现金流净额
            result["free_cashflow"] = n_cashflow_act

        return result

    # ============================================================
    # 综合计算
    # ============================================================

    @classmethod
    def calc_all_indicators(
        cls,
        abstract_df: Optional[pd.DataFrame] = None,
        income_df: Optional[pd.DataFrame] = None,
        balance_df: Optional[pd.DataFrame] = None,
        cashflow_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        计算所有财务指标

        Args:
            abstract_df: 财务摘要 DataFrame
            income_df: 利润表 DataFrame
            balance_df: 资产负债表 DataFrame
            cashflow_df: 现金流量表 DataFrame

        Returns:
            按维度组织的指标字典：
            {
                "profitability": {...},
                "growth": {...},
                "solvency": {...},
                "operational_efficiency": {...},
                "cashflow_quality": {...},
            }
        """
        return {
            "profitability": cls.calc_profitability(abstract_df, income_df, balance_df),
            "growth": cls.calc_growth(abstract_df, income_df),
            "solvency": cls.calc_solvency(abstract_df, balance_df),
            "operational_efficiency": cls.calc_operational_efficiency(income_df, balance_df),
            "cashflow_quality": cls.calc_cashflow_quality(income_df, cashflow_df),
        }

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _safe_float(row: pd.Series, column_name: str) -> Optional[float]:
        """
        安全地从 DataFrame 行中提取浮点数

        支持多种列名匹配（模糊匹配），避免因列名细微差异导致提取失败。

        Args:
            row: DataFrame 的一行
            column_name: 目标列名

        Returns:
            提取的浮点数值，失败返回 None
        """
        # 精确匹配
        if column_name in row.index:
            return FinancialIndicators._to_float(row[column_name])

        # 模糊匹配：列名包含关键词
        for col in row.index:
            if column_name in str(col) or str(col) in column_name:
                return FinancialIndicators._to_float(row[col])

        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """将值安全转换为浮点数"""
        if value is None:
            return None
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 2)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def to_flat_dict(
        indicators: Dict[str, Dict[str, Optional[float]]]
    ) -> Dict[str, Optional[float]]:
        """
        将嵌套的指标字典扁平化为一级字典

        Args:
            indicators: calc_all_indicators 的返回结果

        Returns:
            扁平化的指标字典，如 {"roe": 15.5, "roa": 8.2, ...}
        """
        flat: Dict[str, Optional[float]] = {}
        for category, metrics in indicators.items():
            for key, value in metrics.items():
                flat[key] = value
        return flat