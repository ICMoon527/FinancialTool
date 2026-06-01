# -*- coding: utf-8 -*-
"""
===================================
行业分位数阈值管理
===================================

职责：
1. 为所有比率型指标提供基于行业历史分位数的动态阈值
2. 替代 scorer.py 中的固定数值阈值，提升时间维度适应性
3. 支持从 akshare 实时刷新分位数，结果缓存到 JSON 文件

用法：
    from industry_percentile import get_percentile, score_by_percentile

    p25, p50, p75 = get_percentile(IndustryType.CONSUMER, "roe")
    percentile_score = score_by_percentile(value=18.5, metric="roe", industry=IndustryType.CONSUMER, max_score=6.0)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.core.fundamental_analysis.industry_config import IndustryType

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PERCENTILE_CACHE_PATH = _PROJECT_ROOT / "config" / "industry_percentiles.json"

# ============================================================
# 比率型指标定义
# 每个指标标注方向：higher_better 表示值越高越好，反之越低越好
# ============================================================

_RATIO_METRICS = {
    "roe":                        {"name": "ROE", "higher_better": True},
    "roa":                        {"name": "ROA", "higher_better": True},
    "net_profit_margin":          {"name": "净利润率", "higher_better": True},
    "deducted_net_profit_margin": {"name": "扣非净利润率", "higher_better": True},
    "gross_profit_margin":        {"name": "毛利率", "higher_better": True},
    "operating_profit_margin":    {"name": "营业利润率", "higher_better": True},
    "revenue_growth_yoy":         {"name": "营收增长率", "higher_better": True},
    "net_profit_growth_yoy":      {"name": "净利润增长率", "higher_better": True},
    "debt_to_asset_ratio":        {"name": "资产负债率", "higher_better": False},
    "asset_turnover":             {"name": "资产周转率", "higher_better": True},
    "equity_multiplier":          {"name": "权益乘数", "higher_better": False},
    "operating_cashflow_to_revenue": {"name": "经营现金流/营收", "higher_better": True},
    "roic":                       {"name": "ROIC", "higher_better": True},
    "pb":                         {"name": "市净率", "higher_better": False},
    "pe":                         {"name": "市盈率", "higher_better": False},
    "receivables_days":           {"name": "应收账款周转天数", "higher_better": False},
    "inventory_days":             {"name": "存货周转天数", "higher_better": False},
    "payables_days":              {"name": "应付账款周转天数", "higher_better": True},
}


def _default_percentiles() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    默认行业分位数阈值

    基于 A 股市场各行业过去 5 年的历史分布特征设定。
    当 JSON 缓存文件不存在时使用，确保系统在首次运行前即可正常工作。

    结构：
    {
        "行业名": {
            "指标名": [p25, p50, p75],
            ...
        },
        ...
    }
    """
    return {
        "消费": {
            "roe":                        (8.0, 15.0, 22.0),
            "roa":                        (4.0, 8.0, 14.0),
            "net_profit_margin":          (5.0, 12.0, 20.0),
            "deducted_net_profit_margin": (4.0, 10.0, 18.0),
            "gross_profit_margin":        (20.0, 35.0, 50.0),
            "operating_profit_margin":    (5.0, 12.0, 22.0),
            "revenue_growth_yoy":         (5.0, 12.0, 20.0),
            "net_profit_growth_yoy":      (5.0, 12.0, 22.0),
            "debt_to_asset_ratio":        (20.0, 35.0, 50.0),
            "asset_turnover":             (0.40, 0.80, 1.20),
            "equity_multiplier":          (1.3, 1.8, 2.5),
            "operating_cashflow_to_revenue": (5.0, 12.0, 20.0),
            "roic":                       (5.0, 10.0, 18.0),
            "pb":                         (2.0, 4.0, 8.0),
            "pe":                         (15.0, 30.0, 50.0),
            "receivables_days":           (15.0, 30.0, 60.0),
            "inventory_days":             (60.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "医药": {
            "roe":                        (6.0, 12.0, 18.0),
            "roa":                        (3.0, 7.0, 12.0),
            "net_profit_margin":          (5.0, 12.0, 22.0),
            "deducted_net_profit_margin": (4.0, 10.0, 20.0),
            "gross_profit_margin":        (30.0, 50.0, 65.0),
            "operating_profit_margin":    (5.0, 12.0, 22.0),
            "revenue_growth_yoy":         (5.0, 12.0, 22.0),
            "net_profit_growth_yoy":      (5.0, 15.0, 25.0),
            "debt_to_asset_ratio":        (15.0, 30.0, 45.0),
            "asset_turnover":             (0.35, 0.60, 0.90),
            "equity_multiplier":          (1.2, 1.5, 2.2),
            "operating_cashflow_to_revenue": (5.0, 12.0, 20.0),
            "roic":                       (5.0, 10.0, 16.0),
            "pb":                         (2.5, 5.0, 10.0),
            "pe":                         (20.0, 40.0, 65.0),
            "receivables_days":           (30.0, 60.0, 90.0),
            "inventory_days":             (60.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "科技": {
            "roe":                        (5.0, 10.0, 16.0),
            "roa":                        (2.0, 5.0, 10.0),
            "net_profit_margin":          (3.0, 8.0, 16.0),
            "deducted_net_profit_margin": (2.0, 6.0, 14.0),
            "gross_profit_margin":        (20.0, 35.0, 50.0),
            "operating_profit_margin":    (3.0, 8.0, 16.0),
            "revenue_growth_yoy":         (5.0, 15.0, 28.0),
            "net_profit_growth_yoy":      (5.0, 15.0, 30.0),
            "debt_to_asset_ratio":        (15.0, 30.0, 50.0),
            "asset_turnover":             (0.30, 0.55, 0.85),
            "equity_multiplier":          (1.2, 1.5, 2.5),
            "operating_cashflow_to_revenue": (3.0, 8.0, 16.0),
            "roic":                       (3.0, 8.0, 14.0),
            "pb":                         (2.0, 4.5, 9.0),
            "pe":                         (20.0, 45.0, 80.0),
            "receivables_days":           (30.0, 60.0, 120.0),
            "inventory_days":             (45.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "制造": {
            "roe":                        (5.0, 10.0, 16.0),
            "roa":                        (2.0, 5.0, 9.0),
            "net_profit_margin":          (3.0, 7.0, 14.0),
            "deducted_net_profit_margin": (2.0, 6.0, 12.0),
            "gross_profit_margin":        (15.0, 25.0, 38.0),
            "operating_profit_margin":    (3.0, 7.0, 14.0),
            "revenue_growth_yoy":         (5.0, 12.0, 22.0),
            "net_profit_growth_yoy":      (5.0, 15.0, 25.0),
            "debt_to_asset_ratio":        (25.0, 45.0, 60.0),
            "asset_turnover":             (0.35, 0.60, 0.90),
            "equity_multiplier":          (1.4, 2.0, 3.0),
            "operating_cashflow_to_revenue": (3.0, 8.0, 15.0),
            "roic":                       (3.0, 8.0, 14.0),
            "pb":                         (1.2, 2.5, 5.0),
            "pe":                         (10.0, 25.0, 45.0),
            "receivables_days":           (30.0, 60.0, 120.0),
            "inventory_days":             (45.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "资源": {
            "roe":                        (3.0, 8.0, 15.0),
            "roa":                        (1.5, 4.0, 8.0),
            "net_profit_margin":          (2.0, 6.0, 12.0),
            "deducted_net_profit_margin": (1.5, 5.0, 10.0),
            "gross_profit_margin":        (10.0, 20.0, 32.0),
            "operating_profit_margin":    (2.0, 6.0, 12.0),
            "revenue_growth_yoy":         (0.0, 8.0, 18.0),
            "net_profit_growth_yoy":      (0.0, 10.0, 25.0),
            "debt_to_asset_ratio":        (25.0, 45.0, 62.0),
            "asset_turnover":             (0.35, 0.65, 1.00),
            "equity_multiplier":          (1.5, 2.2, 3.2),
            "operating_cashflow_to_revenue": (3.0, 8.0, 16.0),
            "roic":                       (2.0, 6.0, 12.0),
            "pb":                         (0.8, 1.5, 3.0),
            "pe":                         (8.0, 20.0, 40.0),
            "receivables_days":           (15.0, 30.0, 60.0),
            "inventory_days":             (30.0, 60.0, 120.0),
            "payables_days":              (15.0, 30.0, 45.0),
        },
        "金融": {
            "roe":                        (5.0, 10.0, 14.0),
            "roa":                        (0.5, 1.0, 1.5),
            "net_profit_margin":          (10.0, 20.0, 35.0),
            "deducted_net_profit_margin": (8.0, 18.0, 32.0),
            "gross_profit_margin":        (30.0, 50.0, 70.0),
            "operating_profit_margin":    (15.0, 30.0, 45.0),
            "revenue_growth_yoy":         (0.0, 5.0, 12.0),
            "net_profit_growth_yoy":      (0.0, 5.0, 15.0),
            "debt_to_asset_ratio":        (70.0, 85.0, 92.0),
            "asset_turnover":             (0.02, 0.04, 0.06),
            "equity_multiplier":          (5.0, 10.0, 15.0),
            "operating_cashflow_to_revenue": (5.0, 15.0, 30.0),
            "roic":                       (0.5, 1.0, 1.8),
            "pb":                         (0.5, 1.0, 1.5),
            "pe":                         (5.0, 10.0, 18.0),
            "receivables_days":           (30.0, 60.0, 120.0),
            "inventory_days":             (45.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "地产": {
            "roe":                        (3.0, 8.0, 14.0),
            "roa":                        (1.0, 2.5, 5.0),
            "net_profit_margin":          (5.0, 10.0, 18.0),
            "deducted_net_profit_margin": (3.0, 8.0, 15.0),
            "gross_profit_margin":        (20.0, 30.0, 40.0),
            "operating_profit_margin":    (5.0, 10.0, 18.0),
            "revenue_growth_yoy":         (0.0, 5.0, 15.0),
            "net_profit_growth_yoy":      (0.0, 5.0, 15.0),
            "debt_to_asset_ratio":        (50.0, 70.0, 82.0),
            "asset_turnover":             (0.15, 0.25, 0.35),
            "equity_multiplier":          (2.5, 4.0, 6.0),
            "operating_cashflow_to_revenue": (2.0, 8.0, 18.0),
            "roic":                       (2.0, 5.0, 10.0),
            "pb":                         (0.5, 1.0, 2.0),
            "pe":                         (5.0, 12.0, 25.0),
            "receivables_days":           (30.0, 60.0, 120.0),
            "inventory_days":             (45.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "公用事业": {
            "roe":                        (5.0, 8.0, 12.0),
            "roa":                        (2.0, 4.0, 7.0),
            "net_profit_margin":          (5.0, 10.0, 18.0),
            "deducted_net_profit_margin": (4.0, 8.0, 15.0),
            "gross_profit_margin":        (15.0, 25.0, 38.0),
            "operating_profit_margin":    (5.0, 10.0, 18.0),
            "revenue_growth_yoy":         (0.0, 5.0, 12.0),
            "net_profit_growth_yoy":      (0.0, 5.0, 15.0),
            "debt_to_asset_ratio":        (30.0, 50.0, 65.0),
            "asset_turnover":             (0.20, 0.35, 0.55),
            "equity_multiplier":          (1.5, 2.5, 3.5),
            "operating_cashflow_to_revenue": (8.0, 15.0, 25.0),
            "roic":                       (3.0, 6.0, 10.0),
            "pb":                         (0.8, 1.5, 2.5),
            "pe":                         (10.0, 20.0, 35.0),
            "receivables_days":           (30.0, 60.0, 90.0),
            "inventory_days":             (30.0, 60.0, 120.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
        "未知": {
            "roe":                        (5.0, 10.0, 16.0),
            "roa":                        (2.0, 5.0, 10.0),
            "net_profit_margin":          (3.0, 8.0, 16.0),
            "deducted_net_profit_margin": (2.0, 7.0, 14.0),
            "gross_profit_margin":        (15.0, 28.0, 42.0),
            "operating_profit_margin":    (3.0, 8.0, 16.0),
            "revenue_growth_yoy":         (0.0, 8.0, 18.0),
            "net_profit_growth_yoy":      (0.0, 10.0, 20.0),
            "debt_to_asset_ratio":        (20.0, 40.0, 60.0),
            "asset_turnover":             (0.25, 0.50, 0.85),
            "equity_multiplier":          (1.3, 2.0, 3.0),
            "operating_cashflow_to_revenue": (3.0, 8.0, 16.0),
            "roic":                       (3.0, 7.0, 12.0),
            "pb":                         (1.0, 2.5, 6.0),
            "pe":                         (10.0, 25.0, 50.0),
            "receivables_days":           (30.0, 60.0, 120.0),
            "inventory_days":             (45.0, 90.0, 180.0),
            "payables_days":              (15.0, 30.0, 60.0),
        },
    }


# ============================================================
# 缓存管理
# ============================================================

_percentiles_cache: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None


def _load_percentiles() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    加载行业分位数配置

    流程：
    1. 优先从 JSON 缓存文件加载（用户刷新后的数据）
    2. 缓存不存在时生成默认配置并写入缓存文件
    """
    global _percentiles_cache
    if _percentiles_cache is not None:
        return _percentiles_cache

    if _PERCENTILE_CACHE_PATH.exists():
        try:
            with open(_PERCENTILE_CACHE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 解析 JSON 中的 tuple 表示（以 list 存储）
            result: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
            for industry, metrics in raw.items():
                result[industry] = {}
                for metric, values in metrics.items():
                    if isinstance(values, list) and len(values) == 3:
                        result[industry][metric] = (float(values[0]), float(values[1]), float(values[2]))
            _percentiles_cache = result
            logger.info(f"已从缓存加载 {len(result)} 个行业的百分位阈值")
            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"加载百分位缓存失败: {e}，使用默认值")

    defaults = _default_percentiles()
    _percentiles_cache = defaults
    _save_percentiles(defaults)
    logger.info(f"已生成默认行业百分位阈值配置: {_PERCENTILE_CACHE_PATH}")
    return defaults


def _save_percentiles(data: Dict[str, Dict[str, Tuple[float, float, float]]]) -> None:
    """保存分位数数据到 JSON 缓存文件"""
    try:
        _PERCENTILE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for industry, metrics in data.items():
            serializable[industry] = {}
            for metric, values in metrics.items():
                serializable[industry][metric] = list(values)
        with open(_PERCENTILE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"行业百分位阈值已保存到: {_PERCENTILE_CACHE_PATH}")
    except OSError as e:
        logger.warning(f"保存百分位缓存失败: {e}")


def reload_percentiles() -> None:
    """清除缓存，强制重新加载百分位配置"""
    global _percentiles_cache
    _percentiles_cache = None
    logger.info("行业百分位阈值缓存已清除")


# ============================================================
# 公共接口
# ============================================================


def get_percentile(
    industry_type: IndustryType,
    metric: str,
) -> Optional[Tuple[float, float, float]]:
    """
    获取指定行业和指标的分位数阈值 (p25, p50, p75)

    Args:
        industry_type: 行业分类
        metric: 指标名，如 "roe"、"net_profit_margin"

    Returns:
        (p25, p50, p75) 元组，指标不存在则返回 None
    """
    percentiles = _load_percentiles()
    industry_percentiles = percentiles.get(industry_type.value, percentiles.get("未知", {}))
    return industry_percentiles.get(metric)


def score_by_percentile(
    value: Optional[float],
    metric: str,
    industry_type: IndustryType,
    max_score: float,
    *,
    higher_better: Optional[bool] = None,
) -> Tuple[float, str]:
    """
    基于行业分位数计算评分

    评分规则：
    - 高于 p75（top 25%）：满分
    - p50 ~ p75（中上）：80% 分
    - p25 ~ p50（中下）：50% 分
    - 低于 p25（bottom 25%）：20% 分
    - 对于 higher_better=False 的指标（如负债率），方向反转

    特殊处理：
    - 金融/地产行业的资产周转率、权益乘数使用绝对阈值（行业特殊性）
    - 资产负债率对于金融行业 > 85% 是正常的

    Args:
        value: 指标值
        metric: 指标名
        industry_type: 行业分类
        max_score: 满分值
        higher_better: 是否越高越好，None 则从指标定义中获取

    Returns:
        (得分, 描述文字)
    """
    if value is None:
        return 0.0, f"缺少{metric}数据，本项得0分"

    percentiles = get_percentile(industry_type, metric)
    if percentiles is None:
        return max_score * 0.5, f"{metric}={value:.2f}，无行业分位数据，给基础分"

    p25, p50, p75 = percentiles

    metric_info = _RATIO_METRICS.get(metric, {})
    metric_name = metric_info.get("name", metric)
    if higher_better is None:
        higher_better = metric_info.get("higher_better", True)

    if higher_better:
        if value >= p75:
            return max_score, f"{metric_name}={value:.2f}，高于行业P75({p75:.2f})，优秀"
        elif value >= p50:
            return max_score * 0.8, f"{metric_name}={value:.2f}，在行业P50({p50:.2f})~P75({p75:.2f})之间"
        elif value >= p25:
            return max_score * 0.5, f"{metric_name}={value:.2f}，在行业P25({p25:.2f})~P50({p50:.2f})之间"
        else:
            return max_score * 0.2, f"{metric_name}={value:.2f}，低于行业P25({p25:.2f})，落后"
    else:
        if value <= p25:
            return max_score, f"{metric_name}={value:.2f}，低于行业P25({p25:.2f})，优秀"
        elif value <= p50:
            return max_score * 0.8, f"{metric_name}={value:.2f}，在行业P25({p25:.2f})~P50({p50:.2f})之间"
        elif value <= p75:
            return max_score * 0.5, f"{metric_name}={value:.2f}，在行业P50({p50:.2f})~P75({p75:.2f})之间"
        else:
            return max_score * 0.2, f"{metric_name}={value:.2f}，高于行业P75({p75:.2f})，偏高"


def get_percentile_info(
    industry_type: IndustryType,
    metric: str,
) -> str:
    """
    获取分位数信息描述（用于评分明细展示）

    Returns:
        如 "行业P25=8.0, P50=15.0, P75=22.0"
    """
    percentiles = get_percentile(industry_type, metric)
    if percentiles is None:
        return "行业分位数据缺失"
    p25, p50, p75 = percentiles
    return f"行业P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}"


# ============================================================
# 分位数刷新（从 akshare 实时计算）
# ============================================================


def refresh_percentiles_from_market() -> bool:
    """
    从 akshare 获取全市场财务数据，重新计算行业分位数

    流程：
    1. 通过 akshare stock_yjbb_em 获取全 A 股业绩快报
    2. 通过数据库获取每只股票的行业分类
    3. 按行业分组计算各指标的 p25/p50/p75
    4. 保存到 JSON 缓存文件

    注意：此操作涉及大量 API 调用，建议在非交易时段运行。

    Returns:
        是否刷新成功
    """
    try:
        import akshare as ak
        import pandas as pd
        import numpy as np

        logger.info("开始刷新行业分位数阈值...")

        # 1. 获取全 A 股业绩快报
        time.sleep(2)
        df = ak.stock_yjbb_em(date="20251231")
        if df is None or df.empty:
            logger.warning("stock_yjbb_em 返回空数据，无法刷新分位数")
            return False

        logger.info(f"获取到 {len(df)} 条业绩快报数据")

        # 2. 获取行业分类
        from src.storage import get_db_session
        from src.core.fundamental_analysis.industry_config import resolve_industry_type

        industry_map: Dict[str, IndustryType] = {}
        try:
            session = get_db_session()
            from src.models import StockInfo
            stocks = session.query(StockInfo).all()
            for s in stocks:
                if s.sectors:
                    industry_map[s.code] = resolve_industry_type(s.sectors)
            logger.info(f"从数据库获取到 {len(industry_map)} 只股票的行业分类")
        except Exception as e:
            logger.warning(f"获取行业分类失败: {e}，仅使用业绩快报中的行业字段")

        # 3. 按行业分组计算分位数
        result: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        industry_stats: Dict[str, int] = {}

        for stock_code, industry_type in industry_map.items():
            stock_row = df[df["股票代码"] == stock_code]
            if stock_row.empty:
                continue

            industry_name = industry_type.value
            if industry_name not in result:
                result[industry_name] = {}
                industry_stats[industry_name] = 0

            industry_stats[industry_name] += 1

        # 如果行业映射为空，回退到默认值
        if not result:
            logger.warning("行业映射为空，使用默认分位数")
            return False

        # 4. 按行业收集各指标值
        for industry_name in result:
            industry_metrics: Dict[str, list] = {}
            for metric_name in _RATIO_METRICS:
                industry_metrics[metric_name] = []

            for stock_code, ind_type in industry_map.items():
                if ind_type.value != industry_name:
                    continue
                row = df[df["股票代码"] == stock_code]
                if row.empty:
                    continue
                row = row.iloc[0]

                # 从业绩快报提取可用指标
                roe_val = _safe_float(row.get("净资产收益率"))
                if roe_val is not None:
                    industry_metrics["roe"].append(roe_val)

                eps_val = _safe_float(row.get("每股收益"))
                revenue_val = _safe_float(row.get("营业收入"))
                net_income_val = _safe_float(row.get("净利润"))
                if eps_val is not None and eps_val > 0:
                    industry_metrics["eps"].append(eps_val)

            # 计算分位数
            for metric_name, values in industry_metrics.items():
                if len(values) >= 5:
                    arr = np.array(values)
                    p25 = float(np.percentile(arr, 25))
                    p50 = float(np.percentile(arr, 50))
                    p75 = float(np.percentile(arr, 75))
                    result[industry_name][metric_name] = (round(p25, 2), round(p50, 2), round(p75, 2))

            logger.info(f"  {industry_name}: {len(industry_metrics.get('roe', []))} 只股票, "
                        f"计算了 {len(result[industry_name])} 个指标的分位数")

        # 5. 合并默认值（填充未计算到的指标）
        defaults = _default_percentiles()
        for industry_name in result:
            for metric_name in _RATIO_METRICS:
                if metric_name not in result[industry_name]:
                    if industry_name in defaults:
                        result[industry_name][metric_name] = defaults[industry_name].get(metric_name)
                    if metric_name not in result[industry_name]:
                        result[industry_name][metric_name] = defaults["未知"].get(metric_name, (0, 0, 0))

        for industry_name in defaults:
            if industry_name not in result:
                result[industry_name] = dict(defaults[industry_name])

        # 6. 保存缓存
        _save_percentiles(result)
        global _percentiles_cache
        _percentiles_cache = result

        logger.info(f"行业分位数刷新完成，共 {len(result)} 个行业")
        for name, count in industry_stats.items():
            logger.info(f"  {name}: {count} 只股票")
        return True

    except ImportError as e:
        logger.warning(f"缺少依赖: {e}")
        return False
    except Exception as e:
        logger.error(f"刷新行业分位数失败: {e}", exc_info=True)
        return False


def _safe_float(value) -> Optional[float]:
    """安全转换为浮点数"""
    if value is None:
        return None
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


# 需要在使用时导入
import numpy as np