# -*- coding: utf-8 -*-
"""
===================================
评分器配置加载器
===================================

职责：
1. 从 YAML 配置文件加载评分权重、阈值和完备性要求
2. 提供默认值，确保配置文件缺失时评分器仍可运行
3. 支持运行时重新加载配置
"""

import logging
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# 配置文件路径
SCORER_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "scorer_config.yaml"


def _load_raw_config() -> dict:
    """从 YAML 文件加载原始配置，如果文件不存在则返回空字典"""
    if not SCORER_CONFIG_PATH.exists():
        logger.warning(f"评分器配置文件不存在: {SCORER_CONFIG_PATH}，将使用默认值")
        return {}
    try:
        with open(SCORER_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"加载评分器配置文件失败: {e}，将使用默认值")
        return {}


@dataclass
class SubItemConfig:
    """子项配置"""
    weight: float = 0.0
    thresholds: List[float] = field(default_factory=list)


@dataclass
class IntegrityConfig:
    """三表勾稽配置"""
    retained_earnings: SubItemConfig = field(default_factory=SubItemConfig)
    ocf_vs_ni: SubItemConfig = field(default_factory=SubItemConfig)
    cash_change: SubItemConfig = field(default_factory=SubItemConfig)


@dataclass
class InterestCoverageConfig:
    """利息保障倍数配置"""
    weight: float = 5.0
    thresholds: List[float] = field(default_factory=lambda: [3.0, 2.0])
    zero_interest_debt_ratio_low: float = 30.0
    zero_interest_score_no_debt: float = 5.0
    zero_interest_score_low_leverage: float = 4.0
    zero_interest_score_has_debt: float = 3.0
    zero_interest_score_capitalized: float = 1.0


@dataclass
class CashEarningsConfig:
    """盈利与现金流配置"""
    ebit_positive: SubItemConfig = field(default_factory=SubItemConfig)
    interest_coverage: InterestCoverageConfig = field(default_factory=InterestCoverageConfig)
    roic: SubItemConfig = field(default_factory=SubItemConfig)
    fcf_positive: SubItemConfig = field(default_factory=SubItemConfig)
    fcf_vs_ni: SubItemConfig = field(default_factory=SubItemConfig)
    fcf_vs_dividend: SubItemConfig = field(default_factory=SubItemConfig)
    high_growth_threshold: float = 20.0


@dataclass
class EfficiencyConfig:
    """营运效率配置"""
    receivables: SubItemConfig = field(default_factory=SubItemConfig)
    inventory: SubItemConfig = field(default_factory=SubItemConfig)
    payables: SubItemConfig = field(default_factory=SubItemConfig)
    fin_re_base_score: float = 2.5
    liquor_base_reserve_threshold: float = 180.0
    liquor_abundant_threshold: float = 730.0


@dataclass
class DupontConfig:
    """杜邦分析配置"""
    profit_margin: SubItemConfig = field(default_factory=SubItemConfig)
    profit_quality: SubItemConfig = field(default_factory=SubItemConfig)
    asset_turnover: SubItemConfig = field(default_factory=SubItemConfig)
    leverage: SubItemConfig = field(default_factory=SubItemConfig)
    deducted_weight: float = 6.0
    fallback_weight: float = 4.0
    missing_deducted_score: float = 2.0
    fallback_roe_high: float = 20.0
    fallback_roe_mid: float = 10.0
    fallback_score_high: float = 8.0
    fallback_score_mid: float = 5.0
    fallback_score_low: float = 2.0


@dataclass
class ValuationConfig:
    """估值配置"""
    pe_industry: SubItemConfig = field(default_factory=SubItemConfig)
    dcf: SubItemConfig = field(default_factory=SubItemConfig)
    pb: SubItemConfig = field(default_factory=SubItemConfig)
    premium_ratio: float = 1.2
    dcf_range_ratio_limit: float = 5.0
    dcf_price_low_ratio: float = 3.0
    dcf_price_high_ratio: float = 0.3


@dataclass
class DataCompletenessConfig:
    """数据完备性配置"""
    min_ratio: float = 0.6
    required_fields: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ScorerConfig:
    """评分器完整配置"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "integrity": 20, "cash_earnings": 30, "efficiency": 15,
        "dupont": 20, "valuation": 15,
    })
    rating_thresholds: List[float] = field(default_factory=lambda: [85, 70, 60])
    rating_labels: List[str] = field(default_factory=lambda: ["优秀", "良好", "一般", "较差"])
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    cash_earnings: CashEarningsConfig = field(default_factory=CashEarningsConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    dupont: DupontConfig = field(default_factory=DupontConfig)
    valuation: ValuationConfig = field(default_factory=ValuationConfig)
    data_completeness: DataCompletenessConfig = field(default_factory=DataCompletenessConfig)


def _parse_sub_item(raw: dict, default_weight: float, default_thresholds: List[float]) -> SubItemConfig:
    """从原始配置字典解析子项配置"""
    return SubItemConfig(
        weight=float(raw.get("weight", default_weight)),
        thresholds=[float(t) for t in raw.get("thresholds", default_thresholds)],
    )


def load_scorer_config() -> ScorerConfig:
    """加载评分器配置，缺失时使用默认值"""
    raw = _load_raw_config()

    config = ScorerConfig()

    # 权重
    if "weights" in raw:
        config.weights = {
            k: float(v) for k, v in raw["weights"].items()
        }

    # 评级阈值
    if "rating" in raw:
        r = raw["rating"]
        config.rating_thresholds = [float(t) for t in r.get("thresholds", [85, 70, 60])]
        config.rating_labels = r.get("labels", ["优秀", "良好", "一般", "较差"])

    # 三表勾稽
    if "integrity" in raw:
        ri = raw["integrity"]
        config.integrity.retained_earnings = _parse_sub_item(
            ri.get("retained_earnings", {}), 5.0, [0.05, 0.15, 0.30])
        config.integrity.ocf_vs_ni = _parse_sub_item(
            ri.get("ocf_vs_ni", {}), 10.0, [1.0, 0.8, 0.5])
        config.integrity.cash_change = _parse_sub_item(
            ri.get("cash_change", {}), 5.0, [0.02, 0.10, 0.25])

    # 盈利与现金流
    if "cash_earnings" in raw:
        rc = raw["cash_earnings"]
        config.cash_earnings.ebit_positive = _parse_sub_item(
            rc.get("ebit_positive", {}), 5.0, [])
        config.cash_earnings.roic = _parse_sub_item(
            rc.get("roic", {}), 5.0, [])
        config.cash_earnings.fcf_positive = _parse_sub_item(
            rc.get("fcf_positive", {}), 5.0, [])
        config.cash_earnings.fcf_vs_ni = _parse_sub_item(
            rc.get("fcf_vs_ni", {}), 5.0, [1.0, 0.8])
        config.cash_earnings.fcf_vs_dividend = _parse_sub_item(
            rc.get("fcf_vs_dividend", {}), 5.0, [])

        # 利息保障倍数
        ic_raw = rc.get("interest_coverage", {})
        zi = ic_raw.get("zero_interest", {})
        config.cash_earnings.interest_coverage = InterestCoverageConfig(
            weight=float(ic_raw.get("weight", 5.0)),
            thresholds=[float(t) for t in ic_raw.get("thresholds", [3.0, 2.0])],
            zero_interest_debt_ratio_low=float(zi.get("debt_ratio_low", 30.0)),
            zero_interest_score_no_debt=float(zi.get("score_no_debt", 5.0)),
            zero_interest_score_low_leverage=float(zi.get("score_low_leverage", 4.0)),
            zero_interest_score_has_debt=float(zi.get("score_has_debt", 3.0)),
            zero_interest_score_capitalized=float(zi.get("score_capitalized", 1.0)),
        )

        config.cash_earnings.high_growth_threshold = float(
            rc.get("high_growth_threshold", 20.0))

    # 营运效率
    if "efficiency" in raw:
        re = raw["efficiency"]
        config.efficiency.receivables = _parse_sub_item(
            re.get("receivables", {}), 5.0, [])
        config.efficiency.inventory = _parse_sub_item(
            re.get("inventory", {}), 5.0, [])
        config.efficiency.payables = _parse_sub_item(
            re.get("payables", {}), 5.0, [])
        config.efficiency.fin_re_base_score = float(
            re.get("fin_re_base_score", 2.5))
        liquor = re.get("inventory", {}).get("liquor", {})
        config.efficiency.liquor_base_reserve_threshold = float(
            liquor.get("base_reserve_threshold", 180.0))
        config.efficiency.liquor_abundant_threshold = float(
            liquor.get("abundant_threshold", 730.0))

    # 杜邦
    if "dupont" in raw:
        rd = raw["dupont"]
        config.dupont.profit_margin = _parse_sub_item(
            rd.get("profit_margin", {}), 6.0, [])
        config.dupont.profit_quality = _parse_sub_item(
            rd.get("profit_quality", {}), 4.0, [5.0, 15.0, 30.0])
        config.dupont.asset_turnover = _parse_sub_item(
            rd.get("asset_turnover", {}), 5.0, [])
        config.dupont.leverage = _parse_sub_item(
            rd.get("leverage", {}), 5.0, [])
        config.dupont.deducted_weight = float(
            rd.get("profit_margin", {}).get("deducted_weight", 6.0))
        config.dupont.fallback_weight = float(
            rd.get("profit_margin", {}).get("fallback_weight", 4.0))
        config.dupont.missing_deducted_score = float(
            rd.get("profit_quality", {}).get("missing_deducted_score", 2.0))
        fb = rd.get("fallback", {})
        config.dupont.fallback_roe_high = float(fb.get("roe_high", 20.0))
        config.dupont.fallback_roe_mid = float(fb.get("roe_mid", 10.0))
        config.dupont.fallback_score_high = float(fb.get("score_high", 8.0))
        config.dupont.fallback_score_mid = float(fb.get("score_mid", 5.0))
        config.dupont.fallback_score_low = float(fb.get("score_low", 2.0))

    # 估值
    if "valuation" in raw:
        rv = raw["valuation"]
        config.valuation.pe_industry = _parse_sub_item(
            rv.get("pe_industry", {}), 5.0, [])
        config.valuation.dcf = _parse_sub_item(
            rv.get("dcf", {}), 5.0, [])
        config.valuation.pb = _parse_sub_item(
            rv.get("pb", {}), 5.0, [])
        config.valuation.premium_ratio = float(
            rv.get("pe_industry", {}).get("premium_ratio", 1.2))
        config.valuation.dcf_range_ratio_limit = float(
            rv.get("dcf", {}).get("range_ratio_limit", 5.0))
        config.valuation.dcf_price_low_ratio = float(
            rv.get("dcf", {}).get("price_low_ratio", 3.0))
        config.valuation.dcf_price_high_ratio = float(
            rv.get("dcf", {}).get("price_high_ratio", 0.3))

    # 数据完备性
    if "data_completeness" in raw:
        rdc = raw["data_completeness"]
        config.data_completeness.min_ratio = float(
            rdc.get("min_ratio", 0.6))
        config.data_completeness.required_fields = rdc.get("required_fields", {})

    return config


# 全局单例
_config: Optional[ScorerConfig] = None


def get_scorer_config() -> ScorerConfig:
    """获取评分器配置单例"""
    global _config
    if _config is None:
        _config = load_scorer_config()
    return _config


def reload_scorer_config() -> ScorerConfig:
    """重新加载评分器配置"""
    global _config
    _config = load_scorer_config()
    logger.info("评分器配置已重新加载")
    return _config


def _make_sub_default(weight: float, thresholds: List[float]) -> SubItemConfig:
    """创建默认子项配置"""
    return SubItemConfig(weight=weight, thresholds=list(thresholds))


def _get_default_threshold_score(
    value: float,
    thresholds: List[float],
    scores: List[float],
) -> float:
    """
    根据阈值阶梯计算分数

    Args:
        value: 待评价值
        thresholds: 阈值列表（升序），如 [0.05, 0.15, 0.30]
        scores: 对应分数列表，长度 = len(thresholds) + 1
                如 [5, 4, 2, 0] 表示 value < 0.05 → 5, value < 0.15 → 4, ...
    """
    for i, t in enumerate(thresholds):
        if value < t:
            return scores[i]
    return scores[-1]