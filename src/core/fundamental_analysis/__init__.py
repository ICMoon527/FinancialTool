# -*- coding: utf-8 -*-
"""
===================================
公司财报基本面分析模块
===================================

职责：
1. 从数据源获取公司财报数据（资产负债表、利润表、现金流量表）
2. 计算各项财务指标（盈利能力、成长能力、偿债能力、运营能力、现金流）
3. 根据财务指标核算基本面评分
4. 提供统一的分析入口

模块结构：
- fetcher.py: 财报数据抓取器
- indicators.py: 财务指标计算
- scorer.py: 基本面评分算法
- industry_config.py: 行业分类与阈值配置
- industry_percentile.py: 行业分位数动态阈值
- analyzer.py: 对外主入口，编排抓取→计算→评分流程
"""

from src.core.fundamental_analysis.fetcher import FinancialReportFetcher
from src.core.fundamental_analysis.indicators import FinancialIndicators
from src.core.fundamental_analysis.analyzer import FundamentalAnalyzer
from src.core.fundamental_analysis.scorer import FinancialScorer, FundamentalScorer
from src.core.fundamental_analysis.industry_config import (
    IndustryType,
    get_industry_pe_range,
    load_pe_ranges,
    reload_pe_config,
    INDUSTRY_PE_CONFIG_PATH,
)
from src.core.fundamental_analysis.industry_percentile import (
    score_by_percentile,
    get_percentile,
    get_percentile_info,
    reload_percentiles,
    refresh_percentiles_from_market,
)
from src.core.fundamental_analysis.scorer_config import (
    get_scorer_config,
    reload_scorer_config,
    ScorerConfig,
)

__all__ = [
    "FinancialReportFetcher",
    "FinancialIndicators",
    "FundamentalAnalyzer",
    "FinancialScorer",
    "FundamentalScorer",
    "IndustryType",
    "get_industry_pe_range",
    "load_pe_ranges",
    "reload_pe_config",
    "INDUSTRY_PE_CONFIG_PATH",
    "score_by_percentile",
    "get_percentile",
    "get_percentile_info",
    "reload_percentiles",
    "refresh_percentiles_from_market",
    "get_scorer_config",
    "reload_scorer_config",
    "ScorerConfig",
]