# -*- coding: utf-8 -*-
"""
===================================
基本面分析器 - 对外主入口
===================================

职责：
1. 协调财报数据抓取 → 财务指标计算 → 基本面评分 的完整流程
2. 提供统一的 analyze() 接口
3. 返回结构化的分析结果

数据补齐说明：
- 自动从已有报表中提取留存收益变动、现金余额、每股净资产等字段
- 通过 DataFetcherManager 获取实时行情（股价、市值、PB）
- 通过 akshare 获取分红数据
- 通过行业配置获取行业合理 PE 区间
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from src.core.fundamental_analysis.fetcher import FinancialReportFetcher
from src.core.fundamental_analysis.indicators import FinancialIndicators
from src.core.fundamental_analysis.scorer import FinancialScorer
from src.core.fundamental_analysis.industry_config import (
    resolve_industry_type,
    get_industry_pe_range,
    IndustryType,
)

logger = logging.getLogger(__name__)


# 重资产行业列表（制造、资源、地产、公用事业）
_ASSET_HEAVY_INDUSTRIES = {
    IndustryType.MANUFACTURING,
    IndustryType.RESOURCE,
    IndustryType.REAL_ESTATE,
    IndustryType.UTILITY,
}


class FundamentalAnalyzer:
    """
    基本面分析器

    编排财报数据抓取、指标计算、评分的一站式分析流程。

    用法：
        analyzer = FundamentalAnalyzer()
        result = analyzer.analyze("600519")
    """

    def __init__(self):
        self.fetcher = FinancialReportFetcher()

    def analyze(self, stock_code: str) -> Dict[str, Any]:
        """
        对指定股票进行基本面分析

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            分析结果字典
        """
        logger.info(f"开始对 {stock_code} 进行基本面分析...")

        result: Dict[str, Any] = {
            "stock_code": stock_code,
            "analysis_time": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "reports": {},
            "indicators": {},
            "score": None,
        }

        try:
            # 获取财报数据
            reports = self.fetcher.fetch_all_reports(stock_code)
            result["reports"] = {
                "abstract": self._summarize_df(reports["abstract"]),
                "income": self._summarize_df(reports["income"]),
                "balance": self._summarize_df(reports["balance"]),
                "cashflow": self._summarize_df(reports["cashflow"]),
            }

            if all(v is None for v in [reports["abstract"], reports["income"], reports["balance"], reports["cashflow"]]):
                result["error"] = f"未能获取 {stock_code} 的任何财报数据"
                logger.warning(result["error"])
                return result

            # 计算财务指标
            indicators = FinancialIndicators.calc_all_indicators(
                abstract_df=reports["abstract"],
                income_df=reports["income"],
                balance_df=reports["balance"],
                cashflow_df=reports["cashflow"],
            )
            result["indicators"] = indicators

            # 基本面评分
            try:
                sectors = self._lookup_sectors(stock_code)
                if not sectors:
                    sectors = None
                    industry_type = IndustryType.UNKNOWN
                else:
                    industry_type = resolve_industry_type(sectors)
                    logger.info(f"{stock_code} 行业标签: {sectors[:6]}, 解析为: {industry_type.value}")

                # 构建 scorer 所需数据（含市场数据 + 分红 + 报表衍生字段）
                scorer_data = self._build_scorer_data(indicators, reports, stock_code, industry_type)
                scorer = FinancialScorer(scorer_data, sectors=sectors)
                total, module_scores, reasons = scorer.full_score()
                result["score"] = {
                    "total_score": total,
                    "dimension_scores": module_scores,
                    "industry": scorer.industry_type.value,
                    "rating": scorer._rating(total),
                    "reasons": reasons,
                }
                logger.info(f"{stock_code} 基本面评分完成: 总分={total:.1f}, 行业={scorer.industry_type.value}")
            except Exception as e:
                logger.error(f"{stock_code} 基本面评分失败: {e}", exc_info=True)
                result["score"] = None

            result["success"] = True
            logger.info(f"{stock_code} 基本面分析完成")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"{stock_code} 基本面分析失败: {e}", exc_info=True)

        return result

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _lookup_sectors(stock_code: str) -> Optional[List[str]]:
        try:
            from src.storage import get_db
            from sqlalchemy import text

            db = get_db()
            with db.get_session() as session:
                row = session.execute(
                    text("SELECT sectors FROM stock_sector WHERE code = :code"),
                    {"code": stock_code},
                ).fetchone()
            if row and row[0]:
                sectors = json.loads(row[0])
                if sectors:
                    return sectors
            return None
        except Exception as e:
            logger.warning(f"查询 {stock_code} 行业标签失败: {e}")
            return None

    @staticmethod
    def _fetch_market_data(stock_code: str) -> Dict[str, Any]:
        """
        获取实时行情数据

        Returns:
            包含 current_price, market_cap, pb 的字典
        """
        data = {}
        try:
            from data_provider.base import DataFetcherManager

            manager = DataFetcherManager()
            quote = manager.get_realtime_quote(stock_code)
            if quote:
                data["current_price"] = getattr(quote, "price", None)
                data["market_cap"] = getattr(quote, "total_mv", None)
                data["pb"] = getattr(quote, "pb_ratio", None)
                logger.info(f"{stock_code} 实时行情获取成功: price={data.get('current_price')}, "
                            f"mv={data.get('market_cap')}, pb={data.get('pb')}")
            else:
                logger.warning(f"{stock_code} 实时行情为空")
        except Exception as e:
            logger.warning(f"获取 {stock_code} 实时行情失败: {e}")
        return data

    @staticmethod
    def _fetch_dividend_data(stock_code: str) -> Dict[str, Any]:
        """
        获取分红数据：最新一期分红总额

        Returns:
            包含 dividends 的字典
        """
        data = {}
        try:
            import akshare as ak

            time.sleep(1)
            df = ak.stock_fhps_detail_ths(symbol=stock_code)
            if df is not None and not df.empty:
                if "分红总额" in df.columns:
                    # 从后往前找（df 从旧到新排列），取最新有效分红总额
                    for _, row in df[::-1].iterrows():
                        val = row.get("分红总额")
                        if val is not None and val not in ("", "-", 0):
                            from src.core.fundamental_analysis.fetcher import _parse_ths_value
                            parsed = _parse_ths_value(val)
                            if parsed and parsed > 0:
                                data["dividends"] = parsed
                                logger.info(f"{stock_code} 最新分红总额: {parsed}")
                                break
            if "dividends" not in data:
                logger.info(f"{stock_code} 未找到有效分红数据")
        except Exception as e:
            logger.warning(f"获取 {stock_code} 分红数据失败: {e}")
        return data

    @staticmethod
    def _calc_retained_earnings_change(balance_df) -> Optional[float]:
        """计算留存收益变动 = 本期未分配利润 - 上期未分配利润"""
        if balance_df is None or len(balance_df) < 2:
            return None
        try:
            row0 = balance_df.iloc[0]
            row1 = balance_df.iloc[1]
            from src.core.fundamental_analysis.indicators import FinancialIndicators
            cur = FinancialIndicators._safe_float(row0, "未分配利润")
            prev = FinancialIndicators._safe_float(row1, "未分配利润")
            if cur is not None and prev is not None:
                return cur - prev
        except Exception as e:
            logger.warning(f"计算留存收益变动失败: {e}")
        return None

    @staticmethod
    def _get_total_debt(balance_df) -> Optional[float]:
        """估算总负债 = 短期借款 + 长期借款（或直接用负债合计）"""
        if balance_df is None or balance_df.empty:
            return None
        try:
            row = balance_df.iloc[0]
            from src.core.fundamental_analysis.indicators import FinancialIndicators
            st = FinancialIndicators._safe_float(row, "短期借款")
            lt = FinancialIndicators._safe_float(row, "长期借款")
            if st is not None and lt is not None:
                return st + lt
            total_liab = FinancialIndicators._safe_float(row, "*负债合计")
            return total_liab
        except Exception as e:
            logger.warning(f"获取总负债失败: {e}")
        return None

    @classmethod
    def _calc_simple_dcf(
        cls,
        cashflow_df,
        data: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        """
        简化 2 阶段 DCF 估值

        假设：
        - WACC = 10%（标准股权成本）
        - 终值增长率 = 3%
        - 预测期 = 5 年
        - 增长率：优先用营收增长率（revenue_growth_yoy），否则默认 5%

        Returns:
            (dcf_low, dcf_high) 每股价值区间，或 None
        """
        fcf = data.get("fcf")
        if fcf is None or fcf <= 0:
            return None

        # 用营收增长率作为FCF增长预期（比季度FCF趋势更稳定）
        revenue_growth = data.get("revenue_growth_yoy")
        if revenue_growth is not None and revenue_growth > 0:
            avg_growth = min(revenue_growth / 100, 0.20)
        else:
            avg_growth = 0.05
        avg_growth = max(0.02, min(0.20, avg_growth))

        wacc = 0.10
        terminal_growth = 0.03
        years = 5

        market_cap = data.get("market_cap")
        current_price = data.get("current_price")
        shares = None
        if market_cap is not None and current_price is not None and current_price > 0:
            shares = market_cap / current_price

        cash = data.get("monetary_cash", 0) or 0
        debt = data.get("total_debt", 0) or 0

        def _dcf_value(g: float) -> Optional[float]:
            present = 0.0
            for y in range(1, years + 1):
                fcf_y = fcf * (1 + g) ** y
                present += fcf_y / (1 + wacc) ** y
            terminal = fcf * (1 + g) ** (years + 1) * (1 + terminal_growth) / (wacc - terminal_growth)
            terminal /= (1 + wacc) ** years
            enterprise_value = present + terminal
            equity_value = enterprise_value + cash - debt
            if shares and shares > 0:
                return equity_value / shares
            return None

        g_low = max(0.02, avg_growth * 0.6)
        g_high = min(0.20, avg_growth * 1.4)
        if g_low >= g_high:
            g_low, g_high = max(0.02, avg_growth - 0.02), min(0.20, avg_growth + 0.02)

        low = _dcf_value(g_low)
        high = _dcf_value(g_high)
        if low is not None and high is not None and low > 0 and high > 0:
            logger.info(f"DCF 估值: 增长率区间[{g_low:.1%}, {g_high:.1%}], "
                        f"每股[{low:.2f}, {high:.2f}]")
            return (round(low, 2), round(high, 2))
        return None

    @classmethod
    def _build_scorer_data(
        cls,
        indicators: Dict[str, Dict[str, Optional[float]]],
        reports: Dict[str, Any],
        stock_code: str,
        industry_type: IndustryType,
    ) -> Dict[str, Any]:
        """
        从指标、原始报表、市场数据、分红数据构建完整的 scorer data 字典

        步骤：
        1. 从已有报表提取明细字段
        2. 计算衍生字段（留存收益变动、资产周转率等）
        3. 获取实时行情（股价、市值、PB）
        4. 获取分红数据
        5. 注入行业PE配置
        """
        data: Dict[str, Any] = {}

        # =========================================
        # 1. 从 indicators 聚合指标展开
        # =========================================
        flat = FinancialIndicators.to_flat_dict(indicators)
        data.update(flat)

        income_df = reports.get("income")
        balance_df = reports.get("balance")
        cashflow_df = reports.get("cashflow")
        abstract_df = reports.get("abstract")

        # =========================================
        # 2. 从原始利润表补充
        # =========================================
        if income_df is not None and not income_df.empty:
            row = income_df.iloc[0]
            def _sf(col):
                return FinancialIndicators._safe_float(row, col)

            data.setdefault("revenue", _sf("营业总收入"))
            data.setdefault("net_income", _sf("净利润"))
            data["ebit"] = _sf("营业利润")
            # 利息费用优先用"其中：利息费用"，回退到"财务费用"
            interest = _sf("其中：利息费用")
            if interest is None:
                interest = _sf("财务费用")
            data["interest_expense"] = interest
            data["tax_expense"] = _sf("所得税费用")
            data["cost_of_goods_sold"] = _sf("营业成本")
            data["eps"] = _sf("基本每股收益")

            # 扣非净利润（用于判断利润可持续性）
            data["deducted_net_profit"] = _sf("扣除非经常性损益后的净利润")

            # 其他综合收益（用于留存收益变动校验，补偿净利润-分配之外的调整项）
            data["other_comprehensive_income"] = _sf("归属母公司所有者的其他综合收益")

            # 尝试补充利息资本化金额（在附注中）
            data["interest_capitalized"] = _sf("利息资本化")

        # =========================================
        # 3. 从原始资产负债表补充
        # =========================================
        if balance_df is not None and not balance_df.empty:
            row = balance_df.iloc[0]
            def _sf(col):
                return FinancialIndicators._safe_float(row, col)

            data["avg_receivables"] = _sf("应收账款")
            data["avg_inventory"] = _sf("存货")
            data["avg_payables"] = _sf("应付账款")
            data["total_assets"] = _sf("*资产合计")
            data["total_liabilities"] = _sf("*负债合计")
            data["equity"] = _sf("*所有者权益（或股东权益）合计")
            data["monetary_cash"] = _sf("货币资金")
            data["total_debt"] = cls._get_total_debt(balance_df)

            # 留存收益变动（两期未分配利润差值）
            re_change = cls._calc_retained_earnings_change(balance_df)
            if re_change is not None:
                data["retained_earnings_change"] = re_change

        # =========================================
        # 4. 从原始现金流量表补充
        # =========================================
        if cashflow_df is not None and not cashflow_df.empty:
            row0 = cashflow_df.iloc[0]
            def _sf(col):
                return FinancialIndicators._safe_float(row0, col)

            data.setdefault("operating_cash_flow", _sf("经营活动产生的现金流量净额"))
            data["capex"] = _sf("购建固定资产、无形资产和其他长期资产支付的现金")
            # 现金净增加额
            data["cash_change_from_cf"] = _sf("现金及现金等价物净增加额")
            # 期末现金余额
            data["cash_balance_end"] = _sf("期末现金及现金等价物余额")
            # 期初现金余额（上一期期末）
            if len(cashflow_df) >= 2:
                row1 = cashflow_df.iloc[1]
                data["cash_balance_start"] = FinancialIndicators._safe_float(row1, "期末现金及现金等价物余额")

        # =========================================
        # 5. 从财务摘要补充
        # =========================================
        if abstract_df is not None and not abstract_df.empty:
            row = abstract_df.iloc[0]
            def _sf(col):
                return FinancialIndicators._safe_float(row, col)

            data.setdefault("roe", _sf("净资产收益率"))
            data.setdefault("net_profit_margin", _sf("销售净利率"))
            data.setdefault("revenue", _sf("营业总收入"))
            data.setdefault("net_income", _sf("净利润"))
            data["book_value_per_share"] = _sf("每股净资产")
            data.setdefault("eps", _sf("基本每股收益"))

        # =========================================
        # 6. 计算衍生指标
        # =========================================
        revenue = data.get("revenue")
        total_assets = data.get("total_assets")
        equity = data.get("equity")
        operating_cf = data.get("operating_cash_flow")
        capex = data.get("capex")

        # 总资产周转率 = 营收 / 总资产
        if revenue is not None and total_assets is not None and total_assets > 0:
            data["asset_turnover"] = round(revenue / total_assets, 4)

        # 权益乘数 = 总资产 / 净资产
        if total_assets is not None and equity is not None and equity > 0:
            data["equity_multiplier"] = round(total_assets / equity, 4)

        # 自由现金流 = 经营现金流 - 资本支出
        if operating_cf is not None:
            if capex is not None:
                data["fcf"] = operating_cf - capex
            else:
                data["fcf"] = operating_cf

        # 净经营资产 = 总负债 + 净资产 - 货币资金（用于 ROIC 分母）
        total_liab = data.get("total_liabilities")
        monetary_cash = data.get("monetary_cash")
        if total_assets is not None and monetary_cash is not None:
            data["net_operating_assets"] = total_assets - monetary_cash
        elif total_liab is not None and equity is not None and monetary_cash is not None:
            data["net_operating_assets"] = total_liab + equity - monetary_cash

        # 简化 DCF 估值（5年FCF折现 + 终值）
        dcf_result = cls._calc_simple_dcf(cashflow_df, data)
        if dcf_result:
            data["dcf_low"], data["dcf_high"] = dcf_result

        # 是否为重资产
        data["is_asset_heavy"] = industry_type in _ASSET_HEAVY_INDUSTRIES

        # =========================================
        # 7. 获取实时行情数据
        # =========================================
        market_data = cls._fetch_market_data(stock_code)
        data.update(market_data)

        # =========================================
        # 8. 获取分红数据（补齐分红总额）
        # =========================================
        div_data = cls._fetch_dividend_data(stock_code)
        data.update(div_data)

        # 分红+回购总额
        dividends = data.get("dividends", 0) or 0
        buyback = data.get("share_buyback", 0) or 0
        data["dividend_plus_buyback"] = dividends + buyback

        # =========================================
        # 9. 注入行业PE配置（低估值行业用区间上限，高估值行业用中值）
        # =========================================
        pe_low, pe_high = get_industry_pe_range(industry_type)
        # 取区间中值作为行业平均PE参考
        data["industry_pe"] = round((pe_low + pe_high) / 2, 1)

        logger.info(f"scorer data 构建完成，共 {len(data)} 个字段")
        return data

    @staticmethod
    def _summarize_df(df) -> Optional[Dict[str, Any]]:
        if df is None:
            return None
        result = {
            "row_count": len(df),
            "columns": list(df.columns),
        }
        if not df.empty:
            latest = df.iloc[0]
            result["latest_period"] = {
                str(k): str(v) if not isinstance(v, (int, float, bool)) else v
                for k, v in latest.items()
                if v is not None and str(v) != "nan"
            }
        return result