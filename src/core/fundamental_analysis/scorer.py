# -*- coding: utf-8 -*-
"""
===================================
基本面评分算法
===================================

职责：
1. 根据财务指标计算公司基本面评分
2. 支持自定义权重和评分规则

用法：
    继承 FundamentalScorer 并实现 score() 方法来自定义评分逻辑。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple

from src.core.fundamental_analysis.industry_config import (
    IndustryType,
    resolve_industry_type,
    get_industry_thresholds,
    is_financial_or_real_estate,
    is_liquor_industry,
)

logger = logging.getLogger(__name__)


class FundamentalScorer(ABC):
    """
    基本面评分器基类

    子类需实现 score() 方法，根据财务指标返回评分结果。
    """

    @abstractmethod
    def score(self, indicators: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
        """
        根据财务指标计算基本面评分

        Args:
            indicators: FinancialIndicators.calc_all_indicators() 返回的指标字典

        Returns:
            评分结果字典，至少包含：
            - total_score: 总分
            - dimension_scores: 各维度得分
            - rating: 评级（如 "优秀"、"良好"、"一般"、"较差"）
        """
        ...


class FinancialScorer(FundamentalScorer):
    """
    企业财报质量评分器

    基于五步分析框架，对上市公司财务质量进行打分（满分100分）。
    支持按行业分类调整营运效率评分阈值，使评分更符合行业特征。

    输入：公司财报关键指标字典
    输出：总分、分项得分、详细理由
    """

    def __init__(
        self,
        data: Dict[str, Any],
        industry_type: Optional[IndustryType] = None,
        sectors: Optional[List[str]] = None,
    ):
        """
        初始化评分器

        Args:
            data: 财报关键指标字典。
                  包含的字段见下，缺失项会影响相应部分得分。
            industry_type: 行业分类。如果提供，将使用行业专属阈值评分。
            sectors: 行业标签列表，与 industry_type 二选一。
                     如果只传 sectors，将自动解析为 IndustryType。

        data 字典字段说明：
        --- 三表勾稽 ---
        'net_income': 净利润 (单位统一为元)
        'retained_earnings_change': 留存收益变动额（期末-期初，可为负数）
        'dividends': 分红总额
        'share_buyback': 股票回购总额
        'operating_cash_flow': 经营活动现金流净额
        'cash_change_from_cf': 现金流量表"现金净变化额"
        'cash_balance_end': 期末现金余额
        'cash_balance_start': 期初现金余额
        --- 盈利与现金流 ---
        'ebit': 息税前利润
        'interest_expense': 利息费用
        'tax_expense': 所得税费用
        'roic': 资本回报率（%），若未提供则尝试计算
        'net_operating_assets': 净经营资产（用于ROIC分母）
        'capex': 资本性支出
        'fcf': 自由现金流（若未提供，自动计算 = operating_cash_flow - capex）
        'dividend_plus_buyback': 分红+回购总额
        --- 营运效率 ---
        'revenue': 营业收入
        'avg_receivables': 平均应收账款
        'avg_inventory': 平均存货
        'avg_payables': 平均应付账款
        'cost_of_goods_sold': 营业成本
        --- 杜邦分解（百分数，如 26.92 表示 26.92%）---
        'roe': 净资产收益率（%）
        'net_profit_margin': 净利润率（%）
        'asset_turnover': 总资产周转率（次数）
        'equity_multiplier': 权益乘数
        --- 估值 ---
        'current_price': 当前股价
        'eps': 每股收益
        'market_cap': 总市值
        'industry_pe': 行业平均市盈率（可选）
        'dcf_low': DCF估值下限
        'dcf_high': DCF估值上限
        'is_asset_heavy': 是否重资产公司（True/False），用于P/B判断
        'pb': 市净率
        'book_value_per_share': 每股净资产
        """
        self.data = data
        self.scores: Dict[str, float] = {}
        self.reasons: List[str] = []

        # 解析行业分类
        if industry_type is not None:
            self.industry_type = industry_type
        elif sectors is not None:
            self.industry_type = resolve_industry_type(sectors)
        else:
            self.industry_type = IndustryType.UNKNOWN

        self.thresholds = get_industry_thresholds(self.industry_type)
        self._is_fin_or_re = is_financial_or_real_estate(self.industry_type)
        self._is_liquor = is_liquor_industry(sectors or [])

        # 归一化：统一货币单位到元，统一百分比到百分数
        self.data = self._normalize_units(self.data)
        self.data = self._normalize_percent(self.data)
        logger.info(f"评分器初始化完毕，行业分类: {self.industry_type.value}")

    def _add_reason(self, item: str, score: float, max_score: float, detail: str):
        """记录单项得分及理由"""
        self.scores[item] = score
        self.reasons.append(f"{item} ({score}/{max_score}): {detail}")

    # ============================================================
    # 单位与数值归一化
    # ============================================================

    @staticmethod
    def _normalize_units(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一货币字段的单位到"元"

        通过检测关键货币字段的量级，自动判断数据是以"元"、"万元"还是"亿元"
        为单位，并统一转为"元"。避免因单位不一致导致 PE 等比率计算错误。

        检测逻辑：
        - 取净利润或营收作为参考值
        - 参考值 < 10^4 → 推测单位为亿（×1e8 转元）
        - 10^4 ≤ 参考值 < 10^8 → 推测单位为万（×1e4 转元）
        - 参考值 ≥ 10^8 → 推测单位为元（不变）
        """
        data = dict(data)

        # 待检测的货币字段
        monetary_fields = {
            "net_income", "market_cap", "revenue", "ebit",
            "operating_cash_flow", "fcf", "capex",
            "dividends", "share_buyback", "dividend_plus_buyback",
            "avg_receivables", "avg_inventory", "avg_payables",
            "cash_balance_end", "cash_balance_start",
            "cash_change_from_cf", "interest_expense", "tax_expense",
            "total_debt",
        }

        # 取参考值：优先用 net_income 或 revenue
        ref = None
        for key in ("net_income", "revenue", "market_cap", "operating_cash_flow"):
            val = data.get(key)
            if val is not None and val != 0:
                ref = abs(val)
                break

        if ref is None:
            return data

        # 判断量级
        if ref < 1e4:       # 0 ~ 9,999 → 亿
            scale = 1e8
            logger.info(f"货币单位归一化：检测到参考值 {ref:.2f}，推测单位为亿，×{scale:.0e} 转元")
        elif ref < 1e8:     # 1万 ~ 9,999万 → 万
            scale = 1e4
            logger.info(f"货币单位归一化：检测到参考值 {ref:.2f}，推测单位为万，×{scale:.0e} 转元")
        else:
            scale = 1        # 已经是元

        if scale != 1:
            for k in monetary_fields:
                v = data.get(k)
                if v is not None:
                    data[k] = v * scale
            logger.info(f"货币单位归一化完成，共处理了 {len(monetary_fields & set(data.keys()))} 个字段")

        return data

    @staticmethod
    def _normalize_percent(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一百分比字段到百分数格式（如 26.92 表示 26.92%）

        检测规则：
        - 如果值在 (0, 1) 范围内，视为小数格式，×100 转为百分数
        - 特殊处理 ROE：茅台 ROE 可达 30+，双位数正常
        """
        percent_fields = {"net_profit_margin", "roe", "roic",
                          "revenue_growth_yoy", "net_profit_growth_yoy",
                          "eps_growth_yoy", "gross_profit_margin",
                          "operating_profit_margin",
                          "operating_cashflow_to_revenue",
                          "debt_to_asset_ratio"}

        for field in percent_fields:
            val = data.get(field)
            if val is not None and 0 < val < 1:
                data[field] = round(val * 100, 2)
                logger.debug(f"百分比归一化: {field} 从 {val} 转为 {data[field]}")

        return data

    def score_integrity(self) -> float:
        """三表勾稽与真实性（20分）"""
        max_score = 20.0
        score = 0.0
        details = []

        # 1. 净利润 vs 留存收益变动 (5分)
        if 'net_income' in self.data and 'retained_earnings_change' in self.data:
            ni = self.data['net_income']
            re_change = self.data['retained_earnings_change']
            # 考虑分红回购会减少留存收益，调整净利润后对比
            adjustment = self.data.get('dividends', 0) + self.data.get('share_buyback', 0)
            adjusted_ni = ni - adjustment
            if abs(adjusted_ni - re_change) < abs(ni) * 0.05:  # 5%误差
                s = 5.0
                detail = "净利润与留存收益变动基本匹配"
            else:
                s = 2.0
                detail = f"净利润{ni}，分红回购{adjustment}，留存收益变动{re_change}，差异较大"
            score += s
            details.append(detail)
        else:
            details.append("未提供净利润或留存收益变动，本项得0分")

        # 2. 经营现金流 vs 净利润 (10分)
        if 'operating_cash_flow' in self.data and 'net_income' in self.data:
            ocf = self.data['operating_cash_flow']
            ni = self.data['net_income']
            if ni != 0:
                ratio = ocf / ni
                if ocf > ni:
                    s = 10.0
                    detail = f"经营现金流({ocf}) > 净利润({ni})，利润含金量高"
                elif ratio > 0.8:
                    s = 7.0
                    detail = f"经营现金流/净利润 = {ratio:.2f}，含金量尚可"
                elif ratio > 0.5:
                    s = 4.0
                    detail = f"经营现金流/净利润 = {ratio:.2f}，利润质量偏低"
                else:
                    s = 0.0
                    detail = f"经营现金流/净利润 = {ratio:.2f}，纸面富贵风险高"
            else:
                s = 0.0
                detail = "净利润为0或负数，无法比较"
            score += s
            details.append(detail)
        else:
            details.append("未提供经营现金流或净利润，本项得0分")

        # 3. 现金变动一致性 (5分)
        if all(k in self.data for k in ['cash_change_from_cf', 'cash_balance_end', 'cash_balance_start']):
            cf_change = self.data['cash_change_from_cf']
            bs_change = self.data['cash_balance_end'] - self.data['cash_balance_start']
            if abs(cf_change - bs_change) < max(abs(cf_change), 1e-6) * 0.02:
                s = 5.0
                detail = "现金流量表现金净变化与资产负债表现金差额一致"
            else:
                s = 0.0
                detail = f"现金变动不一致：CF净变化={cf_change}，BS变动={bs_change}"
            score += s
            details.append(detail)
        else:
            details.append("未提供完整的现金变动数据，本项得0分")

        # 综合记录
        self._add_reason("三表勾稽真实性", score, max_score, "；".join(details))
        return score

    def score_cash_earnings(self) -> float:
        """核心盈利与现金流质量（30分）"""
        max_score = 30.0
        score = 0.0
        details = []

        # EBIT (5分) - 假设连续3年为正需要外部提供，这里仅判断当年
        if 'ebit' in self.data:
            ebit = self.data['ebit']
            if ebit > 0:
                s = 5.0
                detail = f"EBIT = {ebit} > 0，经营盈利为正"
            else:
                s = 0.0
                detail = f"EBIT = {ebit}，经营亏损"
            score += s
            details.append(detail)
        else:
            details.append("未提供EBIT，本项得0分")

        # 利息保障倍数 (5分)
        if 'ebit' in self.data and 'interest_expense' in self.data:
            interest = self.data['interest_expense']
            if interest != 0:
                cover = self.data['ebit'] / interest
                if cover >= 3:
                    s = 5.0
                    detail = f"利息保障倍数 = {cover:.2f} ≥ 3，偿债能力强"
                elif cover >= 2:
                    s = 3.0
                    detail = f"利息保障倍数 = {cover:.2f}，处于2~3之间，需关注"
                else:
                    s = 0.0
                    detail = f"利息保障倍数 = {cover:.2f} < 2，风险信号"
            else:
                # 利息费用为零：需要区分三种情况
                interest_capitalized = self.data.get('interest_capitalized', 0)
                total_debt = self.data.get('total_debt')
                has_debt = total_debt is not None and abs(total_debt) > 0

                if interest_capitalized and abs(interest_capitalized) > 0:
                    s = 1.0
                    detail = f"利息费用为零但存在利息资本化({interest_capitalized:.2f})，可能有粉饰嫌疑"
                elif has_debt:
                    s = 2.0
                    detail = f"有负债({total_debt:.2f})但利息费用为零，可能数据异常"
                else:
                    s = 5.0
                    detail = "无负债且无利息费用，财务稳健"
            score += s
            details.append(detail)
        else:
            details.append("未提供EBIT或利息费用，本项得0分")

        # ROIC (5分)
        if 'roic' in self.data:
            roic = self.data['roic']
            if roic > 15:
                s = 5.0
                detail = f"ROIC = {roic:.1f}% > 15%，资本回报优秀"
            elif roic >= 10:
                s = 3.0
                detail = f"ROIC = {roic:.1f}%，处于10%~15%之间"
            else:
                s = 1.0
                detail = f"ROIC = {roic:.1f}% < 10%，资本回报较差"
            score += s
            details.append(detail)
        elif 'ebit' in self.data and 'tax_expense' in self.data and 'net_operating_assets' in self.data:
            # 尝试计算：ROIC = EBIT*(1-税率)/净经营资产
            tax_rate = self.data['tax_expense'] / (self.data['ebit'] + self.data['tax_expense']) if (self.data['ebit']+self.data['tax_expense']) != 0 else 0
            nopat = self.data['ebit'] * (1 - tax_rate)
            roic = nopat / self.data['net_operating_assets'] * 100
            if roic > 15:
                s = 5.0
                detail = f"计算得ROIC={roic:.1f}% > 15%"
            elif roic >= 10:
                s = 3.0
                detail = f"计算得ROIC={roic:.1f}%"
            else:
                s = 1.0
                detail = f"计算得ROIC={roic:.1f}%"
            score += s
            details.append(detail)
        else:
            details.append("未提供ROIC或计算所需数据，本项得0分")

        # 自由现金流为正 (5分)
        fcf = self.data.get('fcf')
        if fcf is None and 'operating_cash_flow' in self.data and 'capex' in self.data:
            fcf = self.data['operating_cash_flow'] - self.data['capex']
        if fcf is not None:
            if fcf > 0:
                s = 5.0
                detail = f"自由现金流 = {fcf:.2f} > 0"
            else:
                s = 0.0
                detail = f"自由现金流 = {fcf:.2f}，持续为负有烧钱风险"
            score += s
            details.append(detail)
        else:
            details.append("未提供自由现金流数据，本项得0分")

        # FCF vs 净利润 (5分)
        if fcf is not None and 'net_income' in self.data:
            ni = self.data['net_income']
            if ni != 0:
                if fcf >= ni:
                    s = 5.0
                    detail = f"FCF({fcf:.2f}) ≥ 净利润({ni})，盈利质量极佳"
                elif fcf >= ni * 0.8:
                    s = 3.0
                    detail = f"FCF/净利润 = {fcf/ni:.2f}，质量尚可"
                else:
                    s = 1.0
                    detail = f"FCF长期低于净利润，仅{fcf/ni:.2f}倍"
            else:
                s = 0.0
                detail = "净利润为负，不适用比较"
            score += s
            details.append(detail)
        else:
            details.append("未提供FCF或净利润，本项得0分")

        # FCF vs 分红+回购 (5分)
        dpb = self.data.get('dividend_plus_buyback')
        if fcf is not None and dpb is not None:
            if fcf >= dpb:
                s = 5.0
                detail = f"FCF({fcf:.2f}) ≥ 分红回购总额({dpb:.2f})，可持续"
            else:
                s = 0.0
                detail = f"FCF不足以覆盖分红回购，差额{dpb-fcf:.2f}，依赖外部融资"
            score += s
            details.append(detail)
        else:
            details.append("未提供分红回购总额，本项得0分")

        self._add_reason("盈利与现金流质量", score, max_score, "；".join(details))
        return score

    def score_efficiency(self) -> float:
        """营运效率与议价能力（15分）"""
        max_score = 15.0
        score = 0.0
        details = []
        rec_thresholds = self.thresholds["receivables"]
        inv_thresholds = self.thresholds["inventory"]
        ap_thresholds = self.thresholds["payables"]

        # --------------------------------------------------
        # 应收账款周转天数
        # --------------------------------------------------
        if self._is_fin_or_re:
            # 金融/地产行业不适用应收周转，给基础分
            details.append("金融/地产行业，应收账款周转指标不适用，给基础分 2.5/5")
            score += 2.5
        elif 'avg_receivables' in self.data and 'revenue' in self.data:
            rec_days = 365 * self.data['avg_receivables'] / self.data['revenue']
            excellent, good, fair = rec_thresholds
            if rec_days < excellent:
                s = 5.0
                detail = f"应收账款周转天数 = {rec_days:.1f}天，回款极快（行业优秀 < {excellent}天）"
            elif rec_days < good:
                s = 4.0
                detail = f"应收账款周转天数 = {rec_days:.1f}天，回款较快"
            elif rec_days < fair:
                s = 2.0
                detail = f"应收账款周转天数 = {rec_days:.1f}天，偏长"
            else:
                s = 0.0
                detail = f"应收账款周转天数 = {rec_days:.1f}天，资金占用严重"
            score += s
            details.append(detail)
        else:
            details.append("未提供应收账款或营收，本项得0分")

        # --------------------------------------------------
        # 存货周转天数
        # --------------------------------------------------
        if self._is_fin_or_re:
            details.append("金融/地产行业，存货周转指标不适用，给基础分 2.5/5")
            score += 2.5
        elif self._is_liquor:
            # 白酒行业：存货越久越值钱，周转慢反而是优势
            if 'avg_inventory' in self.data and 'cost_of_goods_sold' in self.data:
                inv_days = 365 * self.data['avg_inventory'] / self.data['cost_of_goods_sold']
                if inv_days < 180:
                    s = 3.0
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备充足性一般"
                elif inv_days < 730:
                    s = 5.0
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备充足，越陈越香"
                else:
                    s = 5.0
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备极为丰富"
                score += s
                details.append(detail)
            else:
                details.append("未提供存货或营业成本，本项得0分")
        elif 'avg_inventory' in self.data and 'cost_of_goods_sold' in self.data:
            inv_days = 365 * self.data['avg_inventory'] / self.data['cost_of_goods_sold']
            excellent, fair = inv_thresholds
            if inv_days < excellent:
                s = 5.0
                detail = f"存货周转天数 = {inv_days:.1f}天，管理优秀（行业优秀 < {excellent}天）"
            elif inv_days < fair:
                s = 3.0
                detail = f"存货周转天数 = {inv_days:.1f}天，正常水平"
            else:
                s = 1.0
                detail = f"存货周转天数 = {inv_days:.1f}天，可能积压"
            score += s
            details.append(detail)
        else:
            details.append("未提供存货或营业成本，本项得0分")

        # --------------------------------------------------
        # 应付账款周转天数
        # --------------------------------------------------
        if self._is_fin_or_re:
            details.append("金融/地产行业，应付账款周转指标不适用，给基础分 2.5/5")
            score += 2.5
        elif 'avg_payables' in self.data and 'cost_of_goods_sold' in self.data:
            ap_days = 365 * self.data['avg_payables'] / self.data['cost_of_goods_sold']
            strong, general = ap_thresholds
            if ap_days > strong:
                s = 5.0
                detail = f"应付账款周转天数 = {ap_days:.1f}天，强议价能力（行业强 > {strong}天）"
            elif ap_days > general:
                s = 3.0
                detail = f"应付账款周转天数 = {ap_days:.1f}天，议价能力一般"
            else:
                s = 1.0
                detail = f"应付账款周转天数 = {ap_days:.1f}天，弱势"
            score += s
            details.append(detail)
        else:
            details.append("未提供应付账款或营业成本，本项得0分")

        self._add_reason("营运效率与议价能力", score, max_score, "；".join(details))
        return score

    def score_duPont(self) -> float:
        """杜邦分析 – ROE驱动力质量（20分）"""
        max_score = 20.0
        score = 0.0
        details = []

        # 需要净利润率、资产周转率、权益乘数，或至少ROE
        npm = self.data.get('net_profit_margin')
        turnover = self.data.get('asset_turnover')
        lev = self.data.get('equity_multiplier')
        roe = self.data.get('roe')

        # 如果没有分解因子但给了ROE，尝试粗略判断（无法准确拆解则降低分数）
        if npm is not None and turnover is not None and lev is not None:
            # 高质量：净利润率 > 15%
            if npm > 15:
                s_npm = 10.0
                detail_npm = f"高净利润率驱动({npm:.1f}%)，可持续性强"
            else:
                s_npm = 0.0
                detail_npm = f"净利润率偏低({npm:.1f}%)"

            # 周转：>1为高效
            if turnover > 1:
                s_turn = 5.0
                detail_turn = f"资产周转率{turnover:.2f}次，高效"
            else:
                s_turn = 2.0
                detail_turn = f"资产周转率{turnover:.2f}次，偏低"

            # 杠杆：>3为高风险依赖
            if lev > 3:
                s_lev = 0.0
                detail_lev = f"权益乘数{lev:.2f}倍，高杠杆驱动风险大"
                # 但如果净利润率也很高，可以适当补偿
                if npm > 15:
                    s_lev = 3.0
                    detail_lev = f"虽然杠杆高({lev:.2f})，但高利润率支撑，风险中等"
            elif lev > 2:
                s_lev = 3.0
                detail_lev = f"权益乘数{lev:.2f}倍，中等杠杆"
            else:
                s_lev = 5.0
                detail_lev = f"权益乘数{lev:.2f}倍，低杠杆安全"

            score = s_npm + s_turn + s_lev
            details.append(detail_npm)
            details.append(detail_turn)
            details.append(detail_lev)
        else:
            # 如果只有ROE，采用简化评分
            if roe is not None:
                if roe > 20:
                    details.append(f"ROE为{roe:.1f}%，但缺乏杜邦分解，无法判断驱动质量，给基础分8/20")
                    score = 8.0
                elif roe > 10:
                    score = 5.0
                    details.append(f"ROE为{roe:.1f}%，缺乏分解，给5分")
                else:
                    score = 2.0
                    details.append(f"ROE为{roe:.1f}%，较低且缺乏分解")
            else:
                details.append("未提供杜邦分解数据或ROE，本项得0分")
                score = 0.0

        self._add_reason("杜邦驱动质量", score, max_score, "；".join(details))
        return score

    def score_valuation(self) -> float:
        """估值合理性（15分）"""
        max_score = 15.0
        score = 0.0
        details = []

        # 市盈率相对行业 (5分)
        if 'market_cap' in self.data and 'net_income' in self.data:
            ni = self.data['net_income']
            if ni > 0:
                pe = self.data['market_cap'] / ni
                industry_pe = self.data.get('industry_pe')
                if industry_pe is not None:
                    if pe < industry_pe:
                        s = 5.0
                        detail = f"PE = {pe:.1f}，低于行业均值{industry_pe:.1f}，估值有吸引力"
                    elif pe < industry_pe * 1.2:
                        s = 3.0
                        detail = f"PE = {pe:.1f}，接近行业均值，合理"
                    else:
                        s = 1.0
                        detail = f"PE = {pe:.1f}，显著高于行业{industry_pe:.1f}，可能高估"
                else:
                    # 没有行业PE，用绝对标准
                    if pe < 15:
                        s = 5.0
                        detail = f"PE = {pe:.1f}，绝对值较低"
                    elif pe < 25:
                        s = 3.0
                        detail = f"PE = {pe:.1f}，中等水平"
                    else:
                        s = 1.0
                        detail = f"PE = {pe:.1f}，偏贵"
                score += s
                details.append(detail)
            else:
                details.append("净利润为负，无法使用PE")
        else:
            details.append("未提供市值或净利润，无法计算PE")

        # DCF估值区间 (5分)
        if 'current_price' in self.data and 'dcf_low' in self.data and 'dcf_high' in self.data:
            price = self.data['current_price']
            low = self.data['dcf_low']
            high = self.data['dcf_high']
            if price < low:
                s = 5.0
                detail = f"股价{price}低于DCF下限{low}，明显低估"
            elif price < high:
                s = 3.0
                detail = f"股价{price}在DCF区间[{low},{high}]内，合理"
            else:
                s = 0.0
                detail = f"股价{price}高于DCF上限{high}，高估"
            score += s
            details.append(detail)
        else:
            details.append("未提供当前股价或DCF区间，本项得0分")

        # P/B 结合资产类型 (5分)
        if 'pb' in self.data and 'is_asset_heavy' in self.data:
            pb = self.data['pb']
            heavy = self.data['is_asset_heavy']
            if heavy:
                if pb < 1.5:
                    s = 5.0
                    detail = f"重资产公司，PB={pb:.2f}，低于1.5倍，估值合理"
                elif pb < 2.5:
                    s = 3.0
                    detail = f"重资产公司，PB={pb:.2f}，略高于净资产"
                else:
                    s = 0.0
                    detail = f"重资产公司，PB={pb:.2f}，显著高估"
            else:
                # 轻资产，PB高可接受
                if pb < 3:
                    s = 5.0
                    detail = f"轻资产公司，PB={pb:.2f}，合理偏低"
                elif pb < 10:
                    s = 3.0
                    detail = f"轻资产公司，PB={pb:.2f}，属于正常范围"
                else:
                    s = 1.0
                    detail = f"轻资产公司，PB={pb:.2f}，极高，但若品牌强仍可接受"
            score += s
            details.append(detail)
        else:
            details.append("未提供市净率或资产类型信息，本项得0分")

        self._add_reason("估值合理性", score, max_score, "；".join(details))
        return score

    def score(self, indicators: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
        """
        实现抽象基类接口：根据财务指标计算基本面评分

        Args:
            indicators: FinancialIndicators.calc_all_indicators() 返回的指标字典

        Returns:
            评分结果字典
        """
        # 将 indicators 展开并转为 data 格式（简化版）
        flat_data = {}
        for category, metrics in indicators.items():
            if metrics:
                flat_data.update(metrics)
        self.data.update(flat_data)

        total, module_scores, reasons = self.full_score()
        return {
            "total_score": total,
            "dimension_scores": module_scores,
            "rating": self._rating(total),
            "reasons": reasons,
        }

    @staticmethod
    def _rating(total: float) -> str:
        """根据总分返回评级"""
        if total >= 85:
            return "优秀"
        elif total >= 70:
            return "良好"
        elif total >= 60:
            return "一般"
        else:
            return "较差"

    def full_score(self) -> Tuple[float, Dict[str, float], List[str]]:
        """计算总分，返回(总分, 各模块得分, 所有理由列表)"""
        s1 = self.score_integrity()
        s2 = self.score_cash_earnings()
        s3 = self.score_efficiency()
        s4 = self.score_duPont()
        s5 = self.score_valuation()
        total = s1 + s2 + s3 + s4 + s5
        module_scores = {
            "三表勾稽真实性": s1,
            "盈利与现金流质量": s2,
            "营运效率与议价能力": s3,
            "杜邦驱动质量": s4,
            "估值合理性": s5,
        }
        return total, module_scores, self.reasons

def print_score_result(total: float, module_scores: Dict[str, float], reasons: List[str]):
    """格式化输出评分结果"""
    print("=" * 80)
    print("企业财报质量评分报告")
    print("=" * 80)
    print(f"总分: {total:.1f} / 100")
    print("\n模块得分：")
    for mod, sc in module_scores.items():
        print(f"  {mod}: {sc:.1f}")
    print("\n详细评分理由：")
    for r in reasons:
        print(f"  - {r}")
    # 综合评级
    if total >= 85:
        grade = "高质量公司，可重点关注"
    elif total >= 70:
        grade = "中等偏上，适合跟踪"
    elif total >= 60:
        grade = "一般或存疑，需谨慎"
    else:
        grade = "高风险或存在造假嫌疑，建议回避"
    print(f"\n综合评级：{grade}")
    print("=" * 80)

# ================= 使用示例 =================
if __name__ == "__main__":
    # 模拟一个公司的财报数据（以苹果2025财年部分指标为例）
    sample_data = {
        # 三表勾稽
        "net_income": 937.4e9,          # 净利润 9374亿（示例单位）
        "retained_earnings_change": 800e9,
        "dividends": 150e9,
        "share_buyback": 900e9,
        "operating_cash_flow": 1180e9,
        "cash_change_from_cf": 50e9,
        "cash_balance_end": 600e9,
        "cash_balance_start": 550e9,
        # 盈利与现金流
        "ebit": 1200e9,
        "interest_expense": 10e9,
        "tax_expense": 250e9,
        "roic": 56.0,
        "capex": 100e9,
        "fcf": 1080e9,
        "dividend_plus_buyback": 1050e9,
        # 营运效率
        "revenue": 3910e9,
        "avg_receivables": 300e9,
        "avg_inventory": 60e9,
        "avg_payables": 740e9,
        "cost_of_goods_sold": 2100e9,
        # 杜邦
        "net_profit_margin": 26.92,
        "asset_turnover": 1.16,
        "equity_multiplier": 4.87,
        "roe": 152.0,
        # 估值
        "current_price": 250,
        "dcf_low": 200,
        "dcf_high": 280,
        "market_cap": 3800e9,
        "industry_pe": 25,
        "is_asset_heavy": False,
        "pb": 8.5,
    }

    print("=" * 80)
    print("示例1: 科技行业（苹果公司）")
    print("=" * 80)
    industry_type = IndustryType.TECH
    scorer = FinancialScorer(sample_data, industry_type=industry_type)
    total, module_scores, reasons = scorer.full_score()
    print_score_result(total, module_scores, reasons)

    print()
    print()
    print("=" * 80)
    print("示例2: 白酒行业（模拟数据，演示行业感知）")
    print("=" * 80)
    liquor_data = {
        "net_income": 800e9,
        "retained_earnings_change": 750e9,
        "operating_cash_flow": 900e9,
        "cash_change_from_cf": 30e9,
        "cash_balance_end": 300e9,
        "cash_balance_start": 270e9,
        "ebit": 1000e9,
        "interest_expense": 0,
        "tax_expense": 250e9,
        "roic": 35.0,
        "capex": 50e9,
        "fcf": 850e9,
        "dividend_plus_buyback": 600e9,
        "revenue": 1200e9,
        "avg_receivables": 10e9,
        "avg_inventory": 350e9,
        "avg_payables": 300e9,
        "cost_of_goods_sold": 200e9,
        "net_profit_margin": 60.0,
        "asset_turnover": 0.6,
        "equity_multiplier": 1.5,
        "roe": 45.0,
        "current_price": 150,
        "dcf_low": 120,
        "dcf_high": 180,
        "market_cap": 2000e9,
        "industry_pe": 30,
        "is_asset_heavy": False,
        "pb": 5.0,
    }
    # 白酒存货通常数百万（老酒储备），演示行业感知
    liquor_scorer = FinancialScorer(liquor_data, sectors=["白酒Ⅲ", "食品饮料", "贵州板块"])
    total2, module_scores2, reasons2 = liquor_scorer.full_score()
    print_score_result(total2, module_scores2, reasons2)