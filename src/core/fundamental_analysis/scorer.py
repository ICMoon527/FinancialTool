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
    is_financial_or_real_estate,
    is_liquor_industry,
)
from src.core.fundamental_analysis.industry_percentile import score_by_percentile, get_percentile_info
from src.core.fundamental_analysis.scorer_config import (
    get_scorer_config,
    _get_default_threshold_score,
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
        self._config = get_scorer_config()

        # 解析行业分类
        if industry_type is not None:
            self.industry_type = industry_type
        elif sectors is not None:
            self.industry_type = resolve_industry_type(sectors)
        else:
            self.industry_type = IndustryType.UNKNOWN

        self._is_fin_or_re = is_financial_or_real_estate(self.industry_type)
        self._is_liquor = is_liquor_industry(sectors or [])

        logger.info(f"评分器初始化完毕，行业分类: {self.industry_type.value}")

    def _add_reason(self, item: str, score: float, max_score: float, detail: str):
        """记录单项得分及理由"""
        self.scores[item] = score
        self.reasons.append(f"{item} ({score}/{max_score}): {detail}")

    def score_integrity(self) -> float:
        """三表勾稽与真实性"""
        cfg = self._config.integrity
        max_score = self._config.weights.get("integrity", 20.0)
        score = 0.0
        details = []

        # 1. 净利润 vs 留存收益变动
        #    留存收益变动 = 净利润 - 分红 - 回购 + 其他综合收益 + 会计政策变更
        #                  + 前期差错更正 + 设定受益计划重计量 + ...
        #    注意：分红数据为年度汇总，与单期净利润期间不匹配，不纳入校验
        #    我们可获取：净利润、其他综合收益
        #    不可获取：会计政策变更、前期差错更正、设定受益计划重计量
        if 'net_income' in self.data and 'retained_earnings_change' in self.data:
            ni = self.data['net_income']
            re_change = self.data['retained_earnings_change']
            oci = self.data.get('other_comprehensive_income', 0) or 0

            adjusted_ni = ni + oci
            diff = abs(adjusted_ni - re_change)
            diff_ratio = diff / abs(ni) if ni != 0 else 0

            oci_note = f"，其他综合收益{oci}" if oci else ""

            re_cfg = cfg.retained_earnings
            s = _get_default_threshold_score(
                diff_ratio, re_cfg.thresholds, [re_cfg.weight, re_cfg.weight * 0.8, re_cfg.weight * 0.4, 0.0])

            if diff_ratio < re_cfg.thresholds[0]:
                detail = f"净利润{ni}与留存收益变动{re_change}高度匹配（差异{diff_ratio:.1%}），勾稽严谨"
            elif diff_ratio < re_cfg.thresholds[1]:
                detail = (
                    f"净利润{ni}，加其他综合收益后"
                    f"与留存收益变动{re_change}存在{diff_ratio:.1%}差异，"
                    f"可能由会计政策变更、前期差错更正等未获取的调整项导致，基本可信"
                )
            elif diff_ratio < re_cfg.thresholds[2]:
                detail = (
                    f"净利润{ni}，加其他综合收益后"
                    f"与留存收益变动{re_change}差异{diff_ratio:.1%}，较大，建议关注附注"
                )
            else:
                detail = (
                    f"净利润{ni}，加其他综合收益后"
                    f"与留存收益变动{re_change}严重不符（差异{diff_ratio:.1%}），"
                    f"存在重大勾稽异常"
                )
            score += s
            details.append(detail)
        else:
            details.append("未提供净利润或留存收益变动，本项得0分")

        # 2. 经营现金流 vs 净利润
        if 'operating_cash_flow' in self.data and 'net_income' in self.data:
            ocf = self.data['operating_cash_flow']
            ni = self.data['net_income']
            ocf_cfg = cfg.ocf_vs_ni
            if ni != 0:
                ratio = ocf / ni
                s = _get_default_threshold_score(
                    ratio, ocf_cfg.thresholds, [ocf_cfg.weight, ocf_cfg.weight * 0.7, ocf_cfg.weight * 0.4, 0.0])
                if ocf > ni:
                    detail = f"经营现金流({ocf}) > 净利润({ni})，利润含金量高"
                elif ratio > ocf_cfg.thresholds[1]:
                    detail = f"经营现金流/净利润 = {ratio:.2f}，含金量尚可"
                elif ratio > ocf_cfg.thresholds[2]:
                    detail = f"经营现金流/净利润 = {ratio:.2f}，利润质量偏低"
                else:
                    detail = f"经营现金流/净利润 = {ratio:.2f}，纸面富贵风险高"
            else:
                s = 0.0
                detail = "净利润为0或负数，无法比较"
            score += s
            details.append(detail)
        else:
            details.append("未提供经营现金流或净利润，本项得0分")

        # 3. 现金变动一致性
        #    现金流量表"现金净增加额" vs 资产负债表"货币资金"期末-期初变动
        #    两者可能因以下正常会计分类产生差异：
        #    - 定期存款计入货币资金但不属于现金等价物
        #    - 受限资金重分类（保证金、司法冻结等）
        #    - 三个月以上银行承兑汇票贴现
        if all(k in self.data for k in ['cash_change_from_cf', 'cash_balance_end', 'cash_balance_start']):
            cf_change = self.data['cash_change_from_cf']
            bs_change = self.data['cash_balance_end'] - self.data['cash_balance_start']

            base = max(abs(cf_change), abs(bs_change), 1e-6)
            diff_ratio = abs(cf_change - bs_change) / base

            cc_cfg = cfg.cash_change
            s = _get_default_threshold_score(
                diff_ratio, cc_cfg.thresholds, [cc_cfg.weight, cc_cfg.weight * 0.8, cc_cfg.weight * 0.4, 0.0])

            if diff_ratio < cc_cfg.thresholds[0]:
                detail = f"现金变动高度一致（差异{diff_ratio:.1%}），勾稽严谨"
            elif diff_ratio < cc_cfg.thresholds[1]:
                detail = (
                    f"现金变动存在{diff_ratio:.1%}轻微差异（CF={cf_change}，BS={bs_change}），"
                    f"可能由定期存款分类、受限资金重分类等正常会计处理导致"
                )
            elif diff_ratio < cc_cfg.thresholds[2]:
                detail = (
                    f"现金变动差异{diff_ratio:.1%}较大（CF={cf_change}，BS={bs_change}），"
                    f"建议关注附注中的现金等价物说明"
                )
            else:
                detail = (
                    f"现金变动严重不一致（差异{diff_ratio:.1%}）：CF净变化={cf_change}，"
                    f"BS变动={bs_change}，可能存在分类错误或异常"
                )
            score += s
            details.append(detail)
        else:
            details.append("未提供完整的现金变动数据，本项得0分")

        # 综合记录
        self._add_reason("三表勾稽真实性", score, max_score, "；".join(details))
        return score

    def score_cash_earnings(self) -> float:
        """核心盈利与现金流质量"""
        cfg = self._config.cash_earnings
        max_score = self._config.weights.get("cash_earnings", 30.0)
        score = 0.0
        details = []

        # EBIT - 假设连续3年为正需要外部提供，这里仅判断当年
        if 'ebit' in self.data:
            ebit = self.data['ebit']
            if ebit > 0:
                s = cfg.ebit_positive.weight
                detail = f"EBIT = {ebit} > 0，经营盈利为正"
            else:
                s = 0.0
                detail = f"EBIT = {ebit}，经营亏损"
            score += s
            details.append(detail)
        else:
            details.append("未提供EBIT，本项得0分")

        # 利息保障倍数
        if 'ebit' in self.data and 'interest_expense' in self.data:
            interest = self.data['interest_expense']
            ic_cfg = cfg.interest_coverage
            if interest != 0:
                cover = self.data['ebit'] / interest
                s = _get_default_threshold_score(
                    cover, ic_cfg.thresholds, [ic_cfg.weight, ic_cfg.weight * 0.6, 0.0])
                if cover >= ic_cfg.thresholds[0]:
                    detail = f"利息保障倍数 = {cover:.2f} ≥ {ic_cfg.thresholds[0]}，偿债能力强"
                elif cover >= ic_cfg.thresholds[1]:
                    detail = f"利息保障倍数 = {cover:.2f}，处于{ic_cfg.thresholds[1]}~{ic_cfg.thresholds[0]}之间，需关注"
                else:
                    detail = f"利息保障倍数 = {cover:.2f} < {ic_cfg.thresholds[1]}，风险信号"
            else:
                # 利息费用为零：区分有息负债/无息负债/资本化
                interest_capitalized = self.data.get('interest_capitalized', 0) or 0
                total_debt = self.data.get('total_debt')
                debt_ratio = self.data.get('debt_to_asset_ratio')
                has_debt = total_debt is not None and abs(total_debt) > 0

                if interest_capitalized and abs(interest_capitalized) > 0:
                    s = ic_cfg.zero_interest_score_capitalized
                    detail = f"利息费用为零但存在利息资本化({interest_capitalized:.2f})，可能有粉饰嫌疑"
                elif has_debt:
                    if debt_ratio is not None and debt_ratio < ic_cfg.zero_interest_debt_ratio_low:
                        s = ic_cfg.zero_interest_score_low_leverage
                        detail = (
                            f"有负债({total_debt:.2f})但无利息费用，"
                            f"资产负债率仅{debt_ratio:.1f}%，"
                            f"可能为应付账款、预收账款等无息负债，议价能力强的表现"
                        )
                    else:
                        s = ic_cfg.zero_interest_score_has_debt
                        detail = (
                            f"有负债({total_debt:.2f})但利息费用为零，"
                            f"可能为无息负债或数据缺失，建议关注负债结构"
                        )
                else:
                    s = ic_cfg.zero_interest_score_no_debt
                    detail = "无负债且无利息费用，财务稳健"
            score += s
            details.append(detail)
        else:
            details.append("未提供EBIT或利息费用，本项得0分")

        # ROIC - 基于行业分位数的动态阈值
        if 'roic' in self.data:
            roic = self.data['roic']
            s, detail = score_by_percentile(
                roic, "roic", self.industry_type, max_score=cfg.roic.weight, higher_better=True,
            )
            detail = f"{detail}（{get_percentile_info(self.industry_type, 'roic')}）"
            score += s
            details.append(detail)
        elif 'ebit' in self.data and 'tax_expense' in self.data and 'net_operating_assets' in self.data:
            # 回退：用 EBIT 和所得税费用估算税率，再计算 ROIC
            pretax = self.data['ebit'] + self.data['tax_expense']
            if pretax != 0:
                raw_rate = self.data['tax_expense'] / pretax
                tax_rate = max(0, min(raw_rate, 0.25))
            else:
                tax_rate = 0.15
            nopat = self.data['ebit'] * (1 - tax_rate)
            roic = nopat / self.data['net_operating_assets'] * 100
            s, detail = score_by_percentile(
                roic, "roic", self.industry_type, max_score=cfg.roic.weight, higher_better=True,
            )
            detail = f"估算ROIC={roic:.1f}%（税率约{tax_rate:.0%}），{detail}"
            score += s
            details.append(detail)
        else:
            details.append("未提供ROIC或计算所需数据，本项得0分")

        # 自由现金流为正
        #    注意：FCF = 经营现金流 - capex 是简化定义，未区分维持性capex与扩张性capex
        #    高成长公司扩张性capex巨大导致负FCF，不等于"烧钱"，需结合营收增速判断
        fcf = self.data.get('fcf')
        if fcf is None and 'operating_cash_flow' in self.data and 'capex' in self.data:
            fcf = self.data['operating_cash_flow'] - self.data['capex']
        if fcf is not None:
            if fcf > 0:
                s = cfg.fcf_positive.weight
                detail = f"自由现金流 = {fcf:.2f} > 0"
            else:
                rev_growth = self.data.get('revenue_growth_yoy')
                ocf = self.data.get('operating_cash_flow')
                if rev_growth is not None and rev_growth > cfg.high_growth_threshold:
                    s = cfg.fcf_positive.weight * 0.6
                    detail = (
                        f"自由现金流 = {fcf:.2f} < 0，但营收增速{rev_growth:.1f}%，"
                        f"负FCF可能由战略性扩张capex导致，非经营恶化"
                    )
                elif ocf is not None and ocf > 0:
                    s = cfg.fcf_positive.weight * 0.4
                    detail = (
                        f"自由现金流 = {fcf:.2f} < 0，但经营现金流({ocf:.2f})为正，"
                        f"负FCF由capex导致，需关注投资回报率"
                    )
                else:
                    s = 0.0
                    detail = f"自由现金流 = {fcf:.2f}，经营现金流也为负，存在烧钱风险"
            score += s
            details.append(detail)
        else:
            details.append("未提供自由现金流数据，本项得0分")

        # FCF vs 净利润
        if fcf is not None and 'net_income' in self.data:
            ni = self.data['net_income']
            fcf_ni_cfg = cfg.fcf_vs_ni
            if ni != 0:
                if fcf >= ni:
                    s = fcf_ni_cfg.weight
                    detail = f"FCF({fcf:.2f}) ≥ 净利润({ni})，盈利质量极佳"
                elif fcf >= ni * fcf_ni_cfg.thresholds[1] if len(fcf_ni_cfg.thresholds) > 1 else fcf >= ni * 0.8:
                    s = fcf_ni_cfg.weight * 0.6
                    detail = f"FCF/净利润 = {fcf/ni:.2f}，质量尚可"
                else:
                    s = fcf_ni_cfg.weight * 0.2
                    detail = f"FCF长期低于净利润，仅{fcf/ni:.2f}倍"
            else:
                s = 0.0
                detail = "净利润为负，不适用比较"
            score += s
            details.append(detail)
        else:
            details.append("未提供FCF或净利润，本项得0分")

        # FCF vs 分红+回购
        dpb = self.data.get('dividend_plus_buyback')
        if fcf is not None and dpb is not None:
            if fcf >= dpb:
                s = cfg.fcf_vs_dividend.weight
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
        """营运效率与议价能力"""
        cfg = self._config.efficiency
        max_score = self._config.weights.get("efficiency", 15.0)
        score = 0.0
        details = []

        # --------------------------------------------------
        # 应收账款周转天数（基于行业分位数）
        # --------------------------------------------------
        if self._is_fin_or_re:
            details.append(f"金融/地产行业，应收账款周转指标不适用，给基础分 {cfg.fin_re_base_score}/{cfg.receivables.weight}")
            score += cfg.fin_re_base_score
        elif 'avg_receivables' in self.data and 'revenue' in self.data:
            rec_days = 365 * self.data['avg_receivables'] / self.data['revenue']
            s, detail = score_by_percentile(
                rec_days, "receivables_days", self.industry_type,
                max_score=cfg.receivables.weight, higher_better=False,
            )
            detail = f"{detail}（{get_percentile_info(self.industry_type, 'receivables_days')}）"
            score += s
            details.append(detail)
        else:
            details.append("未提供应收账款或营收，本项得0分")

        # --------------------------------------------------
        # 存货周转天数（基于行业分位数，白酒特例）
        # --------------------------------------------------
        if self._is_fin_or_re:
            details.append(f"金融/地产行业，存货周转指标不适用，给基础分 {cfg.fin_re_base_score}/{cfg.inventory.weight}")
            score += cfg.fin_re_base_score
        elif self._is_liquor:
            # 白酒行业：存货越久越值钱，基酒储备是核心资产
            if 'avg_inventory' in self.data and 'cost_of_goods_sold' in self.data:
                inv_days = 365 * self.data['avg_inventory'] / self.data['cost_of_goods_sold']
                if inv_days < cfg.liquor_base_reserve_threshold:
                    s = cfg.inventory.weight * 0.6
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备充足性一般"
                elif inv_days < cfg.liquor_abundant_threshold:
                    s = cfg.inventory.weight
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备充足，越陈越香"
                else:
                    s = cfg.inventory.weight
                    detail = f"白酒行业存货周转天数 = {inv_days:.1f}天，基酒储备极为丰富"
                score += s
                details.append(detail)
            else:
                details.append("未提供存货或营业成本，本项得0分")
        elif 'avg_inventory' in self.data and 'cost_of_goods_sold' in self.data:
            inv_days = 365 * self.data['avg_inventory'] / self.data['cost_of_goods_sold']
            s, detail = score_by_percentile(
                inv_days, "inventory_days", self.industry_type,
                max_score=cfg.inventory.weight, higher_better=False,
            )
            detail = f"{detail}（{get_percentile_info(self.industry_type, 'inventory_days')}）"
            score += s
            details.append(detail)
        else:
            details.append("未提供存货或营业成本，本项得0分")

        # --------------------------------------------------
        # 应付账款周转天数（基于行业分位数，议价能力）
        # --------------------------------------------------
        if self._is_fin_or_re:
            details.append(f"金融/地产行业，应付账款周转指标不适用，给基础分 {cfg.fin_re_base_score}/{cfg.payables.weight}")
            score += cfg.fin_re_base_score
        elif 'avg_payables' in self.data and 'cost_of_goods_sold' in self.data:
            ap_days = 365 * self.data['avg_payables'] / self.data['cost_of_goods_sold']
            s, detail = score_by_percentile(
                ap_days, "payables_days", self.industry_type,
                max_score=cfg.payables.weight, higher_better=True,
            )
            detail = f"{detail}（{get_percentile_info(self.industry_type, 'payables_days')}）"
            score += s
            details.append(detail)
        else:
            details.append("未提供应付账款或营业成本，本项得0分")

        self._add_reason("营运效率与议价能力", score, max_score, "；".join(details))
        return score

    def score_duPont(self) -> float:
        """杜邦分析 – ROE驱动力质量与利润可持续性"""
        cfg = self._config.dupont
        max_score = self._config.weights.get("dupont", 20.0)
        score = 0.0
        details = []

        npm = self.data.get('net_profit_margin')
        deducted_npm = self.data.get('deducted_net_profit_margin')
        turnover = self.data.get('asset_turnover')
        lev = self.data.get('equity_multiplier')
        roe = self.data.get('roe')

        if npm is not None and turnover is not None and lev is not None:
            # ==========================================
            # 1. 可持续利润率：基于行业分位数的动态阈值
            # ==========================================
            if deducted_npm is not None:
                s_npm, detail_npm = score_by_percentile(
                    deducted_npm, "deducted_net_profit_margin",
                    self.industry_type, max_score=cfg.deducted_weight, higher_better=True,
                )
                detail_npm = f"{detail_npm}（{get_percentile_info(self.industry_type, 'deducted_net_profit_margin')}）"
            else:
                # 回退：无扣非数据时使用净利润率
                s_npm, detail_npm = score_by_percentile(
                    npm, "net_profit_margin",
                    self.industry_type, max_score=cfg.fallback_weight, higher_better=True,
                )
                detail_npm = f"无扣非数据，{detail_npm}（{get_percentile_info(self.industry_type, 'net_profit_margin')}）"

            # ==========================================
            # 2. 利润质量：非经常性损益占比（绝对标准，不依赖行业分位）
            # ==========================================
            net_income = self.data.get('net_income')
            deducted_net = self.data.get('deducted_net_profit')
            pq_cfg = cfg.profit_quality
            if net_income is not None and deducted_net is not None and net_income != 0:
                non_recurring = net_income - deducted_net
                non_recurring_ratio = abs(non_recurring) / abs(net_income) * 100

                if deducted_net < 0 and net_income > 0:
                    s_quality = 0.0
                    detail_quality = (
                        f"净利润{net_income:.2f}元但扣非净利润{deducted_net:.2f}元为负，"
                        f"利润完全依赖非经常性损益，质量极差"
                    )
                else:
                    s_quality = _get_default_threshold_score(
                        non_recurring_ratio, pq_cfg.thresholds,
                        [pq_cfg.weight, pq_cfg.weight * 0.75, pq_cfg.weight * 0.25, 0.0])

                    if non_recurring_ratio < pq_cfg.thresholds[0]:
                        detail_quality = f"非经常性损益占比{non_recurring_ratio:.1f}%，利润质量高"
                    elif non_recurring_ratio <= pq_cfg.thresholds[1]:
                        detail_quality = f"非经常性损益占比{non_recurring_ratio:.1f}%，利润质量尚可"
                    elif non_recurring_ratio <= pq_cfg.thresholds[2]:
                        detail_quality = f"非经常性损益占比{non_recurring_ratio:.1f}%，偏高需关注"
                    else:
                        detail_quality = f"非经常性损益占比{non_recurring_ratio:.1f}%，利润质量差"
            else:
                s_quality = cfg.missing_deducted_score
                detail_quality = f"缺少扣非净利润数据，无法评估利润质量，给基础分{cfg.missing_deducted_score}/{pq_cfg.weight}"

            # ==========================================
            # 3. 资产周转率：基于行业分位数的动态阈值
            # ==========================================
            s_turn, detail_turn = score_by_percentile(
                turnover, "asset_turnover",
                self.industry_type, max_score=cfg.asset_turnover.weight, higher_better=True,
            )
            detail_turn = f"{detail_turn}（{get_percentile_info(self.industry_type, 'asset_turnover')}）"

            # ==========================================
            # 4. 杠杆质量：基于行业分位数的动态阈值
            # ==========================================
            s_lev, detail_lev = score_by_percentile(
                lev, "equity_multiplier",
                self.industry_type, max_score=cfg.leverage.weight, higher_better=False,
            )
            detail_lev = f"{detail_lev}（{get_percentile_info(self.industry_type, 'equity_multiplier')}）"

            score = s_npm + s_quality + s_turn + s_lev
            details.append(detail_npm)
            details.append(detail_quality)
            details.append(detail_turn)
            details.append(detail_lev)
        else:
            if roe is not None:
                if roe > cfg.fallback_roe_high:
                    details.append(f"ROE为{roe:.1f}%，但缺乏杜邦分解，无法判断驱动质量，给基础分{cfg.fallback_score_high}/{max_score}")
                    score = cfg.fallback_score_high
                elif roe > cfg.fallback_roe_mid:
                    score = cfg.fallback_score_mid
                    details.append(f"ROE为{roe:.1f}%，缺乏分解，给{cfg.fallback_score_mid}分")
                else:
                    score = cfg.fallback_score_low
                    details.append(f"ROE为{roe:.1f}%，较低且缺乏分解")
            else:
                details.append("未提供杜邦分解数据或ROE，本项得0分")
                score = 0.0

        self._add_reason("杜邦驱动质量", score, max_score, "；".join(details))
        return score

    def score_valuation(self) -> float:
        """估值合理性"""
        cfg = self._config.valuation
        max_score = self._config.weights.get("valuation", 15.0)
        score = 0.0
        details = []

        # 市盈率相对行业
        if 'market_cap' in self.data and 'net_income' in self.data:
            ni = self.data['net_income']
            if ni > 0:
                pe = self.data['market_cap'] / ni
                industry_pe = self.data.get('industry_pe')
                if industry_pe is not None:
                    if pe < industry_pe:
                        s = cfg.pe_industry.weight
                        detail = f"PE = {pe:.1f}，低于行业均值{industry_pe:.1f}，估值有吸引力"
                    elif pe < industry_pe * cfg.premium_ratio:
                        s = cfg.pe_industry.weight * 0.6
                        detail = f"PE = {pe:.1f}，接近行业均值，合理"
                    else:
                        s = cfg.pe_industry.weight * 0.2
                        detail = f"PE = {pe:.1f}，显著高于行业{industry_pe:.1f}，可能高估"
                else:
                    # 无行业PE均值时，使用行业分位数（自动适应不同利率环境下的市场整体估值水平）
                    s, detail = score_by_percentile(
                        pe, "pe", self.industry_type, max_score=cfg.pe_industry.weight, higher_better=False,
                    )
                    detail = f"{detail}（{get_percentile_info(self.industry_type, 'pe')}）"
                score += s
                details.append(detail)
            else:
                details.append("净利润为负，无法使用PE")
        else:
            details.append("未提供市值或净利润，无法计算PE")

        # DCF估值区间
        #    注意：DCF模型依赖简化假设（WACC=10%, 终值增长率=3%, 营收增速推导FCF增长）
        #    区间过宽或与股价偏离过大时降低置信度
        if 'current_price' in self.data and 'dcf_low' in self.data and 'dcf_high' in self.data:
            price = self.data['current_price']
            low = self.data['dcf_low']
            high = self.data['dcf_high']
            dcf_range_ratio = high / low if low > 0 else 0

            # 合理性检验
            if dcf_range_ratio > cfg.dcf_range_ratio_limit:
                if price < high:
                    s = cfg.dcf.weight * 0.4
                    detail = (
                        f"股价{price}在DCF区间[{low},{high}]内，"
                        f"但区间跨度{dcf_range_ratio:.1f}倍过宽，"
                        f"模型对增长假设高度敏感，参考价值有限"
                    )
                else:
                    s = cfg.dcf.weight * 0.2
                    detail = (
                        f"DCF区间[{low},{high}]跨度{dcf_range_ratio:.1f}倍过宽，"
                        f"参考价值有限；股价{price}高于上限"
                    )
            elif low > price * cfg.dcf_price_low_ratio:
                s = cfg.dcf.weight * 0.4
                detail = (
                    f"DCF下限{low}为股价{price}的{low/price:.1f}倍，"
                    f"估值模型可能与当前市场定价严重脱节，参考价值有限"
                )
            elif high < price * cfg.dcf_price_high_ratio:
                s = cfg.dcf.weight * 0.4
                detail = (
                    f"DCF上限{high}仅股价{price}的{high/price:.1%}，"
                    f"估值模型可能过于保守，参考价值有限"
                )
            elif price < low:
                s = cfg.dcf.weight
                detail = (
                    f"股价{price}低于DCF下限{low}，明显低估"
                    f"（基于WACC=10%、终值增长率=3%的简化两阶段模型）"
                )
            elif price < high:
                s = cfg.dcf.weight * 0.6
                detail = (
                    f"股价{price}在DCF区间[{low},{high}]内，合理"
                    f"（基于WACC=10%、终值增长率=3%的简化两阶段模型）"
                )
            else:
                s = 0.0
                detail = f"股价{price}高于DCF上限{high}，高估"
            score += s
            details.append(detail)
        else:
            details.append("未提供当前股价或DCF区间，本项得0分")

        # P/B 结合行业分位数
        if 'pb' in self.data and 'is_asset_heavy' in self.data:
            pb = self.data['pb']
            heavy = self.data['is_asset_heavy']
            s, detail = score_by_percentile(
                pb, "pb", self.industry_type, max_score=cfg.pb.weight, higher_better=False,
            )
            asset_label = "重资产" if heavy else "轻资产"
            detail = f"{asset_label}公司，{detail}（{get_percentile_info(self.industry_type, 'pb')}）"
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

        # 创建 data 副本，避免副作用累积
        original_data = self.data
        self.data = {**self.data, **flat_data}
        try:
            # 数据完备性检查
            completeness = self._check_data_completeness()
            total, module_scores, reasons = self.full_score()
            return {
                "total_score": total,
                "dimension_scores": module_scores,
                "rating": self._rating(total),
                "reasons": reasons,
                "data_completeness": completeness,
            }
        finally:
            self.data = original_data

    def _check_data_completeness(self) -> Dict[str, Any]:
        """检查数据完备性，返回完备性比例和缺失字段列表"""
        dc_cfg = self._config.data_completeness
        required_fields = dc_cfg.required_fields

        all_required = []
        for module_fields in required_fields.values():
            all_required.extend(module_fields)

        available = sum(1 for f in all_required if f in self.data and self.data[f] is not None)
        total = len(all_required)
        ratio = available / total if total > 0 else 1.0

        missing = [f for f in all_required if f not in self.data or self.data[f] is None]

        module_completeness = {}
        for module, fields in required_fields.items():
            mod_avail = sum(1 for f in fields if f in self.data and self.data[f] is not None)
            module_completeness[module] = mod_avail / len(fields) if fields else 1.0

        return {
            "overall_ratio": round(ratio, 3),
            "is_sufficient": ratio >= dc_cfg.min_ratio,
            "missing_fields": missing,
            "module_ratios": module_completeness,
        }

    @staticmethod
    def _rating(total: float) -> str:
        """根据总分返回评级"""
        config = get_scorer_config()
        thresholds = config.rating_thresholds
        labels = config.rating_labels
        for i, t in enumerate(thresholds):
            if total >= t:
                return labels[i]
        return labels[-1] if len(labels) > len(thresholds) else "较差"

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