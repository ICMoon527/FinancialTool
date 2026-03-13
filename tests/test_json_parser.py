# -*- coding: utf-8 -*-
"""
单元测试 - JSON解析工具
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.json_parser import (
    clean_and_extract_json,
    extract_stock_name,
    extract_one_sentence_decision,
    extract_position_advice,
    extract_sniper_points,
    extract_checklist,
    determine_signal_and_score,
    build_complete_dashboard,
    extract_key_points,
    extract_risk_warning,
)


class TestJSONParser(unittest.TestCase):
    """测试JSON解析工具函数"""
    
    def test_clean_and_extract_json_with_chinese_keys(self):
        """测试提取中文键名的JSON"""
        response_text = '''```json
{
    "股票名称": "贵州茅台",
    "核心结论": "买入，短期看高一线",
    "持仓分类建议": {
        "空仓者": "建议买入",
        "持仓者": "建议持有"
    },
    "具体狙击点位": {
        "买入价": "1800",
        "止损价": "1750",
        "目标价": "1900"
    },
    "检查清单": {
        "均线形态": "多头排列",
        "量能配合": "良好"
    }
}
```'''
        json_str, data = clean_and_extract_json(response_text)
        self.assertIsNotNone(data)
        self.assertEqual(data['股票名称'], '贵州茅台')
        self.assertEqual(data['核心结论'], '买入，短期看高一线')
    
    def test_extract_stock_name_chinese(self):
        """测试提取中文键名的股票名称"""
        data = {'股票名称': '贵州茅台', 'stock_name': 'Moutai'}
        name = extract_stock_name(data, '股票600519', '600519')
        self.assertEqual(name, '贵州茅台')
    
    def test_extract_stock_name_english(self):
        """测试提取英文键名的股票名称"""
        data = {'stock_name': 'Tesla Inc'}
        name = extract_stock_name(data, '股票TSLA', 'TSLA')
        self.assertEqual(name, 'Tesla Inc')
    
    def test_extract_one_sentence_decision_chinese(self):
        """测试提取中文键名的一句话决策"""
        data = {'核心结论': '买入，短期看涨'}
        decision = extract_one_sentence_decision(data)
        self.assertEqual(decision, '买入，短期看涨')
    
    def test_extract_one_sentence_decision_english(self):
        """测试提取英文键名的一句话决策"""
        data = {'core_conclusion': {'one_sentence': 'Buy for short term gains'}}
        decision = extract_one_sentence_decision(data)
        self.assertEqual(decision, 'Buy for short term gains')
    
    def test_extract_one_sentence_decision_from_investment(self):
        """测试从investment_recommendation提取决策"""
        data = {'investment_recommendation': {'rating': '买入'}}
        decision = extract_one_sentence_decision(data)
        self.assertEqual(decision, '买入')
    
    def test_extract_position_advice_chinese(self):
        """测试提取中文键名的持仓建议"""
        data = {'持仓分类建议': {'空仓者': '建议买入', '持仓者': '建议持有'}}
        no_pos, has_pos = extract_position_advice(data, '观望')
        self.assertEqual(no_pos, '建议买入')
        self.assertEqual(has_pos, '建议持有')
    
    def test_extract_sniper_points_chinese(self):
        """测试提取中文键名的狙击点位"""
        data = {'具体狙击点位': {'买入价': '1800', '止损价': '1750', '目标价': '1900'}}
        points = extract_sniper_points(data)
        self.assertEqual(points['ideal_buy'], '1800')
        self.assertEqual(points['stop_loss'], '1750')
        self.assertEqual(points['take_profit'], '1900')
    
    def test_extract_checklist_chinese(self):
        """测试提取中文键名的检查清单"""
        data = {'检查清单': {'均线形态': '多头排列', '量能配合': '良好'}}
        checklist = extract_checklist(data)
        self.assertEqual(checklist['均线形态'], '多头排列')
        self.assertEqual(checklist['量能配合'], '良好')
    
    def test_determine_signal_buy(self):
        """测试确定买入信号"""
        signal, score, advice, decision, trend = determine_signal_and_score('买入')
        self.assertEqual(signal, '🟢买入信号')
        self.assertEqual(score, 75)
        self.assertEqual(advice, '买入')
        self.assertEqual(decision, 'buy')
        self.assertEqual(trend, '看多')
    
    def test_determine_signal_hold(self):
        """测试确定持有信号"""
        signal, score, advice, decision, trend = determine_signal_and_score('持有观望')
        self.assertEqual(signal, '🟡持有观望')
        self.assertEqual(score, 50)
        self.assertEqual(advice, '持有')
        self.assertEqual(decision, 'hold')
        self.assertEqual(trend, '震荡')
    
    def test_determine_signal_sell(self):
        """测试确定卖出信号"""
        signal, score, advice, decision, trend = determine_signal_and_score('卖出')
        self.assertEqual(signal, '🔴卖出信号')
        self.assertEqual(score, 25)
        self.assertEqual(advice, '卖出')
        self.assertEqual(decision, 'sell')
        self.assertEqual(trend, '看空')
    
    def test_build_complete_dashboard(self):
        """测试构建完整的dashboard"""
        dashboard = build_complete_dashboard(
            '买入',
            '🟢买入信号',
            '建议买入',
            '建议持有',
            {'ideal_buy': '1800', 'stop_loss': '1750', 'take_profit': '1900'},
            {'均线形态': '多头排列', '量能配合': '良好'}
        )
        self.assertIsNotNone(dashboard)
        self.assertIn('core_conclusion', dashboard)
        self.assertIn('data_perspective', dashboard)
        self.assertIn('intelligence', dashboard)
        self.assertIn('battle_plan', dashboard)
        self.assertEqual(dashboard['core_conclusion']['one_sentence'], '买入')
    
    def test_extract_key_points(self):
        """测试提取关键要点"""
        checklist = {'均线形态': '多头排列', '量能配合': '良好', '筹码结构': '健康'}
        key_points = extract_key_points(checklist)
        self.assertIn('均线形态', key_points)
        self.assertIn('量能配合', key_points)
    
    def test_extract_risk_warning(self):
        """测试提取风险警告"""
        checklist = {'消息面': '有减持公告', '技术面': '破位风险'}
        warning = extract_risk_warning(checklist)
        self.assertIn('消息面', warning)
        self.assertIn('技术面', warning)


if __name__ == '__main__':
    unittest.main()
