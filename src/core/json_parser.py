# -*- coding: utf-8 -*-
"""
===================================
统一的JSON解析工具 - 支持多种LLM返回格式
===================================

职责：
1. 统一解析不同LLM返回的JSON格式
2. 支持中英文键名兼容
3. 智能提取关键信息
4. 为分析器和Agent模式提供统一的解析接口
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from json_repair import repair_json

logger = logging.getLogger(__name__)


def clean_and_extract_json(response_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    清理响应文本并提取JSON内容
    
    Args:
        response_text: LLM返回的原始响应文本
        
    Returns:
        Tuple[清理后的JSON字符串, 解析后的字典数据]
    """
    try:
        cleaned_text = response_text
        
        if '```json' in cleaned_text:
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
        elif '```' in cleaned_text:
            cleaned_text = cleaned_text.replace('```', '')
        
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned_text[json_start:json_end]
            json_str = fix_common_json_issues(json_str)
            data = json.loads(json_str)
            logger.info(f"成功解析JSON，顶层键: {list(data.keys())}")
            return json_str, data
        else:
            logger.warning("未找到有效的JSON内容")
            return None, None
    except Exception as e:
        logger.error(f"提取JSON失败: {e}", exc_info=True)
        return None, None


def fix_common_json_issues(json_str: str) -> str:
    """修复常见的JSON格式问题"""
    import re
    
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = json_str.replace('True', 'true').replace('False', 'false')
    json_str = repair_json(json_str)
    
    return json_str


def get_smart_value(data: Dict[str, Any], chinese_key: str, english_key: str, default: Any = None) -> Any:
    """
    智能获取值 - 优先中文键名，其次英文键名
    
    Args:
        data: 数据字典
        chinese_key: 中文键名
        english_key: 英文键名
        default: 默认值
        
    Returns:
        获取到的值或默认值
    """
    if chinese_key in data:
        return data[chinese_key]
    if english_key in data:
        return data[english_key]
    return default


def extract_stock_name(data: Dict[str, Any], default_name: str = "", code: str = "") -> str:
    """
    从数据中智能提取股票名称
    
    Args:
        data: JSON数据
        default_name: 默认名称
        code: 股票代码
        
    Returns:
        提取的股票名称
    """
    name = get_smart_value(data, "股票名称", "stock_name", default_name)
    
    if name and (default_name.startswith('股票') or default_name == code or 'Unknown' in default_name):
        return str(name).strip()
    
    return default_name


def extract_one_sentence_decision(data: Dict[str, Any], default: str = "观望，等待明确信号") -> str:
    """
    从数据中智能提取一句话决策
    
    Args:
        data: JSON数据
        default: 默认值
        
    Returns:
        一句话决策
    """
    one_sentence = get_smart_value(data, "核心结论", "one_sentence", "")
    
    if not one_sentence and 'core_conclusion' in data:
        if isinstance(data['core_conclusion'], dict):
            one_sentence = data['core_conclusion'].get('one_sentence', '')
    
    if not one_sentence:
        one_sentence = data.get('investment_recommendation', {}).get('rating', '')
    if not one_sentence:
        one_sentence = data.get('investment_recommendation', {}).get('short_term', '')
    if not one_sentence:
        one_sentence = data.get('investment_decision', {}).get('short_term', '')
    if not one_sentence:
        one_sentence = data.get('investment_decision', {}).get('short_term_recommendation', '')
    if not one_sentence:
        one_sentence = data.get('decision_dashboard', {}).get('short_term_recommendation', '')
    
    if not one_sentence:
        one_sentence = default
    
    return str(one_sentence).strip()


def extract_position_advice(data: Dict[str, Any], one_sentence: str = "") -> Tuple[str, str]:
    """
    从数据中提取持仓分类建议
    
    Args:
        data: JSON数据
        one_sentence: 一句话决策（作为备选）
        
    Returns:
        Tuple[空仓者建议, 持仓者建议]
    """
    position_advice = data.get('持仓分类建议', {})
    
    if not position_advice and 'core_conclusion' in data:
        if isinstance(data['core_conclusion'], dict):
            position_advice = data['core_conclusion'].get('position_advice', {})
    
    no_position = position_advice.get('空仓者', one_sentence)
    has_position = position_advice.get('持仓者', one_sentence)
    
    if not no_position and 'no_position' in position_advice:
        no_position = position_advice['no_position']
    if not has_position and 'has_position' in position_advice:
        has_position = position_advice['has_position']
    
    return str(no_position).strip(), str(has_position).strip()


def extract_sniper_points(data: Dict[str, Any]) -> Dict[str, str]:
    """
    从数据中提取狙击点位
    
    Args:
        data: JSON数据
        
    Returns:
        狙击点位置字典
    """
    sniper_points = data.get('具体狙击点位', {})
    
    if not sniper_points and 'battle_plan' in data:
        if isinstance(data['battle_plan'], dict):
            sniper_points = data['battle_plan'].get('sniper_points', {})
    
    if not sniper_points:
        sniper_points = data.get('investment_recommendation', {})
    
    buy_price = str(sniper_points.get('买入价', sniper_points.get('ideal_buy', ''))).strip()
    stop_loss = str(sniper_points.get('止损价', sniper_points.get('stop_loss', ''))).strip()
    target_price = str(sniper_points.get('目标价', sniper_points.get('take_profit', ''))).strip()
    secondary_buy = str(sniper_points.get('次优买入价', sniper_points.get('secondary_buy', ''))).strip()
    
    return {
        'ideal_buy': buy_price,
        'secondary_buy': secondary_buy,
        'stop_loss': stop_loss,
        'take_profit': target_price
    }


def extract_checklist(data: Dict[str, Any]) -> Dict[str, str]:
    """
    从数据中提取检查清单
    
    Args:
        data: JSON数据
        
    Returns:
        检查清单字典
    """
    checklist = data.get('检查清单', {})
    
    if not checklist and 'battle_plan' in data:
        if isinstance(data['battle_plan'], dict):
            action_list = data['battle_plan'].get('action_checklist', [])
            if isinstance(action_list, list):
                checklist = {}
                for item in action_list:
                    if isinstance(item, str):
                        if ':' in item:
                            parts = item.split(':', 1)
                            checklist[parts[0].strip()] = parts[1].strip()
                        else:
                            checklist[f'检查项{len(checklist)+1}'] = item
    
    if not isinstance(checklist, dict):
        checklist = {}
    
    return checklist


def determine_signal_and_score(one_sentence: str) -> Tuple[str, int, str, str, str]:
    """
    根据一句话决策确定信号类型、评分、操作建议、决策类型和趋势预测
    
    Args:
        one_sentence: 一句话决策
        
    Returns:
        Tuple[信号类型, 评分, 操作建议, 决策类型, 趋势预测]
    """
    signal_type = '🟡持有观望'
    sentiment_score = 50
    operation_advice = '观望'
    decision_type = 'hold'
    trend_prediction = '震荡'
    
    one_sentence_lower = str(one_sentence).lower()
    
    if '买入' in one_sentence or 'buy' in one_sentence_lower:
        signal_type = '🟢买入信号'
        sentiment_score = 75
        operation_advice = '买入'
        decision_type = 'buy'
        trend_prediction = '看多'
    elif '增持' in one_sentence or 'overweight' in one_sentence_lower:
        signal_type = '🟢加仓信号'
        sentiment_score = 65
        operation_advice = '持有'
        decision_type = 'buy'
        trend_prediction = '震荡偏多'
    elif '卖出' in one_sentence or 'sell' in one_sentence_lower:
        signal_type = '🔴卖出信号'
        sentiment_score = 25
        operation_advice = '卖出'
        decision_type = 'sell'
        trend_prediction = '看空'
    elif '减持' in one_sentence or 'underweight' in one_sentence_lower:
        signal_type = '🔴减仓信号'
        sentiment_score = 35
        operation_advice = '减仓'
        decision_type = 'sell'
        trend_prediction = '震荡偏空'
    elif '持有' in one_sentence or 'hold' in one_sentence_lower:
        signal_type = '🟡持有观望'
        sentiment_score = 50
        operation_advice = '持有'
        decision_type = 'hold'
        trend_prediction = '震荡'
    
    return signal_type, sentiment_score, operation_advice, decision_type, trend_prediction


def build_complete_dashboard(
    one_sentence: str,
    signal_type: str,
    no_position_advice: str,
    has_position_advice: str,
    sniper_points: Dict[str, str],
    checklist: Dict[str, str],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    构建完整的dashboard结构
    
    Args:
        one_sentence: 一句话决策
        signal_type: 信号类型
        no_position_advice: 空仓者建议
        has_position_advice: 持仓者建议
        sniper_points: 狙击点位
        checklist: 检查清单
        context: 上下文数据
        
    Returns:
        完整的dashboard结构
    """
    context = context or {}
    today = context.get('today', {})
    realtime = context.get('realtime', {})
    chip = context.get('chip', {})
    
    return {
        'core_conclusion': {
            'one_sentence': one_sentence,
            'signal_type': signal_type,
            'time_sensitivity': '本周内',
            'position_advice': {
                'no_position': no_position_advice,
                'has_position': has_position_advice
            }
        },
        'data_perspective': {
            'trend_status': {
                'ma_alignment': checklist.get('均线形态', checklist.get('均线多头排列', '')),
                'is_bullish': '多头' in checklist.get('均线形态', '') or '✅' in checklist.get('均线多头排列', ''),
                'trend_score': 50
            },
            'price_position': {
                'current_price': today.get('close', realtime.get('price', 0)),
                'ma5': today.get('ma5', realtime.get('ma5', 0)),
                'ma10': today.get('ma10', realtime.get('ma10', 0)),
                'ma20': today.get('ma20', realtime.get('ma20', 0)),
                'bias_ma5': 0,
                'bias_status': '安全',
                'support_level': sniper_points.get('stop_loss', 0),
                'resistance_level': sniper_points.get('take_profit', 0)
            },
            'volume_analysis': {
                'volume_ratio': realtime.get('volume_ratio', 0),
                'volume_status': checklist.get('量能配合', ''),
                'turnover_rate': realtime.get('turnover_rate', 0),
                'volume_meaning': checklist.get('量能配合', '')
            },
            'chip_structure': {
                'profit_ratio': chip.get('profit_ratio', 0),
                'avg_cost': chip.get('avg_cost', 0),
                'concentration': chip.get('concentration_90', 0),
                'chip_health': checklist.get('筹码结构', checklist.get('筹码结构健康', ''))
            }
        },
        'intelligence': {
            'latest_news': '',
            'risk_alerts': [f'{k}: {v}' for k, v in checklist.items()] if checklist else [],
            'positive_catalysts': [],
            'earnings_outlook': '',
            'sentiment_summary': checklist.get('消息面', '')
        },
        'battle_plan': {
            'sniper_points': {
                'ideal_buy': f'{sniper_points.get("ideal_buy", "")}元' if sniper_points.get("ideal_buy") else '',
                'secondary_buy': f'{sniper_points.get("secondary_buy", "")}元' if sniper_points.get("secondary_buy") else '',
                'stop_loss': f'{sniper_points.get("stop_loss", "")}元' if sniper_points.get("stop_loss") else '',
                'take_profit': f'{sniper_points.get("take_profit", "")}元' if sniper_points.get("take_profit") else ''
            },
            'position_strategy': {
                'suggested_position': '',
                'entry_plan': no_position_advice,
                'risk_control': has_position_advice
            },
            'action_checklist': [f'{k}: {v}' for k, v in checklist.items()] if checklist else []
        }
    }


def extract_key_points(checklist: Dict[str, str], max_points: int = 3) -> str:
    """
    从检查清单中提取关键要点
    
    Args:
        checklist: 检查清单
        max_points: 最大要点数量
        
    Returns:
        关键要点字符串
    """
    if not checklist:
        return ''
    return ', '.join([f'{k}: {v}' for k, v in list(checklist.items())[:max_points]])


def extract_risk_warning(checklist: Dict[str, str]) -> str:
    """
    从检查清单中提取风险警告
    
    Args:
        checklist: 检查清单
        
    Returns:
        风险警告字符串
    """
    if not checklist:
        return ''
    return '; '.join([f'{k}: {v}' for k, v in checklist.items()])


def is_aliyun_simple_format(data: Dict[str, Any]) -> bool:
    """
    检测是否是阿里云LLM返回的简单格式
    
    Args:
        data: JSON数据
        
    Returns:
        是否是阿里云简单格式
    """
    # 阿里云简单格式的特征
    has_aliyun_keys = (
        'stock_code' in data and
        'company_name' in data and
        'investment_recommendation' in data
    )
    # 同时没有我们期望的完整格式的键
    missing_full_format_keys = (
        'sentiment_score' not in data and
        'dashboard' not in data
    )
    return has_aliyun_keys and missing_full_format_keys


def parse_aliyun_simple_format(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    解析阿里云LLM返回的简单格式
    
    Args:
        data: 阿里云格式的JSON数据
        context: 上下文数据
        
    Returns:
        完整的分析结果结构
    """
    context = context or {}
    
    # 提取基本信息
    stock_code = data.get('stock_code', '')
    stock_name = data.get('company_name', '')
    industry = data.get('industry', '')
    
    # 提取投资建议作为一句话决策
    investment_recommendation = data.get('investment_recommendation', '观望，等待明确信号')
    risk_warning = data.get('risk_warning', '')
    
    # 从投资建议中确定信号和评分
    signal_type, sentiment_score, operation_advice, decision_type, trend_prediction = determine_signal_and_score(investment_recommendation)
    
    # 提取技术分析数据
    technical = data.get('technical_analysis', {})
    current_price = technical.get('current_price', 0)
    ma5 = technical.get('ma_5d', 0)
    ma10 = technical.get('ma_10d', 0)
    rsi = technical.get('rsi_14d', 0)
    volume_trend = technical.get('volume_trend', '')
    short_term_trend = technical.get('short_term_trend', '')
    
    # 提取财务分析数据
    financial = data.get('financial_analysis', {})
    revenue_growth = financial.get('revenue_growth_qoq', 0)
    net_profit_margin = financial.get('net_profit_margin', 0)
    debt_equity = financial.get('debt_equity_ratio', 0)
    
    # 提取新闻情绪
    news = data.get('news_sentiment', {})
    recent_news = news.get('recent_news', [])
    sentiment_score_num = news.get('sentiment_score', 0.5)
    
    # 提取行业趋势
    industry_trends = data.get('industry_trends', {})
    sector_performance = industry_trends.get('sector_performance', '')
    key_drivers = industry_trends.get('key_drivers', [])
    risk_factors = industry_trends.get('risk_factors', [])
    
    # 提取估值指标
    valuation = data.get('valuation_metrics', {})
    pe_ratio = valuation.get('pe_ratio', 0)
    pb_ratio = valuation.get('pb_ratio', 0)
    
    # 构建分析摘要
    analysis_summary = f"{stock_name}({stock_code})，{industry}行业。{investment_recommendation}"
    if risk_warning:
        analysis_summary += f" {risk_warning}"
    
    # 提取关键要点
    key_points = []
    if recent_news:
        key_points.extend(recent_news[:2])
    if key_drivers:
        key_points.extend([f"驱动因素: {d}" for d in key_drivers[:2]])
    key_points_str = ', '.join(key_points) if key_points else ''
    
    # 构建检查清单
    checklist = {}
    if ma5 and ma10:
        if ma5 > ma10:
            checklist['均线形态'] = 'MA5 > MA10，短期向好'
        else:
            checklist['均线形态'] = 'MA5 < MA10，需要观察'
    if rsi:
        if rsi > 70:
            checklist['RSI指标'] = f'RSI={rsi:.1f}，超买区域'
        elif rsi < 30:
            checklist['RSI指标'] = f'RSI={rsi:.1f}，超卖区域'
        else:
            checklist['RSI指标'] = f'RSI={rsi:.1f}，正常区间'
    if pe_ratio:
        checklist['估值水平'] = f'PE={pe_ratio:.1f}，PB={pb_ratio:.1f}'
    if volume_trend:
        checklist['量能趋势'] = volume_trend
    if short_term_trend:
        checklist['短期趋势'] = short_term_trend
    if revenue_growth:
        checklist['营收增长'] = f'{revenue_growth:.1f}%'
    if net_profit_margin:
        checklist['净利率'] = f'{net_profit_margin:.1f}%'
    if risk_factors:
        for rf in risk_factors[:3]:
            checklist[f'风险因素{len(checklist)+1}'] = rf
    
    # 构建狙击点位 - 从估值或技术面推断
    sniper_points = {}
    if current_price:
        sniper_points['ideal_buy'] = ''  # 简单格式没有具体价格
        sniper_points['secondary_buy'] = ''
        sniper_points['stop_loss'] = ''
        sniper_points['take_profit'] = ''
    
    # 构建完整的dashboard
    dashboard = build_complete_dashboard(
        one_sentence=investment_recommendation,
        signal_type=signal_type,
        no_position_advice=investment_recommendation,
        has_position_advice=investment_recommendation,
        sniper_points=sniper_points,
        checklist=checklist,
        context=context
    )
    
    # 返回完整结构
    return {
        'stock_name': stock_name,
        'sentiment_score': sentiment_score,
        'trend_prediction': trend_prediction,
        'operation_advice': operation_advice,
        'decision_type': decision_type,
        'confidence_level': '中',
        'dashboard': dashboard,
        'analysis_summary': analysis_summary,
        'key_points': key_points_str,
        'risk_warning': risk_warning,
        'buy_reason': '',
        'trend_analysis': short_term_trend,
        'short_term_outlook': short_term_trend,
        'medium_term_outlook': '',
        'technical_analysis': json.dumps(technical, ensure_ascii=False) if technical else '',
        'ma_analysis': '',
        'volume_analysis': volume_trend,
        'pattern_analysis': '',
        'fundamental_analysis': json.dumps(financial, ensure_ascii=False) if financial else '',
        'sector_position': sector_performance,
        'company_highlights': '',
        'news_summary': '; '.join(recent_news) if recent_news else '',
        'market_sentiment': json.dumps({'sentiment_score': sentiment_score_num}, ensure_ascii=False) if sentiment_score_num else '',
        'hot_topics': ''
    }
