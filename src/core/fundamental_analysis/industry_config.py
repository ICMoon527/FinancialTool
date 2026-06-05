# -*- coding: utf-8 -*-
"""
===================================
行业配置与阈值模板
===================================

职责：
1. 定义行业分类体系（8 大类别 + 特殊子类）
2. 将数据库中的行业标签映射到统一分类
3. 为每个行业提供营运效率评分阈值模板
4. 提供行业合理 PE 区间配置（从 industry_percentiles.yaml 读取）
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import yaml

logger = logging.getLogger(__name__)


class IndustryType(str, Enum):
    """行业分类枚举"""
    CONSUMER = "消费"          # 消费/品牌（白酒、食品饮料、家电、服装等）
    PHARMA = "医药"           # 医药生物（化学制药、中药、医疗器械等）
    TECH = "科技"             # TMT/科技（电子、计算机、通信、半导体等）
    MANUFACTURING = "制造"    # 制造/工业（机械设备、电力设备、军工等）
    RESOURCE = "资源"         # 资源/周期（煤炭、钢铁、有色金属、化工等）
    FINANCE = "金融"          # 金融（银行、证券、保险等）
    REAL_ESTATE = "地产"      # 地产/基建（房地产开发、建筑装饰等）
    UTILITY = "公用事业"      # 公用事业（电力、燃气、水务等）
    UNKNOWN = "未知"          # 无法识别的行业


# ============================================================
# 行业关键词映射
# 将东方财富行业标签中的关键词映射到 IndustryType
# 匹配优先级：从上到下，第一个匹配的生效
# ============================================================

INDUSTRY_KEYWORD_MAP: List[Tuple[str, IndustryType]] = [
    # 金融
    ("银行", IndustryType.FINANCE),
    ("保险", IndustryType.FINANCE),
    ("证券", IndustryType.FINANCE),
    ("多元金融", IndustryType.FINANCE),
    ("非银金融", IndustryType.FINANCE),
    ("金融控股", IndustryType.FINANCE),
    ("信托", IndustryType.FINANCE),
    ("期货", IndustryType.FINANCE),
    # 地产/基建
    ("房地产", IndustryType.REAL_ESTATE),
    ("房地产开发", IndustryType.REAL_ESTATE),
    ("住宅开发", IndustryType.REAL_ESTATE),
    ("商业地产", IndustryType.REAL_ESTATE),
    ("建筑装饰", IndustryType.REAL_ESTATE),
    ("工程建设", IndustryType.REAL_ESTATE),
    ("基础建设", IndustryType.REAL_ESTATE),
    ("装修建材", IndustryType.REAL_ESTATE),
    ("水泥", IndustryType.REAL_ESTATE),
    ("房屋建设", IndustryType.REAL_ESTATE),
    ("基建", IndustryType.REAL_ESTATE),
    # 资源/周期
    ("煤炭", IndustryType.RESOURCE),
    ("钢铁", IndustryType.RESOURCE),
    ("有色金属", IndustryType.RESOURCE),
    ("基础化工", IndustryType.RESOURCE),
    ("化学原料", IndustryType.RESOURCE),
    ("化学制品", IndustryType.RESOURCE),
    ("石油石化", IndustryType.RESOURCE),
    ("炼化", IndustryType.RESOURCE),
    ("采掘", IndustryType.RESOURCE),
    ("贵金属", IndustryType.RESOURCE),
    ("小金属", IndustryType.RESOURCE),
    ("能源金属", IndustryType.RESOURCE),
    ("普钢", IndustryType.RESOURCE),
    ("特钢", IndustryType.RESOURCE),
    ("农化", IndustryType.RESOURCE),
    ("农药", IndustryType.RESOURCE),
    ("化肥", IndustryType.RESOURCE),
    ("磷化工", IndustryType.RESOURCE),
    ("有机硅", IndustryType.RESOURCE),
    ("氟化工", IndustryType.RESOURCE),
    ("氯碱", IndustryType.RESOURCE),
    ("纯碱", IndustryType.RESOURCE),
    # 公用事业
    ("电力", IndustryType.UTILITY),
    ("燃气", IndustryType.UTILITY),
    ("水务", IndustryType.UTILITY),
    ("环保", IndustryType.UTILITY),
    ("固废治理", IndustryType.UTILITY),
    ("环境治理", IndustryType.UTILITY),
    ("热力", IndustryType.UTILITY),
    ("水力发电", IndustryType.UTILITY),
    ("火力发电", IndustryType.UTILITY),
    ("风力发电", IndustryType.UTILITY),
    ("光伏发电", IndustryType.UTILITY),
    ("生物质能", IndustryType.UTILITY),
    ("核能", IndustryType.UTILITY),
    ("绿色电力", IndustryType.UTILITY),
    ("公用事业", IndustryType.UTILITY),
    # 医药
    ("医药", IndustryType.PHARMA),
    ("制药", IndustryType.PHARMA),
    ("中药", IndustryType.PHARMA),
    ("生物制品", IndustryType.PHARMA),
    ("医疗器械", IndustryType.PHARMA),
    ("医疗", IndustryType.PHARMA),
    ("化学制剂", IndustryType.PHARMA),
    ("原料药", IndustryType.PHARMA),
    ("医药商业", IndustryType.PHARMA),
    ("医药流通", IndustryType.PHARMA),
    ("疫苗", IndustryType.PHARMA),
    ("CRO", IndustryType.PHARMA),
    ("医院", IndustryType.PHARMA),
    ("医疗服务", IndustryType.PHARMA),
    ("动物保健", IndustryType.PHARMA),
    # 消费/品牌（含白酒特殊处理）
    ("白酒", IndustryType.CONSUMER),
    ("食品饮料", IndustryType.CONSUMER),
    ("食品", IndustryType.CONSUMER),
    ("饮料", IndustryType.CONSUMER),
    ("乳", IndustryType.CONSUMER),
    ("调味", IndustryType.CONSUMER),
    ("休闲食品", IndustryType.CONSUMER),
    ("零食", IndustryType.CONSUMER),
    ("化妆品", IndustryType.CONSUMER),
    ("美容", IndustryType.CONSUMER),
    ("服装", IndustryType.CONSUMER),
    ("纺织", IndustryType.CONSUMER),
    ("家纺", IndustryType.CONSUMER),
    ("鞋帽", IndustryType.CONSUMER),
    ("饰品", IndustryType.CONSUMER),
    ("家用电器", IndustryType.CONSUMER),
    ("白色家电", IndustryType.CONSUMER),
    ("黑色家电", IndustryType.CONSUMER),
    ("小家电", IndustryType.CONSUMER),
    ("厨房", IndustryType.CONSUMER),
    ("家居", IndustryType.CONSUMER),
    ("家具", IndustryType.CONSUMER),
    ("造纸", IndustryType.CONSUMER),
    ("包装", IndustryType.CONSUMER),
    ("印刷", IndustryType.CONSUMER),
    ("个护", IndustryType.CONSUMER),
    ("宠物", IndustryType.CONSUMER),
    ("文娱", IndustryType.CONSUMER),
    ("体育", IndustryType.CONSUMER),
    ("教育", IndustryType.CONSUMER),
    ("出版", IndustryType.CONSUMER),
    ("传媒", IndustryType.CONSUMER),
    ("广告", IndustryType.CONSUMER),
    ("影视", IndustryType.CONSUMER),
    ("游戏", IndustryType.CONSUMER),
    ("互联网电商", IndustryType.CONSUMER),
    ("新零售", IndustryType.CONSUMER),
    ("一般零售", IndustryType.CONSUMER),
    ("百货", IndustryType.CONSUMER),
    ("超市", IndustryType.CONSUMER),
    ("商贸", IndustryType.CONSUMER),
    ("旅游", IndustryType.CONSUMER),
    ("酒店", IndustryType.CONSUMER),
    ("餐饮", IndustryType.CONSUMER),
    # TMT/科技
    ("电子", IndustryType.TECH),
    ("计算机", IndustryType.TECH),
    ("通信", IndustryType.TECH),
    ("半导体", IndustryType.TECH),
    ("软件开发", IndustryType.TECH),
    ("IT服务", IndustryType.TECH),
    ("互联网服务", IndustryType.TECH),
    ("消费电子", IndustryType.TECH),
    ("元件", IndustryType.TECH),
    ("面板", IndustryType.TECH),
    ("LED", IndustryType.TECH),
    ("光学", IndustryType.TECH),
    ("安防设备", IndustryType.TECH),
    ("印制电路板", IndustryType.TECH),
    ("集成电路", IndustryType.TECH),
    ("芯片", IndustryType.TECH),
    ("分立器件", IndustryType.TECH),
    ("被动元件", IndustryType.TECH),
    ("模拟芯片", IndustryType.TECH),
    ("数字芯片", IndustryType.TECH),
    ("传感器", IndustryType.TECH),
    ("军工电子", IndustryType.TECH),
    ("通信设备", IndustryType.TECH),
    ("通信网络", IndustryType.TECH),
    ("通信服务", IndustryType.TECH),
    ("数据中心", IndustryType.TECH),
    ("云计算", IndustryType.TECH),
    ("大数据", IndustryType.TECH),
    ("人工智能", IndustryType.TECH),
    ("区块链", IndustryType.TECH),
    ("物联网", IndustryType.TECH),
    ("信创", IndustryType.TECH),
    ("网络安全", IndustryType.TECH),
    ("软件", IndustryType.TECH),
    ("游戏", IndustryType.TECH),
    # 制造/工业（放最后兜底）
    ("机械", IndustryType.MANUFACTURING),
    ("设备", IndustryType.MANUFACTURING),
    ("电力设备", IndustryType.MANUFACTURING),
    ("电网设备", IndustryType.MANUFACTURING),
    ("输变电", IndustryType.MANUFACTURING),
    ("配电", IndustryType.MANUFACTURING),
    ("电机", IndustryType.MANUFACTURING),
    ("仪器仪表", IndustryType.MANUFACTURING),
    ("自动化", IndustryType.MANUFACTURING),
    ("机器人", IndustryType.MANUFACTURING),
    ("通用设备", IndustryType.MANUFACTURING),
    ("专用设备", IndustryType.MANUFACTURING),
    ("工程机械", IndustryType.MANUFACTURING),
    ("轨交", IndustryType.MANUFACTURING),
    ("航空装备", IndustryType.MANUFACTURING),
    ("航天", IndustryType.MANUFACTURING),
    ("航海装备", IndustryType.MANUFACTURING),
    ("船舶", IndustryType.MANUFACTURING),
    ("军工", IndustryType.MANUFACTURING),
    ("汽车零部件", IndustryType.MANUFACTURING),
    ("汽车电子", IndustryType.MANUFACTURING),
    ("车身", IndustryType.MANUFACTURING),
    ("底盘", IndustryType.MANUFACTURING),
    ("轮胎", IndustryType.MANUFACTURING),
    ("电池", IndustryType.MANUFACTURING),
    ("锂电", IndustryType.MANUFACTURING),
    ("光伏", IndustryType.MANUFACTURING),
    ("风电", IndustryType.MANUFACTURING),
    ("储能", IndustryType.MANUFACTURING),
    ("激光", IndustryType.MANUFACTURING),
    ("机床", IndustryType.MANUFACTURING),
    ("模具", IndustryType.MANUFACTURING),
    ("金属制品", IndustryType.MANUFACTURING),
    ("钢结构", IndustryType.MANUFACTURING),
    ("线缆", IndustryType.MANUFACTURING),
    ("机床", IndustryType.MANUFACTURING),
    ("工控", IndustryType.MANUFACTURING),
    ("叉车", IndustryType.MANUFACTURING),
    ("电梯", IndustryType.MANUFACTURING),
    ("物流", IndustryType.MANUFACTURING),
    ("航运", IndustryType.MANUFACTURING),
    ("港口", IndustryType.MANUFACTURING),
    ("铁路", IndustryType.MANUFACTURING),
    ("公路", IndustryType.MANUFACTURING),
    ("机场", IndustryType.MANUFACTURING),
    ("航空", IndustryType.MANUFACTURING),
    ("汽车整车", IndustryType.MANUFACTURING),
    ("商用", IndustryType.MANUFACTURING),
    ("乘用车", IndustryType.MANUFACTURING),
    ("摩托车", IndustryType.MANUFACTURING),
    ("交运设备", IndustryType.MANUFACTURING),
]


# ============================================================
# 行业专属效率评分阈值
# 结构: {
#     IndustryType: {
#         "receivables": (优秀天数, 良好天数, 偏长天数),  # 对应 5/4/2 分
#         "inventory": (优秀天数, 正常天数),               # 对应 5/3/1 分
#         "payables": (强议价天数, 一般天数),              # 对应 5/3/1 分
#     }
# }
# ============================================================

# 默认阈值（用于未知行业）
_DEFAULT_THRESHOLDS = {
    "receivables": (30, 60, 90),
    "inventory": (60, 120),
    "payables": (45, 30),
}

INDUSTRY_THRESHOLDS: Dict[IndustryType, Dict[str, Tuple[float, float, float]]] = {
    IndustryType.CONSUMER: {
        # 消费行业：应收短、存货快（白酒除外需特殊处理）、议价能力强
        "receivables": (20, 40, 60),
        "inventory": (60, 120),
        "payables": (60, 30),
    },
    IndustryType.PHARMA: {
        # 医药行业：应收中等、存货较长（效期管理）、议价能力一般
        "receivables": (30, 60, 90),
        "inventory": (90, 180),
        "payables": (45, 30),
    },
    IndustryType.TECH: {
        # TMT行业：应收中等偏短、存货快、上游强势应付短
        "receivables": (30, 60, 90),
        "inventory": (60, 120),
        "payables": (45, 30),
    },
    IndustryType.MANUFACTURING: {
        # 制造行业：应收长、存货长、应付中等
        "receivables": (60, 90, 120),
        "inventory": (90, 180),
        "payables": (45, 30),
    },
    IndustryType.RESOURCE: {
        # 资源行业：应收短、存货较快、应付中等
        "receivables": (20, 40, 60),
        "inventory": (45, 90),
        "payables": (45, 30),
    },
    IndustryType.FINANCE: {
        # 金融行业：不适用传统营运效率指标，特殊处理
        "receivables": (9999, 9999, 9999),  # 几乎不适用，给基础分
        "inventory": (9999, 9999),
        "payables": (9999, 9999),
    },
    IndustryType.REAL_ESTATE: {
        # 地产行业：应收不适用、存货特慢（楼盘）、应付长
        "receivables": (9999, 9999, 9999),  # 不适用
        "inventory": (9999, 9999),  # 不适用，存货是楼盘
        "payables": (90, 60),
    },
    IndustryType.UTILITY: {
        # 公用事业：应收较长（政府客户）、存货少、应付一般
        "receivables": (45, 90, 120),
        "inventory": (30, 60),
        "payables": (45, 30),
    },
    IndustryType.UNKNOWN: {
        "receivables": (30, 60, 90),
        "inventory": (60, 120),
        "payables": (45, 30),
    },
}


def resolve_industry_type(sectors: List[str]) -> IndustryType:
    """
    根据行业标签列表解析行业分类

    按优先级从高到低匹配关键词，返回第一个匹配的行业类型。

    Args:
        sectors: 数据库中的行业标签列表

    Returns:
        匹配的 IndustryType，无匹配返回 UNKNOWN
    """
    if not sectors:
        return IndustryType.UNKNOWN

    for keyword, industry_type in INDUSTRY_KEYWORD_MAP:
        for sector in sectors:
            if keyword in sector:
                return industry_type

    return IndustryType.UNKNOWN


def is_financial_or_real_estate(industry_type: IndustryType) -> bool:
    """判断是否为金融或地产行业（不适用传统营运效率指标）"""
    return industry_type in (IndustryType.FINANCE, IndustryType.REAL_ESTATE)


def is_liquor_industry(sectors: List[str]) -> bool:
    """判断是否为白酒行业（存货周转特殊处理）"""
    return any("白酒" in s for s in sectors)


def get_industry_thresholds(industry_type: IndustryType) -> Dict[str, Tuple[float, ...]]:
    """
    获取指定行业的效率评分阈值

    Args:
        industry_type: 行业分类

    Returns:
        阈值字典
    """
    return INDUSTRY_THRESHOLDS.get(industry_type, _DEFAULT_THRESHOLDS)


# ============================================================
# 行业合理 PE 区间配置（从 industry_percentiles.yaml 读取）
# ============================================================

# 默认 PE 区间（当 YAML 中 pe_range 缺失时使用）
_DEFAULT_PE_RANGES: Dict[IndustryType, Tuple[float, float]] = {
    IndustryType.CONSUMER: (15, 40),
    IndustryType.PHARMA: (20, 50),
    IndustryType.TECH: (20, 60),
    IndustryType.MANUFACTURING: (10, 30),
    IndustryType.RESOURCE: (8, 25),
    IndustryType.FINANCE: (5, 15),
    IndustryType.REAL_ESTATE: (5, 15),
    IndustryType.UTILITY: (10, 25),
    IndustryType.UNKNOWN: (10, 30),
}

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# YAML 配置文件路径（与 industry_percentile.py 共用）
_PERCENTILE_YAML_PATH = _PROJECT_ROOT / "config" / "industry_percentiles.yaml"

# 缓存加载结果，避免频繁读文件
_pe_ranges_cache: Optional[Dict[IndustryType, Tuple[float, float]]] = None


def load_pe_ranges() -> Dict[IndustryType, Tuple[float, float]]:
    """
    从 industry_percentiles.yaml 加载行业 PE 区间

    优先读取 YAML 中每个行业的 pe_range 字段，
    如果不存在或解析失败则回退到默认值。

    Returns:
        行业类型到 PE 区间 [下限, 上限] 的映射
    """
    global _pe_ranges_cache
    if _pe_ranges_cache is not None:
        return _pe_ranges_cache

    # 构建中文名 -> IndustryType 的映射
    name_to_type = {t.value: t for t in IndustryType}

    # 从 YAML 加载
    result: Dict[IndustryType, Tuple[float, float]] = {}
    if _PERCENTILE_YAML_PATH.exists():
        try:
            with open(_PERCENTILE_YAML_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            for name, industry_data in raw.items():
                industry_type = name_to_type.get(name)
                if industry_type is None or not isinstance(industry_data, dict):
                    continue
                pe_range = industry_data.get("pe_range")
                if pe_range is not None and len(pe_range) == 2:
                    low, high = float(pe_range[0]), float(pe_range[1])
                    if low > 0 and high > 0 and low < high:
                        result[industry_type] = (low, high)
                        continue
                logger.debug(f"行业 {name} 的 pe_range 不合法或不存在，使用默认值")
        except Exception as e:
            logger.warning(f"加载行业 PE 区间失败: {e}，使用默认值")

    # 未配置的行业使用默认值
    for t in IndustryType:
        if t not in result:
            result[t] = _DEFAULT_PE_RANGES.get(t, (10, 30))

    _pe_ranges_cache = result
    logger.debug(f"已从 JSON 加载 {len(result)} 个行业的 PE 区间配置")
    return result


def get_industry_pe_range(industry_type: IndustryType) -> Tuple[float, float]:
    """
    获取指定行业的合理 PE 区间

    Args:
        industry_type: 行业分类

    Returns:
        (下限PE, 上限PE) 的元组
    """
    ranges = load_pe_ranges()
    return ranges.get(industry_type, (10, 30))


def reload_pe_config() -> None:
    """
    重新加载行业 PE 配置

    清除缓存，强制重新从 YAML 读取配置。
    适用于用户在运行时修改了配置文件后需要立即生效的场景。
    """
    global _pe_ranges_cache
    _pe_ranges_cache = None
    logger.info("行业PE配置缓存已清除，下次获取时将重新加载")