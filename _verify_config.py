# -*- coding: utf-8 -*-
"""验证 stock_selector 配置项前后端一致性

检查项：
1. Schema default_value 与 StockSelectorConfig dataclass 默认值是否一致
2. get_config API 回退逻辑是否正确（key 不在 .env 时回退到 schema default）
"""
from src.core.config_registry import build_schema_response
from src.core.config_manager import ConfigManager
from src.services.system_config_service import SystemConfigService
from pathlib import Path

# 读取 .env
env_path = Path(__file__).parent / ".env"
mgr = ConfigManager(env_path=env_path)
env_map = mgr.read_config_map()

# 获取 schema
schema = build_schema_response()
ss = [c for c in schema["categories"] if c["category"] == "stock_selector"][0]

# Dataclass 默认值映射（来自 stock_selector/config.py StockSelectorConfig）
DATACLASS_DEFAULTS = {
    "UPDATE_SECTOR_DATA": False,
    "STOCK_SELECTOR_DEFAULT_TOP_N": 10,
    "STOCK_SELECTOR_MIN_MATCH_SCORE": 50.0,
    "STOCK_SELECTOR_AUTO_ACTIVATE_ALL": False,
    "STOCK_SELECTOR_DEFAULT_ACTIVE_STRATEGIES": None,  # field(default_factory=list)
    "STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE": None,
    "STOCK_SELECTOR_PRICE_WEIGHT": 20.0,
    "STOCK_SELECTOR_VOLUME_WEIGHT": 20.0,
    "STOCK_SELECTOR_TECHNICAL_WEIGHT": 25.0,
    "STOCK_SELECTOR_MARKET_WEIGHT": 15.0,
    "STOCK_SELECTOR_TREND_WEIGHT": 20.0,
    "STOCK_SELECTOR_SHORT_TERM_GAIN_MIN": 1.0,
    "STOCK_SELECTOR_SHORT_TERM_GAIN_MAX": 5.0,
    "STOCK_SELECTOR_SHORT_TERM_VOLUME_RATIO": 1.5,
    "STOCK_SELECTOR_SHORT_TERM_VOLUME_AVG": 1.5,
    "STOCK_SELECTOR_MA_GOLDEN_CROSS_FAST": 5,
    "STOCK_SELECTOR_MA_GOLDEN_CROSS_SLOW": 20,
    "STOCK_SELECTOR_VOLUME_BREAKOUT_LOOKBACK": 20,
    "STOCK_SELECTOR_VOLUME_BREAKOUT_MULTIPLIER": 2.0,
    "STOCK_SELECTOR_LOG_LEVEL": "DEBUG",
    "STOCK_SELECTOR_DEBUG_EXECUTION": False,
    "STOCK_SELECTOR_ENABLE_SECTOR_ANALYSIS": True,
    "STOCK_SELECTOR_SECTOR_THRESHOLD": 2.0,
    "STOCK_SELECTOR_SECTOR_CACHE_TTL": 1800,
    "STOCK_SELECTOR_ENABLE_MULTITHREADING": True,
    "STOCK_SELECTOR_MULTITHREADING_WORKERS": 15,
    "STOCK_SELECTOR_EFINANCE_BATCH_SIZE": 10,
    "STOCK_SELECTOR_TUSHARE_BATCH_SIZE": 20,
    "STOCK_SELECTOR_UPDATE_DATA_DEFAULT_DAYS": 150,
    "SIX_DIMENSION_MAIN_TRADING_WEIGHT": 1.0,
    "SIX_DIMENSION_BANK_CONTROL_WEIGHT": 1.0,
    "SIX_DIMENSION_MOMENTUM_V2_WEIGHT": 1.0,
    "SIX_DIMENSION_RESONANCE_WEIGHT": 1.0,
    "SIX_DIMENSION_STRONG_BLAST_WEIGHT": 1.0,
    "SIX_DIMENSION_SECTOR_WEIGHT": 1.0,
    "SIX_DIMENSION_MIN_MATCHED_DIMENSIONS": 4,
}


def dc_default_to_str(val):
    """将 dataclass 默认值转为与 schema default_value 可比较的字符串。"""
    if val is None:
        return None
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


print("=" * 100)
print("1. Schema default_value vs Dataclass 默认值 一致性检查")
print("=" * 100)
print(f"{'Key':<45} {'schema_default':<16} {'dataclass_default':<18} {'match':<6}")
print("-" * 90)

dc_mismatches = 0
for field in ss["fields"]:
    key = field["key"]
    schema_dv = field.get("default_value")
    dc_val = DATACLASS_DEFAULTS.get(key)
    dc_str = dc_default_to_str(dc_val)

    schema_str = "" if schema_dv is None else str(schema_dv)
    dc_display = "" if dc_str is None else dc_str

    match = schema_str == dc_display
    status = "✓" if match else "✗"
    if not match:
        dc_mismatches += 1
    print(f"{key:<45} {schema_str:<16} {dc_display:<18} {status:<6}")

print(f"\nSchema vs Dataclass: 总计 {len(ss['fields'])} 项, 不一致 {dc_mismatches} 项")

print()
print("=" * 100)
print("2. get_config API 返回值检查（验证回退逻辑）")
print("=" * 100)
print(f"{'Key':<45} {'.env 有值?':<10} {'API 返回值':<20} {'来源':<15}")
print("-" * 95)

service = SystemConfigService(manager=mgr)
api_result = service.get_config(include_schema=True)
api_items = {item["key"]: item for item in api_result["items"]}

for field in ss["fields"]:
    key = field["key"]
    schema_dv = field.get("default_value")
    env_val = env_map.get(key, "")
    has_env = "是" if env_val else "否"

    api_item = api_items.get(key, {})
    api_value = api_item.get("value", "")
    api_exists = api_item.get("raw_value_exists", False)

    if env_val:
        source = ".env 文件"
    elif api_value and schema_dv is not None:
        source = "schema 默认值"
    else:
        source = "无默认值"

    print(f"{key:<45} {has_env:<10} {str(api_value):<20} {source:<15}")

print()
print("=" * 100)
print("3. 总结")
print("=" * 100)
if dc_mismatches == 0:
    print("✓ Schema default_value 与 Dataclass 默认值完全一致，前后端配置已同步。")
else:
    print(f"✗ 存在 {dc_mismatches} 项不一致，需要修复。")

# 检查 .env 中实际值与 dataclass 默认值的差异（用户主动修改的配置）
print()
print("--- .env 中用户自定义值（与 dataclass 默认值不同）---")
for field in ss["fields"]:
    key = field["key"]
    env_val = env_map.get(key, "")
    if not env_val:
        continue
    dc_val = DATACLASS_DEFAULTS.get(key)
    dc_str = dc_default_to_str(dc_val)
    if dc_str is not None and env_val != dc_str:
        print(f"  {key}: .env={env_val}, 默认={dc_str}")