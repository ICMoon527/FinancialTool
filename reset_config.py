#!/usr/bin/env python3
"""重置策略配置文件到新格式"""

from strategy.strategy_config import strategy_config

print("重置策略配置文件...")
strategy_config.reset_to_default()
print("✓ 配置文件已重置为新格式！")
