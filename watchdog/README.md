# Watchdog Module - A股智能盯盘助手

## 概述

Watchdog 模块是一个独立的智能盯盘助手，允许用户选择股票、设定交易策略，并自动监控市场动态。当市场行情触发预设策略条件时，系统会及时发出提示，并基于既定策略提供明确的决策建议。

## 功能特性

- 📊 **自选股管理**：添加、删除、管理关注的股票
- 🎯 **策略配置**：支持多种预设策略和自定义策略
- 🔔 **实时监控**：自动监控市场行情变化
- 📢 **预警通知**：触发策略条件时发送通知
- 💡 **决策建议**：基于策略提供明确的操作建议
- 💾 **数据持久化**：自选股和预警记录自动保存

## 目录结构

```
watchdog/
├── __init__.py          # 模块入口，导出公共API
├── base.py              # 核心数据结构和基类
├── config.py            # 配置管理和数据持久化
├── monitor.py           # 实时行情监控引擎
├── notifier.py          # 预警通知集成
├── service.py           # 盯盘服务主类
├── strategies/          # 策略目录
│   └── __init__.py      # 内置策略实现
└── examples/            # 示例目录
    └── usage_example.py # 使用示例
```

## 快速开始

### 1. 基本使用

```python
from watchdog import WatchdogService

# 创建服务实例
service = WatchdogService()

# 添加股票到自选股（使用内置策略）
service.add_stock_to_watchlist(
    stock_code="600519",
    strategy_ids=["price_drop_5pct", "price_rise_5pct"],
    notes="贵州茅台"
)

# 执行一次检查
alerts = service.check_once()

# 查看最近的预警
recent_alerts = service.get_recent_alerts(max_count=50)
```

### 2. 启动持续监控

```python
from watchdog import WatchdogService

service = WatchdogService()

# 启动监控（后台线程）
service.start()

# ... 做其他事情 ...

# 停止监控
service.stop()
```

### 3. 自定义策略

```python
from watchdog import (
    ActionType,
    AlertLevel,
    ConditionType,
    StrategyType,
    WatchdogCondition,
    WatchdogStrategy,
    WatchdogService,
)

# 创建自定义策略
custom_strategy = WatchdogStrategy(
    id="my_custom_strategy",
    name="我的自定义策略",
    description="当价格达到目标时触发",
    strategy_type=StrategyType.ANY_CONDITION,
    conditions=[
        WatchdogCondition(
            condition_type=ConditionType.PRICE_ABOVE,
            parameters={"threshold": 1800.0},
            description="价格超过1800元",
        )
    ],
    alert_level=AlertLevel.WARNING,
    default_action=ActionType.SELL,
    default_reason="达到预设价格目标",
    target_price=1850.0,
    stop_loss=1750.0,
)

# 注册策略
service = WatchdogService()
service.register_strategy(custom_strategy)

# 使用自定义策略
service.add_stock_to_watchlist(
    stock_code="600519",
    strategy_ids=["my_custom_strategy"]
)
```

## 内置策略

| 策略ID | 策略名称 | 描述 | 预警级别 |
|--------|----------|------|----------|
| price_drop_5pct | 跌幅超5%预警 | 当股票跌幅超过5%时触发 | WARNING |
| price_rise_5pct | 涨幅超5%预警 | 当股票涨幅超过5%时触发 | INFO |
| volume_surge_2x | 量能放大2倍 | 当量比超过2倍时触发 | INFO |
| sharp_drop_8pct | 急跌超8%预警 | 当股票跌幅超过8%时触发 | CRITICAL |

## 配置

可以通过环境变量或 `.env` 文件配置模块：

```env
# 自选股和预警数据保存路径
WATCHDOG_WATCHLIST_FILE=data/watchdog_watchlist.json
WATCHDOG_ALERTS_FILE=data/watchdog_alerts.json

# 监控间隔（秒）
WATCHDOG_CHECK_INTERVAL=60

# 预警冷却时间（分钟）
WATCHDOG_ALERT_COOLDOWN=30

# 交易时间设置
WATCHDOG_MARKET_OPEN_START=09:30
WATCHDOG_MARKET_OPEN_END=15:00
WATCHDOG_MONITOR_MARKET_HOURS_ONLY=true

# 通知设置
WATCHDOG_ENABLE_NOTIFICATIONS=true
```

## 数据持久化

- 自选股数据保存在 `data/watchdog_watchlist.json`
- 预警记录保存在 `data/watchdog_alerts.json`
- 数据会自动加载和保存

## 示例代码

完整的使用示例请参考 `watchdog/examples/usage_example.py`：

```bash
python watchdog/examples/usage_example.py
```

## API 参考

### WatchdogService

主要的服务类，提供以下方法：

- `add_stock_to_watchlist(stock_code, strategy_ids, notes)`: 添加股票到自选股
- `remove_stock_from_watchlist(stock_code)`: 从自选股移除股票
- `check_once()`: 执行一次检查，返回触发的预警
- `start()`: 启动后台监控
- `stop()`: 停止后台监控
- `is_running()`: 检查监控是否运行中
- `get_recent_alerts(max_count)`: 获取最近的预警记录
- `register_strategy(strategy)`: 注册自定义策略
- `unregister_strategy(strategy_id)`: 注销策略

## 注意事项

1. 本模块仅作为辅助工具，不构成投资建议
2. 请根据实际情况合理设置策略和预警条件
3. 确保网络连接正常以获取实时行情数据
4. 建议在交易时间外进行测试
