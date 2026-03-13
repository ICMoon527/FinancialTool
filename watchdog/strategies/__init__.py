# -*- coding: utf-8 -*-
"""
Watchdog Strategies Module

Built-in watch strategies.
"""

from typing import List

from watchdog.base import (
    ActionType,
    AlertLevel,
    ConditionType,
    StrategyType,
    WatchdogCondition,
    WatchdogStrategy,
)


def get_builtin_strategies() -> List[WatchdogStrategy]:
    """
    Get all built-in watch strategies.

    Returns:
        List of WatchdogStrategy
    """
    strategies = []

    price_drop_strategy = WatchdogStrategy(
        id="price_drop_3pct",
        name="跌幅超3%提醒",
        description="当股票跌幅超过3%时触发提醒，建议加仓做T",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_BELOW,
                parameters={"threshold": -3.0},
                description="跌幅超过3%",
            )
        ],
        alert_level=AlertLevel.WARNING,
        default_action=ActionType.BUY,
        default_reason="股票跌幅超过3%，建议加仓做T",
    )
    strategies.append(price_drop_strategy)

    price_rise_strategy = WatchdogStrategy(
        id="price_rise_3pct",
        name="涨幅超3%提醒",
        description="当股票涨幅超过3%时触发提醒，建议减仓做T",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_ABOVE,
                parameters={"threshold": 3.0},
                description="涨幅超过3%",
            )
        ],
        alert_level=AlertLevel.INFO,
        default_action=ActionType.SELL,
        default_reason="股票涨幅超过3%，建议减仓做T",
    )
    strategies.append(price_rise_strategy)

    volume_surge_strategy = WatchdogStrategy(
        id="volume_surge_2x",
        name="量能放大2倍",
        description="当量比超过2倍时触发预警",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.VOLUME_RATIO_ABOVE,
                parameters={"threshold": 2.0},
                description="量比超过2倍",
            )
        ],
        alert_level=AlertLevel.INFO,
        default_action=ActionType.WATCH,
        default_reason="量能明显放大，关注资金动向",
    )
    strategies.append(volume_surge_strategy)

    sharp_drop_strategy = WatchdogStrategy(
        id="sharp_drop_8pct",
        name="急跌超8%预警",
        description="当股票跌幅超过8%时触发严重预警",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_BELOW,
                parameters={"threshold": -8.0},
                description="跌幅超过8%",
            )
        ],
        alert_level=AlertLevel.CRITICAL,
        default_action=ActionType.SELL,
        default_reason="股票跌幅较大，建议评估风险",
    )
    strategies.append(sharp_drop_strategy)

    # 支撑位买入策略
    support_buy_strategy = WatchdogStrategy(
        id="support_buy",
        name="支撑位买入",
        description="当股票接近支撑位时触发提醒，建议买入",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.PRICE_BELOW,
                parameters={"threshold": 0},  # 实际支撑位需要根据个股设置
                description="接近支撑位",
            )
        ],
        alert_level=AlertLevel.INFO,
        default_action=ActionType.BUY,
        default_reason="股票接近支撑位，建议买入",
    )
    strategies.append(support_buy_strategy)

    # 压力位卖出策略
    resistance_sell_strategy = WatchdogStrategy(
        id="resistance_sell",
        name="压力位卖出",
        description="当股票接近压力位时触发提醒，建议卖出",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.PRICE_ABOVE,
                parameters={"threshold": 0},  # 实际压力位需要根据个股设置
                description="接近压力位",
            )
        ],
        alert_level=AlertLevel.INFO,
        default_action=ActionType.SELL,
        default_reason="股票接近压力位，建议卖出",
    )
    strategies.append(resistance_sell_strategy)

    # 急拉（放量上涨）策略
    sharp_rise_strategy = WatchdogStrategy(
        id="sharp_rise_with_volume",
        name="放量急拉提醒",
        description="当股票快速上涨且放量时触发提醒",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_ABOVE,
                parameters={"threshold": 4.0},
                description="快速上涨超过4%",
            ),
            WatchdogCondition(
                condition_type=ConditionType.VOLUME_RATIO_ABOVE,
                parameters={"threshold": 2.5},
                description="量比超过2.5倍",
            )
        ],
        alert_level=AlertLevel.INFO,
        default_action=ActionType.SELL,
        default_reason="股票放量急拉，建议减仓或止盈",
    )
    strategies.append(sharp_rise_strategy)

    # 急跌（放量下跌）策略
    sharp_fall_strategy = WatchdogStrategy(
        id="sharp_fall_with_volume",
        name="放量急跌提醒",
        description="当股票快速下跌且放量时触发提醒",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_BELOW,
                parameters={"threshold": -4.0},
                description="快速下跌超过4%",
            ),
            WatchdogCondition(
                condition_type=ConditionType.VOLUME_RATIO_ABOVE,
                parameters={"threshold": 2.5},
                description="量比超过2.5倍",
            )
        ],
        alert_level=AlertLevel.CRITICAL,
        default_action=ActionType.BUY,
        default_reason="股票放量急跌，建议关注或分批建仓",
    )
    strategies.append(sharp_fall_strategy)

    # 缩量下跌策略
    low_volume_fall_strategy = WatchdogStrategy(
        id="low_volume_fall",
        name="缩量下跌提醒",
        description="当股票下跌且缩量时触发提醒",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_BELOW,
                parameters={"threshold": -2.0},
                description="下跌超过2%",
            ),
            WatchdogCondition(
                condition_type=ConditionType.VOLUME_RATIO_ABOVE,
                parameters={"threshold": 0.5},
                description="量比低于0.5倍",
            )
        ],
        alert_level=AlertLevel.WARNING,
        default_action=ActionType.WATCH,
        default_reason="股票缩量下跌，建议观望",
    )
    strategies.append(low_volume_fall_strategy)

    # 缩量上涨策略
    low_volume_rise_strategy = WatchdogStrategy(
        id="low_volume_rise",
        name="缩量上涨提醒",
        description="当股票上涨且缩量时触发提醒",
        strategy_type=StrategyType.ANY_CONDITION,
        conditions=[
            WatchdogCondition(
                condition_type=ConditionType.CHANGE_ABOVE,
                parameters={"threshold": 2.0},
                description="上涨超过2%",
            ),
            WatchdogCondition(
                condition_type=ConditionType.VOLUME_RATIO_ABOVE,
                parameters={"threshold": 0.5},
                description="量比低于0.5倍",
            )
        ],
        alert_level=AlertLevel.WARNING,
        default_action=ActionType.WATCH,
        default_reason="股票缩量上涨，建议谨慎",
    )
    strategies.append(low_volume_rise_strategy)

    return strategies
