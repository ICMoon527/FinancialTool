# -*- coding: utf-8 -*-
"""
===================================
API v1 路由聚合
===================================

职责：
1. 聚合 v1 版本的所有 endpoint 路由
2. 统一添加 /api/v1 前缀
"""

from fastapi import APIRouter

from api.v1.endpoints import health, analysis, auth, history, stocks, backtest, system_config, agent, stock_selector, stock_search, visualization, intraday

# 创建 v1 版本主路由
router = APIRouter(prefix="/api/v1")

router.include_router(health.router)

router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Auth"]
)

router.include_router(
    agent.router,
    prefix="/agent",
    tags=["Agent"]
)

router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["Analysis"]
)

router.include_router(
    history.router,
    prefix="/history",
    tags=["History"]
)

# 搜索路由必须在 stocks 路由之前注册，确保 /stocks/search 不会被 /stocks/{stock_code} 误匹配
router.include_router(
    stock_search.router,
    prefix="/stocks",
    tags=["Stocks Search"]
)

router.include_router(
    stocks.router,
    prefix="/stocks",
    tags=["Stocks"]
)

router.include_router(
    backtest.router,
    prefix="/backtest",
    tags=["Backtest"]
)

router.include_router(
    system_config.router,
    prefix="/system",
    tags=["SystemConfig"]
)

router.include_router(
    stock_selector.router,
    prefix="/stock-selector",
    tags=["Stock Selector"]
)

router.include_router(
    visualization.router,
    prefix="/visualization",
    tags=["Visualization"]
)

router.include_router(
    intraday.router,
    prefix="/intraday",
    tags=["Intraday"]
)
