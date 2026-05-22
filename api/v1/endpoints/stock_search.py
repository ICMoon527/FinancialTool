# -*- coding: utf-8 -*-
"""
===================================
股票搜索 API 端点
===================================

职责：
1. GET /api/v1/stocks/search - 多维度股票搜索
2. POST /api/v1/stocks/search/refresh - 刷新搜索索引
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Query

from api.v1.schemas.stocks import (
    MatchSegment,
    StockSearchResponse,
    StockSearchResult,
    StockSearchRefreshResponse,
)
from src.stock_search_index import get_search_index, refresh_search_index

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# 后端 LRU 缓存
# ---------------------------------------------------------------------------

_search_cache: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL: float = 600.0  # 10 分钟
_CACHE_MAX: int = 500


def _cache_key(query: str, limit: int) -> str:
    """生成缓存键"""
    return f"{query.lower()}|{limit}"


def _cache_get(key: str) -> Optional[list]:
    """从缓存获取结果"""
    entry = _search_cache.get(key)
    if entry is None:
        return None
    ts, results = entry
    if time.time() - ts > _CACHE_TTL:
        del _search_cache[key]
        return None
    return results


def _cache_set(key: str, results: list) -> None:
    """写入缓存"""
    if len(_search_cache) >= _CACHE_MAX:
        # 淘汰最旧的条目
        oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
        del _search_cache[oldest]
    _search_cache[key] = (time.time(), results)


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=StockSearchResponse,
    summary="股票多维度搜索",
    description=(
        "支持以下搜索方式:\n"
        "- 股票代码精确/前缀搜索 (如 600519, 6005)\n"
        "- 中文名称模糊搜索 (如 贵州, 茅台)\n"
        "- 拼音全拼搜索 (如 guizhoumaotai, guizhou)\n"
        "- 拼音简拼搜索 (如 gzmt, gz)\n"
        "- 拼音首字母搜索 (如 gzm)"
    ),
)
def search_stocks(
    q: str = Query(..., description="搜索关键词", min_length=0),
    limit: int = Query(20, ge=1, le=100, description="最大返回结果数"),
) -> StockSearchResponse:
    """股票多维度搜索"""
    query = q.strip()
    if not query:
        return StockSearchResponse(query=q, total=0, results=[], time_ms=0.0)

    limit = max(1, min(limit, 100))

    # 检查缓存
    ck = _cache_key(query, limit)
    cached = _cache_get(ck)
    if cached is not None:
        return StockSearchResponse(query=q, total=len(cached), results=cached, time_ms=0.0)

    start = time.perf_counter()
    try:
        index = get_search_index()
        raw_results = index.search(query, limit)
    except Exception as e:
        logger.error("搜索异常: %s", e, exc_info=True)
        return StockSearchResponse(query=q, total=0, results=[], time_ms=0.0)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # 转换为 Pydantic 模型
    results: List[StockSearchResult] = []
    for item in raw_results:
        segments = [
            MatchSegment(field=seg["field"], start=seg["start"], end=seg["end"])
            for seg in item.get("match_segments", [])
        ]
        results.append(
            StockSearchResult(
                code=item["code"],
                name=item["name"],
                market=item.get("market", ""),
                match_type=item.get("match_type", ""),
                match_segments=segments,
                score=item.get("score", 0),
            )
        )

    _cache_set(ck, results)
    return StockSearchResponse(query=q, total=len(results), results=results, time_ms=elapsed_ms)


@router.post(
    "/search/refresh",
    response_model=StockSearchRefreshResponse,
    summary="刷新搜索索引",
    description="手动重建搜索索引。当股票池更新后调用此端点使索引与数据库同步。",
)
def refresh_index() -> StockSearchRefreshResponse:
    """刷新搜索索引"""
    try:
        count = refresh_search_index()
        # 清空缓存
        _search_cache.clear()
        return StockSearchRefreshResponse(status="ok", entry_count=count)
    except Exception as e:
        logger.error("索引刷新失败: %s", e, exc_info=True)
        return StockSearchRefreshResponse(status="error", entry_count=0)
