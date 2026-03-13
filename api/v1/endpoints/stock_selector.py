# -*- coding: utf-8 -*-
"""
Stock Selector API Endpoints.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from api.v1.schemas.stock_selector import (
    StrategyInfo,
    StrategyMatchInfo,
    StockCandidateInfo,
    StockSelectorRequest,
    StockSelectorResponse,
    StrategiesResponse,
    ActivateStrategyRequest,
    DeactivateStrategyRequest,
)
from stock_selector import StockSelectorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stock-selector", tags=["Stock Selector"])

# Global service instance
_stock_selector_service: Optional[StockSelectorService] = None


def get_stock_selector_service() -> StockSelectorService:
    """
    Get or create the stock selector service.

    Returns:
        StockSelectorService instance
    """
    global _stock_selector_service
    if _stock_selector_service is None:
        _stock_selector_service = StockSelectorService()
        logger.info("Stock Selector Service initialized (using config)")
    return _stock_selector_service


def _convert_strategy_metadata_to_info(metadata) -> StrategyInfo:
    """Convert StrategyMetadata to StrategyInfo schema."""
    return StrategyInfo(
        id=metadata.id,
        name=metadata.name,
        display_name=metadata.display_name,
        description=metadata.description,
        strategy_type=metadata.strategy_type.name,
        category=metadata.category,
        source=metadata.source,
        version=metadata.version,
        created_at=metadata.created_at,
        is_active=metadata.enabled,
    )


def _convert_stock_candidate_to_info(candidate, active_strategy_ids: list[str]) -> StockCandidateInfo:
    """Convert StockCandidate to StockCandidateInfo schema."""
    match_infos = [
        StrategyMatchInfo(
            strategy_id=m.strategy_id,
            strategy_name=m.strategy_name,
            matched=m.matched,
            score=m.score,
            reason=m.reason,
            match_details=m.match_details,
        )
        for m in candidate.strategy_matches
        if m.strategy_id in active_strategy_ids
    ]
    return StockCandidateInfo(
        stock_code=candidate.code,
        stock_name=candidate.name,
        current_price=candidate.current_price,
        overall_score=candidate.match_score,
        strategy_matches=match_infos,
        created_at=candidate.created_at,
        extra_data=candidate.extra_data,
    )


@router.get("/strategies", response_model=StrategiesResponse)
async def get_strategies(
    service: StockSelectorService = Depends(get_stock_selector_service),
):
    """
    Get all available stock selection strategies.
    """
    try:
        strategies_meta = service.get_available_strategies()
        active_ids = service.get_active_strategy_ids()
        strategy_infos = [_convert_strategy_metadata_to_info(meta) for meta in strategies_meta]
        for info in strategy_infos:
            info.is_active = info.id in active_ids
        return StrategiesResponse(
            success=True,
            strategies=strategy_infos,
            active_strategy_ids=active_ids,
        )
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/activate")
async def activate_strategy(
    request: ActivateStrategyRequest,
    service: StockSelectorService = Depends(get_stock_selector_service),
):
    """
    Activate a specific strategy.
    """
    try:
        service.activate_strategies([request.strategy_id])
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to activate strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/deactivate")
async def deactivate_strategy(
    request: DeactivateStrategyRequest,
    service: StockSelectorService = Depends(get_stock_selector_service),
):
    """
    Deactivate a specific strategy.
    """
    try:
        service.deactivate_strategies([request.strategy_id])
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to deactivate strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screen", response_model=StockSelectorResponse)
async def screen_stocks(
    request: StockSelectorRequest,
    service: StockSelectorService = Depends(get_stock_selector_service),
):
    """
    Screen stocks using the specified strategies.
    """
    start_time = time.time()
    try:
        candidates = service.screen_stocks(
            stock_codes=request.stock_codes,
            strategy_ids=request.strategy_ids,
            top_n=request.top_n,
        )

        active_ids = service.get_active_strategy_ids()
        candidate_infos = [_convert_stock_candidate_to_info(c, active_ids) for c in candidates]

        execution_time_ms = (time.time() - start_time) * 1000
        return StockSelectorResponse(
            success=True,
            candidates=candidate_infos,
            total_screened=len(request.stock_codes) if request.stock_codes else 0,
            execution_time_ms=execution_time_ms,
        )
    except Exception as e:
        logger.error(f"Failed to screen stocks: {e}")
        execution_time_ms = (time.time() - start_time) * 1000
        return StockSelectorResponse(
            success=False,
            candidates=[],
            total_screened=len(request.stock_codes) if request.stock_codes else 0,
            execution_time_ms=execution_time_ms,
            error=str(e),
        )
