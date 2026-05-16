# -*- coding: utf-8 -*-
"""
Stock Selector API Endpoints.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Depends, Query
from api.utils.json_encoder import jsonable_encoder_with_numpy

from api.v1.schemas.stock_selector import (
    ScreenProgressStatus,
    StrategyInfo,
    StrategyMatchInfo,
    StockCandidateInfo,
    StockSelectorRequest,
    StockSelectorResponse,
    StrategiesResponse,
    ActivateStrategyRequest,
    DeactivateStrategyRequest,
    StockSelectorConfigResponse,
)
from stock_selector import StockSelectorService
from stock_selector.config import get_config
from stock_selector.market_data_cache import MarketDataCache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Stock Selector"])


def _update_market_data():
    """
    更新大盘数据缓存（支持智能时间窗策略）
    """
    try:
        logger.info("开始更新大盘数据缓存...")
        
        from data_provider import DataFetcherManager
        from stock_selector.market_data_cache import MarketDataCache
        
        data_fetcher = DataFetcherManager()
        
        # 更新上证指数 (sh000001) - 使用智能时间窗方法（会自动保存缓存）
        sh_data = MarketDataCache.get_complete_index_data("sh000001", data_provider=data_fetcher)
        if sh_data is not None and not sh_data.empty:
            logger.info(f"上证指数数据获取成功，共 {len(sh_data)} 条")
        
        # 更新深证成指 (sz399001) - 使用智能时间窗方法（会自动保存缓存）
        sz_data = MarketDataCache.get_complete_index_data("sz399001", data_provider=data_fetcher)
        if sz_data is not None and not sz_data.empty:
            logger.info(f"深证成指数据获取成功，共 {len(sz_data)} 条")
        
        logger.info("大盘数据缓存更新完成！")
    except Exception as e:
        logger.warning(f"更新大盘数据缓存失败: {e}")

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
        # 设置数据提供者，与命令行调用保持一致
        from data_provider import DataFetcherManager
        data_fetcher_manager = DataFetcherManager()
        _stock_selector_service.set_data_provider(data_fetcher_manager)
        logger.info("Stock Selector Service initialized (using config with data provider)")
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


def _convert_stock_candidate_to_info(candidate, active_strategy_ids: list[str], sector_manager=None) -> StockCandidateInfo:
    """Convert StockCandidate to StockCandidateInfo schema."""
    match_infos = [
        StrategyMatchInfo(
            strategy_id=m.strategy_id,
            strategy_name=m.strategy_name,
            matched=bool(m.matched),  # 确保转换为普通 bool
            score=float(m.score),    # 确保转换为普通 float
            reason=m.reason,
            match_details=jsonable_encoder_with_numpy(m.match_details),  # 处理 match_details 中的 numpy 类型
        )
        for m in candidate.strategy_matches
    ]
    
    sectors = []
    if sector_manager:
        try:
            sectors = sector_manager.get_stock_sectors(candidate.code)
        except Exception as e:
            logger.warning(f"Failed to get sector info for {candidate.code}: {e}")
    
    return StockCandidateInfo(
        stock_code=candidate.code,
        stock_name=candidate.name,
        current_price=float(candidate.current_price),  # 确保转换为普通 float
        overall_score=float(candidate.match_score),    # 确保转换为普通 float
        strategy_matches=match_infos,
        created_at=candidate.created_at,
        sectors=sectors,
        extra_data=jsonable_encoder_with_numpy(candidate.extra_data),  # 处理 extra_data 中的 numpy 类型
    )


@router.get("/strategies", response_model=StrategiesResponse)
async def get_strategies(
    service: StockSelectorService = Depends(get_stock_selector_service),
):
    """
    Get all available stock selection strategies.
    """
    logger.info("Getting strategies...")
    try:
        strategies_meta = service.get_available_strategies()
        logger.info(f"Found {len(strategies_meta)} strategies")
        active_ids = service.get_active_strategy_ids()
        strategy_infos = [_convert_strategy_metadata_to_info(meta) for meta in strategies_meta]
        for info in strategy_infos:
            info.is_active = info.id in active_ids
        response = StrategiesResponse(
            success=True,
            strategies=strategy_infos,
            active_strategy_ids=active_ids,
        )
        logger.info(f"Returning {len(strategy_infos)} strategies")
        return response
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}", exc_info=True)
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


@router.get("/config", response_model=StockSelectorConfigResponse)
async def get_stock_selector_config():
    """
    获取选股器配置信息
    """
    logger.info("Getting stock selector config...")
    try:
        config = get_config()
        logger.info(f"Config loaded: default_top_n={config.default_top_n}")
        response = StockSelectorConfigResponse(
            success=True,
            default_top_n=config.default_top_n,
        )
        logger.info(f"Returning response: {response.model_dump()}")
        return response
    except Exception as e:
        logger.error(f"Failed to get stock selector config: {e}", exc_info=True)
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
        # 设置市场数据缓存的强制更新模式
        # 只有在 update_realtime 或 update_data 时才更新大盘数据缓存
        MarketDataCache.set_force_update(request.update_realtime or request.update_data)
        
        # 检查 update_realtime 和 update_data 优先级
        # 如果同时指定，update_realtime 优先
        use_update_realtime = request.update_realtime
        use_update_data = request.update_data
        
        if use_update_realtime and use_update_data:
            logger.info("同时指定了 update_realtime 和 update_data，将优先使用 update_realtime")
            use_update_data = False
        
        # 如果需要先更新实时数据
        if use_update_realtime:
            await _update_realtime_stock_data(request.stock_codes, service)
        elif use_update_data:
            await _update_stock_data(request.stock_codes, service)
        
        # 更新完数据后，关闭强制更新模式，让 screen 可以正常使用缓存
        MarketDataCache.set_force_update(False)

        candidates = service.screen_stocks(
            stock_codes=request.stock_codes,
            strategy_ids=request.strategy_ids,
            top_n=request.top_n,
        )

        # 获取 sector_manager
        sector_manager = None
        if service.strategy_manager:
            sector_manager = service.strategy_manager.get_sector_manager()
        
        active_ids = service.get_active_strategy_ids()
        candidate_infos = [_convert_stock_candidate_to_info(c, active_ids, sector_manager) for c in candidates]

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


async def _update_realtime_stock_data(stock_codes: Optional[list[str]], service: StockSelectorService) -> None:
    """
    更新实时股票数据

    Args:
        stock_codes: 股票代码列表，如果为 None 则更新所有股票
        service: StockSelectorService 实例
    """
    from stock_selector.realtime_data_updater import get_realtime_updater
    from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
    
    logger.info(f"开始更新实时股票数据...")
    
    # 处理股票代码列表
    if stock_codes is None:
        stock_code_name_pairs = get_all_stock_code_name_pairs(force_refresh=True)
        # 过滤ST股票
        stock_code_name_pairs = filter_st_stocks(stock_code_name_pairs)
        # 过滤特定板块的股票代码（科创板、创业板、北交所等）
        stock_codes = [code for code, name in stock_code_name_pairs]
        stock_codes = filter_special_stock_codes(stock_codes)
    else:
        # 如果用户指定了股票代码，先获取它们的名称，然后过滤ST股票
        try:
            all_pairs = get_all_stock_code_name_pairs(force_refresh=True)
            code_to_name = {code: name for code, name in all_pairs}
            # 过滤ST股票
            filtered_codes = []
            for code in stock_codes:
                name = code_to_name.get(code, "")
                if not any(keyword in name.upper() for keyword in ['ST', '*ST', 'SST', 'S*ST']):
                    filtered_codes.append(code)
            stock_codes = filtered_codes
        except Exception:
            pass
    
    # 使用实时数据更新器
    realtime_updater = get_realtime_updater()
    stats = realtime_updater.update_realtime_data(stock_codes=stock_codes)
    
    # 更新大盘数据缓存
    _update_market_data()
    
    # 更新板块数据
    _update_sector_data(service)


async def _update_stock_data(stock_codes: Optional[list[str]], service: StockSelectorService) -> None:
    """
    更新股票数据

    Args:
        stock_codes: 股票代码列表，如果为 None 则更新所有股票
        service: StockSelectorService 实例
    """
    from datetime import date, timedelta
    from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
    
    config = get_config()
    logger.info(f"开始更新股票数据 (最后 {config.update_data_default_days} 天)...")
    
    # 处理股票代码列表
    if stock_codes is None:
        stock_code_name_pairs = get_all_stock_code_name_pairs(force_refresh=True)
        # 过滤ST股票
        stock_code_name_pairs = filter_st_stocks(stock_code_name_pairs)
        # 过滤特定板块的股票代码（科创板、创业板、北交所等）
        stock_codes = [code for code, name in stock_code_name_pairs]
        stock_codes = filter_special_stock_codes(stock_codes)
    else:
        # 如果用户指定了股票代码，先获取它们的名称，然后过滤ST股票
        try:
            all_pairs = get_all_stock_code_name_pairs(force_refresh=True)
            code_to_name = {code: name for code, name in all_pairs}
            # 过滤ST股票
            filtered_codes = []
            for code in stock_codes:
                name = code_to_name.get(code, "")
                if not any(keyword in name.upper() for keyword in ['ST', '*ST', 'SST', 'S*ST']):
                    filtered_codes.append(code)
            stock_codes = filtered_codes
        except Exception:
            pass
    
    # 使用 Tushare 专用下载器强制更新数据
    try:
        from stock_selector.tushare_data_downloader import get_tushare_downloader
        
        logger.info(f"使用 Tushare 数据下载器 (速率限制: 50 次/分钟)")
        
        downloader = get_tushare_downloader(rate_limit_per_minute=50)
        stats = downloader.download_data(
            stock_codes=stock_codes,
            days=config.update_data_default_days
        )
        
        logger.info(f"Tushare 数据更新完成!")
        
    except Exception as e:
        logger.warning(f"Tushare 下载器失败: {e}, 回退到旧版更新器")
        # 如果 Tushare 不可用，回退到旧版更新器
        try:
            from stock_selector.batch_data_updater import get_batch_updater
            
            end_date = date.today()
            target_start_date = end_date - timedelta(days=config.update_data_default_days - 1)
            actual_start_date = target_start_date
            logger.info(f"强制更新全部 {config.update_data_default_days} 天数据")
            logger.info(f"日期范围：{actual_start_date} 至 {end_date}")
            
            if actual_start_date <= end_date:
                batch_updater = get_batch_updater()
                stats = batch_updater.update_stocks_for_date_range(
                    stock_codes=stock_codes,
                    start_date=actual_start_date,
                    end_date=end_date
                )
                
                logger.info(f"数据更新完成: {stats['stocks_updated']} 已更新, {stats['stocks_failed']} 失败")
            else:
                logger.warning("无效的日期范围!")
        except Exception as e2:
            logger.error(f"旧版更新器也失败: {e2}")
            raise
    
    # 更新大盘数据缓存
    _update_market_data()
    
    # 更新板块数据
    _update_sector_data(service)


def _update_sector_data(service: StockSelectorService) -> None:
    """
    更新板块历史数据
    
    Args:
        service: StockSelectorService 实例
    """
    logger.info("开始更新板块数据...")
    
    try:
        from datetime import date, timedelta
        from stock_selector.trading_calendar import is_trading_day, get_trading_calendar
        
        # 获取 sector manager
        sector_manager = None
        if service.strategy_manager:
            sector_manager = service.strategy_manager.get_sector_manager()
        
        if not sector_manager:
            logger.warning("Sector manager 不可用，跳过板块数据更新")
            return
        
        # 获取数据提供者
        data_manager = None
        if hasattr(service.strategy_manager, '_data_provider'):
            data_manager = service.strategy_manager._data_provider
        
        if not data_manager:
            logger.warning("Data manager 不可用，跳过板块数据更新")
            return
        
        # 确定应该保存到哪个交易日
        current_date = date.today()
        target_date = current_date
        
        # 检查是否为交易日
        if not is_trading_day(current_date):
            logger.info(f"{current_date} 不是交易日，寻找最近的交易日...")
            
            # 向前查找最近的交易日
            trading_calendar = get_trading_calendar()
            all_trading_days = trading_calendar.get_all_trading_days()
            
            if all_trading_days:
                # 筛选出 <= 当前日期的交易日
                valid_days = [d for d in all_trading_days if d <= current_date]
                if valid_days:
                    target_date = valid_days[-1]
                    logger.info(f"使用最近的交易日: {target_date}")
                else:
                    logger.warning("找不到合适的交易日，跳过板块数据更新")
                    return
            else:
                logger.warning("交易日历为空，跳过板块数据更新")
                return
        
        # 获取所有板块数据
        all_sectors, _ = data_manager.get_sector_rankings(n=50, return_all=True)
        
        if not all_sectors:
            logger.warning("未获取到板块数据")
            return
        
        logger.info(f"获取到 {len(all_sectors)} 个板块")
        
        # 保存到数据库
        from src.storage import DatabaseManager
        db_manager = DatabaseManager.get_instance()
        
        # 保存所有板块数据
        saved_count = 0
        for sector in all_sectors:
            name = sector.get('name')
            change_pct = sector.get('change_pct')
            stock_count = sector.get('stock_count', 0)
            limit_up_count = sector.get('limit_up_count', 0)
            if name:
                if db_manager.save_sector_daily(
                    name=name,
                    date=target_date,
                    change_pct=change_pct,
                    stock_count=stock_count,
                    limit_up_count=limit_up_count,
                    data_source="data_fetcher"
                ):
                    saved_count += 1
        
        if saved_count > 0:
            logger.info(f"成功保存 {saved_count} 个板块数据，日期: {target_date}")
        else:
            logger.warning("保存板块数据失败")
            
    except Exception as e:
        logger.error(f"更新板块数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ---------- 异步选股（带进度追踪）----------

_screen_tasks: dict = {}
_screen_lock = threading.Lock()


def _run_screen_async(task_id: str, request: StockSelectorRequest):
    """后台线程：运行选股流程并更新进度"""
    import uuid as uuid_module

    task = None
    with _screen_lock:
        task = _screen_tasks.get(task_id)
        if not task:
            return

    try:
        service = get_stock_selector_service()

        # 获取股票列表（优先使用预计算的数据）
        all_codes = task.get("precomputed_all_codes", [])
        code_to_name = task.get("precomputed_code_to_name", {})
        if not all_codes:
            from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
            stock_code_name_pairs = get_all_stock_code_name_pairs(force_refresh=True)
            stock_code_name_pairs = filter_st_stocks(stock_code_name_pairs)
            all_codes = [code for code, name in stock_code_name_pairs]
            all_codes = filter_special_stock_codes(all_codes)
            code_to_name = {code: name for code, name in stock_code_name_pairs}
            with _screen_lock:
                task["total_stocks"] = len(all_codes)
        else:
            with _screen_lock:
                if not task.get("total_stocks"):
                    task["total_stocks"] = len(all_codes)

        total_stocks = len(all_codes)

        # 设置市场数据缓存的强制更新模式
        MarketDataCache.set_force_update(request.update_realtime or request.update_data)

        use_update_realtime = request.update_realtime
        use_update_data = request.update_data
        if use_update_realtime and use_update_data:
            use_update_data = False

        # --- 阶段1: 更新实时数据 ---
        if use_update_realtime:
            with _screen_lock:
                task["stage"] = "update_realtime"
                task["stage_progress"] = 0
                task["current_code"] = ""
                task["current_name"] = ""

            from stock_selector.realtime_data_updater import get_realtime_updater

            realtime_codes = request.stock_codes if request.stock_codes else all_codes
            realtime_updater = get_realtime_updater()

            logger.info(f"阶段1: 开始更新实时数据，共 {len(realtime_codes)} 只股票")

            # 分批次更新，报告进度
            batch_size = 300
            total_batches = (len(realtime_codes) + batch_size - 1) // batch_size
            for i in range(0, len(realtime_codes), batch_size):
                with _screen_lock:
                    if task.get("cancelled"):
                        task["status"] = "cancelled"
                        task["end_time"] = time.time()
                        return

                batch = realtime_codes[i : i + batch_size]
                with _screen_lock:
                    task["current_code"] = batch[0] if batch else ""
                    task["current_name"] = code_to_name.get(batch[0], "") if batch else ""
                    task["processed_stocks"] = i
                    task["stage_progress"] = round((i / len(realtime_codes)) * 100, 1)
                    task["elapsed_seconds"] = time.time() - task["start_time"]

                try:
                    realtime_updater.update_realtime_data(stock_codes=batch)
                except Exception as e:
                    logger.warning(f"更新实时数据批次失败: {batch[0]}-{batch[-1]}: {e}")

            _update_market_data()
            _update_sector_data(service)

            with _screen_lock:
                task["stage_progress"] = 100
                task["processed_stocks"] = len(realtime_codes)

        # --- 阶段2: 更新历史数据 ---
        if use_update_data:
            with _screen_lock:
                task["stage"] = "update_data"
                task["stage_progress"] = 0
                task["processed_stocks"] = 0

            from datetime import date, timedelta

            config = get_config()
            update_codes = request.stock_codes if request.stock_codes else all_codes

            logger.info(f"阶段2: 开始更新历史数据，共 {len(update_codes)} 只股票")

            try:
                from stock_selector.tushare_data_downloader import get_tushare_downloader

                downloader = get_tushare_downloader(rate_limit_per_minute=50)
                downloader.reset_rate_limit()
                task["tushare_downloader"] = downloader

                with _screen_lock:
                    task["current_code"] = update_codes[0] if update_codes else ""
                    task["current_name"] = code_to_name.get(update_codes[0], "") if update_codes else ""

                def download_progress(completed: int, total: int, current_code: str, current_name: str):
                    with _screen_lock:
                        if task.get("cancelled"):
                            downloader.stop()
                            return
                        task["stage_progress"] = round((completed / total) * 100, 1) if total > 0 else 0
                        task["processed_stocks"] = completed
                        task["current_code"] = current_code
                        task["current_name"] = code_to_name.get(current_code, current_name)
                        task["total_stocks"] = total
                        task["elapsed_seconds"] = time.time() - task["start_time"]

                downloader.download_data(
                    stock_codes=update_codes,
                    days=config.update_data_default_days,
                    verbose=False,
                    progress_callback=download_progress,
                )

            except Exception as e:
                logger.warning(f"Tushare下载器失败: {e}")
            finally:
                _update_market_data()
                _update_sector_data(service)

            with _screen_lock:
                task["stage_progress"] = 100
                task["processed_stocks"] = len(update_codes)
                task["elapsed_seconds"] = time.time() - task["start_time"]

        # --- 阶段3: 选股 ---
        with _screen_lock:
            task["stage"] = "screening"
            task["stage_progress"] = 0
            task["processed_stocks"] = 0
            if not task.get("total_stocks"):
                task["total_stocks"] = len(all_codes)

        logger.info(f"阶段3: 开始策略筛选，共 {task['total_stocks']} 只股票")

        MarketDataCache.set_force_update(False)

        def screening_progress(completed: int, total: int, current_code: str, current_name: str):
            with _screen_lock:
                if task.get("cancelled"):
                    cancel_event = task.get("cancel_event")
                    if cancel_event:
                        cancel_event.set()
                    return
                task["stage_progress"] = round((completed / total) * 100, 1)
                task["processed_stocks"] = completed
                task["current_code"] = current_code
                task["current_name"] = current_name
                task["elapsed_seconds"] = time.time() - task["start_time"]

        cancel_event = threading.Event()
        task["cancel_event"] = cancel_event

        candidates = service.screen_stocks(
            stock_codes=request.stock_codes,
            strategy_ids=request.strategy_ids,
            top_n=request.top_n,
            progress_callback=screening_progress,
            cancel_event=cancel_event,
            verbose=False,
        )

        with _screen_lock:
            task["stage_progress"] = 100

        sector_manager = None
        if service.strategy_manager:
            sector_manager = service.strategy_manager.get_sector_manager()

        active_ids = set(request.strategy_ids or [])
        candidate_infos = []
        if candidates:
            for candidate in candidates:
                info = _convert_stock_candidate_to_info(candidate, list(active_ids), sector_manager)
                candidate_infos.append(info)

        execution_time_ms = (time.time() - task["start_time"]) * 1000

        response = StockSelectorResponse(
            success=True,
            candidates=candidate_infos,
            total_screened=len(request.stock_codes) if request.stock_codes else total_stocks,
            execution_time_ms=execution_time_ms,
        )

        with _screen_lock:
            task["status"] = "completed"
            task["stage"] = "done"
            task["stage_progress"] = 100
            task["end_time"] = time.time()
            task["elapsed_seconds"] = task["end_time"] - task["start_time"]
            task["result"] = response

        logger.info(
            f"异步选股完成: {task_id}, 耗时{task['elapsed_seconds']:.1f}s, "
            f"候选{len(candidate_infos)}只"
        )

    except Exception as e:
        logger.error(f"异步选股异常: {task_id}: {e}", exc_info=True)
        with _screen_lock:
            task["status"] = "failed"
            task["stage"] = "done"
            task["end_time"] = time.time()
            task["elapsed_seconds"] = task["end_time"] - task["start_time"]
            task["result"] = StockSelectorResponse(
                success=False,
                candidates=[],
                total_screened=0,
                execution_time_ms=task["elapsed_seconds"] * 1000,
                error=str(e),
            )


@router.post(
    "/screen-async",
    response_model=ScreenProgressStatus,
    summary="异步选股（带进度追踪）",
    description="启动后台选股任务，支持实时进度查询",
)
def screen_stocks_async(
    request: StockSelectorRequest,
    service: StockSelectorService = Depends(get_stock_selector_service),
) -> ScreenProgressStatus:
    import uuid as uuid_module

    task_id = uuid_module.uuid4().hex[:12]

    # 预计算 total_stocks，避免前端轮询时显示"正在初始化..."
    try:
        from stock_selector.stock_pool import get_all_stock_code_name_pairs, filter_special_stock_codes, filter_st_stocks
        pairs = get_all_stock_code_name_pairs(force_refresh=True)
        pairs = filter_st_stocks(pairs)
        codes = [c for c, _ in pairs]
        codes = filter_special_stock_codes(codes)
        precomputed_total = len(codes)
        precomputed_code_to_name = {code: name for code, name in pairs}
    except Exception as e:
        logger.warning(f"预计算股票列表失败: {e}")
        precomputed_total = 0
        precomputed_code_to_name = {}

    task = {
        "task_id": task_id,
        "status": "running",
        "stage": "preparing",
        "stage_progress": 0,
        "total_stocks": precomputed_total,       # 预计算的值
        "processed_stocks": 0,
        "current_code": "",
        "current_name": "",
        "start_time": time.time(),
        "end_time": None,
        "errors": [],
        "result": None,
        "cancelled": False,
        "precomputed_all_codes": codes,           # 传递给后台线程
        "precomputed_code_to_name": precomputed_code_to_name,
    }

    with _screen_lock:
        _screen_tasks[task_id] = task

    thread = threading.Thread(
        target=_run_screen_async,
        args=(task_id, request),
        daemon=True,
    )
    thread.start()

    logger.info(
        f"启动异步选股: {task_id}, update_realtime={request.update_realtime}, "
        f"update_data={request.update_data}"
    )

    return ScreenProgressStatus(
        task_id=task_id,
        status="running",
        stage="preparing",
        stage_progress=0,
        total_stocks=precomputed_total,
        processed_stocks=0,
        current_code="",
        current_name="",
        elapsed_seconds=0.0,
    )


@router.get(
    "/screen-async/status",
    response_model=ScreenProgressStatus,
    summary="查询异步选股进度",
)
def get_screen_async_status(
    task_id: Optional[str] = Query(None, description="任务ID"),
) -> ScreenProgressStatus:
    with _screen_lock:
        if not task_id:
            if not _screen_tasks:
                return ScreenProgressStatus(status="idle")
            task_id = list(_screen_tasks.keys())[-1]

        task = _screen_tasks.get(task_id)
        if not task:
            return ScreenProgressStatus(task_id=task_id, status="idle")

        elapsed = time.time() - task["start_time"] if task["start_time"] else 0
        if task.get("end_time"):
            elapsed = task["end_time"] - task["start_time"]

        return ScreenProgressStatus(
            task_id=task_id,
            status=task["status"],
            stage=task["stage"],
            stage_progress=task["stage_progress"],
            total_stocks=task["total_stocks"],
            processed_stocks=task["processed_stocks"],
            current_code=task["current_code"],
            current_name=task["current_name"],
            elapsed_seconds=round(elapsed, 1),
            errors=task.get("errors", []),
            result=task.get("result"),
        )


@router.post(
    "/screen-async/cancel",
    response_model=ScreenProgressStatus,
    summary="取消异步选股任务",
)
def cancel_screen_async(
    task_id: Optional[str] = Query(None, description="任务ID"),
) -> ScreenProgressStatus:
    with _screen_lock:
        if not task_id:
            if not _screen_tasks:
                return ScreenProgressStatus(status="idle")
            task_id = list(_screen_tasks.keys())[-1]

        task = _screen_tasks.get(task_id)
        if not task:
            return ScreenProgressStatus(task_id=task_id, status="idle")

        if task["status"] == "running":
            task["cancelled"] = True
            task["status"] = "cancelled"
            task["stage"] = "done"
            task["end_time"] = time.time()
            cancel_event = task.get("cancel_event")
            if cancel_event:
                cancel_event.set()
            tushare_downloader = task.get("tushare_downloader")
            if tushare_downloader:
                tushare_downloader.stop()

        elapsed = task.get("end_time", time.time()) - task["start_time"]

        return ScreenProgressStatus(
            task_id=task_id,
            status=task["status"],
            stage=task["stage"],
            stage_progress=task.get("stage_progress", 0),
            total_stocks=task.get("total_stocks", 0),
            processed_stocks=task.get("processed_stocks", 0),
            current_code=task.get("current_code", ""),
            current_name=task.get("current_name", ""),
            elapsed_seconds=round(elapsed, 1),
            errors=task.get("errors", []),
            result=task.get("result"),
        )
