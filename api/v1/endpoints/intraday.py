# -*- coding: utf-8 -*-
"""
分时做T API 端点

提供分时K线数据获取、做T信号计算和支撑/压力参考线生成服务。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_database_manager
from api.v1.schemas.intraday import (
    DeleteHistoryResponse,
    IntradayDataResponse,
    IntradayKlinePoint,
    IntradaySignal,
    ReferenceLine,
    SearchHistoryItem,
    SearchHistoryRequest,
    SearchHistoryResponse,
)
from api.v1.schemas.common import ErrorResponse
from src.storage import DatabaseManager
from watchdog.strategies.reference_line_generator import (
    ReferenceLineGenerator,
    apply_gravitational_field,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _normalize_stock_code(code: str) -> str:
    """标准化股票代码，去除前缀 zh_ / sh / sz 等"""
    code = code.strip().upper()
    for prefix in ["ZH_", "SH.", "SZ.", "SH", "SZ"]:
        if code.startswith(prefix):
            code = code[len(prefix) :]
    return code


def _get_intraday_klines(stock_code: str, date_str: Optional[str] = None) -> list:
    """通过 akshare 获取分时K线数据

    Args:
        stock_code: 股票代码（如 000001, 600519）
        date_str: 日期字符串 YYYYMMDD 或 YYYY-MM-DD，None 为当日

    Returns:
        K线数据列表，每项为包含 Open/High/Low/Close/Volume/timestamp 的字典
    """
    try:
        import akshare as ak

        if date_str is not None:
            date_str = date_str.replace("-", "")

        # 判断市场
        code = _normalize_stock_code(stock_code)
        if code.startswith("6"):
            full_code = f"1.{code}" if len(code) == 6 else f"sh{code}"
        else:
            full_code = f"0.{code}" if len(code) == 6 else f"sz{code}"

        # 尝试使用 stock_intraday_em 获取分时数据
        try:
            if date_str:
                df = ak.stock_intraday_em(symbol=code, period="1", start_date=date_str, end_date=date_str)
            else:
                df = ak.stock_intraday_em(symbol=code, period="1")
        except Exception as e1:
            logger.warning(f"stock_intraday_em 失败: {e1}，尝试备选方案")
            try:
                df = ak.stock_bid_ask_em(symbol=code)
                logger.info("使用 stock_bid_ask_em 作为备选")
            except Exception as e2:
                logger.error(f"所有分时数据获取方案均失败: {e2}")
                raise HTTPException(
                    status_code=500,
                    detail={"error": "data_fetch_failed", "message": f"获取分时数据失败: {str(e2)}"},
                )

        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail={"error": "no_data", "message": f"未找到股票 {stock_code} 的分时数据"},
            )

        # 标准化列名（akshare 不同接口返回的列名可能不同）
        column_mapping = {
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
            "成交额": "Amount",
            "时间": "timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "amount": "Amount",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # 添加缺失列
        if "Amount" not in df.columns:
            if "Close" in df.columns and "Volume" in df.columns:
                df["Amount"] = df["Close"] * df["Volume"]
            else:
                df["Amount"] = 0.0

        # 转换时间戳
        klines = []
        for _, row in df.iterrows():
            try:
                ts = row.get("timestamp", row.get("时间", ""))
                if isinstance(ts, (datetime,)):
                    ts_str = ts.isoformat()
                elif isinstance(ts, str):
                    ts_str = ts
                else:
                    ts_str = str(ts)
            except Exception:
                ts_str = ""

            kline = {
                "Open": float(row.get("Open", row.get("open", 0))),
                "High": float(row.get("High", row.get("high", 0))),
                "Low": float(row.get("Low", row.get("low", 0))),
                "Close": float(row.get("Close", row.get("close", 0))),
                "Volume": float(row.get("Volume", row.get("volume", 0))),
                "Amount": float(row.get("Amount", row.get("amount", 0))),
                "timestamp": ts_str,
            }
            # 跳过价格为0的无效数据
            if kline["Close"] <= 0:
                continue
            klines.append(kline)

        logger.info(f"获取到 {len(klines)} 根K线数据")
        return klines

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail={"error": "import_error", "message": "akshare 未安装，无法获取分时数据"},
        )
    except Exception as e:
        logger.error(f"获取分时数据失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"获取分时数据失败: {str(e)}"},
        )


def _get_daily_history(stock_code: str, days: int = 120) -> list:
    """获取历史日线数据，用于参考线计算

    Args:
        stock_code: 股票代码
        days: 获取天数

    Returns:
        日线K线数据列表，每项为字典
    """
    try:
        import akshare as ak

        code = _normalize_stock_code(stock_code)
        market = "sh" if code.startswith("6") else "sz"
        full_code = f"{market}{code}"

        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

        if df is None or df.empty:
            logger.warning(f"日线历史数据为空，stock_code={stock_code}")
            return []

        column_mapping = {
            "日期": "date",
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
            "成交额": "Amount",
            "换手率": "turnover_rate",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        daily_data = []
        for _, row in df.iterrows():
            daily_data.append(
                {
                    "date": str(row.get("date", "")),
                    "Open": float(row.get("Open", 0)),
                    "High": float(row.get("High", 0)),
                    "Low": float(row.get("Low", 0)),
                    "Close": float(row.get("Close", 0)),
                    "Volume": float(row.get("Volume", 0)),
                    "Amount": float(row.get("Amount", 0)) if "Amount" in df.columns else 0.0,
                    "turnover_rate": float(row.get("turnover_rate", 0)) if "turnover_rate" in df.columns else 0.0,
                }
            )

        return daily_data

    except ImportError:
        logger.warning("akshare 未安装，无法获取日线数据")
        return []
    except Exception as e:
        logger.warning(f"获取日线数据失败: {e}")
        return []


def _run_t0_strategy(klines: list, reference_lines: list = None) -> tuple:
    """对分时K线数据运行做T策略，生成信号列表

    Args:
        klines: 分时K线字典列表
        reference_lines: 参考线列表（用于引力场置信度修正）

    Returns:
        (signals_list, signal_summary_dict)
    """
    try:
        from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy

        strategy = IntradayT0Strategy(stock_code="temp", stock_name="")

        signals = []
        for kline in klines:
            sig = strategy.feed_kline(kline)
            if sig is not None:
                signals.append(sig)

        result_signals = []
        buy_count = 0
        sell_count = 0
        strong_count = 0
        medium_count = 0
        weak_count = 0
        total_return = 0.0
        last_buy_price = None

        for sig in signals:
            base_conf = sig.confidence

            # 应用引力场模型
            gravity_adjust = 0.0
            support_f = 0.0
            pressure_f = 0.0
            if reference_lines and sig.price > 0:
                adj_conf = apply_gravitational_field(
                    current_price=sig.price,
                    reference_lines=reference_lines,
                    signal_type=sig.signal_type,
                    base_confidence=base_conf,
                )
                gravity_adjust = round(adj_conf - base_conf, 4)
                # 近似计算support/pressure force
                for line in reference_lines:
                    lp = line.get("price", 0)
                    bw = line.get("base_weight", 1.0)
                    if lp <= 0:
                        continue
                    rd = (lp - sig.price) / sig.price
                    ad = abs(rd)
                    if ad > 0.03:
                        continue
                    inf = bw / (1.0 + ad / 0.01)
                    if rd < 0:
                        support_f += inf
                    else:
                        pressure_f += inf
                support_f = round(support_f, 4)
                pressure_f = round(pressure_f, 4)
                # 重新计算置信度
                final_conf = base_conf + gravity_adjust
                final_conf = max(0.0, min(1.0, final_conf))
            else:
                final_conf = base_conf

            # 仓位建议（根据最终置信度）
            if final_conf >= 0.80:
                pos = "全仓"
            elif final_conf >= 0.55:
                pos = "半仓"
            elif final_conf >= 0.30:
                pos = "1/3仓"
            else:
                pos = "观望"

            # 信号统计
            if sig.signal_type == "buy":
                buy_count += 1
                last_buy_price = sig.price
            elif sig.signal_type == "sell":
                sell_count += 1
                if last_buy_price is not None and last_buy_price > 0:
                    trade_return = (sig.price - last_buy_price) / last_buy_price * 100
                    total_return += trade_return
                    last_buy_price = None

            if final_conf >= 0.75:
                strong_count += 1
            elif final_conf >= 0.50:
                medium_count += 1
            else:
                weak_count += 1

            result_signals.append(
                IntradaySignal(
                    stock_code=sig.stock_code,
                    signal_type=sig.signal_type,
                    trigger_time=str(sig.trigger_time),
                    price=round(sig.price, 2),
                    score=sig.score,
                    max_score=sig.max_score,
                    confidence=round(final_conf, 4),
                    position_advice=pos,
                    reasoning=sig.reasoning,
                    gravity_adjustment=round(gravity_adjust, 4),
                    support_force=support_f,
                    pressure_f=pressure_f,
                )
            )

        summary = {
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "total_signals": buy_count + sell_count,
            "strong_signals": strong_count,
            "medium_signals": medium_count,
            "weak_signals": weak_count,
            "simulated_return_pct": round(total_return, 2),
        }

        return result_signals, summary

    except ImportError as e:
        logger.warning(f"导入做T策略失败: {e}，返回空信号")
        return [], {"buy_signals": 0, "sell_signals": 0, "total_signals": 0}
    except Exception as e:
        logger.error(f"运行做T策略失败: {e}", exc_info=True)
        return [], {"buy_signals": 0, "sell_signals": 0, "total_signals": 0, "error": str(e)}


# ============================================================
# 路由定义
# ============================================================


@router.get(
    "/data/{stock_code}",
    response_model=IntradayDataResponse,
    responses={
        200: {"description": "分时K线数据 + 信号 + 参考线"},
        404: {"description": "未找到数据", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取股票分时做T数据",
    description="获取指定股票的分时K线数据、做T买卖信号和支撑/压力参考线",
)
def get_intraday_data(
    stock_code: str,
    date: Optional[str] = Query(None, description="日期 YYYYMMDD 或 YYYY-MM-DD，默认当日"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> IntradayDataResponse:
    """获取分时K线数据、信号和参考线"""
    try:
        code = _normalize_stock_code(stock_code)
        date_str = date

        # 获取分时K线
        klines = _get_intraday_klines(code, date_str)
        if not klines:
            raise HTTPException(status_code=404, detail={"error": "no_data", "message": "未获取到分时K线数据"})

        # 获取日线历史数据并生成参考线
        daily_raw = _get_daily_history(code, days=120)
        reference_lines_dicts = []
        if daily_raw:
            import pandas as pd

            df_daily = pd.DataFrame(daily_raw)
            if "date" in df_daily.columns:
                df_daily["date"] = pd.to_datetime(df_daily["date"])
                df_daily = df_daily.set_index("date").sort_index()
            try:
                generator = ReferenceLineGenerator(df_daily)
                reference_lines_dicts = generator.generate_all()
            except Exception as e:
                logger.warning(f"参考线生成失败: {e}")
        reference_lines = [ReferenceLine(**rl) for rl in reference_lines_dicts]

        # 运行做T策略 + 引力场修正
        signals, summary = _run_t0_strategy(klines, reference_lines_dicts)

        # 构建K线响应
        kline_points = [IntradayKlinePoint(**k) for k in klines]

        # 确定日期
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        elif len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        return IntradayDataResponse(
            stock_code=code,
            stock_name="",
            date=date_str,
            kline_data=kline_points,
            signals=signals,
            reference_lines=reference_lines,
            signal_summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取分时数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(e)})


# ---------- 搜索历史 ----------


@router.get(
    "/history",
    response_model=SearchHistoryResponse,
    summary="获取分时搜索历史",
)
def get_search_history(
    limit: int = Query(20, ge=1, le=100),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> SearchHistoryResponse:
    """获取分时搜索历史记录"""
    try:
        rows = db_manager.execute_query(
            """
            SELECT id, stock_code, stock_name, search_date, search_time
            FROM intraday_search_history
            ORDER BY search_time DESC
            LIMIT ?
            """,
            (limit,),
        )

        items = []
        for row in rows:
            items.append(
                SearchHistoryItem(
                    id=row[0],
                    stock_code=row[1],
                    stock_name=row[2] or "",
                    date=row[3] or "",
                    search_time=row[4] or "",
                )
            )

        total = db_manager.execute_query("SELECT COUNT(*) FROM intraday_search_history", ())
        total_count = total[0][0] if total else 0

        return SearchHistoryResponse(items=items, total=total_count)

    except Exception as e:
        logger.error(f"获取搜索历史失败: {e}")
        # 表不存在时返回空
        return SearchHistoryResponse(items=[], total=0)


@router.post(
    "/history",
    response_model=SearchHistoryItem,
    summary="保存分时搜索历史",
)
def save_search_history(
    request: SearchHistoryRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> SearchHistoryItem:
    """保存一条搜索历史"""
    try:
        now = datetime.now().isoformat()

        db_manager.execute_query(
            """
            CREATE TABLE IF NOT EXISTS intraday_search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                search_date TEXT DEFAULT '',
                search_time TEXT DEFAULT ''
            )
            """,
            (),
        )

        db_manager.execute_query(
            """
            INSERT INTO intraday_search_history (stock_code, stock_name, search_date, search_time)
            VALUES (?, ?, ?, ?)
            """,
            (request.stock_code, request.stock_name or "", request.date or "", now),
        )

        last_id = db_manager.execute_query("SELECT last_insert_rowid()", ())
        item_id = last_id[0][0] if last_id else 0

        return SearchHistoryItem(
            id=item_id,
            stock_code=request.stock_code,
            stock_name=request.stock_name or "",
            date=request.date or "",
            search_time=now,
        )

    except Exception as e:
        logger.error(f"保存搜索历史失败: {e}")
        raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(e)})


@router.delete(
    "/history/{history_id}",
    response_model=DeleteHistoryResponse,
    summary="删除分时搜索历史",
)
def delete_search_history(
    history_id: int,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> DeleteHistoryResponse:
    """删除一条搜索历史"""
    try:
        db_manager.execute_query("DELETE FROM intraday_search_history WHERE id = ?", (history_id,))
        return DeleteHistoryResponse(success=True, message="删除成功")
    except Exception as e:
        logger.error(f"删除搜索历史失败: {e}")
        return DeleteHistoryResponse(success=False, message=str(e))
