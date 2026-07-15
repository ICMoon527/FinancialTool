# -*- coding: utf-8 -*-
"""
分时做T API 端点

提供分时K线数据获取和做T信号计算服务。
支持多数据源 fallback 机制，提高数据获取成功率。
"""

import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import yaml
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Optional, Callable, Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from requests.exceptions import (
    ConnectionError,
    ReadTimeout,
    ConnectTimeout,
    RequestException,
)
from sqlalchemy import text

from api.deps import get_database_manager
from api.v1.schemas.intraday import (
    BatchDownloadStatus,
    BatchStatusRequest,
    BatchStatusResponse,
    DeleteHistoryResponse,
    FiveMinKlinePoint,
    IndicatorLine,
    IndicatorLinePoint,
    IndicatorSubChart,
    IntradayDataResponse,
    IntradayKlinePoint,
    IntradaySignal,
    SearchHistoryItem,
    SearchHistoryRequest,
    SearchHistoryResponse,
    SignalAlert,
    SimulatedTradeItem,
    SimulationReportResponse,
    StockSnapshot,
    TiandaoSignal,
    TiandaoSubChart,
    WeightContribution,
)
from api.v1.schemas.common import ErrorResponse
from src.storage import DatabaseManager
from data_provider import DataFetcherManager
from watchdog.strategies.intraday_t0_strategy import IntradayIndicatorEngine
from src.config import get_config

logger = logging.getLogger(__name__)

# ── 天道5分钟信号缓存：key=(code, date)，value=已生成的信号列表 ──
# 确保信号按时间顺序产生，仅使用评估时刻的历史数据，避免XMA未来数据泄露
_tiandao_5min_signals_cache: dict = {}

# ── 股票名称缓存 ──
_stock_display_cache: Dict[str, str] = {}

def _format_stock_display(code: str) -> str:
    """格式化股票显示名，如 '工业富联 (601138)'"""
    if not _stock_display_cache:
        try:
            from stock_selector.stock_pool import get_all_stock_code_name_pairs
            for c, n in get_all_stock_code_name_pairs(force_refresh=False):
                _stock_display_cache[c] = n
        except Exception:
            pass
    name = _stock_display_cache.get(code, "")
    return f"{name} ({code})" if name else code

# ── 分时数据内存缓存（包含完整 K线+信号+参考线+指标子图，避免轮询与手动点击之间的重复计算）──
_full_response_cache: dict = {}
_full_response_cache_lock = threading.Lock()

# ── 分时K线缓存（信号检测时写入，供 get_intraday_data 跳过外部API）──
_klines_cache: dict = {}
_klines_cache_lock = threading.Lock()


def _get_polling_interval() -> int:
    """获取分时轮询间隔（秒），从统一配置读取"""
    return get_config().intraday_polling_interval


def _get_cache_ttl() -> int:
    """获取缓存 TTL（秒），= 轮询间隔 + 5s 缓冲"""
    return _get_polling_interval() + 5


def _get_cached_full_response(code: str, warmup_enabled: bool) -> Optional[IntradayDataResponse]:
    """获取缓存的完整分时数据响应"""
    key = (code, warmup_enabled)
    with _full_response_cache_lock:
        entry = _full_response_cache.get(key)
        if entry and time.time() - entry['timestamp'] < _get_cache_ttl():
            return entry['response']
    return None


def _set_cached_full_response(code: str, warmup_enabled: bool, response: IntradayDataResponse):
    """缓存完整分时数据响应"""
    key = (code, warmup_enabled)
    with _full_response_cache_lock:
        _full_response_cache[key] = {
            'timestamp': time.time(),
            'response': response,
        }


def _get_cached_klines(code: str) -> Optional[list]:
    """获取缓存的K线数据（来自信号检测）"""
    with _klines_cache_lock:
        entry = _klines_cache.get(code)
        if entry and time.time() - entry['timestamp'] < _get_cache_ttl():
            return entry['klines']
    return None


def _set_cached_klines(code: str, klines: list):
    """缓存K线数据"""
    with _klines_cache_lock:
        _klines_cache[code] = {
            'timestamp': time.time(),
            'klines': klines,
        }

def _get_project_root():
    """获取项目根目录（兼容 PyInstaller 打包和开发模式）"""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent.parent


CONFIG_PATH = _get_project_root() / "watchdog" / "strategies" / "intraday_t0_config.yaml"


def _load_indicator_config() -> dict:
    """每次调用时重新加载YAML配置，支持热更新"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f'加载指标配置文件失败，使用默认参数: {e}')
    return {}


def _get_rsi_thresholds():
    """获取RSI超买超卖阈值，来自策略配置文件"""
    config = _load_indicator_config()
    indicators = config.get("indicators", {})
    rsi_cfg = indicators.get("rsi", {})
    return rsi_cfg.get("overbought", 65), rsi_cfg.get("oversold", 20)


def _get_mfi_thresholds():
    """获取MFI超买超卖阈值，来自策略配置文件"""
    config = _load_indicator_config()
    indicators = config.get("indicators", {})
    mfi_cfg = indicators.get("mfi", {})
    return mfi_cfg.get("overbought", 80), mfi_cfg.get("oversold", 20)


def _get_signal_weights():
    """获取买卖信号权重配置，来自策略配置文件"""
    config = _load_indicator_config()
    signals_cfg = config.get("signals", {})
    buy_cfg = signals_cfg.get("buy", {})
    sell_cfg = signals_cfg.get("sell", {})
    return buy_cfg.get("weights", {}), sell_cfg.get("weights", {})


router = APIRouter()


@router.get("/config")
def get_intraday_config() -> dict:
    """返回分时页面前端所需配置（轮询间隔、交易状态等），前端启动时调用一次"""
    config = get_config()
    trading_status = _get_trading_status()
    return {
        "polling_interval_ms": config.intraday_polling_interval * 1000,
        "batch_download_polling_interval_ms": config.batch_download_polling_interval * 1000,
        "screen_async_polling_interval_ms": config.screen_async_polling_interval * 1000,
        "trading_status": trading_status,
    }


def _get_trading_status() -> dict:
    """
    获取当前交易状态，供前端判断是否启动轮询及自动启动时机。

    返回字段：
    - is_trading_day: 今日是否为交易日（基于交易日历）
    - is_trading_time: 当前是否在盘中交易时段（9:30-11:30 或 13:00-15:00，排除午休）
    - next_session_start: 下一个交易时段开始时间（ISO格式），若无需等待则为 null

    前端使用场景：
    - is_trading_day=true → startPolling() 的交易日守卫，防止节假日误启动
    - is_trading_time=false 且 next_session_start 不为 null → 设定时器到该时间再启动
    - 定时器触发时前端更新 is_trading_day=true，确保轮询在正确的交易日启动
    """
    now = datetime.now()
    today = now.date()

    is_td = _is_trading_day(today)
    is_tt = _is_in_trading_window(now)

    next_start = _calc_next_session_start(now, today, is_td, is_tt)

    return {
        "is_trading_day": is_td,
        "is_trading_time": is_tt,
        "next_session_start": next_start.isoformat() if next_start else None,
    }


def _calc_next_session_start(
    now: datetime, today: date_type, is_trading_day: bool, is_trading_time: bool
) -> Optional[datetime]:
    """
    计算下一个交易时段开始时间。

    逻辑：
    - 盘中 → 返回 None（无需等待）
    - 交易日盘前 → 返回今日 9:30
    - 交易日午休 → 返回今日 13:00
    - 交易日收盘后 / 非交易日 → 返回下一个交易日 9:30
    """
    if is_trading_time:
        return None

    # 交易日盘前
    if is_trading_day:
        morning_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0)
        if now < morning_start:
            return morning_start
        if now < afternoon_start:
            return afternoon_start

    # 交易日收盘后 或 非交易日：找下一个交易日
    next_day = today + timedelta(days=1)
    for _ in range(30):  # 最多向前找30天
        if _is_trading_day(next_day):
            return datetime(next_day.year, next_day.month, next_day.day, 9, 30, 0)
        next_day = next_day + timedelta(days=1)

    return None


def _normalize_stock_code(code: str) -> str:
    """标准化股票代码，去除前缀 zh_ / sh / sz 等"""
    code = code.strip().upper()
    for prefix in ["ZH_", "SH.", "SZ.", "SH", "SZ"]:
        if code.startswith(prefix):
            code = code[len(prefix) :]
    return code


# 沪市 ETF 代码前缀
_SH_ETF_PREFIXES = ("51", "52", "56", "58")


def _get_market_prefix(code: str) -> str:
    """根据股票代码判断交易所前缀（sh/sz），兼容 ETF 代码规则

    沪市：6xxxxx（A股）、51xxxx/52xxxx/56xxxx/58xxxx（ETF）
    深市：0xxxxx/3xxxxx（A股）、15xxxx/16xxxx/18xxxx（ETF）
    """
    if code.startswith("6") or code.startswith(_SH_ETF_PREFIXES):
        return "sh"
    return "sz"


def _extract_date_from_klines(klines: list, fallback: str = "") -> str:
    """从K线数据提取实际日期，避免请求参数与实际数据日期不一致"""
    if klines and len(klines) > 0:
        ts = klines[0].get('timestamp', '')
        if ts:
            return ts[:10]
    return fallback or datetime.now().strftime("%Y-%m-%d")


def _is_data_fresh(klines: list, target_date: str) -> bool:
    """检查K线数据相对于目标日期是否新鲜

    校验规则:
      1. klines 为空 → 陈旧
      2. 数据日期 ≠ 目标日期 → 陈旧
      3. 非交易时段: 最后一条K线时间戳在 15:00-15:30（收盘及尾盘集合竞价）→ 新鲜，否则 → 陈旧
      4. 交易时段内: 最后一根K线距今 ≤ 30秒 → 新鲜，否则 → 陈旧
    """
    if not klines or len(klines) == 0:
        return False

    actual_date = _extract_date_from_klines(klines)
    if actual_date != target_date:
        logger.debug(f"数据日期 {actual_date} 与目标日期 {target_date} 不匹配，标记为陈旧")
        return False

    now = datetime.now()
    if _is_in_trading_window(now):
        last_ts = klines[-1].get('timestamp', '')
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
                if (now - last_dt).total_seconds() > _get_polling_interval():
                    logger.debug(
                        f"最后一根K线时间 {last_ts} 距今 {(now - last_dt).total_seconds():.0f} 秒，"
                        f"超过{_get_polling_interval()}秒阈值，标记为陈旧"
                    )
                    return False
            except (ValueError, TypeError):
                pass
    else:
        last_ts = klines[-1].get('timestamp', '')
        if last_ts:
            import re
            match = re.search(r'(\d{2}):(\d{2})', last_ts)
            if match:
                hh, mm = int(match.group(1)), int(match.group(2))
                if hh != 15 or mm > 30:
                    logger.debug(f"非交易时段，最后一条K线时间 {last_ts} 非15:00-15:30收盘范围，标记为陈旧")
                    return False

    return True


def _is_in_trading_window(now: datetime = None) -> bool:
    """判断当前是否处于A股盘中交易时段（9:30-11:30, 13:00-15:00）

    使用交易日历判断，避免误判节假日。
    排除午休时段（11:30-13:00），与前端轮询保持一致。
    返回 True 表示当前是交易日且处于盘中交易时段。
    """
    if now is None:
        now = datetime.now()

    today = now.date()
    try:
        from stock_selector.trading_calendar import is_trading_day
        if not is_trading_day(today):
            return False
    except ImportError:
        # 回退：仅用工作日判断
        if now.weekday() >= 5:
            return False

    # 上午盘：9:30-11:30
    morning_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    morning_close = now.replace(hour=11, minute=30, second=0, microsecond=0)
    if morning_open <= now <= morning_close:
        return True

    # 下午盘：13:00-15:00
    afternoon_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return afternoon_open <= now <= afternoon_close


def _is_trading_day(target_date):
    """判断指定日期是否为A股交易日（使用交易日历）"""
    try:
        from stock_selector.trading_calendar import is_trading_day
        return is_trading_day(target_date)
    except ImportError:
        return target_date.weekday() < 5


def _get_nearest_trading_day(target_date):
    """获取目标日期之前（含当天）的最近一个交易日，最多向前查找30天"""
    try:
        from stock_selector.trading_calendar import get_previous_trading_day
        result = get_previous_trading_day(target_date)
        if result is not None:
            return result
    except ImportError:
        pass

    current = target_date
    for _ in range(30):
        if current.weekday() < 5:
            return current
        current = current - timedelta(days=1)
    return target_date


def _try_multiple_sources(
    fetch_functions: list[tuple[str, Callable]],
    data_type: str = "数据"
) -> Any:
    """
    尝试多个数据源获取数据，直到成功或全部失败

    Args:
        fetch_functions: 数据源函数列表，格式为 [(source_name, fetch_func), ...]
        data_type: 数据类型描述，用于日志

    Returns:
        获取到的数据

    Raises:
        Exception: 当所有数据源都失败时抛出异常
    """
    last_exception = None

    for source_name, fetch_func in fetch_functions:
        try:
            logger.debug(f"尝试使用 {source_name} 获取{data_type}...")
            result = fetch_func()

            # 检查结果是否有效
            if result is not None and (not hasattr(result, "empty") or not result.empty):
                logger.debug(f"使用 {source_name} 成功获取{data_type}")
                return result
            else:
                logger.debug(f"{source_name} 返回空数据，尝试下一个数据源")
                continue

        except Exception as e:
            logger.debug(f"{source_name} 获取{data_type}失败: {type(e).__name__}: {e}")
            last_exception = e
            continue

    # 所有数据源都失败
    error_msg = f"所有数据源获取{data_type}均失败"
    if last_exception:
        error_msg += f"，最后失败原因: {type(last_exception).__name__}: {last_exception}"
    logger.error(error_msg)
    raise Exception(error_msg)


def _get_intraday_klines(stock_code: str, date_str: Optional[str] = None) -> list:
    """通过多数据源获取分时K线数据

    支持的数据源（按优先级）:
    1. 腾讯财经1分钟分时 (高粒度，首选)
    2. 新浪财经5分钟K线 (标准OHLC，降级备选)
    （东方财富及akshare其他接口已禁用，因IP被限制）

    Args:
        stock_code: 股票代码（如 000001，600519）或指数代码（如 sh000001）
        date_str: 日期字符串 YYYYMMDD 或 YYYY-MM-DD，None 为当日

    Returns:
        K线数据列表，每项为包含 Open/High/Low/Close/Volume/timestamp 的字典
    """
    try:
        import akshare as ak
        import requests

        # 检测是否为指数代码（带 sh/sz 前缀）
        is_index_code = bool(re.match(r'^[Ss][Hh]\d{6}$', stock_code) or re.match(r'^[Ss][Zz]\d{6}$', stock_code))
        if is_index_code:
            code = stock_code  # 使用完整代码（如 sh000001），跳过归一化
        else:
            code = _normalize_stock_code(stock_code)

        if date_str is not None:
            date_str = date_str.replace("-", "")
            if len(date_str) == 8:
                target_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                target_date = datetime.now().strftime("%Y-%m-%d")
        else:
            target_date = datetime.now().strftime("%Y-%m-%d")

        # 定义多个数据源函数
        fetch_functions = []

        # 数据源 1: 腾讯财经1分钟分时 (高粒度，首选)
        def fetch_tencent_1min():
            import pandas as pd

            if is_index_code:
                # 指数代码已带前缀，直接使用
                symbol = code
                market_prefix = code[:2]
            else:
                market_prefix = _get_market_prefix(code)
                symbol = f"{market_prefix}{code}"
            url = f"https://ifzq.gtimg.cn/appstock/app/minute/query?code={symbol}"
            r = requests.get(url, timeout=8)
            data = r.json()
            stock_data = data.get("data", {}).get(symbol, {})
            point_list = stock_data.get("data", {}).get("data", [])
            # 尝试从腾讯响应中提取实际日期
            resp_date = stock_data.get("data", {}).get("date", "")
            if resp_date and len(resp_date) == 8:
                actual_day = f"{resp_date[:4]}-{resp_date[4:6]}-{resp_date[6:8]}"
            else:
                actual_day = target_date
            if not point_list:
                return None
            records = []
            prev_vol = 0.0
            for line in point_list:
                parts = line.split()
                if len(parts) < 3:
                    continue
                hour = parts[0][:2]
                minute = parts[0][2:]
                price = float(parts[1])
                cur_vol = float(parts[2])
                per_vol = max(cur_vol - prev_vol, 0.0)
                prev_vol = cur_vol
                records.append({
                    "timestamp": f"{actual_day}T{hour}:{minute}:00",
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": per_vol,
                })
            if not records:
                return None
            df = pd.DataFrame(records)
            # 如果腾讯返回的日期与请求日期不一致，记录警告
            if actual_day != target_date:
                logger.warning(
                    f"腾讯API返回日期({actual_day})与请求日期({target_date})不一致，"
                    f"可能出现日期错配"
                )
            return df

        fetch_functions.append(("腾讯财经1分钟分时", fetch_tencent_1min))

        # 数据源 2: 新浪财经5分钟K线 (标准OHLC，降级备选)
        def fetch_sina_5min():
            import pandas as pd

            symbol_str = code if is_index_code else f"{_get_market_prefix(code)}{code}"
            url = (
                "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                "CN_MarketData.getKLineData"
                f"?symbol={symbol_str}&scale=5&ma=no&datalen=250"
            )
            headers = {"Referer": "https://finance.sina.com.cn"}
            r = requests.get(url, headers=headers, timeout=8)
            data = r.json()
            if not data or not isinstance(data, list):
                return None
            records = []
            for item in data:
                day_str = item.get("day", "")
                if not day_str.startswith(target_date):
                    continue
                time_part = day_str.split()[-1] if " " in day_str else ""
                actual_day = day_str.split()[0] if " " in day_str else target_date
                records.append({
                    "timestamp": f"{actual_day}T{time_part}",
                    "Open": float(item["open"]),
                    "High": float(item["high"]),
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                    "Volume": float(item["volume"]),
                })
            if not records:
                return None
            return pd.DataFrame(records)

        fetch_functions.append(("新浪财经5分钟K线", fetch_sina_5min))

        # 数据源 3: 东方财富 (stock_zh_a_hist_min_em) — 已禁用，因 IP 被限制
        # if hasattr(ak, 'stock_zh_a_hist_min_em'):
        #     def fetch_em():
        #         df = ak.stock_zh_a_hist_min_em(
        #             symbol=code,
        #             period="1",
        #             adjust="",
        #         )
        #         return df
        #     fetch_functions.append(("东方财富接口", fetch_em))

        # 数据源 4: akshare 其他分时接口 — 已禁用，因 IP 被限制
        # for attr_name in dir(ak):
        #     if 'hist_min' in attr_name and (attr_name.startswith('stock') or attr_name.startswith('ak')):
        #         try:
        #             func = getattr(ak, attr_name)
        #             def create_fetch_func(f):
        #                 def fetch_func():
        #                     try:
        #                         return f(symbol=code, period="1", adjust="")
        #                     except:
        #                         try:
        #                             return f(symbol=code)
        #                         except:
        #                             try:
        #                                 return f(code)
        #                             except:
        #                                 raise
        #                 return fetch_func
        #             fetch_functions.append((attr_name, create_fetch_func(func)))
        #         except:
        #             continue

        if not fetch_functions:
            raise HTTPException(
                status_code=500,
                detail={"error": "no_data_source", "message": "没有可用的数据源"},
            )

        # 尝试所有数据源
        df = _try_multiple_sources(fetch_functions, "分时K线数据")

        # 按目标日期筛选（如果有时间字段）
        if '时间' in df.columns:
            df = df[df['时间'].astype(str).str.startswith(target_date)].copy()
        elif 'datetime' in df.columns:
            df = df[df['datetime'].astype(str).str.startswith(target_date)].copy()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail={"error": "no_data", "message": f"股票 {stock_code} 在 {target_date} 无分时数据"},
            )

        # 标准化列名
        column_mapping = {
            '开盘': 'Open', '最高': 'High', '最低': 'Low', '收盘': 'Close',
            '成交量': 'Volume', '成交额': 'Amount', '时间': 'timestamp',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'amount': 'Amount', 'datetime': 'timestamp',
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # 添加缺失列
        if 'Amount' not in df.columns:
            if 'Close' in df.columns and 'Volume' in df.columns:
                df['Amount'] = df['Close'] * df['Volume']
            else:
                df['Amount'] = 0.0

        # 转换时间戳
        klines = []
        for _, row in df.iterrows():
            try:
                ts = row.get('timestamp', row.get('时间', row.get('datetime', '')))
                if isinstance(ts, (datetime,)):
                    ts_str = ts.isoformat()
                elif isinstance(ts, str):
                    ts_str = ts
                else:
                    ts_str = str(ts)
            except Exception:
                ts_str = ''

            kline = {
                'Open': float(row.get('Open', row.get('open', 0))),
                'High': float(row.get('High', row.get('high', 0))),
                'Low': float(row.get('Low', row.get('low', 0))),
                'Close': float(row.get('Close', row.get('close', 0))),
                'Volume': float(row.get('Volume', row.get('volume', 0))),
                'Amount': float(row.get('Amount', row.get('amount', 0))),
                'timestamp': ts_str,
            }
            # 跳过价格为0的无效数据
            if kline['Close'] <= 0:
                continue
            klines.append(kline)

        logger.debug(f"获取到 {len(klines)} 根K线数据")
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


def _run_t0_strategy(klines: list, reference_lines: list = None, warmup_klines: list = None) -> tuple:
    """对分时K线数据运行做T策略，生成信号列表

    Args:
        klines: 分时K线字典列表
        reference_lines: 日线级参考线列表
        warmup_klines: 前一交易日最后N根K线，用于指标预热

    Returns:
        (signals_list, signal_summary_dict, precomputed_result, precomputed_engine)
        其中 precomputed_result 和 precomputed_engine 用于复用，避免 _generate_indicator_sub_charts 重复计算
    """
    try:
        from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy

        config_path = str(_get_project_root() / "watchdog" / "strategies" / "intraday_t0_config.yaml")
        config_path = os.path.normpath(config_path)
        strategy = IntradayT0Strategy(stock_code="temp", stock_name="", config_path=config_path)

        # 预热：将前一日最后N根K线加载到buffer中
        if warmup_klines:
            strategy.buffer.warmup(warmup_klines)

        # 一次性预计算全量指标（避免循环中重复计算）
        import pandas as pd
        df_klines = pd.DataFrame(klines)
        if 'Amount' not in df_klines.columns:
            df_klines['Amount'] = df_klines['Close'] * df_klines['Volume'] if 'Close' in df_klines.columns and 'Volume' in df_klines.columns else 0.0
        warmup_rows = 0
        if warmup_klines:
            warmup_df = pd.DataFrame(warmup_klines)
            if 'Amount' not in warmup_df.columns:
                warmup_df['Amount'] = warmup_df['Close'] * warmup_df['Volume']
            warmup_rows = len(warmup_df)
            full_df = pd.concat([warmup_df, df_klines], ignore_index=True)
        else:
            full_df = df_klines
        precomputed_result = strategy.engine.calculate_all(full_df)

        # 统一前后端 MACD_Bar_Sum 计算逻辑：仅累计当日分时数据（预热数据不参与累加）
        # 前端从 indicator_sub_charts（仅当日数据）计算 runningMacdBarSum，
        # 后端原逻辑对全量（含预热）做 cumsum()，导致 MACD_Bar_Sum 被预热期负值压低
        if warmup_rows > 0 and "MACD_Bar" in precomputed_result.columns:
            today_macd_bars = precomputed_result.iloc[warmup_rows:]["MACD_Bar"].fillna(0)
            today_cumsum = today_macd_bars.cumsum()
            precomputed_result.loc[precomputed_result.index[warmup_rows:], "MACD_Bar_Sum"] = today_cumsum.values

        ref_lines_raw = reference_lines or []
        ref_lines = [
            {"id": rl.id if hasattr(rl, "id") else rl.get("id", ""),
             "price": rl.price if hasattr(rl, "price") else rl.get("price", 0)}
            for rl in ref_lines_raw
        ]

        signals = []
        for i, kline in enumerate(klines):
            sig = strategy.feed_kline(kline, ref_lines, precomputed_df=precomputed_result.iloc[:warmup_rows + i + 1])
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
            if sig.confidence >= 0.80:
                pos = "全仓"
            elif sig.confidence >= 0.55:
                pos = "半仓"
            elif sig.confidence >= 0.30:
                pos = "1/3仓"
            else:
                pos = "观望"

            if sig.signal_type == "buy":
                buy_count += 1
                last_buy_price = sig.price
            elif sig.signal_type == "sell":
                sell_count += 1
                if last_buy_price is not None and last_buy_price > 0:
                    trade_return = (sig.price - last_buy_price) / last_buy_price * 100
                    total_return += trade_return
                    last_buy_price = None

            if sig.confidence >= 0.75:
                strong_count += 1
            elif sig.confidence >= 0.50:
                medium_count += 1
            else:
                weak_count += 1

            buy_details_raw = getattr(sig, "buy_weight_details", []) or []
            sell_details_raw = getattr(sig, "sell_weight_details", []) or []
            buy_weight_details = [
                WeightContribution(
                    key=d.get("key", ""),
                    label=d.get("label", ""),
                    weight=d.get("weight", 0),
                    triggered=d.get("triggered", False),
                    score=d.get("score", 0),
                )
                for d in buy_details_raw
            ]
            sell_weight_details = [
                WeightContribution(
                    key=d.get("key", ""),
                    label=d.get("label", ""),
                    weight=d.get("weight", 0),
                    triggered=d.get("triggered", False),
                    score=d.get("score", 0),
                )
                for d in sell_details_raw
            ]

            result_signals.append(
                IntradaySignal(
                    stock_code=sig.stock_code,
                    signal_type=sig.signal_type,
                    trigger_time=str(sig.trigger_time),
                    price=round(sig.price, 2),
                    score=sig.score,
                    max_score=sig.max_score,
                    confidence=round(sig.confidence, 4),
                    position_advice=pos,
                    reasoning=sig.reasoning,
                    support_force=getattr(sig, "support_force", 0.0),
                    pressure_force=getattr(sig, "pressure_force", 0.0),
                    buy_weight_details=buy_weight_details,
                    sell_weight_details=sell_weight_details,
                )
            )

        summary = {
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'total_signals': buy_count + sell_count,
            'strong_signals': strong_count,
            'medium_signals': medium_count,
            'weak_signals': weak_count,
            'simulated_return_pct': round(total_return, 2),
        }

        return result_signals, summary, precomputed_result, strategy.engine

    except ImportError as e:
        logger.warning(f"导入做T策略失败: {e}，返回空信号")
        return [], {
            'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0,
            'strong_signals': 0, 'medium_signals': 0, 'weak_signals': 0,
            'simulated_return_pct': 0.0,
        }, None, None
    except Exception as e:
        logger.error(f"运行做T策略失败: {e}", exc_info=True)
        return [], {
            'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0,
            'strong_signals': 0, 'medium_signals': 0, 'weak_signals': 0,
            'simulated_return_pct': 0.0, 'error': str(e),
        }, None, None


def _cross_up(a_series, b_series, lookback: int = 3) -> bool:
    """检测 a 上穿 b（最近 lookback 根K线内发生过交叉）"""
    import pandas as pd

    if len(a_series) < 2 or len(b_series) < 2:
        return False
    start = max(0, len(a_series) - lookback)
    for i in range(start, len(a_series)):
        if i == 0:
            continue
        a_prev = a_series.iloc[i - 1]
        b_prev = b_series.iloc[i - 1]
        a_cur = a_series.iloc[i]
        b_cur = b_series.iloc[i]
        if pd.isna(a_prev) or pd.isna(b_prev) or pd.isna(a_cur) or pd.isna(b_cur):
            continue
        if a_prev <= b_prev and a_cur > b_cur:
            return True
    return False


def _cross_down(a_series, b_series, lookback: int = 3) -> bool:
    """检测 a 下穿 b（最近 lookback 根K线内发生过交叉）"""
    import pandas as pd

    if len(a_series) < 2 or len(b_series) < 2:
        return False
    start = max(0, len(a_series) - lookback)
    for i in range(start, len(a_series)):
        if i == 0:
            continue
        a_prev = a_series.iloc[i - 1]
        b_prev = b_series.iloc[i - 1]
        a_cur = a_series.iloc[i]
        b_cur = b_series.iloc[i]
        if pd.isna(a_prev) or pd.isna(b_prev) or pd.isna(a_cur) or pd.isna(b_cur):
            continue
        if a_prev >= b_prev and a_cur < b_cur:
            return True
    return False


def _detect_kdj_top_divergence(close_series, k_series, lookback: int = 10) -> bool:
    """检测KDJ顶背离：价格创新高但K值未创新高 → 卖出信号"""
    import pandas as pd

    n = min(len(close_series), len(k_series))
    if n < lookback:
        return False
    recent_close = close_series.iloc[-lookback:]
    recent_k = k_series.iloc[-lookback:]
    half = lookback // 2
    left_close = recent_close.iloc[:half].dropna()
    right_close = recent_close.iloc[half:].dropna()
    left_k = recent_k.iloc[:half].dropna()
    right_k = recent_k.iloc[half:].dropna()
    if len(left_close) == 0 or len(right_close) == 0 or len(left_k) == 0 or len(right_k) == 0:
        return False
    left_close_max = float(left_close.max())
    right_close_max = float(right_close.max())
    left_k_max = float(left_k.max())
    right_k_max = float(right_k.max())
    return right_close_max > left_close_max and right_k_max < left_k_max


def _detect_kdj_bottom_divergence(close_series, k_series, lookback: int = 10) -> bool:
    """检测KDJ底背离：价格创新低但K值未创新低 → 买入信号"""
    import pandas as pd

    n = min(len(close_series), len(k_series))
    if n < lookback:
        return False
    recent_close = close_series.iloc[-lookback:]
    recent_k = k_series.iloc[-lookback:]
    half = lookback // 2
    left_close = recent_close.iloc[:half].dropna()
    right_close = recent_close.iloc[half:].dropna()
    left_k = recent_k.iloc[:half].dropna()
    right_k = recent_k.iloc[half:].dropna()
    if len(left_close) == 0 or len(right_close) == 0 or len(left_k) == 0 or len(right_k) == 0:
        return False
    left_close_min = float(left_close.min())
    right_close_min = float(right_close.min())
    left_k_min = float(left_k.min())
    right_k_min = float(right_k.min())
    return right_close_min < left_close_min and right_k_min > left_k_min


def _compute_macd_signal(result) -> str:
    """计算MACD信号文本"""
    try:
        if 'DIF' not in result.columns or 'DEA' not in result.columns:
            return ''
        dif = result['DIF']
        dea = result['DEA']
        up = _cross_up(dif, dea)
        down = _cross_down(dif, dea)
        dif_vals = dif.dropna()
        dea_vals = dea.dropna()
        last_dif = float(dif_vals.iloc[-1]) if len(dif_vals) > 0 else 0
        last_dea = float(dea_vals.iloc[-1]) if len(dea_vals) > 0 else 0
        if up:
            return 'MACD金叉 ↑'
        elif down:
            return 'MACD死叉 ↓'
        elif last_dif > last_dea:
            return 'MACD多头 ↗'
        else:
            return 'MACD空头 ↘'
    except Exception as e:
        logger.warning(f'计算MACD信号失败: {e}')
        return ''


def _compute_rsi_signal(result, overbought: float = 70, oversold: float = 30) -> str:
    """计算RSI信号文本"""
    try:
        if 'RSI' not in result.columns:
            return ''
        rsi = result['RSI']
        rsi_vals = rsi.dropna()
        last_rsi = float(rsi_vals.iloc[-1]) if len(rsi_vals) > 0 else 50
        if last_rsi <= oversold:
            return 'RSI超卖 ↑'
        elif last_rsi >= overbought:
            return 'RSI超买 ↓'
        elif last_rsi < 50:
            return 'RSI偏弱 ↘'
        else:
            return 'RSI偏强 ↗'
    except Exception as e:
        logger.warning(f'计算RSI信号失败: {e}')
        return ''


def _compute_kdj_signal(
    result,
    overbought: float = 80,
    oversold: float = 20,
    golden_cross_max: float = 50,
    death_cross_min: float = 80,
    divergence_lookback: int = 10,
) -> str:
    """计算KDJ信号文本，包含超买超卖、金叉死叉、顶底背离"""
    try:
        if not all(c in result.columns for c in ['K', 'D', 'J']):
            return ''
        k_vals = result['K'].dropna()
        d_vals = result['D'].dropna()
        j_vals = result['J'].dropna()
        last_k = float(k_vals.iloc[-1]) if len(k_vals) > 0 else 50
        last_d = float(d_vals.iloc[-1]) if len(d_vals) > 0 else 50
        last_j = float(j_vals.iloc[-1]) if len(j_vals) > 0 else 50

        close_vals = None
        if 'Close' in result.columns:
            close_vals = result['Close']

        # 超买/超卖判断
        if last_k > overbought and last_d > overbought and last_j > overbought:
            return 'KDJ超买 \u2193'
        if last_k < oversold and last_d < oversold and last_j < oversold:
            return 'KDJ超卖 \u2191'

        # 金叉/死叉判断
        up = _cross_up(result['K'], result['D'])
        down = _cross_down(result['K'], result['D'])
        if up and last_k < golden_cross_max:
            return 'KDJ金叉(低位) \u2191'
        elif up:
            return 'KDJ金叉 \u2197'
        if down and last_k > death_cross_min:
            return 'KDJ死叉(高位) \u2193'
        elif down:
            return 'KDJ死叉 \u2198'

        # 顶背离/底背离判断
        if close_vals is not None and len(result) >= divergence_lookback:
            if _detect_kdj_top_divergence(close_vals, result['K'], divergence_lookback):
                return 'KDJ顶背离 \u2193'
            if _detect_kdj_bottom_divergence(close_vals, result['K'], divergence_lookback):
                return 'KDJ底背离 \u2191'

        if last_k > last_d:
            return 'KDJ多头 \u2197'
        else:
            return 'KDJ空头 \u2198'
    except Exception as e:
        logger.warning(f'计算KDJ信号失败: {e}')
        return ''


def _compute_mfi_signal(result, overbought: float = 80, oversold: float = 20) -> str:
    """计算MFI最新信号文本"""
    if result is None or result.empty:
        return ""
    last = result.iloc[-1]
    signals = []
    if last.get("mfi_overbought"):
        signals.append("超买")
    if last.get("mfi_oversold"):
        signals.append("超卖")
    if last.get("mfi_cross_50_up"):
        signals.append("上穿50线")
    if last.get("mfi_cross_50_down"):
        signals.append("下穿50线")
    if last.get("mfi_bottom_divergence"):
        signals.append("底背离")
    if last.get("mfi_top_divergence"):
        signals.append("顶背离")
    return "、".join(signals)


def _synthesize_5min_klines(klines_1min: list) -> list:
    """将1分钟K线合成为5分钟K线

    腾讯1分钟数据每点 Open=High=Low=Close=同一价格，合成为标准OHLC 5分钟K线。
    最后一个桶不足5根K线时仍然输出（用已有数据）。

    Args:
        klines_1min: 1分钟K线字典列表，每项含 Open/High/Low/Close/Volume/timestamp

    Returns:
        5分钟K线字典列表，每项含 Open/High/Low/Close/Volume/timestamp
    """
    if not klines_1min:
        return []

    from collections import defaultdict

    # 按5分钟桶分组
    buckets: dict[str, list] = defaultdict(list)
    for k in klines_1min:
        ts = k.get("timestamp", "")
        if not ts:
            continue
        ts_str = str(ts)
        if "T" in ts_str:
            date_part, time_part = ts_str.split("T", 1)
        elif " " in ts_str:
            date_part, time_part = ts_str.split(" ", 1)
        else:
            continue
        time_str = time_part.strip()[:5]  # "HH:MM"
        try:
            hour, minute = time_str.split(":")
            h, m = int(hour), int(minute)
        except (ValueError, IndexError):
            continue
        # 过滤 15:01 及之后的盘后数据，保留 15:00 整的K线桶（用于十字线定位到最新交易时间）
        if h > 15 or (h == 15 and m > 0):
            continue
        bucket_minute = (m // 5) * 5
        bucket_time = f"{date_part}T{h:02d}:{bucket_minute:02d}:00"
        buckets[bucket_time].append(k)

    # 聚合每个桶
    result = []
    for bucket_time in sorted(buckets.keys()):
        bucket_klines = buckets[bucket_time]
        prices = [float(k["Close"]) for k in bucket_klines]
        volumes = [float(k.get("Volume", 0)) for k in bucket_klines]
        result.append({
            "timestamp": bucket_time,
            "Open": prices[0],
            "High": max(prices),
            "Low": min(prices),
            "Close": prices[-1],
            "Volume": sum(volumes),
        })

    return result


def _filter_post_market_klines(klines: list) -> list:
    """过滤掉盘后交易数据（15:00-15:30），保留15:00整的K线。

    用于预热数据（前一交易日K线）的清洗，确保指标计算和图表展示
    不受盘后集合竞价数据的影响。
    """
    if not klines:
        return klines
    filtered = []
    for k in klines:
        ts = k.get("timestamp", "")
        if not ts:
            filtered.append(k)
            continue
        ts_str = str(ts)
        # 提取时间部分
        if "T" in ts_str:
            time_part = ts_str.split("T", 1)[1]
        elif " " in ts_str:
            time_part = ts_str.split(" ", 1)[1]
        else:
            filtered.append(k)
            continue
        time_str = time_part.strip()[:5]  # "HH:MM"
        try:
            hour, minute = time_str.split(":")
            h, m = int(hour), int(minute)
        except (ValueError, IndexError):
            filtered.append(k)
            continue
        # 过滤 15:01 及之后的盘后数据，保留 15:00 整
        if h > 15 or (h == 15 and m > 0):
            continue
        filtered.append(k)
    return filtered


def _compute_xma_truncated(series: 'np.ndarray', n: int) -> 'np.ndarray':
    """计算XMA，仅使用当前位置及之前的数据（右对齐截断），无未来数据泄露。

    与标准XMA（居中对称窗口）不同，此版本将窗口右边界截断到当前位置 i，
    模拟真实交易中只能看到历史数据的约束。

    Args:
        series: 输入序列
        n: XMA周期（默认25，窗口半宽h=12）

    Returns:
        与输入等长的截断XMA序列
    """
    import numpy as np
    length = len(series)
    h = n // 2
    result = np.full(length, np.nan)
    for i in range(length):
        left = max(0, i - h)
        right = i  # 仅使用当前及之前的数据
        result[i] = np.mean(series[left:right + 1])
    return result


def _evaluate_signals_incremental(full_df: 'pd.DataFrame', prev_len: int) -> list:
    """逐bar评估天道信号，每根bar仅使用当时可用的历史数据。

    模拟真实交易中信号按时间顺序产生的场景：
    - 对每根当日bar，基于prev_day + 当日bar[0..i] 的数据计算截断XMA
    - 判断收盘价与金钻线/金牛线的关系，产生买入/卖出信号
    - 无未来数据泄露

    Args:
        full_df: 包含前日+当日完整数据的DataFrame
        prev_len: 前日数据长度

    Returns:
        TiandaoSignal列表，按时间顺序排列
    """
    import pandas as pd
    import numpy as np

    n = 25
    signals = []

    high_all = full_df["High"].values.astype(np.float64)
    low_all = full_df["Low"].values.astype(np.float64)
    close_all = full_df["Close"].values.astype(np.float64)

    # 提取时间标签
    def _extract_ts(row):
        ts_str = str(row.get("timestamp", "")).strip()
        match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{1,2}):(\d{2})", ts_str)
        if match:
            return f"{match.group(1)}T{match.group(2).zfill(2)}:{match.group(3)}"
        return ""

    # 逐bar评估（仅当日数据）
    for i in range(prev_len, len(full_df)):
        # 使用prev_day + 当日bar[0..i]的数据
        end_idx = i + 1  # 切片不包含end，所以+1

        # 计算截断双重XMA（仅使用end_idx之前的数据）
        xma1_high = _compute_xma_truncated(high_all[:end_idx], n)
        xma1_low = _compute_xma_truncated(low_all[:end_idx], n)
        xma2_high = _compute_xma_truncated(xma1_high, n)
        xma2_low = _compute_xma_truncated(xma1_low, n)

        # 金牛线（上轨）和金钻趋势线（下轨）
        jinniu_val = 2 * xma2_high[-1] - xma2_low[-1]
        jinzuan_val = 2 * xma2_low[-1] - xma2_high[-1]
        close_val = close_all[i]

        if np.isnan(jinzuan_val) or np.isnan(jinniu_val) or np.isnan(close_val):
            continue

        tl = _extract_ts(full_df.iloc[i])
        close_f = float(close_val)
        jinzuan_f = float(jinzuan_val)
        jinniu_f = float(jinniu_val)

        if close_f < jinzuan_f:
            logger.info(
                f"[天道信号-逐bar] {tl} 买入: close={close_f:.4f}, jinzuan={jinzuan_f:.4f}, "
                f"jinniu={jinniu_f:.4f}, 差值={jinzuan_f - close_f:.4f}"
            )
            signals.append(
                TiandaoSignal(
                    signal_type="buy",
                    trigger_time=tl,
                    price=round(close_f, 2),
                    reason="价格跌破金钻趋势线",
                )
            )
        elif close_f > jinniu_f:
            logger.info(
                f"[天道信号-逐bar] {tl} 卖出: close={close_f:.4f}, jinniu={jinniu_f:.4f}, "
                f"jinzuan={jinzuan_f:.4f}, 差值={close_f - jinniu_f:.4f}"
            )
            signals.append(
                TiandaoSignal(
                    signal_type="sell",
                    trigger_time=tl,
                    price=round(close_f, 2),
                    reason="价格突破金牛线",
                )
            )
        else:
            # 在区间内，无信号，记录调试信息（仅在13:00-14:00时段输出，避免日志过多）
            if tl and "T13:" in tl:
                logger.info(
                    f"[天道信号-逐bar] {tl} 无信号: close={close_f:.4f}, jinzuan={jinzuan_f:.4f}, "
                    f"jinniu={jinniu_f:.4f}, 区间内"
                )

    return signals


def _compute_tiandao_5min(klines_5min: list, prev_day_klines: list = None,
                         cache_key: str = None) -> dict:
    """对5分钟K线计算天道指标并生成信号

    【重要】整个天道子图的指标线和信号均使用截断XMA（右对齐），不使用未来数据：
    - 指标线（金牛/金钻）：每个位置 i 的 XMA 窗口为 [i-12, i]，不包含 i 之后的未来数据
    - 首次加载信号：逐bar评估，每根bar仅使用当时可用的历史数据
    - 后续轮询信号：仅评估最新bar，历史信号从缓存累积

    信号生成策略：
    - 首次加载（缓存为空）：逐bar评估，模拟真实交易中信号按时间顺序产生
    - 后续轮询（缓存非空）：仅评估最新一根K线，历史信号从缓存获取

    Args:
        klines_5min: 当前日5分钟K线列表
        prev_day_klines: 前一日5分钟K线列表（用于指标预热和显示）
        cache_key: 信号缓存键（格式: "code_date"），用于累积历史信号

    Returns:
        {
            "jinzuan_line": [IndicatorLinePoint],   # 金钻趋势线（截断XMA，无未来数据）
            "jinniu_line": [IndicatorLinePoint],    # 金牛线（截断XMA，无未来数据）
            "signals": [TiandaoSignal],             # 当前日天道信号（从缓存累积）
        }
    """
    import pandas as pd
    import numpy as np

    if not klines_5min:
        return {"jinzuan_line": [], "jinniu_line": [], "signals": []}

    # 构造当日DataFrame
    today_df = pd.DataFrame(klines_5min)
    for col in ["Open", "High", "Low", "Close"]:
        if col in today_df.columns:
            today_df[col] = today_df[col].astype(float)

    # 预热：拼接前一日数据
    if prev_day_klines:
        prev_df = pd.DataFrame(prev_day_klines)
        for col in ["Open", "High", "Low", "Close"]:
            if col in prev_df.columns:
                prev_df[col] = prev_df[col].astype(float)
        full_df = pd.concat([prev_df, today_df], ignore_index=True)
        prev_len = len(prev_df)
    else:
        full_df = today_df.copy()
        prev_len = 0

    # 计算天道指标
    try:
        from indicators.indicators.tiandao import Tiandao
        tiandao = Tiandao(n=25)
        result = tiandao.calculate(full_df)
    except Exception as e:
        logger.warning(f"天道指标计算失败: {e}")
        return {"jinzuan_line": [], "jinniu_line": [], "signals": []}

    today_result = result.iloc[prev_len:]

    # 提取时间标签（前日+当日完整数据，用于画线）
    def _extract_datetime_label(ts_val):
        """从时间戳提取 YYYY-MM-DDTHH:MM 格式"""
        ts_str = str(ts_val).strip()
        match = re.search(r"(\d{4}-\d{2}-\d{2})[T\s](\d{1,2}):(\d{2})", ts_str)
        if match:
            return f"{match.group(1)}T{match.group(2).zfill(2)}:{match.group(3)}"
        return ""

    # 完整数据的时间标签（前日+当日），含日期
    full_time_labels = [_extract_datetime_label(row.get("timestamp", "")) for _, row in full_df.iterrows()]

    # 计算截断天道指标线（每个位置仅使用历史数据，无未来数据泄露）
    # 与标准XMA（居中对称窗口）不同，截断XMA将窗口右边界限制在当前位置
    high_all = full_df["High"].values.astype(np.float64)
    low_all = full_df["Low"].values.astype(np.float64)

    xma1_high = _compute_xma_truncated(high_all, 25)
    xma1_low = _compute_xma_truncated(low_all, 25)
    xma2_high = _compute_xma_truncated(xma1_high, 25)
    xma2_low = _compute_xma_truncated(xma1_low, 25)

    jinniu_truncated = 2 * xma2_high - xma2_low
    jinzuan_truncated = 2 * xma2_low - xma2_high

    # 金钻趋势线（截断，无未来数据）
    jinzuan_line = []
    for i in range(len(full_df)):
        v = jinzuan_truncated[i]
        if not np.isnan(v):
            jinzuan_line.append(
                IndicatorLinePoint(
                    time=full_time_labels[i] if i < len(full_time_labels) else "",
                    value=round(float(v), 4),
                )
            )

    # 金牛线（截断，无未来数据）
    jinniu_line = []
    for i in range(len(full_df)):
        v = jinniu_truncated[i]
        if not np.isnan(v):
            jinniu_line.append(
                IndicatorLinePoint(
                    time=full_time_labels[i] if i < len(full_time_labels) else "",
                    value=round(float(v), 4),
                )
            )

    # 当日截断值（用于信号评估）
    today_jinzuan_truncated = jinzuan_truncated[prev_len:]
    today_jinniu_truncated = jinniu_truncated[prev_len:]

    # 调试：输出13:00-14:00时段的指标值与收盘价对比
    # for i in range(prev_len, len(full_df)):
    #     tl = full_time_labels[i] if i < len(full_time_labels) else ""
    #     if tl and "T13:" in tl:
    #         close_v = float(full_df["Close"].iloc[i])
    #         jinzuan_v = float(jinzuan_truncated[i]) if not np.isnan(jinzuan_truncated[i]) else float('nan')
    #         jinniu_v = float(jinniu_truncated[i]) if not np.isnan(jinniu_truncated[i]) else float('nan')
    #         logger.info(
    #             f"[天道指标-显示] {tl} close={close_v:.4f}, jinzuan={jinzuan_v:.4f}, "
    #             f"jinniu={jinniu_v:.4f}, "
    #             f"close>jinniu={close_v > jinniu_v}, close<jinzuan={close_v < jinzuan_v}"
    #         )

    # 生成天道信号
    # - 首次加载（缓存为空）：逐bar评估，使用右对齐截断XMA，模拟真实信号产生顺序
    # - 后续轮询（缓存非空）：仅评估最新bar，历史信号从缓存获取
    signals: list = []
    is_first_load = cache_key and cache_key not in _tiandao_5min_signals_cache

    if is_first_load:
        # 首次加载：逐bar评估所有历史信号（无未来数据泄露）
        signals = _evaluate_signals_incremental(full_df, prev_len)
        if cache_key:
            _tiandao_5min_signals_cache[cache_key] = signals
            logger.info(
                f"[天道信号-首次] 共评估{len(today_result)}根K线, 产生{len(signals)}个信号"
            )
    else:
        # 后续轮询：从缓存加载历史信号，仅评估最新bar（使用截断值）
        if cache_key and cache_key in _tiandao_5min_signals_cache:
            signals = list(_tiandao_5min_signals_cache[cache_key])

        last_i = len(today_result) - 1
        if last_i >= 0 and last_i < len(today_jinzuan_truncated):
            close_val = today_result["Close"].iloc[last_i] if "Close" in today_result.columns else None
            jinzuan_val = today_jinzuan_truncated[last_i]
            jinniu_val = today_jinniu_truncated[last_i]
            if not (np.isnan(jinzuan_val) or np.isnan(jinniu_val) or close_val is None or pd.isna(close_val)):
                tl = _extract_datetime_label(today_result.iloc[last_i].get("timestamp", ""))
                close_f = float(close_val)
                jinzuan_f = float(jinzuan_val)
                jinniu_f = float(jinniu_val)

                # 检查是否已有相同时刻的信号，避免重复
                existing_times = set()
                for s in signals:
                    if hasattr(s, 'trigger_time'):
                        existing_times.add(s.trigger_time)

                if tl not in existing_times:
                    if close_f < jinzuan_f:
                        new_signal = TiandaoSignal(
                            signal_type="buy",
                            trigger_time=tl,
                            price=round(close_f, 2),
                            reason="价格跌破金钻趋势线",
                        )
                        signals.append(new_signal)
                        logger.info(
                            f"[天道信号] {tl} 买入: close={close_f:.2f} < jinzuan={jinzuan_f:.2f}, "
                            f"jinniu={jinniu_f:.2f}, 差值={jinzuan_f - close_f:.2f}"
                        )
                    elif close_f > jinniu_f:
                        new_signal = TiandaoSignal(
                            signal_type="sell",
                            trigger_time=tl,
                            price=round(close_f, 2),
                            reason="价格突破金牛线",
                        )
                        signals.append(new_signal)
                        logger.info(
                            f"[天道信号] {tl} 卖出: close={close_f:.2f} > jinniu={jinniu_f:.2f}, "
                            f"jinzuan={jinzuan_f:.2f}, 差值={close_f - jinniu_f:.2f}"
                        )

        # 更新缓存
        if cache_key:
            _tiandao_5min_signals_cache[cache_key] = signals

    return {
        "jinzuan_line": jinzuan_line,
        "jinniu_line": jinniu_line,
        "signals": signals,
    }


def _generate_indicator_sub_charts(klines: list, warmup_klines: list = None,
                                   precomputed_result: 'pd.DataFrame' = None,
                                   precomputed_engine: 'IntradayIndicatorEngine' = None,
                                   ma5_price: float = None) -> list:
    """根据分时K线计算四大指标，生成子图数据

    Args:
        klines: 分时K线字典列表
        warmup_klines: 前一交易日最后N根K线，用于指标预热（可选）
        precomputed_result: 预计算的指标DataFrame（可选，用于避免重复计算）
        precomputed_engine: 预创建的IntradayIndicatorEngine（可选，与precomputed_result配套使用）

    Returns:
        List[IndicatorSubChart] 四个指标的子图数据
    """
    try:
        import pandas as pd

        if precomputed_result is not None and precomputed_engine is not None:
            result = precomputed_result
            engine = precomputed_engine
            warmup_rows = len(warmup_klines) if warmup_klines else 0
        else:
            df = pd.DataFrame(klines)
            if 'Amount' not in df.columns:
                df['Amount'] = df['Close'] * df['Volume'] if 'Close' in df.columns and 'Volume' in df.columns else 0.0

            warmup_rows = 0
            if warmup_klines:
                warmup_df = pd.DataFrame(warmup_klines)
                if 'Amount' not in warmup_df.columns:
                    warmup_df['Amount'] = warmup_df['Close'] * warmup_df['Volume'] if 'Close' in warmup_df.columns and 'Volume' in warmup_df.columns else 0.0
                warmup_rows = len(warmup_df)
                df = pd.concat([warmup_df, df], ignore_index=True)

            engine = IntradayIndicatorEngine(config=_load_indicator_config())
            result = engine.calculate_all(df)

        # 预热行数后的数据才是当天数据，生成时间标签时跳过预热行
        today_result = result.iloc[warmup_rows:]

        # 提取时间标签 "HH:MM"，尽量对齐K线的timestamp格式
        time_labels = []
        for _, row in today_result.iterrows():
            ts = row.get('timestamp', '')
            if isinstance(ts, str):
                ts = ts.strip()
            elif hasattr(ts, 'isoformat'):
                ts = ts.isoformat()
            else:
                ts = str(ts)
            # 从时间字符串中提取 HH:MM
            import re
            match = re.search(r'(\d{1,2}):(\d{2})', ts)
            if match:
                time_labels.append(f"{match.group(1).zfill(2)}:{match.group(2)}")
            else:
                time_labels.append("")

        sub_charts = []

        macd_signal = _compute_macd_signal(result)
        rsi_signal = _compute_rsi_signal(result, overbought=engine.rsi_overbought, oversold=engine.rsi_oversold)
        kdj_signal = _compute_kdj_signal(
            result,
            overbought=engine.kdj_overbought,
            oversold=engine.kdj_oversold,
            golden_cross_max=engine.kdj_golden_cross_max,
            death_cross_min=engine.kdj_death_cross_min,
            divergence_lookback=engine.kdj_divergence_lookback,
        )

        # ── 1. 主力吸筹 ──
        if 'absorption' in today_result.columns:
            absorption_data = []
            for i, v in enumerate(today_result['absorption']):
                if not pd.isna(v):
                    absorption_data.append(IndicatorLinePoint(time=time_labels[i] if i < len(time_labels) else '', value=round(float(v), 4)))
            sub_charts.append(
                IndicatorSubChart(
                    id="absorption",
                    label="主力吸筹",
                    height=110,
                    lines=[
                        IndicatorLine(
                            name="absorption",
                            label="吸筹",
                            color="#AA44FF",
                            data=absorption_data,
                        ),
                        IndicatorLine(
                            name="distribution_label",
                            label="出货",
                            color="#44AA44",
                            data=[],
                        ),
                    ],
                    signal_text="",
                )
            )

        # ── 2. MACD ──
        macd_dif_data = []
        macd_dea_data = []
        macd_bar_data = []
        macd_metadata = None
        if all(c in today_result.columns for c in ['DIF', 'DEA', 'MACD_Bar']):
            # 从完整结果（含预热数据）提取MACD元数据，确保与后端策略算法一致
            if all(c in result.columns for c in ['MACD_Bar_Sum', 'MACD_Bar_Diff']):
                last_row = result.iloc[-1]
                bar_sum = float(last_row.get('MACD_Bar_Sum', 0)) if pd.notna(last_row.get('MACD_Bar_Sum')) else 0
                bar_diff = float(last_row.get('MACD_Bar_Diff', 0)) if pd.notna(last_row.get('MACD_Bar_Diff')) else 0
                bar_sums = []
                bar_diffs = []
                for i in range(len(today_result)):
                    bs = today_result['MACD_Bar_Sum'].iloc[i]
                    bd = today_result['MACD_Bar_Diff'].iloc[i]
                    bar_sums.append(round(float(bs), 2) if pd.notna(bs) else 0)
                    bar_diffs.append(round(float(bd), 2) if pd.notna(bd) else 0)
                macd_metadata = {
                    "bar_sum": round(bar_sum, 2),
                    "bar_diff": round(bar_diff, 2),
                    "bar_sums": bar_sums,
                    "bar_diffs": bar_diffs,
                }
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(today_result['DIF'].iloc[i]):
                    macd_dif_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['DIF'].iloc[i]), 4)))
                if not pd.isna(today_result['DEA'].iloc[i]):
                    macd_dea_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['DEA'].iloc[i]), 4)))
                if not pd.isna(today_result['MACD_Bar'].iloc[i]):
                    macd_bar_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['MACD_Bar'].iloc[i]), 4)))
            sub_charts.append(
                IndicatorSubChart(
                    id="macd",
                    label="MACD",
                    height=110,
                    lines=[
                        IndicatorLine(name="DIF", label="DIF", color="#FFFFFF", data=macd_dif_data),
                        IndicatorLine(name="DEA", label="DEA", color="#FFD700", data=macd_dea_data),
                        IndicatorLine(name="MACD_Bar", label="MACD柱", color="#FF4444", data=macd_bar_data),
                    ],
                    signal_text=macd_signal,
                    metadata=macd_metadata,
                )
            )

        

        # ── 3. KDJ ──
        kdj_k_data = []
        kdj_d_data = []
        kdj_j_data = []
        if all(c in today_result.columns for c in ['K', 'D', 'J']):
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(today_result['K'].iloc[i]):
                    kdj_k_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['K'].iloc[i]), 2)))
                if not pd.isna(today_result['D'].iloc[i]):
                    kdj_d_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['D'].iloc[i]), 2)))
                if not pd.isna(today_result['J'].iloc[i]):
                    kdj_j_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['J'].iloc[i]), 2)))
            sub_charts.append(
                IndicatorSubChart(
                    id="kdj",
                    label="KDJ",
                    height=110,
                    lines=[
                        IndicatorLine(name="K", label="K", color="#FFFF00", data=kdj_k_data),
                        IndicatorLine(name="D", label="D", color="#4488FF", data=kdj_d_data),
                        IndicatorLine(name="J", label="J", color="#AA44FF", data=kdj_j_data),
                    ],
                    signal_text=kdj_signal,
                )
            )

        # ── 4. RSI ──
        rsi_data = []
        if 'RSI' in today_result.columns:
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(today_result['RSI'].iloc[i]):
                    rsi_data.append(IndicatorLinePoint(time=tl, value=round(float(today_result['RSI'].iloc[i]), 2)))
            rsi_ob_data = []
            rsi_os_data = []
            rsi_ob_val = engine.rsi_overbought
            rsi_os_val = engine.rsi_oversold
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                rsi_ob_data.append(IndicatorLinePoint(time=tl, value=rsi_ob_val))
                rsi_os_data.append(IndicatorLinePoint(time=tl, value=rsi_os_val))
            sub_charts.append(
                IndicatorSubChart(
                    id="rsi",
                    label="RSI",
                    height=110,
                    lines=[
                        IndicatorLine(name="RSI", label="RSI", color="#4488FF", data=rsi_data),
                        IndicatorLine(name="rsi_overbought", label=f"超买({rsi_ob_val})", color="#FF4444", data=rsi_ob_data),
                        IndicatorLine(name="rsi_oversold", label=f"超卖({rsi_os_val})", color="#44FF44", data=rsi_os_data),
                    ],
                    signal_text=rsi_signal,
                )
            )

        

        # ── 均价偏离度（基于当日 VWAP，不含预热数据） ──
        deviation_pct_data = []
        if "Close" in today_result.columns and "Volume" in today_result.columns:
            close = today_result["Close"]
            volume = today_result["Volume"].fillna(0)
            cum_amount = (close * volume).cumsum()
            cum_vol = volume.cumsum().replace(0, float("nan"))
            today_avg_price = cum_amount / cum_vol
            today_avg_price = today_avg_price.fillna(close)
            today_deviation = (close - today_avg_price) / today_avg_price * 100

            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ""
                if not pd.isna(today_deviation.iloc[i]):
                    deviation_pct_data.append(
                        IndicatorLinePoint(time=tl, value=round(float(today_deviation.iloc[i]), 2))
                    )
            sub_charts.append(
                IndicatorSubChart(
                    id="avg_price_deviation",
                    label="均价偏离",
                    height=110,
                    lines=[
                        IndicatorLine(
                            name="deviation_pct",
                            label="均价偏离度",
                            color="#d1d4dc",
                            data=deviation_pct_data,
                        ),
                    ],
                    signal_text="",
                )
            )

        # ── MA5乖离率（基于日线级MA5）──
        if ma5_price is not None and ma5_price > 0 and "Close" in today_result.columns:
            ma5_dev_data = []
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ""
                close_val = today_result["Close"].iloc[i]
                if not pd.isna(close_val):
                    dev = (close_val - ma5_price) / ma5_price * 100
                    ma5_dev_data.append(
                        IndicatorLinePoint(time=tl, value=round(float(dev), 2))
                    )
            sub_charts.append(
                IndicatorSubChart(
                    id="ma5_deviation",
                    label="MA5乖离率",
                    height=110,
                    lines=[
                        IndicatorLine(
                            name="ma5_dev_pct",
                            label="MA5乖离率",
                            color="#FFA500",
                            data=ma5_dev_data,
                        ),
                    ],
                    signal_text="",
                )
            )

        logger.debug(f"生成 {len(sub_charts)} 个指标子图")

        # ── 5. MFI 资金流量 ──
        mfi_data = []
        if "mfi_value" in today_result.columns:
            for i in range(len(today_result)):
                tl = time_labels[i] if i < len(time_labels) else ""
                if not pd.isna(today_result["mfi_value"].iloc[i]):
                    mfi_data.append(
                        IndicatorLinePoint(
                            time=tl,
                            value=round(float(today_result["mfi_value"].iloc[i]), 2),
                        )
                    )
            ob_pts = []
            os_pts = []
            if mfi_data:
                first_t = mfi_data[0].time
                last_t = mfi_data[-1].time
                ob_val = engine.mfi_overbought
                os_val = engine.mfi_oversold
                ob_pts = [
                    IndicatorLinePoint(time=first_t, value=ob_val),
                    IndicatorLinePoint(time=last_t, value=ob_val),
                ]
                os_pts = [
                    IndicatorLinePoint(time=first_t, value=os_val),
                    IndicatorLinePoint(time=last_t, value=os_val),
                ]
            mfi_signal = _compute_mfi_signal(
                result,
                overbought=engine.mfi_overbought,
                oversold=engine.mfi_oversold,
            )
            sub_charts.append(
                IndicatorSubChart(
                    id="mfi",
                    label="MFI",
                    height=120,
                    lines=[
                        IndicatorLine(
                            name="mfi_value",
                            label="MFI",
                            color="#FF8C00",
                            data=mfi_data,
                        ),
                        IndicatorLine(
                            name="mfi_ob",
                            label=f"超买{engine.mfi_overbought}",
                            color="#FF4444",
                            data=ob_pts,
                        ),
                        IndicatorLine(
                            name="mfi_os",
                            label=f"超卖{engine.mfi_oversold}",
                            color="#44FF44",
                            data=os_pts,
                        ),
                    ],
                    signal_text=mfi_signal,
                )
            )

        return sub_charts

    except ImportError as e:
        logger.warning(f"导入指标计算依赖失败: {e}，返回空列表")
        return []
    except Exception as e:
        logger.error(f"生成指标子图失败: {e}", exc_info=True)
        return []


def _ensure_turnover_rate(daily_df, code, db_manager, end_date):
    """
    确保DataFrame有有效的换手率数据

    如果换手率全为0，则依次尝试：
    1. 从StockBasic表获取流通股本
    2. 从AKShare API获取带换手率的日线数据来估算流通股本
    然后用 成交量 / 流通股本 计算换手率并回写到数据库
    """
    if 'turnover_rate' not in daily_df.columns:
        return daily_df, False

    turnover_vals = daily_df['turnover_rate']
    non_zero = turnover_vals[turnover_vals.notna() & (turnover_vals > 0)]
    if len(non_zero) > 0:
        return daily_df, False

    logger.debug(f"换手率全为0，尝试为 {code} 补充换手率数据...")

    from datetime import timedelta

    circulating_shares = None

    # ── Step 1: 从 StockBasic 获取流通股本 ──
    try:
        from src.services.turnover_service import TurnoverService
        turnover_service = TurnoverService(db_manager)
        stock_basic = turnover_service.get_stock_basic(code, end_date)
        if stock_basic and stock_basic.circulating_shares and stock_basic.circulating_shares > 0:
            circulating_shares = stock_basic.circulating_shares
            logger.debug(f"从StockBasic获取 {code} 流通股本: {circulating_shares:.0f} 股")
    except Exception as e:
        logger.warning(f"从StockBasic获取 {code} 流通股本失败: {e}")

    # ── Step 2: DB中没有，从AKShare获取 ──
    if circulating_shares is None or circulating_shares <= 0:
        try:
            import akshare as ak
            akshare_df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=(end_date - timedelta(days=120)).strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'), adjust="qfq",
            )
            if akshare_df is not None and not akshare_df.empty:
                akshare_df = akshare_df.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '换手率': 'turnover_rate', '成交额': 'amount',
                })
                basic_info = turnover_service.estimate_circulating_shares_from_akshare(
                    akshare_df, code
                )
                if basic_info and basic_info.get('circulating_shares', 0) > 0:
                    turnover_service.save_stock_basic(basic_info)
                    circulating_shares = basic_info['circulating_shares']
                    logger.info(
                        f"从AKShare估算并保存 {code} 流通股本: {circulating_shares:.0f} 股"
                    )
        except Exception as e:
            logger.warning(f"从AKShare获取 {code} 换手率数据失败: {e}")

    if circulating_shares is None or circulating_shares <= 0:
        logger.warning(f"无法获取 {code} 的流通股本，跳过换手率补充")
        return daily_df, False

    # ── 计算换手率 ──
    daily_df = daily_df.copy()
    daily_df['turnover_rate'] = daily_df['Volume'] / circulating_shares
    daily_df['turnover_rate'] = daily_df['turnover_rate'].clip(upper=1.0)

    # ── 回写到数据库 ──
    try:
        from datetime import datetime as dt
        with db_manager.get_session() as session:
            from src.storage import StockDaily
            from sqlalchemy import and_
            count = 0
            for idx, row in daily_df.iterrows():
                day = idx.date() if hasattr(idx, 'date') else idx
                record = session.query(StockDaily).filter(
                    StockDaily.code == code,
                    StockDaily.date == day,
                ).first()
                if record:
                    record.turnover_rate = float(min(row['turnover_rate'], 1.0))
                    record.updated_at = dt.now()
                    count += 1
            session.commit()
        logger.debug(f"回写 {code} 换手率到数据库: {count} 条")
    except Exception as e:
        logger.warning(f"回写 {code} 换手率到数据库失败: {e}")

    return daily_df, True


def _try_load_index_cache_and_generate_ref_lines(
    ref_lines: list, code: str, klines: list, q_date, today_open: float, today_high: float, today_low: float
):
    """从大盘指数缓存读取日线数据，生成参考线

    当数据库无日线数据时（如 sh000001/sz399001），尝试从 data/cache/ 下的
    market_*.pkl 缓存加载日线数据，用于生成参考线。
    """
    import re as _re
    import pandas as pd

    # 仅处理指数代码
    is_index = bool(_re.match(r'^[Ss][Hh]\d{6}$', code) or _re.match(r'^[Ss][Zz]\d{6}$', code))
    if not is_index:
        logger.debug(f"非指数代码，跳过缓存读取: {code}")
        return

    symbol = code.lower()
    try:
        from stock_selector.market_data_cache import MarketDataCache

        cached_df = MarketDataCache.load(symbol)
        if cached_df is None or cached_df.empty:
            logger.debug(f"大盘指数缓存无数据: {symbol}")
            return

        logger.debug(f"从大盘指数缓存加载 {symbol} 日线数据，共 {len(cached_df)} 条")

        # 转换为 ReferenceLineGenerator 需要的格式
        df_rows = []
        for _, row in cached_df.iterrows():
            df_rows.append({
                'Date': row['date'],
                'Open': float(row['open']),
                'High': float(row['high']),
                'Low': float(row['low']),
                'Close': float(row['close']),
                'Volume': float(row['volume']),
            })

        daily_df = pd.DataFrame(df_rows)
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date').set_index('Date')

        # 将当日分时数据转换为日线OHLC，追加到历史DataFrame
        today_close = klines[-1]['Close']
        today_vol = sum(k.get('Volume', 0) or 0 for k in klines)
        today_row = pd.DataFrame({
            'Open':  [today_open],
            'High':  [today_high],
            'Low':   [today_low],
            'Close': [today_close],
            'Volume': [today_vol],
        }, index=[pd.Timestamp(q_date)])
        daily_df = pd.concat([daily_df, today_row])
        daily_df = daily_df.sort_index()

        from watchdog.strategies.reference_line_generator import ReferenceLineGenerator
        from api.v1.schemas.intraday import ReferenceLine

        gen = ReferenceLineGenerator(daily_df)
        daily_refs = gen.generate_all()

        id_map = {
            'attack_line': 'attack_line',
            'trading_line': 'operation_line',
            'defense_line': 'defense_line',
            'ma_5': 'ma5',
            'ma_10': 'ma10',
            'ma_20': 'ma20',
            'prev_close': 'prev_close',
        }
        skip_ids = {'previous_high_30', 'previous_low_30', 'chip_dense_upper', 'chip_dense_lower'}
        for ref_dict in daily_refs:
            rid = ref_dict['id']
            if rid in skip_ids:
                continue
            mapped_id = id_map.get(rid, rid)
            ref_lines.append(ReferenceLine(
                id=mapped_id,
                label=ref_dict['label'],
                price=ref_dict['price'],
                category=ref_dict['category'],
                color=ref_dict['color'],
                style=ref_dict['style'],
                base_weight=ref_dict['base_weight'],
            ))

        # ── 前高/前低 30个自然日HHV/LLV ──
        _add_30day_extreme_lines(ref_lines, daily_df, q_date, today_low, today_high)

        logger.debug(f"从大盘指数缓存生成 {len(daily_refs)} 条日线级参考线: {symbol}")

    except Exception as e:
        logger.warning(f"从大盘指数缓存生成参考线失败: {symbol}, {e}", exc_info=True)


def _compute_reference_lines(klines: list, code: str, db_manager=None, query_date: str = None) -> list:
    """计算支撑/压力参考线

    日内参考线（基于分时数据）：
    - 今开 / 今日高 / 今日低 (蓝/红/绿 dotted)

    日线级别参考线（需要数据库，与 ReferenceLineGenerator 保持一致）：
    - 昨收 (黄 solid)
    - 主力操盘三线：攻击线/操盘线/防守线 (红/橙/绿 dashed)
    - MA5/MA10/MA20 (红/橙/绿 dotted)
    - 前高/前低 30个自然日HHV/LLV (红/绿 solid)
    - 筹码密集区 上下沿 (紫 dashed)
    """
    from api.v1.schemas.intraday import ReferenceLine

    if not klines or len(klines) < 2:
        return []

    try:
        ref_lines = []
        highs = [k['High'] for k in klines]
        lows = [k['Low'] for k in klines]

        today_open = klines[0]['Open']
        today_high = max(highs)
        today_low = min(lows)

        # ── 今开/今高/今低 已去除──

        # ════════════════════════════════════════
        # 日线级别参考线（需要数据库）
        # ════════════════════════════════════════
        if db_manager is not None and code:
            try:
                from datetime import date, timedelta
                import math
                import pandas as pd

                # 解析查询日期，排除查询日当天
                from datetime import datetime as dt
                try:
                    if query_date and len(query_date) == 8:
                        q_date = dt.strptime(query_date, '%Y%m%d').date()
                    elif query_date:
                        q_date = dt.strptime(query_date.replace('-', '')[:8], '%Y%m%d').date()
                    else:
                        q_date = date.today()
                except (ValueError, TypeError):
                    q_date = date.today()
                end_date = q_date - timedelta(days=1)
                # 取 365 自然日历史数据，确保筹码分布和各指标计算有足够数据
                start_date = end_date - timedelta(days=365)
                daily_data = db_manager.get_data_range(code, start_date, end_date)
                daily_list = [d.to_dict() for d in daily_data] if daily_data else []

                if daily_list and len(daily_list) >= 5:
                    # 构建 DataFrame（按日期排序）
                    df_rows = []
                    for d in daily_list:
                        if d.get('close') is not None and not (
                            isinstance(d.get('close'), float) and math.isnan(d['close'])
                        ):
                            df_rows.append({
                                'Date': d['date'],
                                'Open': float(d['open'] or 0),
                                'High': float(d['high'] or 0),
                                'Low': float(d['low'] or 0),
                                'Close': float(d['close'] or 0),
                                'Volume': float(d.get('volume', 0) or 0),
                                'turnover_rate': float(d.get('turnover_rate', 0) or 0),
                            })
                    if not df_rows:
                        logger.debug("日线数据库无有效数据，跳过日线级参考线")
                    else:
                        daily_df = pd.DataFrame(df_rows)
                        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
                        daily_df = daily_df.sort_values('Date').set_index('Date')

                        # ── 换手率补充：全为0时从流通股本计算 ──
                        daily_df, turnover_filled = _ensure_turnover_rate(
                            daily_df, code, db_manager, end_date
                        )

                        # ── 筹码密集区缓存读写（仅使用历史数据，不含当日分时）──
                        # 如果换手率刚被修复，跳过缓存（旧缓存基于坏数据）
                        cache = None
                        if not turnover_filled:
                            cache = db_manager.load_chip_distribution_cache(code, end_date)
                        if cache is not None:
                            chip_upper_val = cache["chip_upper"]
                            chip_lower_val = cache["chip_lower"]
                            logger.debug(
                                f"命中筹码分布缓存: {code} end_date={end_date}, "
                                f"upper={chip_upper_val}, lower={chip_lower_val}"
                            )
                        else:
                            if turnover_filled:
                                logger.debug(f"换手率已补充，重新计算筹码分布: {code}")
                            try:
                                from indicators.indicators.chip_distribution import ChipDistribution
                                import numpy as np

                                chip_calc = ChipDistribution(enable_smooth=True, max_days=120)
                                chip_result = chip_calc.calculate(daily_df)

                                if (chip_result["max_chip_price"] is not None
                                        and chip_result["current_price"] is not None):
                                    chip_vols = np.array(chip_result["chip_volumes"])
                                    price_bins = np.array(chip_result["price_bins"])
                                    max_density = float(chip_vols.max())
                                    if max_density > 0:
                                        peak_idx = int(np.argmax(chip_vols))
                                        threshold = max_density * 0.5
                                        lower_idx = peak_idx
                                        while lower_idx > 0 and float(chip_vols[lower_idx]) >= threshold:
                                            lower_idx -= 1
                                        upper_idx = peak_idx
                                        while upper_idx < len(chip_vols) - 1 and float(chip_vols[upper_idx]) >= threshold:
                                            upper_idx += 1
                                        chip_upper_val = round(float(price_bins[upper_idx]), 2)
                                        chip_lower_val = round(float(price_bins[lower_idx]), 2)
                                        db_manager.save_chip_distribution_cache(
                                            code, end_date, chip_upper_val, chip_lower_val
                                        )
                                        logger.debug(
                                            f"计算并缓存筹码分布: {code} end_date={end_date}, "
                                            f"upper={chip_upper_val}, lower={chip_lower_val}"
                                        )
                                    else:
                                        chip_upper_val, chip_lower_val = None, None
                                else:
                                    chip_upper_val, chip_lower_val = None, None
                            except Exception as chip_err:
                                logger.warning(f"计算筹码分布失败: {chip_err}")
                                chip_upper_val, chip_lower_val = None, None

                        if chip_upper_val is not None and chip_lower_val is not None:
                            ref_lines.append(ReferenceLine(
                                id='chip_upper', label='筹码密集区上沿',
                                price=chip_upper_val, category='chip_dense',
                                color='#AA44FF', style='dashed', base_weight=1.5,
                            ))
                            ref_lines.append(ReferenceLine(
                                id='chip_lower', label='筹码密集区下沿',
                                price=chip_lower_val, category='chip_dense',
                                color='#AA44FF', style='dashed', base_weight=1.5,
                            ))

                        # 将当日分时数据转换为日线OHLC，追加到历史DataFrame
                        today_close = klines[-1]['Close']
                        today_vol = sum(k.get('Volume', 0) or 0 for k in klines)
                        today_row = pd.DataFrame({
                            'Open':  [today_open],
                            'High':  [today_high],
                            'Low':   [today_low],
                            'Close': [today_close],
                            'Volume': [today_vol],
                        }, index=[pd.Timestamp(q_date)])
                        daily_df = pd.concat([daily_df, today_row])
                        daily_df = daily_df.sort_index()
                        logger.debug(
                            f"追加当日OHLC: O={today_open} H={today_high} L={today_low} "
                            f"C={today_close} V={today_vol}, 总计{len(daily_df)}行"
                        )

                        from watchdog.strategies.reference_line_generator import ReferenceLineGenerator

                        gen = ReferenceLineGenerator(daily_df)
                        daily_refs = gen.generate_all()

                        # 映射 id 前缀与内部一致性
                        id_map = {
                            'attack_line': 'attack_line',
                            'trading_line': 'operation_line',
                            'defense_line': 'defense_line',
                            'ma_5': 'ma5',
                            'ma_10': 'ma10',
                            'ma_20': 'ma20',
                            'prev_close': 'prev_close',
                        }
                        # 排除 ReferenceLineGenerator 的 HHV/LLV 和筹码密集区（已单独计算）
                        skip_ids = {'previous_high_30', 'previous_low_30', 'chip_dense_upper', 'chip_dense_lower'}
                        for ref_dict in daily_refs:
                            rid = ref_dict['id']
                            if rid in skip_ids:
                                continue
                            mapped_id = id_map.get(rid, rid)
                            ref_lines.append(ReferenceLine(
                                id=mapped_id,
                                label=ref_dict['label'],
                                price=ref_dict['price'],
                                category=ref_dict['category'],
                                color=ref_dict['color'],
                                style=ref_dict['style'],
                                base_weight=ref_dict['base_weight'],
                            ))
                        logger.debug(f"ReferenceLineGenerator 生成 {len(daily_refs)} 条日线级参考线")

                        # ── 天道指标参考线（金牛 / 金钻）──
                        try:
                            from indicators.indicators.tiandao import Tiandao

                            tiandao = Tiandao()
                            tiandao_result = tiandao.calculate(daily_df)
                            if len(tiandao_result) > 0:
                                latest = tiandao_result.iloc[-1]
                                jinniu_val = latest.get('td_jinniu')
                                jinzuan_val = latest.get('td_jinzuan')

                                if (jinniu_val is not None
                                        and not (isinstance(jinniu_val, float) and math.isnan(jinniu_val))
                                        and jinniu_val > 0):
                                    ref_lines.append(ReferenceLine(
                                        id='tiandao_jinniu',
                                        label='金牛',
                                        price=round(float(jinniu_val), 2),
                                        category='tiandao',
                                        color='#FFFF00',
                                        style='dashed',
                                        base_weight=1.0,
                                    ))

                                if (jinzuan_val is not None
                                        and not (isinstance(jinzuan_val, float) and math.isnan(jinzuan_val))
                                        and jinzuan_val > 0):
                                    ref_lines.append(ReferenceLine(
                                        id='tiandao_jinzuan',
                                        label='金钻',
                                        price=round(float(jinzuan_val), 2),
                                        category='tiandao',
                                        color='#FF0000',
                                        style='dashed',
                                        base_weight=1.0,
                                    ))

                                bbi_val = latest.get('td_bbi')
                                if (bbi_val is not None
                                        and not (isinstance(bbi_val, float) and math.isnan(bbi_val))
                                        and bbi_val > 0):
                                    ref_lines.append(ReferenceLine(
                                        id='tiandao_bbi',
                                        label='BBI',
                                        price=round(float(bbi_val), 2),
                                        category='tiandao',
                                        color='#FFFFFF',
                                        style='dashed',
                                        base_weight=1.0,
                                    ))
                        except Exception as tiandao_err:
                            logger.warning(f"天道指标参考线计算失败: {tiandao_err}")

                        # ── 前高/前低 30个自然日HHV/LLV ──
                        _add_30day_extreme_lines(ref_lines, daily_df, q_date, today_low, today_high)

                else:
                    # 数据库无足够日线数据，尝试从大盘指数缓存读取（如 sh000001/sz399001）
                    _try_load_index_cache_and_generate_ref_lines(
                        ref_lines, code, klines, q_date, today_open, today_high, today_low
                    )

            except Exception as db_err:
                logger.warning(f"从数据库获取日线数据失败，跳过日线级参考线: {db_err}", exc_info=True)

        logger.debug(f"生成 {len(ref_lines)} 条参考线")
        return ref_lines

    except Exception as e:
        logger.error(f"计算参考线失败: {e}", exc_info=True)
        return []


def _add_30day_extreme_lines(ref_lines: list, daily_df, query_date, today_low: float, today_high: float):
    """计算30个自然日内的最高价和最低价（HHV/LLV）

    从查询日往前推30个自然日，取该区间内交易日的日线最高/最低
    排除查询日当天的分时转换数据
    """
    try:
        from datetime import timedelta

        target = query_date - timedelta(days=30)

        # 排除查询日当天，仅使用历史日线数据
        hist_df = daily_df[daily_df.index.date != query_date]
        if hist_df.empty:
            return

        # 从历史日线中找到 <= target 的最近交易日
        eligible = hist_df[hist_df.index.date <= target]
        if eligible.empty:
            range_df = hist_df
        else:
            start_idx = eligible.index[-1]
            range_df = hist_df.loc[start_idx:]

        if range_df.empty:
            return

        hhv_30 = float(range_df['High'].max())
        llv_30 = float(range_df['Low'].min())

        from api.v1.schemas.intraday import ReferenceLine

        ref_lines.append(ReferenceLine(
            id='hhv_30', label='前高30日', price=round(hhv_30, 2),
            category='极值', color='#FF4444', style='solid', base_weight=1.1,
        ))
        ref_lines.append(ReferenceLine(
            id='llv_30', label='前低30日', price=round(llv_30, 2),
            category='极值', color='#44FF44', style='solid', base_weight=1.1,
        ))
        logger.debug(f"30日极值: HHV={hhv_30:.2f}, LLV={llv_30:.2f}")
    except Exception as e:
        logger.warning(f"计算30日极值失败: {e}", exc_info=True)


def _inject_avg_price(klines: list):
    """为每根K线注入累计分时均价

    均价(t) = 累计成交额[0..t] / 累计成交量[0..t]
    结果写入每根K线字典的 AvgPrice 字段
    """
    cum_amount = 0.0
    cum_vol = 0.0
    for k in klines:
        c = k.get('Close', 0)
        c = c if c is not None else 0.0
        v = k.get('Volume', 0) or 0
        v = v if v is not None else 0.0
        if v > 0:
            cum_amount += c * v
            cum_vol += v
            k['AvgPrice'] = round(cum_amount / cum_vol, 2)
        else:
            # 无成交时沿用上一条均价，第一条用Close
            prev = klines[klines.index(k) - 1].get('AvgPrice') if klines.index(k) > 0 else None
            k['AvgPrice'] = prev if prev is not None else round(float(c), 2)


# ============================================================
# 路由定义
# ============================================================


@router.get(
    "/data/{stock_code}",
    response_model=IntradayDataResponse,
    responses={
        200: {"description": "分时K线数据 + 信号"},
        404: {"description": "未找到数据", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取股票分时做T数据",
    description="获取指定股票的分时K线数据、做T买卖信号",
)
def get_intraday_data(
    stock_code: str,
    date: Optional[str] = Query(None, description="日期 YYYYMMDD 或 YYYY-MM-DD，默认当日"),
    strategy: str = Query(
        "auto",
        description="数据获取策略: auto=自动判断(盘中优先缓存,盘后可联网), cache_only=仅缓存/DB, full=包括API联网",
    ),
    warmup_enabled: bool = Query(True, description="是否启用指标预热（前一日K线数据）"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> IntradayDataResponse:
    """获取分时K线数据和信号"""
    try:
        # 检测指数代码（带 sh/sz 前缀），跳过归一化
        is_index_intraday = bool(re.match(r'^[Ss][Hh]\d{6}$', stock_code) or re.match(r'^[Ss][Zz]\d{6}$', stock_code))
        if is_index_intraday:
            code = stock_code  # 使用完整代码（如 sh000001）
            logger.info(f"检测到指数代码: {code}")
        else:
            code = _normalize_stock_code(stock_code)

        # 确定查询日期
        # 原则: 如果今天非交易日或盘前(<9:30), 查询上一交易日; 否则查询今天
        if date is None:
            raw_date = date_type.today()
        elif len(date) == 8:
            raw_date = date_type(int(date[:4]), int(date[4:6]), int(date[6:8]))
        else:
            raw_date = date_type.fromisoformat(date)

        # 非交易日或盘前，回退至最近交易日
        q_date = raw_date
        if _is_trading_day(raw_date):
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now < market_open:
                q_date = _get_nearest_trading_day(raw_date - timedelta(days=1))
        else:
            q_date = _get_nearest_trading_day(raw_date)

        if q_date != raw_date:
            logger.info(f"日期调整: {raw_date} → {q_date} (今天{'非交易日' if not _is_trading_day(raw_date) else '盘前'})")

        date_str = q_date.isoformat()
        is_today = q_date == date_type.today()

        # 解析实际策略
        # auto: 优先走缓存/DB，无数据时自动降级为 full（允许 API 兜底）
        # cache_only: 严格不联网，无数据时返回 404
        # full: 完整 fallback（缓存→DB→API）
        actual_strategy = strategy
        if strategy == "auto":
            actual_strategy = "cache_only"  # 优先只走缓存/DB
        logger.debug(f"[手动查询] {code}: strategy={strategy} → actual={actual_strategy} (is_today={is_today})")

        # 优先走完整响应缓存（TTL自动管理新鲜度，无需is_today限制）
        cached = _get_cached_full_response(code, warmup_enabled)
        if cached is not None:
            logger.info(f"[手动查询] {_format_stock_display(code)}: 命中完整内存缓存")
            return cached
        else:
            if is_today:
                logger.debug(f"[手动查询] {_format_stock_display(code)}: 缓存未命中(is_today=True)")

        klines = None
        # 先从数据库获取请求日期的数据（保留作为备选，不预判丢弃）
        db_klines = db_manager.load_intraday_klines(code, q_date)
        db_is_fresh = db_klines and _is_data_fresh(db_klines, date_str)

        if db_klines:
            if db_is_fresh:
                logger.info(f"[手动查询] {_format_stock_display(code)}: 命中数据库，共 {len(db_klines)} 条K线")
                klines = db_klines
            else:
                logger.info(f"[手动查询] {_format_stock_display(code)}: 数据库数据不完整，保留作为备选")

        # DB数据新鲜则直接使用，否则尝试内存缓存
        if not klines:
            klines = _get_cached_klines(code)
            if klines:
                if _is_data_fresh(klines, date_str):
                    logger.info(f"[手动查询] {_format_stock_display(code)}: 命中K线内存缓存，共 {len(klines)} 条K线")
                else:
                    logger.info(f"[手动查询] {_format_stock_display(code)}: K线内存缓存已陈旧，丢弃并重新获取")
                    klines = None

        # cache_only 策略：不联网，缓存/DB 无数据时返回 404
        # 但如果是 auto 策略降级来的，则允许回退到 API
        if not klines and actual_strategy == "cache_only":
            if strategy == "auto":
                logger.info(f"[手动查询] {_format_stock_display(code)}: auto策略下缓存/DB均无数据，降级为full走API")
            else:
                if is_today:
                    logger.info(f"[手动查询] {_format_stock_display(code)}: cache_only 策略下缓存/DB均无数据，可能尚未开盘或数据尚未到达")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "no_data",
                        "message": "缓存和数据库中暂无该股票的最新分时数据，请等待盘中轮询更新后再查看",
                    },
                )

        if not klines:
            klines = _get_intraday_klines(code, date_str)
            if klines:
                api_actual_date = _extract_date_from_klines(klines, date_str)
                if api_actual_date == date_str:
                    logger.info(f"[手动查询] {_format_stock_display(code)}: 通过API联网获取，共 {len(klines)} 条K线")
                else:
                    # API 返回日期与请求日期不一致（如盘前请求昨日数据，API 返回今日第一根K线）
                    # 使用 API 返回的最新日期数据展示，DB 中请求日期的数据保留不覆盖
                    logger.info(
                        f"[手动查询] {_format_stock_display(code)}: API返回日期({api_actual_date})与请求日期({date_str})不一致，"
                        f"使用API最新数据展示，DB中{date_str}的数据保留不覆盖，共 {len(klines)} 条K线"
                    )
        if not klines:
            raise HTTPException(status_code=404, detail={"error": "no_data", "message": "未获取到分时K线数据"})

        # 从实际K线数据提取日期（比请求参数更可靠）
        actual_date = _extract_date_from_klines(klines, date_str)
        if actual_date != date_str:
            logger.info(f"[数据源] {code}: 日期修正 {date_str} → {actual_date}")

        # 过滤当日K线中的盘后交易数据（15:00-15:30），保留15:00整的K线
        klines = _filter_post_market_klines(klines)

        # 注入累计分时均价（策略需要 AvgPrice）
        _inject_avg_price(klines)

        # 异步存储分时K线到数据库
        _schedule_intraday_storage(code, klines, q_date, db_manager)

        # 计算并异步存储每日分时快照
        _schedule_daily_summary(code, klines, q_date, db_manager)

        # 计算日线级参考线（引力场模型需要）
        reference_lines = _compute_reference_lines(klines, code, db_manager, date_str)

        # 从K线表加载前日数据用于指标预热（直接查K线表，不依赖快照表）
        warmup_klines: list = []
        prev_day_data = None
        if warmup_enabled:
            prev_day_data = db_manager.load_previous_day_klines(code, q_date)
            warmup_klines = prev_day_data.get("klines", []) if prev_day_data else []
            warmup_klines = _filter_post_market_klines(warmup_klines)
            if warmup_klines:
                logger.info(f"[预热] {code} 从K线表加载前日 {len(warmup_klines)} 根K线进行指标预热（已过滤盘后数据），来源日期={prev_day_data.get('date', '')}")
            else:
                logger.warning(f"[预热] {code} 无前日K线数据，指标将从零状态开始计算")

        warmup_info = {
            "enabled": warmup_enabled,
            "last_klines_count": len(warmup_klines),
            "prev_date": str(prev_day_data.get("date", "")) if prev_day_data else "",
            "klines": warmup_klines,
        }

        # 运行做T策略（含引力场）
        signals, summary, precomputed_result, precomputed_engine = _run_t0_strategy(klines, reference_lines, warmup_klines)
        ma5_price = None
        for rl in reference_lines:
            if rl.id == 'ma5':
                ma5_price = rl.price
                break

        # 生成指标子图数据（指数也生成）
        indicator_sub_charts = _generate_indicator_sub_charts(
            klines, warmup_klines, precomputed_result, precomputed_engine,
            ma5_price=ma5_price,
        )

        # 5分钟K线合成 + 天道指标计算
        tiandao_sub_chart = None
        try:
            klines_5min = _synthesize_5min_klines(klines)
            # 前一日5分钟K线（加载完整前日数据用于图表显示，与预热用的80根不同）
            prev_day_5min = []
            if warmup_enabled:
                full_prev_data = db_manager.load_previous_day_klines(code, q_date, limit=0)
                full_prev_klines = full_prev_data.get("klines", []) if full_prev_data else []
                prev_day_5min = _synthesize_5min_klines(full_prev_klines)
            tiandao_result = _compute_tiandao_5min(
                klines_5min, prev_day_5min,
                cache_key=f"{code}_{q_date}",
            )
            tiandao_sub_chart = TiandaoSubChart(
                klines=[FiveMinKlinePoint(**k) for k in klines_5min],
                prev_day_klines=[FiveMinKlinePoint(**k) for k in prev_day_5min],
                jinzuan_line=tiandao_result["jinzuan_line"],
                jinniu_line=tiandao_result["jinniu_line"],
                signals=tiandao_result["signals"],
            )
            logger.info(
                f"[天道5分钟] {_format_stock_display(code)}: "
                f"合成{len(klines_5min)}根5分钟K线, "
                f"前日{len(prev_day_5min)}根, "
                f"天道信号{len(tiandao_result['signals'])}个"
            )
        except Exception as e:
            logger.warning(f"[天道5分钟] {_format_stock_display(code)}: 计算失败: {e}")

        # 构建K线响应
        kline_points = [IntradayKlinePoint(**k) for k in klines]

        # 获取股票名称（指数使用预设映射，个股从 Stock Pool 数据库）
        _INDEX_NAME_MAP = {
            "sh000001": "上证指数", "sz399001": "深证成指",
            "sz399006": "创业板指", "sh000688": "科创50",
            "sh000016": "上证50", "sh000300": "沪深300",
        }
        if is_index_intraday:
            stock_name = _INDEX_NAME_MAP.get(code.lower(), code)
        else:
            try:
                fetcher_manager = DataFetcherManager()
                stock_name = fetcher_manager.get_stock_name(code, skip_realtime=True) or ""
            except Exception as e:
                logger.warning(f"获取股票名称失败 {code}: {e}")
                stock_name = ""

        rsi_ob, rsi_os = _get_rsi_thresholds()
        mfi_ob, mfi_os = _get_mfi_thresholds()
        buy_weights, sell_weights = _get_signal_weights()

        response = IntradayDataResponse(
            stock_code=code,
            stock_name=stock_name,
            date=actual_date,
            kline_data=kline_points,
            signals=signals,
            reference_lines=reference_lines,
            indicator_sub_charts=indicator_sub_charts,
            tiandao_sub_chart=tiandao_sub_chart,
            signal_summary=summary,
            warm_up_summary=None,
            warmup_info=warmup_info,
            rsi_overbought=rsi_ob,
            rsi_oversold=rsi_os,
            mfi_overbought=mfi_ob,
            mfi_oversold=mfi_os,
            buy_weights=buy_weights,
            sell_weights=sell_weights,
        )

        # 写入完整响应缓存（不限is_today，切换股票时可直接命中）
        _set_cached_full_response(code, warmup_enabled, response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取分时数据失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "internal_error", "message": str(e)})


def _compute_daily_summary_from_klines(klines: list) -> dict:
    """从K线数据计算每日终值快照的指标数据
    Args:
        klines: K线字典列表
    Returns:
        指标快照字典
    """
    if not klines:
        return {}

    # 用于次日指标预热的K线数量
    WARMUP_BARS = 80

    last_k = klines[-1]
    result = {
        "last_price": last_k.get("Close") or 0.0,
        "avg_price": last_k.get("AvgPrice") or 0.0,
        "open_price": klines[0].get("Open") or 0.0,
        "high_price": max((k.get("High") or 0.0 for k in klines), default=0.0),
        "low_price": min((k.get("Low") or 0.0 for k in klines), default=0.0),
        "total_volume": sum(k.get("Volume") or 0.0 for k in klines),
        "total_amount": sum(k.get("Amount") or 0.0 for k in klines),
        "kline_count": len(klines),
        "last_time": last_k.get("timestamp", "")[11:16] if len(last_k.get("timestamp", "")) >= 16 else "",
        "last_klines": klines[-WARMUP_BARS:],  # 最后N根K线，用于次日指标预热
    }
    return result


def _schedule_intraday_storage(code: str, klines: list, q_date: date_type, db_manager: DatabaseManager):
    """异步调度分时K线存储任务（不阻塞主请求）
    使用 K 线实际日期而非查询日期，防止盘后/凌晨获取到昨日数据时日期错配。
    注意：使用 daemon 线程，如果主请求结束但 uvicon 进程存活，线程将继续执行
    """
    actual_date_str = _extract_date_from_klines(klines, q_date.isoformat())
    try:
        actual_q_date = date_type.fromisoformat(actual_date_str)
    except (ValueError, TypeError):
        actual_q_date = q_date

    def _store():
        try:
            count = db_manager.save_intraday_klines(code, actual_q_date, klines)
            logger.debug(f"后台存储完成: {code} {actual_q_date}, {count} 条")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"后台存储分时K线失败 {code} {actual_q_date}: {e}")

    t = threading.Thread(target=_store, daemon=True)
    t.start()
    logger.debug(f"已调度后台存储: {code} {actual_q_date}, 共 {len(klines)} 条K线")


def _schedule_daily_summary(code: str, klines: list, q_date: date_type, db_manager: DatabaseManager):
    """异步调度每日快照存储任务（不阻塞主请求）
    使用 K 线实际日期而非查询日期，防止日期错配。
    注意：使用 daemon 线程，如果主请求结束但 uvicon 进程存活，线程将继续执行
    """
    actual_date_str = _extract_date_from_klines(klines, q_date.isoformat())
    try:
        actual_q_date = date_type.fromisoformat(actual_date_str)
    except (ValueError, TypeError):
        actual_q_date = q_date

    def _store():
        try:
            # 计算指标快照
            indicators = _compute_daily_summary_from_klines(klines)
            ok = db_manager.save_daily_summary(code, actual_q_date, klines, indicators)
            logger.debug(f"后台快照存储完成: {code} {actual_q_date}, 成功={ok}")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"后台存储每日快照失败 {code} {actual_q_date}: {e}")

    t = threading.Thread(target=_store, daemon=True)
    t.start()
    logger.debug(f"已调度快照存储: {code} {actual_q_date}")


# ---------- 批量状态查询 ----------


def _fetch_one_and_store(code: str, db_manager: DatabaseManager) -> Optional[dict]:
    """获取单只股票K线并写入DB和缓存"""
    try:
        klines = _get_intraday_klines(code, None)
        if not klines:
            return None
        actual_date = _extract_date_from_klines(klines)
        _schedule_intraday_storage(code, klines, date_type.today(), db_manager)
        _schedule_daily_summary(code, klines, date_type.today(), db_manager)
        _set_cached_klines(code, klines)
        return {"klines": klines, "actual_date": actual_date}
    except Exception as e:
        logger.warning(f"[batch-fetch] {code}: 获取K线失败: {e}")
        return None


def _batch_fetch_and_store_all(codes: list, db_manager: DatabaseManager) -> dict:
    """并行获取所有标的K线，写DB和缓存"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not codes:
        return {}

    results: dict = {}
    with ThreadPoolExecutor(max_workers=min(5, len(codes))) as executor:
        future_to_code = {
            executor.submit(_fetch_one_and_store, code, db_manager): code
            for code in codes
        }
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                result = future.result(timeout=15)
                if result:
                    results[code] = result
            except Exception as e:
                logger.warning(f"[batch-fetch] {code}: 超时或异常: {e}")

    return results


def _compute_signal_alert(code: str, db_manager=None) -> Optional[dict]:
    """计算单只股票的最新信号告警
    优先从步骤2已更新的K线缓存获取，避免重复API请求和DB写入。
    """
    try:
        klines = _get_cached_klines(code)
        if not klines:
            klines = _get_intraday_klines(code, None)
        if not klines or len(klines) < 2:
            return None
        _inject_avg_price(klines)

        warmup_klines = []
        if db_manager:
            actual_date_str = _extract_date_from_klines(klines)
            try:
                q_date = date_type.fromisoformat(actual_date_str)
            except (ValueError, TypeError):
                q_date = date_type.today()
            prev_day_data = db_manager.load_previous_day_klines(code, q_date)
            warmup_klines = prev_day_data.get("klines", []) if prev_day_data else []
            warmup_klines = _filter_post_market_klines(warmup_klines)

        signals, _summary, _precomputed, _engine = _run_t0_strategy(klines, None, warmup_klines)
        if not signals:
            return None
        latest = signals[-1]
        return {
            "stock_code": code,
            "signal_type": latest.signal_type,
            "trigger_time": str(latest.trigger_time),
            "price": round(latest.price, 2),
        }
    except Exception as e:
        logger.warning(f"计算信号告警失败 {code}: {e}")
        return None


def _batch_compute_signal_alerts(codes: list, skip_code: str = None, db_manager=None) -> dict:
    """批量并行计算多只股票的信号告警
    使用 ThreadPoolExecutor 并行获取K线和运行策略，单只超时 10 秒
    Args:
        codes: 股票代码列表
        skip_code: 跳过的股票代码（已在 batch-status 中刷新过）
        db_manager: 数据库管理器（可选，用于交易时段写入DB）
    Returns:
        {code: SignalAlert字典 或 None}
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    if not codes:
        return {}

    results: dict = {}
    codes_to_process = [c for c in codes if c != skip_code]
    if not codes_to_process:
        return results
    with ThreadPoolExecutor(max_workers=min(5, len(codes_to_process))) as executor:
        future_to_code = {executor.submit(_compute_signal_alert, c, db_manager): c for c in codes_to_process}
        for future in future_to_code:
            code = future_to_code[future]
            try:
                results[code] = future.result(timeout=10)
            except FutureTimeoutError:
                logger.warning(f"信号检测超时: {code}")
                results[code] = None
            except Exception as e:
                logger.warning(f"信号检测异常 {code}: {e}")
                results[code] = None

    return results


def _run_simulated_trading(code: str, db_manager: DatabaseManager) -> Optional[SimulationReportResponse]:
    """对指定股票运行模拟交易，返回统计报告

    使用已有的K线数据和做T策略，模拟逐根K线的买卖交易。
    """
    try:
        q_date = date_type.today()
        klines = db_manager.load_intraday_klines(code, q_date)
        if not klines:
            klines = _get_cached_klines(code)
        if not klines:
            klines = _get_intraday_klines(code, None)
        if not klines or len(klines) < 2:
            return None

        _inject_avg_price(klines)
        reference_lines = _compute_reference_lines(klines, code, db_manager, None)
        signals, _summary, _precomputed, _engine = _run_t0_strategy(klines, reference_lines)

        from watchdog.strategies.simulator import T0Simulator
        from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy

        config_path = str(_get_project_root() / "watchdog" / "strategies" / "intraday_t0_config.yaml")
        config_path = os.path.normpath(config_path)
        strategy = IntradayT0Strategy(stock_code=code, stock_name="", config_path=config_path)
        simulator = T0Simulator(strategy)
        report = simulator.run(klines, verbose=False)

        trade_items = []
        for t in report.trades:
            if t.status == "closed" and t.return_pct is not None:
                trade_items.append(SimulatedTradeItem(
                    buy_time=str(t.buy_time) if t.buy_time else "",
                    buy_price=round(t.buy_price, 2),
                    sell_time=str(t.sell_time) if t.sell_time else "",
                    sell_price=round(t.sell_price, 2),
                    return_pct=round(t.return_pct, 2),
                ))

        return SimulationReportResponse(
            stock_code=code,
            total_klines=report.total_klines,
            total_signals=report.total_signals,
            buy_signals=report.buy_signals,
            sell_signals=report.sell_signals,
            total_trades=report.total_trades,
            win_trades=report.win_trades,
            lose_trades=report.lose_trades,
            win_rate=round(report.win_rate, 4),
            avg_return_pct=round(report.avg_return_pct, 4),
            max_return_pct=round(report.max_return_pct, 4),
            min_return_pct=round(report.min_return_pct, 4),
            total_return_pct=round(report.total_return_pct, 4),
            max_drawdown_pct=round(report.max_drawdown_pct, 4),
            profit_factor=round(report.profit_factor, 4),
            trades=trade_items,
        )
    except Exception as e:
        logger.warning(f"模拟交易失败 {code}: {e}", exc_info=True)
        return None


def _parse_tencent_realtime_batch(stock_codes: list) -> dict:
    """通过腾讯 qt.gtimg.cn 接口批量获取股票实时行情（含五档盘口）

    腾讯接口字段说明（~分隔）:
    [1]=名称 [3]=最新价 [4]=昨收 [5]=今开 [6]=成交量(手)
    [9]~[18]=买五价/量→买一价/量  [19]~[28]=卖一价/量→卖五价/量
    [30]=时间戳 [32]=涨跌幅(%) [34]=最高 [35]=最低/成交量/成交额
    [38]=换手率 [39]=市盈率 [43]=振幅 [44]=流通市值 [45]=总市值
    [46]=市净率 [47]=涨停价 [48]=跌停价 [49]=量比

    Returns:
        {stock_code: {stock_code, stock_name, latest_price, change_pct,
                      open_price, high, low, pre_close, volume, timestamp,
                      bid_prices, ask_prices, bid_volumes, ask_volumes}}
    """
    import requests as req

    # 构建代码 → 完整symbol的映射（保留原始输入代码用于结果key）
    symbol_map = {}  # {input_code: full_symbol}
    for code in stock_codes:
        code = code.strip()
        code_lower = code.lower()
        if re.match(r'^sh\d{6}$', code_lower):
            symbol_map[code] = code_lower  # sh000001 → sh000001
        elif re.match(r'^sz\d{6}$', code_lower):
            symbol_map[code] = code_lower  # sz399001 → sz399001
        elif code_lower.startswith(("6", "5", "9")):
            symbol_map[code] = f"sh{code_lower}"
        else:
            symbol_map[code] = f"sz{code_lower}"

    if not symbol_map:
        return {}

    symbols = list(symbol_map.values())
    # 反向映射：完整symbol → 原始输入代码
    symbol_to_input = {v: k for k, v in symbol_map.items()}
    url = f"http://qt.gtimg.cn/q={','.join(symbols)}"
    headers = {
        "Referer": "http://finance.qq.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        r = req.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
    except Exception as e:
        logger.warning(f"腾讯批量行情接口请求失败: {e}")
        return {}

    result = {}
    content = r.text.strip()
    # 腾讯返回格式: v_sh600519="...";v_sz000001="..."
    lines = content.split(";")

    for line in lines:
        line = line.strip()
        if not line or "=" not in line:
            continue

        eq_pos = line.find("=")
        code_part = line[:eq_pos].strip()
        if code_part.startswith("v_"):
            code_part = code_part[2:]

        # 将响应中的symbol映射回原始输入代码
        stock_code = symbol_to_input.get(code_part)
        if stock_code is None:
            continue

        # 提取数据
        data_start = line.find('"')
        data_end = line.rfind('"')
        if data_start == -1 or data_end == -1:
            continue

        data_str = line[data_start + 1 : data_end]
        fields = data_str.split("~")

        if len(fields) < 30:
            continue

        try:
            name = fields[1] if len(fields) > 1 else ""
            price = float(fields[3]) if fields[3] else 0.0
            pre_close = float(fields[4]) if fields[4] else 0.0
            open_price = float(fields[5]) if fields[5] else 0.0
            vol_hands = int(float(fields[6])) if fields[6] else 0  # 成交量(手)
            volume = vol_hands * 100  # 转为股
            change_pct = float(fields[32]) if len(fields) > 32 and fields[32] else 0.0
            high = float(fields[34]) if len(fields) > 34 and fields[34] else 0.0
            # fields[35] 格式: "最低/成交量/成交额"，取最低这一部分
            low_str = (fields[35].split("/")[0]) if len(fields) > 35 and "/" in str(fields[35]) else (fields[35] if len(fields) > 35 else "0")
            low = float(low_str) if low_str else 0.0
            timestamp = fields[30] if len(fields) > 30 else ""

            # 解析五档盘口：fields[9]~[18]为买方(买一→买五)，fields[19]~[28]为卖方(卖一→卖五)
            # 腾讯原始: 买一价/量(9/10)→买二(11/12)→买三(13/14)→买四(15/16)→买五(17/18)→卖一(19/20)→...→卖五(27/28)
            # 统一存为 买一→买五, 卖一→卖五（从近到远）
            bid_prices = []
            bid_volumes = []
            for idx in range(9, 18, 2):  # 9, 11, 13, 15, 17 (买一价→买五价)
                if idx < len(fields) and fields[idx]:
                    bid_prices.append(float(fields[idx]))
                else:
                    bid_prices.append(0.0)
            for idx in range(10, 19, 2):  # 10, 12, 14, 16, 18 (买一量→买五量)
                if idx < len(fields) and fields[idx]:
                    bid_volumes.append(int(float(fields[idx])))
                else:
                    bid_volumes.append(0)

            ask_prices = []
            ask_volumes = []
            for idx in range(19, 28, 2):  # 19, 21, 23, 25, 27 (卖一价→卖五价)
                if idx < len(fields) and fields[idx]:
                    ask_prices.append(float(fields[idx]))
                else:
                    ask_prices.append(0.0)
            for idx in range(20, 29, 2):  # 20, 22, 24, 26, 28 (卖一量→卖五量)
                if idx < len(fields) and fields[idx]:
                    ask_volumes.append(int(float(fields[idx])))
                else:
                    ask_volumes.append(0)

            # 估值指标
            volume_ratio = float(fields[49]) if len(fields) > 49 and fields[49] else None
            turnover_rate = float(fields[38]) if len(fields) > 38 and fields[38] else None
            pe_ratio = float(fields[39]) if len(fields) > 39 and fields[39] else None
            pb_ratio = float(fields[46]) if len(fields) > 46 and fields[46] else None

            result[stock_code] = {
                "stock_code": stock_code,
                "stock_name": name,
                "latest_price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "open_price": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "pre_close": round(pre_close, 2),
                "volume": volume,
                "timestamp": timestamp,
                "ask_prices": ask_prices,
                "ask_volumes": ask_volumes,
                "bid_prices": bid_prices,
                "bid_volumes": bid_volumes,
                "volume_ratio": volume_ratio,
                "turnover_rate": turnover_rate,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
            }

        except (ValueError, IndexError) as e:
            logger.debug(f"腾讯解析 {stock_code} 失败: {e}")
            continue

    return result


def _parse_sina_realtime_batch(stock_codes: list) -> dict:
    """通过新浪接口批量获取股票实时行情

    新浪接口支持逗号分隔的多股票批量查询：
    http://hq.sinajs.cn/list=sh600519,sz000001,sh600000

    Returns:
        {stock_code: {name, price, change_pct, open, high, low, timestamp}}
    """
    import re
    import requests as req

    # 构建代码 → 完整symbol的映射（保留原始输入代码用于结果key）
    symbol_map = {}
    for code in stock_codes:
        code = code.strip()
        code_lower = code.lower()
        if re.match(r'^sh\d{6}$', code_lower):
            symbol_map[code] = code_lower
        elif re.match(r'^sz\d{6}$', code_lower):
            symbol_map[code] = code_lower
        elif code_lower.startswith(("6", "5", "9")):
            symbol_map[code] = f"sh{code_lower}"
        else:
            symbol_map[code] = f"sz{code_lower}"

    if not symbol_map:
        return {}

    symbol_to_input = {v: k for k, v in symbol_map.items()}
    url = f"http://hq.sinajs.cn/list={','.join(symbol_map.values())}"
    headers = {
        "Referer": "http://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        r = req.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        text = r.text
    except Exception as e:
        logger.warning(f"新浪批量行情接口请求失败: {e}")
        return {}

    result = {}
    pattern = re.compile(r'var hq_str_(sh|sz)(\d+)="([^"]*)"')
    for match in pattern.finditer(text):
        prefix = match.group(1)
        numeric_code = match.group(2)
        data_str = match.group(3)
        raw_symbol = f"{prefix}{numeric_code}"
        fields = data_str.split(",")
        if len(fields) < 33:
            continue

        stock_code = symbol_to_input.get(raw_symbol)
        if stock_code is None:
            continue
        try:
            name = fields[0]
            open_price = float(fields[1]) if fields[1] else 0.0
            pre_close = float(fields[2]) if fields[2] else 0.0
            price = float(fields[3]) if fields[3] else 0.0
            high = float(fields[4]) if fields[4] else 0.0
            low = float(fields[5]) if fields[5] else 0.0
            change_pct = ((price - pre_close) / pre_close * 100) if pre_close > 0 else 0.0
            timestamp = fields[31] if len(fields) > 31 else ""
        except (ValueError, IndexError):
            continue

        result[stock_code] = {
            "stock_code": stock_code,
            "stock_name": name,
            "latest_price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "open_price": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "timestamp": timestamp,
        }

    return result


@router.post(
    "/batch-status",
    response_model=BatchStatusResponse,
    responses={
        200: {"description": "批量实时行情 + 可选完整分时数据"},
    },
    summary="批量获取搜索历史股票的实时状态",
    description="批量获取搜索历史中所有股票的实时行情快照，同时检查当前展示股票是否有新数据需要刷新。",
)
def get_batch_status(
    body: BatchStatusRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BatchStatusResponse:
    """批量获取股票实时行情，并检查当前展示股票是否需要刷新"""
    try:
        stock_codes = [c.strip() for c in body.stock_codes if c.strip()]
        if not stock_codes:
            return BatchStatusResponse(snapshots={}, current_updated=False)

        # 1. 批量获取实时行情
        raw_snapshots = _parse_tencent_realtime_batch(stock_codes)

        snapshots: dict = {}
        for code in stock_codes:
            if code in raw_snapshots:
                snapshots[code] = StockSnapshot(**raw_snapshots[code])
            else:
                snapshots[code] = StockSnapshot(stock_code=code)

        # 2. 并行获取所有标的的K线 → 写DB → 写klines缓存（可视化页面跳过）
        batch_results = {}
        if not body.skip_kline_fetch:
            batch_results = _batch_fetch_and_store_all(stock_codes, db_manager)

        # 3. 对 current_code 计算信号并更新 full 缓存
        current_code = body.current_code.strip()
        current_updated = False
        current_full_data = None

        if current_code and current_code in batch_results:
            # 检测指数代码，与 get_intraday_data 保持一致
            is_index_current = bool(re.match(r'^[Ss][Hh]\d{6}$', current_code) or re.match(r'^[Ss][Zz]\d{6}$', current_code))
            if is_index_current:
                code = current_code  # 指数保持完整代码（如 sh000001）
            else:
                code = _normalize_stock_code(current_code)
            info = batch_results[current_code]
            actual_date = info["actual_date"]
            klines = info["klines"]
            today_str = datetime.now().strftime("%Y-%m-%d")

            if actual_date == today_str:
                current_updated = True
                try:
                    klines = _filter_post_market_klines(klines)
                    _inject_avg_price(klines)
                    reference_lines = _compute_reference_lines(klines, code, db_manager, None)
                    # 从K线表加载前日数据用于指标预热
                    batch_prev_day = db_manager.load_previous_day_klines(code, actual_date)
                    batch_warmup_klines = batch_prev_day.get("klines", []) if batch_prev_day else []
                    batch_warmup_klines = _filter_post_market_klines(batch_warmup_klines)
                    if batch_warmup_klines:
                        logger.info(f"[预热] batch-status {code} 从K线表加载前日 {len(batch_warmup_klines)} 根K线")
                    else:
                        logger.warning(f"[预热] batch-status {code} 无前日K线数据")
                    signals, summary, precomputed_result, precomputed_engine = _run_t0_strategy(klines, reference_lines, batch_warmup_klines)
                    batch_ma5_price = None
                    for rl in reference_lines:
                        if rl.id == 'ma5':
                            batch_ma5_price = rl.price
                            break
                    indicator_sub_charts = _generate_indicator_sub_charts(
                        klines, batch_warmup_klines, precomputed_result, precomputed_engine,
                        ma5_price=batch_ma5_price,
                    )
                    # 5分钟K线合成 + 天道指标
                    batch_tiandao_sub_chart = None
                    try:
                        batch_5min = _synthesize_5min_klines(klines)
                        # 前一日5分钟K线（加载完整前日数据用于图表显示）
                        batch_prev_5min = []
                        batch_full_prev = db_manager.load_previous_day_klines(code, actual_date, limit=0)
                        batch_full_prev_klines = batch_full_prev.get("klines", []) if batch_full_prev else []
                        batch_prev_5min = _synthesize_5min_klines(batch_full_prev_klines)
                        batch_td_result = _compute_tiandao_5min(
                            batch_5min, batch_prev_5min,
                            cache_key=f"{code}_{actual_date}",
                        )
                        batch_tiandao_sub_chart = TiandaoSubChart(
                            klines=[FiveMinKlinePoint(**k) for k in batch_5min],
                            prev_day_klines=[FiveMinKlinePoint(**k) for k in batch_prev_5min],
                            jinzuan_line=batch_td_result["jinzuan_line"],
                            jinniu_line=batch_td_result["jinniu_line"],
                            signals=batch_td_result["signals"],
                        )
                    except Exception as e:
                        logger.warning(f"[天道5分钟-批量] {code}: 计算失败: {e}")
                    kline_points = [IntradayKlinePoint(**k) for k in klines]
                    sn = raw_snapshots.get(current_code, {})
                    stock_name = sn.get("stock_name", "")
                    if not stock_name:
                        if is_index_current:
                            _INDEX_NAME_MAP_BATCH = {
                                "sh000001": "上证指数", "sz399001": "深证成指",
                                "sz399006": "创业板指", "sh000688": "科创50",
                                "sh000016": "上证50", "sh000300": "沪深300",
                            }
                            stock_name = _INDEX_NAME_MAP_BATCH.get(code.lower(), code)
                        else:
                            try:
                                fetcher_manager = DataFetcherManager()
                                stock_name = fetcher_manager.get_stock_name(code, skip_realtime=True) or ""
                            except Exception:
                                pass

                    rsi_ob, rsi_os = _get_rsi_thresholds()
                    mfi_ob, mfi_os = _get_mfi_thresholds()
                    buy_weights, sell_weights = _get_signal_weights()

                    current_full_data = IntradayDataResponse(
                        stock_code=code,
                        stock_name=stock_name,
                        date=actual_date,
                        kline_data=kline_points,
                        signals=signals,
                        reference_lines=reference_lines,
                        indicator_sub_charts=indicator_sub_charts,
                        tiandao_sub_chart=batch_tiandao_sub_chart,
                        signal_summary=summary,
                        warm_up_summary=None,
                        warmup_info={
                            "enabled": True,
                            "last_klines_count": len(batch_warmup_klines),
                            "prev_date": str(batch_prev_day.get("date", "")) if batch_prev_day else "",
                            "klines": batch_warmup_klines,
                        },
                        rsi_overbought=rsi_ob,
                        rsi_oversold=rsi_os,
                        mfi_overbought=mfi_ob,
                        mfi_oversold=mfi_os,
                        buy_weights=buy_weights,
                        sell_weights=sell_weights,
                    )
                    _set_cached_full_response(code, True, current_full_data)
                except Exception as e:
                    logger.warning(f"计算当前股票 {current_code} 信号失败: {e}")
            else:
                logger.warning(
                    f"[batch-status] {code}: K线日期({actual_date})与今日({today_str})不一致，跳过UI更新"
                )

        # 4. 信号检测（复用步骤2已缓存的klines，不再重复获取API/写DB）
        signal_alerts = None
        if body.include_signals:
            try:
                signal_alerts = _batch_compute_signal_alerts(stock_codes, skip_code=current_code, db_manager=db_manager)
            except Exception as e:
                logger.warning(f"批量信号检测失败: {e}")
                signal_alerts = None

        return BatchStatusResponse(
            snapshots=snapshots,
            current_updated=current_updated,
            current_full_data=current_full_data,
            signal_alerts=signal_alerts,
        )

    except Exception as e:
        logger.error(f"批量状态查询失败: {e}", exc_info=True)
        return BatchStatusResponse(snapshots={}, current_updated=False)


@router.post(
    "/{stock_code}/simulate-trading",
    response_model=SimulationReportResponse,
    summary="模拟做T交易盈亏",
    description="基于当日分时K线数据，完整回放做T策略，统计模拟交易的盈亏报告",
)
def get_simulate_trading(
    stock_code: str,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> SimulationReportResponse:
    """运行模拟交易并返回统计报告"""
    code = _normalize_stock_code(stock_code)
    report = _run_simulated_trading(code, db_manager)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "no_data", "message": f"股票 {stock_code} 模拟交易数据不足"},
        )
    logger.info(f"模拟交易完成: {code}, 交易{report.total_trades}次, 累计收益{report.total_return_pct:+.2f}%")
    return report


# ---------- 批量下载分时数据 ----------

_batch_download_tasks: dict = {}
_batch_download_lock = threading.Lock()


def _run_batch_download(task_id: str, target_date: str, max_workers: int, force: bool = False):
    """后台线程：批量下载所有A股分时数据"""
    import uuid

    with _batch_download_lock:
        task = _batch_download_tasks.get(task_id)
        if not task:
            return

    try:
        from stock_selector.stock_pool import (
            get_all_stock_code_name_pairs,
            filter_beijing_stock_exchange,
            filter_special_stock_codes,
        )

        stock_pairs = get_all_stock_code_name_pairs(force_refresh=False)
        code_to_name = {code: name for code, name in stock_pairs}

        # 预过滤不可下载的标的（北交所、科创板、创业板等），避免 total 虚高导致进度条走不满
        codes = [code for code, _ in stock_pairs]
        codes = filter_beijing_stock_exchange(codes)
        codes = filter_special_stock_codes(codes)

        with _batch_download_lock:
            task["total"] = len(codes)
            task["current_code"] = ""
            task["current_name"] = ""

        # 创建独立的 DatabaseManager 实例（避免 session 共享问题）
        from src.storage import DatabaseManager as DBManager
        from datetime import date as date_type

        db = DBManager.get_instance()
        q_date = date_type.fromisoformat(target_date) if target_date else date_type.today()

        processed = 0
        failed = 0
        skipped = 0

        # 按市场分组：沪市优先（代码 6 开头）
        sh_codes = [c for c in codes if c.startswith("6")]
        sz_codes = [c for c in codes if not c.startswith("6")]
        ordered_codes = sh_codes + sz_codes

        # 分段批量处理，每段内并行
        batch_size = 20
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 记录需要重试的股票代码（因频率限制导致的连接错误）
        retry_codes: list = []

        for i in range(0, len(ordered_codes), batch_size):
            # 检查是否被暂停
            while True:
                with _batch_download_lock:
                    if task.get("cancelled"):
                        task["status"] = "cancelled"
                        task["end_time"] = time.time()
                        return
                    paused = task.get("paused", False)
                if not paused:
                    break
                time.sleep(1)

            with _batch_download_lock:
                if task.get("cancelled"):
                    task["status"] = "cancelled"
                    task["end_time"] = time.time()
                    return

            batch = ordered_codes[i : i + batch_size]

            def _fetch_one(code: str):
                """获取单只股票的分时K线"""
                nonlocal db, q_date, target_date
                try:
                    # 非强制模式下检查是否已在 DB 中
                    if not force:
                        existing = db.load_intraday_klines(code, q_date)
                        if existing and len(existing) > 0:
                            return code, True, True  # skipped

                    klines = _get_intraday_klines(code, target_date)
                    if not klines:
                        return code, False, "无分时数据"

                    _inject_avg_price(klines)
                    # 使用K线实际日期存储，防止盘后获取昨日数据时日期错配
                    actual_date_str = _extract_date_from_klines(klines, q_date.isoformat())
                    try:
                        actual_q_date = date_type.fromisoformat(actual_date_str)
                    except (ValueError, TypeError):
                        actual_q_date = q_date
                    db.save_intraday_klines(code, actual_q_date, klines)
                    # 缓存 K 线供后续使用
                    _set_cached_klines(code, klines)
                    return code, True, False  # success, not skipped
                except Exception as e:
                    return code, False, str(e)[:200]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_one, code): code for code in batch}
                # 记录未完成的 future，超时时将其标记为失败并继续下一批
                from concurrent.futures import TimeoutError as FuturesTimeoutError
                remaining_futures = dict(futures)

                batch_failed_codes = []  # 当前批次因连接错误失败的股票

                try:
                    for future in as_completed(futures, timeout=30):
                        code = futures[future]
                        remaining_futures.pop(future, None)
                        try:
                            code, ok, info = future.result(timeout=5)
                        except Exception as e:
                            ok, info = False, str(e)[:200]

                        if ok:
                            if info is True:  # skipped
                                skipped += 1
                            else:
                                processed += 1
                        else:
                            failed += 1
                            fail_reason: str = str(info)
                            # 检查是否为连接相关错误（频率限制），但不包括"所有数据源耗尽"
                            is_all_sources_failed = "所有数据源获取" in fail_reason
                            if is_all_sources_failed:
                                # 所有数据源均失败，记录到数据库供后续"失败重试"
                                try:
                                    db.mark_batch_download_failed(code, q_date, str(info))
                                except Exception:
                                    pass
                            is_connection_error = (
                                not is_all_sources_failed
                                and any(
                                    kw in fail_reason.lower()
                                    for kw in [
                                        "connection", "remote end closed", "timeout",
                                        "remoteDisconnected", "ConnectionError",
                                        "Connection aborted", "too many", "rate limit",
                                    ]
                                )
                            )
                            if is_connection_error:
                                batch_failed_codes.append(code)
                            with _batch_download_lock:
                                if len(task["errors"]) < 20:
                                    task["errors"].append({"code": code, "error": str(info)})

                        with _batch_download_lock:
                            task["completed"] = processed + skipped
                            task["failed"] = failed
                            task["skipped"] = skipped
                            task["current_code"] = code
                            task["current_name"] = code_to_name.get(code, "")
                            task["elapsed_seconds"] = time.time() - task["start_time"]

                except FuturesTimeoutError:
                    # as_completed 超时: 将剩余未完成的 futures 标记为失败并继续下一批
                    timeout_count = len(remaining_futures)
                    logger.warning(
                        f"批次 {i // batch_size + 1}: {timeout_count}/{len(batch)} 只股票超时，"
                        f"标记为失败并继续下一批"
                    )
                    for future, code in remaining_futures.items():
                        future.cancel()
                        failed += 1
                        batch_failed_codes.append(code)
                        with _batch_download_lock:
                            if len(task["errors"]) < 20:
                                task["errors"].append({"code": code, "error": "获取分时数据超时"})
                            task["failed"] = failed
                            task["current_code"] = code
                            task["current_name"] = code_to_name.get(code, "")
                            task["elapsed_seconds"] = time.time() - task["start_time"]

            # 如果当前批次有因频率限制导致的连接错误，等待1分钟后重试
            if batch_failed_codes:
                logger.warning(
                    f"批次 {i // batch_size + 1}: {len(batch_failed_codes)} 只股票因连接错误/频率限制失败，"
                    f"等待 60 秒后重试..."
                )
                retry_codes.extend(batch_failed_codes)
                # 更新任务状态，通知前端即将等待
                with _batch_download_lock:
                    task["waiting_retry"] = True
                    task["retry_countdown"] = 60

                # 倒计时等待，同时检查暂停/取消
                for sec in range(60, 0, -1):
                    with _batch_download_lock:
                        if task.get("cancelled"):
                            task["status"] = "cancelled"
                            task["end_time"] = time.time()
                            return
                        paused = task.get("paused", False)
                    if paused:
                        # 暂停期间不减少倒计时，等待恢复
                        while True:
                            with _batch_download_lock:
                                if task.get("cancelled"):
                                    task["status"] = "cancelled"
                                    task["end_time"] = time.time()
                                    return
                                paused = task.get("paused", False)
                            if not paused:
                                break
                            time.sleep(1)
                    with _batch_download_lock:
                        task["retry_countdown"] = sec
                    time.sleep(1)

                with _batch_download_lock:
                    task["waiting_retry"] = False
                    task["retry_countdown"] = 0

                # 重试失败的股票（逐个重试，降低频率压力）
                logger.info(f"开始重试 {len(batch_failed_codes)} 只失败股票...")
                for code in batch_failed_codes:
                    with _batch_download_lock:
                        if task.get("cancelled"):
                            task["status"] = "cancelled"
                            task["end_time"] = time.time()
                            return
                        paused = task.get("paused", False)
                    while paused:
                        with _batch_download_lock:
                            if task.get("cancelled"):
                                task["status"] = "cancelled"
                                task["end_time"] = time.time()
                                return
                            paused = task.get("paused", False)
                        if not paused:
                            break
                        time.sleep(1)

                    with _batch_download_lock:
                        task["current_code"] = code
                        task["current_name"] = code_to_name.get(code, "")
                        task["elapsed_seconds"] = time.time() - task["start_time"]

                    try:
                        klines_retry = _get_intraday_klines(code, target_date)
                        if klines_retry:
                            _inject_avg_price(klines_retry)
                            # 使用K线实际日期存储
                            actual_date_str = _extract_date_from_klines(klines_retry, q_date.isoformat())
                            try:
                                actual_q_date = date_type.fromisoformat(actual_date_str)
                            except (ValueError, TypeError):
                                actual_q_date = q_date
                            db.save_intraday_klines(code, actual_q_date, klines_retry)
                            _set_cached_klines(code, klines_retry)
                            processed += 1
                            failed -= 1
                            with _batch_download_lock:
                                task["completed"] = processed + skipped
                                task["failed"] = failed
                            logger.debug(f"重试成功: {code}")
                        else:
                            logger.debug(f"重试仍无数据: {code}")
                    except Exception as retry_err:
                        logger.debug(f"重试失败: {code}: {retry_err}")

                    # 重试之间间隔 0.5 秒
                    time.sleep(0.5)

                    with _batch_download_lock:
                        task["completed"] = processed + skipped
                        task["elapsed_seconds"] = time.time() - task["start_time"]
            else:
                # 每批之间暂停 0.3s，避免 API 限流
                time.sleep(0.3)

        with _batch_download_lock:
            task["status"] = "completed"
            task["end_time"] = time.time()
            task["elapsed_seconds"] = task["end_time"] - task["start_time"]

        logger.info(
            f"批量下载完成: {task_id}, 成功{processed}, 跳过{skipped}, 失败{failed}, "
            f"耗时{task['elapsed_seconds']:.1f}s"
        )
    except Exception as e:
        logger.error(f"批量下载异常: {task_id}: {e}", exc_info=True)
        with _batch_download_lock:
            task["status"] = "failed"
            task["end_time"] = time.time()
            task["errors"].append({"code": "system", "error": str(e)[:200]})


@router.post(
    "/batch-download",
    response_model=BatchDownloadStatus,
    summary="批量下载A股分时数据",
    description="启动后台任务，批量下载所有A股的当天分时K线数据并存储到数据库",
)
def start_batch_download(
    date: Optional[str] = Body(None, description="目标日期 YYYY-MM-DD，默认当日"),
    max_workers: int = Body(8, ge=1, le=20, description="并行线程数"),
    force: bool = Body(False, description="强制更新，忽略数据库中已有数据"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BatchDownloadStatus:
    import threading
    import uuid as uuid_module

    # 检查是否已有正在运行的任务
    with _batch_download_lock:
        for tid, t in _batch_download_tasks.items():
            if t.get("status") == "running":
                return BatchDownloadStatus(
                    task_id=tid,
                    status="running",
                    total=t["total"],
                    completed=t["completed"],
                    failed=t["failed"],
                    skipped=t.get("skipped", 0),
                    current_code=t["current_code"],
                    current_name=t["current_name"],
                    elapsed_seconds=time.time() - t["start_time"],
                    errors=t.get("errors", []),
                    date=t.get("date", ""),
                )

    target_date = date or datetime.now().strftime("%Y-%m-%d")
    task_id = uuid_module.uuid4().hex[:12]

    # 清除上一个交易日的失败记录，节省空间
    from datetime import date as date_type
    q_date = date_type.fromisoformat(target_date) if target_date else date_type.today()
    try:
        db_manager.clear_previous_date_failed(q_date)
    except Exception:
        pass

    task = {
        "task_id": task_id,
        "status": "running",
        "total": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_code": "",
        "current_name": "",
        "start_time": time.time(),
        "end_time": None,
        "errors": [],
        "date": target_date,
        "cancelled": False,
        "paused": False,
        "waiting_retry": False,
        "retry_countdown": 0,
    }

    with _batch_download_lock:
        _batch_download_tasks[task_id] = task

    thread = threading.Thread(
        target=_run_batch_download,
        args=(task_id, target_date, max_workers, force),
        daemon=True,
    )
    thread.start()

    logger.info(f"启动批量下载: {task_id}, 日期={target_date}, 线程数={max_workers}")

    return BatchDownloadStatus(
        task_id=task_id,
        status="running",
        total=0,
        completed=0,
        failed=0,
        skipped=0,
        current_code="",
        current_name="",
        elapsed_seconds=0.0,
        errors=[],
        date=target_date,
    )


@router.get(
    "/batch-download/status",
    response_model=BatchDownloadStatus,
    summary="查询批量下载进度",
)
def get_batch_download_status(
    task_id: Optional[str] = Query(None, description="任务ID，不传则返回最近一个任务"),
) -> BatchDownloadStatus:
    with _batch_download_lock:
        if not task_id:
            # 返回最近的任务
            if not _batch_download_tasks:
                return BatchDownloadStatus(status="idle")
            task_id = list(_batch_download_tasks.keys())[-1]

        task = _batch_download_tasks.get(task_id)
        if not task:
            return BatchDownloadStatus(task_id=task_id, status="idle")

        elapsed = time.time() - task["start_time"] if task["start_time"] else 0
        if task.get("end_time"):
            elapsed = task["end_time"] - task["start_time"]

        return BatchDownloadStatus(
            task_id=task_id,
            status=task["status"],
            total=task["total"],
            completed=task["completed"],
            failed=task["failed"],
            skipped=task.get("skipped", 0),
            current_code=task["current_code"],
            current_name=task["current_name"],
            elapsed_seconds=round(elapsed, 1),
            errors=task.get("errors", [])[-20:],
            date=task.get("date", ""),
            paused=task.get("paused", False),
            waiting_retry=task.get("waiting_retry", False),
            retry_countdown=task.get("retry_countdown", 0),
        )


@router.post(
    "/batch-download/cancel",
    summary="取消批量下载任务",
)
def cancel_batch_download(
    task_id: str = Body(..., embed=True),
) -> dict:
    with _batch_download_lock:
        task = _batch_download_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task["status"] != "running":
            raise HTTPException(status_code=400, detail="任务已结束")
        task["cancelled"] = True
        task["status"] = "cancelled"
        task["end_time"] = time.time()
    return {"message": "已发送取消请求", "task_id": task_id}


@router.post(
    "/batch-download/pause",
    summary="暂停/继续批量下载任务",
)
def toggle_pause_batch_download(
    task_id: str = Body(..., embed=True),
) -> dict:
    with _batch_download_lock:
        task = _batch_download_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task["status"] != "running":
            raise HTTPException(status_code=400, detail="任务已结束，无法暂停/继续")
        task["paused"] = not task.get("paused", False)
        new_state = task["paused"]
    action = "已暂停" if new_state else "已继续"
    return {"message": action, "task_id": task_id, "paused": new_state}


def _run_batch_download_retry(task_id: str, target_date: str, max_workers: int, codes: list):
    """后台线程：重试下载失败标的的分时数据

    与 _run_batch_download 类似，但只处理指定的 codes 列表，
    重试成功后清除 batch_download_failed 表中的记录。
    """
    from datetime import date as date_type

    with _batch_download_lock:
        task = _batch_download_tasks.get(task_id)
        if not task:
            return

    try:
        from stock_selector.stock_pool import get_all_stock_code_name_pairs

        stock_pairs = get_all_stock_code_name_pairs(force_refresh=False)
        code_to_name = {code: name for code, name in stock_pairs}

        with _batch_download_lock:
            task["total"] = len(codes)
            task["current_code"] = ""
            task["current_name"] = ""

        from src.storage import DatabaseManager as DBManager

        db = DBManager.get_instance()
        q_date = date_type.fromisoformat(target_date) if target_date else date_type.today()

        processed = 0
        failed = 0
        skipped = 0

        batch_size = 20
        from concurrent.futures import ThreadPoolExecutor, as_completed

        for i in range(0, len(codes), batch_size):
            # 检查是否被暂停/取消
            while True:
                with _batch_download_lock:
                    if task.get("cancelled"):
                        task["status"] = "cancelled"
                        task["end_time"] = time.time()
                        return
                    paused = task.get("paused", False)
                if not paused:
                    break
                time.sleep(1)

            with _batch_download_lock:
                if task.get("cancelled"):
                    task["status"] = "cancelled"
                    task["end_time"] = time.time()
                    return

            batch = codes[i : i + batch_size]

            def _fetch_one(code: str):
                nonlocal db, q_date
                try:
                    klines = _get_intraday_klines(code, target_date)
                    if not klines:
                        return code, False, "无分时数据"

                    _inject_avg_price(klines)
                    actual_date_str = _extract_date_from_klines(klines, q_date.isoformat())
                    try:
                        actual_q_date = date_type.fromisoformat(actual_date_str)
                    except (ValueError, TypeError):
                        actual_q_date = q_date
                    db.save_intraday_klines(code, actual_q_date, klines)
                    _set_cached_klines(code, klines)
                    # 重试成功，清除失败记录
                    try:
                        db.clear_batch_download_failed(code, q_date)
                    except Exception:
                        pass
                    return code, True, False
                except Exception as e:
                    return code, False, str(e)[:200]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_one, code): code for code in batch}
                from concurrent.futures import TimeoutError as FuturesTimeoutError

                try:
                    for future in as_completed(futures, timeout=30):
                        code = futures[future]
                        try:
                            code, ok, info = future.result(timeout=5)
                        except Exception as e:
                            ok, info = False, str(e)[:200]

                        if ok:
                            processed += 1
                        else:
                            failed += 1
                            with _batch_download_lock:
                                if len(task["errors"]) < 20:
                                    task["errors"].append({"code": code, "error": str(info)})

                        with _batch_download_lock:
                            task["completed"] = processed + skipped
                            task["failed"] = failed
                            task["current_code"] = code
                            task["current_name"] = code_to_name.get(code, "")
                            task["elapsed_seconds"] = time.time() - task["start_time"]

                except FuturesTimeoutError:
                    for future in futures:
                        future.cancel()
                    with _batch_download_lock:
                        task["failed"] = failed + len(futures)

            time.sleep(0.3)

        with _batch_download_lock:
            task["status"] = "completed"
            task["end_time"] = time.time()
            task["elapsed_seconds"] = task["end_time"] - task["start_time"]

        logger.info(
            f"失败重试完成: {task_id}, 成功{processed}, 失败{failed}, "
            f"耗时{task['elapsed_seconds']:.1f}s"
        )

    except Exception as e:
        logger.error(f"失败重试异常: {task_id}: {e}", exc_info=True)
        with _batch_download_lock:
            task["status"] = "failed"
            task["end_time"] = time.time()


@router.post(
    "/batch-download/retry-failed",
    response_model=BatchDownloadStatus,
    summary="重试批量下载失败的标的",
)
def retry_batch_download_failed(
    date: Optional[str] = Body(None, description="目标日期 YYYY-MM-DD，默认当日"),
    max_workers: int = Body(4, ge=1, le=10, description="并行线程数"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BatchDownloadStatus:
    """从 batch_download_failed 表读取失败标的，重新下载分时数据"""
    import threading
    import uuid as uuid_module

    from datetime import date as date_type

    # 检查是否已有正在运行的任务
    with _batch_download_lock:
        for tid, t in _batch_download_tasks.items():
            if t.get("status") == "running":
                return BatchDownloadStatus(
                    task_id=tid,
                    status="running",
                    total=t["total"],
                    completed=t["completed"],
                    failed=t["failed"],
                    skipped=t.get("skipped", 0),
                    current_code=t.get("current_code", ""),
                    current_name=t.get("current_name", ""),
                    elapsed_seconds=time.time() - t.get("start_time", time.time()),
                    errors=t.get("errors", []),
                    date=t.get("date", ""),
                    paused=t.get("paused", False),
                    waiting_retry=t.get("waiting_retry", False),
                    retry_countdown=t.get("retry_countdown", 0),
                )

    target_date = date or datetime.now().strftime("%Y-%m-%d")
    q_date = date_type.fromisoformat(target_date) if target_date else date_type.today()

    # 查询失败列表
    failed_records = db_manager.get_batch_download_failed(q_date)
    if not failed_records:
        return BatchDownloadStatus(
            task_id="",
            status="completed",
            total=0,
            completed=0,
            failed=0,
            date=target_date,
        )

    failed_codes = [r["code"] for r in failed_records]
    task_id = uuid_module.uuid4().hex[:12]

    task = {
        "task_id": task_id,
        "status": "running",
        "total": len(failed_codes),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_code": "",
        "current_name": "",
        "start_time": time.time(),
        "end_time": None,
        "errors": [],
        "date": target_date,
        "cancelled": False,
        "paused": False,
        "waiting_retry": False,
        "retry_countdown": 0,
    }

    with _batch_download_lock:
        _batch_download_tasks[task_id] = task

    thread = threading.Thread(
        target=_run_batch_download_retry,
        args=(task_id, target_date, max_workers, failed_codes),
        daemon=True,
    )
    thread.start()

    logger.info(f"启动失败重试: {task_id}, 日期={target_date}, 标的数={len(failed_codes)}")

    return BatchDownloadStatus(
        task_id=task_id,
        status="running",
        total=len(failed_codes),
        completed=0,
        failed=0,
        date=target_date,
    )


@router.get(
    "/batch-download/failed-list",
    summary="查询批量下载失败列表",
)
def get_batch_download_failed_list(
    date: Optional[str] = Query(None, description="目标日期 YYYY-MM-DD，默认当日"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> dict:
    """查询批量下载失败的标的列表"""
    from datetime import date as date_type

    target_date = date or datetime.now().strftime("%Y-%m-%d")
    q_date = date_type.fromisoformat(target_date) if target_date else date_type.today()

    failed_records = db_manager.get_batch_download_failed(q_date)
    return {
        "date": target_date,
        "failed_list": failed_records,
        "count": len(failed_records),
    }


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
        with db_manager.session_scope() as session:
            result = session.execute(text(
                "SELECT id, stock_code, stock_name, search_date, search_time "
                "FROM intraday_search_history "
                "ORDER BY search_time DESC "
                "LIMIT :limit"
            ), {"limit": limit})
            rows = result.fetchall()

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

        with db_manager.session_scope() as session:
            result = session.execute(text("SELECT COUNT(*) FROM intraday_search_history"))
            total_count = result.scalar() or 0

        return SearchHistoryResponse(items=items, total=total_count)

    except Exception as e:
        logger.error(f"获取搜索历史失败: {e}")
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

        with db_manager.session_scope() as session:
            session.execute(text(
                "CREATE TABLE IF NOT EXISTS intraday_search_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "stock_code TEXT NOT NULL, "
                "stock_name TEXT DEFAULT '', "
                "search_date TEXT DEFAULT '', "
                "search_time TEXT DEFAULT ''"
                ")"
            ))

            session.execute(
                text("DELETE FROM intraday_search_history WHERE stock_code = :stock_code"),
                {"stock_code": request.stock_code},
            )

            session.execute(text(
                "INSERT INTO intraday_search_history (stock_code, stock_name, search_date, search_time) "
                "VALUES (:stock_code, :stock_name, :search_date, :search_time)"
            ), {
                "stock_code": request.stock_code,
                "stock_name": request.stock_name or "",
                "search_date": request.date or "",
                "search_time": now,
            })

            result = session.execute(text("SELECT last_insert_rowid()"))
            item_id = result.scalar() or 0

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
        with db_manager.session_scope() as session:
            session.execute(
                text("DELETE FROM intraday_search_history WHERE id = :hid"),
                {"hid": history_id},
            )
        return DeleteHistoryResponse(success=True, message="删除成功")
    except Exception as e:
        logger.error(f"删除搜索历史失败: {e}")
        return DeleteHistoryResponse(success=False, message=str(e))


@router.put(
    "/history/{history_id}/timestamp",
    response_model=SearchHistoryItem,
    summary="更新分时搜索历史时间戳",
    description="将指定记录的时间戳更新为当前时间，使其在列表中置顶显示",
)
def update_search_history_timestamp(
    history_id: int,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> SearchHistoryItem:
    """更新时间戳使记录置顶"""
    try:
        now = datetime.now().isoformat()
        with db_manager.session_scope() as session:
            session.execute(
                text("UPDATE intraday_search_history SET search_time = :t WHERE id = :hid"),
                {"t": now, "hid": history_id},
            )
            result = session.execute(
                text("SELECT id, stock_code, stock_name, search_date, search_time "
                     "FROM intraday_search_history WHERE id = :hid"),
                {"hid": history_id},
            )
            row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {history_id} 的记录")

        return SearchHistoryItem(
            id=row[0],
            stock_code=row[1],
            stock_name=row[2] or "",
            date=row[3] or "",
            search_time=row[4] or "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新时间戳失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
