# -*- coding: utf-8 -*-
"""
分时做T API 端点

提供分时K线数据获取和做T信号计算服务。
支持多数据源 fallback 机制，提高数据获取成功率。
"""

import logging
import os
import traceback
import yaml
import json
import threading
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Optional, Callable, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from requests.exceptions import (
    ConnectionError,
    ReadTimeout,
    ConnectTimeout,
    RequestException,
)
from sqlalchemy import text

from api.deps import get_database_manager
from api.v1.schemas.intraday import (
    BatchStatusRequest,
    BatchStatusResponse,
    DeleteHistoryResponse,
    IndicatorLine,
    IndicatorLinePoint,
    IndicatorSubChart,
    IntradayDataResponse,
    IntradayKlinePoint,
    IntradaySignal,
    SearchHistoryItem,
    SearchHistoryRequest,
    SearchHistoryResponse,
    StockSnapshot,
    WeightContribution,
)
from api.v1.schemas.common import ErrorResponse
from src.storage import DatabaseManager
from data_provider import DataFetcherManager
from watchdog.strategies.intraday_t0_strategy import IntradayIndicatorEngine

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "watchdog" / "strategies" / "intraday_t0_config.yaml"


def _load_indicator_config() -> dict:
    """每次调用时重新加载YAML配置，支持热更新"""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f'加载指标配置文件失败，使用默认参数: {e}')
    return {}

router = APIRouter()


def _normalize_stock_code(code: str) -> str:
    """标准化股票代码，去除前缀 zh_ / sh / sz 等"""
    code = code.strip().upper()
    for prefix in ["ZH_", "SH.", "SZ.", "SH", "SZ"]:
        if code.startswith(prefix):
            code = code[len(prefix) :]
    return code


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
            logger.info(f"尝试使用 {source_name} 获取{data_type}...")
            result = fetch_func()

            # 检查结果是否有效
            if result is not None and (not hasattr(result, "empty") or not result.empty):
                logger.info(f"使用 {source_name} 成功获取{data_type}")
                return result
            else:
                logger.warning(f"{source_name} 返回空数据，尝试下一个数据源")
                continue

        except Exception as e:
            logger.warning(f"{source_name} 获取{data_type}失败: {type(e).__name__}: {e}")
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
    3. 东方财富 (stock_zh_a_hist_min_em)
    4. 其他 akshare 分时接口

    Args:
        stock_code: 股票代码（如 000001，600519）
        date_str: 日期字符串 YYYYMMDD 或 YYYY-MM-DD，None 为当日

    Returns:
        K线数据列表，每项为包含 Open/High/Low/Close/Volume/timestamp 的字典
    """
    try:
        import akshare as ak
        import requests

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

            market = "sh" if code.startswith("6") else "sz"
            url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={market}{code}"
            r = requests.get(url, timeout=15)
            data = r.json()
            point_list = (
                data.get("data", {})
                .get(f"{market}{code}", {})
                .get("data", {})
                .get("data", [])
            )
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
                    "timestamp": f"{target_date}T{hour}:{minute}:00",
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": per_vol,
                })
            if not records:
                return None
            return pd.DataFrame(records)

        fetch_functions.append(("腾讯财经1分钟分时", fetch_tencent_1min))

        # 数据源 2: 新浪财经5分钟K线 (标准OHLC，降级备选)
        def fetch_sina_5min():
            import pandas as pd

            market = "sh" if code.startswith("6") else "sz"
            symbol_str = f"{market}{code}"
            url = (
                "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                "CN_MarketData.getKLineData"
                f"?symbol={symbol_str}&scale=5&ma=no&datalen=250"
            )
            headers = {"Referer": "https://finance.sina.com.cn"}
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            if not data or not isinstance(data, list):
                return None
            records = []
            for item in data:
                day_str = item.get("day", "")
                if not day_str.startswith(target_date):
                    continue
                time_part = day_str.split()[-1] if " " in day_str else ""
                records.append({
                    "Open": float(item["open"]),
                    "High": float(item["high"]),
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                    "Volume": float(item["volume"]),
                    "timestamp": f"{target_date}T{time_part}",
                })
            if not records:
                return None
            return pd.DataFrame(records)

        fetch_functions.append(("新浪财经5分钟K线", fetch_sina_5min))

        # 数据源 3: 东方财富 (stock_zh_a_hist_min_em)
        if hasattr(ak, 'stock_zh_a_hist_min_em'):
            def fetch_em():
                df = ak.stock_zh_a_hist_min_em(
                    symbol=code,
                    period="1",
                    adjust="",
                )
                return df

            fetch_functions.append(("东方财富接口", fetch_em))

        # 数据源 4: 尝试 akshare 的其他可能的分时接口
        for attr_name in dir(ak):
            if 'hist_min' in attr_name and (attr_name.startswith('stock') or attr_name.startswith('ak')):
                try:
                    func = getattr(ak, attr_name)

                    def create_fetch_func(f):
                        def fetch_func():
                            try:
                                return f(symbol=code, period="1", adjust="")
                            except:
                                try:
                                    return f(symbol=code)
                                except:
                                    try:
                                        return f(code)
                                    except:
                                        raise
                        return fetch_func

                    fetch_functions.append((attr_name, create_fetch_func(func)))
                except:
                    continue

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


def _run_t0_strategy(klines: list, reference_lines: list = None) -> tuple:
    """对分时K线数据运行做T策略，生成信号列表

    Args:
        klines: 分时K线字典列表
        reference_lines: 日线级参考线列表

    Returns:
        (signals_list, signal_summary_dict)
    """
    try:
        from watchdog.strategies.intraday_t0_strategy import IntradayT0Strategy

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "watchdog", "strategies", "intraday_t0_config.yaml"
        )
        config_path = os.path.normpath(config_path)
        strategy = IntradayT0Strategy(stock_code="temp", stock_name="", config_path=config_path)

        ref_lines_raw = reference_lines or []
        ref_lines = [
            {"id": rl.id if hasattr(rl, "id") else rl.get("id", ""),
             "price": rl.price if hasattr(rl, "price") else rl.get("price", 0)}
            for rl in ref_lines_raw
        ]

        signals = []
        for kline in klines:
            sig = strategy.feed_kline(kline, ref_lines)
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

        return result_signals, summary

    except ImportError as e:
        logger.warning(f"导入做T策略失败: {e}，返回空信号")
        return [], {
            'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0,
            'strong_signals': 0, 'medium_signals': 0, 'weak_signals': 0,
            'simulated_return_pct': 0.0,
        }
    except Exception as e:
        logger.error(f"运行做T策略失败: {e}", exc_info=True)
        return [], {
            'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0,
            'strong_signals': 0, 'medium_signals': 0, 'weak_signals': 0,
            'simulated_return_pct': 0.0, 'error': str(e),
        }


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


def _compute_main_in_out_signal(result) -> str:
    """计算主力进出信号：基于交叉与阈值判断正T/反T"""
    try:
        if 'main_in' not in result.columns or 'main_out' not in result.columns:
            return ''
        main_in = result['main_in']
        main_out = result['main_out']
        up = _cross_up(main_in, main_out)
        down = _cross_down(main_in, main_out)
        in_vals = main_in.dropna()
        out_vals = main_out.dropna()
        last_in = float(in_vals.iloc[-1]) if len(in_vals) > 0 else 50
        last_out = float(out_vals.iloc[-1]) if len(out_vals) > 0 else 50
        if up:
            recent_min = float(in_vals.iloc[-min(5, len(in_vals)):].min())
            if recent_min < 30:
                return '反T买回 ↑'
            return '正T买入 ↑'
        if down:
            recent_max = float(in_vals.iloc[-min(5, len(in_vals)):].max())
            if recent_max > 70:
                return '反T卖出 ↓'
            return '正T卖出 ↓'
        if last_in > last_out:
            return '主力流入 ↗'
        else:
            return '主力流出 ↘'
    except Exception as e:
        logger.warning(f'计算主力进出信号失败: {e}')
        return ''


def _compute_cyw_signal(result) -> str:
    """计算CYW控盘信号：基于数值正负与CYW/MA交叉判断"""
    try:
        if 'CYW' not in result.columns or 'CYW_MA' not in result.columns:
            return ''
        cyw = result['CYW']
        cyw_ma = result['CYW_MA']
        up = _cross_up(cyw, cyw_ma)
        down = _cross_down(cyw, cyw_ma)
        cyw_vals = cyw.dropna()
        last = float(cyw_vals.iloc[-1]) if len(cyw_vals) > 0 else 0
        if up:
            prefix = '控盘中' if last > 0 else '弱控盘'
            return f'{prefix} 买入 ↑'
        elif down:
            return '未控盘 卖出 ↓'
        elif last > 0:
            return '控盘中 ↗'
        else:
            return '未控盘 ↘'
    except Exception as e:
        logger.warning(f'计算CYW信号失败: {e}')
        return ''


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


def _generate_indicator_sub_charts(klines: list) -> list:
    """根据分时K线计算四大指标，生成子图数据

    Args:
        klines: 分时K线字典列表

    Returns:
        List[IndicatorSubChart] 四个指标的子图数据
    """
    try:
        import pandas as pd

        df = pd.DataFrame(klines)
        if 'Amount' not in df.columns:
            df['Amount'] = df['Close'] * df['Volume'] if 'Close' in df.columns and 'Volume' in df.columns else 0.0

        engine = IntradayIndicatorEngine(config=_load_indicator_config())
        result = engine.calculate_all(df)

        # 提取时间标签 "HH:MM"，尽量对齐K线的timestamp格式
        time_labels = []
        for _, row in result.iterrows():
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

        main_signal = _compute_main_in_out_signal(result)
        cyw_signal = _compute_cyw_signal(result)
        macd_signal = _compute_macd_signal(result)
        rsi_signal = _compute_rsi_signal(result, overbought=engine.rsi_overbought, oversold=engine.rsi_oversold)

        # ── 1. 主力吸筹 ──
        if 'absorption' in result.columns:
            absorption_data = []
            for i, v in enumerate(result['absorption']):
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
        if all(c in result.columns for c in ['DIF', 'DEA', 'MACD_Bar']):
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['DIF'].iloc[i]):
                    macd_dif_data.append(IndicatorLinePoint(time=tl, value=round(float(result['DIF'].iloc[i]), 4)))
                if not pd.isna(result['DEA'].iloc[i]):
                    macd_dea_data.append(IndicatorLinePoint(time=tl, value=round(float(result['DEA'].iloc[i]), 4)))
                if not pd.isna(result['MACD_Bar'].iloc[i]):
                    macd_bar_data.append(IndicatorLinePoint(time=tl, value=round(float(result['MACD_Bar'].iloc[i]), 4)))
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
                )
            )

        # ── 3. RSI ──
        rsi_data = []
        if 'RSI' in result.columns:
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['RSI'].iloc[i]):
                    rsi_data.append(IndicatorLinePoint(time=tl, value=round(float(result['RSI'].iloc[i]), 2)))
            rsi_ob_data = []
            rsi_os_data = []
            rsi_ob_val = engine.rsi_overbought
            rsi_os_val = engine.rsi_oversold
            for i in range(len(result)):
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

        # ── 4. 主力进出 ──
        main_in_data = []
        main_out_data = []
        in_out_line_data = []
        if all(c in result.columns for c in ['main_in', 'main_out', 'in_out_line']):
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['main_in'].iloc[i]):
                    main_in_data.append(IndicatorLinePoint(time=tl, value=round(float(result['main_in'].iloc[i]), 4)))
                if not pd.isna(result['main_out'].iloc[i]):
                    main_out_data.append(IndicatorLinePoint(time=tl, value=round(float(result['main_out'].iloc[i]), 4)))
                if not pd.isna(result['in_out_line'].iloc[i]):
                    in_out_line_data.append(IndicatorLinePoint(time=tl, value=round(float(result['in_out_line'].iloc[i]), 4)))
            sub_charts.append(
                IndicatorSubChart(
                    id="main_in_out",
                    label="主力进出",
                    height=110,
                    lines=[
                        IndicatorLine(name="main_in", label="主力流入", color="#FF4444", data=main_in_data),
                        IndicatorLine(name="main_out", label="主力流出", color="#4488FF", data=main_out_data),
                        IndicatorLine(name="in_out_line", label="进出线", color="#FFAA00", data=in_out_line_data),
                    ],
                    signal_text=main_signal,
                )
            )

        # ── 5. CYW 主力控盘 ──
        cyw_data = []
        cyw_ma_data = []
        if all(c in result.columns for c in ['CYW', 'CYW_MA']):
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['CYW'].iloc[i]):
                    cyw_data.append(IndicatorLinePoint(time=tl, value=round(float(result['CYW'].iloc[i]), 4)))
                if not pd.isna(result['CYW_MA'].iloc[i]):
                    cyw_ma_data.append(IndicatorLinePoint(time=tl, value=round(float(result['CYW_MA'].iloc[i]), 4)))
            sub_charts.append(
                IndicatorSubChart(
                    id="cyw",
                    label="CYW 控盘",
                    height=110,
                    lines=[
                        IndicatorLine(name="CYW", label="CYW", color="#44BBFF", data=cyw_data),
                        IndicatorLine(name="CYW_MA", label="MA", color="#EEEEEE", data=cyw_ma_data),
                    ],
                    signal_text=cyw_signal,
                )
            )

        # ── 6. 价格均线关系 ──
        ma5_data = []
        ma20_data = []
        close_data = []
        if all(c in result.columns for c in ['ma5', 'ma20']):
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['ma5'].iloc[i]):
                    ma5_data.append(IndicatorLinePoint(time=tl, value=round(float(result['ma5'].iloc[i]), 2)))
                if not pd.isna(result['ma20'].iloc[i]):
                    ma20_data.append(IndicatorLinePoint(time=tl, value=round(float(result['ma20'].iloc[i]), 2)))
                if 'Close' in result.columns and not pd.isna(result['Close'].iloc[i]):
                    close_data.append(IndicatorLinePoint(time=tl, value=round(float(result['Close'].iloc[i]), 2)))
            sub_charts.append(
                IndicatorSubChart(
                    id="price_ma",
                    label="价格均线",
                    height=110,
                    lines=[
                        IndicatorLine(name="close", label="价格", color="#FFFFFF", data=close_data),
                        IndicatorLine(name="ma5", label="MA5", color="#FF4444", data=ma5_data),
                        IndicatorLine(name="ma20", label="MA20", color="#44FF44", data=ma20_data),
                    ],
                    signal_text="",
                )
            )

        # ── 7. 均价偏离度 ──
        deviation_data = []
        if 'deviation_pct' in result.columns:
            for i in range(len(result)):
                tl = time_labels[i] if i < len(time_labels) else ''
                if not pd.isna(result['deviation_pct'].iloc[i]):
                    deviation_data.append(IndicatorLinePoint(time=tl, value=round(float(result['deviation_pct'].iloc[i]), 2)))
            sub_charts.append(
                IndicatorSubChart(
                    id="avg_price_deviation",
                    label="均价偏离",
                    height=110,
                    lines=[
                        IndicatorLine(name="deviation_pct", label="偏离%", color="#FFAA00", data=deviation_data),
                    ],
                    signal_text="",
                )
            )

        logger.info(f"生成 {len(sub_charts)} 个指标子图")
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

    logger.info(f"换手率全为0，尝试为 {code} 补充换手率数据...")

    from datetime import timedelta

    circulating_shares = None

    # ── Step 1: 从 StockBasic 获取流通股本 ──
    try:
        from src.services.turnover_service import TurnoverService
        turnover_service = TurnoverService(db_manager)
        stock_basic = turnover_service.get_stock_basic(code, end_date)
        if stock_basic and stock_basic.circulating_shares and stock_basic.circulating_shares > 0:
            circulating_shares = stock_basic.circulating_shares
            logger.info(f"从StockBasic获取 {code} 流通股本: {circulating_shares:.0f} 股")
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
        logger.info(f"回写 {code} 换手率到数据库: {count} 条")
    except Exception as e:
        logger.warning(f"回写 {code} 换手率到数据库失败: {e}")

    return daily_df, True


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

        # ── 今开 ──
        ref_lines.append(ReferenceLine(
            id='today_open', label='今开', price=round(today_open, 2),
            category='key_level', color='#4488FF', style='dotted', base_weight=1.5,
        ))

        # ── 今日高/今日低 ──
        ref_lines.append(ReferenceLine(
            id='today_high', label='今高', price=round(today_high, 2),
            category='key_level', color='#FF444488', style='dotted', base_weight=1.0,
        ))
        ref_lines.append(ReferenceLine(
            id='today_low', label='今低', price=round(today_low, 2),
            category='key_level', color='#44FF4488', style='dotted', base_weight=1.0,
        ))

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
                        logger.info("日线数据库无有效数据，跳过日线级参考线")
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
                            logger.info(
                                f"命中筹码分布缓存: {code} end_date={end_date}, "
                                f"upper={chip_upper_val}, lower={chip_lower_val}"
                            )
                        else:
                            if turnover_filled:
                                logger.info(f"换手率已补充，重新计算筹码分布: {code}")
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
                                        logger.info(
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
                        logger.info(
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
                        logger.info(f"ReferenceLineGenerator 生成 {len(daily_refs)} 条日线级参考线")

                        # ── 前高/前低 30个自然日HHV/LLV ──
                        _add_30day_extreme_lines(ref_lines, daily_df, q_date, today_low, today_high)

            except Exception as db_err:
                logger.warning(f"从数据库获取日线数据失败，跳过日线级参考线: {db_err}", exc_info=True)

        logger.info(f"生成 {len(ref_lines)} 条参考线")
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
        logger.info(f"30日极值: HHV={hhv_30:.2f}, LLV={llv_30:.2f}")
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
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> IntradayDataResponse:
    """获取分时K线数据和信号"""
    try:
        code = _normalize_stock_code(stock_code)
        date_str = date

        # 确定查询日期
        if date_str is None:
            q_date = date_type.today()
            date_str = q_date.isoformat()
        elif len(date_str) == 8:
            q_date = date_type(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            q_date = date_type.fromisoformat(date_str)

        is_today = q_date == date_type.today()

        # 获取分时K线
        klines = None
        if not is_today:
            # 历史日期优先从数据库加载
            klines = db_manager.load_intraday_klines(code, q_date)

        if not klines:
            klines = _get_intraday_klines(code, date_str)
        if not klines:
            raise HTTPException(status_code=404, detail={"error": "no_data", "message": "未获取到分时K线数据"})

        # 注入累计分时均价（策略需要 AvgPrice）
        _inject_avg_price(klines)

        # 异步存储分时K线到数据库
        _schedule_intraday_storage(code, klines, q_date, db_manager)

        # 计算并异步存储每日分时快照
        _schedule_daily_summary(code, klines, q_date, db_manager)

        # 计算日线级参考线（引力场模型需要）
        reference_lines = _compute_reference_lines(klines, code, db_manager, date_str)

        # 运行做T策略（含引力场）
        signals, summary = _run_t0_strategy(klines, reference_lines)

        # 生成指标子图数据（含新增价格均线和均价偏离）
        indicator_sub_charts = _generate_indicator_sub_charts(klines)

        # 构建K线响应
        kline_points = [IntradayKlinePoint(**k) for k in klines]

        # 获取股票名称（优先从 Stock Pool 数据库）
        try:
            fetcher_manager = DataFetcherManager()
            stock_name = fetcher_manager.get_stock_name(code, skip_realtime=True) or ""
        except Exception as e:
            logger.warning(f"获取股票名称失败 {code}: {e}")
            stock_name = ""

        # 加载前日快照用于指标预热
        warm_up = db_manager.load_previous_daily_summary(code, q_date)

        return IntradayDataResponse(
            stock_code=code,
            stock_name=stock_name,
            date=date_str,
            kline_data=kline_points,
            signals=signals,
            reference_lines=reference_lines,
            indicator_sub_charts=indicator_sub_charts,
            signal_summary=summary,
            warm_up_summary=warm_up,
        )

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
    }
    return result


def _schedule_intraday_storage(code: str, klines: list, q_date: date_type, db_manager: DatabaseManager):
    """异步调度分时K线存储任务（不阻塞主请求）
    注意：使用 daemon 线程，如果主请求结束但 uvicon 进程存活，线程将继续执行
    """
    def _store():
        try:
            count = db_manager.save_intraday_klines(code, q_date, klines)
            logger.info(f"后台存储完成: {code} {q_date}, {count} 条")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"后台存储分时K线失败 {code} {q_date}: {e}")

    t = threading.Thread(target=_store, daemon=True)
    t.start()
    logger.info(f"已调度后台存储: {code} {q_date}, 共 {len(klines)} 条K线")


def _schedule_daily_summary(code: str, klines: list, q_date: date_type, db_manager: DatabaseManager):
    """异步调度每日快照存储任务（不阻塞主请求）
    注意：使用 daemon 线程，如果主请求结束但 uvicon 进程存活，线程将继续执行
    """
    def _store():
        try:
            # 计算指标快照
            indicators = _compute_daily_summary_from_klines(klines)
            ok = db_manager.save_daily_summary(code, q_date, klines, indicators)
            logger.info(f"后台快照存储完成: {code} {q_date}, 成功={ok}")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"后台存储每日快照失败 {code} {q_date}: {e}")

    t = threading.Thread(target=_store, daemon=True)
    t.start()
    logger.info(f"已调度快照存储: {code} {q_date}")


# ---------- 批量状态查询 ----------


def _parse_sina_realtime_batch(stock_codes: list) -> dict:
    """通过新浪接口批量获取股票实时行情

    新浪接口支持逗号分隔的多股票批量查询：
    http://hq.sinajs.cn/list=sh600519,sz000001,sh600000

    Returns:
        {stock_code: {name, price, change_pct, open, high, low, timestamp}}
    """
    import re
    import requests as req

    symbols = []
    for code in stock_codes:
        code = code.strip()
        if code.startswith(("6", "5", "9")):
            symbols.append(f"sh{code}")
        else:
            symbols.append(f"sz{code}")

    if not symbols:
        return {}

    url = f"http://hq.sinajs.cn/list={','.join(symbols)}"
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
        code = match.group(2)
        data_str = match.group(3)
        fields = data_str.split(",")
        if len(fields) < 33:
            continue

        stock_code = code
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
        raw_snapshots = _parse_sina_realtime_batch(stock_codes)

        snapshots: dict = {}
        for code in stock_codes:
            if code in raw_snapshots:
                snapshots[code] = StockSnapshot(**raw_snapshots[code])
            else:
                snapshots[code] = StockSnapshot(stock_code=code)

        # 2. 检查当前展示股票是否有新数据
        current_code = body.current_code.strip()
        current_updated = False
        current_full_data = None

        if current_code and current_code in raw_snapshots:
            code = _normalize_stock_code(current_code)
            sn = raw_snapshots[current_code]

            # 获取分时K线，比对最新时间戳判断是否有新数据
            try:
                klines = _get_intraday_klines(code, None)
                if klines:
                    latest_kline_time = klines[-1].get("时间", "") if klines else ""
                    latest_snapshot_time = sn.get("timestamp", "")
                    # 比较：如果K线最新时间和快照时间接近，说明数据已同步
                    if latest_kline_time and latest_snapshot_time:
                        kt = latest_kline_time.replace(":", "")
                        st = latest_snapshot_time.replace(":", "").replace(" ", "")
                        if kt[:4] != st[:4]:
                            current_updated = True
                    else:
                        current_updated = True

                    if current_updated:
                        # 注入累计分时均价
                        _inject_avg_price(klines)
                        # 计算参考线
                        reference_lines = _compute_reference_lines(klines, code, db_manager, None)
                        # 运行做T策略
                        signals, summary = _run_t0_strategy(klines, reference_lines)
                        # 生成指标子图
                        indicator_sub_charts = _generate_indicator_sub_charts(klines)
                        # 构建K线响应
                        kline_points = [IntradayKlinePoint(**k) for k in klines]
                        # 确定日期
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        # 获取股票名称
                        stock_name = sn.get("stock_name", "")
                        if not stock_name:
                            try:
                                fetcher_manager = DataFetcherManager()
                                stock_name = fetcher_manager.get_stock_name(code, skip_realtime=True) or ""
                            except Exception:
                                pass

                        current_full_data = IntradayDataResponse(
                            stock_code=code,
                            stock_name=stock_name,
                            date=date_str,
                            kline_data=kline_points,
                            signals=signals,
                            reference_lines=reference_lines,
                            indicator_sub_charts=indicator_sub_charts,
                            signal_summary=summary,
                        )
            except Exception as e:
                logger.warning(f"检查当前股票 {current_code} 更新失败: {e}")

        return BatchStatusResponse(
            snapshots=snapshots,
            current_updated=current_updated,
            current_full_data=current_full_data,
        )

    except Exception as e:
        logger.error(f"批量状态查询失败: {e}", exc_info=True)
        return BatchStatusResponse(snapshots={}, current_updated=False)


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
