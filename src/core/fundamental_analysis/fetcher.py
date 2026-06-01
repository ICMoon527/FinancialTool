# -*- coding: utf-8 -*-
"""
===================================
财报数据抓取器
===================================

职责：
1. 从 akshare 获取公司财报数据（资产负债表、利润表、现金流量表、财务摘要）
2. 从 tushare 获取财报数据作为备选
3. 标准化数据格式，统一返回 DataFrame
4. 实现缓存机制，避免重复请求

数据源说明：
- 财务摘要: akshare stock_financial_abstract_ths（同花顺）
- 利润表: akshare stock_financial_benefit_ths（同花顺）
- 资产负债表: akshare stock_financial_debt_ths（同花顺）
- 现金流量表: akshare stock_financial_cash_ths（同花顺）
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from src.config import get_config

logger = logging.getLogger(__name__)

# 财报数据缓存（模块级，避免同一进程内重复请求）
_financial_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 3600  # 财报数据缓存 1 小时


def _get_cache_key(stock_code: str, report_type: str) -> str:
    return f"{stock_code}_{report_type}"


def _get_from_cache(cache_key: str) -> Optional[Any]:
    if cache_key in _financial_cache:
        entry = _financial_cache[cache_key]
        if time.time() - entry["timestamp"] < entry["ttl"]:
            logger.debug(f"缓存命中: {cache_key}")
            return entry["data"]
        else:
            del _financial_cache[cache_key]
    return None


def _set_to_cache(cache_key: str, data: Any, ttl: int = _CACHE_TTL) -> None:
    _financial_cache[cache_key] = {
        "data": data,
        "timestamp": time.time(),
        "ttl": ttl,
    }


def _parse_ths_value(value: Any) -> Optional[float]:
    """
    解析同花顺 THS 接口返回的字符串值为 float

    处理格式：
    - "281.54亿" → 281.54 × 1e8
    - "3234.46万" → 3234.46 × 1e4
    - "23.38%"   → 23.38（保持百分比数值）
    - "21.76"    → 21.76（纯数字）
    - "False"/None/NaN → None

    Args:
        value: 原始值（通常为字符串）

    Returns:
        解析后的浮点数，失败返回 None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), 2)

    if not isinstance(value, str):
        try:
            v = float(value)
            return round(v, 2)
        except (ValueError, TypeError):
            return None

    s = value.strip()
    if not s or s.lower() in ("false", "true", "none", "nan", "-"):
        return None

    # 匹配 "数字 + 亿/万" 或 "数字 + %"
    multiplier = 1
    if "亿" in s:
        multiplier = 1e8
        s = s.replace("亿", "")
    elif "万" in s:
        multiplier = 1e4
        s = s.replace("万", "")
    elif "元" in s:
        s = s.replace("元", "")

    # 去除百分号（保持数值不变，即 "23.38%" → 23.38）
    s = s.replace("%", "")

    try:
        v = float(s) * multiplier
        return round(v, 2)
    except (ValueError, TypeError):
        return None


def _normalize_ths_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 THS 接口返回的 DataFrame：
    1. 将字符串值解析为 float
    2. 按报告期降序排列（最新在前）

    Args:
        df: 原始 THS DataFrame

    Returns:
        标准化后的 DataFrame
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 对除 "报告期"、"报表核心指标"、"报表全部指标" 外的所有列进行值解析
    skip_cols = {"报告期", "报表核心指标", "报表全部指标"}
    for col in df.columns:
        if col in skip_cols:
            continue
        df[col] = df[col].apply(_parse_ths_value)

    # 按报告期降序排列
    if "报告期" in df.columns:
        df["报告期"] = pd.to_datetime(df["报告期"], errors="coerce")
        df = df.sort_values("报告期", ascending=False).reset_index(drop=True)

    return df


class FinancialReportFetcher:
    """
    财报数据抓取器

    数据源：akshare 同花顺 THS 接口（免费，无需 Token）。
    回退方案：tushare Pro API（需要配置 token）。

    返回的财报数据统一为 pandas DataFrame 格式，
    数值字段已解析为 float，按报告期降序排列（最新在前）。
    """

    def __init__(self):
        self.config = get_config()
        self._tushare_pro = None

    def _get_tushare_pro(self):
        if self._tushare_pro is None:
            token = self.config.tushare_token
            if not token:
                logger.warning("tushare token 未配置，将跳过 tushare 数据源")
                return None
            try:
                import tushare as ts
                self._tushare_pro = ts.pro_api(token)
                logger.info("tushare pro 接口初始化成功")
            except Exception as e:
                logger.error(f"tushare pro 初始化失败: {e}")
                return None
        return self._tushare_pro

    # ============================================================
    # 财务摘要
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def fetch_financial_abstract(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取公司财务摘要数据

        包含 ROE、ROA、毛利率、净利率、营业收入、净利润等核心财务指标。

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            已标准化的财务摘要 DataFrame，获取失败返回 None
        """
        cache_key = _get_cache_key(stock_code, "abstract")
        cached = _get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            logger.info(f"正在获取 {stock_code} 的财务摘要数据...")
            time.sleep(2 + (hash(stock_code) % 3))

            df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按报告期")
            if df is None or df.empty:
                logger.warning(f"{stock_code} 财务摘要数据为空")
                return None

            df = _normalize_ths_dataframe(df)
            logger.info(f"{stock_code} 财务摘要数据获取成功，共 {len(df)} 条记录")
            _set_to_cache(cache_key, df)
            return df

        except ImportError:
            logger.error("akshare 未安装，无法获取财务摘要数据")
            return None
        except Exception as e:
            logger.error(f"获取 {stock_code} 财务摘要失败: {e}")
            return None

    # ============================================================
    # 利润表
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def fetch_income_statement(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取公司利润表（同花顺 THS 接口）

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            已标准化的利润表 DataFrame，获取失败返回 None
        """
        cache_key = _get_cache_key(stock_code, "income")
        cached = _get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            logger.info(f"正在获取 {stock_code} 的利润表数据（THS）...")
            time.sleep(2 + (hash(stock_code) % 3))

            df = ak.stock_financial_benefit_ths(symbol=stock_code)
            if df is None or df.empty:
                logger.warning(f"{stock_code} 利润表数据为空，尝试 tushare...")
                return self._fetch_income_from_tushare(stock_code)

            df = _normalize_ths_dataframe(df)
            logger.info(f"{stock_code} 利润表数据获取成功，共 {len(df)} 条记录")
            _set_to_cache(cache_key, df)
            return df

        except ImportError:
            logger.error("akshare 未安装")
            return None
        except Exception as e:
            logger.error(f"获取 {stock_code} 利润表失败: {e}")
            return self._fetch_income_from_tushare(stock_code)

    def _fetch_income_from_tushare(self, stock_code: str) -> Optional[pd.DataFrame]:
        pro = self._get_tushare_pro()
        if pro is None:
            return None

        try:
            ts_code = self._to_tushare_code(stock_code)
            df = pro.income(ts_code=ts_code, fields="ts_code,ann_date,f_ann_date,end_date,report_type,"
                            "total_revenue,revenue,oper_cost,operate_profit,total_profit,income_tax,n_income,"
                            "basic_eps,diluted_eps")
            if df is not None and not df.empty:
                logger.info(f"{stock_code} 从 tushare 获取利润表成功，共 {len(df)} 条记录")
                return df
        except Exception as e:
            logger.error(f"从 tushare 获取 {stock_code} 利润表失败: {e}")
        return None

    # ============================================================
    # 资产负债表
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def fetch_balance_sheet(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取公司资产负债表（同花顺 THS 接口）

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            已标准化的资产负债表 DataFrame，获取失败返回 None
        """
        cache_key = _get_cache_key(stock_code, "balance")
        cached = _get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            logger.info(f"正在获取 {stock_code} 的资产负债表数据（THS）...")
            time.sleep(2 + (hash(stock_code) % 3))

            df = ak.stock_financial_debt_ths(symbol=stock_code)
            if df is None or df.empty:
                logger.warning(f"{stock_code} 资产负债表数据为空，尝试 tushare...")
                return self._fetch_balance_from_tushare(stock_code)

            df = _normalize_ths_dataframe(df)
            logger.info(f"{stock_code} 资产负债表数据获取成功，共 {len(df)} 条记录")
            _set_to_cache(cache_key, df)
            return df

        except ImportError:
            logger.error("akshare 未安装")
            return None
        except Exception as e:
            logger.error(f"获取 {stock_code} 资产负债表失败: {e}")
            return self._fetch_balance_from_tushare(stock_code)

    def _fetch_balance_from_tushare(self, stock_code: str) -> Optional[pd.DataFrame]:
        pro = self._get_tushare_pro()
        if pro is None:
            return None

        try:
            ts_code = self._to_tushare_code(stock_code)
            df = pro.balancesheet(ts_code=ts_code, fields="ts_code,ann_date,f_ann_date,end_date,report_type,"
                                  "total_assets,total_liab,total_hldr_eqy_exc_min_int,"
                                  "total_cur_assets,total_cur_liab,inventories,"
                                  "accounts_receiv,accounts_payable")
            if df is not None and not df.empty:
                logger.info(f"{stock_code} 从 tushare 获取资产负债表成功，共 {len(df)} 条记录")
                return df
        except Exception as e:
            logger.error(f"从 tushare 获取 {stock_code} 资产负债表失败: {e}")
        return None

    # ============================================================
    # 现金流量表
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def fetch_cash_flow_statement(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取公司现金流量表（同花顺 THS 接口）

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            已标准化的现金流量表 DataFrame，获取失败返回 None
        """
        cache_key = _get_cache_key(stock_code, "cashflow")
        cached = _get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import akshare as ak

            logger.info(f"正在获取 {stock_code} 的现金流量表数据（THS）...")
            time.sleep(2 + (hash(stock_code) % 3))

            df = ak.stock_financial_cash_ths(symbol=stock_code)
            if df is None or df.empty:
                logger.warning(f"{stock_code} 现金流量表数据为空，尝试 tushare...")
                return self._fetch_cashflow_from_tushare(stock_code)

            df = _normalize_ths_dataframe(df)
            logger.info(f"{stock_code} 现金流量表数据获取成功，共 {len(df)} 条记录")
            _set_to_cache(cache_key, df)
            return df

        except ImportError:
            logger.error("akshare 未安装")
            return None
        except Exception as e:
            logger.error(f"获取 {stock_code} 现金流量表失败: {e}")
            return self._fetch_cashflow_from_tushare(stock_code)

    def _fetch_cashflow_from_tushare(self, stock_code: str) -> Optional[pd.DataFrame]:
        pro = self._get_tushare_pro()
        if pro is None:
            return None

        try:
            ts_code = self._to_tushare_code(stock_code)
            df = pro.cashflow(ts_code=ts_code, fields="ts_code,ann_date,f_ann_date,end_date,report_type,"
                              "n_cashflow_act,c_fr_sale_sg,free_cashflow")
            if df is not None and not df.empty:
                logger.info(f"{stock_code} 从 tushare 获取现金流量表成功，共 {len(df)} 条记录")
                return df
        except Exception as e:
            logger.error(f"从 tushare 获取 {stock_code} 现金流量表失败: {e}")
        return None

    # ============================================================
    # 批量获取
    # ============================================================

    def fetch_all_reports(self, stock_code: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        获取公司的全部财报数据

        Args:
            stock_code: 股票代码，如 '600519'

        Returns:
            包含各报表的字典：
            {
                "abstract": DataFrame or None,
                "income": DataFrame or None,
                "balance": DataFrame or None,
                "cashflow": DataFrame or None,
            }
        """
        logger.info(f"开始批量获取 {stock_code} 的财报数据...")

        abstract = self.fetch_financial_abstract(stock_code)
        income = self.fetch_income_statement(stock_code)
        balance = self.fetch_balance_sheet(stock_code)
        cashflow = self.fetch_cash_flow_statement(stock_code)

        result = {
            "abstract": abstract,
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
        }

        success_count = sum(1 for v in result.values() if v is not None)
        logger.info(f"{stock_code} 财报数据获取完成: {success_count}/4 成功")

        return result

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _to_tushare_code(stock_code: str) -> str:
        code = stock_code.strip()
        if code.startswith("6"):
            return f"{code}.SH"
        elif code.startswith(("0", "3")):
            return f"{code}.SZ"
        else:
            return code

    @staticmethod
    def clear_cache() -> None:
        _financial_cache.clear()
        logger.info("财报数据缓存已清空")