# -*- coding: utf-8 -*-
"""
250首板战法 —— 筛选股价首次突破250日均线且当日接近涨停的强势股。

必备条件（全部满足才匹配）:
  1. 股价上穿250日均线（昨日 close ≤ MA250，今日 close > MA250）
  2. 量比 > 1（腾讯快照优先，日线 volume 兜底）
  3. 涨幅 ≥ 9.9%（腾讯快照优先，日线 pct_chg 兜底）
  4. 换手率 5% - 15%（腾讯快照优先，日线 turnover_rate 兜底）
  5. 流通市值 10亿 - 100亿（仅腾讯快照可获取，不可用时放行）

加分条件（仅影响排名）:
  F1. 90自然日横盘（波幅 < 20%）→ +5分
  F2. 180自然日横盘（F1通过后被判断，波幅 < 20%）→ +5分
  G.  横盘期间有涨停板（pct_chg ≥ 9.9%）→ +5分

评分: 基础70 + 加分0~15 = 70~85

数据源: 腾讯行情接口 qt.gtimg.cn（批量拉取，2批 × 0.1s ≈ 0.2s，替代已不可用的东方财富 stock_zh_a_spot_em）
"""

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from stock_selector.base import (
    StockSelectorStrategy,
    StrategyMatch,
    StrategyMetadata,
    StrategyType,
)
from stock_selector.strategies.python_strategy_loader import register_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义异常：快照不可用时中止整个 Screen 流程
# ---------------------------------------------------------------------------
class SnapshotUnavailableError(Exception):
    """快照 API 连接失败，无法获取实时数据。"""


# ---------------------------------------------------------------------------
# 模块级快照缓存（腾讯 qt.gtimg.cn 批量行情，缓存20分钟）
# ---------------------------------------------------------------------------
_SNAPSHOT_CACHE: Dict[str, Any] = {"df": None, "ts": 0.0, "failed_ts": 0.0, "ttl": 1200}
_SNAPSHOT_LOCK = threading.Lock()

# 腾讯接口每批最大股票数（避免 Request-URI Too Large 414 错误）
_TENCENT_BATCH_SIZE = 500


def _get_all_stock_codes() -> List[str]:
    """从股票池获取全市场代码列表，转为腾讯格式（sh600519, sz000858）。"""
    try:
        from stock_selector.stock_pool import get_all_stock_code_name_pairs

        pairs = get_all_stock_code_name_pairs()
        codes = []
        for code, name in pairs:
            # 过滤 ST 股票
            if any(kw in name.upper() for kw in ["ST", "*ST", "SST", "S*ST"]):
                continue
            # 根据代码首字符判断交易所前缀
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            codes.append(f"{prefix}{code}")
        logger.debug("[250首板] 股票池共 %d 只（已过滤ST）", len(codes))
        return codes
    except Exception as e:
        logger.warning("[250首板] 获取股票池失败: %s", e)
        return []


def _get_spot_snapshot() -> Optional[pd.DataFrame]:
    """获取全市场实时快照 DataFrame（腾讯 qt.gtimg.cn），带 20 分钟缓存（含失败节流）。"""
    global _SNAPSHOT_CACHE

    # 快速路径：缓存命中 — 无锁检查，不记录日志
    now = time.time()
    if _SNAPSHOT_CACHE["df"] is not None and (now - _SNAPSHOT_CACHE["ts"]) < _SNAPSHOT_CACHE["ttl"]:
        return _SNAPSHOT_CACHE["df"]
    if _SNAPSHOT_CACHE["failed_ts"] > 0 and (now - _SNAPSHOT_CACHE["failed_ts"]) < 300:
        return None

    # 慢路径：需要拉取API — 加锁防止多线程并发请求
    with _SNAPSHOT_LOCK:
        # 双重检查：锁内再次确认缓存状态
        now = time.time()
        if _SNAPSHOT_CACHE["df"] is not None and (now - _SNAPSHOT_CACHE["ts"]) < _SNAPSHOT_CACHE["ttl"]:
            return _SNAPSHOT_CACHE["df"]
        if _SNAPSHOT_CACHE["failed_ts"] > 0 and (now - _SNAPSHOT_CACHE["failed_ts"]) < 300:
            return None

        is_first_attempt = _SNAPSHOT_CACHE["failed_ts"] == 0.0 or (now - _SNAPSHOT_CACHE["failed_ts"]) >= 300

        try:
            import requests
            import logging as _logging

            codes = _get_all_stock_codes()
            if not codes:
                raise SnapshotUnavailableError("无法获取股票代码列表，请检查股票池数据")

            logger.info("[250首板] 通过腾讯API获取全市场快照，共 %d 只股票 ...", len(codes))

            # 抑制 requests/urllib3 的 DEBUG 日志（避免打印完整URL）
            _logging.getLogger("urllib3").setLevel(_logging.WARNING)
            _logging.getLogger("requests").setLevel(_logging.WARNING)

            all_rows: List[Dict[str, Any]] = []
            for i in range(0, len(codes), _TENCENT_BATCH_SIZE):
                batch = codes[i : i + _TENCENT_BATCH_SIZE]
                url = "http://qt.gtimg.cn/q=" + ",".join(batch)
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()

                for line in resp.text.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("~")
                    if len(parts) < 47:
                        continue
                    raw_code = parts[2]
                    code = raw_code[2:] if len(raw_code) > 2 else raw_code
                    all_rows.append(
                        {
                            "代码": code,
                            "名称": parts[1],
                            "现价": _safe_float(parts[3]),
                            "涨跌幅": _safe_float(parts[32]),
                            "换手率": _safe_float(parts[38]),
                            "流通市值": _safe_float(parts[44]),  # 腾讯单位：亿
                            "量比": _safe_float(parts[46]),
                            "日期": parts[30][:8] if len(parts[30]) >= 8 else "",  # YYYYMMDD
                        }
                    )

            # 恢复默认日志级别
            _logging.getLogger("urllib3").setLevel(_logging.NOTSET)
            _logging.getLogger("requests").setLevel(_logging.NOTSET)

            df = pd.DataFrame(all_rows)
            if not df.empty and "代码" in df.columns:
                df = df.set_index("代码", drop=False)

            _SNAPSHOT_CACHE["df"] = df
            _SNAPSHOT_CACHE["ts"] = now
            _SNAPSHOT_CACHE["failed_ts"] = 0.0
            logger.info("[250首板] 快照获取成功，共 %d 只股票（股票池 %d 只）", len(df), len(codes))
            return df

        except Exception as e:
            logger.warning("[250首板] 快照拉取失败: %s", e)
            _SNAPSHOT_CACHE["failed_ts"] = now
            if is_first_attempt:
                raise SnapshotUnavailableError(f"快照 API 连接失败，无法获取实时行情数据：{e}") from e

    return None


def _get_stock_from_snapshot(code: str) -> Optional[Dict[str, Any]]:
    """从快照 DataFrame 中查找指定股票的数据行。"""
    df = _get_spot_snapshot()
    if df is None:
        return None
    try:
        if code in df.index:
            row = df.loc[code]
            snap_date_str = str(row.get("日期", ""))
            snap_date = (
                datetime.strptime(snap_date_str, "%Y%m%d").date()
                if snap_date_str and len(snap_date_str) == 8
                else None
            )
            return {
                "现价": _safe_float(row.get("现价")),
                "涨跌幅": _safe_float(row.get("涨跌幅")),
                "换手率": _safe_float(row.get("换手率")),
                "流通市值": _safe_float(row.get("流通市值")),
                "量比": _safe_float(row.get("量比")),
                "日期": snap_date,
            }
    except Exception:
        pass
    return None


def _safe_float(value: Any) -> Optional[float]:
    """安全转为 float，失败返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 策略类
# ---------------------------------------------------------------------------


@register_strategy
class FirstBoard250MAStrategy(StockSelectorStrategy):
    """
    250首板战法

    筛选条件:
      必备: MA250上穿 + 量比>1 + 涨幅≥9.9% + 换手率5-15% + 流通市值10-100亿
      加分: 90日横盘 + 180日横盘 + 横盘期间有涨停
    """

    def __init__(self):
        metadata = StrategyMetadata(
            id="first_board_250_ma",
            name="first_board_250_ma",
            display_name="250首板战法",
            description="基于250日均线突破的首板股票筛选策略。必备条件：股价上穿MA250、量比>1、涨幅≥9.9%、"
            "换手率5%-15%、流通市值10-100亿。加分条件：90/180日横盘、横盘期内有涨停。",
            strategy_type=StrategyType.PYTHON,
            category="short_term",
            source="builtin",
            version="1.0.0",
            score_multiplier=1.0,
            max_raw_score=85.0,
        )
        super().__init__(metadata)

    # ------------------------------------------------------------------
    # 必备条件（任一失败 → matched=False 立即返回）
    # ------------------------------------------------------------------

    def _check_ma250_cross(self, daily_data: pd.DataFrame, snapshot: Optional[Dict] = None) -> Dict[str, Any]:
        """条件A：股价上穿250日均线。根据快照日期与DB最新交易日的先后关系对齐比较窗口。"""
        data = daily_data.copy()
        if len(data) < 250:
            return {"passed": False, "reason": "日线数据不足250天", "detail": {}}

        data["MA250"] = data["close"].rolling(window=250).mean()

        # 解析 DB 最新交易日
        db_dates = pd.to_datetime(data["date"]).dt.date
        db_today = db_dates.iloc[-1]

        # 解析快照日期（若可用）
        snap_date = snapshot.get("日期") if snapshot else None

        detail: Dict[str, Any] = {}

        if snap_date is not None and snap_date > db_today:
            # ── 快照比 DB 新：快照现价 = 今日，DB 最新行 = 昨日 ──
            today_price = snapshot.get("现价")
            if today_price is None or pd.isna(today_price):
                return {"passed": False, "reason": "快照现价缺失", "detail": {}}

            yesterday_close = float(data["close"].iloc[-1])
            ma250_ref = float(data["MA250"].iloc[-1])

            detail = {
                "source": "snapshot_aligned",
                "snap_date": str(snap_date),
                "db_date": str(db_today),
                "today_price": today_price,
                "yesterday_close": yesterday_close,
                "ma250": ma250_ref,
            }

            if pd.isna(ma250_ref):
                return {"passed": False, "reason": "MA250 计算失败（数据不足）", "detail": detail}

            if yesterday_close <= ma250_ref < today_price:
                return {
                    "passed": True,
                    "reason": f"上穿MA250 (今日 {today_price:.2f} > MA250 {ma250_ref:.2f})",
                    "detail": detail,
                }
            return {
                "passed": False,
                "reason": f"未上穿MA250 (今日 {today_price:.2f} vs MA250 {ma250_ref:.2f})",
                "detail": detail,
            }
        else:
            # ── 快照与 DB 同步（或无快照）：DB 最后两行 = 昨日 + 今日 ──
            today_close = float(data["close"].iloc[-1])
            ma250_today = float(data["MA250"].iloc[-1])
            yesterday_close = float(data["close"].iloc[-2])
            ma250_yesterday = float(data["MA250"].iloc[-2])

            detail = {
                "source": "db_only",
                "db_date": str(db_today),
                "today_close": today_close,
                "ma250": ma250_today,
                "yesterday_close": yesterday_close,
                "yesterday_ma250": ma250_yesterday,
            }

            if pd.isna(ma250_today) or pd.isna(ma250_yesterday):
                return {"passed": False, "reason": "MA250 计算失败（数据不足）", "detail": detail}

            if yesterday_close <= ma250_yesterday and today_close > ma250_today:
                return {
                    "passed": True,
                    "reason": f"上穿MA250 (今日 {today_close:.2f} > MA250 {ma250_today:.2f})",
                    "detail": detail,
                }
            return {
                "passed": False,
                "reason": f"未上穿MA250 (今日 {today_close:.2f} vs MA250 {ma250_today:.2f})",
                "detail": detail,
            }

    def _check_volume_ratio(self, snapshot: Optional[Dict], daily_data: pd.DataFrame) -> Dict[str, Any]:
        """条件B：量比 > 1。快照优先，日线 volume 兜底。"""
        # 优先快照
        if snapshot and snapshot.get("量比") is not None:
            vol_ratio = snapshot["量比"]
            if vol_ratio > 1.0:
                return {"passed": True, "reason": f"量比 {vol_ratio:.2f} > 1（快照）", "detail": {"volume_ratio": vol_ratio, "source": "snapshot"}}
            else:
                return {"passed": False, "reason": f"量比 {vol_ratio:.2f} ≤ 1（快照）", "detail": {"volume_ratio": vol_ratio, "source": "snapshot"}}

        # 日线兜底：当日 volume > 前5日均量
        if daily_data is not None and len(daily_data) >= 6:
            today_vol = float(daily_data["volume"].iloc[-1])
            avg_5_vol = float(daily_data["volume"].iloc[-6:-1].mean())
            if today_vol > avg_5_vol:
                return {"passed": True, "reason": f"日线放量 {today_vol:.0f} > 5日均量 {avg_5_vol:.0f}（日线兜底）", "detail": {"today_volume": today_vol, "avg_5_volume": avg_5_vol, "source": "daily"}}
            else:
                return {"passed": False, "reason": f"日线未放量 {today_vol:.0f} ≤ 5日均量 {avg_5_vol:.0f}（日线兜底）", "detail": {"today_volume": today_vol, "avg_5_volume": avg_5_vol, "source": "daily"}}

        return {"passed": False, "reason": "量比数据不可用（快照+日线均失败）", "detail": {}}

    def _check_change_pct(self, snapshot: Optional[Dict], daily_data: pd.DataFrame) -> Dict[str, Any]:
        """条件C：涨幅 ≥ 9.9%。快照优先，日线 pct_chg 兜底。"""
        # 优先快照
        if snapshot and snapshot.get("涨跌幅") is not None:
            change = snapshot["涨跌幅"]
            if change >= 9.9:
                return {"passed": True, "reason": f"涨幅 {change:.2f}% ≥ 9.9%（快照）", "detail": {"change_pct": change, "source": "snapshot"}}
            else:
                return {"passed": False, "reason": f"涨幅 {change:.2f}% < 9.9%（快照）", "detail": {"change_pct": change, "source": "snapshot"}}

        # 日线兜底
        if daily_data is not None and len(daily_data) >= 1:
            pct_chg = float(daily_data["pct_chg"].iloc[-1])
            if pct_chg >= 9.9:
                return {"passed": True, "reason": f"涨幅 {pct_chg:.2f}% ≥ 9.9%（日线兜底）", "detail": {"change_pct": pct_chg, "source": "daily"}}
            else:
                return {"passed": False, "reason": f"涨幅 {pct_chg:.2f}% < 9.9%（日线兜底）", "detail": {"change_pct": pct_chg, "source": "daily"}}

        return {"passed": False, "reason": "涨幅数据不可用", "detail": {}}

    def _check_turnover_rate(self, snapshot: Optional[Dict], daily_data: pd.DataFrame) -> Dict[str, Any]:
        """条件D：换手率 5% - 15%。快照优先，日线 turnover_rate 兜底。"""
        # 优先快照
        if snapshot and snapshot.get("换手率") is not None:
            turnover = snapshot["换手率"]
            if 5.0 <= turnover <= 15.0:
                return {"passed": True, "reason": f"换手率 {turnover:.2f}%（快照）", "detail": {"turnover_rate": turnover, "source": "snapshot"}}
            else:
                return {"passed": False, "reason": f"换手率 {turnover:.2f}% 不在 5%-15%（快照）", "detail": {"turnover_rate": turnover, "source": "snapshot"}}

        # 日线兜底
        if daily_data is not None and len(daily_data) >= 1:
            turnover = float(daily_data["turnover_rate"].iloc[-1])
            if not pd.isna(turnover) and 5.0 <= turnover <= 15.0:
                return {"passed": True, "reason": f"换手率 {turnover:.2f}%（日线兜底）", "detail": {"turnover_rate": turnover, "source": "daily"}}
            else:
                return {"passed": False, "reason": f"换手率 {turnover:.2f}% 不在 5%-15%（日线兜底）", "detail": {"turnover_rate": turnover, "source": "daily"}}

        return {"passed": False, "reason": "换手率数据不可用", "detail": {}}

    def _check_circ_mv(self, snapshot: Optional[Dict]) -> Dict[str, Any]:
        """条件E：流通市值 10亿 - 100亿。仅快照可获取，不可用时放行。"""
        if snapshot is None or snapshot.get("流通市值") is None:
            return {"passed": True, "reason": "流通市值数据不可用，放行", "detail": {"circ_mv": None, "source": "unavailable"}}

        circ_mv = snapshot["流通市值"]  # 腾讯单位：亿
        if circ_mv is None or pd.isna(circ_mv):
            return {"passed": True, "reason": "流通市值数据不可用，放行", "detail": {"circ_mv": None, "source": "unavailable"}}

        if 10.0 <= circ_mv <= 100.0:
            return {"passed": True, "reason": f"流通市值 {circ_mv:.1f}亿（快照）", "detail": {"circ_mv": circ_mv, "source": "snapshot"}}
        else:
            return {"passed": False, "reason": f"流通市值 {circ_mv:.1f}亿 不在 10-100亿", "detail": {"circ_mv": circ_mv, "source": "snapshot"}}

    # ------------------------------------------------------------------
    # 加分条件（仅在所有必备通过后被调用）
    # ------------------------------------------------------------------

    def _check_sideways(self, daily_data: pd.DataFrame, natural_days: int) -> bool:
        """判断上穿日往前 natural_days 个自然日内是否横盘（波幅 < 20%）。"""
        if daily_data is None or len(daily_data) < 5:
            return False

        df = daily_data.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        last_date = df["date_dt"].iloc[-1]
        start_date = last_date - timedelta(days=natural_days)

        # 裁剪自然日区间
        period_df = df[df["date_dt"] >= start_date]
        if len(period_df) < 10:  # 至少需要10个交易日
            return False

        max_high = float(period_df["high"].max())
        min_low = float(period_df["low"].min())
        mean_close = float(period_df["close"].mean())

        if mean_close == 0:
            return False

        amplitude = (max_high - min_low) / mean_close
        return amplitude < 0.20

    def _has_limit_up_in_period(self, daily_data: pd.DataFrame, natural_days: int) -> bool:
        """判断上穿日往前 natural_days 个自然日内是否有涨停（pct_chg ≥ 9.9%）。"""
        if daily_data is None or len(daily_data) < 5:
            return False

        df = daily_data.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        last_date = df["date_dt"].iloc[-1]
        start_date = last_date - timedelta(days=natural_days)

        period_df = df[(df["date_dt"] >= start_date) & (df["date_dt"] < last_date)]  # 排除上穿日当天
        if period_df.empty:
            return False

        return (period_df["pct_chg"] >= 9.9).any()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def select(
        self,
        stock_code: str,
        stock_name: str = "",
        daily_data: Optional[pd.DataFrame] = None,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> StrategyMatch:
        """执行250首板战法选股。"""
        match_details: Dict[str, Any] = {
            "conditions": {},
            "score_breakdown": {},
        }

        # 日线数据检查
        if daily_data is None or not isinstance(daily_data, pd.DataFrame) or daily_data.empty:
            return StrategyMatch(
                strategy_id=self.metadata.id,
                strategy_name=self.metadata.display_name,
                score=0.0,
                matched=False,
                reason="无日线数据",
                match_details=match_details,
            )

        # 获取快照（一次性全市场拉取，已缓存）
        snapshot = _get_stock_from_snapshot(stock_code)

        # ── 阶段1: 必备条件（快速失败） ──

        # 条件A: MA250 上穿
        result_a = self._check_ma250_cross(daily_data, snapshot=snapshot)
        match_details["conditions"]["A_ma250_cross"] = result_a
        if not result_a["passed"]:
            return StrategyMatch(
                strategy_id=self.metadata.id, strategy_name=self.metadata.display_name,
                score=0.0, matched=False,
                reason=result_a["reason"], match_details=match_details,
            )

        # 条件B: 量比 > 1
        result_b = self._check_volume_ratio(snapshot, daily_data)
        match_details["conditions"]["B_volume_ratio"] = result_b
        if not result_b["passed"]:
            return StrategyMatch(
                strategy_id=self.metadata.id, strategy_name=self.metadata.display_name,
                score=0.0, matched=False,
                reason=result_b["reason"], match_details=match_details,
            )

        # 条件C: 涨幅 ≥ 9.9%
        result_c = self._check_change_pct(snapshot, daily_data)
        match_details["conditions"]["C_change_pct"] = result_c
        if not result_c["passed"]:
            return StrategyMatch(
                strategy_id=self.metadata.id, strategy_name=self.metadata.display_name,
                score=0.0, matched=False,
                reason=result_c["reason"], match_details=match_details,
            )

        # 条件D: 换手率 5%-15%
        result_d = self._check_turnover_rate(snapshot, daily_data)
        match_details["conditions"]["D_turnover_rate"] = result_d
        if not result_d["passed"]:
            return StrategyMatch(
                strategy_id=self.metadata.id, strategy_name=self.metadata.display_name,
                score=0.0, matched=False,
                reason=result_d["reason"], match_details=match_details,
            )

        # 条件E: 流通市值 10-100 亿
        result_e = self._check_circ_mv(snapshot)
        match_details["conditions"]["E_circ_mv"] = result_e
        if not result_e["passed"]:
            return StrategyMatch(
                strategy_id=self.metadata.id, strategy_name=self.metadata.display_name,
                score=0.0, matched=False,
                reason=result_e["reason"], match_details=match_details,
            )

        # ── 阶段2: 加分条件（全部必备通过后才到这里） ──
        score = 70.0
        match_details["score_breakdown"]["base"] = 70.0

        # F1: 90 自然日横盘
        f1_passed = self._check_sideways(daily_data, 90)
        if f1_passed:
            score += 5.0
            match_details["score_breakdown"]["F1_sideways_90d"] = 5.0
            match_details["conditions"]["F1_sideways_90d"] = {"passed": True, "reason": "90自然日横盘"}
        else:
            match_details["score_breakdown"]["F1_sideways_90d"] = 0.0

        # F2: 180 自然日横盘（F1 通过后才判断）
        f2_passed = False
        if f1_passed:
            f2_passed = self._check_sideways(daily_data, 180)
        if f2_passed:
            score += 5.0
            match_details["score_breakdown"]["F2_sideways_180d"] = 5.0
            match_details["conditions"]["F2_sideways_180d"] = {"passed": True, "reason": "180自然日横盘"}
        else:
            match_details["score_breakdown"]["F2_sideways_180d"] = 0.0

        # G: 横盘期间有涨停
        g_passed = self._has_limit_up_in_period(daily_data, 180)
        if g_passed:
            score += 5.0
            match_details["score_breakdown"]["G_limit_up"] = 5.0
            match_details["conditions"]["G_limit_up"] = {"passed": True, "reason": "横盘期间有涨停"}
        else:
            match_details["score_breakdown"]["G_limit_up"] = 0.0

        # 构建 reason
        reasons = ["250首板战法匹配"]
        reasons.append(result_a["reason"])
        reasons.append(result_b["reason"])
        reasons.append(result_c["reason"])
        reasons.append(result_d["reason"])
        reasons.append(result_e["reason"])
        if f1_passed:
            reasons.append("横盘3个月 +5")
            if f2_passed:
                reasons.append("横盘6个月 +5")
        if g_passed:
            reasons.append("横盘期涨停 +5")

        return StrategyMatch(
            strategy_id=self.metadata.id,
            strategy_name=self.metadata.display_name,
            score=score,
            matched=True,
            reason="; ".join(reasons),
            match_details=match_details,
        )