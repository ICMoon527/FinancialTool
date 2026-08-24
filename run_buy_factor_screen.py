# -*- coding: utf-8 -*-
"""
买入侧 alpha 因子筛查：分析天道超跌反弹策略中"什么样的超跌票反弹最猛"。

直接复用基线回测产物（backtest_results.json），不重跑回测：
- 从逐笔 trades 做 FIFO 配对，聚合成 (stock, entry_date, exit_date, realized_pnl)
- 对每笔补四个因子：
  (a) 跌幅深度   = 买入前60日(最高收盘价 - 买入收盘价)/最高收盘价
  (b) 下跌速度   = 跌幅深度 / 距前期高点的交易日数（日均跌幅）
  (c) 板块       = SectorManager.get_stock_sectors（DB缓存URL）
  (d) 市值       = 流通股本(当前spot) × 买入日收盘价
- 按四分位分桶，输出 realized_pnl 对各因子的单调性。

数据说明：
- 跌幅/速度/市值中的"买入日收盘价"来自日线数据（精确历史）
- 市值用"当前流通股本 × 历史买入收盘价"估算，流通股本相对稳定，误差可接受
- 板块为当前归属，相对稳定，仅供分层参考
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def load_paired_trades(results_json: str):
    """
    读取 backtest_results.json 的逐笔 trades，按 (stock, 日期) 做 FIFO 配对，
    聚合为买入批次级已实现交易。

    Returns:
        List[Dict] 每笔 {stock, entry_date, exit_date, pnl, entry_close}
    """
    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    trades = results["trades"]

    ordered = sorted(trades, key=lambda t: (t["stock_code"], t["date"]))
    buys: dict = {}
    realized: list = []
    for t in ordered:
        stock = t["stock_code"]
        qty = t["quantity"]
        if t["order_type"] == "buy":
            buy_cost = qty * t["price"] + t["commission"] + t["slippage"]
            buys.setdefault(stock, []).append(
                {"entry_date": t["date"], "entry_close": t["price"], "remaining": qty, "buy_cost": buy_cost, "sell_income": 0.0, "exit_date": None}
            )
        else:
            income = qty * t["price"] - t["commission"] - t["slippage"]
            queue = buys.get(stock, [])
            while qty > 0 and queue:
                b = queue[0]
                take = min(qty, b["remaining"])
                b["sell_income"] += income * (take / t["quantity"])
                b["remaining"] -= take
                qty -= take
                b["exit_date"] = t["date"]
                if b["remaining"] <= 0:
                    pnl = (b["sell_income"] - b["buy_cost"]) / b["buy_cost"]
                    realized.append(
                        {
                            "stock": stock,
                            "entry_date": b["entry_date"],
                            "entry_close": b["entry_close"],
                            "exit_date": b["exit_date"],
                            "pnl": pnl,
                        }
                    )
                    queue.pop(0)
    return realized


def add_price_factors(paired, data_provider, lookback_days=60):
    """
    对每笔交易补齐跌幅深度、下跌速度、买入日收盘价。

    Args:
        paired: aggregate_trades 输出
        data_provider: DataFetcherManager 实例
        lookback_days: 买入前回看交易日数（用于确定前期高点窗口）

    Returns:
        增强后的交易列表（仅含成功补全因子的），dict 增加 drop_depth、drop_speed 字段
    """
    from datetime import timedelta

    enriched = []
    for t in paired:
        try:
            code = t["stock"]
            entry_dt = pd.to_datetime(t["entry_date"]).date()
            # 取买入日前约 lookback*2 个自然日，确保覆盖足够交易日
            start = entry_dt - timedelta(days=lookback_days * 2)
            end = entry_dt + timedelta(days=1)
            df, _ = data_provider.get_daily_data(code, start_date=str(start), end_date=str(end))
            if df is None or df.empty or "close" not in df.columns:
                logger.debug("无日线数据: %s", code)
                continue

            df = df.sort_values("date").reset_index(drop=True)
            df = df[pd.to_datetime(df["date"]).dt.date <= entry_dt]
            if len(df) < 10:
                logger.debug("日线太少: %s (%d)", code, len(df))
                continue

            closes = df["close"].astype(float).values
            entry_close = t["entry_close"]

            # 跌幅深度：买入前窗口内(含当日)的最高收盘价相对买入收盘价的最大回撤
            # 使用窗口 MA 高点近似（避免单日毛刺被误判为前期高点）
            rolling_high = pd.Series(closes).rolling(20, min_periods=5).max()
            prev_high = rolling_high.iloc[-1] if not np.isnan(rolling_high.iloc[-1]) else closes[-1]
            drop_depth = max(0.0, (prev_high - entry_close) / prev_high)

            # 下跌速度：从前期高点到买入日的日均跌幅
            # 找最近一次收盘价 >= prev_high*0.98 的位置作为前期高点日
            high_idx = None
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] >= prev_high * 0.95:
                    high_idx = i
                    break
            if high_idx is None:
                high_idx = 0
            days_to_peak = max(1, (len(df) - 1 - high_idx))
            drop_speed = drop_depth / days_to_peak if days_to_peak > 0 else 0.0

            t["drop_depth"] = drop_depth
            t["drop_speed"] = drop_speed
            t["prev_high"] = prev_high

            # 市值：买入日流通市值 = 成交额 / 换手率（当日精确历史值，换手率为小数）
            cap = np.nan
            if {"amount", "turnover_rate"}.issubset(df.columns):
                last = df.iloc[-1]
                amt = last.get("amount")
                tr = last.get("turnover_rate")
                if amt is not None and tr is not None:
                    amt = float(amt)
                    tr = float(tr)
                    if amt > 0 and 0 < tr <= 1:
                        cap = amt / tr / 1e8  # 元 → 亿元
            t["mkt_cap"] = cap

            enriched.append(t)
        except Exception as e:
            logger.debug("补因子失败 %s: %s", t.get("stock"), e)
            continue
    return enriched


def add_sector_and_mv(enriched, data_provider, sector_manager):
    """
    补齐板块与市值。

    - 板块：取股票所属板块列表的首个作为主板块桶（用于按板块分层）。
      不从网络拉取，优先复用数据库缓存的板块信息。
    - 市值：流通股本(数据库 stock_basic) × 买入日收盘价。

    Args:
        enriched: 已含 price factors 的交易列表
        data_provider: DataFetcherManager（保留参数，供数据源扩展）
        sector_manager: SectorManager（提供 get_stock_sectors）

    Returns:
        增强交易列表，dict 增加 sector（主板块）、mkt_cap（亿元）字段
    """
    from datetime import date as _date

    from src.services.turnover_service import TurnoverService
    ts = TurnoverService()

    for t in enriched:
        code = t["stock"]
        # 板块：数据库缓存优先，取首个作为主板块
        sector = None
        if sector_manager is not None:
            try:
                sectors = sector_manager.get_stock_sectors(code)
                if sectors:
                    sector = sectors[0]
            except Exception:
                sector = None
        t["sector"] = sector

        # 市值：优先用日线反推的买入日流通市值（add_price_factors 已算）；
        # 缺失时回退为「DB 流通股本 × 买入日收盘价」
        if not (cap_ := t.get("mkt_cap")) or not np.isfinite(cap_):
            cap = np.nan
            try:
                entry_date = _date.fromisoformat(t["entry_date"])
                b = ts.get_stock_basic(code, entry_date)
                if b is not None and b.circulating_shares and b.circulating_shares > 0:
                    cap = b.circulating_shares * t["entry_close"] / 1e8
            except Exception:
                cap = np.nan
            t["mkt_cap"] = cap
    return enriched


def quartile_report(enriched, value_key, label, direction="desc"):
    """
    按某因子四分位分桶，输出各桶 realized_pnl 的中位数/均值/样本量。
    """
    valid = [t for t in enriched if t.get(value_key) is not None and pd.notna(t.get(value_key))]
    if len(valid) < 8:
        logger.info("  %-12s 有效样本 %d 过少，跳过", label, len(valid))
        return
    df = pd.DataFrame(valid)
    qc = pd.qcut(df[value_key], 4, duplicates="drop")
    grp = df.groupby(qc, observed=True)["pnl"].agg(["count", "median", "mean", "std"])
    logger.info("\n  【%s】（%s）", label, "从高到低" if direction == "desc" else "")
    logger.info("  %-22s %6s %8s %8s", "区间", "n", "中位数", "均值")
    for idx, row in grp.iterrows():
        iv = getattr(idx, "left", None)
        logger.info("  %-22s %6d %8.2f%% %8.2f%%", str(idx), int(row["count"]), row["median"] * 100, row["mean"] * 100)


def main():
    # 默认读基线回测产物
    base_dir = project_root / "strategy_backtest_results" / "天道超跌反弹买入策略_动态分级止盈_2021-07-31_2026-07-31"
    results_json = base_dir / "backtest_results.json"
    if not results_json.exists():
        logger.error("未找到回测产物: %s", results_json)
        logger.error("请先运行一次动态分级止盈(基线)回测生成 backtest_results.json")
        return

    paired = load_paired_trades(str(results_json))
    logger.info("聚合买入批次交易数: %d", len(paired))

    from data_provider import DataFetcherManager
    data_provider = DataFetcherManager(enable_realtime=False)

    # 板块管理
    sector_manager = None
    try:
        from stock_selector.sector_manager import SectorManager
        sector_manager = SectorManager()
    except Exception as e:
        logger.warning("SectorManager 初始化失败，板块因子将跳过: %s", e)

    enriched = add_price_factors(paired, data_provider)
    logger.info("补全价格因子后的有效样本: %d", len(enriched))

    # 补充板块（主板块），并作为市值缺失时的回退
    enriched = add_sector_and_mv(enriched, data_provider, sector_manager)

    logger.info("=" * 60)
    logger.info("买入侧 alpha 因子筛查（基线动态分级止盈回测产物）")
    logger.info("样本: 逐笔已实现交易，pnl 为整笔收益率")
    logger.info("=" * 60)

    logger.info("\n  === 跌幅深度（买入前最大回撤，越大=跌得越深）===")
    quartile_report(enriched, "drop_depth", "跌幅深度")

    logger.info("\n  === 下跌速度（日均跌幅，越大=跌得越急）===")
    quartile_report(enriched, "drop_speed", "下跌速度")

    # 板块：按板块分组统计
    sector_valid = [t for t in enriched if t.get("sector")]
    if sector_valid:
        df = pd.DataFrame(sector_valid)
        logger.info("\n  === 板块分组（出现≥3次的板块）===")
        grp = df.groupby("sector")["pnl"].agg(["count", "median", "mean"]).reset_index()
        grp_f = grp[grp["count"] >= 3].sort_values("median", ascending=False)
        if grp_f.empty:
            logger.info("  无满足≥3次的板块分组")
        else:
            for _, r in grp_f.iterrows():
                logger.info("  %-20s n=%3d 中位数=%7.2f%% 均值=%7.2f%%", r["sector"], int(r["count"]), r["median"] * 100, r["mean"] * 100)

    logger.info("\n  === 市值（DB流通股本 × 买入收盘价，越大=市值越高）===")
    quartile_report(enriched, "mkt_cap", "市值", direction="desc")

    logger.info("\n  === 总体参照：全部交易 pnl 分布 ===")
    p = pd.Series([t["pnl"] for t in enriched])
    logger.info("  样本 n=%d | 中位数=%7.2f%% | 均值=%7.2f%% | 胜率=%5.1f%%", len(p), p.median() * 100, p.mean() * 100, (p > 0).mean() * 100)


if __name__ == "__main__":
    main()