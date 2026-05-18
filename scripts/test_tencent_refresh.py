"""
腾讯财经分时API刷新规律测试脚本

在交易时段运行，轮询腾讯1分钟分时接口，
记录每次返回的最新数据点时间和服务端响应时间，
分析数据到达的延迟规律。

用法（交易时段内运行）:
    python scripts/test_tencent_refresh.py --code 000001 --duration 600
    python scripts/test_tencent_refresh.py --code 600519 --duration 1800 --interval 3
"""
import argparse
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict

import requests

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_tencent_1min(code: str) -> tuple:
    """获取腾讯1分钟分时数据，返回(最新数据点时间, 数据点列表)"""
    market = "sh" if code.startswith("6") else "sz"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={market}{code}"
    request_time = datetime.now()
    try:
        r = requests.get(url, timeout=8)
        response_time = datetime.now()
        data = r.json()
        stock_data = data.get("data", {}).get(f"{market}{code}", {})
        point_list = stock_data.get("data", {}).get("data", [])
        resp_date = stock_data.get("data", {}).get("date", "")

        if not point_list:
            return request_time, response_time, None, []

        latest_time = point_list[-1].split()[0] if point_list else ""
        return request_time, response_time, latest_time, point_list
    except Exception as e:
        return request_time, datetime.now(), None, []


def analyze(samples: List[Dict]) -> None:
    """分析采集到的数据"""
    if not samples:
        print("无数据")
        return

    # 找出每次数据更新（最新点变化）的时刻
    updates = []
    prev_latest = ""
    for s in samples:
        if s["latest_time"] and s["latest_time"] != prev_latest:
            updates.append({
                "latest_time": s["latest_time"],
                "detected_at": s["response_time"].strftime("%H:%M:%S.%f")[:-3],
                "delay_ms": int((s["response_time"] - s["request_time"]).total_seconds() * 1000),
            })
            prev_latest = s["latest_time"]

    print(f"\n{'='*60}")
    print(f"采样总数: {len(samples)}")
    print(f"时间跨度: {samples[0]['request_time'].strftime('%H:%M:%S')} ~ {samples[-1]['request_time'].strftime('%H:%M:%S')}")
    print(f"数据更新次数: {len(updates)}")

    if not updates:
        print("无数据更新（可能非交易时段）")
        return

    print(f"\n--- 每次数据更新详情 ---")
    print(f"{'数据时间':>8} | {'检测到时间':>14} | {'延迟(ms)':>8}")
    print("-" * 40)
    for u in updates:
        print(f"{u['latest_time']:>8} | {u['detected_at']:>14} | {u['delay_ms']:>6}")

    # 分析时间间隔规律
    print(f"\n--- 到达时间分析 ---")
    for u in updates:
        # 将数据时间转换为秒数
        h = int(u["latest_time"][:2])
        m = int(u["latest_time"][2:4])
        data_second = h * 3600 + m * 60
        # 将检测时间转换为秒数
        parts = u["detected_at"].split(":")
        detected_second = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        offset = detected_second - data_second
        print(f"  {u['latest_time']}: 检测到于 {u['detected_at']}, 数据时间之后 {offset:.1f}s")

    # 延迟统计
    delays = [u["delay_ms"] for u in updates]
    print(f"\n--- 请求延迟统计 ---")
    print(f"  最小: {min(delays)}ms")
    print(f"  最大: {max(delays)}ms")
    print(f"  平均: {sum(delays)/len(delays):.0f}ms")

    # 检查是否是按分钟或半分钟刷新
    print(f"\n--- 刷新规律分析 ---")
    secs = []
    for u in updates:
        parts = u["detected_at"].split(":")
        secs.append(float(parts[2]))
    print(f"  检测时刻的秒数: {[f'{s:.1f}' for s in secs]}")
    # 检查是否集中在某个秒数附近
    if len(secs) >= 3:
        avg_sec = sum(secs) / len(secs)
        variance = sum((s - avg_sec) ** 2 for s in secs) / len(secs)
        print(f"  平均检测秒数: {avg_sec:.1f}s, 方差: {variance:.1f}")
        if variance < 25:
            print(f"  ✅ 数据更新时间集中在 {avg_sec:.0f}s 附近（±{variance**0.5:.0f}s）")
        else:
            print(f"  ❌ 数据更新时间分散，无明显规律")


def main():
    parser = argparse.ArgumentParser(description="测试腾讯财经分时API刷新规律")
    parser.add_argument("--code", type=str, default="000001", help="股票代码")
    parser.add_argument("--duration", type=int, default=600, help="采样时长（秒），默认600秒=10分钟")
    parser.add_argument("--interval", type=int, default=5, help="轮询间隔（秒），默认5秒")
    args = parser.parse_args()

    # 检查是否在交易时段
    now = datetime.now()
    current_hhmm = now.hour * 100 + now.minute
    # A股交易时段: 9:30-11:30, 13:00-15:00
    in_trading = (930 <= current_hhmm <= 1130) or (1300 <= current_hhmm <= 1500)

    if not in_trading:
        logger.warning("⚠️ 当前非交易时段（A股 9:30-11:30, 13:00-15:00），API可能返回旧数据或无数据")
        logger.warning("建议在交易时段运行此脚本以获得准确结果")
        logger.warning("继续运行以测试API响应...")
        print()

    logger.info(f"开始轮询腾讯分时API: code={args.code}")
    logger.info(f"采样时长: {args.duration}s, 轮询间隔: {args.interval}s")
    print()

    samples: List[Dict] = []
    start = time.time()
    iteration = 0

    try:
        while time.time() - start < args.duration:
            iteration += 1
            req_time, resp_time, latest_time, points = fetch_tencent_1min(args.code)
            sample = {
                "iteration": iteration,
                "request_time": req_time,
                "response_time": resp_time,
                "latest_time": latest_time,
                "point_count": len(points),
            }
            samples.append(sample)

            if latest_time:
                logger.info(
                    f"[{iteration:4d}] 最新数据点: {latest_time}, "
                    f"总点数: {len(points):3d}, "
                    f"延迟: {(resp_time - req_time).total_seconds()*1000:5.0f}ms"
                )
            else:
                logger.warning(f"[{iteration:4d}] 无数据返回")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("用户中断")

    print()
    analyze(samples)

    # 保存原始数据
    output_file = f"logs/tencent_refresh_test_{now.strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([{**s, "request_time": s["request_time"].isoformat(), "response_time": s["response_time"].isoformat()} for s in samples], f, ensure_ascii=False, indent=2)
    logger.info(f"原始数据已保存到: {output_file}")


if __name__ == "__main__":
    main()