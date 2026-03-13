#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看最新的分析历史记录
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage import get_db
from datetime import datetime, date
import json


def main():
    db = get_db()
    
    with db.session_scope() as session:
        from src.storage import AnalysisHistory
        
        # 获取最新的分析历史
        latest = session.query(AnalysisHistory).order_by(
            AnalysisHistory.created_at.desc()
        ).first()
        
        if latest:
            print("=" * 80)
            print(f"📊 最新分析历史记录: {latest.code} ({latest.name})")
            print("=" * 80)
            print(f"创建时间: {latest.created_at}")
            print(f"评分: {latest.sentiment_score}")
            print(f"操作建议: {latest.operation_advice}")
            print(f"趋势预测: {latest.trend_prediction}")
            print(f"分析摘要: {latest.analysis_summary}")
            print()
            
            if latest.raw_result:
                print("=" * 80)
                print("📋 原始结果 (raw_result):")
                print("=" * 80)
                try:
                    raw_data = json.loads(latest.raw_result)
                    print(json.dumps(raw_data, ensure_ascii=False, indent=2))
                except:
                    print(latest.raw_result)
            print()
            
            if latest.context_snapshot:
                print("=" * 80)
                print("🔍 上下文快照 (context_snapshot):")
                print("=" * 80)
                try:
                    ctx_data = json.loads(latest.context_snapshot)
                    print(json.dumps(ctx_data, ensure_ascii=False, indent=2))
                except:
                    print(latest.context_snapshot)
        else:
            print("❌ 没有找到分析历史记录")


if __name__ == '__main__':
    main()
