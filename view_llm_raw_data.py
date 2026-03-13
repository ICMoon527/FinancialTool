# -*- coding: utf-8 -*-
"""
查看阿里云LLM返回的原始JSON数据
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage import get_db


def main():
    print("=" * 80)
    print("📊 查看阿里云LLM返回的原始JSON数据")
    print("=" * 80)
    
    db = get_db()
    
    with db.session_scope() as session:
        from src.storage import AnalysisHistory
        
        print("\n📋 获取最新的10条分析历史...")
        print("=" * 80)
        
        records = session.query(AnalysisHistory).order_by(
            AnalysisHistory.created_at.desc()
        ).limit(10).all()
        
        if not records:
            print("❌ 没有找到分析历史记录")
            return
        
        for i, record in enumerate(records, 1):
            print(f"\n{'=' * 80}")
            print(f"📝 记录 #{i}: {record.code} ({record.name})")
            print(f"⏰ 创建时间: {record.created_at}")
            print(f"⭐ 评分: {record.sentiment_score}")
            print(f"📌 操作建议: {record.operation_advice}")
            print(f"📈 趋势预测: {record.trend_prediction}")
            print(f"✅ 分析摘要: {record.analysis_summary[:100] if record.analysis_summary else '无'}...")
            print(f"{'=' * 80}")
            
            if record.raw_result:
                print("\n📄 原始结果 (raw_result):")
                print("-" * 80)
                try:
                    raw_data = json.loads(record.raw_result)
                    print(json.dumps(raw_data, ensure_ascii=False, indent=2))
                    
                    print("\n🔍 数据结构分析:")
                    print("-" * 80)
                    
                    if isinstance(raw_data, dict):
                        print(f"顶层键: {list(raw_data.keys())}")
                        
                        if 'dashboard' in raw_data:
                            dash = raw_data['dashboard']
                            if isinstance(dash, dict):
                                print(f"dashboard键: {list(dash.keys())}")
                                
                                if 'core_conclusion' in dash:
                                    cc = dash['core_conclusion']
                                    if isinstance(cc, dict):
                                        print(f"  - core_conclusion键: {list(cc.keys())}")
                                        if 'one_sentence' in cc:
                                            print(f"    - one_sentence: {cc['one_sentence']}")
                        
                        if 'raw_response' in raw_data and raw_data['raw_response']:
                            print(f"\n📋 raw_response (LLM原始响应):")
                            print("-" * 80)
                            print(str(raw_data['raw_response'])[:1000])
                            if len(str(raw_data['raw_response'])) > 1000:
                                print("... (截断，超过1000字符)")
                    
                except Exception as e:
                    print(f"解析失败: {e}")
                    print(record.raw_result)
            
            if record.context_snapshot:
                print("\n🎯 上下文快照 (context_snapshot):")
                print("-" * 80)
                try:
                    ctx_data = json.loads(record.context_snapshot)
                    print(json.dumps(ctx_data, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"解析失败: {e}")
                    print(record.context_snapshot)
            
            if i == 5:
                print("\n" + "=" * 80)
                print("💡 提示: 已显示5条记录。")
                print("=" * 80)
                break


if __name__ == '__main__':
    main()
