"""验证 *stb 不再误匹配 ST百灵"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from src.stock_search_index import get_search_index
index = get_search_index()

tests = ["*stb", "*stbs", "*st"]
for q in tests:
    results = index.search(q, 10)
    print(f"\n--- 搜索 '{q}' ---")
    for r in results:
        print(f"  {r['code']} {r['name']}  match={r['match_type']}")
    
    has_st_bailing = any(r['code'] == '002424' for r in results)
    print(f"  >> {'✗ 误匹配ST百灵' if has_st_bailing else '✓ 没有ST百灵'}")