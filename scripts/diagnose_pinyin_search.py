"""
诊断：检查生意宝 (002095) 拼音搜索索引
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stock_search_index import get_search_index

index = get_search_index()

# 1. 查找 002095 在索引中的位置
entry_002095 = None
for entry in index._entries:
    if entry.code == "002095":
        entry_002095 = entry
        break

if entry_002095:
    print(f"找到 002095: name={entry_002095.name}")
    print(f"  pinyin_full={entry_002095.pinyin_full}")
    print(f"  pinyin_initials={entry_002095.pinyin_initials}")
    print(f"  pinyin_first_char={entry_002095.pinyin_first_char}")

    # 2. 检查 Trie
    init_result = index._pinyin_initials_trie.search_prefix("syb")
    if entry_002095 and id(entry_002095) in [id(index._entries[i]) for i in init_result]:
        # 变通方式找 idx
        for i in init_result:
            if index._entries[i].code == "002095":
                print(f"  ✓ 'syb' 在 initials_trie 中找到，idx={i}, score 基数=50+3=53")
                break
        else:
            print("  ✗ 'syb' 在 initials_trie 中 CANNOT 找到 002095")
    else:
        print(f"  ✗ 'syb' 在 initials_trie 中搜索到 {len(init_result)} 个结果，但不含 002095")

    # 3. 直接测试 search 方法
    results = index.search("syb", 20)
    print(f"\n搜索 'syb' 返回 {len(results)} 个结果:")
    for r in results:
        print(f"  {r['code']} {r['name']} match_type={r['match_type']} score={r['score']}")

    # 搜索002095作为对照
    results2 = index.search("002095", 20)
    print(f"\n搜索 '002095' 返回 {len(results2)} 个结果:")
    for r in results2:
        print(f"  {r['code']} {r['name']} match_type={r['match_type']} score={r['score']}")
else:
    print("002095 不在索引中！")
    # 打印所有创业板/中小板股票看看
    print("所有索引中的股票:")
    for e in index._entries:
        if e.code.startswith("00"):
            print(f"  {e.code} {e.name}")