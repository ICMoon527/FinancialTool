# -*- coding: utf-8 -*-
"""
===================================
股票搜索索引模块
===================================

职责：
1. 全量加载 stock_pool 表并建立内存索引
2. 支持中文名称、拼音（全拼/简拼/首字母）、代码的多维度搜索
3. 基于 Trie + Trigram 倒排索引实现高速模糊匹配（<1ms）
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pypinyin import pinyin, Style

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class StockIndexEntry:
    """单只股票的索引条目"""

    code: str
    name: str
    market: str
    name_lower: str = ""
    pinyin_full: str = ""  # 全拼（如 guizhoumaotai）
    pinyin_initials: str = ""  # 简拼（如 gzmt）
    pinyin_first_char: str = ""  # 每字首字母（如 gzm）
    pinyin_initials_alt: Set[str] = field(default_factory=set)  # 多音字简拼变体（如 长电科技 → cdjk）
    trigrams: Set[str] = field(default_factory=set)  # 2-gram 集合

    def __repr__(self) -> str:
        return f"<StockIndexEntry code={self.code} name={self.name}>"


# ---------------------------------------------------------------------------
# Trie 树实现
# ---------------------------------------------------------------------------


class TrieNode:
    """Trie 树节点"""

    __slots__ = ("children", "indices")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.indices: Set[int] = set()


class Trie:
    """前缀树，支持插入和前缀搜索"""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str, entry_idx: int) -> None:
        """插入一个词及其对应的条目索引"""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.indices.add(entry_idx)

    def search_prefix(self, prefix: str) -> Set[int]:
        """搜索前缀，返回匹配的条目索引集合"""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return set()
            node = node.children[ch]
        return node.indices.copy()


# ---------------------------------------------------------------------------
# 过滤规则
# ---------------------------------------------------------------------------

# ST 股票名称关键词
_ST_KEYWORDS = ("ST", "*ST", "SST", "S*ST")

# 需过滤的股票代码前缀
_BLOCKED_CODE_PREFIXES: Tuple[str, ...] = (
    # 北交所
    "8",
    "92",
    "43",
    "83",
    "87",
    # 科创板
    "688",
    "689",
    # 创业板
    "300",
    "301",
    "302",
)


def _is_valid_stock(name: str, code: str, _market: str) -> bool:
    """判断股票是否应该包含在搜索索引中"""
    # 过滤 ST 股票
    name_upper = name.upper()
    for kw in _ST_KEYWORDS:
        if kw in name_upper:
            return False

    # 过滤特定板块
    for prefix in _BLOCKED_CODE_PREFIXES:
        if code.startswith(prefix):
            return False

    return True


# ---------------------------------------------------------------------------
# 搜索索引
# ---------------------------------------------------------------------------


class StockSearchIndex:
    """
    股票搜索索引（常驻内存）

    索引结构：
    - _entries: 所有有效股票的 StockIndexEntry 列表
    - _code_map: code → entry_idx 映射
    - _name_trie: 中文名称前缀 Trie
    - _pinyin_full_trie: 全拼前缀 Trie
    - _pinyin_initials_trie: 简拼前缀 Trie
    - _trigram_index: 2-gram → Set[entry_idx]
    - _sorted_by_code: 按 code 升序的 idx 列表
    """

    def __init__(self) -> None:
        self._entries: List[StockIndexEntry] = []
        self._code_map: Dict[str, int] = {}
        self._name_trie = Trie()
        self._pinyin_full_trie = Trie()
        self._pinyin_initials_trie = Trie()
        self._trigram_index: Dict[str, Set[int]] = {}
        self._sorted_by_code: List[int] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def build(self) -> int:
        """
        从 stock_pool 表全量加载并构建索引。

        Returns:
            成功加载的股票条目数量
        """
        start = time.perf_counter()

        # 清空旧索引
        self._entries.clear()
        self._code_map.clear()
        self._name_trie = Trie()
        self._pinyin_full_trie = Trie()
        self._pinyin_initials_trie = Trie()
        self._trigram_index.clear()
        self._sorted_by_code.clear()

        # 从 stock_pool 表加载
        stocks = self._load_from_stock_pool()
        if not stocks:
            logger.warning("stock_pool 表为空，搜索索引为空")
            return 0

        # 逐条过滤并生成索引字段
        for code, name, market in stocks:
            if not _is_valid_stock(name, code, market):
                continue

            entry = self._build_entry(code, name, market)
            self._entries.append(entry)

        # 构建排序列表
        self._sorted_by_code = sorted(
            range(len(self._entries)), key=lambda i: self._entries[i].code
        )

        # 建立 code 映射
        for i, entry in enumerate(self._entries):
            self._code_map[entry.code] = i

        # 构建 Trie 和 Trigram 索引
        self._build_tries()
        self._build_trigram_index()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "搜索索引构建完成，共 %d 只股票，耗时 %.1fms",
            len(self._entries),
            elapsed_ms,
        )
        return len(self._entries)

    def _load_from_stock_pool(self) -> List[Tuple[str, str, str]]:
        """从 stock_pool 数据库表加载股票列表"""
        try:
            from sqlalchemy import select
            from src.storage import DatabaseManager
            from stock_selector.stock_pool import StockPoolItem

            db = DatabaseManager.get_instance()
            with db.get_session() as session:
                rows = session.execute(
                    select(StockPoolItem.code, StockPoolItem.name, StockPoolItem.market)
                ).all()

            result = [(row.code, row.name, row.market or "") for row in rows]
            logger.info("从 stock_pool 表读取了 %d 条股票记录", len(result))
            return result
        except Exception as e:
            logger.error("从 stock_pool 加载股票列表失败: %s", e)
            return []

    @staticmethod
    def _build_entry(code: str, name: str, market: str) -> "StockIndexEntry":
        """为单只股票生成所有索引字段"""
        from itertools import product

        name_lower = name.lower()

        # 使用 heteronym=True 获取所有读音
        all_py = pinyin(name, style=Style.TONE3, heteronym=True)
        # all_py: [['zha3ng', 'cha2ng'], ['dia4n'], ['ke1'], ['ji4']]

        # 主读音（每字第一个）
        primary_py = [pron[0] for pron in all_py if pron]
        pinyin_full = "".join(re.sub(r"\d", "", s) for s in primary_py)
        pinyin_initials = "".join(s[0] for s in primary_py if s)

        # 首字母（去重连续相同）
        initials_chars = [s[0] for s in primary_py if s]
        unique_initials: List[str] = []
        for ch in initials_chars:
            if not unique_initials or unique_initials[-1] != ch:
                unique_initials.append(ch)
        pinyin_first_char = "".join(unique_initials)

        # 多音字：生成所有可能的简拼组合
        alt_initials: Set[str] = set()
        has_heteronym = any(len(p) > 1 for p in all_py)
        if has_heteronym:
            # 每字取首字母的所有排列
            char_initials = [[pron[0] for pron in char_py if pron] for char_py in all_py]
            for combo in product(*char_initials):
                alt_initials.add("".join(combo))
            alt_initials.discard(pinyin_initials)  # 排除主读音

        # 2-gram 切分
        trigrams: Set[str] = set()
        for i in range(len(name) - 1):
            trigrams.add(name[i : i + 2])
        # 单字股名也加入
        if len(name) == 1:
            trigrams.add(name)

        return StockIndexEntry(
            code=code,
            name=name,
            market=market,
            name_lower=name_lower,
            pinyin_full=pinyin_full,
            pinyin_initials=pinyin_initials,
            pinyin_first_char=pinyin_first_char,
            pinyin_initials_alt=alt_initials,
            trigrams=trigrams,
        )

    def _build_tries(self) -> None:
        """构建三棵 Trie 树"""
        for i, entry in enumerate(self._entries):
            # 名称前缀
            self._name_trie.insert(entry.name, i)
            # 全拼前缀
            if entry.pinyin_full:
                self._pinyin_full_trie.insert(entry.pinyin_full, i)
            # 简拼前缀（含多音字变体）
            if entry.pinyin_initials:
                self._pinyin_initials_trie.insert(entry.pinyin_initials, i)
            for alt in entry.pinyin_initials_alt:
                if alt:
                    self._pinyin_initials_trie.insert(alt, i)

    def _build_trigram_index(self) -> None:
        """构建 2-gram 倒排索引"""
        for i, entry in enumerate(self._entries):
            for gram in entry.trigrams:
                if gram not in self._trigram_index:
                    self._trigram_index[gram] = set()
                self._trigram_index[gram].add(i)

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    def search(
        self, query: str, limit: int = 20
    ) -> List[Dict]:
        """
        主搜索入口。

        Args:
            query: 搜索关键词
            limit: 最大返回结果数

        Returns:
            搜索结果列表，每项含 code, name, market, match_type, match_segments, score
        """
        query = query.strip()
        if not query:
            return []

        limit = max(1, min(limit, 100))

        # 识别查询类型
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        is_digits = query.isdigit()
        is_lower_ascii = query.isascii() and query.islower() and not is_digits

        # 收集匹配结果：{entry_idx: (best_score, best_match_type, segments)}
        matches: Dict[int, Tuple[int, str, List[Dict]]] = {}

        if is_digits:
            # 代码搜索
            self._search_by_code(query, matches)
        elif is_lower_ascii and not has_chinese:
            # 纯拼音搜索
            self._search_by_pinyin(query, matches)
        elif has_chinese:
            # 中文名称搜索
            self._search_by_name(query, matches)
        else:
            # 混合查询，尝试所有方式
            if query.isdigit():
                self._search_by_code(query, matches)
            if has_chinese:
                self._search_by_name(query, matches)
            if is_lower_ascii or (query.isascii() and not query.isdigit()):
                self._search_by_pinyin(query, matches)

        # 排序：按得分降序，同分按代码升序
        sorted_indices = sorted(
            matches.keys(),
            key=lambda idx: (-matches[idx][0], self._entries[idx].code),
        )

        # 截取 limit 条并构造结果
        results: List[Dict] = []
        for idx in sorted_indices[:limit]:
            entry = self._entries[idx]
            score, match_type, segments = matches[idx]
            results.append({
                "code": entry.code,
                "name": entry.name,
                "market": entry.market,
                "match_type": match_type,
                "match_segments": segments,
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # 搜索子方法
    # ------------------------------------------------------------------

    def _search_by_code(self, query: str, matches: Dict) -> None:
        """代码精确/前缀匹配"""
        # 精确匹配
        if query in self._code_map:
            idx = self._code_map[query]
            segments = [{"field": "code", "start": 0, "end": len(query)}]
            self._set_match(matches, idx, 100, "code_exact", segments)

        # 前缀匹配
        for idx in self._sorted_by_code:
            if self._entries[idx].code.startswith(query):
                if idx not in matches:
                    segments = [{"field": "code", "start": 0, "end": len(query)}]
                    self._set_match(matches, idx, 80, "code_prefix", segments)

    def _search_by_name(self, query: str, matches: Dict) -> None:
        """中文名称匹配：前缀 + trigram"""
        # 名称前缀匹配（Trie）
        prefix_indices = self._name_trie.search_prefix(query)
        for idx in prefix_indices:
            entry = self._entries[idx]
            pos = entry.name.find(query)
            score = 90 + len(query)  # 前缀匹配基础分 + 长度加成
            segments = [{"field": "name", "start": pos, "end": pos + len(query)}]
            self._set_match(matches, idx, score, "name_prefix", segments)

        # Trigram 索引匹配
        if len(query) >= 2:
            # 将查询也切成 2-gram
            query_grams = {query[i : i + 2] for i in range(len(query) - 1)}
            trigram_hits: Dict[int, int] = {}  # idx → hit count
            for gram in query_grams:
                for idx in self._trigram_index.get(gram, set()):
                    trigram_hits[idx] = trigram_hits.get(idx, 0) + 1

            for idx, hit_count in trigram_hits.items():
                if idx not in matches:
                    entry = self._entries[idx]
                    pos = entry.name.find(query)
                    if pos < 0 and len(query) <= len(entry.name):
                        # 多 gram 命中但完整 query 不在名称中 → 仍然认为是 trigram 匹配
                        pass
                    score = 70 + hit_count  # trigram 匹配基础分 + 命中数加成
                    segments = self._build_name_segments(entry, query)
                    self._set_match(matches, idx, score, "name_trigram", segments)

        # 单汉字查询：遍历 trigram 索引
        if len(query) == 1:
            for idx in self._trigram_index.get(query, set()):
                if idx not in matches:
                    entry = self._entries[idx]
                    pos = entry.name.find(query)
                    score = 65
                    segments = [{"field": "name", "start": pos, "end": pos + 1}]
                    self._set_match(matches, idx, score, "name_trigram", segments)

    def _search_by_pinyin(self, query: str, matches: Dict) -> None:
        """拼音匹配：全拼、简拼、首字母"""
        query_lower = query.lower()

        # 全拼前缀匹配
        full_indices = self._pinyin_full_trie.search_prefix(query_lower)
        for idx in full_indices:
            if idx not in matches:
                entry = self._entries[idx]
                py_pos = entry.pinyin_full.find(query_lower)
                score = 60 + len(query_lower)
                segments = self._pinyin_to_name_segments(entry, py_pos, len(query_lower))
                self._set_match(matches, idx, score, "pinyin_full", segments)

        # 简拼前缀匹配
        init_indices = self._pinyin_initials_trie.search_prefix(query_lower)
        for idx in init_indices:
            if idx not in matches:
                entry = self._entries[idx]
                score = 50 + len(query_lower)
                segments = self._build_name_segments(entry, entry.name)
                self._set_match(matches, idx, score, "pinyin_initials", segments)

        # 首字母匹配（精确匹配首字母序列）
        if not matches:
            for idx in self._sorted_by_code:
                entry = self._entries[idx]
                if entry.pinyin_first_char == query_lower:
                    segments = self._build_name_segments(entry, entry.name)
                    self._set_match(matches, idx, 43, "pinyin_first_char", segments)
                elif (
                    entry.pinyin_first_char
                    and query_lower
                    and entry.pinyin_first_char.startswith(query_lower)
                ):
                    segments = self._build_name_segments(entry, entry.name)
                    self._set_match(
                        matches, idx, 40, "pinyin_first_char_prefix", segments
                    )

    @staticmethod
    def _set_match(
        matches: Dict,
        idx: int,
        score: int,
        match_type: str,
        segments: List[Dict],
    ) -> None:
        """设置或更新匹配结果（保留最高分）"""
        if idx not in matches or score > matches[idx][0]:
            matches[idx] = (score, match_type, segments)

    @staticmethod
    def _build_name_segments(entry: StockIndexEntry, keyword: str) -> List[Dict]:
        """生成名称中关键词的高亮区间"""
        segments: List[Dict] = []
        pos = entry.name.find(keyword)
        if pos >= 0:
            segments.append({"field": "name", "start": pos, "end": pos + len(keyword)})
        return segments

    @staticmethod
    def _pinyin_to_name_segments(
        entry: StockIndexEntry, py_start: int, py_len: int
    ) -> List[Dict]:
        """
        将拼音匹配位置映射回中文名称的高亮区间。

        简单策略：高亮整个名称，因为拼音到汉字的精确映射较复杂。
        """
        # 全拼匹配 → 高亮整个名称
        return [{"field": "name", "start": 0, "end": len(entry.name)}]


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_search_index: Optional[StockSearchIndex] = None


def get_search_index() -> StockSearchIndex:
    """获取搜索索引单例（懒加载）"""
    global _search_index
    if _search_index is None:
        _search_index = StockSearchIndex()
        _search_index.build()
    return _search_index


def refresh_search_index() -> int:
    """
    刷新搜索索引（原子替换）。

    Returns:
        新索引的条目数量
    """
    global _search_index
    new_index = StockSearchIndex()
    count = new_index.build()
    _search_index = new_index
    logger.info("搜索索引已刷新，共 %d 只股票", count)
    return count


def has_search_index() -> bool:
    """检查搜索索引是否已构建"""
    return _search_index is not None and len(_search_index._entries) > 0
