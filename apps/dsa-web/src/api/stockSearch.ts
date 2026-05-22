import apiClient from './index';

/** 匹配高亮区间 */
export interface MatchSegment {
  field: 'name' | 'code';
  start: number;
  end: number;
}

/** 单条股票搜索结果 */
export interface StockSearchResult {
  code: string;
  name: string;
  market: string;
  match_type: string;
  match_segments: MatchSegment[];
  score: number;
}

/** 搜索响应 */
export interface StockSearchResponse {
  query: string;
  total: number;
  results: StockSearchResult[];
  time_ms: number;
}

/** 索引刷新响应 */
export interface StockSearchRefreshResponse {
  status: string;
  entry_count: number;
}

// ---------------------------------------------------------------------------
// 前端 LRU 缓存
// ---------------------------------------------------------------------------

interface CacheEntry {
  results: StockSearchResult[];
  ts: number;
}

const searchCache = new Map<string, CacheEntry>();
const CACHE_TTL = 5 * 60 * 1000; // 5 分钟
const CACHE_MAX = 100;

function cacheGet(key: string): StockSearchResult[] | null {
  const entry = searchCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL) {
    searchCache.delete(key);
    return null;
  }
  return entry.results;
}

function cacheSet(key: string, results: StockSearchResult[]): void {
  if (searchCache.size >= CACHE_MAX) {
    const oldest = searchCache.keys().next().value;
    if (oldest) searchCache.delete(oldest);
  }
  searchCache.set(key, { results, ts: Date.now() });
}

// ---------------------------------------------------------------------------
// API 函数
// ---------------------------------------------------------------------------

/**
 * 搜索股票
 * @param query 搜索关键词
 * @param limit 最大返回数（默认 20，最大 100）
 */
export async function searchStocks(
  query: string,
  limit: number = 20,
): Promise<StockSearchResult[]> {
  const key = query.trim().toLowerCase();
  if (!key) return [];

  // 检查前端缓存
  const cached = cacheGet(key);
  if (cached) return cached;

  const response = await apiClient.get<StockSearchResponse>(
    '/api/v1/stocks/search',
    { params: { q: query.trim(), limit: Math.max(1, Math.min(limit, 100)) } },
  );

  const results = response.data.results ?? [];
  cacheSet(key, results);
  return results;
}

/**
 * 刷新搜索索引
 */
export async function refreshSearchIndex(): Promise<StockSearchRefreshResponse> {
  const response = await apiClient.post<StockSearchRefreshResponse>(
    '/api/v1/stocks/search/refresh',
  );
  // 刷新后清空前端缓存
  searchCache.clear();
  return response.data;
}