/**
 * 分时做T API 模块
 *
 * 封装分时K线数据获取、做T信号和参考线的请求接口。
 */

const API_BASE = '/api/v1/intraday';

// ============================================================
// TypeScript 类型定义
// ============================================================

export interface IntradayKlinePoint {
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Amount?: number;
  AvgPrice?: number;
  timestamp: string;
  time?: string;
}

export interface WeightContribution {
  key: string;
  label: string;
  weight: number;
  triggered: boolean;
  score: number;
}

export interface IntradaySignal {
  stock_code: string;
  signal_type: 'buy' | 'sell' | 'hold';
  trigger_time: string;
  price: number;
  score: number;
  max_score: number;
  confidence: number;
  position_advice: string;
  reasoning: string;
  gravity_adjustment?: number;
  support_force?: number;
  pressure_force?: number;
  buy_weight_details?: WeightContribution[];
  sell_weight_details?: WeightContribution[];
}

export interface ReferenceLine {
  id: string;
  label: string;
  price: number;
  category: string;
  color: string;
  style: string;
  base_weight: number;
}

export interface IndicatorLinePoint {
  time: string;
  value: number;
}

export interface IndicatorLine {
  name: string;
  label: string;
  color: string;
  data: IndicatorLinePoint[];
}

export interface IndicatorSubChart {
  id: string;
  label: string;
  height: number;
  lines: IndicatorLine[];
  signal_text: string;
}

export interface IntradayDataResponse {
  stock_code: string;
  stock_name: string;
  date: string;
  kline_data: IntradayKlinePoint[];
  signals: IntradaySignal[];
  reference_lines: ReferenceLine[];
  indicator_sub_charts: IndicatorSubChart[];
  signal_summary: {
    buy_signals: number;
    sell_signals: number;
    total_signals: number;
    strong_signals: number;
    medium_signals: number;
    weak_signals: number;
    simulated_return_pct: number;
  };
}

export interface SearchHistoryItem {
  id: number;
  stock_code: string;
  stock_name: string;
  date: string;
  search_time: string;
}

export interface SearchHistoryResponse {
  items: SearchHistoryItem[];
  total: number;
}

export interface StockSnapshot {
  stock_code: string;
  stock_name: string;
  latest_price: number;
  change_pct: number;
  open_price: number;
  high: number;
  low: number;
  timestamp: string;
}

export interface BatchStatusResponse {
  snapshots: Record<string, StockSnapshot>;
  current_updated: boolean;
  current_full_data: IntradayDataResponse | null;
}

// ============================================================
// API 函数
// ============================================================

export async function getIntradayData(
  stockCode: string,
  date?: string,
): Promise<IntradayDataResponse> {
  const params = new URLSearchParams();
  if (date) {
    params.set('date', date.replace(/-/g, ''));
  }
  const query = params.toString();
  const url = `${API_BASE}/data/${stockCode}${query ? `?${query}` : ''}`;

  const resp = await fetch(url);
  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ message: resp.statusText }));
    throw new Error(error.detail?.message || error.message || `请求失败 (${resp.status})`);
  }
  return resp.json();
}

export async function getSearchHistory(
  limit: number = 20,
): Promise<SearchHistoryResponse> {
  const resp = await fetch(`${API_BASE}/history?limit=${limit}`);
  if (!resp.ok) return { items: [], total: 0 };
  return resp.json();
}

export async function saveSearchHistory(
  stockCode: string,
  stockName: string = '',
  date: string = '',
): Promise<SearchHistoryItem> {
  const resp = await fetch(`${API_BASE}/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stock_code: stockCode, stock_name: stockName, date }),
  });
  if (!resp.ok) throw new Error('保存搜索历史失败');
  return resp.json();
}

export async function deleteSearchHistory(id: number): Promise<boolean> {
  const resp = await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
  return resp.ok;
}

export async function updateSearchHistoryTimestamp(id: number): Promise<SearchHistoryItem> {
  const resp = await fetch(`${API_BASE}/history/${id}/timestamp`, { method: 'PUT' });
  if (!resp.ok) throw new Error('更新时间戳失败');
  return resp.json();
}

export async function getBatchStatus(
  stockCodes: string[],
  currentCode: string,
): Promise<BatchStatusResponse> {
  const resp = await fetch(`${API_BASE}/batch-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stock_codes: stockCodes, current_code: currentCode }),
  });
  if (!resp.ok) throw new Error(`批量状态查询失败: ${resp.status}`);
  return resp.json();
}
