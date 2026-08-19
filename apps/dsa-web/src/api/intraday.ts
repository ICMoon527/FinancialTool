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
  metadata?: Record<string, any> | null;
}

export interface FiveMinKlinePoint {
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  timestamp: string;
}

export interface TiandaoSignal {
  signal_type: 'buy' | 'sell';
  trigger_time: string;
  price: number;
  reason: string;
}

export interface TiandaoSubChart {
  klines: FiveMinKlinePoint[];
  prev_day_klines: FiveMinKlinePoint[];
  jinzuan_line: IndicatorLinePoint[];
  jinniu_line: IndicatorLinePoint[];
  signals: TiandaoSignal[];
}

export interface IntradayDataResponse {
  stock_code: string;
  stock_name: string;
  date: string;
  kline_data: IntradayKlinePoint[];
  signals: IntradaySignal[];
  reference_lines: ReferenceLine[];
  indicator_sub_charts: IndicatorSubChart[];
  tiandao_sub_chart?: TiandaoSubChart | null;
  signal_summary: {
    buy_signals: number;
    sell_signals: number;
    total_signals: number;
    strong_signals: number;
    medium_signals: number;
    weak_signals: number;
    simulated_return_pct: number;
  };
  rsi_overbought?: number;
  rsi_oversold?: number;
  mfi_overbought?: number;
  mfi_oversold?: number;
  buy_weights?: Record<string, number>;
  sell_weights?: Record<string, number>;
  warm_up_summary?: Record<string, any> | null;
  warmup_info?: {
    enabled: boolean;
    last_klines_count: number;
    prev_date: string;
    klines?: IntradayKlinePoint[];
  } | null;
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
  pre_close: number;
  volume: number;
  timestamp: string;
  // 五档盘口（卖1→卖5，买1→买5，量单位为手）
  ask_prices: number[];    // [卖一价, 卖二价, 卖三价, 卖四价, 卖五价]
  ask_volumes: number[];   // [卖一量, 卖二量, 卖三量, 卖四量, 卖五量] (手)
  bid_prices: number[];    // [买一价, 买二价, 买三价, 买四价, 买五价]
  bid_volumes: number[];   // [买一量, 买二量, 买三量, 买四量, 买五量] (手)
  // 估值指标（可选）
  volume_ratio?: number | null;
  turnover_rate?: number | null;
  pe_ratio?: number | null;
  pb_ratio?: number | null;
}

export interface SignalAlert {
  stock_code: string;
  signal_type: 'buy' | 'sell';
  trigger_time: string;
  price: number;
}

export interface BatchStatusResponse {
  snapshots: Record<string, StockSnapshot>;
  current_updated: boolean;
  current_full_data: IntradayDataResponse | null;
  signal_alerts?: Record<string, SignalAlert | null> | null;
}

export interface BatchStatusRequest {
  stock_codes: string[];
  current_code: string;
  include_signals?: boolean;
}

export interface SimulatedTradeItem {
  buy_time: string;
  buy_price: number;
  sell_time: string;
  sell_price: number;
  return_pct: number;
}

export interface SimulationReportResponse {
  stock_code: string;
  total_klines: number;
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  total_trades: number;
  win_trades: number;
  lose_trades: number;
  win_rate: number;
  avg_return_pct: number;
  max_return_pct: number;
  min_return_pct: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  profit_factor: number;
  trades: SimulatedTradeItem[];
}

export interface TradingStatus {
  is_trading_day: boolean;
  is_trading_time: boolean;
  next_session_start: string | null;
}

export interface IntradayConfig {
  polling_interval_ms: number;
  batch_download_polling_interval_ms: number;
  screen_async_polling_interval_ms: number;
  signal_sound_cooldown_seconds: number;
  signal_sound_volume: number;
  trading_status: TradingStatus;
}

// ============================================================
// API 函数
// ============================================================

export async function getIntradayConfig(): Promise<IntradayConfig> {
  const defaultConfig: IntradayConfig = {
    polling_interval_ms: 30000,
    batch_download_polling_interval_ms: 1000,
    screen_async_polling_interval_ms: 1000,
    signal_sound_cooldown_seconds: 30,
    signal_sound_volume: 0.3,
    trading_status: {
      is_trading_day: false,
      is_trading_time: false,
      next_session_start: null,
    },
  };
  const resp = await fetch(`${API_BASE}/config`);
  if (!resp.ok) return defaultConfig;
  return resp.json();
}

export async function getIntradayData(
  stockCode: string,
  date?: string,
  strategy?: string,
  warmupEnabled?: boolean,
): Promise<IntradayDataResponse> {
  const params = new URLSearchParams();
  if (date) {
    params.set('date', date.replace(/-/g, ''));
  }
  if (strategy) {
    params.set('strategy', strategy);
  }
  if (warmupEnabled !== undefined) {
    params.set('warmup_enabled', String(warmupEnabled));
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
  includeSignals: boolean = false,
  existingKlineCount: number = 0,
  skipKlineFetch: boolean = false,
): Promise<BatchStatusResponse> {
  const resp = await fetch(`${API_BASE}/batch-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stock_codes: stockCodes,
      current_code: currentCode,
      include_signals: includeSignals,
      existing_kline_count: existingKlineCount,
      skip_kline_fetch: skipKlineFetch,
    }),
  });
  if (!resp.ok) throw new Error(`批量状态查询失败: ${resp.status}`);
  return resp.json();
}

export async function simulateTrading(
  stockCode: string,
): Promise<SimulationReportResponse> {
  const resp = await fetch(`${API_BASE}/${stockCode}/simulate-trading`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!resp.ok) throw new Error(`模拟交易失败: ${resp.status}`);
  return resp.json();
}

// ============================================================
// 批量下载分时数据
// ============================================================

export interface BatchDownloadStatus {
  task_id: string;
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'failed';
  total: number;
  completed: number;
  failed: number;
  skipped: number;
  current_code: string;
  current_name: string;
  elapsed_seconds: number;
  errors: { code: string; error: string }[];
  date: string;
  paused: boolean;
  waiting_retry: boolean;
  retry_countdown: number;
}

export async function startBatchDownload(
  date?: string,
  maxWorkers?: number,
  force?: boolean,
): Promise<BatchDownloadStatus> {
  const body: Record<string, any> = {};
  if (date) body.date = date;
  if (maxWorkers) body.max_workers = maxWorkers;
  if (force !== undefined) body.force = force;
  const resp = await fetch(`${API_BASE}/batch-download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`批量下载启动失败: ${resp.status}`);
  return resp.json();
}

export async function getBatchDownloadStatus(
  taskId?: string,
): Promise<BatchDownloadStatus> {
  // 添加时间戳防止浏览器缓存 GET 请求
  const ts = Date.now();
  const params = taskId
    ? `?task_id=${encodeURIComponent(taskId)}&_=${ts}`
    : `?_=${ts}`;
  const resp = await fetch(`${API_BASE}/batch-download/status${params}`, {
    headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
  });
  if (!resp.ok) throw new Error(`查询进度失败: ${resp.status}`);
  return resp.json();
}

export async function cancelBatchDownload(
  taskId: string,
): Promise<{ message: string }> {
  const resp = await fetch(`${API_BASE}/batch-download/cancel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });
  if (!resp.ok) throw new Error(`取消失败: ${resp.status}`);
  return resp.json();
}

export async function togglePauseBatchDownload(
  taskId: string,
): Promise<{ message: string; paused: boolean }> {
  const resp = await fetch(`${API_BASE}/batch-download/pause`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  });
  if (!resp.ok) throw new Error(`暂停/继续失败: ${resp.status}`);
  return resp.json();
}

export async function retryFailedBatchDownload(
  date?: string,
  maxWorkers?: number,
): Promise<BatchDownloadStatus> {
  const body: Record<string, any> = {};
  if (date) body.date = date;
  if (maxWorkers) body.max_workers = maxWorkers;
  const resp = await fetch(`${API_BASE}/batch-download/retry-failed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`失败重试启动失败: ${resp.status}`);
  return resp.json();
}

export interface FailedListItem {
  code: string;
  error_msg: string;
  retry_count: number;
}

export interface FailedListResponse {
  date: string;
  failed_list: FailedListItem[];
  count: number;
}

export async function getBatchDownloadFailedList(
  date?: string,
): Promise<FailedListResponse> {
  const ts = Date.now();
  const params = date
    ? `?date=${encodeURIComponent(date)}&_=${ts}`
    : `?_=${ts}`;
  const resp = await fetch(`${API_BASE}/batch-download/failed-list${params}`, {
    headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
  });
  if (!resp.ok) throw new Error(`查询失败列表失败: ${resp.status}`);
  return resp.json();
}
