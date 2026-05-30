import type React from 'react';
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as lightweightCharts from 'lightweight-charts';
import { createSeriesMarkers } from 'lightweight-charts';
import type { Time } from 'lightweight-charts';
import {
  getIntradayData,
  getSearchHistory,
  saveSearchHistory,
  deleteSearchHistory,
  getBatchStatus,
  updateSearchHistoryTimestamp,
  simulateTrading,
  startBatchDownload,
  getBatchDownloadStatus,
  cancelBatchDownload,
  togglePauseBatchDownload,
  getIntradayConfig,
  type IntradayDataResponse,
  type IntradayKlinePoint,
  type SearchHistoryItem,
  type SimulationReportResponse,
  type StockSnapshot,
  type BatchDownloadStatus,
} from '../api/intraday';
import type { WeightContribution, IntradaySignal } from '../api/intraday';
import { validateStockCode } from '../utils/validation';
import { Card, StockSearchInput } from '../components/common';
import { CrosshairSyncEngine } from './CrosshairSyncEngine';

/** 计算成交量的 N 日简单移动平均（过滤掉不足N天的数据点） */
function calculateVolumeMA(
  volumeData: Array<{ time: any; value: number }>,
  days: number,
): Array<{ time: any; value: number }> {
  const result: Array<{ time: any; value: number }> = [];
  for (let idx = days - 1; idx < volumeData.length; idx++) {
    let sum = 0;
    for (let i = idx - days + 1; i <= idx; i++) {
      sum += volumeData[i].value;
    }
    result.push({ time: volumeData[idx].time, value: sum / days });
  }
  return result;
}

const CHART_HEIGHT = 460;

function timeFormatter(time: Time) {
  const d = new Date((time as number) * 1000);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function formatDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function signalLevel(confidence: number): { label: string; color: string } {
  if (confidence >= 0.75) return { label: '强', color: '#FF4444' };
  if (confidence >= 0.50) return { label: '中', color: '#FFAA00' };
  return { label: '弱', color: '#888888' };
}

/** 信号文本颜色：买入(红)/卖出(绿)/中性 */
function signalColor(text: string): { fg: string; bg: string } {
  const buy = /买入|买回|流入|金叉|控盘中|反弹|超卖|上穿|底背离/;
  const sell = /卖出|流出|死叉|弱控盘|未控盘|出货|破位|超买|下穿|顶背离/;
  if (buy.test(text)) return { fg: '#FF6644', bg: 'rgba(255,100,68,0.12)' };
  if (sell.test(text)) return { fg: '#44DD44', bg: 'rgba(68,221,68,0.12)' };
  return { fg: '#00D4FF', bg: 'rgba(0,212,255,0.10)' };
}

/** 将 hex 颜色（支持 #RGB / #RRGGBB / #RRGGBBAA）转换为带 alpha 的 hex 格式 #RRGGBBAA */
function hexToRgba(hex: string, alpha: number): string {
  let h = hex.replace('#', '');
  // 展开简写 #RGB → RRGGBB
  if (h.length === 3) {
    h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  } else if (h.length === 4) {
    h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2] + h[3] + h[3];
  }
  // 取前6位作为 RGB，忽略已有的 alpha 通道
  const rgb = h.substring(0, 6);
  const a = Math.round(alpha * 255).toString(16).padStart(2, '0');
  return `#${rgb}${a}`;
}

/**
 * 将分时K线数据转换为 lightweight-charts 可用格式
 * dateStr: 当前查询日期 "YYYY-MM-DD"
 */
function convertKlineData(
  klineData: IntradayKlinePoint[],
  dateStr: string,
): Array<{
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rawTimestamp: string;
}> {
  const result: Array<{
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    avgPrice?: number;
    rawTimestamp: string;
  }> = [];

  for (const k of klineData) {
    const ts = k.timestamp || k.time || '';
    const utcMs = parseTimestamp(ts, dateStr);
    const unixSec = Math.floor(utcMs / 1000);
    result.push({
      time: unixSec,
      open: k.Open,
      high: k.High,
      low: k.Low,
      close: k.Close,
      volume: k.Volume,
      avgPrice: k.AvgPrice,
      rawTimestamp: ts,
    });
  }
  result.sort((a, b) => a.time - b.time);
  return result;
}

/**
 * 确保子图数据时间范围与K线对齐
 * 如果数据首点晚于首根K线，前插一个 null 值点（lightweight-charts 中 null = 断点不绘制）
 * 防止子图 auto-fit 到较晚的起始时间，进而反向传播到主图
 */
function padDataStart(pts: any[], firstKlineTime: number): any[] {
  if (firstKlineTime > 0 && pts.length > 0 && (pts[0].time as number) > firstKlineTime) {
    return [{ time: firstKlineTime, value: null }, ...pts];
  }
  return pts;
}

const _tsCache = new Map<string, number>();

/**
 * 解析 akshare 返回的时间字符串为 UTC 毫秒
 */
function parseTimestamp(tsStr: string, dateStr: string): number {
  if (!tsStr) return 0;

  const cacheKey = `${tsStr}||${dateStr}`;
  const cached = _tsCache.get(cacheKey);
  if (cached !== undefined) return cached;

  const s = tsStr.trim();

  // 带日期的完整格式 "2024-01-01 09:30:00"
  const fullMatch = s.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (fullMatch) {
    const d = new Date(
      Number(fullMatch[1]),
      Number(fullMatch[2]) - 1,
      Number(fullMatch[3]),
      Number(fullMatch[4]),
      Number(fullMatch[5]),
      Number(fullMatch[6]) || 0,
    );
    _tsCache.set(cacheKey, d.getTime());
    return d.getTime();
  }

  // ISO 格式
  const isoMatch = s.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  if (isoMatch) {
    const d = new Date(
      Number(isoMatch[1]),
      Number(isoMatch[2]) - 1,
      Number(isoMatch[3]),
      Number(isoMatch[4]),
      Number(isoMatch[5]),
      Number(isoMatch[6]),
    );
    _tsCache.set(cacheKey, d.getTime());
    return d.getTime();
  }

  // 纯时间 "09:30" / "09:30:00"
  const timeMatch = s.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (timeMatch) {
    let hour = Number(timeMatch[1]);
    const min = Number(timeMatch[2]);
    const sec = Number(timeMatch[3]) || 0;
    if (hour < 0 || hour > 23) hour = 0;

    const [y, mon, day] = dateStr.split('-').map(Number);
    if (y && mon && day) {
      const d = new Date(y, mon - 1, day, hour, min, sec);
      _tsCache.set(cacheKey, d.getTime());
      return d.getTime();
    }

    // fallback: 用当天的日期
    const today = new Date();
    const d = new Date(today.getFullYear(), today.getMonth(), today.getDate(), hour, min, sec);
    _tsCache.set(cacheKey, d.getTime());
    return d.getTime();
  }

  // 数字类时间戳
  const num = Number(s);
  if (!Number.isNaN(num)) {
    // 秒级时间戳
    if (num > 1e9) {
      _tsCache.set(cacheKey, num * 1000);
      return num * 1000;
    }
    // 毫秒级
    _tsCache.set(cacheKey, num);
    return num;
  }

  _tsCache.set(cacheKey, 0);
  return 0;
}

/**
 * 计算五档盘口买卖总手数
 * @param volumes 五档量数组（手）
 * @returns 总量（手）
 */
function calcDepthTotal(volumes: number[]): number {
  return volumes.reduce((sum, v) => sum + v, 0);
}

const IntradayPage: React.FC = () => {
  // ── 状态 ──
  const [stockCode, setStockCode] = useState('');
  const [inputError, setInputError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [intradayData, setIntradayData] = useState<IntradayDataResponse | null>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<number | null>(null);
  const [historySnapshots, setHistorySnapshots] = useState<Record<string, StockSnapshot>>({});
  const [signalBells, setSignalBells] = useState<Record<string, { type: 'buy' | 'sell'; time: string; price: number }>>({});
  const [configLoaded, setConfigLoaded] = useState(false);  // 轮询配置是否已从后端加载
  const signalBellsRef = useRef(signalBells);
  signalBellsRef.current = signalBells;
  const seenSignalTimesRef = useRef<Record<string, string>>({});
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [priceRangeEnabled, setPriceRangeEnabled] = useState(false);
  const todayDateStr = useMemo(() => formatDate(new Date()), []);
  const [crosshairSignals, setCrosshairSignals] = useState<Record<string, string>>({});
  const [isCrosshairActive, setIsCrosshairActive] = useState(false);
  const [crosshairMacdSum, setCrosshairMacdSum] = useState<number | null>(null);
  const [crosshairMacdDiff, setCrosshairMacdDiff] = useState<number | null>(null);
  const [crosshairRsiValue, setCrosshairRsiValue] = useState<number | null>(null);
  const [crosshairKdjKValue, setCrosshairKdjKValue] = useState<number | null>(null);
  const [crosshairKdjDValue, setCrosshairKdjDValue] = useState<number | null>(null);
  const [crosshairKdjJValue, setCrosshairKdjJValue] = useState<number | null>(null);
  const [crosshairDeviationPct, setCrosshairDeviationPct] = useState<number | null>(null);
  const [crosshairMa5DevPct, setCrosshairMa5DevPct] = useState<number | null>(null);
  const [warmupEnabled, setWarmupEnabled] = useState(true);
  const [hoveredWeightDetails, setHoveredWeightDetails] = useState<{
    buy: WeightContribution[];
    sell: WeightContribution[];
    supportForce: number;
    pressureForce: number;
    signalType: string;
  } | null>(null);

  // 模拟交易盈亏报告
  const [simulationReport, setSimulationReport] = useState<SimulationReportResponse | null>(null);
  const [simulationLoading, setSimulationLoading] = useState(false);

  // 复盘功能
  const [isReplaying, setIsReplaying] = useState(false);
  const [isReplayPaused, setIsReplayPaused] = useState(false);
  const isReplayingRef = useRef(false);
  const replayIndexRef = useRef<number>(0);
  const replayTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fullKlineBackupRef = useRef<any[] | null>(null);
  const fullSubChartDataBackupRef = useRef<any | null>(null);
  const indicatorSubChartsBackupRef = useRef<any[] | null>(null);
  const signalsBackupRef = useRef<IntradaySignal[] | null>(null);
  const isTencentDataRef = useRef(false);

  useEffect(() => {
    isReplayingRef.current = isReplaying;
  }, [isReplaying]);

  // 批量下载分时数据
  const [batchDownload, setBatchDownload] = useState<BatchDownloadStatus | null>(null);
  const [showBatchModal, setShowBatchModal] = useState(false);
  const batchDownloadRef = useRef<BatchDownloadStatus | null>(null);
  batchDownloadRef.current = batchDownload;
  const batchPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Canvas 进度条引用：直接用 JS 绘制，完全绕过 React/CSS 渲染管线
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const progressTextRef = useRef<HTMLDivElement | null>(null);

  // Canvas 绘制进度条函数（纯 JS，不受 React 影响）
  const drawProgressBar = useCallback((completed: number, total: number, status: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    // 如果 canvas 尚未布局（宽度为 0），延迟一帧重试
    if (w <= 0 || h <= 0) {
      requestAnimationFrame(() => drawProgressBar(completed, total, status));
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const radius = h / 2;
    const pct = total > 0 ? Math.max(0.008, completed / total) : 0;

    // 背景
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.moveTo(radius, 0);
    ctx.arcTo(w, 0, w, h, radius);
    ctx.arcTo(w, h, 0, h, radius);
    ctx.arcTo(0, h, 0, 0, radius);
    ctx.arcTo(0, 0, w, 0, radius);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fill();

    // 进度
    const progressW = Math.max(radius * 2, w * pct);
    ctx.beginPath();
    ctx.moveTo(radius, 0);
    ctx.arcTo(progressW, 0, progressW, h, radius);
    ctx.arcTo(progressW, h, 0, h, radius);
    ctx.arcTo(0, h, 0, 0, radius);
    ctx.arcTo(0, 0, progressW, 0, radius);
    ctx.closePath();
    ctx.fillStyle = status === 'completed' ? '#34d399' :
                    status === 'cancelled' ? 'rgba(234,179,8,0.6)' :
                    status === 'failed' ? 'rgba(239,68,68,0.6)' :
                    '#06b6d4';
    ctx.fill();
  }, []);

  // 当 batchDownload 变化时重绘 Canvas（处理初始渲染和状态颜色切换）
  useEffect(() => {
    if (batchDownload && batchDownload.total > 0) {
      requestAnimationFrame(() => {
        drawProgressBar(batchDownload.completed, batchDownload.total, batchDownload.status);
      });
    }
  }, [batchDownload?.completed, batchDownload?.total, batchDownload?.status, drawProgressBar]);

  // 渲染计数器（调试用：验证组件是否在 setBatchDownload 后重新渲染）
  const renderCountRef = useRef(0);
  renderCountRef.current += 1;
  if (renderCountRef.current % 20 === 0) {
    console.log(
      `[Render #${renderCountRef.current}] batchDownload:`,
      batchDownload ? `${batchDownload.completed}/${batchDownload.total} ${batchDownload.status}` : 'null'
    );
  }

  const handleStartBatchDownload = useCallback(async () => {
    setShowBatchModal(true);
    try {
      const status = await startBatchDownload(undefined, undefined, true);
      setBatchDownload(status);
      return status.task_id;
    } catch (err: any) {
      console.error('批量下载启动失败:', err);
      setShowBatchModal(false);
    }
  }, []);

  const handleCancelBatchDownload = useCallback(async () => {
    const task = batchDownloadRef.current;
    if (!task?.task_id) return;
    try {
      await cancelBatchDownload(task.task_id);
    } catch {
      // 取消请求可能失败，轮询会自动更新状态
    }
  }, []);

  const handleTogglePause = useCallback(async () => {
    const task = batchDownloadRef.current;
    if (!task?.task_id) return;
    try {
      const result = await togglePauseBatchDownload(task.task_id);
      // 立即更新本地状态（轮询也会同步，但这里即时反馈用户体验更好）
      setBatchDownload(prev => prev ? { ...prev, paused: result.paused } : null);
    } catch {
      // 请求失败，轮询会自动同步状态
    }
  }, []);

  // 批量下载进度轮询
  useEffect(() => {
    if (batchDownload?.status === 'running') {
      if (batchPollingRef.current) return;
      const taskId = batchDownload.task_id;
      console.log('[BatchDownload] 开始轮询, taskId:', taskId);
      batchPollingRef.current = setInterval(async () => {
        try {
          const status = await getBatchDownloadStatus(taskId);
          console.log('[BatchDownload] 轮询结果:', status.status, `${status.completed}/${status.total}`);
          setBatchDownload(status);
          // 用 requestAnimationFrame + Canvas 绘制，完全绕过 React 渲染管线
          requestAnimationFrame(() => {
            if (status.total > 0) {
              drawProgressBar(status.completed, status.total, status.status);
            }
            if (progressTextRef.current) {
              progressTextRef.current.textContent =
                `${status.completed} / ${status.total} (${Math.round(status.completed / status.total * 100)}%)`;
            }
          });
          if (status.status !== 'running') {
            if (batchPollingRef.current) {
              clearInterval(batchPollingRef.current);
              batchPollingRef.current = null;
            }
          }
        } catch (err) {
          console.warn('[BatchDownload] 轮询失败:', err);
        }
      }, batchPollingIntervalRef.current);
    } else {
      if (batchPollingRef.current) {
        clearInterval(batchPollingRef.current);
        batchPollingRef.current = null;
      }
    }
    return () => {
      if (batchPollingRef.current) {
        clearInterval(batchPollingRef.current);
        batchPollingRef.current = null;
      }
      if (replayTimerRef.current) {
        clearInterval(replayTimerRef.current);
        replayTimerRef.current = null;
      }
    };
  }, [batchDownload?.status]);

  // 存储从API返回的所有信号
  const allSignalsRef = useRef<IntradaySignal[]>([]);
  const filteredSignalsRef = useRef<IntradaySignal[]>([]);
  const macdMetadataRef = useRef<Record<string, any> | null>(null);

  // 信号筛选状态
  const [activeFilters, setActiveFilters] = useState<Set<string>>(new Set());

  // ── 图表引用 ──
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const volumeContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const volumeChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candleSeriesRef = useRef<lightweightCharts.ISeriesApi<'Candlestick'> | lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<lightweightCharts.ISeriesApi<'Histogram'> | null>(null);
  const volume5MASeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const volume10MASeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const refLineSeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'>[]>([]);
  const refLinePriceLinesRef = useRef<lightweightCharts.IPriceLine[]>([]);
  const chipAreaRef = useRef<lightweightCharts.ISeriesApi<'Baseline'> | null>(null);
  const avgPriceLineRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const priceRangeSeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const seriesMarkersRef = useRef<any>(null);
  const timeSyncSubRef = useRef<(() => void) | null>(null);
  const isTimeSyncingRef = useRef(false);
  const isInitialRenderRef = useRef(false);
  const currentDateRef = useRef(todayDateStr);
  const priceRangeEnabledRef = useRef(false);
  const syncEngineRef = useRef(new CrosshairSyncEngine());
  const currentCrosshairTimeRef = useRef<Time | null>(null);
  const klineRawDataRef = useRef<any[]>([]);
  const crosshairSignalRef = useRef<Record<string, string>>({});
  const crosshairSubsRef = useRef<Array<any>>([]);

  // 指标子图容器 refs
  const absorptionContainerRef = useRef<HTMLDivElement>(null);
  const macdContainerRef = useRef<HTMLDivElement>(null);
  const rsiContainerRef = useRef<HTMLDivElement>(null);
  const kdjContainerRef = useRef<HTMLDivElement>(null);
  const mfiContainerRef = useRef<HTMLDivElement>(null);

  // 指标子图实例管理
  const indicatorChartsRef = useRef<Map<string, lightweightCharts.IChartApi>>(new Map());
  const indicatorSeriesRef = useRef<Map<string, any[]>>(new Map());

  currentDateRef.current = todayDateStr;
  priceRangeEnabledRef.current = priceRangeEnabled;

  // 加载搜索历史
  const loadHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const resp = await getSearchHistory(50);
      setSearchHistory(resp.items);
    } catch (err) {
      console.error('加载搜索历史失败:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // ── 搜索历史股票实时行情轮询（仅盘中） ──
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollingIntervalRef = useRef(30000);      // 从后端配置动态获取
  const batchPollingIntervalRef = useRef(1000);  // 批量下载进度轮询间隔
  const searchHistoryRef = useRef<SearchHistoryItem[]>([]);
  searchHistoryRef.current = searchHistory;

  const isTradingTime = useCallback(() => {
    const now = new Date();
    const day = now.getDay();
    if (day === 0 || day === 6) return false;
    const totalMinutes = now.getHours() * 60 + now.getMinutes();
    // 9:30-11:30 和 13:00-15:00，排除午休
    return (totalMinutes >= 570 && totalMinutes <= 690) || (totalMinutes >= 780 && totalMinutes <= 900);
  }, []);

  const fetchAndUpdate = useCallback(async (includeSignals: boolean) => {
    const codes = searchHistoryRef.current.map(h => h.stock_code);
    if (codes.length === 0) return;
    const requestedCode = intradayData?.stock_code || '';
    try {
      const resp = await getBatchStatus(codes, requestedCode, includeSignals);
      setHistorySnapshots(resp.snapshots);
      if (resp.current_updated && resp.current_full_data && requestedCode) {
        setIntradayData(prev => {
          if (prev?.stock_code === requestedCode) {
            return resp.current_full_data;
          }
          return prev;
        });
      }
      // 处理信号铃铛（从同一响应中提取）
      if (resp.signal_alerts) {
        const seen = seenSignalTimesRef.current;
        const newBells: Record<string, { type: 'buy' | 'sell'; time: string; price: number }> = {};
        for (const [code, alert] of Object.entries(resp.signal_alerts)) {
          if (!alert) continue;
          if (code === requestedCode) continue;
          const lastSeen = seen[code];
          if (alert.trigger_time !== lastSeen) {
            newBells[code] = {
              type: alert.signal_type,
              time: alert.trigger_time,
              price: alert.price,
            };
          }
        }
        if (Object.keys(newBells).length > 0) {
          setSignalBells(prev => ({ ...prev, ...newBells }));
        }
      }
    } catch {
      // 静默失败，不打扰用户
    }
  }, [intradayData?.stock_code]);

  const fetchAndUpdateRef = useRef(fetchAndUpdate);
  fetchAndUpdateRef.current = fetchAndUpdate;

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    stopPolling();
    if (!isTradingTime()) return; // 非交易时间不启动轮询
    console.log('[Polling] 启动行情轮询, 间隔:', pollingIntervalRef.current + 'ms');
    pollingRef.current = setInterval(() => {
      if (!isTradingTime()) {
        stopPolling(); // 收盘后自动停止
        return;
      }
      fetchAndUpdateRef.current(true);
    }, pollingIntervalRef.current);
  }, [stopPolling, isTradingTime]);

  // 首次加载时自动展示搜索历史第一条标的
  const autoLoadedRef = useRef(false);

  useEffect(() => {
    if (autoLoadedRef.current) return;
    if (searchHistory.length > 0 && !intradayData) {
      autoLoadedRef.current = true;
      const first = searchHistory[0];
      setStockCode(first.stock_code);
      setSelectedHistoryId(first.id);
      setIsLoading(true);
      const dateParam = todayDateStr.replace(/-/g, '');
      const loadData = async () => {
        // 盘中优先从缓存/DB读取（轮询已保证数据最新）
        if (isTradingTime()) {
          try {
            const data = await getIntradayData(first.stock_code, dateParam, 'cache_only', warmupEnabled);
            return data;
          } catch (e: any) {
            // cache_only 无数据时降级走完整API链路
            console.log('缓存无数据，降级走API:', e.message);
          }
        }
        // 盘后直接走 full 链路
        return getIntradayData(first.stock_code, dateParam, 'full', warmupEnabled);
      };
      loadData()
        .then(data => {
          setIntradayData(data);
          setInputError(undefined);
          return data;
        })
        .then(() => {
          // 自动加载完成后获取侧边栏快照（不含信号检测，避免与 auto-load 竞争资源）
          if (searchHistory.length > 0) {
            fetchAndUpdate(false);
          }
        })
        .catch(err => {
          console.error('自动加载分时数据失败:', err);
          setInputError(err.message || '获取数据失败');
        })
        .finally(() => setIsLoading(false));
    }
  }, [searchHistory.length, intradayData, todayDateStr, isTradingTime, warmupEnabled]);

  // 从后端获取轮询配置（页面加载时调用一次）
  useEffect(() => {
    getIntradayConfig().then(cfg => {
      pollingIntervalRef.current = cfg.polling_interval_ms;
      batchPollingIntervalRef.current = cfg.batch_download_polling_interval_ms;
      console.log('[PollingConfig] 从后端获取轮询配置:', {
        polling: cfg.polling_interval_ms + 'ms',
        batchDownload: cfg.batch_download_polling_interval_ms + 'ms',
        screenAsync: cfg.screen_async_polling_interval_ms + 'ms',
      });
    }).catch((err) => {
      // 降级使用默认值，已在 ref 初始化时设置
      console.warn('[PollingConfig] 获取后端配置失败，使用默认值:', err);
    }).finally(() => {
      setConfigLoaded(true);
    });
  }, []);

  // 盘中轮询（等待配置加载完毕后才启动）
  useEffect(() => {
    if (!configLoaded) return;

    startPolling();

    const handleVisibility = () => {
      if (document.hidden) {
        stopPolling();
      } else if (isTradingTime()) {
        fetchAndUpdateRef.current(true); // 恢复时立即刷新含信号
        startPolling(); // 仅盘中会启动
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [startPolling, stopPolling, configLoaded]);

  // ── 初始化图表 ──
  useEffect(() => {
    if (!chartContainerRef.current || !volumeContainerRef.current) return;
    if (chartRef.current) return;

    // 主K线图

    const chart = lightweightCharts.createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: CHART_HEIGHT,
      layout: {
        background: { type: 'solid', color: '#1a1a2e' } as any,
        textColor: '#d1d4dc',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      rightPriceScale: {
        scaleMargins: { top: 0.1, bottom: 0.1 },
        borderVisible: false,
      },
      crosshair: {
        mode: 1,
        vertLine: { color: '#9B7DFF', width: 1, style: 2 },
        horzLine: { color: '#9B7DFF', width: 1, style: 2 },
      },
      localization: {
        timeFormatter,
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          const d = new Date(time * 1000);
          const h = String(d.getHours()).padStart(2, '0');
          const m = String(d.getMinutes()).padStart(2, '0');
          return `${h}:${m}`;
        },
        barSpacing: 8,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      handleScroll: {
        mouseWheel: false,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        mouseWheel: true,
        axisPressedMouseMove: false,
        pinch: true,
      },
    });

    const candleSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
      upColor: '#FF4444',
      downColor: '#44AA44',
      borderDownColor: '#44AA44',
      borderUpColor: '#FF4444',
      wickDownColor: '#44AA44',
      wickUpColor: '#FF4444',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // 成交量子图
    const volChart = lightweightCharts.createChart(volumeContainerRef.current, {
      width: volumeContainerRef.current.clientWidth,
      height: 100,
      layout: {
        background: { type: 'solid', color: '#1a1a2e' } as any,
        textColor: '#d1d4dc',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      rightPriceScale: {
        scaleMargins: { top: 0.05, bottom: 0.05 },
        borderVisible: false,
      },
      crosshair: { mode: 0 },
      localization: {
        timeFormatter,
      },
      timeScale: {
        timeVisible: false,
        secondsVisible: false,
        tickMarkFormatter: () => '',
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      handleScroll: {
        mouseWheel: false,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        mouseWheel: true,
        axisPressedMouseMove: false,
        pinch: true,
      },
    });

    const volSeries = volChart.addSeries(lightweightCharts.HistogramSeries, {
      color: '#66666688',
      priceFormat: { type: 'volume', precision: 0, minMove: 1 },
    });

    // 五日平均成交量线（白色）
    const vol5ma = volChart.addSeries(lightweightCharts.LineSeries, {
      color: '#FFFFFF',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    volume5MASeriesRef.current = vol5ma;

    // 十日平均成交量线（金色）
    const vol10ma = volChart.addSeries(lightweightCharts.LineSeries, {
      color: '#FFD700',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    volume10MASeriesRef.current = vol10ma;

    volumeChartRef.current = volChart;
    volumeSeriesRef.current = volSeries;

    // 同步主图和成交量图的时间轴范围
    const syncTimeRange = () => {
      if (isTimeSyncingRef.current) return;
      isTimeSyncingRef.current = true;
      try {
        const range = chart.timeScale().getVisibleRange();
        if (range) {
          volChart.timeScale().setVisibleRange(range);
          // 同步所有指标子图
          indicatorChartsRef.current.forEach((ic) => {
            try { ic.timeScale().setVisibleRange(range); } catch (e) { /* ignore */ }
          });
        }
      } catch (e) { /* ignore */ }
      setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
    };

    chart.timeScale().subscribeVisibleTimeRangeChange(syncTimeRange);
    timeSyncSubRef.current = () => {
      chart.timeScale().unsubscribeVisibleTimeRangeChange(syncTimeRange);
    };

    // 响应窗口尺寸
    const mainRo = new ResizeObserver((entries) => {
      if (!chartRef.current || entries.length === 0) return;
      const { width } = entries[0].contentRect;
      chartRef.current.applyOptions({ width, height: CHART_HEIGHT });
    });
    mainRo.observe(chartContainerRef.current);

    const volRo = new ResizeObserver((entries) => {
      if (!volumeChartRef.current || entries.length === 0) return;
      const { width } = entries[0].contentRect;
      volumeChartRef.current.applyOptions({ width, height: 100 });
    });
    volRo.observe(volumeContainerRef.current);

    return () => {
      mainRo.disconnect();
      volRo.disconnect();
      if (timeSyncSubRef.current) timeSyncSubRef.current();
      try { chart.remove(); } catch (e) { /* ignore */ }
      try { volChart.remove(); } catch (e) { /* ignore */ }
      chartRef.current = null;
      volumeChartRef.current = null;
      volume5MASeriesRef.current = null;
      volume10MASeriesRef.current = null;
      // 清理指标子图
      indicatorChartsRef.current.forEach((c) => {
        try { c.remove(); } catch (e) { /* ignore */ }
      });
      indicatorChartsRef.current.clear();
      indicatorSeriesRef.current.clear();
    };
  }, []);

  // ── 前端信号计算（无反函数，基于已加载的完整指标线数据切片） ──

  // 存储每根K线的指标值快照，在 renderData 时填充
  const snapshotRef = useRef<Array<{
    time: number;
    dominant_power: number;
    absorption: number;
    close: number;
    ma5: number;
    ma20: number;
    deviation_pct: number;
    ma5_dev_pct: number;
    absorption_active: boolean;
    distribution_active: boolean;
    price_above_ma5: boolean;
    price_above_ma20: boolean;
    price_cross_ma5_up: boolean;
    price_cross_ma5_down: boolean;
    deviation_oversold: boolean;
    deviation_overbought: boolean;
    deviation_narrowing: boolean;
    deviation_peaking: boolean;
    DIF: number;
    DEA: number;
    MACD_Bar: number;
    macd_golden_cross: boolean;
    macd_death_cross: boolean;
    macd_bullish_weakening: boolean;
    macd_bearish_recovering: boolean;
    macd_bar_sum: number;
    macd_bar_diff: number;
    RSI: number;
    K: number;
    D: number;
    J: number;
    rsi_oversold: boolean;
    rsi_overbought: boolean;
    mfi_value: number;
    mfi_oversold: boolean;
    mfi_overbought: boolean;
  }>>([]);

  const weightsConfigRef = useRef<{ buy: Record<string, number>; sell: Record<string, number> }>({
    buy: {},
    sell: {},
  });

  /** lookback=3 设计意图:
   *  穿越事件发生时, 右侧面板连续3根K线保持显示金叉/死叉, 让用户有足够时间注意到状态变化。
   *  这仅是展示层面的效果, 不影响买卖信号权重计算(权重字段macd_golden_cross为单根事件)。 */
  const _crossUpTs = (aArr: number[], bArr: number[], lookback: number = 3): boolean => {
    if (aArr.length < 2 || bArr.length < 2) return false;
    const start = Math.max(0, aArr.length - lookback);
    for (let i = start; i < aArr.length; i++) {
      if (i === 0) continue;
      if (aArr[i - 1] <= bArr[i - 1] && aArr[i] > bArr[i]) return true;
    }
    return false;
  };

  /** lookback=3 同上, 与_crossUpTs对称 */
  const _crossDownTs = (aArr: number[], bArr: number[], lookback: number = 3): boolean => {
    if (aArr.length < 2 || bArr.length < 2) return false;
    const start = Math.max(0, aArr.length - lookback);
    for (let i = start; i < aArr.length; i++) {
      if (i === 0) continue;
      if (aArr[i - 1] >= bArr[i - 1] && aArr[i] < bArr[i]) return true;
    }
    return false;
  };

  const _detectTopDivergence = (closeArr: number[], kArr: number[], lookback: number = 10): boolean => {
    const n = Math.min(closeArr.length, kArr.length);
    if (n < lookback) return false;
    const recentClose = closeArr.slice(-lookback);
    const recentK = kArr.slice(-lookback);
    const half = Math.floor(lookback / 2);
    const leftClose = recentClose.slice(0, half);
    const rightClose = recentClose.slice(half);
    const leftK = recentK.slice(0, half);
    const rightK = recentK.slice(half);
    const validLeftClose = leftClose.filter((v) => !isNaN(v));
    const validRightClose = rightClose.filter((v) => !isNaN(v));
    const validLeftK = leftK.filter((v) => !isNaN(v));
    const validRightK = rightK.filter((v) => !isNaN(v));
    if (validLeftClose.length === 0 || validRightClose.length === 0 || validLeftK.length === 0 || validRightK.length === 0) return false;
    const leftCloseMax = Math.max(...validLeftClose);
    const rightCloseMax = Math.max(...validRightClose);
    const leftKMax = Math.max(...validLeftK);
    const rightKMax = Math.max(...validRightK);
    return rightCloseMax > leftCloseMax && rightKMax < leftKMax;
  };

  const _detectBottomDivergence = (closeArr: number[], kArr: number[], lookback: number = 10): boolean => {
    const n = Math.min(closeArr.length, kArr.length);
    if (n < lookback) return false;
    const recentClose = closeArr.slice(-lookback);
    const recentK = kArr.slice(-lookback);
    const half = Math.floor(lookback / 2);
    const leftClose = recentClose.slice(0, half);
    const rightClose = recentClose.slice(half);
    const leftK = recentK.slice(0, half);
    const rightK = recentK.slice(half);
    const validLeftClose = leftClose.filter((v) => !isNaN(v));
    const validRightClose = rightClose.filter((v) => !isNaN(v));
    const validLeftK = leftK.filter((v) => !isNaN(v));
    const validRightK = rightK.filter((v) => !isNaN(v));
    if (validLeftClose.length === 0 || validRightClose.length === 0 || validLeftK.length === 0 || validRightK.length === 0) return false;
    const leftCloseMin = Math.min(...validLeftClose);
    const rightCloseMin = Math.min(...validRightClose);
    const leftKMin = Math.min(...validLeftK);
    const rightKMin = Math.min(...validRightK);
    return rightCloseMin < leftCloseMin && rightKMin > leftKMin;
  };

  const computeSignalsAtTime = (time: Time) => {
    const snapshots = snapshotRef.current;
    const idx = snapshots.findIndex((s) => s.time === time);
    if (idx < 0) return;

    const slice = snapshots.slice(0, idx + 1);
    const signals: Record<string, string> = {};

    // MACD
    const difArr = slice.map((s) => s.DIF);
    const deaArr = slice.map((s) => s.DEA);
    const lastDif = difArr[difArr.length - 1];
    const lastDea = deaArr[deaArr.length - 1];
    if (!isNaN(lastDif) && !isNaN(lastDea)) {
      const up = _crossUpTs(difArr, deaArr);
      const down = _crossDownTs(difArr, deaArr);
      if (up) {
        signals.macd = 'MACD金叉 \u2191';
      } else if (down) {
        signals.macd = 'MACD死叉 \u2193';
      } else if (lastDif > lastDea) {
        signals.macd = 'MACD多头 \u2197';
      } else {
        signals.macd = 'MACD空头 \u2198';
      }
    }

    // RSI：使用后端配置阈值
    const rsiArr = slice.map((s) => s.RSI);
    const lastRsi = rsiArr[rsiArr.length - 1];
    if (!isNaN(lastRsi)) {
      const rsiOb = intradayData?.rsi_overbought ?? 65;
      const rsiOs = intradayData?.rsi_oversold ?? 20;
      if (lastRsi <= rsiOs) {
        signals.rsi = `RSI超卖 \u2191`;
      } else if (lastRsi >= rsiOb) {
        signals.rsi = `RSI超买 \u2193`;
      } else if (lastRsi < 50) {
        signals.rsi = `RSI偏弱 \u2198`;
      } else {
        signals.rsi = `RSI偏强 \u2197`;
      }
    }

    // KDJ
    const kArr = slice.map((s) => s.K);
    const dArr = slice.map((s) => s.D);
    const jArr = slice.map((s) => s.J);
    const closeArr = slice.map((s) => s.close);
    const lastK = kArr[kArr.length - 1];
    const lastD = dArr[dArr.length - 1];
    const lastJ = jArr[jArr.length - 1];
    if (!isNaN(lastK) && !isNaN(lastD) && !isNaN(lastJ)) {
      const kdjOb = 80;
      const kdjOs = 20;
      if (lastK > kdjOb && lastD > kdjOb && lastJ > kdjOb) {
        signals.kdj = 'KDJ超买 \u2193';
      } else if (lastK < kdjOs && lastD < kdjOs && lastJ < kdjOs) {
        signals.kdj = 'KDJ超卖 \u2191';
      } else if (_crossUpTs(kArr, dArr)) {
        if (lastK < 50) {
          signals.kdj = 'KDJ金叉(低位) \u2191';
        } else {
          signals.kdj = 'KDJ金叉 \u2197';
        }
      } else if (_crossDownTs(kArr, dArr)) {
        if (lastK > 80) {
          signals.kdj = 'KDJ死叉(高位) \u2193';
        } else {
          signals.kdj = 'KDJ死叉 \u2198';
        }
      } else if (_detectTopDivergence(closeArr, kArr)) {
        signals.kdj = 'KDJ顶背离 \u2193';
      } else if (_detectBottomDivergence(closeArr, kArr)) {
        signals.kdj = 'KDJ底背离 \u2191';
      } else if (lastK > lastD) {
        signals.kdj = 'KDJ多头 \u2197';
      } else {
        signals.kdj = 'KDJ空头 \u2198';
      }
    }

    // MFI
    const mfiArr = slice.map((s) => s.mfi_value);
    const lastMfi = mfiArr[mfiArr.length - 1];
    const mfiOb = 80;
    const mfiOs = 20;
    if (!isNaN(lastMfi)) {
      const fiftyArr = mfiArr.map(() => 50);
      const crossUp = _crossUpTs(mfiArr, fiftyArr);
      const crossDown = _crossDownTs(mfiArr, fiftyArr);
      const topDiv = _detectTopDivergence(closeArr, mfiArr);
      const bottomDiv = _detectBottomDivergence(closeArr, mfiArr);
      const parts: string[] = [];
      if (lastMfi >= mfiOb) {
        parts.push('超买');
      } else if (lastMfi <= mfiOs) {
        parts.push('超卖');
      }
      if (crossUp) parts.push('上穿50线');
      if (crossDown) parts.push('下穿50线');
      if (topDiv) parts.push('顶背离');
      if (bottomDiv) parts.push('底背离');
      if (parts.length === 0) {
        if (lastMfi > 50) {
          parts.push('偏强');
        } else {
          parts.push('偏弱');
        }
      }
      signals.mfi = parts.filter(Boolean).join('、');
    }

    // 主力吸筹不输出信号
    signals.absorption = '';

    crosshairSignalRef.current = signals;
    setCrosshairSignals({ ...signals });

    setCrosshairRsiValue(isNaN(lastRsi) ? null : lastRsi);

    setCrosshairKdjKValue(isNaN(lastK) ? null : lastK);
    setCrosshairKdjDValue(isNaN(lastD) ? null : lastD);
    setCrosshairKdjJValue(isNaN(lastJ) ? null : lastJ);

    // 累计到当前时间点的MACD柱高度和 / 柱高度差：使用后端metadata（与策略算法一致）
    const meta = macdMetadataRef.current;
    if (meta?.bar_sums && idx < meta.bar_sums.length) {
      setCrosshairMacdSum(meta.bar_sums[idx]);
    } else {
      setCrosshairMacdSum(null);
    }
    if (meta?.bar_diffs && idx < meta.bar_diffs.length) {
      setCrosshairMacdDiff(meta.bar_diffs[idx]);
    } else {
      setCrosshairMacdDiff(null);
    }

    const curSn = snapshots[idx];
    setCrosshairDeviationPct(curSn && !isNaN(curSn.deviation_pct) ? curSn.deviation_pct : null);
    setCrosshairMa5DevPct(curSn && !isNaN(curSn.ma5_dev_pct) ? curSn.ma5_dev_pct : null);

    setIsCrosshairActive(true);

    // 查找该时间点附近的信号
    const allSignals = allSignalsRef.current;
    const sigAtTime = allSignals.find((s) => {
      const sigTime = Math.floor(new Date(s.trigger_time).getTime() / 1000);
      return sigTime === time;
    });
    if (sigAtTime) {
      setHoveredWeightDetails({
        buy: sigAtTime.buy_weight_details || [],
        sell: sigAtTime.sell_weight_details || [],
        supportForce: sigAtTime.support_force || 0,
        pressureForce: sigAtTime.pressure_force || 0,
        signalType: sigAtTime.signal_type,
      });
    } else {
      // 无后端信号时，基于快照数据在前端重建权重贡献
      const sn = snapshots[idx];
      const absActive = sn.absorption_active;
      const distActive = sn.distribution_active;
      const macdGoldenCross = sn.macd_golden_cross;
      const rsiOversold = sn.rsi_oversold;
      const macdDeathCross = sn.macd_death_cross;
      const rsiOverbought = sn.rsi_overbought;
      const kdjK = sn.K;
      const kdjD = sn.D;
      const kdjJ = sn.J;
      const kdjKArr = slice.map((s: any) => s.K);
      const kdjDArr = slice.map((s: any) => s.D);
      const kdjOversold = !isNaN(kdjK) && !isNaN(kdjD) && !isNaN(kdjJ) && kdjK < 20 && kdjD < 20 && kdjJ < 20;
      const kdjOverbought = !isNaN(kdjK) && !isNaN(kdjD) && !isNaN(kdjJ) && kdjK > 80 && kdjD > 80 && kdjJ > 80;
      const kdjGoldenCross = _crossUpTs(kdjKArr, kdjDArr);
      const kdjDeathCross = _crossDownTs(kdjKArr, kdjDArr);

      const buyWeights: WeightContribution[] = [];
      const sellWeights: WeightContribution[] = [];

      const bwMap = weightsConfigRef.current.buy;
      const swMap = weightsConfigRef.current.sell;

      // 买入权重
      if (!absActive) {
        buyWeights.push({ key: 'absorption_required', label: '主力吸筹(必备条件)', weight: 0, triggered: false, score: 0 });
      } else {
        const factors: [string, string, number, boolean][] = [
          ['absorption_active', '主力吸筹活跃', bwMap.absorption_active ?? 5, absActive],
          ['price_cross_ma5_up', '价格上穿MA5', bwMap.price_cross_ma5_up ?? 1, sn.price_cross_ma5_up],
          ['avg_price_oversold_fix', '均价超卖修复', bwMap.avg_price_oversold_fix ?? 2, sn.deviation_oversold && sn.deviation_narrowing],
          ['price_above_ma20', '价格>MA20趋势', bwMap.price_above_ma20 ?? 1, sn.price_above_ma20],
          ['volume_surge', '量能放大', bwMap.volume_surge ?? 1, false],
          ['macd_golden_cross', 'MACD金叉', bwMap.macd_golden_cross ?? 2, macdGoldenCross],
          ['macd_bearish_recovering', 'MACD空头动能衰竭', bwMap.macd_bearish_recovering ?? 5, sn.macd_bearish_recovering],
          ['rsi_oversold', 'RSI超卖', bwMap.rsi_oversold ?? 5, rsiOversold],
          ['kdj_oversold', 'KDJ超卖', bwMap.kdj_oversold ?? 5, kdjOversold],
          ['kdj_golden_cross', 'KDJ金叉', bwMap.kdj_golden_cross ?? 3, kdjGoldenCross],
          ['mfi_oversold', 'MFI超卖', bwMap.mfi_oversold ?? 3, sn.mfi_oversold],
        ];
        let buyScore = 0;
        for (const [key, label, w, trig] of factors) {
          if (w === 0) continue;
          buyScore += trig ? w : 0;
          buyWeights.push({ key, label, weight: w, triggered: trig, score: trig ? w : 0 });
        }
        buyWeights.push({ key: 'gravity', label: '引力场', weight: 0, triggered: false, score: 0 });
      }

      // 卖出权重
      if (!distActive) {
        sellWeights.push({ key: 'distribution_required', label: '主力出货(必备条件)', weight: 0, triggered: false, score: 0 });
      } else {
        const factors: [string, string, number, boolean][] = [
          ['distribution_active', '主力出货活跃', swMap.distribution_active ?? 0, distActive],
          ['volume_stagnation', '放量滞涨', swMap.volume_stagnation ?? 3, false],
          ['price_cross_ma5_down', '价格下穿MA5', swMap.price_cross_ma5_down ?? 2, sn.price_cross_ma5_down],
          ['avg_price_overbought_fix', '均价超买回落', swMap.avg_price_overbought_fix ?? 2, sn.deviation_overbought && sn.deviation_peaking],
          ['macd_death_cross', 'MACD死叉', swMap.macd_death_cross ?? 2, macdDeathCross],
          ['macd_bullish_weakening', 'MACD多头动能衰减', swMap.macd_bullish_weakening ?? 5, sn.macd_bullish_weakening],
          ['rsi_overbought', 'RSI超买', swMap.rsi_overbought ?? 5, rsiOverbought],
          ['kdj_overbought', 'KDJ超买', swMap.kdj_overbought ?? 5, kdjOverbought],
          ['kdj_death_cross', 'KDJ死叉', swMap.kdj_death_cross ?? 3, kdjDeathCross],
          ['mfi_overbought', 'MFI超买', swMap.mfi_overbought ?? 3, sn.mfi_overbought],
        ];
        let sellScore = 0;
        for (const [key, label, w, trig] of factors) {
          if (w === 0) continue;
          sellScore += trig ? w : 0;
          sellWeights.push({ key, label, weight: w, triggered: trig, score: trig ? w : 0 });
        }
        sellWeights.push({ key: 'gravity', label: '引力场', weight: 0, triggered: false, score: 0 });
      }

      setHoveredWeightDetails({
        buy: buyWeights,
        sell: sellWeights,
        supportForce: 0,
        pressureForce: 0,
        signalType: 'none',
      });
    }
  };

  // 复盘交互锁定/恢复
  const applyReplayInteractionLock = useCallback(() => {
    const disableInteraction = {
      handleScroll: { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false },
      handleScale: { mouseWheel: false, axisPressedMouseMove: false, pinch: false },
    };
    if (chartRef.current) {
      chartRef.current.applyOptions(disableInteraction);
    }
    if (volumeChartRef.current) {
      volumeChartRef.current.applyOptions(disableInteraction);
    }
    indicatorChartsRef.current.forEach((c) => {
      try { c.applyOptions(disableInteraction); } catch (e) { /* ignore */ }
    });
  }, []);

  const restoreChartInteraction = useCallback(() => {
    const enableInteraction = {
      handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
      handleScale: { mouseWheel: true, axisPressedMouseMove: false, pinch: true },
    };
    if (chartRef.current) {
      chartRef.current.applyOptions(enableInteraction);
    }
    if (volumeChartRef.current) {
      volumeChartRef.current.applyOptions(enableInteraction);
    }
    indicatorChartsRef.current.forEach((c) => {
      try { c.applyOptions(enableInteraction); } catch (e) { /* ignore */ }
    });
  }, []);

  // ── 渲染数据 ──
  const renderData = useCallback(
    (data: IntradayDataResponse, date: string) => {
      const container = chartContainerRef.current;
      const volContainer = volumeContainerRef.current;
      if (!container || !volContainer) return;

      // ── 标记初始化阶段开始：阻止子图时间轴变更反向传播到主图 ──
      isInitialRenderRef.current = true;

      // ── 先退订十字线事件，防止销毁图表时触发旧图表 crosshair 事件扰乱引擎状态 ──
      crosshairSubsRef.current.forEach((unsub) => {
        try { unsub(); } catch (e) { /* ignore */ }
      });
      crosshairSubsRef.current = [];
      syncEngineRef.current.clear();

      // ── 销毁旧图表以完全重置内部状态（包括用户手动调整的 scale）──
      if (chartRef.current) {
        try { chartRef.current.remove(); } catch (e) { /* ignore */ }
        chartRef.current = null;
        candleSeriesRef.current = null;
        chipAreaRef.current = null;
        avgPriceLineRef.current = null;
        priceRangeSeriesRef.current = null;
        seriesMarkersRef.current = null;
        refLineSeriesRef.current = [];
        refLinePriceLinesRef.current = [];
      }
      if (volumeChartRef.current) {
        try { volumeChartRef.current.remove(); } catch (e) { /* ignore */ }
        volumeChartRef.current = null;
        volumeSeriesRef.current = null;
      }
      indicatorChartsRef.current.forEach((c) => {
        try { c.remove(); } catch (e) { /* ignore */ }
      });
      indicatorChartsRef.current.clear();
      indicatorSeriesRef.current.clear();

      // ── 清空所有容器DOM，确保完全干净（防御性措施，解决chart.remove()可能残留canvas的问题）──
      try { container.innerHTML = ''; } catch (e) { /* ignore */ }
      try { volContainer.innerHTML = ''; } catch (e) { /* ignore */ }
      [absorptionContainerRef, macdContainerRef, rsiContainerRef, kdjContainerRef].forEach((ref) => {
        if (ref.current) {
          try { ref.current.innerHTML = ''; } catch (e) { /* ignore */ }
        }
      });

      // ── 创建全新图表 ──
      const chart = lightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: CHART_HEIGHT,
        layout: {
          background: { type: 'solid', color: '#1a1a2e' } as any,
          textColor: '#d1d4dc',
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: '#2b2b43' },
          horzLines: { color: '#2b2b43' },
        },
        rightPriceScale: {
          scaleMargins: { top: 0.1, bottom: 0.1 },
          borderVisible: false,
        },
        crosshair: {
          mode: 1,
          vertLine: { color: '#9B7DFF', width: 1, style: 2 },
          horzLine: { color: '#9B7DFF', width: 1, style: 2 },
        },
        localization: {
          timeFormatter,
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time: number) => {
            const d = new Date(time * 1000);
            const h = String(d.getHours()).padStart(2, '0');
            const m = String(d.getMinutes()).padStart(2, '0');
            return `${h}:${m}`;
          },
          barSpacing: 8,
          fixLeftEdge: true,
          fixRightEdge: true,
        },
        handleScroll: {
          mouseWheel: false,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          mouseWheel: true,
          axisPressedMouseMove: false,
          pinch: true,
        },
      });
      chartRef.current = chart;

      const volChart = lightweightCharts.createChart(volContainer, {
        width: volContainer.clientWidth,
        height: 100,
        layout: {
          background: { type: 'solid', color: '#1a1a2e' } as any,
          textColor: '#d1d4dc',
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: '#2b2b43' },
          horzLines: { color: '#2b2b43' },
        },
        rightPriceScale: {
          borderVisible: false,
        },
        crosshair: {
          mode: 0,
        },
        localization: {
          timeFormatter,
        },
        timeScale: {
          timeVisible: false,
          secondsVisible: false,
          tickMarkFormatter: () => '',
          fixLeftEdge: true,
          fixRightEdge: true,
        },
        handleScroll: {
          mouseWheel: false,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          mouseWheel: true,
          axisPressedMouseMove: false,
          pinch: true,
        },
      });
      const volSeries = volChart.addSeries(lightweightCharts.HistogramSeries, {
        color: '#66666688',
        priceFormat: { type: 'volume', precision: 0, minMove: 1 },
      });
      volumeChartRef.current = volChart;
      volumeSeriesRef.current = volSeries;

      // 五日平均成交量线（白色）
      const vol5maLine = volChart.addSeries(lightweightCharts.LineSeries, {
        color: '#FFFFFF',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      volume5MASeriesRef.current = vol5maLine;

      // 十日平均成交量线（金色）
      const vol10maLine = volChart.addSeries(lightweightCharts.LineSeries, {
        color: '#FFD700',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      volume10MASeriesRef.current = vol10maLine;

      const klines = convertKlineData(data.kline_data, date);
      klineRawDataRef.current = klines;
      allSignalsRef.current = data.signals || [];
      const firstKlineTime: number = klines.length > 0 ? (klines[0].time as number) : 0;

      // 默认显示最后一个信号的权重贡献
      const sigs = data.signals || [];
      if (sigs.length > 0) {
        const lastSig = sigs[sigs.length - 1];
        setHoveredWeightDetails({
          buy: lastSig.buy_weight_details || [],
          sell: lastSig.sell_weight_details || [],
          supportForce: lastSig.support_force || 0,
          pressureForce: lastSig.pressure_force || 0,
          signalType: lastSig.signal_type,
        });
      }

      // 检测腾讯数据：每根K线中 Open===High===Low===Close 的比例
      const flatCount = klines.filter((k) => k.open === k.high && k.high === k.low && k.low === k.close).length;
      const isTencentData = klines.length > 10 && flatCount / klines.length > 0.95;
      isTencentDataRef.current = isTencentData;

      // 总是重建主系列以重置价格 scale，避免手动调整后切换股票 scale 不重置
      // 注意：分时白线延迟添加，待筹码区色带先添加后再添加，使白线处于最上层
      if (candleSeriesRef.current) {
        try { chart.removeSeries(candleSeriesRef.current); } catch (e) { /* ignore */ }
        candleSeriesRef.current = null;
      }

      volSeries.setData(
        klines.map((k, i) => {
          const prevClose = i > 0 ? klines[i - 1].close : klines[0].close;
          const isUp = isTencentData ? k.close >= prevClose : k.close >= k.open;
          return {
            time: k.time as any,
            value: k.volume,
            color: isUp ? '#FF444466' : '#44AA4466',
          };
        }),
      );

      // 更新五日/十日平均成交量线（预热模式下用预热数据参与计算，再只保留当日部分）
      const volumeValues = klines.map(k => ({ time: k.time as any, value: k.volume }));
      const warmupKlines = data.warmup_info?.klines;
      let allVolumeValues = volumeValues;
      if (warmupKlines && warmupKlines.length > 0) {
        const warmupVolumeValues = warmupKlines.map(k => ({ time: k.time as any, value: k.Volume }));
        allVolumeValues = [...warmupVolumeValues, ...volumeValues];
      }
      const allVolume5MA = calculateVolumeMA(allVolumeValues, 5);
      const allVolume10MA = calculateVolumeMA(allVolumeValues, 10);
      const volume5MA = warmupKlines?.length ? allVolume5MA.slice(-klines.length) : allVolume5MA;
      const volume10MA = warmupKlines?.length ? allVolume10MA.slice(-klines.length) : allVolume10MA;
      if (volume5MASeriesRef.current) volume5MASeriesRef.current.setData(volume5MA);
      if (volume10MASeriesRef.current) volume10MASeriesRef.current.setData(volume10MA);

      // ── 注册成交量到同步引擎 ──
      syncEngineRef.current.register(
        'volume',
        volChart,
        volSeries,
        klines.map((k) => ({ time: k.time as number, value: (k as any).volume || 0 })),
      );

      // ── 分时均价线（累计成交额/累计成交量曲线）──
      if (avgPriceLineRef.current) {
        try { chart.removeSeries(avgPriceLineRef.current); } catch (e) { /* ignore */ }
        avgPriceLineRef.current = null;
      }
      if (priceRangeSeriesRef.current) {
        try { chart.removeSeries(priceRangeSeriesRef.current); } catch (e) { /* ignore */ }
        priceRangeSeriesRef.current = null;
      }
      const avgPriceData = klines.filter((k: any) => k.avgPrice != null).map((k: any) => ({
        time: k.time,
        value: k.avgPrice,
      }));
      if (avgPriceData.length > 0) {
        const avgSeries = chart.addSeries(lightweightCharts.LineSeries, {
          color: '#FF8800',
          lineWidth: 1,
          lineStyle: 1, // dotted
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        avgSeries.setData(avgPriceData);
        avgPriceLineRef.current = avgSeries;
      }

      // ── 支撑位/压力位参考线清理 ──
      refLineSeriesRef.current.forEach((s) => {
        try { chart.removeSeries(s); } catch (e) { /* ignore */ }
      });
      refLineSeriesRef.current = [];
      {
        const cs = candleSeriesRef.current;
        if (cs) {
          refLinePriceLinesRef.current.forEach((pl) => {
            try { (cs as any).removePriceLine(pl); } catch (e) { /* ignore */ }
          });
          refLinePriceLinesRef.current = [];
        }
      }
      if (chipAreaRef.current) {
        try { chart.removeSeries(chipAreaRef.current); } catch (e) { /* ignore */ }
        chipAreaRef.current = null;
      }

      // ── 筹码密集区重叠判定（需在参考线和色带渲染前计算，避免 PriceLine 拉伸 Y 轴）──
      let chipOverlaps = false;
      let chipUpperPrice: number | null = null;
      let chipLowerPrice: number | null = null;
      let clampedUpperPrice: number | null = null;
      let autoAnchorHigh: number | null = null;
      let autoAnchorLow: number | null = null;
      if (data.reference_lines && data.reference_lines.length > 0 && klines.length > 0) {
        const chipUpper = data.reference_lines.find((rl) => rl.id === 'chip_upper');
        const chipLower = data.reference_lines.find((rl) => rl.id === 'chip_lower');
        if (chipUpper && chipLower) {
          chipUpperPrice = chipUpper.price;
          chipLowerPrice = chipLower.price;
          const klineMin = Math.min(...klines.map((k) => k.low));
          const klineMax = Math.max(...klines.map((k) => k.high));
          const prevCloseRef = data.reference_lines.find((rl: any) => rl.id === 'prev_close');
          if (priceRangeEnabledRef.current && prevCloseRef) {
            const rangeLow = prevCloseRef.price * 0.9;
            const rangeHigh = prevCloseRef.price * 1.1;
            chipOverlaps = chipUpper.price >= rangeLow && chipLower.price <= rangeHigh;
            if (chipOverlaps) {
              clampedUpperPrice = Math.min(chipUpperPrice, rangeHigh);
            }
          } else {
            chipOverlaps = chipUpper.price >= klineMin && chipLower.price <= klineMax;
            if (chipOverlaps) {
              const visibleMax = klineMax + (klineMax - klineMin) * 0.125;
              clampedUpperPrice = Math.min(chipUpperPrice, visibleMax);
              autoAnchorLow = klineMin;
              autoAnchorHigh = klineMax;
            }
          }
        }
      }

      // ── 隐藏参考线图层：支撑/压力位 PriceLine 附着于此类（位于底层，白线在其上方）──
      if (data.reference_lines && data.reference_lines.length > 0) {
        const refLayer = chart.addSeries(lightweightCharts.LineSeries, {
          lineVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        } as any);
        refLayer.setData(
          klines.map((k) => ({
            time: k.time as any,
            value: k.close,
          })),
        );
        refLineSeriesRef.current.push(refLayer);

        data.reference_lines.forEach((rl) => {
          const isChipLine = rl.id === 'chip_upper' || rl.id === 'chip_lower';
          if (isChipLine && !chipOverlaps) return; // 筹码区不在可见范围内，跳过该参考线
          const ls: 0 | 1 | 2 | 3 | 4 = rl.style === 'dashed' ? 2 : (rl.style === 'dotted' ? 1 : 0);
          // 将颜色 hex 转换为半透明 hex 作为标签背景色
          const labelBgColor = hexToRgba(rl.color, 0.35);
          const priceLine = refLayer.createPriceLine({
            price: rl.price,
            color: rl.color,
            lineWidth: 1,
            lineStyle: ls,
            axisLabelVisible: !isChipLine,
            axisLabelColor: labelBgColor,
            axisLabelTextColor: '#000000',
            title: rl.label,
          });
          refLinePriceLinesRef.current.push(priceLine);
        });
      }

      // ── 分时白线/K线（在参考线图层之后添加，使其始终处于最上层，仅次于箭头标记）──
      if (isTencentData) {
        const lineSeries = chart.addSeries(lightweightCharts.LineSeries, {
          color: '#EEEEEE',
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        lineSeries.setData(
          klines.map((k) => ({
            time: k.time as any,
            value: k.close,
          })),
        );
        candleSeriesRef.current = lineSeries as any;
      } else {
        const newCandleSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
          upColor: '#FF4444',
          downColor: '#44AA44',
          borderDownColor: '#44AA44',
          borderUpColor: '#FF4444',
          wickDownColor: '#44AA44',
          wickUpColor: '#FF4444',
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        });
        newCandleSeries.setData(
          klines.map((k) => ({
            time: k.time as any,
            open: k.open,
            high: k.high,
            low: k.low,
            close: k.close,
          })),
        );
        candleSeriesRef.current = newCandleSeries;
      }

      chart.timeScale().fitContent();

      // ── 设置引擎回调和注册主图 ──
      syncEngineRef.current.setCallbacks({
        onMove: (time) => {
          currentCrosshairTimeRef.current = time;
          try {
            computeSignalsAtTime(time);
          } catch (e) {
            console.error('[Engine] computeSignalsAtTime error', e);
          }
        },
        onLeave: () => {
          setIsCrosshairActive(false);
          setCrosshairMacdSum(null);
          setCrosshairMacdDiff(null);
          setCrosshairRsiValue(null);
          setCrosshairKdjKValue(null);
          setCrosshairKdjDValue(null);
          setCrosshairKdjJValue(null);
          setCrosshairDeviationPct(null);
          setCrosshairMa5DevPct(null);
        },
      });
      syncEngineRef.current.register(
        'main',
        chart,
        candleSeriesRef.current!,
        klines.map((k) => ({ time: k.time as number, value: k.close })),
      );

      // ── 设置初始价格范围为昨收的±10%（受 toggle 控制）──
      if (priceRangeEnabledRef.current) {
        const prevCloseRef = (data.reference_lines || []).find((rl: any) => rl.id === 'prev_close');
        if (prevCloseRef && klines.length >= 2) {
          const prevClose = prevCloseRef.price;
          const ycLow = prevClose * 0.9;
          const ycHigh = prevClose * 1.1;
          const firstTime = klines[0].time;
          const lastTime = klines[klines.length - 1].time;
          if (firstTime != null && lastTime != null) {
            const ghostSeries = chart.addSeries(lightweightCharts.LineSeries, {
              lineVisible: false,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
            } as any);
            ghostSeries.setData([
              { time: firstTime as any, value: ycLow },
              { time: lastTime as any, value: ycHigh },
            ]);
            priceRangeSeriesRef.current = ghostSeries;

            // 锁定价格轴，防止后续添加的系列（如筹码密集区）拉宽Y轴范围
            chart.priceScale('right').applyOptions({ autoScale: false });
          }
        }
      } else if (autoAnchorLow != null && autoAnchorHigh != null) {
        const firstTime = klines[0].time;
        const lastTime = klines[klines.length - 1].time;
        if (firstTime != null && lastTime != null) {
          const ghostSeries = chart.addSeries(lightweightCharts.LineSeries, {
            lineVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          } as any);
          ghostSeries.setData([
            { time: firstTime as any, value: autoAnchorLow },
            { time: lastTime as any, value: autoAnchorHigh },
          ]);
          priceRangeSeriesRef.current = ghostSeries;
          chart.priceScale('right').applyOptions({ autoScale: false });
        }
      } else {
        chart.priceScale('right').applyOptions({ autoScale: true });
      }

      // ── 筹码密集区色带（仅在价格范围内可见时才渲染，上沿裁剪避免拉伸Y轴）──
      if (chipOverlaps && chipLowerPrice != null && clampedUpperPrice != null) {
        const purple = 'rgba(187, 68, 255, 0.06)';
        const chipSeries = chart.addSeries(lightweightCharts.BaselineSeries, {
          baseValue: { type: 'price', price: chipLowerPrice },
          lineWidth: 0,
          topLineColor: 'rgba(0,0,0,0)',
          bottomLineColor: 'rgba(0,0,0,0)',
          topFillColor1: purple,
          topFillColor2: purple,
          bottomFillColor1: purple,
          bottomFillColor2: purple,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        } as any);
        chipSeries.setData(klines.map((k) => ({
          time: k.time as any,
          value: clampedUpperPrice,
        })));
        chipAreaRef.current = chipSeries;
      }

      // ── 构建快照数据：融合K线和指标子图 ──
      const buildSnapshot = () => {
        const map = new Map<number, {
          dominant_power: number; absorption: number;
          close: number; ma5: number; ma20: number; deviation_pct: number; ma5_dev_pct: number;
          DIF: number; DEA: number; MACD_Bar: number; RSI: number;
          K: number; D: number; J: number;
          mfi_value: number;
        }>();
        (data.indicator_sub_charts || []).forEach((sc) => {
          (sc.lines || []).forEach((line) => {
            (line.data || []).forEach((pt: any) => {
              const ms = parseTimestamp(pt.time, date);
              const t = Math.floor(ms / 1000);
              if (t <= 0) return;
              if (!map.has(t)) map.set(t, {
                dominant_power: NaN, absorption: NaN,
                close: NaN, ma5: NaN, ma20: NaN, deviation_pct: NaN, ma5_dev_pct: NaN,
                DIF: NaN, DEA: NaN, MACD_Bar: NaN, RSI: NaN,
                K: NaN, D: NaN, J: NaN,
                mfi_value: NaN,
              });
              const entry = map.get(t)!;
              if (line.name === 'dominant_power') entry.dominant_power = pt.value;
              else if (line.name === 'absorption') entry.absorption = pt.value;
              else if (line.name === 'close') entry.close = pt.value;
              else if (line.name === 'ma5') entry.ma5 = pt.value;
              else if (line.name === 'ma20') entry.ma20 = pt.value;
              else if (line.name === 'deviation_pct') entry.deviation_pct = pt.value;
              else if (line.name === 'ma5_dev_pct') entry.ma5_dev_pct = pt.value;
              else if (line.name === 'DIF') entry.DIF = pt.value;
              else if (line.name === 'DEA') entry.DEA = pt.value;
              else if (line.name === 'MACD_Bar') entry.MACD_Bar = pt.value;
              else if (line.name === 'RSI') entry.RSI = pt.value;
              else if (line.name === 'K') entry.K = pt.value;
              else if (line.name === 'D') entry.D = pt.value;
              else if (line.name === 'J') entry.J = pt.value;
              else if (line.name === 'mfi_value') entry.mfi_value = pt.value;
            });
          });
        });
        const OVERSOLD = -2.5;
        const OVERBOUGHT = 2.5;
        const RSI_OVERSOLD = 20;
        const RSI_OVERBOUGHT = 65;
        const MFI_OVERSOLD = 20;
        const MFI_OVERBOUGHT = 80;
        const raw = Array.from(map.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([time, v]) => ({ time, ...v }));
        snapshotRef.current = (() => {
          let runningMacdBarSum = 0;
          return raw.map((cur, i) => {
          const prev = i > 0 ? raw[i - 1] : null;
          const absorption_val = isNaN(cur.absorption) ? 0 : cur.absorption;
          const dev = isNaN(cur.deviation_pct) ? 0 : cur.deviation_pct;
          const prevDev = prev && !isNaN(prev.deviation_pct) ? prev.deviation_pct : 0;
          const curDif = isNaN(cur.DIF) ? 0 : cur.DIF;
          const curDea = isNaN(cur.DEA) ? 0 : cur.DEA;
          const prevDif = prev && !isNaN(prev.DIF) ? prev.DIF : 0;
          const prevDea = prev && !isNaN(prev.DEA) ? prev.DEA : 0;
          // 检查前一根K线是否发生了金叉/死叉
          const prevPrev = i > 1 ? raw[i - 2] : null;
          const prevPrevDif = prevPrev && !isNaN(prevPrev.DIF) ? prevPrev.DIF : 0;
          const prevPrevDea = prevPrev && !isNaN(prevPrev.DEA) ? prevPrev.DEA : 0;
          const prevGoldenCross = prev ? (prevPrevDif <= prevPrevDea && prevDif > prevDea) : false;
          const prevDeathCross = prev ? (prevPrevDif >= prevPrevDea && prevDif < prevDea) : false;
          // 检查前两根K线是否发生了金叉/死叉
          const prevPrevPrev = i > 2 ? raw[i - 3] : null;
          const prevPrevPrevDif = prevPrevPrev && !isNaN(prevPrevPrev.DIF) ? prevPrevPrev.DIF : 0;
          const prevPrevPrevDea = prevPrevPrev && !isNaN(prevPrevPrev.DEA) ? prevPrevPrev.DEA : 0;
          const prevPrevGoldenCross = prevPrev ? (prevPrevPrevDif <= prevPrevPrevDea && prevPrevDif > prevPrevDea) : false;
          const prevPrevDeathCross = prevPrev ? (prevPrevPrevDif >= prevPrevPrevDea && prevPrevDif < prevPrevDea) : false;
          const curRsi = isNaN(cur.RSI) ? 50 : cur.RSI;
          const curMfi = isNaN(cur.mfi_value) ? 50 : cur.mfi_value;
          const macdBarSum = (() => {
            const v = cur.MACD_Bar;
            if (!isNaN(v)) runningMacdBarSum += v;
            return runningMacdBarSum;
          })();
          const macdBarDiff = prev
            ? (isNaN(cur.MACD_Bar) ? 0 : cur.MACD_Bar) - (isNaN(prev.MACD_Bar) ? 0 : prev.MACD_Bar)
            : 0;
          return {
            ...cur,
            absorption_active: absorption_val > 0,
            distribution_active: absorption_val < 0,
            price_above_ma5: !isNaN(cur.close) && !isNaN(cur.ma5) ? cur.close > cur.ma5 : false,
            price_above_ma20: !isNaN(cur.close) && !isNaN(cur.ma20) ? cur.close > cur.ma20 : false,
            price_cross_ma5_up: prev ? (prev.close <= prev.ma5 && cur.close > cur.ma5) : false,
            price_cross_ma5_down: prev ? (prev.close >= prev.ma5 && cur.close < cur.ma5) : false,
            deviation_oversold: dev <= OVERSOLD,
            deviation_overbought: dev >= OVERBOUGHT,
            deviation_narrowing: dev <= OVERSOLD && prev ? (dev > prevDev) : false,
            deviation_peaking: dev >= OVERBOUGHT && prev ? (dev < prevDev) : false,
            macd_golden_cross: prev ? (prevDif <= prevDea && curDif > curDea) : false,
            macd_death_cross: prev ? (prevDif >= prevDea && curDif < curDea) : false,
            macd_bar_sum: macdBarSum,
            macd_bar_diff: macdBarDiff,
            // 多头动能衰减→卖出: prev/prev2只排除金叉(见后端策略注释同款逻辑)
            macd_bullish_weakening: curDif > curDea && macdBarSum >= -0.015 && macdBarDiff >= -0.005 && !(prev ? (prevDif <= prevDea && curDif > curDea) : false) && !(prev ? (prevDif >= prevDea && curDif < curDea) : false) && !prevGoldenCross && !prevPrevGoldenCross,
            // 空头动能衰竭→买入: prev/prev2只排除死叉(见后端策略注释同款逻辑)
            macd_bearish_recovering: curDif < curDea && macdBarSum <= 0 && macdBarDiff <= 0 && !(prev ? (prevDif <= prevDea && curDif > curDea) : false) && !(prev ? (prevDif >= prevDea && curDif < curDea) : false) && !prevDeathCross && !prevPrevDeathCross,
            rsi_oversold: curRsi <= RSI_OVERSOLD,
            rsi_overbought: curRsi >= RSI_OVERBOUGHT,
            mfi_value: isNaN(cur.mfi_value) ? 50 : cur.mfi_value,
            mfi_oversold: curMfi <= MFI_OVERSOLD,
            mfi_overbought: curMfi >= MFI_OVERBOUGHT,
          };
        });
      })();
    };
      buildSnapshot();
      macdMetadataRef.current = data?.indicator_sub_charts?.find((sc: any) => sc.id === 'macd')?.metadata ?? null;
      weightsConfigRef.current = {
        buy: (data as any).buy_weights || {},
        sell: (data as any).sell_weights || {},
      };

      // ── 主图十字线联动 ──
      const mainHandleCrosshairMove = (param: any) => {
        syncEngineRef.current.handleMove('main', param);
      };
      crosshairSubsRef.current.push(chart.subscribeCrosshairMove(mainHandleCrosshairMove));

      const mainHandleTimeScaleChange = () => {
        if (isTimeSyncingRef.current) return;
        isTimeSyncingRef.current = true;
        try {
          const range = chart.timeScale().getVisibleRange();
          if (range && range.from && range.to) {
            syncEngineRef.current.syncTimeRange(range);
          }
        } finally {
          setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
        }
      };
      chart.timeScale().subscribeVisibleTimeRangeChange(mainHandleTimeScaleChange);

      // ── 成交量图十字线联动 ──
      if (volChart && volSeries) {
        const volHandleCrosshairMove = (param: any) => {
          syncEngineRef.current.handleMove('volume', param);
        };
        crosshairSubsRef.current.push(volChart.subscribeCrosshairMove(volHandleCrosshairMove));

        const volHandleTimeScaleChange = () => {
          if (isTimeSyncingRef.current) return;
          isTimeSyncingRef.current = true;
          try {
            const vr = volChart.timeScale().getVisibleRange();
            if (vr && vr.from && vr.to) {
              syncEngineRef.current.syncTimeRange(vr);
            }
          } finally {
            setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
          }
        };
        volChart.timeScale().subscribeVisibleTimeRangeChange(volHandleTimeScaleChange);
      }

      // 存储清理引用
      (window as any)._intradayCrosshairHandler = mainHandleCrosshairMove;
      (window as any)._intradayTimeHandler = mainHandleTimeScaleChange;

      // ── 指标子图 ──
      const containerMap: Array<{ id: string; ref: React.RefObject<HTMLDivElement | null> }> = [
        { id: 'absorption', ref: absorptionContainerRef },
        { id: 'macd', ref: macdContainerRef },
        { id: 'rsi', ref: rsiContainerRef },
        { id: 'kdj', ref: kdjContainerRef },
        { id: 'mfi', ref: mfiContainerRef },
      ];

      const subCharts = data.indicator_sub_charts || [];

      // 先清理所有旧的指标子图
      indicatorChartsRef.current.forEach((c) => {
        try { c.remove(); } catch (e) { /* ignore */ }
      });
      indicatorChartsRef.current.clear();
      indicatorSeriesRef.current.clear();

      for (const sc of subCharts) {
        const entry = containerMap.find((c) => c.id === sc.id);
        const container = entry?.ref.current;
        if (!container) continue;

        const subChart = lightweightCharts.createChart(container, {
          width: container.clientWidth,
          height: sc.height || 110,
          layout: {
            background: { type: 'solid', color: '#1a1a2e' } as any,
            textColor: '#d1d4dc',
            attributionLogo: false,
          },
          grid: {
            vertLines: { color: '#2b2b43' },
            horzLines: { color: '#2b2b43' },
          },
          rightPriceScale: {
            scaleMargins: { top: 0.1, bottom: 0.1 },
            borderVisible: false,
          },
          crosshair: { mode: 1 },
          localization: {
            timeFormatter,
          },
          timeScale: {
            timeVisible: true,
            visible: true,
            tickMarkFormatter: (time: number) => {
              const d = new Date(time * 1000);
              const h = String(d.getHours()).padStart(2, '0');
              const m = String(d.getMinutes()).padStart(2, '0');
              return `${h}:${m}`;
            },
          },
        });

        const lineSeriesList: any[] = [];

        let engineDataCollected = false;
        let engineData: { time: number; value: number }[] = [];
        let engineLastValueSeries: any = undefined;

        for (const line of sc.lines) {
          if (sc.id === 'absorption') {
            if (!line.data || line.data.length === 0) continue;
            const hs = subChart.addSeries(lightweightCharts.HistogramSeries, {
              color: '#AA44FF',
              priceFormat: { type: 'volume' },
            } as any);

            const points = line.data
              .filter((pt) => pt.time)
              .map((pt) => {
                const ms = parseTimestamp(pt.time, date);
                return {
                  time: Math.floor(ms / 1000) as any,
                  value: pt.value,
                  color: pt.value >= 0 ? '#AA44FF' : '#44AA44',
                };
              })
              .sort((a, b) => (a.time as number) - (b.time as number));

            if (!engineDataCollected) {
              engineData = padDataStart(points, firstKlineTime);
              engineDataCollected = true;
            }
            hs.setData(points);
            lineSeriesList.push(hs);
          } else if (sc.id === 'macd') {
            if (!line.data || line.data.length === 0) continue;
            const pts = line.data
              .filter((pt: any) => pt.time)
              .map((pt: any) => {
                const ms = parseTimestamp(pt.time, date);
                return { time: Math.floor(ms / 1000) as any, value: pt.value };
              })
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));
            if (!engineDataCollected && line.name === 'DIF') {
              engineData = padDataStart(pts, firstKlineTime);
              engineDataCollected = true;
            }
            if (line.name === 'DIF' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FFFFFF',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if (line.name === 'DEA' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FFD700',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if (line.name === 'MACD_Bar' && pts.length > 0) {
              const barPoints = pts.map((p: any) => ({
                ...p,
                color: p.value >= 0 ? '#FF4444' : '#44FF44',
              }));
              const hs = subChart.addSeries(lightweightCharts.HistogramSeries, {
                priceFormat: { type: 'volume' },
              } as any);
              hs.setData(barPoints);
              lineSeriesList.push(hs);
            }
          } else if (sc.id === 'rsi') {
            if (!line.data || line.data.length === 0) continue;
            const pts = line.data
              .filter((pt: any) => pt.time)
              .map((pt: any) => {
                const ms = parseTimestamp(pt.time, date);
                return { time: Math.floor(ms / 1000) as any, value: pt.value };
              })
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));
            if (!engineDataCollected && line.name === 'RSI') {
              engineData = padDataStart(pts, firstKlineTime);
              engineDataCollected = true;
            }
            if (line.name === 'RSI' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#4488FF',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
              engineLastValueSeries = ls;
            } else if ((line.name === 'rsi_overbought' || line.name === 'RSI_Overbought') && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FF444488',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if ((line.name === 'rsi_oversold' || line.name === 'RSI_Oversold') && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#44FF4488',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            }
          } else if (sc.id === 'kdj') {
            if (!line.data || line.data.length === 0) continue;
            const pts = line.data
              .filter((pt: any) => pt.time)
              .map((pt: any) => {
                const ms = parseTimestamp(pt.time, date);
                return { time: Math.floor(ms / 1000) as any, value: pt.value };
              })
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));

            if (!engineDataCollected && line.name === 'K') {
              engineData = padDataStart(pts, firstKlineTime);
              engineDataCollected = true;
            }

            if (line.name === 'K' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FFFF00',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if (line.name === 'D' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#4488FF',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if (line.name === 'J' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#AA44FF',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            }
          } else if (sc.id === 'mfi') {
            if (!line.data || line.data.length === 0) continue;
            const pts = line.data
              .filter((pt: any) => pt.time)
              .map((pt: any) => {
                const ms = parseTimestamp(pt.time, date);
                return { time: Math.floor(ms / 1000) as any, value: pt.value };
              })
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));

            if (!engineDataCollected && line.name === 'mfi_value') {
              engineData = padDataStart(pts, firstKlineTime);
              engineDataCollected = true;
            }

            if (line.name === 'mfi_value' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FF8C00',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
              engineLastValueSeries = ls;
            } else if (line.name === 'mfi_ob' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#FF444488',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            } else if (line.name === 'mfi_os' && pts.length > 0) {
              const ls = subChart.addSeries(lightweightCharts.LineSeries, {
                color: '#44FF4488',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              ls.setData(pts);
              lineSeriesList.push(ls);
            }
          } else {
            const ls = subChart.addSeries(lightweightCharts.LineSeries, {
              color: line.color,
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: true,
              crosshairMarkerVisible: true,
            });

            const points = line.data
              .filter((pt) => pt.time)
              .map((pt) => {
                const ms = parseTimestamp(pt.time, date);
                return {
                  time: Math.floor(ms / 1000) as any,
                  value: pt.value,
                };
              })
              .sort((a, b) => (a.time as number) - (b.time as number));

            if (!engineDataCollected) {
              engineData = padDataStart(points, firstKlineTime);
              engineDataCollected = true;
            }
            ls.setData(points);
            lineSeriesList.push(ls);
          }
        }

        // ── 注册子图到同步引擎 ──
        const primarySeries = lineSeriesList.find(
          (s: any) => {
            try { return s.seriesType?.() !== 'Histogram'; } catch { return true; }
          },
        ) || lineSeriesList[0];
        if (engineData.length > 0 && primarySeries) {
          syncEngineRef.current.register(
            sc.id,
            subChart,
            primarySeries,
            engineData,
            engineLastValueSeries ? { lastValueSeries: engineLastValueSeries } : undefined,
          );
          console.log(`[SubChart] REGISTERED: id=${sc.id}`);
        } else {
          console.warn(`[SubChart] SKIPPED registration: id=${sc.id} engineDataLen=${engineData.length} hasPrimarySeries=${!!primarySeries}`);
        }

        // 十字线订阅
        crosshairSubsRef.current.push(
          subChart.subscribeCrosshairMove((param) =>
            syncEngineRef.current.handleMove(sc.id, param),
          ),
        );

        // 时间轴同步
        const subHandleTimeScaleChange = () => {
          if (isTimeSyncingRef.current) return;
          if (isInitialRenderRef.current) return;
          isTimeSyncingRef.current = true;
          try {
            const sr = subChart.timeScale().getVisibleRange();
            if (sr && sr.from && sr.to) {
              syncEngineRef.current.syncTimeRange(sr);
            }
          } finally {
            setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
          }
        };
        subChart.timeScale().subscribeVisibleTimeRangeChange(subHandleTimeScaleChange);

        indicatorChartsRef.current.set(sc.id, subChart);
        indicatorSeriesRef.current.set(sc.id, lineSeriesList);
      }

      // 初次同步：多帧重试确保所有子图时间轴对齐
      // lightweight-charts 内部布局是异步的，单次 setVisibleRange 可能在布局完成前被忽略
      if (klines.length >= 2) {
        const firstTime = klines[0].time as number;
        const lastTime = klines[klines.length - 1].time as number;
        if (firstTime > 0 && lastTime > firstTime) {
          const range = { from: firstTime as any, to: lastTime as any };
          syncEngineRef.current.syncTimeRange(range);
          requestAnimationFrame(() => {
            syncEngineRef.current.syncTimeRange(range);
            requestAnimationFrame(() => {
              syncEngineRef.current.syncTimeRange(range);
            });
          });
        }
      }

      // ── 直接在 renderData 中创建信号标记，避免时序问题 ──
      const markSeries = candleSeriesRef.current;
      if (markSeries) {
        if (seriesMarkersRef.current) {
          try { seriesMarkersRef.current.setMarkers([]); } catch (e) { /* ignore */ }
          try { seriesMarkersRef.current.detach(); } catch (e) { /* ignore */ }
          seriesMarkersRef.current = null;
        }
        const signalsForMarkers = filteredSignalsRef.current;
        const markers: lightweightCharts.SeriesMarker<lightweightCharts.Time>[] = [];
        for (const sig of signalsForMarkers) {
          const sigMs = parseTimestamp(sig.trigger_time, date);
          const sigUnix = Math.floor(sigMs / 1000);
          if (sigUnix <= 0) continue;
          if (sig.signal_type === 'buy') {
            markers.push({
              time: sigUnix as any,
              position: 'belowBar',
              color: '#FF2222',
              shape: 'arrowUp',
              text: '',
              size: 1,
            });
          } else {
            markers.push({
              time: sigUnix as any,
              position: 'aboveBar',
              color: '#22DD44',
              shape: 'arrowDown',
              text: '',
              size: 1,
            });
          }
        }
        seriesMarkersRef.current = createSeriesMarkers(markSeries, markers as any);
      }

      // ── 延迟清除初始化标志：多帧等待后启用双向同步，防止子图异步布局覆盖主图范围 ──
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            isInitialRenderRef.current = false;
          });
        });
      });
    },
    [],
  );

  const stopReplay = useCallback(() => {
    if (replayTimerRef.current) {
      clearInterval(replayTimerRef.current);
      replayTimerRef.current = null;
    }

    restoreChartInteraction();

    setIsReplaying(false);
    setIsReplayPaused(false);
    replayIndexRef.current = 0;

    if (fullSubChartDataBackupRef.current) {
      const data = fullSubChartDataBackupRef.current;
      const date = data.date || todayDateStr;
      requestAnimationFrame(() => {
        renderData(data, date);
      });
    }

    fullKlineBackupRef.current = null;
    fullSubChartDataBackupRef.current = null;
    indicatorSubChartsBackupRef.current = null;
    signalsBackupRef.current = null;
  }, [renderData, restoreChartInteraction, todayDateStr]);

  // 复盘：将累积的K线数据 slice 设置到图表各系列中
  const applyReplayFrame = useCallback(
    (klines: any[], idx: number, dateStr: string, subChartsData: any[] | null) => {
      const slice = klines.slice(0, idx + 1);
      const currentKline = klines[idx];
      const currentTime = currentKline.time as number;

      // 更新主图K线/白线
      if (candleSeriesRef.current) {
        if (isTencentDataRef.current) {
          (candleSeriesRef.current as any).setData(
            slice.map((k) => ({
              time: k.time as any,
              value: k.close,
            })),
          );
        } else {
          (candleSeriesRef.current as any).setData(
            slice.map((k) => ({
              time: k.time as any,
              open: k.open,
              high: k.high,
              low: k.low,
              close: k.close,
            })),
          );
        }
      }

      // 更新成交量
      if (volumeSeriesRef.current) {
        volumeSeriesRef.current.setData(
          slice.map((k, i) => {
            const prevClose = i > 0 ? slice[i - 1].close : slice[0].close;
            const isUp = isTencentDataRef.current ? k.close >= prevClose : k.close >= k.open;
            return {
              time: k.time as any,
              value: k.volume,
              color: isUp ? '#FF444466' : '#44AA4466',
            };
          }),
        );
      }

      // 更新均价线
      if (avgPriceLineRef.current) {
        avgPriceLineRef.current.setData(
          slice
            .filter((k: any) => k.avgPrice != null)
            .map((k: any) => ({
              time: k.time,
              value: k.avgPrice,
            })),
        );
      }

      // 更新成交量均线
      const volumeValues = slice.map((k) => ({ time: k.time as any, value: k.volume }));
      const volume5MA = slice.length >= 5 ? calculateVolumeMA(volumeValues, 5) : [];
      const volume10MA = slice.length >= 10 ? calculateVolumeMA(volumeValues, 10) : [];
      if (volume5MASeriesRef.current) volume5MASeriesRef.current.setData(volume5MA);
      if (volume10MASeriesRef.current) volume10MASeriesRef.current.setData(volume10MA);

      // 时间窗口管理：数据始终填满画布（与真实交易日行为一致）
      // 使用 setVisibleRange 替代 fitContent，避免 ghost 系列（全时段数据）干扰时间范围计算
      const sliceFirstTime = slice[0].time as number;
      const sliceLastTime = slice[slice.length - 1].time as number;
      // 保证最小时间跨度（至少60秒），避免 setVisibleRange from===to 报错
      const minSpan = 60;
      const effectiveFrom = sliceFirstTime as any;
      const effectiveTo =
        sliceLastTime > sliceFirstTime ? (sliceLastTime as any) : ((sliceFirstTime + minSpan) as any);

      if (chartRef.current) {
        chartRef.current.timeScale().setVisibleRange({ from: effectiveFrom, to: effectiveTo });
      }
      if (volumeChartRef.current) {
        volumeChartRef.current.timeScale().setVisibleRange({ from: effectiveFrom, to: effectiveTo });
      }

      // 更新指标子图
      if (subChartsData && subChartsData.length > 0) {
        subChartsData.forEach((sc) => {
          const lineSeriesList = indicatorSeriesRef.current.get(sc.id);
          if (!lineSeriesList || lineSeriesList.length === 0) return;

          sc.lines.forEach((line: any, li: number) => {
            if (li >= lineSeriesList.length) return;
            const series = lineSeriesList[li];
            if (!series || !line.data) return;

            const filteredData = line.data
              .filter((pt: any) => {
                if (!pt.time) return false;
                const ptMs = parseTimestamp(pt.time, dateStr);
                const ptSec = Math.floor(ptMs / 1000);
                return ptSec <= currentTime;
              })
              .map((pt: any) => {
                const ms = parseTimestamp(pt.time, dateStr);
                return { time: Math.floor(ms / 1000) as any, value: pt.value };
              })
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));

            if (filteredData.length > 0) {
              try {
                series.setData(filteredData);
              } catch (_e) { /* ignore */ }
            }
          });
        });

        // 同步子图时间轴
        indicatorChartsRef.current.forEach((ic) => {
          try {
            ic.timeScale().setVisibleRange({ from: effectiveFrom, to: effectiveTo });
          } catch (_e) { /* ignore */ }
        });
      }

      // 更新信号标记（模拟真实交易日信号逐一出现）
      if (candleSeriesRef.current && signalsBackupRef.current && signalsBackupRef.current.length > 0) {
        const activeSignals = signalsBackupRef.current.filter((sig) => {
          const sigMs = parseTimestamp(sig.trigger_time, dateStr);
          const sigUnix = Math.floor(sigMs / 1000);
          return sigUnix > 0 && sigUnix <= currentTime;
        });
        const newMarkers: lightweightCharts.SeriesMarker<lightweightCharts.Time>[] = [];
        for (const sig of activeSignals) {
          const sigUnix = Math.floor(parseTimestamp(sig.trigger_time, dateStr) / 1000);
          if (sigUnix <= 0) continue;
          if (sig.signal_type === 'buy') {
            newMarkers.push({
              time: sigUnix as any,
              position: 'belowBar',
              color: '#FF2222',
              shape: 'arrowUp',
              text: '',
              size: 1,
            });
          } else if (sig.signal_type === 'sell') {
            newMarkers.push({
              time: sigUnix as any,
              position: 'aboveBar',
              color: '#22DD44',
              shape: 'arrowDown',
              text: '',
              size: 1,
            });
          }
        }
        if (seriesMarkersRef.current) {
          try { seriesMarkersRef.current.setMarkers([]); } catch (_e) { /* ignore */ }
          try { seriesMarkersRef.current.detach(); } catch (_e) { /* ignore */ }
          seriesMarkersRef.current = null;
        }
        if (newMarkers.length > 0) {
          seriesMarkersRef.current = createSeriesMarkers(
            candleSeriesRef.current as any,
            newMarkers as any,
          );
        }
      }
    },
    [],
  );

  const startReplay = useCallback(() => {
    const klines = klineRawDataRef.current;
    if (!klines || klines.length === 0) return;

    fullKlineBackupRef.current = [...klines];
    fullSubChartDataBackupRef.current = intradayData;
    indicatorSubChartsBackupRef.current =
      intradayData?.indicator_sub_charts
        ? [...intradayData.indicator_sub_charts]
        : null;
    signalsBackupRef.current = filteredSignalsRef.current
      ? [...filteredSignalsRef.current]
      : null;

    // 清除当前信号标记
    if (seriesMarkersRef.current) {
      try { seriesMarkersRef.current.setMarkers([]); } catch (_e) { /* ignore */ }
      try { seriesMarkersRef.current.detach(); } catch (_e) { /* ignore */ }
      seriesMarkersRef.current = null;
    }

    const dateStr = intradayData?.date || todayDateStr;

    setIsReplaying(true);
    setIsReplayPaused(false);
    replayIndexRef.current = 0;

    applyReplayInteractionLock();

    applyReplayFrame(klines, 0, dateStr, indicatorSubChartsBackupRef.current);

    replayTimerRef.current = setInterval(() => {
      replayIndexRef.current++;
      const idx = replayIndexRef.current;
      if (idx >= klines.length) {
        stopReplay();
        return;
      }
      applyReplayFrame(klines, idx, dateStr, indicatorSubChartsBackupRef.current);
    }, 80);
  }, [intradayData, applyReplayInteractionLock, applyReplayFrame, stopReplay, todayDateStr]);

  const pauseReplay = useCallback(() => {
    if (replayTimerRef.current) {
      clearInterval(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    setIsReplayPaused(true);
  }, []);

  const resumeReplay = useCallback(() => {
    const klines = klineRawDataRef.current;
    if (!klines || klines.length === 0) return;

    const dateStr = fullSubChartDataBackupRef.current?.date || todayDateStr;

    setIsReplayPaused(false);

    replayTimerRef.current = setInterval(() => {
      replayIndexRef.current++;
      const idx = replayIndexRef.current;
      if (idx >= klines.length) {
        stopReplay();
        return;
      }
      applyReplayFrame(klines, idx, dateStr, indicatorSubChartsBackupRef.current);
    }, 80);
  }, [applyReplayFrame, stopReplay, todayDateStr]);

  // 当数据变更时重新渲染
  useEffect(() => {
    if (intradayData) {
      renderData(intradayData, intradayData.date || todayDateStr);
    }
  }, [intradayData, renderData, priceRangeEnabled]);

  // ── 搜索 ──
  const handleSearch = useCallback(async (overrideWarmup?: boolean, overrideCode?: string) => {
    const code = overrideCode ?? stockCode;
    const v = validateStockCode(code);
    setInputError(v.valid ? undefined : v.message);
    if (!v.valid) return;

    if (isReplayingRef.current) {
      if (replayTimerRef.current) {
        clearInterval(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      restoreChartInteraction();
      setIsReplaying(false);
      setIsReplayPaused(false);
      replayIndexRef.current = 0;
      fullKlineBackupRef.current = null;
    fullSubChartDataBackupRef.current = null;
    indicatorSubChartsBackupRef.current = null;
    signalsBackupRef.current = null;
    }

    setIsLoading(true);
    try {
      const dateParam = todayDateStr.replace(/-/g, '');
      // 盘中轮询已保证缓存/DB最新，手动搜索用 cache_only 避免额外API压力
      // 盘后直接走 full 链路，跳过 cache_only 的无效尝试
      const dataStrategy = isTradingTime() ? 'auto' : 'full';
      const data = await getIntradayData(code, dateParam, dataStrategy, overrideWarmup ?? warmupEnabled);
      setIntradayData(data);
      setInputError(undefined);

      // 保存搜索历史
      try {
        await saveSearchHistory(code, data.stock_name || '', todayDateStr);
        await loadHistory();
      } catch (e) {
        console.warn('保存搜索历史失败:', e);
      }
    } catch (err: any) {
      console.error('获取分时数据失败:', err);
      setInputError(err.message || '获取数据失败');
      setIntradayData(null);
    } finally {
      setIsLoading(false);
    }
  }, [stockCode, loadHistory, isTradingTime, todayDateStr, warmupEnabled, restoreChartInteraction]);

  // 点击历史记录
  const handleHistoryClick = useCallback(
    async (item: SearchHistoryItem) => {
      setStockCode(item.stock_code);
      setSelectedHistoryId(item.id);
      // 清除该股票的铃铛（通过 ref 读取最新值，避免 stale closure）
      const bell = signalBellsRef.current[item.stock_code];
      if (bell) {
        seenSignalTimesRef.current = {
          ...seenSignalTimesRef.current,
          [item.stock_code]: bell.time,
        };
        setSignalBells(prev => {
          const next = { ...prev };
          delete next[item.stock_code];
          return next;
        });
      }
      if (isReplayingRef.current) {
        if (replayTimerRef.current) {
          clearInterval(replayTimerRef.current);
          replayTimerRef.current = null;
        }
        restoreChartInteraction();
        setIsReplaying(false);
        setIsReplayPaused(false);
        replayIndexRef.current = 0;
        fullKlineBackupRef.current = null;
        fullSubChartDataBackupRef.current = null;
        indicatorSubChartsBackupRef.current = null;
        signalsBackupRef.current = null;
      }
      setIsLoading(true);
      try {
        const dateParam = todayDateStr.replace(/-/g, '');
        // 盘中轮询已保证缓存/DB最新，手动点击用 cache_only 避免额外API压力
        // 盘后直接走 full 链路，跳过 cache_only 的无效尝试
        const dataStrategy = isTradingTime() ? 'auto' : 'full';
        const data = await getIntradayData(item.stock_code, dateParam, dataStrategy, warmupEnabled);
        setIntradayData(data);
        setInputError(undefined);
        // 更新搜索历史时间戳，使该纪录置顶
        try {
          await updateSearchHistoryTimestamp(item.id);
          await loadHistory();
        } catch {
          // 即使更新失败也不影响用户使用数据
        }
      } catch (err: any) {
        console.error('获取分时数据失败:', err);
        setInputError(err.message || '获取数据失败');
        setIntradayData(null);
      } finally {
        setIsLoading(false);
      }
    },
    [todayDateStr, loadHistory, isTradingTime, warmupEnabled, restoreChartInteraction],
  );

  // 删除历史
  const handleDeleteHistory = useCallback(
    async (id: number) => {
      await deleteSearchHistory(id);
      await loadHistory();
    },
    [loadHistory],
  );

  // 模拟交易盈亏
  const handleSimulateTrading = useCallback(async () => {
    const code = intradayData?.stock_code;
    if (!code) return;
    setSimulationLoading(true);
    setSimulationReport(null);
    try {
      const report = await simulateTrading(code);
      setSimulationReport(report);
    } catch (err: any) {
      console.error('模拟交易失败:', err);
    } finally {
      setSimulationLoading(false);
    }
  }, [intradayData?.stock_code]);

  // ── 计算信号统计颜色 ──
  const summary = intradayData?.signal_summary;
  const signalStats = summary
    ? [
        { label: '买入信号', value: summary.buy_signals, color: '#FF4444', filterKey: 'buy' },
        { label: '卖出信号', value: summary.sell_signals, color: '#44FF44', filterKey: 'sell' },
        { label: '强信号', value: summary.strong_signals, color: '#FFFF44', filterKey: 'strong' },
        { label: '中信号', value: summary.medium_signals, color: '#FFAA44', filterKey: 'medium' },
        { label: '弱信号', value: summary.weak_signals, color: '#AAAAAA', filterKey: 'weak' },
      ]
    : [];

  const toggleFilter = (key: string) => {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const filteredSignals = useMemo(() => {
    const signals = intradayData?.signals || [];
    if (activeFilters.size === 0) return signals;

    const wantBuy = activeFilters.has('buy');
    const wantSell = activeFilters.has('sell');
    const wantStrong = activeFilters.has('strong');
    const wantMedium = activeFilters.has('medium');
    const wantWeak = activeFilters.has('weak');

    // 如果同时或都不选类型，不限制类型；否则按选中的类型过滤
    const filterType = (wantBuy && wantSell) || (!wantBuy && !wantSell) ? null
      : wantBuy ? 'buy' : 'sell';
    // 强度同理
    const filterStrength = (wantStrong && wantMedium && wantWeak) || (!wantStrong && !wantMedium && !wantWeak) ? null
      : { strong: wantStrong, medium: wantMedium, weak: wantWeak };

    return signals.filter((sig) => {
      if (filterType && sig.signal_type !== filterType) return false;
      if (filterStrength) {
        const conf = sig.confidence;
        if (filterStrength.strong && conf >= 0.75) return true;
        if (filterStrength.medium && conf >= 0.50 && conf < 0.75) return true;
        if (filterStrength.weak && conf < 0.50) return true;
        return false;
      }
      return true;
    });
  }, [intradayData?.signals, activeFilters]);
  filteredSignalsRef.current = filteredSignals;

  // 根据当前筛选后的信号计算模拟收益（仅考虑显示中的做T记录）
  const filteredSimulReturn = useMemo(() => {
    const parseWeight = (advice: string): number => {
      if (!advice) return 1.0;
      if (advice.includes('全仓')) return 1.0;
      if (advice.includes('半仓')) return 0.5;
      if (advice.includes('1/3仓')) return 0.33;
      return 1.0;
    };

    let totalReturn = 0;
    let lastBuy: { price: number; weight: number; time: string } | null = null;
    let unsettledBuy: { price: number; weight: number; time: string } | null = null;

    for (const sig of filteredSignals) {
      if (sig.signal_type === 'buy') {
        const weight = parseWeight(sig.position_advice || '');
        lastBuy = { price: sig.price, weight, time: sig.trigger_time };
      } else if (sig.signal_type === 'sell') {
        const weight = parseWeight(sig.position_advice || '');
        if (lastBuy !== null) {
          const profit = ((sig.price - lastBuy.price) / lastBuy.price * weight - 0.001) * 100;
          totalReturn += profit;
          lastBuy = null;
        }
      }
    }

    if (lastBuy !== null) {
      unsettledBuy = { ...lastBuy };
    }

    return { totalReturn, unsettledBuy };
  }, [filteredSignals]);

  // MACD柱高度和 / 柱高度差：从后端metadata获取（与后端策略算法一致）
  const macdMetadata = useMemo(() => {
    return intradayData?.indicator_sub_charts?.find(sc => sc.id === 'macd')?.metadata ?? null;
  }, [intradayData]);

  const macdBarSum: number = macdMetadata?.bar_sum ?? 0;
  const macdBarDiff: number = macdMetadata?.bar_diff ?? 0;

  // ── 信号标记（受筛选条件调控，随 filteredSignals 变化而更新）──
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;

    if (seriesMarkersRef.current) {
      try {
        seriesMarkersRef.current.setMarkers([]);
        seriesMarkersRef.current.detach();
      } catch (e) { /* ignore */ }
      seriesMarkersRef.current = null;
    }

    const markers: lightweightCharts.SeriesMarker<lightweightCharts.Time>[] = [];
    for (const sig of filteredSignals) {
      const sigMs = parseTimestamp(sig.trigger_time, todayDateStr);
      const sigUnix = Math.floor(sigMs / 1000);
      if (sigUnix <= 0) continue;

      if (sig.signal_type === 'buy') {
        markers.push({
          time: sigUnix as any,
          position: 'belowBar',
          color: '#FF2222',
          shape: 'arrowUp',
          text: '',
          size: 1,
        });
      } else {
        markers.push({
          time: sigUnix as any,
          position: 'aboveBar',
          color: '#22DD44',
          shape: 'arrowDown',
          text: '',
          size: 1,
        });
      }
    }
    seriesMarkersRef.current = createSeriesMarkers(series, markers as any);
  }, [filteredSignals]);

  // ── 侧边栏 ──
  const sidebarContent = (
    <div className="flex flex-col overflow-hidden min-h-0 h-full">
      <div className="p-3 border-b border-white/5 flex-shrink-0">
        <h3 className="text-sm font-medium text-white">搜索历史</h3>
      </div>
      <div className="overflow-y-auto px-3 py-1 h-full">
        {isLoadingHistory ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
          </div>
        ) : searchHistory.length === 0 ? (
          <p className="text-xs text-muted text-center py-4">暂无搜索历史</p>
        ) : (
          <div className="space-y-2">
            {searchHistory.map(item => {
              const isCurrentlyDisplayed = intradayData &&
                intradayData.stock_code === item.stock_code;

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleHistoryClick(item)}
                  className={`history-item w-full text-left ${
                    isCurrentlyDisplayed
                      ? 'ring-2 ring-cyan/60 bg-cyan/10 border-transparent'
                      : (selectedHistoryId === item.id ? 'active' : '')
                  }`}
                >
                  <div className="flex items-center gap-2 w-full">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="font-medium text-white truncate text-xs">
                          {item.stock_name ? `${item.stock_name} (${item.stock_code})` : item.stock_code}
                        </span>
                        {isCurrentlyDisplayed && !signalBells[item.stock_code] ? (
                          <span className="flex-shrink-0 text-cyan text-xs font-medium">
                            展示中
                          </span>
                        ) : signalBells[item.stock_code] ? (
                          <span
                            className="flex-shrink-0"
                            title={`${signalBells[item.stock_code].type === 'buy' ? '买入' : '卖出'}信号 ${signalBells[item.stock_code].time} ¥${signalBells[item.stock_code].price.toFixed(2)}`}
                          >
                            <svg
                              className="w-4 h-4"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M5.5 18a6 6 0 0 1 13 0H5.5z"
                                stroke={signalBells[item.stock_code].type === 'buy' ? '#FF4444' : '#44FF44'}
                                strokeWidth="1.5"
                              />
                              <path
                                d="M6 10a6 6 0 1 1 12 0v4H6v-4z"
                                stroke={signalBells[item.stock_code].type === 'buy' ? '#FF4444' : '#44FF44'}
                                strokeWidth="1.5"
                              />
                              <path
                                d="M9.5 20a2.5 2.5 0 0 0 5 0h-5z"
                                stroke={signalBells[item.stock_code].type === 'buy' ? '#FF4444' : '#44FF44'}
                                strokeWidth="1.5"
                              />
                            </svg>
                          </span>
                        ) : null}
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        {(() => {
                          const snap = historySnapshots[item.stock_code];
                          if (snap && snap.latest_price > 0) {
                            const isUp = snap.change_pct >= 0;
                            return (
                              <>
                                <span className="text-xs text-white font-mono font-medium">
                                  ¥{snap.latest_price.toFixed(2)}
                                </span>
                                <span
                                  className="text-xs font-mono font-medium"
                                  style={{ color: isUp ? '#FF4444' : '#44FF44' }}
                                >
                                  {isUp ? '+' : ''}{snap.change_pct.toFixed(2)}%
                                </span>
                                <span className="text-xs text-muted/50">·</span>
                                <span className="text-xs text-muted">{snap.timestamp.length >= 14 ? snap.timestamp.slice(8, 10) + ':' + snap.timestamp.slice(10, 12) + ':' + snap.timestamp.slice(12, 14) : snap.timestamp}</span>
                              </>
                            );
                          }
                          return (
                            <>
                              <span className="text-xs text-muted font-mono">{item.stock_code}</span>
                              <span className="text-xs text-muted/50">·</span>
                              <span className="text-xs text-muted">
                                {item.search_time ? new Date(item.search_time).toLocaleString('zh-CN') : item.date}
                              </span>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteHistory(item.id);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.stopPropagation();
                          handleDeleteHistory(item.id);
                        }
                      }}
                      className="p-1 text-muted hover:text-danger transition-colors flex-shrink-0"
                      title="删除"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div
      className="min-h-screen flex flex-col md:grid overflow-hidden w-full"
      style={{
        gridTemplateColumns: 'minmax(12px, 1fr) 256px 24px minmax(auto, 1200px) minmax(12px, 1fr)',
        gridTemplateRows: 'auto 1fr auto',
      }}
    >
      {/* 顶部搜索栏 */}
      <header className="py-3 px-3 md:px-0 border-b border-white/5 flex-shrink-0 flex items-center min-w-0 overflow-hidden md:col-start-2 md:col-end-5 md:row-start-1">
        <div className="flex items-center gap-2 w-full min-w-0 flex-1">
          <button
            onClick={() => setSidebarOpen(true)}
            className="md:hidden p-1.5 -ml-1 rounded-lg hover:bg-white/10 transition-colors text-secondary hover:text-white flex-shrink-0"
            title="历史记录"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <div className="flex-1 relative min-w-0">
            <StockSearchInput
              placeholder="输入股票代码，如 600519、00700、AAPL"
              disabled={isLoading}
              error={inputError}
              onChange={(q) => {
                setStockCode(q);
                setInputError(undefined);
              }}
              onSelect={(code) => {
                setStockCode(code);
                setInputError(undefined);
              }}
              onSubmit={(query) => {
                handleSearch(undefined, query.toUpperCase());
              }}
              className="w-full"
            />
            {inputError && (
              <p className="absolute -bottom-4 left-0 text-xs text-danger">{inputError}</p>
            )}
          </div>

          <div className="flex-shrink-0">
            <button
              type="button"
              onClick={() => setPriceRangeEnabled((v) => !v)}
              disabled={isLoading}
              title={priceRangeEnabled ? '昨收±10%范围已激活，点击关闭' : '自动范围，点击激活昨收±10%'}
              className={`text-sm border rounded-lg px-3 py-2 transition-colors flex items-center gap-1.5 ${
                priceRangeEnabled
                  ? 'bg-cyan/15 border-cyan/40 text-cyan hover:bg-cyan/20'
                  : 'bg-[#1a1a2e] border-white/10 text-muted hover:border-white/20'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
              <span>{priceRangeEnabled ? '昨收±10%' : '自动范围'}</span>
            </button>
          </div>

          <button
            type="button"
            onClick={() => handleSearch()}
            disabled={!stockCode || isLoading}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap flex-shrink-0"
          >
            {isLoading ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                加载中
              </>
            ) : (
              '搜索'
            )}
          </button>

          {/* 预热状态切换按钮 */}
            <button
              type="button"
              onClick={async () => {
                const newVal = !warmupEnabled;
                setWarmupEnabled(newVal);
                if (intradayData && stockCode) {
                  setIsLoading(true);
                  try {
                    const dateParam = todayDateStr.replace(/-/g, '');
                    const data = await getIntradayData(stockCode, dateParam, 'full', newVal);
                    setIntradayData(data);
                  } catch (err: any) {
                    console.error('切换预热状态失败:', err);
                  } finally {
                    setIsLoading(false);
                  }
                }
              }}
              disabled={!intradayData}
              className={`text-sm border rounded-lg px-3 py-2 transition-colors flex items-center gap-1.5 ${
                warmupEnabled && intradayData?.warmup_info?.last_klines_count != null && intradayData.warmup_info.last_klines_count > 0
                  ? 'bg-green/15 border-green/40 text-green hover:bg-green/20'
                  : 'bg-[#1a1a2e] border-white/10 text-muted hover:border-white/20'
              }`}
              title={
                warmupEnabled && intradayData?.warmup_info?.last_klines_count != null && intradayData.warmup_info.last_klines_count > 0
                  ? `预热已启用，前日 ${intradayData.warmup_info.last_klines_count} 根K线 (${intradayData.warmup_info.prev_date}) — 点击关闭`
                  : warmupEnabled
                    ? '预热已启用但无前日数据，指标从零状态计算 — 点击关闭'
                    : intradayData
                      ? '预热未启用 — 点击开启'
                      : '请先搜索股票'
              }
            >
              {warmupEnabled && intradayData?.warmup_info?.last_klines_count != null && intradayData.warmup_info.last_klines_count > 0 && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="#ef4444" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0">
                  <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
                </svg>
              )}
              {warmupEnabled && intradayData?.warmup_info?.last_klines_count != null && intradayData.warmup_info.last_klines_count > 0
                ? '预热'
                : intradayData
                  ? '无预热'
                  : '预热'}
            </button>

          <button
            type="button"
            onClick={handleStartBatchDownload}
            disabled={batchDownload?.status === 'running'}
            title="批量下载全市场A股当天分时数据"
            className={`flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded border transition-colors flex-shrink-0 ${
              batchDownload?.status === 'running'
                ? 'text-cyan border-cyan/30 bg-cyan/10'
                : 'text-muted border-muted/20 bg-transparent hover:text-cyan hover:border-cyan/40'
            } disabled:opacity-60 disabled:cursor-not-allowed`}
          >
            {batchDownload?.status === 'running' ? (
              <>
                <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
                下载中
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                批量下载
              </>
            )}
          </button>
        </div>
      </header>

      {/* 批量下载进度弹窗 */}
      {showBatchModal && batchDownload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60" onClick={() => {
            if (batchDownload?.status !== 'running') setShowBatchModal(false);
          }} />
          <div className="relative terminal-card border border-white/10 rounded-xl shadow-2xl p-5 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-white">
                {batchDownload.waiting_retry ? '等待重试中...' :
                 batchDownload.paused ? '已暂停' :
                 batchDownload.status === 'running' ? '正在批量下载分时数据' :
                 batchDownload.status === 'completed' ? '批量下载完成' :
                 batchDownload.status === 'cancelled' ? '已取消' : '下载异常'}
              </h3>
              <button
                type="button"
                onClick={() => setShowBatchModal(false)}
                className="text-muted hover:text-white p-0.5"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* 进度条 - Canvas 绘制，完全绕过 React/CSS 渲染管线 */}
            {batchDownload.total > 0 ? (
              <div className="mb-3">
                <canvas
                  ref={canvasRef}
                  className="w-full h-3 block"
                  style={{ borderRadius: '9999px' }}
                />
                <div ref={progressTextRef} className="text-xs text-muted mt-1 text-right">
                  {batchDownload.completed} / {batchDownload.total} ({Math.round(batchDownload.completed / batchDownload.total * 100)}%)
                  <span className="ml-2 text-[10px] text-cyan/50">render #{renderCountRef.current}</span>
                </div>
              </div>
            ) : (
              batchDownload.status === 'running' && (
                <div className="text-xs text-muted mb-3 text-center py-2">
                  正在初始化股票列表...
                </div>
              )
            )}

            {/* 统计信息 */}
            <div key={`stats-${batchDownload.completed}`} className="grid grid-cols-3 gap-2 mb-3 text-xs">
              <div className="bg-white/[0.03] rounded px-2 py-1.5">
                <div className="text-muted">成功</div>
                <div className="text-emerald font-mono">{batchDownload.completed - batchDownload.skipped}</div>
              </div>
              <div className="bg-white/[0.03] rounded px-2 py-1.5">
                <div className="text-muted">跳过</div>
                <div className="text-muted font-mono">{batchDownload.skipped}</div>
              </div>
              <div className="bg-white/[0.03] rounded px-2 py-1.5">
                <div className="text-muted">失败</div>
                <div className="text-red-400 font-mono">{batchDownload.failed}</div>
              </div>
            </div>

            {/* 当前处理 */}
            {batchDownload.status === 'running' && batchDownload.current_code && (
              <div className="text-xs text-muted mb-3">
                当前: <span className="text-white font-mono">{batchDownload.current_code}</span>
                {batchDownload.current_name && <span className="ml-1 text-white/70">{batchDownload.current_name}</span>}
              </div>
            )}

            {/* 耗时 */}
            {batchDownload.elapsed_seconds > 0 && (
              <div className="text-xs text-muted mb-3">
                耗时: {Math.floor(batchDownload.elapsed_seconds / 60)}分{Math.floor(batchDownload.elapsed_seconds % 60)}秒
              </div>
            )}

            {/* 等待重试倒计时 */}
            {batchDownload.waiting_retry && batchDownload.retry_countdown > 0 && (
              <div className="mb-3 px-2 py-1.5 rounded bg-yellow-500/10 border border-yellow-500/20">
                <div className="text-xs text-yellow-400 flex items-center gap-1.5">
                  <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                    <path d="M12 2a10 10 0 019.95 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  检测到频率限制，{batchDownload.retry_countdown} 秒后重试失败项...
                </div>
              </div>
            )}

            {/* 暂停状态提示 */}
            {batchDownload.paused && batchDownload.status === 'running' && (
              <div className="mb-3 px-2 py-1.5 rounded bg-blue-500/10 border border-blue-500/20">
                <div className="text-xs text-blue-400 flex items-center gap-1.5">
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                  </svg>
                  下载已暂停，点击"继续下载"恢复
                </div>
              </div>
            )}

            {/* 日期 */}
            {batchDownload.date && (
              <div className="text-xs text-muted mb-3">日期: {batchDownload.date}</div>
            )}

            {/* 错误列表 */}
            {batchDownload.errors.length > 0 && (
              <div className="mb-3">
                <div className="text-xs text-red-400 mb-1">最近错误:</div>
                <div className="max-h-24 overflow-y-auto text-xs font-mono bg-black/20 rounded px-2 py-1">
                  {batchDownload.errors.map((e, i) => (
                    <div key={i} className="text-red-400/80 truncate">
                      {e.code}: {e.error}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 按钮 */}
            <div className="flex items-center gap-2">
              {batchDownload.status === 'running' && (
                <>
                  <button
                    type="button"
                    onClick={handleTogglePause}
                    className={`px-3 py-1.5 text-xs font-medium rounded border transition-colors ${
                      batchDownload.paused
                        ? 'border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10'
                        : 'border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/10'
                    }`}
                  >
                    {batchDownload.paused ? '继续下载' : '暂停下载'}
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelBatchDownload}
                    className="px-3 py-1.5 text-xs font-medium rounded border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    取消下载
                  </button>
                </>
              )}
              {batchDownload.status !== 'running' && (
                <button
                  type="button"
                  onClick={() => setShowBatchModal(false)}
                  className="px-3 py-1.5 text-xs font-medium rounded border border-white/10 text-muted hover:text-white hover:border-white/30 transition-colors"
                >
                  关闭
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Desktop 侧边栏 */}
      <div className="hidden md:flex col-start-2 row-start-2 flex-col overflow-hidden min-h-0 h-full">
        {sidebarContent}
      </div>

      {/* Mobile 侧边栏 */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 md:hidden" onClick={() => setSidebarOpen(false)}>
          <div className="absolute inset-0 bg-black/60" />
          <div
            className="absolute left-0 top-0 bottom-0 w-72 flex flex-col terminal-card overflow-hidden border-r border-white/10 shadow-2xl p-3"
            onClick={(e) => e.stopPropagation()}
          >
            {sidebarContent}
          </div>
        </div>
      )}

      {/* 中央图表区域 */}
      <section className="flex-1 overflow-y-auto overflow-x-auto px-3 md:px-0 md:pl-1 min-w-0 min-h-0 md:col-start-4 md:row-start-2">
        <div className="max-w-6xl">
          {intradayData && (
            <div className="mb-4 mt-2">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-white">
                    {intradayData.stock_name
                      ? `${intradayData.stock_name} (${intradayData.stock_code})`
                      : intradayData.stock_code}
                    <span className="text-sm text-muted ml-3">{intradayData.date}</span>
                  </h2>
                  <div className="text-xs text-muted mt-1">
                    分时K线: {intradayData.kline_data?.length || 0} 条 · 信号: {intradayData.signals?.length || 0} 个
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <button
                    type="button"
                    onClick={handleSimulateTrading}
                    disabled={simulationLoading || isReplaying}
                    className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors ${
                      simulationReport
                        ? 'text-emerald border-emerald/40 bg-emerald/10 hover:bg-emerald/20'
                        : 'text-muted border-muted/30 bg-transparent hover:text-white hover:border-white/50'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {simulationLoading ? (
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
                        计算中
                      </span>
                    ) : simulationReport ? (
                      `模拟交易 ${simulationReport.total_return_pct >= 0 ? '+' : ''}${simulationReport.total_return_pct.toFixed(2)}%`
                    ) : (
                      '模拟交易'
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (isReplaying) {
                        if (isReplayPaused) resumeReplay();
                        else pauseReplay();
                      } else {
                        startReplay();
                      }
                    }}
                    disabled={!intradayData}
                    className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors ${
                      isReplaying
                        ? 'text-cyan border-cyan/40 bg-cyan/10 hover:bg-cyan/20'
                        : 'text-muted border-muted/30 bg-transparent hover:text-white hover:border-white/50'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isReplaying ? (isReplayPaused ? '继续' : '暂停') : '复盘'}
                  </button>
                  {simulationReport && (
                    <button
                      type="button"
                      onClick={() => setSimulationReport(null)}
                      className="text-muted hover:text-white text-xs p-0.5"
                      title="关闭"
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M18 6L6 18M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
              {simulationReport && (
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted bg-white/[0.03] rounded border border-white/[0.06] px-3 py-2">
                  <span>总交易: <strong className="text-white">{simulationReport.total_trades}</strong></span>
                  <span>盈利: <strong className="text-emerald">{simulationReport.win_trades}</strong></span>
                  <span>亏损: <strong className="text-red-400">{simulationReport.lose_trades}</strong></span>
                  <span>胜率: <strong className="text-white">{(simulationReport.win_rate * 100).toFixed(1)}%</strong></span>
                  <span>累计收益: <strong className={simulationReport.total_return_pct >= 0 ? 'text-emerald' : 'text-red-400'}>
                    {simulationReport.total_return_pct >= 0 ? '+' : ''}{simulationReport.total_return_pct.toFixed(2)}%
                  </strong></span>
                  <span>平均收益: <strong className={simulationReport.avg_return_pct >= 0 ? 'text-emerald' : 'text-red-400'}>
                    {simulationReport.avg_return_pct >= 0 ? '+' : ''}{simulationReport.avg_return_pct.toFixed(2)}%
                  </strong></span>
                  <span>最大回撤: <strong className="text-red-400">{simulationReport.max_drawdown_pct.toFixed(2)}%</strong></span>
                  <span>盈亏比: <strong className="text-white">{simulationReport.profit_factor.toFixed(2)}</strong></span>
                </div>
              )}
            </div>
          )}

          <div className="w-full">
            <div className="flex items-start gap-2 mb-2" style={{ height: 165 }}>
              {signalStats.length > 0 && (
                <Card variant="default" padding="sm" className="w-44 flex-shrink-0">
                  <h3 className="text-xs font-medium text-muted mb-2">
                    信号统计
                    {activeFilters.size > 0 && (
                      <button
                        type="button"
                        className="ml-2 text-[10px] text-cyan hover:text-white transition-colors"
                        onClick={() => setActiveFilters(new Set())}
                      >
                        清除筛选
                      </button>
                    )}
                  </h3>
                  <div className="space-y-1">
                    {signalStats.map((s) => {
                      const isActive = s.filterKey && activeFilters.has(s.filterKey);
                      const isClickable = !!s.filterKey;
                      return (
                        <div
                          key={s.label}
                          className={`flex items-center justify-between ${isClickable ? 'cursor-pointer hover:bg-white/5 rounded px-1 -mx-1 py-0.5 transition-colors' : ''}`}
                          style={isActive ? {
                            backgroundColor: `${s.color}18`,
                            borderLeft: `3px solid ${s.color}`,
                            paddingLeft: '6px',
                            borderRadius: '2px',
                            fontWeight: 600,
                          } : undefined}
                          onClick={isClickable ? () => toggleFilter(s.filterKey!) : undefined}
                          title={isClickable ? '点击筛选' : undefined}
                        >
                          <span
                            className="text-xs"
                            style={{ color: isActive ? s.color : undefined }}
                          >
                            {isActive ? '▸ ' : ''}{s.label}
                          </span>
                          <span
                            className="text-xs font-mono font-medium"
                            style={{ color: s.color }}
                          >
                            {typeof s.value === 'number' && s.label.includes('%')
                              ? `${s.value.toFixed(2)}%`
                              : s.value}
                          </span>
                        </div>
                      );
                    })}
                </div>
                </Card>
              )}

              {hoveredWeightDetails && (
                <>
                  <Card variant="default" padding="sm" className="w-44 flex-shrink-0" style={{ height: 165 }}>
                    <div className="overflow-y-auto h-full">
                      <h3 className="text-xs font-medium text-muted mb-2">买入权重贡献</h3>
                      <div className="space-y-0.5">
                        <div className="text-xs font-mono mb-1" style={{ color: '#FF4444' }}>
                          总分: {hoveredWeightDetails.buy.reduce((s, d) => s + d.score, 0)}
                          /{hoveredWeightDetails.buy.reduce((s, d) => s + d.weight, 0)}
                        </div>
                        {[...hoveredWeightDetails.buy].sort((a, b) => (b.triggered ? 1 : 0) - (a.triggered ? 1 : 0)).map((d) => (
                          <div key={d.key} className="flex items-center justify-between text-[11px]">
                            <span className={d.triggered ? 'text-white/80' : 'text-muted/40'}>
                              {d.triggered ? '✓' : '✗'} {d.label}
                            </span>
                            <span className={`font-mono ${d.triggered ? 'text-accent' : 'text-muted/30'}`}>
                              +{d.score}
                            </span>
                          </div>
                        ))}
                        <div className="border-t border-white/10 pt-1 mt-1">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-white/60">支撑力</span>
                            <span className="font-mono" style={{ color: '#FF4444' }}>
                              {hoveredWeightDetails.supportForce.toFixed(1)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-white/60">净力</span>
                            <span className="font-mono" style={{ color: (hoveredWeightDetails.supportForce - hoveredWeightDetails.pressureForce) >= 0 ? '#FF4444' : '#44FF44' }}>
                              {(hoveredWeightDetails.supportForce - hoveredWeightDetails.pressureForce).toFixed(1)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>

                  <Card variant="default" padding="sm" className="w-44 flex-shrink-0" style={{ height: 165 }}>
                    <div className="overflow-y-auto h-full">
                      <h3 className="text-xs font-medium text-muted mb-2">卖出权重贡献</h3>
                      <div className="space-y-0.5">
                        <div className="text-xs font-mono mb-1" style={{ color: '#44FF44' }}>
                          总分: {hoveredWeightDetails.sell.reduce((s, d) => s + d.score, 0)}
                          /{hoveredWeightDetails.sell.reduce((s, d) => s + d.weight, 0)}
                        </div>
                        {[...hoveredWeightDetails.sell].sort((a, b) => (b.triggered ? 1 : 0) - (a.triggered ? 1 : 0)).map((d) => (
                          <div key={d.key} className="flex items-center justify-between text-[11px]">
                            <span className={d.triggered ? 'text-white/80' : 'text-muted/40'}>
                              {d.triggered ? '✓' : '✗'} {d.label}
                            </span>
                            <span className={`font-mono ${d.triggered ? 'text-accent' : 'text-muted/30'}`}>
                              +{d.score}
                            </span>
                          </div>
                        ))}
                        <div className="border-t border-white/10 pt-1 mt-1">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-white/60">压力力</span>
                            <span className="font-mono" style={{ color: '#44FF44' }}>
                              {hoveredWeightDetails.pressureForce.toFixed(1)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-white/60">净力</span>
                            <span className="font-mono" style={{ color: (hoveredWeightDetails.supportForce - hoveredWeightDetails.pressureForce) >= 0 ? '#FF4444' : '#44FF44' }}>
                              {(hoveredWeightDetails.supportForce - hoveredWeightDetails.pressureForce).toFixed(1)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>
                </>
              )}

              {/* 五档买卖盘口面板 - 腾讯 qt.gtimg.cn 数据 */}
              {stockCode && historySnapshots[stockCode] && (() => {
                const snap = historySnapshots[stockCode];
                const hasDepth = snap.bid_prices?.length >= 5 && snap.ask_prices?.length >= 5;
                return (
                  <Card variant="default" padding="sm" className="w-36 flex-shrink-0 flex-[7] min-w-0" style={{ height: 165 }}>
                    <div className="overflow-y-auto h-full">
                      {!hasDepth ? (
                        <div className="text-xs text-muted/40">暂无盘口数据</div>
                      ) : (
                        <table className="w-full text-[10px] leading-tight">
                          <tbody>
                            {/* 卖五→卖一，绿色 */}
                            {[4, 3, 2, 1, 0].map((i) => (
                              <tr key={`ask-${i}`} className="border-b border-white/[0.03]">
                                <td className="text-muted/50 py-0 w-6">卖{i + 1}</td>
                                <td className={`text-right font-mono py-0 ${snap.ask_prices[i] > 0 ? 'text-green-400/80' : 'text-muted/30'}`}>
                                  {snap.ask_prices[i] > 0 ? snap.ask_prices[i].toFixed(2) : '-'}
                                </td>
                                <td className="text-right text-muted/70 font-mono py-0 w-11">
                                  {snap.ask_volumes[i] > 0 ? `${snap.ask_volumes[i]}手` : '-'}
                                </td>
                              </tr>
                            ))}
                            <tr>
                              <td colSpan={3}><div className="border-t border-white/15 my-0.5" /></td>
                            </tr>
                            {/* 买一→买五，红色 */}
                            {[0, 1, 2, 3, 4].map((i) => (
                              <tr key={`bid-${i}`} className="border-b border-white/[0.03]">
                                <td className="text-muted/50 py-0 w-6">买{i + 1}</td>
                                <td className={`text-right font-mono py-0 ${snap.bid_prices[i] > 0 ? 'text-red-400/80' : 'text-muted/30'}`}>
                                  {snap.bid_prices[i] > 0 ? snap.bid_prices[i].toFixed(2) : '-'}
                                </td>
                                <td className="text-right text-muted/70 font-mono py-0 w-11">
                                  {snap.bid_volumes[i] > 0 ? `${snap.bid_volumes[i]}手` : '-'}
                                </td>
                              </tr>
                            ))}
                            <tr>
                              <td colSpan={3} className="pt-1">
                                <div className="flex items-center justify-between text-[9px] font-semibold text-muted/70">
                                  <span>买总：<span className="text-red-400/90">{calcDepthTotal(snap.bid_volumes)}手</span></span>
                                  <span className="text-muted/50">|</span>
                                  <span>卖总：<span className="text-green-400/90">{calcDepthTotal(snap.ask_volumes)}手</span></span>
                                </div>
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      )}
                    </div>
                  </Card>
                );
              })()}

              {intradayData && filteredSignals.length > 0 && (
                <div className="flex-[11] overflow-y-auto" style={{ height: 165 }}>
                  <div className="text-xs text-muted/60 mb-1 flex items-center gap-3">
                    <span>共 {filteredSignals.length} 条信号</span>
                    <span className="font-mono font-medium" style={{ color: filteredSimulReturn.totalReturn >= 0 ? '#FF4444' : '#44FF44' }}>
                      模拟收益 {filteredSimulReturn.totalReturn >= 0 ? '+' : ''}{filteredSimulReturn.totalReturn.toFixed(2)}%
                    </span>
                    {filteredSimulReturn.unsettledBuy && (
                      <span className="text-yellow-400/80" title={`未平仓买入: ¥${filteredSimulReturn.unsettledBuy.price.toFixed(2)} ${filteredSimulReturn.unsettledBuy.time}`}>
                        (浮动盈亏待结算)
                      </span>
                    )}
                    {activeFilters.size > 0 && (
                      <span className="text-cyan">（已筛选）</span>
                    )}
                  </div>
                  <div className="space-y-1">
                    {filteredSignals.slice().reverse().map((sig, idx) => {
                      const sigTimeStr = (() => {
                        const parts = sig.trigger_time.split('T');
                        const tm = parts[1] || parts[0] || '';
                        return tm.length > 5 ? tm.substring(0, 5) : tm;
                      })();
                      const handleSignalClick = () => {
                        const ch = chartRef.current;
                        if (!ch || !intradayData?.kline_data?.length) return;
                        const parts = sig.trigger_time.split('T');
                        const timePart = parts[1] || parts[0] || '';
                        const ms = parseTimestamp(timePart, intradayData.date);
                        const targetSec = Math.floor(ms / 1000);
                        if (targetSec <= 0) return;
                        const klines = intradayData.kline_data;
                        const kp = klines.find((k: any) => {
                          const kMs = parseTimestamp(k.timestamp, intradayData.date);
                          const kSec = Math.floor(kMs / 1000);
                          return kSec === targetSec;
                        });
                        if (!kp) return;
                        syncEngineRef.current.setCrosshairAtTime(targetSec as any);
                        setIsCrosshairActive(true);
                        setHoveredWeightDetails({
                          buy: sig.buy_weight_details || [],
                          sell: sig.sell_weight_details || [],
                          supportForce: sig.support_force || 0,
                          pressureForce: sig.pressure_force || 0,
                          signalType: sig.signal_type,
                        });
                      };
                      return (
                        <div
                          key={idx}
                          className="flex items-center gap-2 text-xs py-1 px-2 rounded bg-white/5 cursor-pointer hover:bg-white/10 transition-colors"
                          onClick={handleSignalClick}
                          title="点击定位到该时间点"
                        >
                          <span
                            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                            style={{
                              backgroundColor: sig.signal_type === 'buy' ? '#FF4444' : '#44FF44',
                            }}
                          />
                          <span className="text-muted font-mono w-16">{sigTimeStr}</span>
                        <span
                          className="font-medium"
                          style={{ color: sig.signal_type === 'buy' ? '#FF4444' : '#44FF44' }}
                        >
                          {sig.signal_type === 'buy' ? '买入' : '卖出'}
                        </span>
                        <span className="text-white font-mono">¥{sig.price.toFixed(2)}</span>
                        <span className="text-muted/70">
                          信号等级 {signalLevel(sig.confidence).label}
                        </span>
                        <span
                          className="text-[10px] px-1 py-0.5 rounded"
                          style={{
                            backgroundColor: sig.signal_type === 'buy' ? '#FF444420' : '#44FF4420',
                            color: sig.signal_type === 'buy' ? '#FF6666' : '#66FF66',
                          }}
                        >
                          {sig.position_advice}
                        </span>
                      </div>
                    );
                  })}
                  </div>
                </div>
              )}
            </div>

            {/* 主K线图 */}
            <Card variant="default" padding="none" className="mb-2">
              <div style={{ position: 'relative' }}>
                <div ref={chartContainerRef} style={{ width: '100%', height: CHART_HEIGHT }} />
                {(() => {
                  const devPct = isCrosshairActive
                    ? crosshairDeviationPct
                    : (intradayData?.indicator_sub_charts
                        ?.find((sc: any) => sc.id === 'avg_price_deviation')
                        ?.lines?.find((l: any) => l.name === 'deviation_pct')
                        ?.data?.slice(-1)[0]?.value ?? null);
                  if (devPct == null) return null;
                  const isOversold = devPct <= -2.5;
                  const isOverbought = devPct >= 2.5;
                  const textColor = isOverbought ? '#FF4444' : isOversold ? '#44FF44' : '#d1d4dc';
                  return (
                    <div
                      style={{
                        position: 'absolute',
                        top: 8,
                        left: 12,
                        zIndex: 10,
                        fontSize: 12,
                        fontFamily: 'monospace',
                        fontWeight: 600,
                        color: textColor,
                        backgroundColor: 'rgba(26,26,46,0.85)',
                        padding: '2px 8px',
                        borderRadius: 4,
                        pointerEvents: 'none',
                      }}
                    >
                      均价偏离 {devPct >= 0 ? '+' : ''}{devPct.toFixed(2)}%
                    </div>
                  );
                })()}
                {/* MA5乖离率浮动标签 */}
                {(() => {
                  const ma5DevPct = isCrosshairActive
                    ? crosshairMa5DevPct
                    : (intradayData?.indicator_sub_charts
                        ?.find((sc: any) => sc.id === 'ma5_deviation')
                        ?.lines?.find((l: any) => l.name === 'ma5_dev_pct')
                        ?.data?.slice(-1)[0]?.value ?? null);
                  if (ma5DevPct == null) return null;
                  const ma5DevAbs = Math.abs(ma5DevPct);
                  const ma5textColor =
                    ma5DevAbs > 7.5 ? '#FF4444' : ma5DevAbs >= 5 ? '#FFAA00' : '#44FF44';
                  return (
                    <div
                      style={{
                        position: 'absolute',
                        top: 8,
                        left: 160,
                        zIndex: 10,
                        fontSize: 12,
                        fontFamily: 'monospace',
                        fontWeight: 600,
                        color: ma5textColor,
                        backgroundColor: 'rgba(26,26,46,0.85)',
                        padding: '2px 8px',
                        borderRadius: 4,
                        pointerEvents: 'none',
                      }}
                    >
                      MA5乖离 {ma5DevPct >= 0 ? '+' : ''}{ma5DevPct.toFixed(2)}%
                    </div>
                  );
                })()}
              </div>
            </Card>

            {/* 成交量子图 */}
            <Card variant="default" padding="none" className="mb-2">
              <div ref={volumeContainerRef} style={{ width: '100%', height: 100 }} />
            </Card>

            {/* 四大指标子图 */}
            {(intradayData?.indicator_sub_charts?.length ?? 0) > 0 && (
              <div className="space-y-2 mb-4">
                {intradayData?.indicator_sub_charts?.filter((sc) => sc.id !== 'price_ma' && sc.id !== 'avg_price_deviation' && sc.id !== 'ma5_deviation').map((sc) => {
                  const containerRefMap: Record<string, React.RefObject<HTMLDivElement | null>> = {
                    absorption: absorptionContainerRef,
                    macd: macdContainerRef,
                    rsi: rsiContainerRef,
                    kdj: kdjContainerRef,
                    mfi: mfiContainerRef,
                  };
                  const ref = containerRefMap[sc.id];

                  return (
                    <Card key={sc.id} variant="default" padding="none">
                      <div className="flex items-center justify-between px-3 pt-2 pb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-muted">{sc.label}</span>
                          {(() => {
                            const displaySignal = isCrosshairActive
                              ? (crosshairSignals[sc.id] || '')
                              : sc.signal_text;
                            return displaySignal ? (() => {
                              const sc = signalColor(displaySignal);
                              return (
                                <span
                                  className="text-[11px] font-semibold px-1.5 py-px rounded"
                                  style={{ color: sc.fg, backgroundColor: sc.bg }}
                                >
                                  {displaySignal}
                                </span>
                              );
                            })() : null;
                          })()}
                          {sc.id === 'rsi' && (() => {
                            const rsiOb = intradayData?.rsi_overbought ?? 65;
                            const rsiOs = intradayData?.rsi_oversold ?? 20;
                            const displayRsi = isCrosshairActive && crosshairRsiValue !== null
                              ? crosshairRsiValue
                              : (() => {
                                  const rsiLine = sc.lines.find((l: any) => l.name === 'RSI');
                                  const rsiData = rsiLine?.data || [];
                                  return rsiData.length > 0 ? rsiData[rsiData.length - 1].value : null;
                                })();
                            if (displayRsi == null) return null;
                            return (
                              <span
                                className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                style={{
                                  color: displayRsi >= rsiOb ? '#FF4444' : displayRsi <= rsiOs ? '#44FF44' : '#888888',
                                  backgroundColor: displayRsi >= rsiOb ? 'rgba(255,68,68,0.1)'
                                    : displayRsi <= rsiOs ? 'rgba(68,255,68,0.1)' : 'rgba(136,136,136,0.1)',
                                }}
                              >
                                RSI {displayRsi.toFixed(1)}
                              </span>
                            );
                          })()}
                          {sc.id === 'macd' && (() => {
                            const displaySum = isCrosshairActive && crosshairMacdSum !== null ? crosshairMacdSum : macdBarSum;
                            return (
                              <span
                                className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                style={{
                                  color: displaySum >= 0 ? '#FF4444' : '#44FF44',
                                  backgroundColor: displaySum >= 0 ? 'rgba(255,68,68,0.1)' : 'rgba(68,255,68,0.1)',
                                }}
                              >
                                柱高和 {displaySum >= 0 ? '+' : ''}{displaySum.toFixed(2)}
                              </span>
                            );
                          })()}
                          {sc.id === 'macd' && (() => {
                            const displayDiff = isCrosshairActive && crosshairMacdDiff !== null ? crosshairMacdDiff : macdBarDiff;
                            return (
                              <span
                                className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                style={{
                                  color: displayDiff >= 0 ? '#FF4444' : '#44FF44',
                                  backgroundColor: displayDiff >= 0 ? 'rgba(255,68,68,0.1)' : 'rgba(68,255,68,0.1)',
                                }}
                              >
                                柱高差 {displayDiff >= 0 ? '+' : ''}{displayDiff.toFixed(2)}
                              </span>
                            );
                          })()}
                          {sc.id === 'kdj' && (() => {
                            const displayK = isCrosshairActive && crosshairKdjKValue !== null
                              ? crosshairKdjKValue
                              : (() => {
                                  const kLine = sc.lines.find((l: any) => l.name === 'K');
                                  const kData = kLine?.data || [];
                                  return kData.length > 0 ? kData[kData.length - 1].value : null;
                                })();
                            const displayD = isCrosshairActive && crosshairKdjDValue !== null
                              ? crosshairKdjDValue
                              : (() => {
                                  const dLine = sc.lines.find((l: any) => l.name === 'D');
                                  const dData = dLine?.data || [];
                                  return dData.length > 0 ? dData[dData.length - 1].value : null;
                                })();
                            const displayJ = isCrosshairActive && crosshairKdjJValue !== null
                              ? crosshairKdjJValue
                              : (() => {
                                  const jLine = sc.lines.find((l: any) => l.name === 'J');
                                  const jData = jLine?.data || [];
                                  return jData.length > 0 ? jData[jData.length - 1].value : null;
                                })();
                            if (displayK == null) return null;
                            return (
                              <span
                                className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                style={{
                                  color: '#FFFF00',
                                  backgroundColor: 'rgba(255,255,0,0.1)',
                                }}
                              >
                                K {displayK.toFixed(2)} D {displayD?.toFixed(2)} J {displayJ?.toFixed(2)}
                              </span>
                            );
                          })()}
                        </div>
                        <div className="flex items-center gap-2">
                          {sc.lines.map((line) => (
                            <div key={line.name} className="flex items-center gap-1">
                              <span className="w-2.5 h-0.5 rounded" style={{ backgroundColor: line.color }} />
                              <span className="text-[10px] text-muted/60">{line.label}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div ref={ref} style={{ width: '100%', height: 115 }} />
                    </Card>
                  );
                })}
              </div>
            )}

            {/* 空状态 */}
          {!intradayData && (
            <div className="flex flex-col items-center justify-center h-full text-center" style={{ minHeight: '500px' }}>
              <div className="w-12 h-12 mb-3 rounded-xl bg-elevated flex items-center justify-center">
                <svg className="w-6 h-6 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <h3 className="text-base font-medium text-white mb-1.5">查看分时走势</h3>
              <p className="text-sm text-muted max-w-md">
                输入股票代码和日期，查看分时K线走势、做T信号和支撑/压力参考线
              </p>
            </div>
          )}
        </div>
      </div>
      </section>
    </div>
  );
};

export default IntradayPage;
