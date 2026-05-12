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
  type IntradayDataResponse,
  type IntradayKlinePoint,
  type SearchHistoryItem,
} from '../api/intraday';
import type { WeightContribution, IntradaySignal } from '../api/intraday';
import { validateStockCode } from '../utils/validation';
import { Card } from '../components/common';

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
  const buy = /买入|买回|流入|金叉|控盘中|反弹/;
  const sell = /卖出|流出|死叉|弱控盘|未控盘|出货|破位/;
  if (buy.test(text)) return { fg: '#FF6644', bg: 'rgba(255,100,68,0.12)' };
  if (sell.test(text)) return { fg: '#44DD44', bg: 'rgba(68,221,68,0.12)' };
  return { fg: '#00D4FF', bg: 'rgba(0,212,255,0.10)' };
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
 * 解析 akshare 返回的时间字符串为 UTC 毫秒
 */
function parseTimestamp(tsStr: string, dateStr: string): number {
  if (!tsStr) return 0;
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
      return d.getTime();
    }

    // fallback: 用当天的日期
    const today = new Date();
    const d = new Date(today.getFullYear(), today.getMonth(), today.getDate(), hour, min, sec);
    return d.getTime();
  }

  // 数字类时间戳
  const num = Number(s);
  if (!Number.isNaN(num)) {
    // 秒级时间戳
    if (num > 1e9) return num * 1000;
    // 毫秒级
    return num;
  }

  return 0;
}

const IntradayPage: React.FC = () => {
  // ── 状态 ──
  const [stockCode, setStockCode] = useState('');
  const [inputError, setInputError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [intradayData, setIntradayData] = useState<IntradayDataResponse | null>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [priceRangeEnabled, setPriceRangeEnabled] = useState(true);
  const todayDateStr = useMemo(() => formatDate(new Date()), []);
  const [crosshairSignals, setCrosshairSignals] = useState<Record<string, string>>({});
  const [isCrosshairActive, setIsCrosshairActive] = useState(false);
  const [hoveredWeightDetails, setHoveredWeightDetails] = useState<{
    buy: WeightContribution[];
    sell: WeightContribution[];
    supportForce: number;
    pressureForce: number;
    signalType: string;
  } | null>(null);

  // 存储从API返回的所有信号
  const allSignalsRef = useRef<IntradaySignal[]>([]);
  const filteredSignalsRef = useRef<IntradaySignal[]>([]);

  // 信号筛选状态
  const [activeFilters, setActiveFilters] = useState<Set<string>>(new Set());

  // ── 图表引用 ──
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const volumeContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const volumeChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candleSeriesRef = useRef<lightweightCharts.ISeriesApi<'Candlestick'> | lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<lightweightCharts.ISeriesApi<'Histogram'> | null>(null);
  const refLineSeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'>[]>([]);
  const refLinePriceLinesRef = useRef<lightweightCharts.IPriceLine[]>([]);
  const chipAreaRef = useRef<lightweightCharts.ISeriesApi<'Area'> | null>(null);
  const avgPriceLineRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const priceRangeSeriesRef = useRef<lightweightCharts.ISeriesApi<'Line'> | null>(null);
  const seriesMarkersRef = useRef<any>(null);
  const timeSyncSubRef = useRef<(() => void) | null>(null);
  const isTimeSyncingRef = useRef(false);
  const currentDateRef = useRef(todayDateStr);
  const priceRangeEnabledRef = useRef(true);
  const isCrosshairUpdatingRef = useRef(false);
  const currentCrosshairTimeRef = useRef<Time | null>(null);
  const klineRawDataRef = useRef<any[]>([]);
  const crosshairSignalRef = useRef<Record<string, string>>({});
  const crosshairSubsRef = useRef<Array<any>>([]);

  // 指标子图容器 refs
  const absorptionContainerRef = useRef<HTMLDivElement>(null);
  const mainInOutContainerRef = useRef<HTMLDivElement>(null);
  const cywContainerRef = useRef<HTMLDivElement>(null);
  const macdContainerRef = useRef<HTMLDivElement>(null);
  const rsiContainerRef = useRef<HTMLDivElement>(null);

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
      },
      handleScroll: {},
      handleScale: {},
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
      },
      handleScroll: {},
      handleScale: {},
    });

    const volSeries = volChart.addSeries(lightweightCharts.HistogramSeries, {
      color: '#66666688',
      priceFormat: { type: 'volume', precision: 0, minMove: 1 },
    });

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
      // 清理指标子图
      indicatorChartsRef.current.forEach((c) => {
        try { c.remove(); } catch (e) { /* ignore */ }
      });
      indicatorChartsRef.current.clear();
      indicatorSeriesRef.current.clear();
    };
  }, []);

  // ── 同步辅助函数 ──
  const syncCrosshairToSubCharts = (time: Time) => {
    indicatorChartsRef.current.forEach((subChart, id) => {
      try {
        const seriesArr = indicatorSeriesRef.current.get(id);
        if (seriesArr && seriesArr.length > 0) {
          const firstSeries = seriesArr[0];
          const data = (firstSeries as any).data?.() || [];
          const dataPt = data.find((d: any) => d.time === time);
          if (dataPt) {
            const value = dataPt.value !== undefined ? dataPt.value :
              (dataPt.close !== undefined ? dataPt.close : 0);
            subChart.setCrosshairPosition(value, time, firstSeries);
          }
        }
      } catch (e) { /* ignore */ }
    });
  };

  const syncTimeRangeToSubCharts = (range: { from: Time; to: Time }) => {
    indicatorChartsRef.current.forEach((subChart) => {
      try { subChart.timeScale().setVisibleRange(range); } catch (e) { /* ignore */ }
    });
  };

  // ── 前端信号计算（无反函数，基于已加载的完整指标线数据切片） ──

  // 存储每根K线的指标值快照，在 renderData 时填充
  const snapshotRef = useRef<Array<{
    time: number;
    dominant_power: number;
    main_in: number;
    main_out: number;
    CYW: number;
    CYW_MA: number;
    absorption: number;
    close: number;
    ma5: number;
    ma20: number;
    deviation_pct: number;
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
    RSI: number;
    rsi_oversold: boolean;
    rsi_overbought: boolean;
  }>>([]);

  const _crossUpTs = (aArr: number[], bArr: number[], lookback: number = 3): boolean => {
    if (aArr.length < 2 || bArr.length < 2) return false;
    const start = Math.max(0, aArr.length - lookback);
    for (let i = start; i < aArr.length; i++) {
      if (i === 0) continue;
      if (aArr[i - 1] <= bArr[i - 1] && aArr[i] > bArr[i]) return true;
    }
    return false;
  };

  const _crossDownTs = (aArr: number[], bArr: number[], lookback: number = 3): boolean => {
    if (aArr.length < 2 || bArr.length < 2) return false;
    const start = Math.max(0, aArr.length - lookback);
    for (let i = start; i < aArr.length; i++) {
      if (i === 0) continue;
      if (aArr[i - 1] >= bArr[i - 1] && aArr[i] < bArr[i]) return true;
    }
    return false;
  };

  const computeSignalsAtTime = (time: Time) => {
    const snapshots = snapshotRef.current;
    const idx = snapshots.findIndex((s) => s.time === time);
    if (idx < 0) return;

    const slice = snapshots.slice(0, idx + 1);
    const signals: Record<string, string> = {};

    // 主力进出
    const mi = slice.map((s) => s.main_in);
    const mo = slice.map((s) => s.main_out);
    const lastMi = mi[mi.length - 1];
    const lastMo = mo[mo.length - 1];
    if (!isNaN(lastMi) && !isNaN(lastMo)) {
      const up = _crossUpTs(mi, mo);
      const down = _crossDownTs(mi, mo);
      if (up) {
        const recentMin = Math.min(...mi.slice(-5));
        signals.main_in_out = recentMin < 30 ? '反T买回 ↑' : '正T买入 ↑';
      } else if (down) {
        const recentMax = Math.max(...mi.slice(-5));
        signals.main_in_out = recentMax > 70 ? '反T卖出 ↓' : '正T卖出 ↓';
      } else if (lastMi > lastMo) {
        signals.main_in_out = '主力流入 ↗';
      } else {
        signals.main_in_out = '主力流出 ↘';
      }
    }

    // CYW
    const cywArr = slice.map((s) => s.CYW);
    const cywMaArr = slice.map((s) => s.CYW_MA);
    const lastCyw = cywArr[cywArr.length - 1];
    if (!isNaN(lastCyw)) {
      const up = _crossUpTs(cywArr, cywMaArr);
      const down = _crossDownTs(cywArr, cywMaArr);
      if (up) {
        signals.cyw = (lastCyw > 0 ? '控盘中' : '弱控盘') + ' 买入 ↑';
      } else if (down) {
        signals.cyw = '未控盘 卖出 ↓';
      } else if (lastCyw > 0) {
        signals.cyw = '控盘中 ↗';
      } else {
        signals.cyw = '未控盘 ↘';
      }
    }

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

    // RSI
    const rsiArr = slice.map((s) => s.RSI);
    const lastRsi = rsiArr[rsiArr.length - 1];
    if (!isNaN(lastRsi)) {
      if (lastRsi <= 30) {
        signals.rsi = 'RSI超卖 \u2191';
      } else if (lastRsi >= 70) {
        signals.rsi = 'RSI超买 \u2193';
      } else if (lastRsi < 50) {
        signals.rsi = 'RSI偏弱 \u2198';
      } else {
        signals.rsi = 'RSI偏强 \u2197';
      }
    }

    // 主力吸筹不输出信号
    signals.absorption = '';

    crosshairSignalRef.current = signals;
    setCrosshairSignals({ ...signals });
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
      const mainInArr = slice.map((s) => s.main_in);
      const mainOutArr = slice.map((s) => s.main_out);
      const cywVals = slice.map((s) => s.CYW);
      const cywMaVals = slice.map((s) => s.CYW_MA);

      const mainInCrossUp = _crossUpTs(mainInArr, mainOutArr);
      const mainInCrossDown = _crossDownTs(mainInArr, mainOutArr);
      const cywCrossUp = _crossUpTs(cywVals, cywMaVals);
      const cywCrossDown = _crossDownTs(cywVals, cywMaVals);

      const buyWeights: WeightContribution[] = [];
      const sellWeights: WeightContribution[] = [];

      // 买入权重
      if (!absActive) {
        buyWeights.push({ key: 'absorption_required', label: '主力吸筹(必备条件)', weight: 0, triggered: false, score: 0 });
      } else {
        const factors: [string, string, number, boolean][] = [
          ['absorption_active', '主力吸筹活跃', 0, absActive],
          ['cyw_cross_ma_up', 'CYW上穿MA', 1, cywCrossUp],
          ['main_in_signal', '主力进出金叉', 1, mainInCrossUp],
          ['price_cross_ma5_up', '价格上穿MA5', 1, sn.price_cross_ma5_up],
          ['avg_price_oversold_fix', '均价超卖修复', 2, sn.deviation_oversold && sn.deviation_narrowing],
          ['price_above_ma20', '价格>MA20趋势', 1, sn.price_above_ma20],
          ['volume_surge', '量能放大', 1, false],
          ['macd_golden_cross', 'MACD金叉', 2, macdGoldenCross],
          ['rsi_oversold', 'RSI超卖', 5, rsiOversold],
        ];
        let buyScore = 0;
        for (const [key, label, w, trig] of factors) {
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
          ['distribution_active', '主力出货活跃', 0, distActive],
          ['main_out_signal', '主力进出死叉', 0, mainInCrossDown],
          ['cyw_cross_ma_down', 'CYW下穿MA', 0, cywCrossDown],
          ['volume_stagnation', '放量滞涨', 3, false],
          ['price_cross_ma5_down', '价格下穿MA5', 2, sn.price_cross_ma5_down],
          ['avg_price_overbought_fix', '均价超买回落', 2, sn.deviation_overbought && sn.deviation_peaking],
          ['macd_death_cross', 'MACD死叉', 2, macdDeathCross],
          ['rsi_overbought', 'RSI超买', 5, rsiOverbought],
        ];
        let sellScore = 0;
        for (const [key, label, w, trig] of factors) {
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

  // ── 渲染数据 ──
  const renderData = useCallback(
    (data: IntradayDataResponse, date: string) => {
      const container = chartContainerRef.current;
      const volContainer = volumeContainerRef.current;
      if (!container || !volContainer) return;

      // ── 销毁旧图表以完全重置内部状态（包括用户手动调整的 scale）──
      if (chartRef.current) {
        if (timeSyncSubRef.current) timeSyncSubRef.current();
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
      crosshairSubsRef.current.forEach((unsub) => {
        try { unsub(); } catch (e) { /* ignore */ }
      });
      crosshairSubsRef.current = [];

      // ── 清空所有容器DOM，确保完全干净（防御性措施，解决chart.remove()可能残留canvas的问题）──
      try { container.innerHTML = ''; } catch (e) { /* ignore */ }
      try { volContainer.innerHTML = ''; } catch (e) { /* ignore */ }
      [absorptionContainerRef, mainInOutContainerRef, cywContainerRef, macdContainerRef, rsiContainerRef].forEach((ref) => {
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
        },
        handleScroll: {},
        handleScale: {},
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
        },
        handleScroll: {},
        handleScale: {},
      });
      const volSeries = volChart.addSeries(lightweightCharts.HistogramSeries, {
        color: '#66666688',
        priceFormat: { type: 'volume', precision: 0, minMove: 1 },
      });
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

      const klines = convertKlineData(data.kline_data, date);
      klineRawDataRef.current = klines;
      allSignalsRef.current = data.signals || [];

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

      // ── 筹码密集区色带（最先添加到底层，避免遮挡分时白线）──
      if (data.reference_lines && data.reference_lines.length > 0) {
        const chipUpper = data.reference_lines.find((rl) => rl.id === 'chip_upper');
        const chipLower = data.reference_lines.find((rl) => rl.id === 'chip_lower');
        if (chipUpper && chipLower && klines.length > 0) {
          const areaSeries = chart.addSeries(lightweightCharts.AreaSeries, {
            lineWidth: 1,
            topColor: 'rgba(187, 68, 255, 0.08)',
            bottomColor: 'rgba(187, 68, 255, 0.02)',
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          const areaData = klines.map((k) => ({
            time: k.time as any,
            value: chipUpper.price,
          }));
          areaSeries.setData(areaData);
          chipAreaRef.current = areaSeries;
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
          const ls: 0 | 1 | 2 | 3 | 4 = rl.style === 'dashed' ? 2 : (rl.style === 'dotted' ? 1 : 0);
          const priceLine = refLayer.createPriceLine({
            price: rl.price,
            color: rl.color,
            lineWidth: 1,
            lineStyle: ls,
            axisLabelVisible: true,
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
          }
        }
      }

      // ── 构建快照数据：融合K线和指标子图 ──
      const buildSnapshot = () => {
        const map = new Map<number, {
          dominant_power: number; main_in: number; main_out: number;
          CYW: number; CYW_MA: number; absorption: number;
          close: number; ma5: number; ma20: number; deviation_pct: number;
          DIF: number; DEA: number; MACD_Bar: number; RSI: number;
        }>();
        (data.indicator_sub_charts || []).forEach((sc) => {
          (sc.lines || []).forEach((line) => {
            (line.data || []).forEach((pt: any) => {
              const ms = parseTimestamp(pt.time, date);
              const t = Math.floor(ms / 1000);
              if (t <= 0) return;
              if (!map.has(t)) map.set(t, {
                dominant_power: NaN, main_in: NaN, main_out: NaN,
                CYW: NaN, CYW_MA: NaN, absorption: NaN,
                close: NaN, ma5: NaN, ma20: NaN, deviation_pct: NaN,
                DIF: NaN, DEA: NaN, MACD_Bar: NaN, RSI: NaN,
              });
              const entry = map.get(t)!;
              if (line.name === 'dominant_power') entry.dominant_power = pt.value;
              else if (line.name === 'main_in') entry.main_in = pt.value;
              else if (line.name === 'main_out') entry.main_out = pt.value;
              else if (line.name === 'CYW') entry.CYW = pt.value;
              else if (line.name === 'CYW_MA') entry.CYW_MA = pt.value;
              else if (line.name === 'absorption') entry.absorption = pt.value;
              else if (line.name === 'close') entry.close = pt.value;
              else if (line.name === 'ma5') entry.ma5 = pt.value;
              else if (line.name === 'ma20') entry.ma20 = pt.value;
              else if (line.name === 'deviation_pct') entry.deviation_pct = pt.value;
              else if (line.name === 'DIF') entry.DIF = pt.value;
              else if (line.name === 'DEA') entry.DEA = pt.value;
              else if (line.name === 'MACD_Bar') entry.MACD_Bar = pt.value;
              else if (line.name === 'RSI') entry.RSI = pt.value;
            });
          });
        });
        const OVERSOLD = -2.5;
        const OVERBOUGHT = 2.5;
        const RSI_OVERSOLD = 20;
        const RSI_OVERBOUGHT = 65;
        const raw = Array.from(map.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([time, v]) => ({ time, ...v }));
        snapshotRef.current = raw.map((cur, i) => {
          const prev = i > 0 ? raw[i - 1] : null;
          const absorption_val = isNaN(cur.absorption) ? 0 : cur.absorption;
          const dev = isNaN(cur.deviation_pct) ? 0 : cur.deviation_pct;
          const prevDev = prev && !isNaN(prev.deviation_pct) ? prev.deviation_pct : 0;
          const curDif = isNaN(cur.DIF) ? 0 : cur.DIF;
          const curDea = isNaN(cur.DEA) ? 0 : cur.DEA;
          const prevDif = prev && !isNaN(prev.DIF) ? prev.DIF : 0;
          const prevDea = prev && !isNaN(prev.DEA) ? prev.DEA : 0;
          const curRsi = isNaN(cur.RSI) ? 50 : cur.RSI;
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
            rsi_oversold: curRsi <= RSI_OVERSOLD,
            rsi_overbought: curRsi >= RSI_OVERBOUGHT,
          };
        });
      };
      buildSnapshot();

      // ── 主图十字线联动 ──
      const mainHandleCrosshairMove = (param: any) => {
        if (isCrosshairUpdatingRef.current) {
          if (param.time) computeSignalsAtTime(param.time);
          return;
        }
        isCrosshairUpdatingRef.current = true;
        try {
          currentCrosshairTimeRef.current = param.time;
          if (param.time) {
            // 同步到成交量图
            if (volChart && volSeries) {
              const kp = klines.find((k: any) => k.time === param.time);
              if (kp) {
                try { volChart.setCrosshairPosition(kp.volume, param.time, volSeries); } catch (e) { /* ignore */ }
              }
            }
            // 同步到所有子图
            syncCrosshairToSubCharts(param.time);
            // 更新信号文本
            computeSignalsAtTime(param.time);
          } else {
            // 离开图表，恢复最新信号
            setIsCrosshairActive(false);
          }
        } finally {
          setTimeout(() => { isCrosshairUpdatingRef.current = false; }, 0);
        }
      };
      crosshairSubsRef.current.push(chart.subscribeCrosshairMove(mainHandleCrosshairMove));

      const mainHandleTimeScaleChange = () => {
        if (isTimeSyncingRef.current) return;
        isTimeSyncingRef.current = true;
        try {
          const range = chart.timeScale().getVisibleRange();
          if (range && range.from && range.to) {
            if (volChart) {
              try { volChart.timeScale().setVisibleRange(range); } catch (e) { /* ignore */ }
            }
            syncTimeRangeToSubCharts(range);
          }
        } finally {
          setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
        }
      };
      chart.timeScale().subscribeVisibleTimeRangeChange(mainHandleTimeScaleChange);

      // ── 成交量图十字线联动 ──
      if (volChart && volSeries) {
        const volHandleCrosshairMove = (param: any) => {
          if (isCrosshairUpdatingRef.current) {
            if (param.time) computeSignalsAtTime(param.time);
            return;
          }
          isCrosshairUpdatingRef.current = true;
          try {
            currentCrosshairTimeRef.current = param.time;
            if (param.time) {
              const kp = klines.find((k: any) => k.time === param.time);
              if (kp) {
                const cs = candleSeriesRef.current;
                try {
                  if (cs) chart.setCrosshairPosition(kp.close, param.time, cs);
                } catch (e) { /* ignore */ }
              }
              syncCrosshairToSubCharts(param.time);
              computeSignalsAtTime(param.time);
            } else {
              setIsCrosshairActive(false);
            }
          } finally {
            setTimeout(() => { isCrosshairUpdatingRef.current = false; }, 0);
          }
        };
        crosshairSubsRef.current.push(volChart.subscribeCrosshairMove(volHandleCrosshairMove));

        const volHandleTimeScaleChange = () => {
          if (isTimeSyncingRef.current) return;
          isTimeSyncingRef.current = true;
          try {
            const vr = volChart.timeScale().getVisibleRange();
            if (vr && vr.from && vr.to) {
              chart.timeScale().setVisibleRange(vr);
              syncTimeRangeToSubCharts(vr);
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
        { id: 'main_in_out', ref: mainInOutContainerRef },
        { id: 'cyw', ref: cywContainerRef },
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
          } else {
            const ls = subChart.addSeries(lightweightCharts.LineSeries, {
              color: line.color,
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
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

            ls.setData(points);
            lineSeriesList.push(ls);
          }
        }

        // 同步主图时间轴
        const range = chart.timeScale().getVisibleRange();
        if (range && range.from && range.to) {
          subChart.timeScale().setVisibleRange(range);
        }

        // 子图十字线联动
        const subHandleCrosshairMove = (param: any) => {
          if (isCrosshairUpdatingRef.current) {
            if (param.time) computeSignalsAtTime(param.time);
            return;
          }
          isCrosshairUpdatingRef.current = true;
          try {
            currentCrosshairTimeRef.current = param.time;
            if (param.time) {
              // 同步到主图
              const kp = klines.find((k: any) => k.time === param.time);
              if (kp) {
                try {
                  const cs = candleSeriesRef.current;
                  const isValidCandleSeries = cs && (cs as any).seriesType?.() === 'Candlestick';
                  if (isValidCandleSeries && cs) {
                    chart.setCrosshairPosition(kp.close, param.time, cs);
                  } else if (cs) {
                    chart.setCrosshairPosition(kp.close, param.time, cs);
                  }
                } catch (e) { /* ignore */ }
              }
              // 同步到成交量图
              if (volChart && volSeries) {
                try { volChart.setCrosshairPosition(kp?.volume || 0, param.time, volSeries); } catch (e) { /* ignore */ }
              }
              // 同步到其他子图
              indicatorChartsRef.current.forEach((otherChart, otherId) => {
                if (otherId === sc.id) return;
                try {
                  const otherSeriesArr = indicatorSeriesRef.current.get(otherId);
                  if (otherSeriesArr && otherSeriesArr.length > 0) {
                    const otherSeries = otherSeriesArr[0];
                    const otherData = (otherSeries as any).data?.() || [];
                    const otherPt = otherData.find((d: any) => d.time === param.time);
                    if (otherPt) {
                      const val = otherPt.value !== undefined ? otherPt.value
                        : (otherPt.close !== undefined ? otherPt.close : 0);
                      otherChart.setCrosshairPosition(val, param.time, otherSeries);
                    }
                  }
                } catch (e) { /* ignore */ }
              });
              // 更新信号文本
              computeSignalsAtTime(param.time);
            } else {
              // 离开图表，恢复最新信号
              setIsCrosshairActive(false);
            }
          } finally {
            setTimeout(() => { isCrosshairUpdatingRef.current = false; }, 0);
          }
        };
        crosshairSubsRef.current.push(subChart.subscribeCrosshairMove(subHandleCrosshairMove));

        const subHandleTimeScaleChange = () => {
          if (isTimeSyncingRef.current) return;
          isTimeSyncingRef.current = true;
          try {
            const sr = subChart.timeScale().getVisibleRange();
            if (sr && sr.from && sr.to) {
              chart.timeScale().setVisibleRange(sr);
              if (volChart) {
                try { volChart.timeScale().setVisibleRange(sr); } catch (e) { /* ignore */ }
              }
              indicatorChartsRef.current.forEach((otherChart, otherId) => {
                if (otherId === sc.id) return;
                try { otherChart.timeScale().setVisibleRange(sr); } catch (e) { /* ignore */ }
              });
            }
          } finally {
            setTimeout(() => { isTimeSyncingRef.current = false; }, 0);
          }
        };
        subChart.timeScale().subscribeVisibleTimeRangeChange(subHandleTimeScaleChange);

        indicatorChartsRef.current.set(sc.id, subChart);
        indicatorSeriesRef.current.set(sc.id, lineSeriesList);
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
    },
    [],
  );

  // 当数据变更时重新渲染
  useEffect(() => {
    if (intradayData) {
      renderData(intradayData, intradayData.date || todayDateStr);
    }
  }, [intradayData, renderData, priceRangeEnabled]);

  // ── 搜索 ──
  const handleSearch = useCallback(async () => {
    const v = validateStockCode(stockCode);
    setInputError(v.valid ? undefined : v.message);
    if (!v.valid) return;

    setIsLoading(true);
    try {
      const dateParam = todayDateStr.replace(/-/g, '');
      const data = await getIntradayData(stockCode, dateParam);
      setIntradayData(data);
      setInputError(undefined);

      // 保存搜索历史
      try {
        await saveSearchHistory(stockCode, data.stock_name || '', todayDateStr);
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
  }, [stockCode, loadHistory]);

  // 回车搜索
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !isLoading && stockCode) {
        handleSearch();
      }
    },
    [handleSearch, isLoading, stockCode],
  );

  // 点击历史记录
  const handleHistoryClick = useCallback(
    async (item: SearchHistoryItem) => {
      setStockCode(item.stock_code);
      setIsLoading(true);
      try {
        const dateParam = todayDateStr.replace(/-/g, '');
        const data = await getIntradayData(item.stock_code, dateParam);
        setIntradayData(data);
        setInputError(undefined);
      } catch (err: any) {
        console.error('获取分时数据失败:', err);
        setInputError(err.message || '获取数据失败');
        setIntradayData(null);
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  // 删除历史
  const handleDeleteHistory = useCallback(
    async (id: number) => {
      await deleteSearchHistory(id);
      await loadHistory();
    },
    [loadHistory],
  );

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
        {isLoadingHistory && (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
          </div>
        )}
        {!isLoadingHistory && searchHistory.length === 0 && (
          <p className="text-xs text-muted text-center py-4">暂无搜索历史</p>
        )}
        {searchHistory.map((item) => (
          <div
            key={item.id}
            className="flex items-center gap-1 py-2 border-b border-white/5 last:border-b-0 cursor-pointer hover:bg-white/5 rounded px-1"
          >
            <button
              type="button"
              className="flex-1 text-left"
              onClick={() => handleHistoryClick(item)}
            >
              <div className="flex items-center justify-between gap-1.5">
                <span className="font-medium text-white truncate text-xs">
                  {item.stock_code}
                </span>
                {item.date && (
                  <span className="flex-shrink-0 text-cyan text-xs font-medium">
                    {item.date}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-xs text-muted font-mono">
                  {item.stock_name || '-'}
                </span>
              </div>
            </button>
            <button
              type="button"
              className="p-1 text-muted hover:text-danger transition-colors flex-shrink-0"
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteHistory(item.id);
              }}
              title="删除"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ))}
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
            <input
              type="text"
              value={stockCode}
              onChange={(e) => {
                setStockCode(e.target.value.toUpperCase());
                setInputError(undefined);
              }}
              onKeyDown={handleKeyDown}
              placeholder="输入股票代码，如 600519"
              disabled={isLoading}
              className={`input-terminal w-full ${inputError ? 'border-danger/50' : ''}`}
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
            onClick={handleSearch}
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
        </div>
      </header>

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
          )}

          <div className="w-full">
            <div className="flex items-start gap-2 mb-2" style={{ height: 165 }}>
              {signalStats.length > 0 && (
                <Card variant="default" padding="sm" className="w-56 flex-shrink-0">
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
                  <Card variant="default" padding="sm" className="w-56 flex-shrink-0" style={{ height: 165 }}>
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

                  <Card variant="default" padding="sm" className="w-56 flex-shrink-0" style={{ height: 165 }}>
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

              {intradayData && filteredSignals.length > 0 && (
                <div className="flex-1 overflow-y-auto" style={{ height: 165 }}>
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
                        try {
                          const cs = candleSeriesRef.current;
                          if (cs && (cs as any).seriesType?.() !== 'Histogram') {
                            ch.setCrosshairPosition(sig.price, targetSec as any, cs as any);
                          } else {
                            ch.setCrosshairPosition(sig.price, targetSec as any, candleSeriesRef.current!);
                          }
                        } catch (e) { /* ignore */ }
                        currentCrosshairTimeRef.current = targetSec as any;
                        syncCrosshairToSubCharts(targetSec as any);
                        if (volumeChartRef.current && volumeSeriesRef.current && kp) {
                          try {
                            volumeChartRef.current.setCrosshairPosition(
                              (kp as any).volume || 0,
                              targetSec as any,
                              volumeSeriesRef.current,
                            );
                          } catch (e) { /* ignore */ }
                        }
                        computeSignalsAtTime(targetSec as any);
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
              <div ref={chartContainerRef} style={{ width: '100%', height: CHART_HEIGHT }} />
            </Card>

            {/* 成交量子图 */}
            <Card variant="default" padding="none" className="mb-2">
              <div ref={volumeContainerRef} style={{ width: '100%', height: 100 }} />
            </Card>

            {/* 四大指标子图 */}
            {(intradayData?.indicator_sub_charts?.length ?? 0) > 0 && (
              <div className="space-y-2 mb-4">
                {intradayData?.indicator_sub_charts?.filter((sc) => sc.id !== 'price_ma' && sc.id !== 'avg_price_deviation').map((sc) => {
                  const containerRefMap: Record<string, React.RefObject<HTMLDivElement | null>> = {
                    absorption: absorptionContainerRef,
                    macd: macdContainerRef,
                    rsi: rsiContainerRef,
                    main_in_out: mainInOutContainerRef,
                    cyw: cywContainerRef,
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
