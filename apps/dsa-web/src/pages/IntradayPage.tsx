import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
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
  const [selectedDate, setSelectedDate] = useState(formatDate(new Date()));
  const [crosshairSignals, setCrosshairSignals] = useState<Record<string, string>>({});
  const [isCrosshairActive, setIsCrosshairActive] = useState(false);

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
  const seriesMarkersRef = useRef<any>(null);
  const timeSyncSubRef = useRef<(() => void) | null>(null);
  const isTimeSyncingRef = useRef(false);
  const currentDateRef = useRef(selectedDate);
  const isCrosshairUpdatingRef = useRef(false);
  const currentCrosshairTimeRef = useRef<Time | null>(null);
  const klineRawDataRef = useRef<any[]>([]);
  const crosshairSignalRef = useRef<Record<string, string>>({});

  // 指标子图容器 refs
  const absorptionContainerRef = useRef<HTMLDivElement>(null);
  const mainInOutContainerRef = useRef<HTMLDivElement>(null);
  const dragonTigerContainerRef = useRef<HTMLDivElement>(null);
  const cywContainerRef = useRef<HTMLDivElement>(null);

  // 指标子图实例管理
  const indicatorChartsRef = useRef<Map<string, lightweightCharts.IChartApi>>(new Map());
  const indicatorSeriesRef = useRef<Map<string, any[]>>(new Map());

  currentDateRef.current = selectedDate;

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
      },
      grid: {
        vertLines: { color: '#2b2b43' },
        horzLines: { color: '#2b2b43' },
      },
      rightPriceScale: {
        scaleMargins: { top: 0.05, bottom: 0.05 },
        borderVisible: false,
      },
      crosshair: { mode: 1 },
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
      chart.remove();
      volChart.remove();
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

    // 龙虎动力
    const dp = slice.map((s) => s.dominant_power);
    const lastDp = dp[dp.length - 1];
    if (!isNaN(lastDp)) {
      if (lastDp >= 0.3) signals.dragon_tiger_power = '买入 ↑';
      else if (lastDp >= 0.1) signals.dragon_tiger_power = '持有偏多 ↗';
      else if (lastDp > -0.1) signals.dragon_tiger_power = '观望 —';
      else if (lastDp >= -0.3) signals.dragon_tiger_power = '减仓 ↘';
      else signals.dragon_tiger_power = '卖出 ↓';
    }

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

    // 主力吸筹不输出信号
    signals.absorption = '';

    crosshairSignalRef.current = signals;
    setCrosshairSignals({ ...signals });
    setIsCrosshairActive(true);
  };

  // ── 渲染数据 ──
  const renderData = useCallback(
    (data: IntradayDataResponse, date: string) => {
      const chart = chartRef.current;
      const volChart = volumeChartRef.current;
      const candleSeries = candleSeriesRef.current;
      const volSeries = volumeSeriesRef.current;
      if (!chart || !volChart || !candleSeries || !volSeries) return;

      // 清理旧标记
      if (seriesMarkersRef.current) {
        try {
          seriesMarkersRef.current.setMarkers([]);
          seriesMarkersRef.current.detach();
        } catch (e) { /* ignore */ }
        seriesMarkersRef.current = null;
      }

      const klines = convertKlineData(data.kline_data, date);
      klineRawDataRef.current = klines;

      // 检测腾讯数据：每根K线中 Open===High===Low===Close 的比例
      const flatCount = klines.filter((k) => k.open === k.high && k.high === k.low && k.low === k.close).length;
      const isTencentData = klines.length > 10 && flatCount / klines.length > 0.95;

      // 设置K线：腾讯数据用白色细实线，非腾讯数据用K线
      if (isTencentData) {
        // 重新创建 series: 替换 Candlestick 为 Line
        chart.removeSeries(candleSeries);
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
        // 常规 K 线
        candleSeries.setData(
          klines.map((k) => ({
            time: k.time as any,
            open: k.open,
            high: k.high,
            low: k.low,
            close: k.close,
          })),
        );
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

      // ── 支撑位/压力位参考线 ──
      refLineSeriesRef.current.forEach((s) => {
        try { chart.removeSeries(s); } catch (e) { /* ignore */ }
      });
      refLineSeriesRef.current = [];
      {
        const cs = candleSeriesRef.current;
        if (cs) {
          refLinePriceLinesRef.current.forEach((pl) => {
            try { cs.removePriceLine(pl); } catch (e) { /* ignore */ }
          });
          refLinePriceLinesRef.current = [];
        }
      }
      // 清理筹码区色带
      if (chipAreaRef.current) {
        try { chart.removeSeries(chipAreaRef.current); } catch (e) { /* ignore */ }
        chipAreaRef.current = null;
      }
      if (data.reference_lines && data.reference_lines.length > 0) {
        const cs = candleSeriesRef.current;
        // 筹码密集区色带（两个PriceLine之间的半透明区域）
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
        if (cs) {
          data.reference_lines.forEach((rl) => {
            const ls: 0 | 1 | 2 | 3 | 4 = rl.style === 'dashed' ? 2 : (rl.style === 'dotted' ? 1 : 0);
            const priceLine = cs.createPriceLine({
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
      }

      chart.timeScale().fitContent();

      // ── 构建快照数据：融合K线和指标子图 ──
      const buildSnapshot = () => {
        const map = new Map<number, { dominant_power: number; main_in: number; main_out: number; CYW: number; CYW_MA: number }>();
        (data.indicator_sub_charts || []).forEach((sc) => {
          (sc.lines || []).forEach((line) => {
            (line.data || []).forEach((pt: any) => {
              const ms = parseTimestamp(pt.time, date);
              const t = Math.floor(ms / 1000);
              if (t <= 0) return;
              if (!map.has(t)) map.set(t, { dominant_power: NaN, main_in: NaN, main_out: NaN, CYW: NaN, CYW_MA: NaN });
              const entry = map.get(t)!;
              if (line.name === 'dominant_power') entry.dominant_power = pt.value;
              else if (line.name === 'main_in') entry.main_in = pt.value;
              else if (line.name === 'main_out') entry.main_out = pt.value;
              else if (line.name === 'CYW') entry.CYW = pt.value;
              else if (line.name === 'CYW_MA') entry.CYW_MA = pt.value;
            });
          });
        });
        snapshotRef.current = Array.from(map.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([time, v]) => ({ time, ...v }));
      };
      buildSnapshot();

      // ── 主图十字线联动 ──
      const mainHandleCrosshairMove = (param: any) => {
        if (isCrosshairUpdatingRef.current) return;
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
      chart.subscribeCrosshairMove(mainHandleCrosshairMove);

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
          if (isCrosshairUpdatingRef.current) return;
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
        volChart.subscribeCrosshairMove(volHandleCrosshairMove);

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
        { id: 'main_in_out', ref: mainInOutContainerRef },
        { id: 'dragon_tiger_power', ref: dragonTigerContainerRef },
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
          // 主力吸筹使用柱状图样式（紫色吸筹柱），与可视化标签页保持一致
          if (sc.id === 'absorption') {
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

        // 自适应时间范围
        subChart.timeScale().fitContent();

        // 同步主图时间轴
        const range = chart.timeScale().getVisibleRange();
        if (range && range.from && range.to) {
          subChart.timeScale().setVisibleRange(range);
        }

        // 子图十字线联动
        const subHandleCrosshairMove = (param: any) => {
          if (isCrosshairUpdatingRef.current) return;
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
        subChart.subscribeCrosshairMove(subHandleCrosshairMove);

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

      // ── 信号标记 ──
      const markers: lightweightCharts.SeriesMarker<lightweightCharts.Time>[] = [];
      for (const sig of data.signals) {
        // 将 trigger_time 转为 Unix 秒
        const sigMs = parseTimestamp(sig.trigger_time, date);
        const sigUnix = Math.floor(sigMs / 1000);
        if (sigUnix <= 0) continue;

        if (sig.signal_type === 'buy') {
          markers.push({
            time: sigUnix as any,
            position: 'belowBar',
            color: '#FF2222',
            shape: 'arrowUp',
            text: `买 ${(sig.confidence * 100).toFixed(0)}%`,
            size: 2 + (sig.confidence * 2),
          });
        } else {
          markers.push({
            time: sigUnix as any,
            position: 'aboveBar',
            color: '#22DD44',
            shape: 'arrowDown',
            text: `卖 ${(sig.confidence * 100).toFixed(0)}%`,
            size: 2 + (sig.confidence * 2),
          });
        }
      }

      // 创建新标记
      if (markers.length > 0) {
        seriesMarkersRef.current = createSeriesMarkers(candleSeries, markers as any);
      }
    },
    [],
  );

  // 当数据变更时重新渲染
  useEffect(() => {
    if (intradayData) {
      renderData(intradayData, intradayData.date || selectedDate);
    }
  }, [intradayData, selectedDate, renderData]);

  // ── 搜索 ──
  const handleSearch = useCallback(async () => {
    const v = validateStockCode(stockCode);
    setInputError(v.valid ? undefined : v.message);
    if (!v.valid) return;

    setIsLoading(true);
    try {
      const dateParam = selectedDate.replace(/-/g, '');
      const data = await getIntradayData(stockCode, dateParam);
      setIntradayData(data);
      setInputError(undefined);

      // 保存搜索历史
      try {
        await saveSearchHistory(stockCode, data.stock_name || '', selectedDate);
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
  }, [stockCode, selectedDate, loadHistory]);

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
      if (item.date) setSelectedDate(item.date);
      setIsLoading(true);
      try {
        const dateParam = (item.date || selectedDate).replace(/-/g, '');
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
        { label: '买入信号', value: summary.buy_signals, color: '#FF4444' },
        { label: '卖出信号', value: summary.sell_signals, color: '#44FF44' },
        { label: '强信号', value: summary.strong_signals, color: '#FFFF44' },
        { label: '中信号', value: summary.medium_signals, color: '#FFAA44' },
        { label: '弱信号', value: summary.weak_signals, color: '#AAAAAA' },
        { label: '模拟收益%', value: summary.simulated_return_pct, color: '#44BBFF' },
      ]
    : [];

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
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              disabled={isLoading}
              className="input-terminal bg-[#1a1a2e] text-sm border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan/50"
            />
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
            <div className="flex items-start gap-2 mb-2">
              {signalStats.length > 0 && (
                <Card variant="default" padding="sm" className="w-56 flex-shrink-0">
                  <h3 className="text-xs font-medium text-muted mb-2">信号统计</h3>
                  <div className="space-y-1">
                    {signalStats.map((s) => (
                      <div key={s.label} className="flex items-center justify-between">
                        <span className="text-xs text-muted">{s.label}</span>
                        <span
                          className="text-xs font-mono font-medium"
                          style={{ color: s.color }}
                        >
                          {typeof s.value === 'number' && s.label.includes('%')
                            ? `${s.value.toFixed(2)}%`
                            : s.value}
                        </span>
                      </div>
                    ))}
                </div>
                </Card>
              )}

              {intradayData && intradayData.signals.length > 0 && (
                <div className="flex-1 max-h-64 overflow-y-auto">
                  <div className="text-xs text-muted/60 mb-1">
                    共 {intradayData.signals.length} 条信号
                  </div>
                  <div className="space-y-1">
                    {intradayData.signals.slice().reverse().map((sig, idx) => {
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
                          置信度 {(sig.confidence * 100).toFixed(0)}%
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
                {intradayData?.indicator_sub_charts?.map((sc) => {
                  const containerRefMap: Record<string, React.RefObject<HTMLDivElement | null>> = {
                    absorption: absorptionContainerRef,
                    main_in_out: mainInOutContainerRef,
                    dragon_tiger_power: dragonTigerContainerRef,
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
                            return displaySignal ? (
                              <span className="text-[11px] font-semibold text-accent px-1.5 py-px rounded bg-accent/10">
                                {displaySignal}
                              </span>
                            ) : null;
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
                      <div ref={ref} style={{ width: '100%', height: sc.id === 'dragon_tiger_power' ? 105 : 115 }} />
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
