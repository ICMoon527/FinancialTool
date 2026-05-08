import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';

import {
  getIntradayData,
  getSearchHistory,
  saveSearchHistory,
  deleteSearchHistory,
  type IntradayDataResponse,
  type IntradaySignal,
  type ReferenceLine,
  type SearchHistoryItem,
} from '../api/intraday';
import { validateStockCode } from '../utils/validation';
import { Card } from '../components/common';

// ============================================================
// 常量
// ============================================================

const CHART_HEIGHT = 480;
const VOLUME_CHART_HEIGHT = 120;

const SIGNAL_COLORS: Record<string, string> = {
  strong: '#FF4444',
  medium: '#FFAA00',
  weak: '#888888',
};

const REFERENCE_LINE_COLORS: Record<string, string> = {
  main_trading_lines: '#FF4444',
  ma_lines: '#4488FF',
  extreme_lines: '#FF8800',
  chip_dense_zone: '#AA44FF',
  prev_close: '#44AAFF',
};

const REFERENCE_LINE_STYLE_MAP: Record<string, 0 | 1 | 2 | 3 | 4> = {
  solid: 0,
  dashed: 2,
  dotted: 1,
  long_dashed: 3,
  dot_dashed: 4,
};

// ============================================================
// 子组件：信号详情抽屉
// ============================================================

interface SignalDetailDrawerProps {
  signal: IntradaySignal | null;
  onClose: () => void;
}

const SignalDetailDrawer: React.FC<SignalDetailDrawerProps> = ({ signal, onClose }) => {
  if (!signal) return null;

  const isBuy = signal.signal_type === 'buy';

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" onClick={onClose}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <div className="absolute inset-y-0 right-0 w-full max-w-md flex">
        <div className="relative w-full flex flex-col bg-[#1a1a2e] border-l border-white/10 shadow-2xl"
          onClick={(e) => e.stopPropagation()}>
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
            <div>
              <span className="text-xs text-muted uppercase tracking-wider">
                {isBuy ? '买入信号' : '卖出信号'}
              </span>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className="inline-block w-3 h-3 rounded-sm"
                  style={{ backgroundColor: isBuy ? '#FF4444' : '#44AA44' }}
                />
                <h2 className="text-base font-semibold text-white">
                  {signal.stock_code}
                </h2>
              </div>
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 text-muted hover:text-white">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted">触发时间</div>
                <div className="text-sm text-white font-mono mt-1">{signal.trigger_time}</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted">触发价格</div>
                <div className="text-sm text-white font-mono mt-1">{signal.price.toFixed(2)}</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted">综合评分</div>
                <div className="text-sm text-white font-mono mt-1">{signal.score}/{signal.max_score}</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted">置信度</div>
                <div className="text-sm font-mono mt-1"
                  style={{ color: signal.confidence >= 0.75 ? '#FF4444' : signal.confidence >= 0.5 ? '#FFAA00' : '#888888' }}>
                  {(signal.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted">仓位建议</div>
                <div className="text-sm text-white font-mono mt-1">{signal.position_advice}</div>
              </div>
              {signal.gravity_adjustment !== undefined && (
                <div className="bg-white/5 rounded-lg p-3">
                  <div className="text-xs text-muted">引力场修正</div>
                  <div className="text-sm font-mono mt-1"
                    style={{ color: signal.gravity_adjustment >= 0 ? '#44AA44' : '#FF4444' }}>
                    {signal.gravity_adjustment >= 0 ? '+' : ''}{signal.gravity_adjustment.toFixed(4)}
                  </div>
                </div>
              )}
            </div>

            {signal.support_force !== undefined && signal.pressure_force !== undefined && (
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted mb-2">受力分析</div>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#44AA44] w-16">支撑力</span>
                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-[#44AA44] rounded-full transition-all"
                        style={{ width: `${Math.min(signal.support_force * 10, 100)}%` }} />
                    </div>
                    <span className="text-xs text-[#44AA44] font-mono w-12 text-right">{signal.support_force.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#FF4444] w-16">压力力</span>
                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-[#FF4444] rounded-full transition-all"
                        style={{ width: `${Math.min(signal.pressure_force * 10, 100)}%` }} />
                    </div>
                    <span className="text-xs text-[#FF4444] font-mono w-12 text-right">{signal.pressure_force.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )}

            {signal.reasoning && (
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-xs text-muted mb-2">信号逻辑</div>
                <div className="text-xs text-secondary leading-relaxed whitespace-pre-wrap">{signal.reasoning}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// 主页面组件
// ============================================================

const IntradayPage: React.FC = () => {
  // ---------- 状态 ----------
  const [stockCode, setStockCode] = useState('');
  const [dateValue, setDateValue] = useState('');
  const [inputError, setInputError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [intradayData, setIntradayData] = useState<IntradayDataResponse | null>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<IntradaySignal | null>(null);
  const [showReferenceLines, setShowReferenceLines] = useState(true);
  const [showSignals, setShowSignals] = useState(true);
  const [referenceLineCategories, setReferenceLineCategories] = useState<Record<string, boolean>>({});

  // ---------- 图表引用 ----------
  const mainChartContainerRef = useRef<HTMLDivElement>(null);
  const mainChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const volumeChartContainerRef = useRef<HTMLDivElement>(null);
  const volumeChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const isChartInitialized = useRef(false);
  const priceLinesRef = useRef<lightweightCharts.IPriceLine[]>([]);
  const signalMarkersRef = useRef<any>(null);

  // ---------- 工具函数 ----------

  /** 解析时间戳为 Unix 秒数 */
  const parseTimestamp = useCallback((ts: string, dateBase?: string): number => {
    if (!ts) return 0;

    // 尝试纯时间格式 "HH:MM:SS" 或 "HH:MM"
    const timeMatch = ts.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
    if (timeMatch && dateBase) {
      const h = parseInt(timeMatch[1], 10);
      const m = parseInt(timeMatch[2], 10);
      const s = parseInt(timeMatch[3] || '0', 10);
      const d = new Date(dateBase);
      d.setHours(h, m, s, 0);
      return Math.floor(d.getTime() / 1000);
    }

    // 尝试 "YYYY-MM-DD HH:MM:SS" 或完整时间字符串
    const d = new Date(ts);
    if (!isNaN(d.getTime())) {
      return Math.floor(d.getTime() / 1000);
    }

    return 0;
  }, []);

  /** 格式化日期 YYYY-MM-DD */
  const formatDate = (d: Date): string => {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  };

  // ---------- 图表初始化 ----------
  useEffect(() => {
    if (!mainChartContainerRef.current || !volumeChartContainerRef.current) return;
    if (isChartInitialized.current) return;

    try {
      // 主图
      const mainChart = lightweightCharts.createChart(mainChartContainerRef.current, {
        width: mainChartContainerRef.current.clientWidth,
        height: CHART_HEIGHT,
        layout: {
          background: { type: 'solid', color: '#1a1a2e' } as any,
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: '#2b2b43' },
          horzLines: { color: '#2b2b43' },
        },
        crosshair: {
          mode: 1,
          vertLine: { color: '#9B7DFF', width: 1, style: 2 },
          horzLine: { color: '#9B7DFF', width: 1, style: 2 },
        },
        localization: {
          timeFormatter: (time: any) => {
            const d = new Date(time * 1000);
            const h = String(d.getHours()).padStart(2, '0');
            const m = String(d.getMinutes()).padStart(2, '0');
            return `${h}:${m}`;
          },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time: any) => {
            const d = new Date(time * 1000);
            const h = String(d.getHours()).padStart(2, '0');
            const m = String(d.getMinutes()).padStart(2, '0');
            return `${h}:${m}`;
          },
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          mouseWheel: true,
          pinch: true,
          axisPressedMouseMove: { time: false, price: false },
          axisDoubleClickReset: { time: true, price: true },
        },
      });

      const candlestickSeries = mainChart.addSeries(lightweightCharts.CandlestickSeries, {
        upColor: '#FF4444',
        downColor: '#44AA44',
        borderDownColor: '#44AA44',
        borderUpColor: '#FF4444',
        wickDownColor: '#44AA44',
        wickUpColor: '#FF4444',
      });

      mainChartRef.current = mainChart;
      candlestickSeriesRef.current = candlestickSeries;

      // 成交量子图
      const volChart = lightweightCharts.createChart(volumeChartContainerRef.current, {
        width: volumeChartContainerRef.current.clientWidth,
        height: VOLUME_CHART_HEIGHT,
        layout: {
          background: { type: 'solid', color: '#1a1a2e' } as any,
          textColor: '#8a8a9a',
        },
        grid: {
          vertLines: { visible: false },
          horzLines: { color: '#2b2b43' },
        },
        crosshair: { mode: 0 },
        timeScale: {
          borderVisible: false,
          timeVisible: false,
          visible: false,
        },
        rightPriceScale: {
          borderVisible: false,
          visible: false,
        },
        handleScroll: { mouseWheel: false, pressedMouseMove: false },
        handleScale: { mouseWheel: false, pinch: false },
      });

      const volSeries = volChart.addSeries(lightweightCharts.HistogramSeries, {
        color: '#444488',
        priceFormat: { type: 'volume' as any },
      });

      volumeChartRef.current = volChart;
      volumeSeriesRef.current = volSeries;
      isChartInitialized.current = true;

      // 时间轴同步：主图 ↔ 成交量
      const syncTime = () => {
        if (!mainChartRef.current || !volumeChartRef.current) return;
        try {
          const range = mainChartRef.current.timeScale().getVisibleRange();
          if (range) {
            volumeChartRef.current.timeScale().setVisibleRange(range, false);
          }
        } catch (_) { /* ignore */ }
      };
      mainChart.timeScale().subscribeVisibleTimeRangeChange(syncTime);

      // 窗口大小自适应
      const mainResize = new ResizeObserver((entries) => {
        if (entries.length === 0 || !mainChartRef.current) return;
        const { width } = entries[0].contentRect;
        mainChartRef.current.applyOptions({ width, height: CHART_HEIGHT });
      });
      mainResize.observe(mainChartContainerRef.current);

      const volResize = new ResizeObserver((entries) => {
        if (entries.length === 0 || !volumeChartRef.current) return;
        const { width } = entries[0].contentRect;
        volumeChartRef.current.applyOptions({ width, height: VOLUME_CHART_HEIGHT });
        volumeChartRef.current.timeScale().applyOptions({ visible: false });
      });
      volResize.observe(volumeChartContainerRef.current);

    } catch (err) {
      console.error('图表初始化失败:', err);
    }
  }, []);

  // ---------- 更新图表数据 ----------
  useEffect(() => {
    if (!intradayData || !candlestickSeriesRef.current) return;

    const dateBase = intradayData.date; // "YYYY-MM-DD"

    // K线数据
    const klineData = intradayData.kline_data
      .map((k) => {
        const t = parseTimestamp(k.timestamp, dateBase);
        if (t <= 0) return null;
        return {
          time: t as any,
          open: k.Open,
          high: k.High,
          low: k.Low,
          close: k.Close,
        };
      })
      .filter(Boolean);

    candlestickSeriesRef.current.setData(klineData);

    // 成交量数据
    const volumeData = intradayData.kline_data
      .map((k) => {
        const t = parseTimestamp(k.timestamp, dateBase);
        if (t <= 0) return null;
        const isRise = k.Close >= k.Open;
        return {
          time: t as any,
          value: k.Volume,
          color: isRise ? '#442222' : '#224422',
        };
      })
      .filter(Boolean);

    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(volumeData);
    }

    // 信号标记
    if (showSignals) {
      updateSignalMarkers(intradayData.signals, dateBase);
    }

    // 参考线
    if (showReferenceLines) {
      updateReferenceLines(intradayData.reference_lines);
    }

    if (mainChartRef.current) {
      mainChartRef.current.timeScale().fitContent();
    }
  }, [intradayData, showSignals, showReferenceLines, referenceLineCategories, parseTimestamp]);

  // ---------- 信号标记渲染 ----------
  const updateSignalMarkers = useCallback((signals: IntradaySignal[], dateBase: string) => {
    if (!candlestickSeriesRef.current) return;

    // 先清旧标记
    if (signalMarkersRef.current) {
      try {
        candlestickSeriesRef.current.setMarkers([]);
      } catch (_) {}
    }

    const markers = signals.map((sig) => {
      const isBuy = sig.signal_type === 'buy';
      const t = parseTimestamp(sig.trigger_time, dateBase);
      return {
        time: t as any,
        position: isBuy ? 'belowBar' as const : 'aboveBar' as const,
        color: isBuy ? '#FF4444' : '#44AA44',
        shape: isBuy ? 'arrowUp' as const : 'arrowDown' as const,
        text: (sig.confidence >= 0.75 ? '★' : sig.confidence >= 0.5 ? '●' : '○'),
        size: 2,
        id: `${sig.signal_type}_${sig.trigger_time}`,
      };
    });

    candlestickSeriesRef.current.setMarkers(markers);
    signalMarkersRef.current = markers;
  }, [parseTimestamp]);

  // ---------- 参考线渲染 ----------
  const updateReferenceLines = useCallback((lines: ReferenceLine[]) => {
    if (!candlestickSeriesRef.current) return;

    // 清除旧线
    priceLinesRef.current.forEach((pl) => {
      try { candlestickSeriesRef.current.removePriceLine(pl); } catch (_) {}
    });
    priceLinesRef.current = [];

    // 分组过滤
    const enabledCategories = Object.keys(referenceLineCategories).length > 0
      ? referenceLineCategories
      : null;

    const filteredLines = enabledCategories
      ? lines.filter((l) => {
          const cat = l.category || 'other';
          if (cat in enabledCategories) return enabledCategories[cat];
          return true;
        })
      : lines;

    const sortedLines = [...filteredLines].sort((a, b) => b.base_weight - a.base_weight);

    sortedLines.forEach((line) => {
      try {
        const pl = candlestickSeriesRef.current.createPriceLine({
          price: line.price,
          color: line.color || '#888888',
          lineWidth: line.base_weight >= 1.5 ? 2 : 1,
          lineStyle: REFERENCE_LINE_STYLE_MAP[line.style] ?? 2,
          axisLabelVisible: true,
          title: line.label,
        });
        priceLinesRef.current.push(pl);
      } catch (_) {}
    });
  }, [referenceLineCategories]);

  // ---------- 搜索历史 ----------
  const loadSearchHistory = useCallback(async () => {
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

  useEffect(() => { loadSearchHistory(); }, [loadSearchHistory]);

  // ---------- 搜索处理 ----------
  const handleSearch = useCallback(async () => {
    const code = stockCode.trim().toUpperCase();
    if (!code) {
      setInputError('请输入股票代码');
      return;
    }

    const validation = validateStockCode(code);
    if (!validation.valid) {
      setInputError(validation.message);
      return;
    }

    setInputError(undefined);
    setIsLoading(true);

    try {
      const data = await getIntradayData(code, dateValue || undefined);
      setIntradayData(data);

      // 初始化参考线类别开关（默认全开）
      const cats: Record<string, boolean> = {};
      data.reference_lines.forEach((l) => {
        const c = l.category || 'other';
        if (!(c in cats)) cats[c] = true;
      });
      setReferenceLineCategories(cats);

      // 保存搜索历史
      try {
        await saveSearchHistory(code, data.stock_name || '', data.date);
        loadSearchHistory();
      } catch (_) {}
    } catch (err: any) {
      setInputError(err.message || '获取数据失败');
      setIntradayData(null);
    } finally {
      setIsLoading(false);
    }
  }, [stockCode, dateValue, loadSearchHistory]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  const handleHistoryClick = useCallback((item: SearchHistoryItem) => {
    setStockCode(item.stock_code);
    setDateValue(item.date || '');
    // 触发搜索
    setIsLoading(true);
    getIntradayData(item.stock_code, item.date || undefined)
      .then((data) => {
        setIntradayData(data);
        const cats: Record<string, boolean> = {};
        data.reference_lines.forEach((l) => {
          const c = l.category || 'other';
          if (!(c in cats)) cats[c] = true;
        });
        setReferenceLineCategories(cats);
      })
      .catch((err: any) => setInputError(err.message || '获取数据失败'))
      .finally(() => setIsLoading(false));
  }, []);

  const handleDeleteHistory = useCallback(async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSearchHistory(id);
      loadSearchHistory();
    } catch (_) {}
  }, [loadSearchHistory]);

  // ---------- 信号点击处理 ----------
  const handleSignalClick = useCallback((signal: IntradaySignal) => {
    setSelectedSignal(signal);
  }, []);

  // ---------- 参考线类别切换 ----------
  const toggleReferenceLineCategory = (cat: string) => {
    setReferenceLineCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  // ---------- 参考线类别名称映射 ----------
  const categoryLabels: Record<string, string> = {
    main_trading_lines: '操盘三线',
    ma_lines: '移动均线',
    extreme_lines: '前高/前低',
    chip_dense_zone: '筹码密集区',
    prev_close: '昨收',
  };

  // ---------- 获取可见参考线列表 ----------
  const visibleCategories = Object.entries(referenceLineCategories)
    .filter(([, v]) => v)
    .map(([k]) => k);
  const visibleLines = intradayData?.reference_lines.filter(
    (l) => visibleCategories.includes(l.category || 'other')
  ) || [];

  // ---------- 构建参考线数据用于交互面板 ----------
  const refLinesByCategory: Record<string, ReferenceLine[]> = {};
  intradayData?.reference_lines.forEach((l) => {
    const c = l.category || 'other';
    if (!refLinesByCategory[c]) refLinesByCategory[c] = [];
    refLinesByCategory[c].push(l);
  });

  // ============================================================
  // 渲染
  // ============================================================

  // 侧边栏
  const sidebarContent = (
    <div className="flex flex-col overflow-hidden min-h-0 h-full">
      <div className="p-3 border-b border-white/5 flex-shrink-0">
        <h3 className="text-sm font-medium text-white">搜索历史</h3>
      </div>
      <div className="overflow-y-auto px-3 py-1 flex-1">
        {isLoadingHistory ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
          </div>
        ) : searchHistory.length === 0 ? (
          <p className="text-xs text-muted text-center py-4">暂无搜索历史</p>
        ) : (
          <div className="space-y-2">
            {searchHistory.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => handleHistoryClick(item)}
                className="w-full text-left p-2 rounded-lg hover:bg-white/5 transition-colors group"
              >
                <div className="flex items-center gap-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1.5">
                      <span className="font-medium text-white truncate text-xs">
                        {item.stock_name || item.stock_code}
                      </span>
                      <span className="flex-shrink-0 text-cyan text-xs font-medium">
                        {item.stock_code}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className="text-xs text-muted font-mono">{item.date}</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteHistory(item.id, e)}
                    className="p-1 text-muted hover:text-danger transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100"
                    title="删除"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </button>
            ))}
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
        <div className="flex items-center gap-2 w-full min-w-0 flex-1" style={{ maxWidth: 'min(100%, 1168px)' }}>
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

          <input
            type="date"
            value={dateValue}
            onChange={(e) => setDateValue(e.target.value)}
            className="input-terminal bg-[#1a1a2e] text-sm border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan/50 flex-shrink-0"
            title="选择历史日期，留空默认当日"
          />

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

      {/* Desktop sidebar */}
      <div className="hidden md:flex col-start-2 row-start-2 flex-col overflow-hidden min-h-0 h-full">
        {sidebarContent}
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 md:hidden" onClick={() => setSidebarOpen(false)}>
          <div className="absolute inset-0 bg-black/60" />
          <div
            className="absolute left-0 top-0 bottom-0 w-72 flex flex-col bg-[#12122a] overflow