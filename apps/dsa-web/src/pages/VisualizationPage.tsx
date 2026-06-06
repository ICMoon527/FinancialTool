import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';

import { visualizationApi, type VisualizationResponse, type VisualizationSearchHistoryItem, type ChipDistributionResponse } from '../api/visualization';
import type { StockSnapshot } from '../api/intraday';
import { validateStockCode } from '../utils/validation';
import { Card, StockSearchInput } from '../components/common';
import { useStockPriceHistory } from '../hooks';
import ChipDistributionChart from '../components/charts/ChipDistributionChart';

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

// 定义指标配置
const INDICATOR_OPTIONS = [
  { id: 'volume', name: '成交量', description: '成交量柱状图', color: '#666666' },
  { id: 'macd', name: 'MACD', description: '指数平滑异同移动平均线 (DIF/DEA/Bar)', color: '#FFFFFF' },
  { id: 'expma', name: 'EXPMA', description: '指数移动平均线 (EXPMA13, EXPMA30)', color: '#FFFFFF' },
  { id: 'main_capital_absorption', name: '主力吸筹', description: '主力资金吸筹情况', color: '#AA44FF' },
  { id: 'main_cost', name: '主力成本', description: '主力资金成本', color: '#FFAA00' },
  { id: 'main_trading', name: '主力操盘', description: '主力操盘三线', color: '#FF4444' },
  { id: 'banker_control', name: '庄家控盘', description: '庄家控盘程度指标', color: '#FFAA00' },
  { id: 'momentum_2', name: '动能二号', description: '动能二号指标，四种颜色表示不同动能状态', color: '#FF4444' },
  { id: 'strong_detonation', name: '强势起爆', description: '强势起爆指标，识别强势起爆阶段', color: '#AA44FF' },
  { id: 'resonance_chase', name: '共振追涨', description: '共振追涨指标，识别共振追涨机会', color: '#AA44FF' },
  { id: 'dmi', name: 'DMI', description: '方向移动指数，显示趋势方向和强度', color: '#44AA44' },
  { id: 'tiandao', name: '天道', description: '天道做T指标 (金牛/金钻趋势/金牛2)', color: '#FFD700' },
  { id: 'bollinger', name: '布林带', description: '布林带指标 (上轨/中轨/下轨)', color: '#FF8C00' },
];

// 子图高度
const SUBCHART_HEIGHT = 150;

// 日期范围选项
const DATE_RANGE_OPTIONS = [
  { id: '30d', name: '最近30天', getStartDate: () => { const d = new Date(); d.setDate(d.getDate() - 30); return d; } },
  { id: '90d', name: '最近90天', getStartDate: () => { const d = new Date(); d.setDate(d.getDate() - 90); return d; } },
  { id: '180d', name: '最近180天', getStartDate: () => { const d = new Date(); d.setDate(d.getDate() - 180); return d; } },
  { id: '1y', name: '最近1年', getStartDate: () => { const d = new Date(); d.setFullYear(d.getFullYear() - 1); return d; } },
  { id: '3y', name: '最近3年', getStartDate: () => { const d = new Date(); d.setFullYear(d.getFullYear() - 3); return d; } },
  { id: '5y', name: '最近5年', getStartDate: () => { const d = new Date(); d.setFullYear(d.getFullYear() - 5); return d; } },
  { id: 'all', name: '全部数据', getStartDate: null },
];

const VisualizationPage: React.FC = () => {
  const validIndicatorIds = INDICATOR_OPTIONS.map(opt => opt.id);
  
  const filterValidIndicators = (indicators: string[]) => {
    return indicators.filter(id => validIndicatorIds.includes(id));
  };

  // 股价记录管理
  const { 
    recordPrice, 
    deletePriceRecord, 
    calculatePriceChange 
  } = useStockPriceHistory();

  // 状态管理
  const [stockCode, setStockCode] = useState('');
  const [inputError, setInputError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [visualizationData, setVisualizationData] = useState<VisualizationResponse | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [searchHistory, setSearchHistory] = useState<VisualizationSearchHistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<number | null>(null);
  const [historySnapshots, setHistorySnapshots] = useState<Record<string, StockSnapshot>>({});
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(
    filterValidIndicators(['volume', 'macd', 'main_capital_absorption', 'main_cost', 'main_trading'])
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedDateRange, setSelectedDateRange] = useState('1y');
  const [chipDistributionData, setChipDistributionData] = useState<ChipDistributionResponse | null>(null);
  const [isLoadingChipDistribution, setIsLoadingChipDistribution] = useState(false);
  const [showChipDistribution, setShowChipDistribution] = useState(false);
  const [priceRange, setPriceRange] = useState<{ min: number; max: number } | null>(null);
  const [cursorPrice, setCursorPrice] = useState<number | null>(null);
  const chipDistributionTimerRef = useRef<number | null>(null); // 防抖定时器
  const handlePriceRangeChangeRef = useRef<(() => void) | null>(null); // 价格范围更新函数引用
  const lastChipDistributionDateRef = useRef<string | null>(null); // 记录上次请求的日期，避免重复请求
  const [cursorValues, setCursorValues] = useState<{
    attackLine?: number;
    tradingLine?: number;
    defenseLine?: number;
    signal?: 'buy' | 'sell';
    bankerControl?: number;
    mainCapitalAbsorption?: number;
    mainCost?: number;
    momentum2?: number;
    strongDetonation?: number;
    resonanceChase?: number;
    dmiPdi?: number;
    dmiMdi?: number;
    dmiAdx?: number;
    closePrice?: number;
    mainNetBuyWan?: number;
    mainDirection?: 'inflow' | 'outflow';
    turnoverRatio?: number;
    macdBarSum?: number;
    macdBarDiff?: number;
    macdSignal?: string;
    expmaFast?: number;
    expmaSlow?: number;
    bollingerUpper?: number;
    bollingerMiddle?: number;
    bollingerLower?: number;
    tiandaoJinzuan?: number;
    tiandaoJinniu?: number;
    tiandaoJinniu2?: number;
    tiandaoBbi?: number;
  }>({});

  useEffect(() => {
    setSelectedIndicators(prev => filterValidIndicators(prev));
  }, []);

  // 图表引用
  const mainChartContainerRef = useRef<HTMLDivElement>(null);
  const mainChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const isChartInitialized = useRef(false);
  
  // 主力操盘系列引用
  const attackLineSeriesRef = useRef<any>(null);
  const tradingLineSeriesRef = useRef<any>(null);
  const defenseLineSeriesRef = useRef<any>(null);
  const buyPointSeriesRef = useRef<any>(null);
  const sellPointSeriesRef = useRef<any>(null);
  
  // 存储主力操盘数据用于十字线值查找
  const mainTradingDataRef = useRef<any>(null);
  
  // EXPMA 系列引用
  const expmaFastSeriesRef = useRef<any>(null);
  const expmaSlowSeriesRef = useRef<any>(null);
  // 存储 EXPMA 数据用于十字线值查找
  const expmaDataRef = useRef<any>(null);
  
  // 天道指标系列引用
  const tiandaoBbiSeriesRef = useRef<any>(null);
  const tiandaoJinzuanSeriesRef = useRef<any>(null);
  const tiandaoJinniuSeriesRef = useRef<any>(null);
  const tiandaoJinniu2SeriesRef = useRef<any>(null);
  // 存储天道数据用于十字线值查找
  const tiandaoDataRef = useRef<any>(null);
  // 天道买入信号标记插件引用
  const tiandaoMarkersRef = useRef<any>(null);
  
  // 布林带系列引用
  const bollingerUpperSeriesRef = useRef<any>(null);
  const bollingerMiddleSeriesRef = useRef<any>(null);
  const bollingerLowerSeriesRef = useRef<any>(null);
  // 存储布林带数据
  const bollingerDataRef = useRef<any>(null);
  
  const klineDataRef = useRef<any>(null);
  const latestDateRef = useRef<any>(null);
  const earliestDateRef = useRef<any>(null);
  const isTimeRangeUpdatingRef = useRef(false);
  
  // 子图引用
  const subChartContainerRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const subChartRefs = useRef<{ [key: string]: lightweightCharts.IChartApi | null }>({});
  const subChartSeriesRefs = useRef<{ [key: string]: any }>({});
  const subChartResizeObservers = useRef<{ [key: string]: ResizeObserver | null }>({});
  const timeSyncSubscription = useRef<any>(null);
  const subChartCrosshairSubscriptions = useRef<{ [key: string]: any }>({});
  const isCrosshairUpdatingRef = useRef(false);
  const currentCrosshairTimeRef = useRef<any>(null);
  
  // 存储所有指标数据用于十字线值查找
  const indicatorsDataRef = useRef<{ [key: string]: any }>({});
  // 存储预计算的筹码分布数据
  const precomputedChipDistributionRef = useRef<{
    latest?: ChipDistributionResponse;
    history?: Array<{ date: string; data: ChipDistributionResponse }>;
  } | null>(null);
  
  // 保存最新的visualizationData到ref中，确保handleCrosshairMove能获取到最新数据
  const visualizationDataRef = useRef<any>(null);
  // 保存showChipDistribution的最新状态到ref
  const showChipDistributionRef = useRef(false);

  // 获取筹码峰数据
  const fetchChipDistribution = useCallback(async (stockCode: string, endDate?: string) => {
    if (!stockCode) return;
    
    console.log('开始获取筹码分布，股票代码:', stockCode, '日期:', endDate);
    
    // 直接从API获取，不使用预计算数据
    setIsLoadingChipDistribution(true);
    try {
      const dateRangeOption = DATE_RANGE_OPTIONS.find(opt => opt.id === selectedDateRange);
      let startDateStr: string | undefined;
      let days = 365; // 默认天数
      
      // 如果有endDate，从endDate往前推；否则从今天往前推
      const baseDate = endDate ? new Date(endDate) : new Date();
      
      if (dateRangeOption?.getStartDate) {
        const startDate = dateRangeOption.getStartDate();
        
        // 计算时间跨度（毫秒）
        const today = new Date();
        const timeSpan = today.getTime() - startDate.getTime();
        
        // 从baseDate往前推相同的时间跨度
        const calculatedStartDate = new Date(baseDate.getTime() - timeSpan);
        startDateStr = calculatedStartDate.toISOString().split('T')[0];
        
        // 计算天数
        days = Math.ceil(timeSpan / (1000 * 60 * 60 * 24));
      } else {
        // 全部数据，不传startDate，后端会使用默认逻辑
        startDateStr = undefined;
        days = 3650; // 10年
      }
      
      const response = await visualizationApi.getChipDistribution(
        stockCode,
        days,
        startDateStr,
        endDate // 传入endDate，后端会根据日期计算end_date_idx
      );
      setChipDistributionData(response);
    } catch (err) {
      console.error('Failed to load chip distribution data:', err);
      setChipDistributionData(null);
    } finally {
      setIsLoadingChipDistribution(false);
    }
  }, [selectedDateRange]);

  // 清理防抖定时器
  useEffect(() => {
    return () => {
      if (chipDistributionTimerRef.current) {
        clearTimeout(chipDistributionTimerRef.current);
      }
    };
  }, []);

  // 加载搜索历史
  const loadSearchHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const response = await visualizationApi.getSearchHistory(100);
      setSearchHistory(response.items);
    } catch (err) {
      console.error('Failed to load search history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  // 初始化加载
  useEffect(() => {
    loadSearchHistory();
  }, [loadSearchHistory]);

  const searchHistoryRef = useRef<VisualizationSearchHistoryItem[]>([]);
  searchHistoryRef.current = searchHistory;

  const fetchAndUpdate = useCallback(async () => {
    const codes = searchHistoryRef.current.map(h => h.stock_code);
    if (codes.length === 0) return;
    try {
      const resp = await visualizationApi.getBatchStatus(codes);
      setHistorySnapshots(resp.snapshots);
    } catch {
      // 静默失败，不打扰用户
    }
  }, []);

  useEffect(() => {
    if (searchHistory.length > 0) {
      fetchAndUpdate();
    }
  }, [searchHistory.length, fetchAndUpdate]);
  
  // 确保visualizationDataRef始终与最新状态同步
  useEffect(() => {
    if (visualizationData) {
      visualizationDataRef.current = visualizationData;
      console.log('visualizationDataRef已更新:', visualizationData);
    }
  }, [visualizationData]);
  
  // 确保showChipDistributionRef始终与最新状态同步
  useEffect(() => {
    showChipDistributionRef.current = showChipDistribution;
  }, [showChipDistribution]);
  
  // 当显示筹码峰且有股票数据时,自动获取筹码峰数据
  useEffect(() => {
    if (showChipDistribution && visualizationData?.stock_code) {
      console.log('显示筹码峰时调用API获取数据');
      // 直接调用API，不使用预计算数据
      fetchChipDistribution(visualizationData.stock_code);
    }
  }, [showChipDistribution, visualizationData?.stock_code]);

  // 通用函数：获取指定指标在指定时间的数据项
  const getIndicatorDataItem = (indicatorId: string, time: any) => {
    if (indicatorsDataRef.current[indicatorId]) {
      return indicatorsDataRef.current[indicatorId].data.find((item: any) => {
        const itemDate = item.date || item.time;
        return itemDate === time;
      });
    }
    return undefined;
  };

  // 获取所有指标在指定时间点的值
  const getAllIndicatorValuesAtTime = (time: any) => {
    const values: {
      bankerControl?: number;
      mainCapitalAbsorption?: number;
      mainCost?: number;
      momentum2?: number;
      strongDetonation?: number;
      resonanceChase?: number;
      dmiPdi?: number;
      dmiMdi?: number;
      dmiAdx?: number;
      mainNetBuyWan?: number;
      mainDirection?: 'inflow' | 'outflow';
      turnoverRatio?: number;
      macdDIF?: number;
      macdDEA?: number;
      macdBar?: number;
      macdBarSum?: number;
      macdBarDiff?: number;
      macdSignal?: string;
      expmaFast?: number;
      expmaSlow?: number;
      tiandaoJinzuan?: number;
      tiandaoJinniu?: number;
      tiandaoJinniu2?: number;
      tiandaoBbi?: number;
    } = {};

    const bankerData = getIndicatorDataItem('banker_control', time);
    if (bankerData) {
      values.bankerControl = bankerData.control_degree;
    }

    const absorptionData = getIndicatorDataItem('main_capital_absorption', time);
    if (absorptionData) {
      values.mainCapitalAbsorption = absorptionData.main_capital_absorption;
    }

    const mainCostData = getIndicatorDataItem('main_cost', time);
    if (mainCostData) {
      values.mainCost = mainCostData.main_cost || mainCostData.cost;
      values.mainNetBuyWan = mainCostData.main_net_buy_wan;
      values.mainDirection = mainCostData.main_direction;
      values.turnoverRatio = mainCostData.turnover_ratio;
    }

    const momentum2Data = getIndicatorDataItem('momentum_2', time);
    if (momentum2Data) {
      values.momentum2 = momentum2Data.strong_momentum || momentum2Data.medium_momentum || 
                         momentum2Data.no_momentum || momentum2Data.recovery_momentum;
    }

    const strongDetonationData = getIndicatorDataItem('strong_detonation', time);
    if (strongDetonationData) {
      values.strongDetonation = strongDetonationData.bull_line;
    }

    const resonanceChaseData = getIndicatorDataItem('resonance_chase', time);
    if (resonanceChaseData && resonanceChaseData.resonance) {
      const originalHeight = Math.abs(resonanceChaseData.OUT1 || 0);
      const topValue = originalHeight - 0.5;
      values.resonanceChase = topValue > 0 ? topValue : undefined;
    }

    const dmiData = getIndicatorDataItem('dmi', time);
    if (dmiData) {
      values.dmiPdi = dmiData.PDI;
      values.dmiMdi = dmiData.MDI;
      values.dmiAdx = dmiData.ADX;
    }

    const macdData = getIndicatorDataItem('macd', time);
    if (macdData) {
      values.macdDIF = macdData.DIF;
      values.macdDEA = macdData.DEA;
      values.macdBar = macdData.MACD_Bar;

      const macdIndicatorData = indicatorsDataRef.current['macd'];
      if (macdIndicatorData && macdIndicatorData.data) {
        let timeStr = time;
        if (typeof time === 'number') {
          const date = new Date(time * 1000);
          timeStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
        }
        const idx = macdIndicatorData.data.findIndex((d: any) => {
          const dTime = d.date || d.time;
          return dTime === timeStr;
        });
        if (idx >= 0) {
          const barDataUpToIdx = macdIndicatorData.data.slice(0, idx + 1);
          values.macdBarSum = barDataUpToIdx.reduce((sum: number, d: any) => sum + (Number(d.MACD_Bar) || 0), 0);
          values.macdBarDiff = idx > 0
            ? (Number(macdIndicatorData.data[idx].MACD_Bar) || 0) - (Number(macdIndicatorData.data[idx - 1].MACD_Bar) || 0)
            : 0;
          const dif = Number(macdIndicatorData.data[idx].DIF) || 0;
          const dea = Number(macdIndicatorData.data[idx].DEA) || 0;
          values.macdSignal = dif > dea ? 'MACD多头 \u2197' : 'MACD空头 \u2198';
        }
      }
    }

    const expmaData = getIndicatorDataItem('expma', time);
    if (expmaData) {
      values.expmaFast = expmaData.EXPMA_Fast;
      values.expmaSlow = expmaData.EXPMA_Slow;
    }

    const tiandaoData = getIndicatorDataItem('tiandao', time);
    if (tiandaoData) {
      values.tiandaoJinzuan = tiandaoData.td_jinzuan;
      values.tiandaoJinniu = tiandaoData.td_jinniu;
      values.tiandaoJinniu2 = tiandaoData.td_jinniu2;
      values.tiandaoBbi = tiandaoData.td_bbi;
    }

    return values;
  };

  // 初始化主图表
  useEffect(() => {
    if (!mainChartContainerRef.current) {
      console.log('Chart container not found');
      return;
    }

    if (isChartInitialized.current) {
      console.log('Chart already initialized');
      return;
    }

    console.log('Initializing chart...');

    try {
      // 创建图表
      const chart = lightweightCharts.createChart(mainChartContainerRef.current, {
        width: mainChartContainerRef.current.clientWidth,
        height: 500,
        layout: {
          background: { type: 'solid', color: '#1a1a2e' } as any,
          textColor: '#d1d4dc',
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: '#2b2b43' },
          horzLines: { color: '#2b2b43' },
        },
        crosshair: {
          mode: 1, // 0=Normal, 1=Magnet, 2=Hidden
          vertLine: {
            color: '#9B7DFF',
            width: 1,
            style: 2,
          },
          horzLine: {
            color: '#9B7DFF',
            width: 1,
            style: 2,
          },
        },
        localization: {
          timeFormatter: (time: any) => {
            let date: Date;
            if (typeof time === 'string' && time.includes('-')) {
              date = new Date(time);
            } else {
              date = new Date(time * 1000);
            }
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          tickMarkFormatter: (time: any) => {
            let date: Date;
            if (typeof time === 'string' && time.includes('-')) {
              date = new Date(time);
            } else {
              date = new Date(time * 1000);
            }
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
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
          axisPressedMouseMove: {
            time: false,
            price: false,
          },
          axisDoubleClickReset: {
            time: true,
            price: true,
          },
        },
      });

      console.log('Chart created');

      // 创建K线系列 - Lightweight Charts v5 新API
      const candlestickSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
        upColor: '#FF4444',
        downColor: '#44AA44',
        borderDownColor: '#44AA44',
        borderUpColor: '#FF4444',
        wickDownColor: '#44AA44',
        wickUpColor: '#FF4444',
      });

      console.log('Candlestick series created');

      mainChartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;
      isChartInitialized.current = true;

      // 处理窗口大小变化
      const resizeObserver = new ResizeObserver(entries => {
        if (entries.length === 0 || !mainChartRef.current) return;
        const { width } = entries[0].contentRect;
        mainChartRef.current.applyOptions({ width, height: 500 });
      });
      resizeObserver.observe(mainChartContainerRef.current);

      // 订阅主图时间轴变化，同步到所有子图
      const handleTimeScaleChange = () => {
        if (!mainChartRef.current || isTimeRangeUpdatingRef.current) return;
        try {
          isTimeRangeUpdatingRef.current = true;
          const timeRange = mainChartRef.current.timeScale().getVisibleRange();
          if (timeRange && timeRange.from && timeRange.to) {
            // 直接使用时间范围，不限制，让缩放更自由
            // 同步时间范围到子图
            Object.values(subChartRefs.current).forEach(subChart => {
              if (subChart) {
                try {
                  subChart.timeScale().setVisibleRange(timeRange);
                } catch (e) {
                  console.warn('Failed to sync time range to subchart:', e);
                }
              }
            });
          }
        } catch (e) {
          console.warn('Failed to get time range from main chart:', e);
        } finally {
          setTimeout(() => {
            isTimeRangeUpdatingRef.current = false;
          }, 0);
        }
      };

      chart.timeScale().subscribeVisibleTimeRangeChange(handleTimeScaleChange);
      timeSyncSubscription.current = handleTimeScaleChange;
      
      // 订阅主图价格范围变化，同步到筹码分布图
      const handlePriceRangeChange = () => {
        if (!mainChartRef.current || !candlestickSeriesRef.current) {
          return;
        }
        try {
          const priceScale = mainChartRef.current.priceScale('right');
          const visibleRange = priceScale.getVisibleRange();
          if (visibleRange) {
            const options = priceScale.options() as any;
            const bottomMargin = options.scaleMargins?.bottom ?? 0.1;
            const topMargin = options.scaleMargins?.top ?? 0.2;
            const totalMargin = bottomMargin + topMargin;
            const visibleSize = visibleRange.to - visibleRange.from;
            const totalSize = visibleSize / (1 - totalMargin);
            const fullFrom = visibleRange.from - totalSize * bottomMargin;
            const fullTo = visibleRange.to + totalSize * topMargin;
            setPriceRange({ min: fullFrom, max: fullTo });
          }
        } catch (e) {
          console.warn('Failed to get price range from main chart:', e);
        }
      };
      handlePriceRangeChangeRef.current = handlePriceRangeChange;
      
      // 初始获取价格范围
      setTimeout(() => {
        handlePriceRangeChange();
      }, 100);
      
      // 监听时间范围变化也更新价格范围
      const updatePriceRangeOnTimeChange = () => {
        handlePriceRangeChange();
      };
      chart.timeScale().subscribeVisibleTimeRangeChange(updatePriceRangeOnTimeChange);
      
      // 订阅十字线移动以更新光标处三线值和信号
      const handleCrosshairMove = (param: any) => {
        console.log('handleCrosshairMove被调用！param:', param);
        console.log('showChipDistributionRef:', showChipDistributionRef.current, 'visualizationDataRef?:', visualizationDataRef.current);
        
        // 先处理筹码分布更新（不受防止循环更新的限制）
        if (showChipDistributionRef.current && visualizationDataRef.current?.stock_code && param.time) {
          // 将时间转换为日期字符串
          let cursorDateStr: string;
          
          console.log('param.time类型:', typeof param.time, '值:', param.time);
          
          if (typeof param.time === 'string') {
            cursorDateStr = param.time;
          } else {
            const date = new Date(param.time * 1000);
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            cursorDateStr = `${year}-${month}-${day}`;
          }
          
          console.log('十字线移动，当前日期:', cursorDateStr, '上次更新日期:', lastChipDistributionDateRef.current);
          
          // 如果日期与上次相同，不重复请求
          if (lastChipDistributionDateRef.current !== cursorDateStr) {
            console.log('更新筹码分布到日期:', cursorDateStr);
            lastChipDistributionDateRef.current = cursorDateStr;
            fetchChipDistribution(visualizationDataRef.current.stock_code, cursorDateStr);
          }
        }
        
        // 防止循环更新
        if (isCrosshairUpdatingRef.current) {
          return;
        }
        
        isCrosshairUpdatingRef.current = true;
        
        try {
          currentCrosshairTimeRef.current = param.time;
          
          // 更新光标价格用于筹码图十字线同步
          let targetPrice: number | undefined;
          
          if (param.price !== undefined) {
            targetPrice = param.price;
          } else if (param.point && param.point.price !== undefined) {
            targetPrice = param.point.price;
          }
          
          // 如果没有直接价格，或者即使有，也始终用 K 线收盘价（更稳定）
          if (param.time) {
            const klinePoint = klineDataRef.current?.find((item: any) => item.time === param.time);
            if (klinePoint) {
              targetPrice = klinePoint.close; // 优先使用 K 线收盘价
            }
          }
          
          if (targetPrice !== undefined) {
            setCursorPrice(targetPrice);
          }
          
          // 同步十字线到所有子图 - 更新光标值显示和十字线位置
          if (param.time) {
            // 获取所有子图指标的值
            const indicatorValues = getAllIndicatorValuesAtTime(param.time);
            const bankerControl = indicatorValues.bankerControl;
            const mainCapitalAbsorption = indicatorValues.mainCapitalAbsorption;
            const mainCost = indicatorValues.mainCost;
            const momentum2 = indicatorValues.momentum2;
            const strongDetonation = indicatorValues.strongDetonation;
            const resonanceChase = indicatorValues.resonanceChase;
            const dmiPdi = indicatorValues.dmiPdi;
            const dmiMdi = indicatorValues.dmiMdi;
            const dmiAdx = indicatorValues.dmiAdx;
            
            // 同步十字线位置到所有子图
            Object.entries(subChartRefs.current).forEach(([indicatorId, subChart]) => {
              if (subChart && subChartSeriesRefs.current[indicatorId]) {
                try {
                  const series = subChartSeriesRefs.current[indicatorId];
                  
                  let price: number | undefined;
                  if (indicatorId === 'volume') {
                    const klinePoint = klineDataRef.current?.find((item: any) => item.time === param.time);
                    if (klinePoint && klinePoint.originalItem) {
                      price = klinePoint.originalItem.volume;
                    } else if (visualizationData && visualizationData.kline_data) {
                      const originalDataPoint = visualizationData.kline_data.find((item: any) => item.date === param.time);
                      price = originalDataPoint?.volume;
                    }
                    console.log('Volume sync:', { paramTime: param.time, klinePoint, price });
                  } else {
                    const dataPoint = getIndicatorDataItem(indicatorId, param.time);
                    
                    if (dataPoint) {
                      if (indicatorId === 'banker_control') {
                        price = dataPoint.control_degree;
                      } else if (indicatorId === 'main_capital_absorption') {
                        price = dataPoint.main_capital_absorption;
                      } else if (indicatorId === 'main_cost') {
                        price = dataPoint.main_cost || dataPoint.cost;
                      } else if (indicatorId === 'momentum_2') {
                        price = dataPoint.strong_momentum || dataPoint.medium_momentum || 
                                dataPoint.no_momentum || dataPoint.recovery_momentum;
                      } else if (indicatorId === 'strong_detonation') {
                        price = dataPoint.bull_line;
                      } else if (indicatorId === 'resonance_chase') {
                        if (dataPoint.resonance) {
                          const originalHeight = Math.abs(dataPoint.OUT1 || 0);
                          const topValue = originalHeight - 0.5;
                          price = topValue > 0 ? topValue : undefined;
                        }
                      } else if (indicatorId === 'dmi') {
                        if (dataPoint.PDI !== undefined && dataPoint.PDI !== null) {
                          price = dataPoint.PDI;
                        }
                      } else if (indicatorId === 'macd') {
                        if (dataPoint.DIF !== undefined && dataPoint.DIF !== null) {
                          price = dataPoint.DIF;
                        }
                      }
                    }
                  }
                  
                  if (price !== undefined && !isNaN(price)) {
                    console.log('Setting crosshair for', indicatorId, { price, time: param.time });
                    subChart.setCrosshairPosition(price, param.time, series);
                  } else {
                    console.log('No valid price for', indicatorId, { price });
                  }
                } catch (e) {
                  console.warn('Failed to sync crosshair to subchart:', e);
                }
              }
            });
            
            // 获取布林带光标值
            let bollingerUpper: number | undefined;
            let bollingerMiddle: number | undefined;
            let bollingerLower: number | undefined;
            const bollingerPoint = bollingerDataRef.current?.data?.find((item: any) => item.date === param.time);
            if (bollingerPoint) {
              bollingerUpper = bollingerPoint.bollinger_upper;
              bollingerMiddle = bollingerPoint.bollinger_middle;
              bollingerLower = bollingerPoint.bollinger_lower;
            }

            // 更新光标值，包括所有子图指标
            if (!mainTradingDataRef.current || !klineDataRef.current) {
              setCursorValues({
                bankerControl,
                mainCapitalAbsorption,
                mainCost,
                momentum2,
                strongDetonation,
                resonanceChase,
                dmiPdi,
                dmiMdi,
                dmiAdx,
                mainNetBuyWan: indicatorValues.mainNetBuyWan,
                mainDirection: indicatorValues.mainDirection,
                turnoverRatio: indicatorValues.turnoverRatio,
                expmaFast: indicatorValues.expmaFast,
                expmaSlow: indicatorValues.expmaSlow,
                bollingerUpper,
                bollingerMiddle,
                bollingerLower,
                tiandaoJinzuan: indicatorValues.tiandaoJinzuan,
                tiandaoJinniu: indicatorValues.tiandaoJinniu,
                tiandaoJinniu2: indicatorValues.tiandaoJinniu2,
                tiandaoBbi: indicatorValues.tiandaoBbi,
              });
            } else {
              const dataPoint = mainTradingDataRef.current.data.find((item: any) => item.date === param.time);
              const klinePoint = klineDataRef.current.find((item: any) => item.time === param.time);
              
              if (dataPoint && klinePoint) {
                let signal: 'buy' | 'sell' | undefined;
                const currentIndex = mainTradingDataRef.current.data.findIndex((item: any) => item.date === param.time);
                if (currentIndex !== -1) {
                  let lastSignalType: 'buy' | 'sell' | null = null;
                  let lastSignalIndex = -1;
                  for (let i = currentIndex; i >= 0; i--) {
                    const item = mainTradingDataRef.current.data[i];
                    if (item.buy_signal === 1) {
                      lastSignalType = 'buy';
                      lastSignalIndex = i;
                      break;
                    }
                    if (item.sell_signal === 1) {
                      lastSignalType = 'sell';
                      lastSignalIndex = i;
                      break;
                    }
                  }
                  
                  let nextOppositeSignalIndex = -1;
                  if (lastSignalType) {
                    for (let i = lastSignalIndex + 1; i < mainTradingDataRef.current.data.length; i++) {
                      const item = mainTradingDataRef.current.data[i];
                      const oppositeSignal = lastSignalType === 'buy' ? item.sell_signal : item.buy_signal;
                      if (oppositeSignal === 1) {
                        nextOppositeSignalIndex = i;
                        break;
                      }
                    }
                  }
                  
                  if (lastSignalType) {
                    if (nextOppositeSignalIndex === -1 || currentIndex < nextOppositeSignalIndex) {
                      signal = lastSignalType;
                    }
                  }
                }
                
                setCursorValues({
                  attackLine: dataPoint.attack_line,
                  tradingLine: dataPoint.trading_line,
                  defenseLine: dataPoint.defense_line,
                  signal: signal,
                  bankerControl,
                  mainCapitalAbsorption,
                  mainCost,
                  momentum2,
                  strongDetonation,
                  resonanceChase,
                  dmiPdi,
                  dmiMdi,
                  dmiAdx,
                  closePrice: klinePoint.close,
                  mainNetBuyWan: indicatorValues.mainNetBuyWan,
                  mainDirection: indicatorValues.mainDirection,
                  turnoverRatio: indicatorValues.turnoverRatio,
                  macdBarSum: indicatorValues.macdBarSum,
                  macdBarDiff: indicatorValues.macdBarDiff,
                  macdSignal: indicatorValues.macdSignal,
                  expmaFast: indicatorValues.expmaFast,
                  expmaSlow: indicatorValues.expmaSlow,
                  bollingerUpper,
                  bollingerMiddle,
                  bollingerLower,
                  tiandaoJinzuan: indicatorValues.tiandaoJinzuan,
                  tiandaoJinniu: indicatorValues.tiandaoJinniu,
                  tiandaoJinniu2: indicatorValues.tiandaoJinniu2,
                  tiandaoBbi: indicatorValues.tiandaoBbi,
                });
              } else {
                const klinePoint = klineDataRef.current.find((item: any) => item.time === param.time);
                setCursorValues({
                  bankerControl,
                  mainCapitalAbsorption,
                  mainCost,
                  momentum2,
                  strongDetonation,
                  resonanceChase,
                  dmiPdi,
                  dmiMdi,
                  dmiAdx,
                  closePrice: klinePoint?.close,
                  mainNetBuyWan: indicatorValues.mainNetBuyWan,
                  mainDirection: indicatorValues.mainDirection,
                  turnoverRatio: indicatorValues.turnoverRatio,
                  macdBarSum: indicatorValues.macdBarSum,
                  macdBarDiff: indicatorValues.macdBarDiff,
                  macdSignal: indicatorValues.macdSignal,
                  expmaFast: indicatorValues.expmaFast,
                  expmaSlow: indicatorValues.expmaSlow,
                  bollingerUpper,
                  bollingerMiddle,
                  bollingerLower,
                  tiandaoJinzuan: indicatorValues.tiandaoJinzuan,
                  tiandaoJinniu: indicatorValues.tiandaoJinniu,
                  tiandaoJinniu2: indicatorValues.tiandaoJinniu2,
                  tiandaoBbi: indicatorValues.tiandaoBbi,
                });
              }
            }
          }
        } finally {
          // 使用 setTimeout 避免立即释放标志，防止同一帧内的多次调用
          setTimeout(() => {
            isCrosshairUpdatingRef.current = false;
          }, 0);
        }
      };
      
      chart.subscribeCrosshairMove(handleCrosshairMove);
      
      // 保存引用以便清理
      (chart as any)._crosshairSubscription = handleCrosshairMove;

      // 清理函数
      return () => {
        resizeObserver.disconnect();
        if (chart) {
          if (timeSyncSubscription.current) {
            chart.timeScale().unsubscribeVisibleTimeRangeChange(timeSyncSubscription.current);
          }
          if ((chart as any)._crosshairSubscription) {
            chart.unsubscribeCrosshairMove((chart as any)._crosshairSubscription);
          }
          chart.remove();
        }
        isChartInitialized.current = false;
      };
    } catch (error) {
      console.error('Error initializing chart:', error);
    }
  }, []);

  // 1. 只更新 K 线数据（不碰主力操盘系列）
  useEffect(() => {
    console.log('Updating K-line data only:', visualizationData, 'refreshKey:', refreshKey);
    
    if (!visualizationData) {
      return;
    }
    
    if (!candlestickSeriesRef.current) {
      return;
    }

    if (!visualizationData.kline_data || visualizationData.kline_data.length === 0) {
      return;
    }

    try {
      const klineData = visualizationData.kline_data.map(item => ({
        time: item.date,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        originalItem: item,
      }));
      
      klineDataRef.current = klineData;
      
      // 保存数据范围
      if (klineData.length > 0) {
        earliestDateRef.current = klineData[0].time;
        latestDateRef.current = klineData[klineData.length - 1].time;
        
        // 计算价格范围
        let minPrice = Infinity;
        let maxPrice = -Infinity;
        klineData.forEach(item => {
          minPrice = Math.min(minPrice, item.low);
          maxPrice = Math.max(maxPrice, item.high);
        });
        // 给价格范围加一点padding
        const padding = (maxPrice - minPrice) * 0.1;
        setPriceRange({
          min: minPrice - padding,
          max: maxPrice + padding
        });
      }

      candlestickSeriesRef.current.setData(klineData);
      
      if (mainChartRef.current) {
        mainChartRef.current.timeScale().fitContent();
      }
      
      // K线数据加载后，等待图表完成价格轴计算再更新价格范围（含 scaleMargins）
      setTimeout(() => {
        handlePriceRangeChangeRef.current?.();
      }, 200);
    } catch (error) {
      console.error('Error updating K-line data:', error);
    }
  }, [visualizationData, refreshKey]);
  
  // 2. 只更新主力操盘系列（不碰 K 线）
  useEffect(() => {
    if (!mainChartRef.current || !visualizationData) return;
    
    try {
      // 清理旧的主力操盘系列
      if (attackLineSeriesRef.current) {
        mainChartRef.current.removeSeries(attackLineSeriesRef.current);
        attackLineSeriesRef.current = null;
      }
      if (tradingLineSeriesRef.current) {
        mainChartRef.current.removeSeries(tradingLineSeriesRef.current);
        tradingLineSeriesRef.current = null;
      }
      if (defenseLineSeriesRef.current) {
        mainChartRef.current.removeSeries(defenseLineSeriesRef.current);
        defenseLineSeriesRef.current = null;
      }
      if (buyPointSeriesRef.current) {
        mainChartRef.current.removeSeries(buyPointSeriesRef.current);
        buyPointSeriesRef.current = null;
      }
      if (sellPointSeriesRef.current) {
        mainChartRef.current.removeSeries(sellPointSeriesRef.current);
        sellPointSeriesRef.current = null;
      }
      
      const isMainTradingSelected = selectedIndicators.includes('main_trading');
      const mainTradingData = visualizationData.indicators.find(ind => ind.indicator_type === 'main_trading');
      
      // 保存主力操盘数据到 ref
      mainTradingDataRef.current = mainTradingData;
      
      if (isMainTradingSelected && mainTradingData) {
        
        // 攻击线
        const attackLineData = mainTradingData.data.map((item: any) => ({
          time: item.date,
          value: item.attack_line !== null && item.attack_line !== undefined 
            ? Number(item.attack_line.toFixed(2)) 
            : null,
        })).filter((d: any) => d.value !== null && d.value !== undefined);
        
        const attackSeries = mainChartRef.current.addSeries(lightweightCharts.LineSeries, {
          color: '#FF4444',
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        attackSeries.setData(attackLineData);
        attackLineSeriesRef.current = attackSeries;
        
        // 操盘线
        const tradingLineData = mainTradingData.data.map((item: any) => ({
          time: item.date,
          value: item.trading_line !== null && item.trading_line !== undefined 
            ? Number(item.trading_line.toFixed(2)) 
            : null,
        })).filter((d: any) => d.value !== null && d.value !== undefined);
        
        const tradingSeries = mainChartRef.current.addSeries(lightweightCharts.LineSeries, {
          color: '#FFAA00',
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        tradingSeries.setData(tradingLineData);
        tradingLineSeriesRef.current = tradingSeries;
        
        // 防守线
        const defenseLineData = mainTradingData.data.map((item: any) => ({
          time: item.date,
          value: item.defense_line !== null && item.defense_line !== undefined 
            ? Number(item.defense_line.toFixed(2)) 
            : null,
        })).filter((d: any) => d.value !== null && d.value !== undefined);
        
        const defenseSeries = mainChartRef.current.addSeries(lightweightCharts.LineSeries, {
          color: '#44AA44',
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        defenseSeries.setData(defenseLineData);
        defenseLineSeriesRef.current = defenseSeries;
      }
    } catch (error) {
      console.error('Error updating main trading:', error);
    }
  }, [selectedIndicators, visualizationData, refreshKey]);

  // 3. 更新 EXPMA 系列（主图叠加）
  useEffect(() => {
    if (!mainChartRef.current || !visualizationData) return;

    try {
      // 清理旧的 EXPMA 系列
      if (expmaFastSeriesRef.current) {
        mainChartRef.current.removeSeries(expmaFastSeriesRef.current);
        expmaFastSeriesRef.current = null;
      }
      if (expmaSlowSeriesRef.current) {
        mainChartRef.current.removeSeries(expmaSlowSeriesRef.current);
        expmaSlowSeriesRef.current = null;
      }

      const isExpmaSelected = selectedIndicators.includes('expma');
      const expmaData = visualizationData.indicators.find(
        (ind: any) => ind.indicator_type === 'expma'
      );

      // 保存 EXPMA 数据到 ref
      expmaDataRef.current = expmaData;

      if (isExpmaSelected && expmaData) {
        // 快速线（白色）
        const fastData = expmaData.data
          .map((item: any) => ({
            time: item.date,
            value: item.EXPMA_Fast != null ? Number(item.EXPMA_Fast.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (fastData.length > 0) {
          const fastSeries = mainChartRef.current!.addSeries(
            lightweightCharts.LineSeries, {
              color: '#FFFFFF',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: true,
              crosshairMarkerVisible: false,
            }
          );
          fastSeries.setData(fastData);
          expmaFastSeriesRef.current = fastSeries;
        }

        // 慢速线（金色）
        const slowData = expmaData.data
          .map((item: any) => ({
            time: item.date,
            value: item.EXPMA_Slow != null ? Number(item.EXPMA_Slow.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (slowData.length > 0) {
          const slowSeries = mainChartRef.current!.addSeries(
            lightweightCharts.LineSeries, {
              color: '#FFD700',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: true,
              crosshairMarkerVisible: false,
            }
          );
          slowSeries.setData(slowData);
          expmaSlowSeriesRef.current = slowSeries;
        }
      }
    } catch (error) {
      console.error('Error updating EXPMA:', error);
    }
  }, [selectedIndicators, visualizationData, refreshKey]);

  // 4. 更新天道系列（主图叠加）
  useEffect(() => {
    if (!mainChartRef.current || !visualizationData) return;

    try {
      // 清理旧的天道系列
      if (tiandaoBbiSeriesRef.current) {
        mainChartRef.current.removeSeries(tiandaoBbiSeriesRef.current);
        tiandaoBbiSeriesRef.current = null;
      }
      if (tiandaoJinzuanSeriesRef.current) {
        mainChartRef.current.removeSeries(tiandaoJinzuanSeriesRef.current);
        tiandaoJinzuanSeriesRef.current = null;
      }
      if (tiandaoJinniuSeriesRef.current) {
        mainChartRef.current.removeSeries(tiandaoJinniuSeriesRef.current);
        tiandaoJinniuSeriesRef.current = null;
      }
      if (tiandaoJinniu2SeriesRef.current) {
        mainChartRef.current.removeSeries(tiandaoJinniu2SeriesRef.current);
        tiandaoJinniu2SeriesRef.current = null;
      }

      const isTiandaoSelected = selectedIndicators.includes('tiandao');
      const tiandaoIndicator = visualizationData.indicators.find(
        (ind: any) => ind.indicator_type === 'tiandao'
      );

      // 保存天道数据到 ref
      tiandaoDataRef.current = tiandaoIndicator;

      // 清理旧的买入信号标记
      if (tiandaoMarkersRef.current) {
        tiandaoMarkersRef.current.setMarkers([]);
        tiandaoMarkersRef.current = null;
      }

      if (isTiandaoSelected && tiandaoIndicator && tiandaoIndicator.data && tiandaoIndicator.data.length > 0) {
        // td_jinzuan 金钻趋势（金色细线，核心趋势线）
        const jinzuanData = tiandaoIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.td_jinzuan != null ? Number(item.td_jinzuan.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (jinzuanData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FF0000',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(jinzuanData);
          tiandaoJinzuanSeriesRef.current = series;
        }

        // td_jinniu 金牛（红色细线，通道上轨）
        const jinniuData = tiandaoIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.td_jinniu != null ? Number(item.td_jinniu.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (jinniuData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FFFF00',
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(jinniuData);
          tiandaoJinniuSeriesRef.current = series;
        }

        // td_jinniu2 金牛2（绿色细线，慢速跟随）
        const jinniu2Data = tiandaoIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.td_jinniu2 != null ? Number(item.td_jinniu2.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (jinniu2Data.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#00FFFF',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(jinniu2Data);
          tiandaoJinniu2SeriesRef.current = series;
        }

        // td_bbi BBI 线
        const bbiData = tiandaoIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.td_bbi != null ? Number(item.td_bbi.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (bbiData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FFFFFF',
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(bbiData);
          tiandaoBbiSeriesRef.current = series;
        }

        // 处理买入信号标记（td_xg 和 td_xg2）
        if (candlestickSeriesRef.current) {
          const markers: any[] = [];

          tiandaoIndicator.data.forEach((item: any) => {
            // td_xg: ▲买入（红色箭头，K线下方）
            if (item.td_xg === 1) {
              markers.push({
                time: item.date,
                position: 'belowBar',
                shape: 'arrowUp',
                color: '#FF0000',
                text: '买入',
                size: 1,
              });
            }
            // td_xg2: ↖金钻起涨（品红色箭头，K线下方）
            if (item.td_xg2 === 1) {
              markers.push({
                time: item.date,
                position: 'belowBar',
                shape: 'arrowUp',
                color: '#FF00FF',
                text: '金钻',
                size: 1,
              });
            }
          });

          tiandaoMarkersRef.current = lightweightCharts.createSeriesMarkers(
            candlestickSeriesRef.current,
            markers,
          );
        }
      }
    } catch (error) {
      console.error('Error updating Tiandao:', error);
    }
  }, [selectedIndicators, visualizationData, refreshKey]);

  // 5. 更新布林带系列（主图叠加）
  useEffect(() => {
    if (!mainChartRef.current || !visualizationData) return;

    try {
      // 清理旧的布林带系列
      if (bollingerUpperSeriesRef.current) {
        mainChartRef.current.removeSeries(bollingerUpperSeriesRef.current);
        bollingerUpperSeriesRef.current = null;
      }
      if (bollingerMiddleSeriesRef.current) {
        mainChartRef.current.removeSeries(bollingerMiddleSeriesRef.current);
        bollingerMiddleSeriesRef.current = null;
      }
      if (bollingerLowerSeriesRef.current) {
        mainChartRef.current.removeSeries(bollingerLowerSeriesRef.current);
        bollingerLowerSeriesRef.current = null;
      }

      const isBollingerSelected = selectedIndicators.includes('bollinger');
      const bollingerIndicator = visualizationData.indicators.find(
        (ind: any) => ind.indicator_type === 'bollinger'
      );

      // 保存布林带数据到 ref
      bollingerDataRef.current = bollingerIndicator;

      if (isBollingerSelected && bollingerIndicator && bollingerIndicator.data && bollingerIndicator.data.length > 0) {
        // 上轨（橙色实线）
        const upperData = bollingerIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.bollinger_upper != null ? Number(item.bollinger_upper.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (upperData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FF8C00',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(upperData);
          bollingerUpperSeriesRef.current = series;
        }

        // 中轨（白色虚线）
        const middleData = bollingerIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.bollinger_middle != null ? Number(item.bollinger_middle.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (middleData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FFFFFF',
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(middleData);
          bollingerMiddleSeriesRef.current = series;
        }

        // 下轨（橙色实线）
        const lowerData = bollingerIndicator.data
          .map((item: any) => ({
            time: item.date,
            value: item.bollinger_lower != null ? Number(item.bollinger_lower.toFixed(2)) : null,
          }))
          .filter((d: any) => d.value !== null && d.value !== undefined);

        if (lowerData.length > 0) {
          const series = mainChartRef.current!.addSeries(lightweightCharts.LineSeries, {
            color: '#FF8C00',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          series.setData(lowerData);
          bollingerLowerSeriesRef.current = series;
        }
      }
    } catch (error) {
      console.error('Error updating Bollinger:', error);
    }
  }, [selectedIndicators, visualizationData, refreshKey]);

  // 更新子图数据
  useEffect(() => {
    // 清理旧的子图
    Object.keys(subChartResizeObservers.current).forEach(indicatorId => {
      const observer = subChartResizeObservers.current[indicatorId];
      if (observer) {
        observer.disconnect();
      }
    });
    subChartResizeObservers.current = {};

    // 清理子图十字线订阅
    Object.keys(subChartCrosshairSubscriptions.current).forEach(indicatorId => {
      const chart = subChartRefs.current[indicatorId];
      const subscription = subChartCrosshairSubscriptions.current[indicatorId];
      if (chart && subscription) {
        try {
          chart.unsubscribeCrosshairMove(subscription);
        } catch (e) {
          console.warn('Failed to unsubscribe crosshair:', e);
        }
      }
    });

    Object.values(subChartRefs.current).forEach(chart => {
      if (chart) chart.remove();
    });
    subChartRefs.current = {};
    subChartSeriesRefs.current = {};
    subChartCrosshairSubscriptions.current = {};
    
    // 还需要一个存储子图时间轴订阅的对象
    (window as any)._subChartTimeSubscriptions = {};

    if (!visualizationData) return;

    // 辅助函数：根据主图时间范围过滤数据
    const filterDataByTimeRange = (data: any[], timeField: string = 'date') => {
      if (!earliestDateRef.current || !latestDateRef.current || !data || data.length === 0) {
        return data;
      }
      
      const toTimestamp = (time: any): number => {
        if (typeof time === 'string' && time.includes('-')) {
          return new Date(time).getTime() / 1000;
        }
        return Number(time);
      };
      
      const earliestTs = toTimestamp(earliestDateRef.current);
      const latestTs = toTimestamp(latestDateRef.current);
      
      return data.filter(item => {
        const itemTime = item[timeField] || item.time;
        const itemTs = toTimestamp(itemTime);
        return itemTs >= earliestTs && itemTs <= latestTs;
      });
    };

    // 保存所有指标数据到 ref，用于十字线值查找（使用过滤后的数据）
    indicatorsDataRef.current = {};
    visualizationData.indicators.forEach(ind => {
      indicatorsDataRef.current[ind.indicator_type] = {
        ...ind,
        data: filterDataByTimeRange(ind.data, 'date')
      };
    });

    // 第一步：为每个选中的指标创建子图（跳过主力操盘，它在主图显示）
    const createdCharts: Array<{ id: string; chart: any }> = [];
    
    selectedIndicators.forEach(indicatorId => {
      if (indicatorId === 'main_trading' || indicatorId === 'expma' || indicatorId === 'tiandao' || indicatorId === 'bollinger') return;
      
      // 成交量不需要从指标数据中获取，它直接使用K线数据
      if (indicatorId !== 'volume') {
        const indicatorData = visualizationData.indicators.find(ind => ind.indicator_type === indicatorId);
        if (!indicatorData || !indicatorData.data || indicatorData.data.length === 0) return;
      }

      const containerRef = subChartContainerRefs.current[indicatorId];
      if (!containerRef) return;

      try {
        // 获取指标数据（成交量不需要）
        const indicatorData = indicatorId !== 'volume' 
          ? visualizationData.indicators.find(ind => ind.indicator_type === indicatorId)
          : null;
          
        // 创建子图
        const chart = lightweightCharts.createChart(containerRef, {
          width: containerRef.clientWidth,
          height: SUBCHART_HEIGHT,
          layout: {
            background: { type: 'solid', color: '#1a1a2e' } as any,
            textColor: '#d1d4dc',
            attributionLogo: false,
          },
          grid: {
            vertLines: { color: '#2b2b43' },
            horzLines: { color: '#2b2b43' },
          },
          crosshair: {
            mode: 1, // 0=Normal, 1=Magnet, 2=Hidden
            vertLine: {
              color: '#9B7DFF',
              width: 1,
              style: 2,
            },
            horzLine: {
              color: '#9B7DFF',
              width: 1,
              style: 2,
            },
          },
          localization: {
            timeFormatter: (time: any) => {
              let date: Date;
              if (typeof time === 'string' && time.includes('-')) {
                date = new Date(time);
              } else {
                date = new Date(time * 1000);
              }
              const year = date.getFullYear();
              const month = String(date.getMonth() + 1).padStart(2, '0');
              const day = String(date.getDate()).padStart(2, '0');
              return `${year}-${month}-${day}`;
            },
          },
          timeScale: {
            timeVisible: true,
            secondsVisible: false,
            tickMarkFormatter: (time: any) => {
              let date: Date;
              if (typeof time === 'string' && time.includes('-')) {
                date = new Date(time);
              } else {
                date = new Date(time * 1000);
              }
              const year = date.getFullYear();
              const month = String(date.getMonth() + 1).padStart(2, '0');
              const day = String(date.getDate()).padStart(2, '0');
              return `${year}-${month}-${day}`;
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
            axisPressedMouseMove: {
              time: false,
              price: false,
            },
            axisDoubleClickReset: {
              time: true,
              price: true,
            },
          },
        });

        subChartRefs.current[indicatorId] = chart;
        createdCharts.push({ id: indicatorId, chart });

        // 根据指标类型渲染不同的内容
        if (indicatorId === 'resonance_chase') {
          // 共振追涨：启用 Y 轴刻度显示，但只显示共振柱高度
          chart.priceScale('right').applyOptions({
            visible: true,
          });
        }
        
        if (indicatorId === 'volume') {
          // 成交量 - 显示柱状图，颜色与K线一致
          // 先过滤数据，确保时间范围与主图对齐
          const filteredKlineData = filterDataByTimeRange(visualizationData.kline_data, 'date');
          const volumeData = filteredKlineData.map((item: any) => {
            let color = '#666666';
            if (item.close > item.open) {
              color = '#FF4444';
            } else if (item.close < item.open) {
              color = '#44AA44';
            }
            return {
              time: item.date,
              value: item.volume || 0,
              color: color,
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          const histogramSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
            color: '#666666',
            priceFormat: {
              type: 'volume',
            },
            crosshairMarkerVisible: false,
          } as any);
          
          const histogramData = volumeData.map((d: any) => ({
            time: d.time,
            value: d.value,
            color: d.color,
          }));
          
          histogramSeries.setData(histogramData);
          subChartSeriesRefs.current[indicatorId] = histogramSeries;

          // 五日平均成交量线（白色）
          const volume5MA = calculateVolumeMA(volumeData, 5);
          const vol5maSeries = chart.addSeries(lightweightCharts.LineSeries, {
            color: '#FFFFFF',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
            priceFormat: { type: 'volume' },
          });
          vol5maSeries.setData(volume5MA);

          // 十日平均成交量线（金色）
          const volume10MA = calculateVolumeMA(volumeData, 10);
          const vol10maSeries = chart.addSeries(lightweightCharts.LineSeries, {
            color: '#FFD700',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
            priceFormat: { type: 'volume' },
          });
          vol10maSeries.setData(volume10MA);

        } else if (indicatorId === 'banker_control' && indicatorData) {
          // 庄家控盘 - 显示能量柱，分三段颜色
          // 先过滤数据，确保时间范围与主图对齐
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const barData = filteredIndicatorData.map((item: any) => {
            const value = Number((item.control_degree || 0).toFixed(2));
            let color = '#666666';
            if (value >= 80) {
              color = '#AA44FF';
            } else if (value >= 60) {
              color = '#FF4444';
            } else if (value >= 50) {
              color = '#FFAA00';
            }
            return {
              time: item.date,
              value: value,
              color: color,
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          const histogramSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
            color: '#FFAA00',
            priceFormat: {
              type: 'price',
              precision: 2,
            },
            priceScaleId: 'right',
            crosshairMarkerVisible: false,
          } as any);
          
          // 将庄家控盘的价格范围设为50-100（零轴为50）
          chart.priceScale('right').applyOptions({
            autoScale: false,
          });
          chart.priceScale('right').setVisibleRange({
            from: 50,
            to: 100,
          });
          
          const histogramData = barData.map((d: any) => ({
            time: d.time,
            value: d.value,
            color: d.color,
          }));
          
          histogramSeries.setData(histogramData);
          subChartSeriesRefs.current[indicatorId] = histogramSeries;

          chart.timeScale().fitContent();

        } else if (indicatorId === 'main_capital_absorption' && indicatorData) {
          // 主力吸筹 - 显示柱状图
          // 先过滤数据，确保时间范围与主图对齐，值小于1.01的柱体高度设为0
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const barData = filteredIndicatorData.map((item: any) => {
            const rawValue = item.main_capital_absorption || 0;
            const value = Math.abs(rawValue) < 1.01 ? 0 : Number(rawValue.toFixed(2));
            return {
              time: item.date,
              value: value,
              color: value >= 0 ? '#AA44FF' : '#44AA44',
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          const histogramSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
            color: '#AA44FF',
            priceFormat: {
              type: 'volume',
            },
            crosshairMarkerVisible: false,
          } as any);
          
          const histogramData = barData.map((d: any) => ({
            time: d.time,
            value: d.value,
            color: d.color,
          }));
          
          histogramSeries.setData(histogramData);
          subChartSeriesRefs.current[indicatorId] = histogramSeries;

        } else if (indicatorId === 'main_cost' && indicatorData) {
          // 主力成本 - 显示成本曲线
          // 先过滤数据，确保时间范围与主图对齐
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          
          // 主力成本线（红实线）
          const mainCostLineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.main_cost || item.cost || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          if (mainCostLineData.length > 0) {
            const mainCostLineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FF4444',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            mainCostLineSeries.setData(mainCostLineData);
            subChartSeriesRefs.current[indicatorId] = mainCostLineSeries;
          }
          
          // 成交均价线（黄虚线）
          const avgPriceLineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.avg_price || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          if (avgPriceLineData.length > 0) {
            const avgPriceLineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FF9900',
              lineWidth: 1,
              lineStyle: 2,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            avgPriceLineSeries.setData(avgPriceLineData);
          }

        } else if (indicatorId === 'momentum_2' && indicatorData) {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const barData = filteredIndicatorData.map((item: any) => {
            let value = 0;
            let color = 'transparent';
            
            if (item.strong_momentum !== 0) {
              value = Number(item.strong_momentum.toFixed(2));
              color = '#FF4444';
            } else if (item.medium_momentum !== 0) {
              value = Number(item.medium_momentum.toFixed(2));
              color = '#FFAA00';
            } else if (item.no_momentum !== 0) {
              value = Number(item.no_momentum.toFixed(2));
              color = '#44AA44';
            } else if (item.recovery_momentum !== 0) {
              value = Number(item.recovery_momentum.toFixed(2));
              color = '#4477FF';
            }
            
            return {
              time: item.date,
              value: value,
              color: color,
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          const histogramSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
            color: '#FF4444',
            priceFormat: {
              type: 'price',
              precision: 2,
            },
            crosshairMarkerVisible: false,
          } as any);
          
          const histogramData = barData.map((d: any) => ({
            time: d.time,
            value: d.value,
            color: d.color,
          }));
          
          histogramSeries.setData(histogramData);
          subChartSeriesRefs.current[indicatorId] = histogramSeries;

        } else if (indicatorId === 'strong_detonation' && indicatorData) {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          
          const bullLineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.bull_line || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);
          
          const bearLineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.bear_line || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);
          
          const marketMidlineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.market_midline || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);
          
          // 红色箱体K线
          const redBoxData = filteredIndicatorData
            .filter((item: any) => item.red_box_open !== null && item.red_box_open !== undefined && !isNaN(item.red_box_open))
            .map((item: any) => ({
              time: item.date,
              open: Number(item.red_box_open.toFixed(2)),
              high: Number(item.red_box_high.toFixed(2)),
              low: Number(item.red_box_low.toFixed(2)),
              close: Number(item.red_box_close.toFixed(2)),
            }));

          if (redBoxData.length > 0) {
            const redBoxSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
              upColor: '#FF4444',
              downColor: '#FF4444',
              borderDownColor: '#FF4444',
              borderUpColor: '#FF4444',
              wickDownColor: '#FF4444',
              wickUpColor: '#FF4444',
              priceLineVisible: false,
              lastValueVisible: false,
            });
            redBoxSeries.setData(redBoxData);
            subChartSeriesRefs.current[indicatorId] = redBoxSeries;
          }

          // 紫色箱体K线
          const purpleBoxData = filteredIndicatorData
            .filter((item: any) => item.purple_box_open !== null && item.purple_box_open !== undefined && !isNaN(item.purple_box_open))
            .map((item: any) => ({
              time: item.date,
              open: Number(item.purple_box_open.toFixed(2)),
              high: Number(item.purple_box_high.toFixed(2)),
              low: Number(item.purple_box_low.toFixed(2)),
              close: Number(item.purple_box_close.toFixed(2)),
            }));

          if (purpleBoxData.length > 0) {
            const purpleBoxSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
              upColor: '#AA44FF',
              downColor: '#AA44FF',
              borderDownColor: '#AA44FF',
              borderUpColor: '#AA44FF',
              wickDownColor: '#AA44FF',
              wickUpColor: '#AA44FF',
              priceLineVisible: false,
              lastValueVisible: false,
            });
            purpleBoxSeries.setData(purpleBoxData);
            if (!subChartSeriesRefs.current[indicatorId]) {
              subChartSeriesRefs.current[indicatorId] = purpleBoxSeries;
            }
          }

          if (bullLineData.length > 0) {
            const bullLineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FFAA00',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            bullLineSeries.setData(bullLineData);
            if (!subChartSeriesRefs.current[indicatorId]) {
              subChartSeriesRefs.current[indicatorId] = bullLineSeries;
            }
          }
          
          if (bearLineData.length > 0) {
            const bearLineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#44AA44',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            bearLineSeries.setData(bearLineData);
          }
          
          if (marketMidlineData.length > 0) {
            const marketMidlineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FF4444',
              lineWidth: 1,
              lineStyle: 2,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            marketMidlineSeries.setData(marketMidlineData);
          }

        } else if (indicatorId === 'resonance_chase' && indicatorData) {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          
          const midBarData = filteredIndicatorData.map((item: any) => {
            const value = item.mid_bullish ? 1 : -1;
            const color = item.mid_bullish ? '#FF4444' : '#44AA44';
            return {
              time: item.date,
              value: value,
              color: color,
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          const out1Data = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.OUT1 || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          const out2Data = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.OUT2 || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          // 共振柱高度数据
          const resonanceHeightData = filteredIndicatorData.map((item: any) => {
            const originalHeight = Math.abs(item.OUT1 || 0);
            const height = Math.max(0, originalHeight - 0.5);
            return {
              time: item.date,
              value: Math.floor(height), // 取整
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined);

          // 共振柱：K线箱体显示，底部固定为0.5，顶部为原高度-0.5
          const resonanceBoxData = filteredIndicatorData
            .filter((item: any) => item.mid_bullish && item.resonance)
            .map((item: any) => {
              const baseValue = 0.5;
              const originalHeight = Math.abs(item.OUT1 || 0);
              const topValue = originalHeight - 0.5;
              const finalTopValue = topValue > baseValue ? topValue : baseValue + 0.1;
              
              return {
                time: item.date,
                open: Number(baseValue.toFixed(2)),
                high: Number(finalTopValue.toFixed(2)),
                low: Number(baseValue.toFixed(2)),
                close: Number(finalTopValue.toFixed(2)),
              };
            });

          // 首先添加共振柱高度的 series（作为主 series，控制十字线价格格式和 Last Value）
          if (resonanceHeightData.length > 0) {
            const lastValueLineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: 'transparent',
              priceLineVisible: false,
              lastValueVisible: true,
              crosshairMarkerVisible: false,
              priceFormat: {
                type: 'custom',
                formatter: (price: number) => {
                  return Math.round(price).toString();
                },
              },
            });
            lastValueLineSeries.setData(resonanceHeightData);
            subChartSeriesRefs.current[indicatorId] = lastValueLineSeries;
          }

          // 然后画中线柱（在底部）
          if (midBarData.length > 0) {
            const midHistogramSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
              color: '#FF4444',
              priceFormat: {
                type: 'price',
                precision: 0,
              },
              crosshairMarkerVisible: false,
              priceLineVisible: false,
              lastValueVisible: false,
            } as any);
            
            const midHistogramData = midBarData.map((d: any) => ({
              time: d.time,
              value: d.value,
              color: d.color,
            }));
            
            midHistogramSeries.setData(midHistogramData);
          }

          // 再画共振柱（在中间，使用K线箱体）
          if (resonanceBoxData.length > 0) {
            const resonanceBoxSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
              upColor: '#AA44FF',
              downColor: '#AA44FF',
              borderDownColor: '#AA44FF',
              borderUpColor: '#AA44FF',
              wickDownColor: '#AA44FF',
              wickUpColor: '#AA44FF',
              priceLineVisible: false,
              lastValueVisible: false,
            });
            resonanceBoxSeries.setData(resonanceBoxData);
          }

          // 最后画OUT1线和OUT2线（在顶部）
          if (out1Data.length > 0) {
            const out1LineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FFAA00',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
              autoscaleInfoProvider: () => null,
            });
            out1LineSeries.setData(out1Data);
          }

          if (out2Data.length > 0) {
            const out2LineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#4477FF',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
              autoscaleInfoProvider: () => null,
            });
            out2LineSeries.setData(out2Data);
          }

        } else if (indicatorId === 'dmi' && indicatorData) {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');

          const pdiData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.PDI || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          const mdiData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.MDI || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          const adxData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.ADX || 0).toFixed(2)),
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          if (pdiData.length > 0) {
            const pdiSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FF4444',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: true,
              crosshairMarkerVisible: false,
              priceFormat: {
                type: 'price',
                precision: 2,
              },
            });
            pdiSeries.setData(pdiData);
            subChartSeriesRefs.current[indicatorId] = pdiSeries;
          }

          if (mdiData.length > 0) {
            const mdiSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#44AA44',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
              priceFormat: {
                type: 'price',
                precision: 2,
              },
            });
            mdiSeries.setData(mdiData);
          }

          if (adxData.length > 0) {
            const adxSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#AA44FF',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false,
              priceFormat: {
                type: 'price',
                precision: 2,
              },
            });
            adxSeries.setData(adxData);
          }

        } else if (indicatorId === 'macd' && indicatorData) {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');

          const difData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.DIF || 0).toFixed(4)),
          })).filter((d: any) => d.value !== null && d.value !== undefined && !isNaN(d.value));

          const deaData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: Number((item.DEA || 0).toFixed(4)),
          })).filter((d: any) => d.value !== null && d.value !== undefined && !isNaN(d.value));

          const barData = filteredIndicatorData.map((item: any) => {
            const val = Number((item.MACD_Bar || 0).toFixed(4));
            return {
              time: item.date,
              value: val,
              color: val >= 0 ? '#FF4444' : '#44FF44',
            };
          }).filter((d: any) => d.value !== null && d.value !== undefined && !isNaN(d.value));

          if (difData.length > 0) {
            const difSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FFFFFF',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            difSeries.setData(difData);
            subChartSeriesRefs.current[indicatorId] = difSeries;
          }

          if (deaData.length > 0) {
            const deaSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FFD700',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            });
            deaSeries.setData(deaData);
          }

          if (barData.length > 0) {
            const barSeries = chart.addSeries(lightweightCharts.HistogramSeries, {
              priceFormat: { type: 'volume' },
            } as any);
            barSeries.setData(barData);
          }

        }
        const resizeObserver = new ResizeObserver(entries => {
          if (entries.length === 0) return;
          const { width } = entries[0].contentRect;
          if (chart) {
            try {
              chart.applyOptions({ width, height: SUBCHART_HEIGHT });
            } catch (e) {
              console.warn('Chart already disposed:', e);
            }
          }
        });
        resizeObserver.observe(containerRef);
        subChartResizeObservers.current[indicatorId] = resizeObserver;
        
        // 添加子图时间轴变化监听，实现双向联动
        const handleSubChartTimeScaleChange = () => {
          if (!chart || isTimeRangeUpdatingRef.current) return;
          
          try {
            isTimeRangeUpdatingRef.current = true;
            const timeRange = chart.timeScale().getVisibleRange();
            if (timeRange && timeRange.from && timeRange.to && mainChartRef.current) {
              // 直接使用时间范围，不限制，让缩放更自由
              // 同步到主图
              mainChartRef.current.timeScale().setVisibleRange(timeRange);
              
              // 同步到其他子图
              Object.keys(subChartRefs.current).forEach(otherIndicatorId => {
                if (otherIndicatorId !== indicatorId && subChartRefs.current[otherIndicatorId]) {
                  try {
                    subChartRefs.current[otherIndicatorId]!.timeScale().setVisibleRange(timeRange);
                  } catch (e) {
                    console.warn('Failed to sync to other subchart:', e);
                  }
                }
              });
            }
          } catch (e) {
            console.warn('Failed to handle subchart time scale change:', e);
          } finally {
            setTimeout(() => {
              isTimeRangeUpdatingRef.current = false;
            }, 0);
          }
        };
        
        chart.timeScale().subscribeVisibleTimeRangeChange(handleSubChartTimeScaleChange);
        
        // 保存订阅以便清理
        (window as any)._subChartTimeSubscriptions = (window as any)._subChartTimeSubscriptions || {};
        (window as any)._subChartTimeSubscriptions[indicatorId] = handleSubChartTimeScaleChange;
        
        // 为子图添加十字线移动监听，实现双向同步
        const handleSubChartCrosshairMove = (param: any) => {
          if (isCrosshairUpdatingRef.current) {
            return;
          }
          
          isCrosshairUpdatingRef.current = true;
          
          try {
            currentCrosshairTimeRef.current = param.time;
            
            // 获取所有指标在当前时间点的值
            const indicatorValues = getAllIndicatorValuesAtTime(param.time);
            
            // 同步十字线位置到主图
            if (param.time && mainChartRef.current && candlestickSeriesRef.current && klineDataRef.current) {
              const klinePoint = klineDataRef.current.find((item: any) => item.time === param.time);
              if (klinePoint) {
                try {
                  mainChartRef.current.setCrosshairPosition(klinePoint.close, param.time, candlestickSeriesRef.current);
                } catch (e) {
                  console.warn('Failed to sync crosshair to main chart:', e);
                }
              }
            }
            
            // 同步十字线位置到其他子图
            if (param.time) {
              Object.entries(subChartRefs.current).forEach(([otherIndicatorId, otherSubChart]) => {
                if (otherIndicatorId !== indicatorId && otherSubChart && subChartSeriesRefs.current[otherIndicatorId]) {
                  try {
                    const otherSeries = subChartSeriesRefs.current[otherIndicatorId];
                    
                    let otherPrice: number | undefined;
                    if (otherIndicatorId === 'volume') {
                      const otherKlinePoint = klineDataRef.current?.find((item: any) => item.time === param.time);
                      if (otherKlinePoint && otherKlinePoint.originalItem) {
                        otherPrice = otherKlinePoint.originalItem.volume;
                      } else if (visualizationData && visualizationData.kline_data) {
                        const originalDataPoint = visualizationData.kline_data.find((item: any) => item.date === param.time);
                        otherPrice = originalDataPoint?.volume;
                      }
                    } else {
                      const otherIndicatorData = indicatorsDataRef.current[otherIndicatorId];
                      if (otherIndicatorData && otherIndicatorData.data) {
                        const otherDataPoint = otherIndicatorData.data.find((item: any) => {
                          const itemDate = item.date || item.time;
                          return itemDate === param.time;
                        });
                        
                        if (otherDataPoint) {
                          if (otherIndicatorId === 'banker_control') {
                            otherPrice = otherDataPoint.control_degree;
                          } else if (otherIndicatorId === 'main_capital_absorption') {
                            otherPrice = otherDataPoint.main_capital_absorption;
                          } else if (otherIndicatorId === 'main_cost') {
                            otherPrice = otherDataPoint.main_cost || otherDataPoint.cost;
                          } else if (otherIndicatorId === 'momentum_2') {
                            otherPrice = otherDataPoint.strong_momentum || otherDataPoint.medium_momentum || 
                                         otherDataPoint.no_momentum || otherDataPoint.recovery_momentum;
                          } else if (otherIndicatorId === 'strong_detonation') {
                            otherPrice = otherDataPoint.bull_line;
                          } else if (otherIndicatorId === 'resonance_chase') {
                            if (otherDataPoint.resonance) {
                              const originalHeight = Math.abs(otherDataPoint.OUT1 || 0);
                              const topValue = originalHeight - 0.5;
                              otherPrice = topValue > 0 ? topValue : undefined;
                            }
                          } else if (otherIndicatorId === 'dmi') {
                            if (otherDataPoint.PDI !== undefined && otherDataPoint.PDI !== null) {
                              otherPrice = otherDataPoint.PDI;
                            }
                          } else if (otherIndicatorId === 'macd') {
                            if (otherDataPoint.DIF !== undefined && otherDataPoint.DIF !== null && !isNaN(otherDataPoint.DIF)) {
                              otherPrice = otherDataPoint.DIF;
                            }
                          }
                        }
                      }
                    }
                        
                    if (otherPrice !== undefined && !isNaN(otherPrice)) {
                      otherSubChart.setCrosshairPosition(otherPrice, param.time, otherSeries);
                    }
                  } catch (e) {
                    console.warn('Failed to sync crosshair to other subchart:', e);
                  }
                }
              });
            }
            
            // 同步到主图
            if (param.time && mainChartRef.current) {
              // 同时更新主图的光标值显示
              if (mainTradingDataRef.current && klineDataRef.current) {
                const dataPoint = mainTradingDataRef.current.data.find((item: any) => item.date === param.time);
                const klinePoint = klineDataRef.current.find((item: any) => item.time === param.time);
                
                if (dataPoint && klinePoint) {
                  let signal: 'buy' | 'sell' | undefined;
                  const currentIndex = mainTradingDataRef.current.data.findIndex((item: any) => item.date === param.time);
                  if (currentIndex !== -1) {
                    let lastSignalType: 'buy' | 'sell' | null = null;
                    let lastSignalIndex = -1;
                    for (let i = currentIndex; i >= 0; i--) {
                      const item = mainTradingDataRef.current.data[i];
                      if (item.buy_signal === 1) {
                        lastSignalType = 'buy';
                        lastSignalIndex = i;
                        break;
                      }
                      if (item.sell_signal === 1) {
                        lastSignalType = 'sell';
                        lastSignalIndex = i;
                        break;
                      }
                    }
                    
                    let nextOppositeSignalIndex = -1;
                    if (lastSignalType) {
                      for (let i = lastSignalIndex + 1; i < mainTradingDataRef.current.data.length; i++) {
                        const item = mainTradingDataRef.current.data[i];
                        const oppositeSignal = lastSignalType === 'buy' ? item.sell_signal : item.buy_signal;
                        if (oppositeSignal === 1) {
                          nextOppositeSignalIndex = i;
                          break;
                        }
                      }
                    }
                    
                    if (lastSignalType) {
                      if (nextOppositeSignalIndex === -1 || currentIndex < nextOppositeSignalIndex) {
                        signal = lastSignalType;
                      }
                    }
                  }
                  
                  setCursorValues({
                    attackLine: dataPoint.attack_line,
                    tradingLine: dataPoint.trading_line,
                    defenseLine: dataPoint.defense_line,
                    signal: signal,
                    bankerControl: indicatorValues.bankerControl,
                    mainCapitalAbsorption: indicatorValues.mainCapitalAbsorption,
                    mainCost: indicatorValues.mainCost,
                    momentum2: indicatorValues.momentum2,
                    strongDetonation: indicatorValues.strongDetonation,
                    resonanceChase: indicatorValues.resonanceChase,
                    dmiPdi: indicatorValues.dmiPdi,
                    dmiMdi: indicatorValues.dmiMdi,
                    dmiAdx: indicatorValues.dmiAdx,
                    closePrice: klinePoint.close,
                    mainNetBuyWan: indicatorValues.mainNetBuyWan,
                    mainDirection: indicatorValues.mainDirection,
                    turnoverRatio: indicatorValues.turnoverRatio,
                    macdBarSum: indicatorValues.macdBarSum,
                    macdBarDiff: indicatorValues.macdBarDiff,
                    macdSignal: indicatorValues.macdSignal,
                    expmaFast: indicatorValues.expmaFast,
                    expmaSlow: indicatorValues.expmaSlow,
                    tiandaoJinzuan: indicatorValues.tiandaoJinzuan,
                    tiandaoJinniu: indicatorValues.tiandaoJinniu,
                    tiandaoJinniu2: indicatorValues.tiandaoJinniu2,
                    tiandaoBbi: indicatorValues.tiandaoBbi,
                  });
                } else {
                  // 即使没有主图数据，也要更新所有子图指标的值
                  const klinePoint = klineDataRef.current.find((item: any) => item.time === param.time);
                  setCursorValues({
                    bankerControl: indicatorValues.bankerControl,
                    mainCapitalAbsorption: indicatorValues.mainCapitalAbsorption,
                    mainCost: indicatorValues.mainCost,
                    momentum2: indicatorValues.momentum2,
                    strongDetonation: indicatorValues.strongDetonation,
                    resonanceChase: indicatorValues.resonanceChase,
                    dmiPdi: indicatorValues.dmiPdi,
                    dmiMdi: indicatorValues.dmiMdi,
                    dmiAdx: indicatorValues.dmiAdx,
                    closePrice: klinePoint?.close,
                    mainNetBuyWan: indicatorValues.mainNetBuyWan,
                    mainDirection: indicatorValues.mainDirection,
                    turnoverRatio: indicatorValues.turnoverRatio,
                    macdBarSum: indicatorValues.macdBarSum,
                    macdBarDiff: indicatorValues.macdBarDiff,
                    macdSignal: indicatorValues.macdSignal,
                    expmaFast: indicatorValues.expmaFast,
                    expmaSlow: indicatorValues.expmaSlow,
                    tiandaoJinzuan: indicatorValues.tiandaoJinzuan,
                    tiandaoJinniu: indicatorValues.tiandaoJinniu,
                    tiandaoJinniu2: indicatorValues.tiandaoJinniu2,
                    tiandaoBbi: indicatorValues.tiandaoBbi,
                  });
                }
              }
            }
          } finally {
            setTimeout(() => {
              isCrosshairUpdatingRef.current = false;
            }, 0);
          }
        };
        
        chart.subscribeCrosshairMove(handleSubChartCrosshairMove);
        subChartCrosshairSubscriptions.current[indicatorId] = handleSubChartCrosshairMove;

      } catch (error) {
        console.error(`Error initializing subchart for ${indicatorId}:`, error);
      }
    });

    // 第二步：延迟让所有图表准备好后，调用 fitContent 并同步
    setTimeout(() => {
      try {
        // 先让主图和所有子图都 fitContent
        if (mainChartRef.current) {
          mainChartRef.current.timeScale().fitContent();
        }
        Object.values(subChartRefs.current).forEach(chart => {
          if (chart) {
            chart.timeScale().fitContent();
          }
        });
        
        // 再从主图获取时间范围并同步到子图
        if (mainChartRef.current) {
          const timeRange = mainChartRef.current.timeScale().getVisibleRange();
          if (timeRange && timeRange.from && timeRange.to) {
            Object.values(subChartRefs.current).forEach(subChart => {
              if (subChart) {
                try {
                  subChart.timeScale().setVisibleRange(timeRange);
                } catch (e) {
                  console.warn('Failed to sync time range after fitContent:', e);
                }
              }
            });
          }
        }
      } catch (e) {
        console.warn('Failed to finalize subchart time sync:', e);
      }
    }, 100);
  }, [visualizationData, selectedIndicators, refreshKey]);

  // 搜索股票
  const handleSearch = async (overrideCode?: string) => {
    const codeToValidate = overrideCode || stockCode;
    const { valid, message, normalized } = validateStockCode(codeToValidate);
    if (!valid) {
      setInputError(message);
      return;
    }

    setInputError(undefined);
    setIsLoading(true);

    try {
      console.log('Fetching visualization data for:', normalized);
      
      // 获取起始日期
      const dateRangeOption = DATE_RANGE_OPTIONS.find(opt => opt.id === selectedDateRange);
      let startDateStr: string | undefined;
      if (dateRangeOption?.getStartDate) {
        const startDate = dateRangeOption.getStartDate();
        startDateStr = startDate.toISOString().split('T')[0];
      }
      
      const response = await visualizationApi.getVisualizationData(
        normalized,
        3650,
        selectedIndicators,
        startDateStr
      );

      console.log('Visualization data received:', response);
      console.log('Response中的筹码分布数据:', response.chip_distribution);
      // 创建全新对象，确保React能检测到变化
      const freshData = JSON.parse(JSON.stringify(response));
      setVisualizationData(freshData);
      visualizationDataRef.current = freshData;
      setRefreshKey(prev => prev + 1);
      
      // 保存预计算的筹码分布数据
      if (freshData.chip_distribution) {
        console.log('保存预计算的筹码分布数据:', freshData.chip_distribution);
        console.log('预计算历史数据数量:', freshData.chip_distribution.history?.length || 0);
        if (freshData.chip_distribution.history?.length > 0) {
          console.log('前几个日期:', freshData.chip_distribution.history.slice(0, 5).map((h: any) => h.date));
        }
        precomputedChipDistributionRef.current = freshData.chip_distribution;
      }

      // 记录股价（如果有K线数据，记录最新的收盘价）
      if (freshData.kline_data && freshData.kline_data.length > 0) {
        const latestKline = freshData.kline_data[freshData.kline_data.length - 1];
        if (latestKline.close !== undefined && latestKline.close !== null) {
          recordPrice(normalized, latestKline.close, latestKline.date);
        }
      }

      // 保存搜索历史
      await visualizationApi.saveSearchHistory({
        stock_code: normalized,
        stock_name: response.stock_name,
        days: 3650,
        indicator_types: selectedIndicators,
      });

      // 刷新搜索历史
      await loadSearchHistory();
      
      // 如果显示筹码峰,尝试使用预计算数据
      if (showChipDistribution) {
        if (freshData.chip_distribution?.latest) {
          console.log('直接使用预计算的最新筹码分布数据');
          setChipDistributionData(freshData.chip_distribution.latest);
        } else {
          await fetchChipDistribution(normalized);
        }
      }
    } catch (err) {
      console.error('Failed to load visualization data:', err);
      setInputError('加载数据失败，请重试');
    } finally {
      setIsLoading(false);
    }
  };

  // 点击历史记录
  const handleHistoryClick = async (item: VisualizationSearchHistoryItem) => {
    setStockCode(item.stock_code);
    setSelectedHistoryId(item.id);
    // 不使用历史记录中的指标，使用默认指标
    setSelectedIndicators(filterValidIndicators(['volume', 'macd', 'main_capital_absorption', 'main_cost', 'main_trading']));
    setSidebarOpen(false);

    setIsLoading(true);

    try {
      // 获取起始日期
      const dateRangeOption = DATE_RANGE_OPTIONS.find(opt => opt.id === selectedDateRange);
      let startDateStr: string | undefined;
      if (dateRangeOption?.getStartDate) {
        const startDate = dateRangeOption.getStartDate();
        startDateStr = startDate.toISOString().split('T')[0];
      }
      
      const response = await visualizationApi.getVisualizationData(
        item.stock_code,
        item.days || 3650,
        filterValidIndicators(['volume', 'macd', 'main_capital_absorption', 'main_cost', 'main_trading']),
        startDateStr
      );
      // 创建全新对象，确保React能检测到变化
      const freshData = JSON.parse(JSON.stringify(response));
      setVisualizationData(freshData);
      setRefreshKey(prev => prev + 1);
      
      // 保存预计算的筹码分布数据
      if (freshData.chip_distribution) {
        console.log('保存预计算的筹码分布数据:', freshData.chip_distribution);
        precomputedChipDistributionRef.current = freshData.chip_distribution;
      }

      // 记录股价（如果有K线数据，记录最新的收盘价）
      if (freshData.kline_data && freshData.kline_data.length > 0) {
        const latestKline = freshData.kline_data[freshData.kline_data.length - 1];
        if (latestKline.close !== undefined && latestKline.close !== null) {
          recordPrice(item.stock_code, latestKline.close, latestKline.date);
        }
      }

      // 更新搜索历史记录时间戳，使其置顶显示
      try {
        await visualizationApi.updateSearchHistoryTimestamp(item.id);
        // 重新加载搜索历史列表，以显示更新后的顺序
        await loadSearchHistory();
      } catch (err) {
        console.error('Failed to update search history timestamp:', err);
        // 即使更新失败也不影响用户使用数据
      }
      
      // 如果显示筹码峰,尝试使用预计算数据
      if (showChipDistribution) {
        if (freshData.chip_distribution?.latest) {
          console.log('直接使用预计算的最新筹码分布数据');
          setChipDistributionData(freshData.chip_distribution.latest);
        } else {
          await fetchChipDistribution(item.stock_code);
        }
      }
    } catch (err) {
      console.error('Failed to load visualization data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // 切换指标选择
  const toggleIndicator = (indicatorId: string) => {
    setSelectedIndicators(prev => {
      let newIndicators;
      if (prev.includes(indicatorId)) {
        newIndicators = prev.filter(id => id !== indicatorId);
      } else {
        newIndicators = [...prev, indicatorId];
      }
      return filterValidIndicators(newIndicators);
    });
  };

  // 删除搜索历史
  const handleDeleteHistory = async (recordId: number, stockCode: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await visualizationApi.deleteSearchHistory(recordId);
      // 删除对应的股价记录
      deletePriceRecord(stockCode);
      await loadSearchHistory();
    } catch (err) {
      console.error('Failed to delete search history:', err);
    }
  };

  const sidebarContent = (
    <div className="flex flex-col overflow-hidden min-h-0 h-full">
      <div className="p-3 border-b border-white/5 flex-shrink-0">
        <h3 className="text-sm font-medium text-white">搜索历史</h3>
      </div>
      <div className="overflow-y-auto px-3 py-1 h-[1200px]">
        {isLoadingHistory ? (
        <div className="flex items-center justify-center py-8">
          <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
        </div>
      ) : searchHistory.length === 0 ? (
        <p className="text-xs text-muted text-center py-4">暂无搜索历史</p>
      ) : (
        <div className="space-y-2">
          {searchHistory.map(item => {
            const isCurrentlyDisplayed = visualizationData && 
              visualizationData.stock_code === item.stock_code;
            
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
                      {isCurrentlyDisplayed && (
                        <span className="flex-shrink-0 text-cyan text-xs font-medium">
                          展示中
                        </span>
                      )}
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
                              <span className="text-xs text-muted">
                                {new Date(item.searched_at).toLocaleDateString('zh-CN')}
                              </span>
                            </>
                          );
                        }
                        return (
                          <>
                            <span className="text-xs text-muted font-mono">{item.stock_code}</span>
                            <span className="text-xs text-muted/50">·</span>
                            <span className="text-xs text-muted">
                              {new Date(item.searched_at).toLocaleDateString('zh-CN')}
                            </span>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteHistory(item.id, item.stock_code, e)}
                    className="p-1 text-muted hover:text-danger transition-colors flex-shrink-0"
                    title="删除"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
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
        gridTemplateRows: 'auto 1fr auto' 
      }}
    >
      {/* 顶部搜索栏 */}
      <header
        className="py-3 px-3 md:px-0 border-b border-white/5 flex-shrink-0 flex items-center min-w-0 overflow-hidden md:col-start-2 md:col-end-5 md:row-start-1"
      >
        <div className="flex items-center gap-2 w-full min-w-0 flex-1" style={{ maxWidth: 'min(100%, 1168px)' }}>
          {/* Mobile hamburger */}
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
                handleSearch(query.toUpperCase());
              }}
              className="w-full"
            />
          </div>
          
          <div className="flex-shrink-0">
            <select
              value={selectedDateRange}
              onChange={(e) => setSelectedDateRange(e.target.value)}
              disabled={isLoading}
              className="input-terminal bg-[#1a1a2e] text-sm border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-cyan/50"
            >
              {DATE_RANGE_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
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
            className="absolute left-0 top-0 bottom-0 w-72 flex flex-col glass-card overflow-hidden border-r border-white/10 shadow-2xl p-3"
            onClick={(e) => e.stopPropagation()}
          >
            {sidebarContent}
          </div>
        </div>
      )}

      {/* 中央图表区域 - 容器始终渲染 */}
      <section className="flex-1 overflow-y-auto overflow-x-auto px-3 md:px-0 md:pl-1 min-w-0 min-h-0 md:col-start-4 md:row-start-2">
        {/* 图表容器始终存在 */}
        <div className="max-w-6xl">
          {/* 标题 - 只在有数据时显示 */}
          {visualizationData && (
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white">
                  {visualizationData.stock_name ? `${visualizationData.stock_name} (${visualizationData.stock_code})` : visualizationData.stock_code}
                </h2>
                {/* 筹码峰切换按钮 */}
                <button
                  type="button"
                  onClick={async () => {
                    const newShow = !showChipDistribution;
                    setShowChipDistribution(newShow);
                    if (newShow && visualizationData?.stock_code) {
                      await fetchChipDistribution(visualizationData.stock_code);
                    }
                  }}
                  className={`
                    px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer
                    ${showChipDistribution
                      ? 'border-purple-500/40 bg-purple-500/10 text-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.15)]'
                      : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                    }
                  `}
                >
                  <div className="flex items-center gap-1.5">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: showChipDistribution ? '#a855f7' : '#666666' }}
                    />
                    筹码峰
                  </div>
                </button>
              </div>
              <div className="flex flex-wrap items-center gap-3 mt-1">
                <div className="text-xs text-muted">
                  K线数据: {visualizationData.kline_data?.length || 0} 条
                </div>
                {/* 展示涨跌幅 */}
                {(() => {
                  if (!visualizationData.kline_data || visualizationData.kline_data.length === 0) {
                    return null;
                  }
                  const latestKline = visualizationData.kline_data[visualizationData.kline_data.length - 1];
                  if (latestKline.close === undefined || latestKline.close === null) {
                    return null;
                  }
                  const priceChange = calculatePriceChange(visualizationData.stock_code, latestKline.close);
                  if (!priceChange.hasRecord) {
                    return (
                      <div className="text-xs text-muted flex items-center gap-1">
                        <span className="inline-block w-2 h-2 rounded-full bg-gray-500"></span>
                        首次搜索
                      </div>
                    );
                  }
                  const isPositive = priceChange.changePercent >= 0;
                  const colorClass = isPositive ? 'text-red-400' : 'text-green-400';
                  const bgClass = isPositive ? 'bg-red-400/10' : 'bg-green-400/10';
                  const sign = isPositive ? '+' : '';
                  return (
                    <div className={`text-xs ${colorClass} ${bgClass} px-2 py-0.5 rounded flex items-center gap-1`}>
                      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: isPositive ? '#f87171' : '#4ade80' }}></span>
                      涨跌幅: {sign}{priceChange.changePercent.toFixed(2)}%
                      {priceChange.firstPrice && (
                        <span className="text-xs text-muted/70 ml-1">
                          (首价: {priceChange.firstPrice.toFixed(2)})
                        </span>
                      )}
                    </div>
                  );
                })()}
              </div>
            </div>
          )}
          
          {/* 图表区域：主图表和指标子图 */}
          <div className="w-full">
              {/* 一体化图表：K线 + 筹码分布 */}
              <div className={showChipDistribution ? "flex gap-0" : ""}>
                {/* 左侧：K线图 */}
                <Card variant="default" padding="none" className={showChipDistribution ? "mb-4 flex-1 rounded-r-none border-r-0" : "mb-4"}>
                  {/* 光标值显示标签 */}
                  <div className="p-2 border-b border-white/5 flex flex-wrap gap-4 items-center">
                    {/* 显示跟随光标的时间 */}
                    {currentCrosshairTimeRef.current !== undefined && currentCrosshairTimeRef.current !== null && (
                      <span className="text-xs text-cyan font-mono">
                        {(() => {
                          let date: Date;
                          const timeValue = currentCrosshairTimeRef.current;
                          if (typeof timeValue === 'string') {
                            if (timeValue.includes('-')) {
                              date = new Date(timeValue);
                            } else {
                              date = new Date(parseInt(timeValue) * 1000);
                            }
                          } else {
                            date = new Date(timeValue * 1000);
                          }
                          const year = date.getFullYear();
                          const month = String(date.getMonth() + 1).padStart(2, '0');
                          const day = String(date.getDate()).padStart(2, '0');
                          return `${year}-${month}-${day}`;
                        })()}
                      </span>
                    )}
                    {selectedIndicators.includes('main_trading') && (
                      <>
                        <h3 className="text-sm font-medium text-white">主力操盘</h3>
                        {cursorValues.attackLine !== undefined && cursorValues.attackLine !== null && !isNaN(cursorValues.attackLine) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FF4444' }} />
                            <span className="text-xs text-white">攻击线: <span className="font-mono">{cursorValues.attackLine.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.tradingLine !== undefined && cursorValues.tradingLine !== null && !isNaN(cursorValues.tradingLine) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FFAA00' }} />
                            <span className="text-xs text-white">操盘线: <span className="font-mono">{cursorValues.tradingLine.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.defenseLine !== undefined && cursorValues.defenseLine !== null && !isNaN(cursorValues.defenseLine) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#44AA44' }} />
                            <span className="text-xs text-white">防守线: <span className="font-mono">{cursorValues.defenseLine.toFixed(2)}</span></span>
                            {cursorValues.signal === 'buy' && (
                              <span className="text-xs font-bold" style={{ color: '#FF0000' }}>买入</span>
                            )}
                            {cursorValues.signal === 'sell' && (
                              <span className="text-xs font-bold" style={{ color: '#00FF00' }}>卖出</span>
                            )}
                          </div>
                        )}
                      </>
                    )}
                    {selectedIndicators.includes('tiandao') && (
                      <>
                        <h3 className="text-sm font-medium text-white">天道</h3>
                        {cursorValues.tiandaoJinzuan !== undefined && cursorValues.tiandaoJinzuan !== null && !isNaN(cursorValues.tiandaoJinzuan) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FF0000' }} />
                            <span className="text-xs text-white">金钻趋势: <span className="font-mono">{cursorValues.tiandaoJinzuan.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.tiandaoJinniu !== undefined && cursorValues.tiandaoJinniu !== null && !isNaN(cursorValues.tiandaoJinniu) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FFFF00' }} />
                            <span className="text-xs text-white">金牛: <span className="font-mono">{cursorValues.tiandaoJinniu.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.tiandaoJinniu2 !== undefined && cursorValues.tiandaoJinniu2 !== null && !isNaN(cursorValues.tiandaoJinniu2) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#00FFFF' }} />
                            <span className="text-xs text-white">金牛2: <span className="font-mono">{cursorValues.tiandaoJinniu2.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.tiandaoBbi !== undefined && cursorValues.tiandaoBbi !== null && !isNaN(cursorValues.tiandaoBbi) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full border border-white" style={{ backgroundColor: 'transparent' }} />
                            <span className="text-xs text-white">BBI: <span className="font-mono">{cursorValues.tiandaoBbi.toFixed(2)}</span></span>
                          </div>
                        )}
                      </>
                    )}
                    {selectedIndicators.includes('bollinger') && (
                      <>
                        <h3 className="text-sm font-medium text-white">布林带</h3>
                        {(cursorValues.bollingerUpper !== undefined && cursorValues.bollingerUpper !== null && !isNaN(cursorValues.bollingerUpper)) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FF8C00' }} />
                            <span className="text-xs text-white">上轨: <span className="font-mono">{cursorValues.bollingerUpper.toFixed(2)}</span></span>
                          </div>
                        )}
                        {(cursorValues.bollingerMiddle !== undefined && cursorValues.bollingerMiddle !== null && !isNaN(cursorValues.bollingerMiddle)) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full border border-white" style={{ backgroundColor: 'transparent' }} />
                            <span className="text-xs text-white">中轨: <span className="font-mono">{cursorValues.bollingerMiddle.toFixed(2)}</span></span>
                          </div>
                        )}
                        {(cursorValues.bollingerLower !== undefined && cursorValues.bollingerLower !== null && !isNaN(cursorValues.bollingerLower)) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FF8C00' }} />
                            <span className="text-xs text-white">下轨: <span className="font-mono">{cursorValues.bollingerLower.toFixed(2)}</span></span>
                          </div>
                        )}
                      </>
                    )}
                    {selectedIndicators.includes('expma') && (
                      <>
                        <h3 className="text-sm font-medium text-white">EXPMA</h3>
                        {cursorValues.expmaFast !== undefined && cursorValues.expmaFast !== null && !isNaN(cursorValues.expmaFast) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FFFFFF' }} />
                            <span className="text-xs text-white">EXPMA13: <span className="font-mono">{cursorValues.expmaFast.toFixed(2)}</span></span>
                          </div>
                        )}
                        {cursorValues.expmaSlow !== undefined && cursorValues.expmaSlow !== null && !isNaN(cursorValues.expmaSlow) && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#FFD700' }} />
                            <span className="text-xs text-white">EXPMA30: <span className="font-mono">{cursorValues.expmaSlow.toFixed(2)}</span></span>
                          </div>
                        )}
                      </>
                    )}
                    {/* 筹码分布标签（只在显示筹码时显示） */}
                    {showChipDistribution && chipDistributionData && (
                      <>
                        {chipDistributionData.current_price !== undefined && chipDistributionData.current_price !== null && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#fbbf24' }} />
                            <span className="text-xs text-white">现价: <span className="font-mono">{chipDistributionData.current_price.toFixed(2)}</span></span>
                          </div>
                        )}
                        {chipDistributionData.avg_cost !== undefined && chipDistributionData.avg_cost !== null && (
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#ff6b6b' }} />
                            <span className="text-xs text-white">成本: <span className="font-mono">{chipDistributionData.avg_cost.toFixed(2)}</span></span>
                          </div>
                        )}
                      </>
                    )}
                  {selectedIndicators.includes('tiandao') && visualizationData && null}
                  </div>
                  <div 
                    ref={mainChartContainerRef} 
                    style={{ 
                      height: '500px', 
                      width: '100%',
                      minHeight: '500px',
                      backgroundColor: '#1a1a2e'
                    }} 
                  />
                </Card>
                
                {/* 右侧：筹码分布图（无Card包装，视觉融合） */}
                {showChipDistribution && (
                  <div 
                    className="mb-4"
                    style={{ width: '300px', minWidth: '300px' }}
                  >
                    {/* 占位区域，与左侧标题对齐 */}
                    <div style={{ height: '40px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}></div>
                    {/* 筹码分布图，无Card */}
                    <div 
                      style={{ 
                        height: '500px', 
                        width: '100%',
                        minHeight: '500px',
                        backgroundColor: '#1a1a2e',
                        border: '1px solid rgba(255,255,255,0.05)',
                        borderLeft: 'none',
                        borderRadius: '0 0.5rem 0.5rem 0'
                      }} 
                    >
                      <ChipDistributionChart 
                        data={chipDistributionData} 
                        loading={isLoadingChipDistribution} 
                        priceRange={priceRange}
                        cursorPrice={cursorPrice}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* 指标子图 - 只在有数据时显示（跳过主力操盘和EXPMA，它们在主图显示） */}
              {visualizationData && selectedIndicators.map(indicatorId => {
                if (indicatorId === 'main_trading' || indicatorId === 'expma' || indicatorId === 'tiandao' || indicatorId === 'bollinger') return null;
                
                const indicatorData = indicatorId !== 'volume' 
                  ? visualizationData.indicators.find(ind => ind.indicator_type === indicatorId)
                  : { data: [] };
                const indicatorOption = INDICATOR_OPTIONS.find(opt => opt.id === indicatorId);
                
                if (!indicatorOption) return null;
                if (indicatorId !== 'volume' && (!indicatorData || !indicatorData.data || indicatorData.data.length === 0)) return null;

                return (
                  <Card key={indicatorId} variant="default" padding="none" className="mb-4">
                    <div className="p-2 border-b border-white/5 flex items-center justify-between">
                      <h3 className="text-sm font-medium text-white flex items-center gap-2">
                        {/* 所有指标都显示圆点和名称 */}
                        <div 
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: indicatorOption.color }}
                        />
                        {indicatorOption.name}
                        {/* 主力成本指标显示资金流向数据，跟随十字线更新，没有十字线时显示最新值 */}
                        {indicatorId === 'main_cost' && indicatorData && (
                          <span className="text-xs text-muted/80 ml-2">
                            {(() => {
                              let mainNetBuyWan = cursorValues.mainNetBuyWan;
                              let mainDirection = cursorValues.mainDirection;
                              let turnoverRatio = cursorValues.turnoverRatio;
                              
                              // 如果十字线没有数据，回退到 metadata 的最新值
                              if (mainNetBuyWan === undefined || mainNetBuyWan === null || isNaN(mainNetBuyWan)) {
                                const metadata = (indicatorData as any).metadata;
                                if (metadata) {
                                  mainNetBuyWan = metadata.main_net_buy_wan;
                                  mainDirection = metadata.main_direction;
                                  turnoverRatio = metadata.turnover_ratio;
                                }
                              }
                              
                              if (mainNetBuyWan === undefined || mainNetBuyWan === null || isNaN(mainNetBuyWan)) {
                                return null;
                              }
                              
                              const formatAmount = (amount: number) => {
                                return `${Math.abs(amount).toFixed(2)}万`;
                              };
                              
                              const isInflow = mainDirection === 'inflow';
                              const color = isInflow ? '#FF4444' : '#44AA44';
                              
                              return (
                                <span className="flex items-center gap-2">
                                  <span style={{ color }}>
                                    主力: {isInflow ? '+' : '-'}{formatAmount(mainNetBuyWan)}
                                  </span>
                                  {turnoverRatio !== undefined && turnoverRatio !== null && !isNaN(turnoverRatio) && (
                                    <span className="text-xs text-muted/70">
                                      占比: {turnoverRatio.toFixed(2)}%
                                    </span>
                                  )}
                                </span>
                              );
                            })()}
                          </span>
                        )}
                        {/* 强势起爆指标显示连续紫色箱体数量 */}
                        {indicatorId === 'strong_detonation' && indicatorData && (
                          <span className="text-xs text-muted/80 ml-2">
                            {(() => {
                              const data = indicatorData.data;
                              if (!data || data.length === 0) return null;
                              
                              let consecutivePurple = 0;
                              for (let i = data.length - 1; i >= 0; i--) {
                                const item = data[i];
                                if (item.purple_box_open !== null && item.purple_box_open !== undefined && !isNaN(item.purple_box_open)) {
                                  consecutivePurple++;
                                } else {
                                  break;
                                }
                              }
                              if (consecutivePurple > 0) {
                                return (
                                  <span style={{ color: '#AA44FF' }}>
                                    连紫: {consecutivePurple}
                                  </span>
                                );
                              }
                              return null;
                            })()}
                          </span>
                        )}
                        {/* 共振追涨指标显示最新共振柱顶部高度 */}
                        {indicatorId === 'resonance_chase' && indicatorData && (
                          <span className="text-xs text-muted/80 ml-2">
                            {(() => {
                              const data = indicatorData.data;
                              if (!data || data.length === 0) return null;
                              
                              const lastItem = data[data.length - 1];
                              if (!lastItem) return null;
                              
                              if (lastItem.resonance) {
                                const originalHeight = Math.abs(lastItem.OUT1 || 0);
                                const topValue = originalHeight - 0.5;
                                if (topValue > 0) {
                                  return (
                                    <span style={{ color: '#AA44FF' }}>
                                      高度: {Math.floor(topValue)}
                                    </span>
                                  );
                                }
                              }
                              return null;
                            })()}
                          </span>
                        )}
                      </h3>
                      {/* 显示光标处的指标值 */}
                      {indicatorId === 'banker_control' && cursorValues.bankerControl !== undefined && cursorValues.bankerControl !== null && !isNaN(cursorValues.bankerControl) && (
                        <span className="text-xs font-mono text-cyan">
                          {cursorValues.bankerControl.toFixed(2)}
                        </span>
                      )}
                      {indicatorId === 'main_capital_absorption' && cursorValues.mainCapitalAbsorption !== undefined && cursorValues.mainCapitalAbsorption !== null && !isNaN(cursorValues.mainCapitalAbsorption) && (
                        <span className="text-xs font-mono text-cyan">
                          {cursorValues.mainCapitalAbsorption.toFixed(2)}
                        </span>
                      )}
                      {indicatorId === 'main_cost' && cursorValues.mainCost !== undefined && cursorValues.mainCost !== null && !isNaN(cursorValues.mainCost) && cursorValues.closePrice !== undefined && cursorValues.closePrice !== null && !isNaN(cursorValues.closePrice) && (
                        <span className="flex items-center gap-2">
                          <span className="text-xs font-mono text-cyan">
                            {cursorValues.mainCost.toFixed(2)}
                          </span>
                          <span className={`text-xs font-mono ${((cursorValues.closePrice - cursorValues.mainCost) / cursorValues.mainCost * 100) >= 0 ? 'text-[#FF4444]' : 'text-[#44AA44]'}`}>
                            {((cursorValues.closePrice - cursorValues.mainCost) / cursorValues.mainCost * 100).toFixed(2)}%
                          </span>
                        </span>
                      )}
                      {indicatorId === 'main_cost' && cursorValues.mainCost !== undefined && cursorValues.mainCost !== null && !isNaN(cursorValues.mainCost) && (cursorValues.closePrice === undefined || cursorValues.closePrice === null || isNaN(cursorValues.closePrice)) && (
                        <span className="text-xs font-mono text-cyan">
                          {cursorValues.mainCost.toFixed(2)}
                        </span>
                      )}
                      {indicatorId === 'resonance_chase' && cursorValues.resonanceChase !== undefined && cursorValues.resonanceChase !== null && !isNaN(cursorValues.resonanceChase) && (
                        <span className="text-xs font-mono text-cyan">
                          {Math.floor(cursorValues.resonanceChase)}
                        </span>
                      )}
                      {indicatorId === 'dmi' && cursorValues.dmiPdi !== undefined && cursorValues.dmiPdi !== null && !isNaN(cursorValues.dmiPdi) && (
                        <span className="flex items-center gap-2">
                          <span className="text-xs font-mono text-[#FF4444]">+DI:{cursorValues.dmiPdi.toFixed(2)}</span>
                          <span className="text-xs font-mono text-[#44AA44]">-DI:{cursorValues.dmiMdi?.toFixed(2)}</span>
                          <span className="text-xs font-mono text-[#AA44FF]">ADX:{cursorValues.dmiAdx?.toFixed(2)}</span>
                        </span>
                      )}
                      {/* MACD 标题栏：信号文本 + 柱高和 + 柱高差 */}
                      {indicatorId === 'macd' && indicatorData && (
                        <div className="flex items-center gap-2 ml-auto">
                          {(() => {
                            const metadata = (indicatorData as any).metadata;
                            let signalText = cursorValues.macdSignal;
                            let barSum = cursorValues.macdBarSum;
                            let barDiff = cursorValues.macdBarDiff;

                            if (signalText === undefined && metadata?.signal_text) {
                              signalText = metadata.signal_text;
                            }
                            if (barSum === undefined && metadata?.bar_sum !== undefined) {
                              barSum = metadata.bar_sum;
                            }
                            if (barDiff === undefined && metadata?.bar_diff !== undefined) {
                              barDiff = metadata.bar_diff;
                            }

                            const isBuySignal = signalText && (signalText.includes('\u91d1') || signalText.includes('\u591a'));
                            const signalColor = signalText ? (isBuySignal ? '#FF4444' : '#44FF44') : '#888';
                            const signalBgColor = signalText ? (isBuySignal ? 'rgba(255,68,68,0.1)' : 'rgba(68,255,68,0.1)') : undefined;

                            return (
                              <>
                                {signalText && (
                                  <span className="text-[11px] font-semibold px-1.5 py-px rounded"
                                    style={{ color: signalColor, backgroundColor: signalBgColor }}>
                                    {signalText}
                                  </span>
                                )}
                                {barSum !== undefined && (
                                  <span className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                    style={{ color: barSum >= 0 ? '#FF4444' : '#44FF44', backgroundColor: barSum >= 0 ? 'rgba(255,68,68,0.1)' : 'rgba(68,255,68,0.1)' }}>
                                    柱高和 {barSum >= 0 ? '+' : ''}{barSum.toFixed(2)}
                                  </span>
                                )}
                                {barDiff !== undefined && (
                                  <span className="text-[10px] font-mono font-medium px-1.5 py-px rounded"
                                    style={{ color: barDiff >= 0 ? '#FF4444' : '#44FF44', backgroundColor: barDiff >= 0 ? 'rgba(255,68,68,0.1)' : 'rgba(68,255,68,0.1)' }}>
                                    柱高差 {barDiff >= 0 ? '+' : ''}{barDiff.toFixed(2)}
                                  </span>
                                )}
                                <div className="flex items-center gap-1 ml-2">
                                  <span className="w-2.5 h-0.5 rounded" style={{ backgroundColor: '#FFFFFF' }}></span>
                                  <span className="text-[10px] text-muted/60">DIF</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <span className="w-2.5 h-0.5 rounded" style={{ backgroundColor: '#FFD700' }}></span>
                                  <span className="text-[10px] text-muted/60">DEA</span>
                                </div>
                                <div className="flex items-center gap-0.5">
                                  <div className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: '#FF4444' }}></div>
                                  <div className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: '#44FF44' }}></div>
                                </div>
                              </>
                            );
                          })()}
                        </div>
                      )}
                      {/* 成交量子图均量线图例 */}
                      {indicatorId === 'volume' && (
                        <div className="flex items-center gap-3 ml-auto">
                          <div className="flex items-center gap-1">
                            <span className="w-4 h-0.5 rounded" style={{ backgroundColor: '#FFFFFF' }} />
                            <span className="text-[10px] text-muted/60">5日均量</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="w-4 h-0.5 rounded" style={{ backgroundColor: '#FFD700' }} />
                            <span className="text-[10px] text-muted/60">10日均量</span>
                          </div>
                        </div>
                      )}
                    </div>
                    <div 
                      ref={(el) => { subChartContainerRefs.current[indicatorId] = el; }}
                      style={{ 
                        height: `${SUBCHART_HEIGHT}px`, 
                        width: '100%',
                        minHeight: `${SUBCHART_HEIGHT}px`,
                        backgroundColor: '#1a1a2e'
                      }} 
                    />
                  </Card>
                );
              })}
          </div>

          {/* 空白状态 - 只在没有数据且没有加载时显示 */}
          {!isLoading && !visualizationData && (
            <div className="flex flex-col items-center justify-center h-full text-center" style={{ minHeight: '500px' }}>
              <div className="w-12 h-12 mb-3 rounded-xl bg-elevated flex items-center justify-center">
                <svg className="w-6 h-6 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-white mb-1.5">开始查看股票走势</h3>
              <p className="text-xs text-muted max-w-xs">
                输入股票代码查看K线图和技术指标
              </p>
            </div>
          )}
        </div>
      </section>




      {/* 底部指标选择区 */}
      <footer
        className="py-3 px-3 md:px-0 border-t border-white/5 flex-shrink-0 md:col-start-2 md:col-end-5 md:row-start-3"
      >
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-white">选择指标</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {INDICATOR_OPTIONS.map((indicator) => (
            <button
              key={indicator.id}
              type="button"
              onClick={() => toggleIndicator(indicator.id)}
              className={`
                px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer
                ${selectedIndicators.includes(indicator.id)
                  ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                  : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                }
              `}
            >
              <div className="flex items-center gap-1.5">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: indicator.color }}
                />
                {indicator.name}
              </div>
            </button>
          ))}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default VisualizationPage;
