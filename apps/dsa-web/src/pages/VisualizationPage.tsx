import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';

import { visualizationApi, type VisualizationResponse, type VisualizationSearchHistoryItem } from '../api/visualization';
import { validateStockCode } from '../utils/validation';
import { Card } from '../components/common';

// 定义指标配置
const INDICATOR_OPTIONS = [
  { id: 'volume', name: '成交量', description: '成交量柱状图', color: '#666666' },
  { id: 'banker_control', name: '庄家控盘', description: '庄家控盘程度指标', color: '#FFAA00' },
  { id: 'main_capital_absorption', name: '主力吸筹', description: '主力资金吸筹情况', color: '#AA44FF' },
  { id: 'main_cost', name: '主力成本', description: '主力资金成本', color: '#FFAA00' },
  { id: 'main_trading', name: '主力操盘', description: '主力操盘三线', color: '#FF4444' },
];

// 子图高度
const SUBCHART_HEIGHT = 150;

const VisualizationPage: React.FC = () => {
  const validIndicatorIds = INDICATOR_OPTIONS.map(opt => opt.id);
  
  const filterValidIndicators = (indicators: string[]) => {
    return indicators.filter(id => validIndicatorIds.includes(id));
  };

  // 状态管理
  const [stockCode, setStockCode] = useState('');
  const [inputError, setInputError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [visualizationData, setVisualizationData] = useState<VisualizationResponse | null>(null);
  const [searchHistory, setSearchHistory] = useState<VisualizationSearchHistoryItem[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<number | null>(null);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(
    filterValidIndicators(['volume', 'main_capital_absorption', 'banker_control', 'main_trading'])
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [cursorValues, setCursorValues] = useState<{
    attackLine?: number;
    tradingLine?: number;
    defenseLine?: number;
    signal?: 'buy' | 'sell';
    bankerControl?: number;
    mainCapitalAbsorption?: number;
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

  // 加载搜索历史
  const loadSearchHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const response = await visualizationApi.getSearchHistory(20);
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

      // 验证并限制时间范围函数
      const validateAndClampTimeRange = (timeRange: any) => {
        if (!timeRange || !timeRange.from || !timeRange.to) {
          return { clampedRange: timeRange, wasClamped: false };
        }

        let clampedFrom = timeRange.from;
        let clampedTo = timeRange.to;
        let wasClamped = false;

        // 转换为时间戳进行比较
        const toTimestamp = (time: any): number => {
          if (typeof time === 'string' && time.includes('-')) {
            return new Date(time).getTime() / 1000;
          }
          return Number(time);
        };

        const fromTs = toTimestamp(clampedFrom);
        const toTs = toTimestamp(clampedTo);
        const earliestTs = earliestDateRef.current ? toTimestamp(earliestDateRef.current) : -Infinity;
        const latestTs = latestDateRef.current ? toTimestamp(latestDateRef.current) : Infinity;

        // 检查并限制边界
        if (fromTs < earliestTs) {
          const originalRange = toTs - fromTs;
          clampedFrom = earliestDateRef.current;
          const newFromTs = earliestTs;
          const newToTs = newFromTs + originalRange;
          
          // 确保新的 to 不超过 latestDateRef.current
          if (newToTs > latestTs && latestDateRef.current) {
            clampedTo = latestDateRef.current;
            clampedFrom = earliestDateRef.current;
          } else {
            clampedTo = timeRange.to;
          }
          wasClamped = true;
        }

        if (toTs > latestTs) {
          const originalRange = toTs - fromTs;
          clampedTo = latestDateRef.current;
          const newToTs = latestTs;
          const newFromTs = newToTs - originalRange;
          
          // 确保新的 from 不小于 earliestDateRef.current
          if (newFromTs < earliestTs && earliestDateRef.current) {
            clampedFrom = earliestDateRef.current;
            clampedTo = latestDateRef.current;
          } else {
            clampedFrom = timeRange.from;
          }
          wasClamped = true;
        }

        // 最终确保边界不超出
        const finalFromTs = toTimestamp(clampedFrom);
        const finalToTs = toTimestamp(clampedTo);
        
        if (finalFromTs < earliestTs && earliestDateRef.current) {
          clampedFrom = earliestDateRef.current;
          wasClamped = true;
        }
        if (finalToTs > latestTs && latestDateRef.current) {
          clampedTo = latestDateRef.current;
          wasClamped = true;
        }

        return { 
          clampedRange: { from: clampedFrom, to: clampedTo }, 
          wasClamped 
        };
      };

      // 订阅主图时间轴变化，同步到所有子图
      const handleTimeScaleChange = () => {
        if (!mainChartRef.current || isTimeRangeUpdatingRef.current) return;
        try {
          isTimeRangeUpdatingRef.current = true;
          const timeRange = mainChartRef.current.timeScale().getVisibleRange();
          if (timeRange && timeRange.from && timeRange.to) {
            // 验证并限制时间范围
            const { clampedRange, wasClamped } = validateAndClampTimeRange(timeRange);
            
            // 如果被限制了，应用到主图
            if (wasClamped) {
              mainChartRef.current.timeScale().setVisibleRange(clampedRange);
            }
            
            // 同步时间范围到子图
            Object.values(subChartRefs.current).forEach(subChart => {
              if (subChart) {
                try {
                  subChart.timeScale().setVisibleRange(clampedRange);
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
      
      // 订阅十字线移动以更新光标处三线值和信号
      const handleCrosshairMove = (param: any) => {
        // 防止循环更新
        if (isCrosshairUpdatingRef.current) {
          return;
        }
        
        isCrosshairUpdatingRef.current = true;
        
        try {
          currentCrosshairTimeRef.current = param.time;
          
          // 同步十字线到所有子图 - 更新光标值显示和十字线位置
          if (param.time) {
            // 获取所有子图指标的值
            let bankerControl: number | undefined = cursorValues.bankerControl;
            let mainCapitalAbsorption: number | undefined = cursorValues.mainCapitalAbsorption;
            
            if (indicatorsDataRef.current['banker_control']) {
              const bankerData = indicatorsDataRef.current['banker_control'].data.find((item: any) => {
                const itemDate = item.date || item.time;
                return itemDate === param.time;
              });
              if (bankerData) {
                bankerControl = bankerData.control_degree;
              }
            }
            
            if (indicatorsDataRef.current['main_capital_absorption']) {
              const absorptionData = indicatorsDataRef.current['main_capital_absorption'].data.find((item: any) => {
                const itemDate = item.date || item.time;
                return itemDate === param.time;
              });
              if (absorptionData) {
                mainCapitalAbsorption = absorptionData.main_capital_absorption;
              }
            }
            
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
                    const indicatorData = indicatorsDataRef.current[indicatorId];
                    if (indicatorData && indicatorData.data) {
                      const dataPoint = indicatorData.data.find((item: any) => {
                        const itemDate = item.date || item.time;
                        return itemDate === param.time;
                      });
                      
                      if (dataPoint) {
                        if (indicatorId === 'banker_control') {
                          price = dataPoint.control_degree;
                        } else if (indicatorId === 'main_capital_absorption') {
                          price = dataPoint.main_capital_absorption;
                        } else if (indicatorId === 'main_cost') {
                          price = dataPoint.main_cost || dataPoint.cost;
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
            
            // 更新光标值，包括所有子图指标
            if (!mainTradingDataRef.current || !klineDataRef.current) {
              setCursorValues({
                bankerControl,
                mainCapitalAbsorption,
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
                });
              } else {
                setCursorValues({
                  bankerControl,
                  mainCapitalAbsorption,
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
    console.log('Updating K-line data only:', visualizationData);
    
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
      }

      candlestickSeriesRef.current.setData(klineData);
      
      if (mainChartRef.current) {
        mainChartRef.current.timeScale().fitContent();
      }
    } catch (error) {
      console.error('Error updating K-line data:', error);
    }
  }, [visualizationData]);
  
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
          value: item.attack_line || null,
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
          value: item.trading_line || null,
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
          value: item.defense_line || null,
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
  }, [selectedIndicators, visualizationData]);

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
      if (indicatorId === 'main_trading') return;
      
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
          chart.timeScale().fitContent();

        } else if (indicatorId === 'banker_control' && indicatorData) {
          // 庄家控盘 - 显示能量柱，分三段颜色
          // 先过滤数据，确保时间范围与主图对齐
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const barData = filteredIndicatorData.map((item: any) => {
            const value = item.control_degree || 0;
            let color = 'transparent';
            if (value >= 80) {
              color = '#AA44FF';
            } else if (value >= 60) {
              color = '#FF4444';
            } else if (value >= 50) {
              color = '#FFAA00';
            }
            return {
              time: item.date,
              value: value >= 50 ? value : 50,
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
          
          // 将庄家控盘的零轴（基准）设为50
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

          // 添加标记线
          const level50Line = chart.addSeries(lightweightCharts.LineSeries, {
            color: '#666666',
            lineWidth: 1,
            lineStyle: 2,
            crosshairMarkerVisible: false,
          });
          level50Line.setData(barData.map((d: any) => ({ time: d.time, value: 50 })));

          const level60Line = chart.addSeries(lightweightCharts.LineSeries, {
            color: '#666666',
            lineWidth: 1,
            lineStyle: 2,
            crosshairMarkerVisible: false,
          });
          level60Line.setData(barData.map((d: any) => ({ time: d.time, value: 60 })));

          const level80Line = chart.addSeries(lightweightCharts.LineSeries, {
            color: '#666666',
            lineWidth: 1,
            lineStyle: 2,
            crosshairMarkerVisible: false,
          });
          level80Line.setData(barData.map((d: any) => ({ time: d.time, value: 80 })));

          chart.timeScale().fitContent();

        } else if (indicatorId === 'main_capital_absorption' && indicatorData) {
          // 主力吸筹 - 显示柱状图
          // 先过滤数据，确保时间范围与主图对齐
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const barData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: item.main_capital_absorption || 0,
            color: (item.main_capital_absorption || 0) >= 0 ? '#AA44FF' : '#44AA44',
          })).filter((d: any) => d.value !== null && d.value !== undefined);

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
          chart.timeScale().fitContent();

        } else if (indicatorId === 'main_cost' && indicatorData) {
          // 主力成本 - 显示成本曲线
          // 先过滤数据，确保时间范围与主图对齐
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          const lineData = filteredIndicatorData.map((item: any) => ({
            time: item.date,
            value: item.main_cost || item.cost || 0,
          })).filter((d: any) => d.value !== null && d.value !== undefined);

          if (lineData.length > 0) {
            const lineSeries = chart.addSeries(lightweightCharts.LineSeries, {
              color: '#FFAA00',
              lineWidth: 2,
            });
            lineSeries.setData(lineData);
            subChartSeriesRefs.current[indicatorId] = lineSeries;
            chart.timeScale().fitContent();
          }

        }

        // 处理窗口大小变化
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
        
        // 子图验证并限制时间范围函数
        const validateAndClampTimeRangeSub = (timeRange: any) => {
          if (!timeRange || !timeRange.from || !timeRange.to) {
            return { clampedRange: timeRange, wasClamped: false };
          }

          let clampedFrom = timeRange.from;
          let clampedTo = timeRange.to;
          let wasClamped = false;

          // 转换为时间戳进行比较
          const toTimestamp = (time: any): number => {
            if (typeof time === 'string' && time.includes('-')) {
              return new Date(time).getTime() / 1000;
            }
            return Number(time);
          };

          const fromTs = toTimestamp(clampedFrom);
          const toTs = toTimestamp(clampedTo);
          const earliestTs = earliestDateRef.current ? toTimestamp(earliestDateRef.current) : -Infinity;
          const latestTs = latestDateRef.current ? toTimestamp(latestDateRef.current) : Infinity;

          // 检查并限制边界
          if (fromTs < earliestTs) {
            const originalRange = toTs - fromTs;
            clampedFrom = earliestDateRef.current;
            const newFromTs = earliestTs;
            const newToTs = newFromTs + originalRange;
            
            // 确保新的 to 不超过 latestDateRef.current
            if (newToTs > latestTs && latestDateRef.current) {
              clampedTo = latestDateRef.current;
              clampedFrom = earliestDateRef.current;
            } else {
              clampedTo = timeRange.to;
            }
            wasClamped = true;
          }

          if (toTs > latestTs) {
            const originalRange = toTs - fromTs;
            clampedTo = latestDateRef.current;
            const newToTs = latestTs;
            const newFromTs = newToTs - originalRange;
            
            // 确保新的 from 不小于 earliestDateRef.current
            if (newFromTs < earliestTs && earliestDateRef.current) {
              clampedFrom = earliestDateRef.current;
              clampedTo = latestDateRef.current;
            } else {
              clampedFrom = timeRange.from;
            }
            wasClamped = true;
          }

          // 最终确保边界不超出
          const finalFromTs = toTimestamp(clampedFrom);
          const finalToTs = toTimestamp(clampedTo);
          
          if (finalFromTs < earliestTs && earliestDateRef.current) {
            clampedFrom = earliestDateRef.current;
            wasClamped = true;
          }
          if (finalToTs > latestTs && latestDateRef.current) {
            clampedTo = latestDateRef.current;
            wasClamped = true;
          }

          return { 
            clampedRange: { from: clampedFrom, to: clampedTo }, 
            wasClamped 
          };
        };

        // 添加子图时间轴变化监听，实现双向联动
        const handleSubChartTimeScaleChange = () => {
          if (!chart || isTimeRangeUpdatingRef.current) return;
          
          try {
            isTimeRangeUpdatingRef.current = true;
            const timeRange = chart.timeScale().getVisibleRange();
            if (timeRange && timeRange.from && timeRange.to && mainChartRef.current) {
              // 验证并限制时间范围
              const { clampedRange, wasClamped } = validateAndClampTimeRangeSub(timeRange);
              
              // 如果被限制了，应用到当前子图
              if (wasClamped) {
                chart.timeScale().setVisibleRange(clampedRange);
              }
              
              // 同步到主图
              mainChartRef.current.timeScale().setVisibleRange(clampedRange);
              
              // 同步到其他子图
              Object.keys(subChartRefs.current).forEach(otherIndicatorId => {
                if (otherIndicatorId !== indicatorId && subChartRefs.current[otherIndicatorId]) {
                  try {
                    subChartRefs.current[otherIndicatorId]!.timeScale().setVisibleRange(clampedRange);
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
            
            // 获取当前子图指标的值
            let indicatorValue: number | undefined;
            if (param.time && indicatorsDataRef.current[indicatorId]) {
              const indicatorData = indicatorsDataRef.current[indicatorId].data.find((item: any) => {
                // 不同指标使用不同的日期字段
                const itemDate = item.date || item.time;
                return itemDate === param.time;
              });
              
              if (indicatorData) {
                // 根据指标类型获取对应的值字段
                if (indicatorId === 'banker_control') {
                  indicatorValue = indicatorData.control_degree;
                } else if (indicatorId === 'main_capital_absorption') {
                  indicatorValue = indicatorData.main_capital_absorption;
                } else if (indicatorId === 'main_cost') {
                  indicatorValue = indicatorData.main_cost || indicatorData.cost;
                }
              }
            }
            
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
                    bankerControl: indicatorId === 'banker_control' ? indicatorValue : cursorValues.bankerControl,
                    mainCapitalAbsorption: indicatorId === 'main_capital_absorption' ? indicatorValue : cursorValues.mainCapitalAbsorption,
                  });
                } else {
                  // 即使没有主图数据，也要保留子图指标的值
                  setCursorValues({
                    bankerControl: indicatorId === 'banker_control' ? indicatorValue : cursorValues.bankerControl,
                    mainCapitalAbsorption: indicatorId === 'main_capital_absorption' ? indicatorValue : cursorValues.mainCapitalAbsorption,
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
  }, [visualizationData, selectedIndicators]);

  // 搜索股票
  const handleSearch = async () => {
    const { valid, message, normalized } = validateStockCode(stockCode);
    if (!valid) {
      setInputError(message);
      return;
    }

    setInputError(undefined);
    setIsLoading(true);

    try {
      console.log('Fetching visualization data for:', normalized);
      const response = await visualizationApi.getVisualizationData(
        normalized,
        3650,
        selectedIndicators
      );

      console.log('Visualization data received:', response);
      setVisualizationData(response);

      // 保存搜索历史
      await visualizationApi.saveSearchHistory({
        stock_code: normalized,
        stock_name: response.stock_name,
        days: 3650,
        indicator_types: selectedIndicators,
      });

      // 刷新搜索历史
      await loadSearchHistory();
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
    setSelectedIndicators(filterValidIndicators(['volume', 'main_capital_absorption', 'banker_control', 'main_trading']));
    setSidebarOpen(false);

    setIsLoading(true);

    try {
      const response = await visualizationApi.getVisualizationData(
        item.stock_code,
        item.days || 3650
      );
      setVisualizationData(response);
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
  const handleDeleteHistory = async (recordId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await visualizationApi.deleteSearchHistory(recordId);
      await loadSearchHistory();
    } catch (err) {
      console.error('Failed to delete search history:', err);
    }
  };

  // 回车提交
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && stockCode && !isLoading) {
      handleSearch();
    }
  };

  const sidebarContent = (
    <div className="flex flex-col gap-3 overflow-hidden min-h-0 h-full">
      <div className="p-3 border-b border-white/5">
        <h3 className="text-sm font-medium text-white">搜索历史</h3>
      </div>
      <div className="flex-1 overflow-y-auto px-3">
        {isLoadingHistory ? (
        <div className="flex items-center justify-center py-8">
          <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
        </div>
      ) : searchHistory.length === 0 ? (
        <p className="text-xs text-muted text-center py-4">暂无搜索历史</p>
      ) : (
        <div className="space-y-1.5">
          {searchHistory.map(item => (
            <button
              key={item.id}
              type="button"
              onClick={() => handleHistoryClick(item)}
              className={`history-item w-full text-left ${selectedHistoryId === item.id ? 'active' : ''}`}
            >
              <div className="flex items-center gap-2 w-full">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="font-medium text-white truncate text-xs">
                      {item.stock_name ? `${item.stock_name} (${item.stock_code})` : item.stock_code}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-xs text-muted font-mono">
                      {item.stock_code}
                    </span>
                    <span className="text-xs text-muted/50">·</span>
                    <span className="text-xs text-muted">
                      {new Date(item.searched_at).toLocaleString('zh-CN')}
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={(e) => handleDeleteHistory(item.id, e)}
                  className="p-1 text-muted hover:text-danger transition-colors flex-shrink-0"
                  title="删除"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
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
      style={{ gridTemplateColumns: 'minmax(12px, 1fr) 256px 24px minmax(auto, 896px) minmax(12px, 1fr)', gridTemplateRows: 'auto 1fr auto' }}
    >
      {/* 顶部搜索栏 */}
      <header
        className="md:col-start-2 md:col-end-5 md:row-start-1 py-3 px-3 md:px-0 border-b border-white/5 flex-shrink-0 flex items-center min-w-0 overflow-hidden"
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
            <input
              type="text"
              value={stockCode}
              onChange={(e) => {
                setStockCode(e.target.value.toUpperCase());
                setInputError(undefined);
              }}
              onKeyDown={handleKeyDown}
              placeholder="输入股票代码，如 600519、00700、AAPL"
              disabled={isLoading}
              className={`input-terminal w-full ${inputError ? 'border-danger/50' : ''}`}
            />
            {inputError && (
              <p className="absolute -bottom-4 left-0 text-xs text-danger">{inputError}</p>
            )}
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

      {/* Desktop sidebar */}
      <div className="hidden md:flex col-start-2 row-start-2 flex-col gap-3 overflow-hidden min-h-0">
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
      <section className="md:col-start-4 md:row-start-2 flex-1 overflow-y-auto overflow-x-auto px-3 md:px-0 md:pl-1 min-w-0 min-h-0">
        {/* 图表容器始终存在 */}
        <div className="max-w-6xl">
          {/* 标题 - 只在有数据时显示 */}
          {visualizationData && (
            <div className="mb-4">
              <h2 className="text-xl font-bold text-white">
                {visualizationData.stock_name ? `${visualizationData.stock_name} (${visualizationData.stock_code})` : visualizationData.stock_code}
              </h2>
              <div className="text-xs text-muted mt-1">
                K线数据: {visualizationData.kline_data?.length || 0} 条
              </div>
            </div>
          )}
          
          {/* 主图表 - K线（始终渲染） */}
          <Card variant="default" padding="none" className="mb-4">
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

          {/* 指标子图 - 只在有数据时显示（跳过主力操盘，它在主图显示） */}
          {visualizationData && selectedIndicators.map(indicatorId => {
            if (indicatorId === 'main_trading') return null;
            
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
        className="md:col-start-2 md:col-end-5 md:row-start-3 py-3 px-3 md:px-0 border-t border-white/5 flex-shrink-0"
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
