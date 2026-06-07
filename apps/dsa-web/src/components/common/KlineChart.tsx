import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';
import { visualizationApi, type VisualizationResponse } from '../../api/visualization';
import { systemConfigApi } from '../../api/systemConfig';
import { TiandaoBandPrimitive } from '../TiandaoBandPrimitive';
import { getCachedKlineChart, setCachedKlineChart } from '../../cache/klineChartCache';

const SUBCHART_HEIGHT = 150;
const DEFAULT_DAYS = 150;

const KlineChart: React.FC<{
  stockCode: string;
  stockName?: string;
}> = ({ stockCode, stockName }) => {
  const [visualizationData, setVisualizationData] = useState<VisualizationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [defaultDays, setDefaultDays] = useState(DEFAULT_DAYS);

  const mainChartContainerRef = useRef<HTMLDivElement>(null);
  const mainChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const isChartInitialized = useRef(false);

  const klineDataRef = useRef<any>(null);
  const earliestDateRef = useRef<any>(null);
  const latestDateRef = useRef<any>(null);
  const isTimeRangeUpdatingRef = useRef(false);
  const isCrosshairUpdatingRef = useRef(false);
  
  const dataFetchedRef = useRef(false);

  const [cursorValues, setCursorValues] = useState<{
    closePrice?: number;
    mainCost?: number;
    mainNetBuyWan?: number;
    mainDirection?: 'inflow' | 'outflow';
    turnoverRatio?: number;
  }>({});

  const subChartContainerRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const subChartRefs = useRef<{ [key: string]: lightweightCharts.IChartApi | null }>({});
  const subChartSeriesRefs = useRef<{ [key: string]: any }>({});
  const subChartResizeObservers = useRef<{ [key: string]: ResizeObserver | null }>({});
  const timeSyncSubscription = useRef<any>(null);
  const indicatorsDataRef = useRef<{ [key: string]: any }>({});

  // 天道指标系列引用
  const tiandaoBbiSeriesRef = useRef<any>(null);
  const tiandaoJinzuanSeriesRef = useRef<any>(null);
  const tiandaoJinniuSeriesRef = useRef<any>(null);
  const tiandaoJinniu2SeriesRef = useRef<any>(null);
  const tiandaoBandPrimitiveRef = useRef<TiandaoBandPrimitive | null>(null);
  const tiandaoStickCandleRef = useRef<any>(null);
  const tiandaoMarkersRef = useRef<any>(null);

  const loadSystemConfig = useCallback(async () => {
    try {
      const config = await systemConfigApi.getConfig(false);
      const daysItem = config?.items?.find(item => item.key === 'stock_selector_update_data_default_days');
      if (daysItem?.value) {
        const days = parseInt(daysItem.value, 10);
        if (!isNaN(days) && days > 0) {
          setDefaultDays(days);
        }
      }
    } catch (err) {
      console.warn('Failed to load system config, using default:', err);
    }
  }, []);

  const loadVisualizationData = useCallback(async (code: string) => {
    if (!code) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await visualizationApi.getVisualizationData(
        code,
        defaultDays,
        ['main_capital_absorption', 'main_capital_distribution', 'main_cost', 'tiandao']
      );
      setVisualizationData(response);
      // 写入缓存：保存最近查看的标的数据，避免重复请求
      setCachedKlineChart({
        stockCode: code,
        stockName: stockName || '',
        visualizationData: response,
      });
    } catch (err) {
      console.error('Failed to load visualization data:', err);
      setError(err instanceof Error ? err.message : '加载数据失败，请稍后重试');
    } finally {
      setIsLoading(false);
    }
  }, [defaultDays, stockName]);

  useEffect(() => {
    loadSystemConfig();
  }, [loadSystemConfig]);

  // 从缓存恢复：若缓存标的代码与当前请求一致，直接使用缓存数据，避免重复请求
  useEffect(() => {
    if (!stockCode) return;
    const cached = getCachedKlineChart();
    if (cached && cached.stockCode === stockCode) {
      console.log('[KlineChart缓存] 从缓存恢复数据:', cached.stockCode, cached.stockName);
      setVisualizationData(cached.visualizationData);
      dataFetchedRef.current = true;
      return;
    }
    // 缓存未命中，发起网络请求
    dataFetchedRef.current = false;
    loadVisualizationData(stockCode);
  }, [stockCode]); // eslint-disable-line react-hooks/exhaustive-deps

  // 当 defaultDays 变更（系统配置加载完毕后），用新天数重新获取数据
  useEffect(() => {
    if (!stockCode || !dataFetchedRef.current) return;
    if (defaultDays === DEFAULT_DAYS) return; // 未加载配置时不触发
    loadVisualizationData(stockCode);
  }, [defaultDays]); // eslint-disable-line react-hooks/exhaustive-deps

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

  const getIndicatorDataItem = (indicatorId: string, time: any) => {
    const indicatorData = indicatorsDataRef.current[indicatorId];
    if (!indicatorData || !indicatorData.data) {
      return null;
    }
    return indicatorData.data.find((item: any) => item.date === time);
  };

  useEffect(() => {
    if (!mainChartContainerRef.current || isChartInitialized.current) return;

    try {
      const chart = lightweightCharts.createChart(mainChartContainerRef.current, {
        width: mainChartContainerRef.current.clientWidth,
        height: 400,
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
          mode: 1,
          vertLine: { color: '#9B7DFF', width: 1, style: 2 },
          horzLine: { color: '#9B7DFF', width: 1, style: 2 },
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
        },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
        handleScale: {
          mouseWheel: true,
          pinch: true,
          axisPressedMouseMove: { time: false, price: false },
          axisDoubleClickReset: { time: true, price: true },
        },
      });

      const candlestickSeries = chart.addSeries(lightweightCharts.CandlestickSeries, {
        upColor: '#FF4444',
        downColor: '#44AA44',
        borderDownColor: '#44AA44',
        borderUpColor: '#FF4444',
        wickDownColor: '#44AA44',
        wickUpColor: '#FF4444',
      });

      mainChartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;
      isChartInitialized.current = true;

      const resizeObserver = new ResizeObserver(entries => {
        if (entries.length === 0 || !mainChartRef.current) return;
        const { width } = entries[0].contentRect;
        mainChartRef.current.applyOptions({ width, height: 400 });
      });
      resizeObserver.observe(mainChartContainerRef.current);

      const handleTimeScaleChange = () => {
        if (!mainChartRef.current || isTimeRangeUpdatingRef.current) return;
        try {
          isTimeRangeUpdatingRef.current = true;
          const timeRange = mainChartRef.current.timeScale().getVisibleRange();
          if (timeRange && timeRange.from && timeRange.to) {
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
      
      const handleCrosshairMove = (param: any) => {
        if (isCrosshairUpdatingRef.current) {
          return;
        }
        
        isCrosshairUpdatingRef.current = true;
        
        try {
          if (param.time) {
            let closePrice: number | undefined;
            let mainCost: number | undefined;
            
            const klinePoint = klineDataRef.current?.find((item: any) => item.time === param.time);
            if (klinePoint) {
              closePrice = klinePoint.close;
            }
            
            const mainCostDataPoint = getIndicatorDataItem('main_cost', param.time);
            if (mainCostDataPoint) {
              mainCost = mainCostDataPoint.main_cost || mainCostDataPoint.cost;
            }
            
            setCursorValues({
              closePrice,
              mainCost,
            });
            
            Object.entries(subChartRefs.current).forEach(([indicatorId, subChart]) => {
              if (subChart && subChartSeriesRefs.current[indicatorId]) {
                try {
                  const series = subChartSeriesRefs.current[indicatorId];
                  
                  let price: number | undefined;
                  if (indicatorId === 'main_capital_absorption') {
                    const dataPoint = getIndicatorDataItem(indicatorId, param.time);
                    if (dataPoint) {
                      const rawValue = dataPoint.main_capital_absorption || 0;
                      price = Math.abs(rawValue) < 1.01 ? 0 : Number(rawValue.toFixed(2));
                    }
                  } else if (indicatorId === 'main_cost') {
                    const dataPoint = getIndicatorDataItem(indicatorId, param.time);
                    if (dataPoint) {
                      price = dataPoint.main_cost || dataPoint.cost;
                    }
                  }
                  
                  if (price !== undefined && !isNaN(price)) {
                    subChart.setCrosshairPosition(price, param.time, series);
                  }
                } catch (e) {
                  console.warn('Failed to sync crosshair to subchart:', e);
                }
              }
            });
          }
        } finally {
          setTimeout(() => {
            isCrosshairUpdatingRef.current = false;
          }, 0);
        }
      };
      
      chart.subscribeCrosshairMove(handleCrosshairMove);
      (chart as any)._crosshairSubscription = handleCrosshairMove;

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

  useEffect(() => {
    if (!visualizationData || !candlestickSeriesRef.current) return;
    if (!visualizationData.kline_data || visualizationData.kline_data.length === 0) return;

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
      candlestickSeriesRef.current.setData(klineData);

      if (klineData.length > 0) {
        earliestDateRef.current = klineData[0].time;
        latestDateRef.current = klineData[klineData.length - 1].time;
      }

      if (mainChartRef.current) {
        mainChartRef.current.timeScale().fitContent();
      }
    } catch (error) {
      console.error('Error updating K-line data:', error);
    }
  }, [visualizationData]);

  // 更新天道指标系列（主图叠加）
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
      if (tiandaoBandPrimitiveRef.current) {
        candlestickSeriesRef.current?.detachPrimitive(tiandaoBandPrimitiveRef.current);
        tiandaoBandPrimitiveRef.current = null;
      }
      if (tiandaoStickCandleRef.current) {
        mainChartRef.current.removeSeries(tiandaoStickCandleRef.current);
        tiandaoStickCandleRef.current = null;
      }

      const tiandaoIndicator = visualizationData.indicators.find(
        (ind: any) => ind.indicator_type === 'tiandao'
      );

      // 清理旧的买入信号标记
      if (tiandaoMarkersRef.current) {
        tiandaoMarkersRef.current.setMarkers([]);
        tiandaoMarkersRef.current = null;
      }

      if (tiandaoIndicator && tiandaoIndicator.data && tiandaoIndicator.data.length > 0) {
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

        // DRAWBAND 红绿带状区域：使用 ISeriesPrimitive 自定义绘制
        const bandPrimitive = new TiandaoBandPrimitive();
        bandPrimitive.data = tiandaoIndicator.data as any;
        candlestickSeriesRef.current!.attachPrimitive(bandPrimitive);
        tiandaoBandPrimitiveRef.current = bandPrimitive;

        // STICKLINE 黄色柱体
        const klineDateMap = new Map<string, any>();
        if (visualizationData.kline_data) {
          visualizationData.kline_data.forEach((kl: any) => {
            klineDateMap.set(kl.date || kl.time, kl);
          });
        }

        const overlayCandleData: any[] = [];
        tiandaoIndicator.data.forEach((item: any) => {
          const jinzuan = item.td_jinzuan != null ? Number(item.td_jinzuan) : null;
          const kl = klineDateMap.get(item.date);
          if (jinzuan == null || !kl) return;

          const high = kl.high != null ? Number(kl.high) : (kl.High != null ? Number(kl.High) : null);
          const low = kl.low != null ? Number(kl.low) : (kl.Low != null ? Number(kl.Low) : null);
          const open = kl.open != null ? Number(kl.open) : (kl.Open != null ? Number(kl.Open) : null);
          const close = kl.close != null ? Number(kl.close) : (kl.Close != null ? Number(kl.Close) : null);

          if (high == null || low == null || open == null || close == null) return;

          const maxOC = Math.max(open, close);
          const minOC = Math.min(open, close);

          if (jinzuan > low && jinzuan < high) {
            const stickLow = Math.min(minOC, jinzuan);
            overlayCandleData.push({
              time: item.date,
              open: stickLow,
              high: jinzuan,
              low: stickLow,
              close: jinzuan,
            });
          } else if (jinzuan >= high) {
            overlayCandleData.push({
              time: item.date,
              open: minOC,
              high: maxOC,
              low: minOC,
              close: maxOC,
            });
          }
        });

        if (overlayCandleData.length > 0) {
          const overlaySeries = mainChartRef.current!.addSeries(lightweightCharts.CandlestickSeries, {
            upColor: '#FFD700',
            downColor: '#FFD700',
            borderUpColor: '#FFD700',
            borderDownColor: '#FFD700',
            wickUpColor: 'transparent',
            wickDownColor: 'transparent',
            priceLineVisible: false,
            lastValueVisible: false,
          });
          overlaySeries.setData(overlayCandleData);
          tiandaoStickCandleRef.current = overlaySeries;
        }

        // 买入信号标记（td_xg 和 td_xg2）
        if (candlestickSeriesRef.current) {
          const markers: any[] = [];

          tiandaoIndicator.data.forEach((item: any) => {
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
  }, [visualizationData]);

  useEffect(() => {
    Object.keys(subChartResizeObservers.current).forEach(indicatorId => {
      const observer = subChartResizeObservers.current[indicatorId];
      if (observer) {
        observer.disconnect();
      }
    });
    subChartResizeObservers.current = {};

    Object.values(subChartRefs.current).forEach(chart => {
      if (chart) {
        if ((chart as any)._crosshairSubscription) {
          chart.unsubscribeCrosshairMove((chart as any)._crosshairSubscription);
        }
        chart.remove();
      }
    });
    subChartRefs.current = {};
    subChartSeriesRefs.current = {};

    if (!visualizationData) return;

    indicatorsDataRef.current = {};
    visualizationData.indicators.forEach(ind => {
      indicatorsDataRef.current[ind.indicator_type] = {
        ...ind,
        data: filterDataByTimeRange(ind.data, 'date')
      };
    });

    const indicatorIds = ['main_capital_absorption', 'main_cost'];
    const createdCharts: Array<{ id: string; chart: any }> = [];

    indicatorIds.forEach(indicatorId => {
      const indicatorData = visualizationData.indicators.find(ind => ind.indicator_type === indicatorId);
      if (!indicatorData || !indicatorData.data || indicatorData.data.length === 0) return;

      const containerRef = subChartContainerRefs.current[indicatorId];
      if (!containerRef) return;

      try {
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
            mode: 1,
            vertLine: { color: '#9B7DFF', width: 1, style: 2 },
            horzLine: { color: '#9B7DFF', width: 1, style: 2 },
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
          },
          handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
          handleScale: {
            mouseWheel: true,
            pinch: true,
            axisPressedMouseMove: { time: false, price: false },
            axisDoubleClickReset: { time: true, price: true },
          },
        });

        subChartRefs.current[indicatorId] = chart;
        createdCharts.push({ id: indicatorId, chart });

        if (indicatorId === 'main_capital_absorption') {
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
        } else if (indicatorId === 'main_cost') {
          const filteredIndicatorData = filterDataByTimeRange(indicatorData.data, 'date');
          
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

        const handleSubChartTimeScaleChange = () => {
          if (!chart || isTimeRangeUpdatingRef.current) return;
          
          try {
            isTimeRangeUpdatingRef.current = true;
            const timeRange = chart.timeScale().getVisibleRange();
            if (timeRange && timeRange.from && timeRange.to && mainChartRef.current) {
              mainChartRef.current.timeScale().setVisibleRange(timeRange);
              
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
            console.warn('Failed to get time range from subchart:', e);
          } finally {
            setTimeout(() => {
              isTimeRangeUpdatingRef.current = false;
            }, 0);
          }
        };

        const handleSubChartCrosshairMove = (param: any) => {
          if (isCrosshairUpdatingRef.current) {
            return;
          }
          
          isCrosshairUpdatingRef.current = true;
          
          try {
            if (param.time) {
              let closePrice: number | undefined;
              let mainCost: number | undefined;
              let mainNetBuyWan: number | undefined;
              let mainDirection: 'inflow' | 'outflow' | undefined;
              let turnoverRatio: number | undefined;
              
              const klinePoint = klineDataRef.current?.find((item: any) => item.time === param.time);
              if (klinePoint) {
                closePrice = klinePoint.close;
              }
              
              const mainCostDataPoint = getIndicatorDataItem('main_cost', param.time);
              if (mainCostDataPoint) {
                mainCost = mainCostDataPoint.main_cost || mainCostDataPoint.cost;
                mainNetBuyWan = mainCostDataPoint.main_net_buy_wan;
                mainDirection = mainCostDataPoint.main_direction;
                turnoverRatio = mainCostDataPoint.turnover_ratio;
              }
              
              setCursorValues({
                closePrice,
                mainCost,
                mainNetBuyWan,
                mainDirection,
                turnoverRatio,
              });
              
              if (mainChartRef.current && candlestickSeriesRef.current) {
                try {
                  if (closePrice !== undefined && !isNaN(closePrice)) {
                    mainChartRef.current.setCrosshairPosition(closePrice, param.time, candlestickSeriesRef.current);
                  }
                } catch (e) {
                  console.warn('Failed to sync crosshair to main chart:', e);
                }
              }
              
              Object.entries(subChartRefs.current).forEach(([otherIndicatorId, subChart]) => {
                if (otherIndicatorId !== indicatorId && subChart && subChartSeriesRefs.current[otherIndicatorId]) {
                  try {
                    const series = subChartSeriesRefs.current[otherIndicatorId];
                    
                    let price: number | undefined;
                    if (otherIndicatorId === 'main_capital_absorption') {
                      const dataPoint = getIndicatorDataItem(otherIndicatorId, param.time);
                      if (dataPoint) {
                        const rawValue = dataPoint.main_capital_absorption || 0;
                        price = Math.abs(rawValue) < 1.01 ? 0 : Number(rawValue.toFixed(2));
                      }
                    } else if (otherIndicatorId === 'main_cost') {
                      const dataPoint = getIndicatorDataItem(otherIndicatorId, param.time);
                      if (dataPoint) {
                        price = dataPoint.main_cost || dataPoint.cost;
                      }
                    }
                    
                    if (price !== undefined && !isNaN(price)) {
                      subChart.setCrosshairPosition(price, param.time, series);
                    }
                  } catch (e) {
                    console.warn('Failed to sync crosshair to other subchart:', e);
                  }
                }
              });
            }
          } finally {
            setTimeout(() => {
              isCrosshairUpdatingRef.current = false;
            }, 0);
          }
        };

        chart.timeScale().subscribeVisibleTimeRangeChange(handleSubChartTimeScaleChange);
        chart.subscribeCrosshairMove(handleSubChartCrosshairMove);
        (chart as any)._crosshairSubscription = handleSubChartCrosshairMove;
      } catch (error) {
        console.error(`Error creating subchart for ${indicatorId}:`, error);
      }
    });

    if (mainChartRef.current) {
      const timeRange = mainChartRef.current.timeScale().getVisibleRange();
      if (timeRange) {
        createdCharts.forEach(({ chart }) => {
          try {
            chart.timeScale().setVisibleRange(timeRange);
          } catch (e) {
            console.warn('Failed to set initial time range on subchart:', e);
          }
        });
      }
    }
  }, [visualizationData]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <div className="text-sm font-medium text-white">
          {stockCode} {stockName && <span className="text-muted">{stockName}</span>}
        </div>
        {isLoading && (
          <div className="text-xs text-muted flex items-center gap-1">
            <div className="w-3 h-3 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
            加载中...
          </div>
        )}
      </div>

      {error ? (
        <div className="flex flex-col items-center justify-center p-8 bg-white/5 rounded-xl gap-4">
          <div className="text-red-400 text-lg font-medium">
            {error}
          </div>
          <button
            onClick={() => loadVisualizationData(stockCode)}
            className="px-4 py-2 bg-cyan/20 text-cyan rounded-lg hover:bg-cyan/30 transition-colors text-sm"
          >
            重试
          </button>
        </div>
      ) : (
        <>
          <div ref={mainChartContainerRef} className="w-full rounded-xl overflow-hidden" />

          <div className="flex flex-col gap-2">
            <div className="flex items-center px-1">
              <div className="text-sm font-medium text-white">主力进出</div>
              <div className="flex items-center gap-2 ml-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#AA44FF' }} />
                  <span className="text-xs text-white">吸筹</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#44AA44' }} />
                  <span className="text-xs text-white">出货</span>
                </div>
              </div>
            </div>
            <div
              ref={el => { subChartContainerRefs.current['main_capital_absorption'] = el; }}
              className="w-full rounded-xl overflow-hidden"
            />

            <div className="flex items-center px-1">
              <div className="text-sm font-medium text-white">主力成本</div>
              {cursorValues.mainCost !== undefined && cursorValues.mainCost !== null && !isNaN(cursorValues.mainCost) && cursorValues.closePrice !== undefined && cursorValues.closePrice !== null && !isNaN(cursorValues.closePrice) && (
                <span className="flex items-center gap-2 ml-2">
                  <span className="text-xs font-mono text-cyan">
                    {cursorValues.mainCost.toFixed(2)}
                  </span>
                  <span className={`text-xs font-mono ${((cursorValues.closePrice - cursorValues.mainCost) / cursorValues.mainCost * 100) >= 0 ? 'text-[#FF4444]' : 'text-[#44AA44]'}`}>
                    {((cursorValues.closePrice - cursorValues.mainCost) / cursorValues.mainCost * 100).toFixed(2)}%
                  </span>
                </span>
              )}
              {cursorValues.mainCost !== undefined && cursorValues.mainCost !== null && !isNaN(cursorValues.mainCost) && (cursorValues.closePrice === undefined || cursorValues.closePrice === null || isNaN(cursorValues.closePrice)) && (
                <span className="text-xs font-mono text-cyan ml-2">
                  {cursorValues.mainCost.toFixed(2)}
                </span>
              )}
              {(() => {
                const buyWan = cursorValues.mainNetBuyWan;
                const ratio = cursorValues.turnoverRatio;
                const isInflow = cursorValues.mainDirection === 'inflow';
                if (buyWan == null || isNaN(buyWan)) return null;
                return (
                  <span className="flex items-center gap-2 ml-2">
                    <span className={`text-xs font-mono ${isInflow ? 'text-[#FF4444]' : 'text-[#44AA44]'}`}>
                      {isInflow ? '▲' : '▼'}净{isInflow ? '买' : '卖'}{Math.abs(buyWan).toFixed(0)}万
                    </span>
                    {ratio != null && !isNaN(ratio) && (
                      <span className="text-xs font-mono text-muted">
                        占比{ratio.toFixed(1)}%
                      </span>
                    )}
                  </span>
                );
              })()}
            </div>
            <div
              ref={el => { subChartContainerRefs.current['main_cost'] = el; }}
              className="w-full rounded-xl overflow-hidden"
            />
          </div>
        </>
      )}
    </div>
  );
};

export default KlineChart;
