import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';
import { visualizationApi, type VisualizationResponse } from '../../api/visualization';
import { systemConfigApi } from '../../api/systemConfig';

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
  
  const [cursorValues, setCursorValues] = useState<{
    closePrice?: number;
    mainCost?: number;
  }>({});

  const subChartContainerRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const subChartRefs = useRef<{ [key: string]: lightweightCharts.IChartApi | null }>({});
  const subChartSeriesRefs = useRef<{ [key: string]: any }>({});
  const subChartResizeObservers = useRef<{ [key: string]: ResizeObserver | null }>({});
  const timeSyncSubscription = useRef<any>(null);
  const indicatorsDataRef = useRef<{ [key: string]: any }>({});

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
        ['main_capital_absorption', 'main_cost']
      );
      setVisualizationData(response);
    } catch (err) {
      console.error('Failed to load visualization data:', err);
      setError(err instanceof Error ? err.message : '加载数据失败，请稍后重试');
    } finally {
      setIsLoading(false);
    }
  }, [defaultDays]);

  useEffect(() => {
    loadSystemConfig();
  }, [loadSystemConfig]);

  useEffect(() => {
    loadVisualizationData(stockCode);
  }, [stockCode, defaultDays, loadVisualizationData]);

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
              lineWidth: 2,
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
              <div className="text-sm font-medium text-white">主力吸筹</div>
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
