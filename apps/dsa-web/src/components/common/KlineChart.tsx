import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import * as lightweightCharts from 'lightweight-charts';
import { visualizationApi, type VisualizationResponse } from '../../api/visualization';

const KlineChart: React.FC<{
  stockCode: string;
  stockName?: string;
}> = ({ stockCode, stockName }) => {
  const [visualizationData, setVisualizationData] = useState<VisualizationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const mainChartContainerRef = useRef<HTMLDivElement>(null);
  const mainChartRef = useRef<lightweightCharts.IChartApi | null>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const isChartInitialized = useRef(false);

  const klineDataRef = useRef<any>(null);

  const loadVisualizationData = useCallback(async (code: string) => {
    if (!code) return;
    setIsLoading(true);
    try {
      const response = await visualizationApi.getVisualizationData(code, 150);
      setVisualizationData(response);
    } catch (err) {
      console.error('Failed to load visualization data:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVisualizationData(stockCode);
  }, [stockCode, loadVisualizationData]);

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

      return () => {
        resizeObserver.disconnect();
        if (chart) {
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
      }));

      klineDataRef.current = klineData;
      candlestickSeriesRef.current.setData(klineData);

      if (mainChartRef.current) {
        mainChartRef.current.timeScale().fitContent();
      }
    } catch (error) {
      console.error('Error updating K-line data:', error);
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

      <div ref={mainChartContainerRef} className="w-full rounded-xl overflow-hidden" />
    </div>
  );
};

export default KlineChart;
