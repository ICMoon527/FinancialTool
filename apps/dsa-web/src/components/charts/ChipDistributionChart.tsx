import React, { useRef, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import type { ChipDistributionResponse } from '../../api/visualization';

type ChipDistributionChartProps = {
  data: ChipDistributionResponse | null;
  loading?: boolean;
  priceRange?: { min: number; max: number } | null;
  cursorPrice?: number | null;
};

const ChipDistributionChart: React.FC<ChipDistributionChartProps> = ({ 
  data, 
  loading = false, 
  priceRange, 
  cursorPrice 
}) => {
  const chartRef = useRef<ReactECharts>(null);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.getEchartsInstance()?.setOption(getOption() as any);
    }
  }, [cursorPrice, data, priceRange]);

  const getOption = () => {
    if (!data || data.price_bins.length === 0) {
      return {};
    }

    const priceBins = data.price_bins;
    const profitVolumes = data.profit_volumes;
    const lossVolumes = data.loss_volumes;
    const totalVolumes = profitVolumes.map((v, i) => (v || 0) + (lossVolumes[i] || 0));

    const markLines = [];
    
    const findClosestIndex = (targetPrice: number) => {
      let closestIndex = 0;
      let minDiff = Infinity;
      for (let i = 0; i < priceBins.length; i++) {
        const diff = Math.abs(priceBins[i] - targetPrice);
        if (diff < minDiff) {
          minDiff = diff;
          closestIndex = i;
        }
      }
      return closestIndex;
    };
    
    if (data.avg_cost !== null && data.avg_cost !== undefined) {
      markLines.push({
        yAxis: findClosestIndex(data.avg_cost),
        lineStyle: { color: '#ff6b6b', type: 'dashed', width: 1.5 },
        label: { show: false }
      });
    }

    if (data.current_price !== null && data.current_price !== undefined) {
      markLines.push({
        yAxis: findClosestIndex(data.current_price),
        lineStyle: { color: '#fbbf24', type: 'dashed', width: 1 },
        label: { show: false }
      });
    }

    let currentPriceIndex = -1;
    if (data.current_price !== null && data.current_price !== undefined) {
      currentPriceIndex = findClosestIndex(data.current_price);
    }

    const barColors = totalVolumes.map((_v, i) => {
      if (currentPriceIndex >= 0 && i < currentPriceIndex) {
        return {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 1,
          y2: 0,
          colorStops: [
            { offset: 0, color: 'rgba(239, 68, 68, 0.95)' },
            { offset: 0.5, color: 'rgba(239, 68, 68, 0.6)' },
            { offset: 1, color: 'rgba(239, 68, 68, 0.25)' }
          ]
        };
      } else {
        return {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 1,
          y2: 0,
          colorStops: [
            { offset: 0, color: 'rgba(59, 130, 246, 0.25)' },
            { offset: 0.5, color: 'rgba(59, 130, 246, 0.6)' },
            { offset: 1, color: 'rgba(59, 130, 246, 0.95)' }
          ]
        };
      }
    });

    return {
      backgroundColor: '#1a1a2e',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(0,0,0,0.9)',
        borderColor: '#2a2e47',
        borderWidth: 1,
        textStyle: { color: '#fff', fontSize: 12 },
        formatter: (params: any) => {
          if (!params || params.length === 0) return '';
          const dataIndex = params[0].dataIndex;
          const price = priceBins[dataIndex];
          const profit = profitVolumes[dataIndex] || 0;
          const loss = lossVolumes[dataIndex] || 0;
          const total = profit + loss;
          return `
            <div style="font-weight: bold; margin-bottom: 4px;">价格: ${price.toFixed(2)}</div>
            <div>持仓量: ${total >= 10000 ? ((total / 10000).toFixed(2) + '万') : total.toFixed(0)}</div>
            <div style="color: #ef4444;">获利盘: ${total > 0 ? (profit / total * 100).toFixed(1) : 0}%</div>
            <div style="color: #3b82f6;">套牢盘: ${total > 0 ? (loss / total * 100).toFixed(1) : 0}%</div>
          `;
        }
      },
      grid: {
        left: 0,
        right: 20,
        top: 10,
        bottom: 10,
        containLabel: false
      },
      yAxis: {
        type: 'category',
        data: priceBins.map(p => p.toFixed(2)),
        position: 'left',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
        splitLine: { 
          lineStyle: { 
            color: '#2b2b43',
            type: 'dashed'
          } 
        }
      },
      xAxis: {
        type: 'value',
        position: 'top',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { show: false },
        splitLine: { 
          lineStyle: { 
            color: '#2b2b43',
            type: 'dashed'
          } 
        }
      },
      series: [
        {
          name: '筹码分布',
          type: 'bar',
          data: totalVolumes.map((v, i) => ({
            value: v,
            itemStyle: {
              color: barColors[i],
              borderRadius: [0, 2, 2, 0]
            }
          })),
          barWidth: '85%',
          markLine: {
            data: markLines,
            symbol: ['none', 'none'],
            silent: true
          }
        }
      ],
      animation: false,
      dataZoom: priceRange && priceRange.min !== undefined && priceRange.max !== undefined ? [
        {
          type: 'inside',
          yAxisIndex: 0,
          startValue: (() => {
            let idx = priceBins.findIndex(p => p >= priceRange.min);
            return idx === -1 ? 0 : idx;
          })(),
          endValue: (() => {
            let idx = priceBins.findIndex(p => p > priceRange.max);
            if (idx === -1) return priceBins.length - 1;
            return Math.max(0, idx - 1);
          })(),
          zoomLock: true
        }
      ] : undefined
    };
  };

  return (
    <div className="w-full h-full">
      {loading ? (
        <div className="w-full h-full flex items-center justify-center bg-[#1a1a2e]">
          <div className="text-xs text-muted">加载中...</div>
        </div>
      ) : !data || data.price_bins.length === 0 ? (
        <div className="w-full h-full flex items-center justify-center p-4 bg-[#1a1a2e]">
          <div className="text-center">
            <div className="text-xs text-yellow-500 mb-2">⚠️ 筹码分布无法计算</div>
            <div className="text-xs text-muted">
              需要真实换手率数据<br/>
              请确保数据包含换手率信息
            </div>
          </div>
        </div>
      ) : (
        <ReactECharts
          ref={chartRef}
          option={getOption()}
          style={{ height: '100%', width: '100%' }}
          notMerge={true}
          lazyUpdate={true}
        />
      )}
    </div>
  );
};

export default ChipDistributionChart;
