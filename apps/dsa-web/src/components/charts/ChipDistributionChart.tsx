import React, { useRef, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import type { ChipDistributionResponse } from '../../api/visualization';

type ChipDistributionChartProps = {
  data: ChipDistributionResponse | null;
  loading?: boolean;
  priceRange?: { min: number; max: number } | null;
  cursorPrice?: number | null;
};

const ChipDistributionChart: React.FC<ChipDistributionChartProps> = ({ data, loading = false, priceRange, cursorPrice }) => {
  const chartRef = useRef<ReactECharts>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // 监听变化，更新图表
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.getEchartsInstance()?.setOption(getOption() as any);
    }
  }, [cursorPrice, data, priceRange]);

  const getOption = () => {
    if (!data || data.price_bins.length === 0) {
      return {};
    }

    let priceBins = data.price_bins;
    let profitVolumes = data.profit_volumes;
    let lossVolumes = data.loss_volumes;
    
    // 如果有价格范围限制，筛选在范围内的数据
    if (priceRange && priceRange.min !== undefined && priceRange.max !== undefined) {
      const filteredIndices = [];
      for (let i = 0; i < priceBins.length; i++) {
        if (priceBins[i] >= priceRange.min && priceBins[i] <= priceRange.max) {
          filteredIndices.push(i);
        }
      }
      if (filteredIndices.length > 0) {
        priceBins = filteredIndices.map(i => priceBins[i]);
        profitVolumes = filteredIndices.map(i => profitVolumes[i]);
        lossVolumes = filteredIndices.map(i => lossVolumes[i]);
      }
    }
    
    const totalVolumes = profitVolumes.map((v, i) => (v || 0) + (lossVolumes[i] || 0));

    const markLines = [];
    
    // 辅助函数：找到最接近的价格索引
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
        label: {
          show: true,
          position: 'end',
          formatter: `成本: ${data.avg_cost.toFixed(2)}`,
          color: '#ff6b6b',
          fontWeight: 'bold',
          backgroundColor: 'rgba(0,0,0,0.7)',
          padding: [2, 5]
        }
      });
    }

    // 添加当前价格线（区分获利盘和套牢盘）
    if (data.current_price !== null && data.current_price !== undefined) {
      markLines.push({
        yAxis: findClosestIndex(data.current_price),
        lineStyle: { color: '#fbbf24', type: 'solid', width: 2 },
        label: {
          show: true,
          position: 'end',
          formatter: `现价: ${data.current_price.toFixed(2)}`,
          color: '#fbbf24',
          fontWeight: 'bold',
          backgroundColor: 'rgba(0,0,0,0.7)',
          padding: [2, 5]
        }
      });
    }



    // 计算当前价格位置，用于区分颜色
    let currentPriceIndex = -1;
    if (data.current_price !== null && data.current_price !== undefined) {
      currentPriceIndex = findClosestIndex(data.current_price);
    }

    // 根据当前价格设置柱子颜色：下方是获利盘（红），上方是套牢盘（蓝）
    const barColors = totalVolumes.map((_v, i) => {
      if (currentPriceIndex >= 0 && i < currentPriceIndex) {
        // 价格低于现价 → 获利盘（红色）
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
        // 价格高于现价 → 套牢盘（蓝色）
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
      backgroundColor: '#0a0e27',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(0,0,0,0.9)',
        borderColor: '#2a2e47',
        borderWidth: 1,
        textStyle: { color: '#fff', fontSize: 12 },
        formatter: (params: any) => {
          const price = params[0]?.name;
          const profit = params[0]?.data?.profit || 0;
          const loss = params[0]?.data?.loss || 0;
          const total = profit + loss;
          return `
            <div style="font-weight: bold; margin-bottom: 4px;">价格: ${price}</div>
            <div>持仓量: ${total >= 10000 ? ((total / 10000).toFixed(2) + '万') : total.toFixed(0)}</div>
            <div style="color: #ef4444;">获利盘: ${total > 0 ? (profit / total * 100).toFixed(1) : 0}%</div>
            <div style="color: #3b82f6;">套牢盘: ${total > 0 ? (loss / total * 100).toFixed(1) : 0}%</div>
          `;
        }
      },
      grid: {
        left: 70,
        right: 20,
        top: 10,
        bottom: 10,
        containLabel: true
      },
      yAxis: {
        type: 'category',
        data: priceBins.map(p => p.toFixed(2)),
        position: 'left',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { 
          color: '#d1d5db',
          fontSize: 11,
          interval: Math.floor(priceBins.length / 20)
        },
        splitLine: { 
          lineStyle: { 
            color: '#1a1e37',
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
            color: '#1a1e37',
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
            profit: profitVolumes[i] || 0,
            loss: lossVolumes[i] || 0
          })),
          itemStyle: {
            color: (params: any) => barColors[params.dataIndex],
            borderRadius: [0, 2, 2, 0]
          },
          barWidth: '88%',
          markLine: {
            data: markLines,
            symbol: ['none', 'none'],
            silent: true
          }
        }
      ],
      animation: true,
      animationDuration: 800,
      animationEasing: 'cubicOut'
    };
  };

  // const totalProfit = data ? data.profit_volumes.reduce((a, b) => a + b, 0) : 0;
  // const totalLoss = data ? data.loss_volumes.reduce((a, b) => a + b, 0) : 0;
  // const total = totalProfit + totalLoss;
  // const profitPercentage = total > 0 ? (totalProfit / total * 100).toFixed(1) : '0';
  // const lossPercentage = total > 0 ? (totalLoss / total * 100).toFixed(1) : '0';

  return (
    <div className="w-full h-full flex flex-col bg-[#0a0e27] rounded-lg border border-[#1a1e37]">
      <div ref={containerRef} className="flex-1 min-h-0">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-gray-500 text-sm">加载中...</div>
          </div>
        ) : !data || data.price_bins.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center p-4">
            <div className="text-center">
              <div className="text-yellow-500 text-sm mb-2">⚠️ 筹码分布无法计算</div>
              <div className="text-gray-400 text-xs">
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
    </div>
  );
};

export default ChipDistributionChart;
