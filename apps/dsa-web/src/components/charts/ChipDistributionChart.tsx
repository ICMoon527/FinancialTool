import React, { useRef, useEffect, useState } from 'react';
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
  const yAxisCanvasRef = useRef<HTMLCanvasElement>(null);
  const [priceBins, setPriceBins] = useState<number[]>([]);

  // 监听变化，更新图表和Y轴
  useEffect(() => {
    if (data && data.price_bins) {
      let bins = data.price_bins;
      
      // 如果有价格范围限制，筛选在范围内的数据
      if (priceRange && priceRange.min !== undefined && priceRange.max !== undefined) {
        bins = bins.filter(p => p >= priceRange.min && p <= priceRange.max);
      }
      
      setPriceBins(bins);
    }
    
    if (chartRef.current) {
      chartRef.current.getEchartsInstance()?.setOption(getOption() as any);
    }
  }, [cursorPrice, data, priceRange]);

  // 绘制独立的Y轴
  useEffect(() => {
    const canvas = yAxisCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置canvas尺寸
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 60 * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    // 清空画布
    ctx.clearRect(0, 0, 60, canvas.offsetHeight);
    
    // 设置样式
    ctx.fillStyle = '#d1d4dc';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const height = canvas.offsetHeight;
    
    // 确定要显示的价格范围
    let displayMin = priceBins[0];
    let displayMax = priceBins[priceBins.length - 1];
    
    if (priceRange && priceRange.min !== undefined && priceRange.max !== undefined) {
      displayMin = priceRange.min;
      displayMax = priceRange.max;
    }
    
    // 生成均匀分布的价格标签（约20个标签）
    const numLabels = Math.min(20, Math.floor(height / 20));
    const priceStep = (displayMax - displayMin) / numLabels;
    
    // 绘制价格标签（从下往上：min在底部，max在顶部）
    for (let i = 0; i <= numLabels; i++) {
      const price = displayMin + priceStep * i;
      
      // 计算价格在当前显示范围中的位置
      const normalizedY = (price - displayMin) / (displayMax - displayMin);
      const y = height - normalizedY * height;  // 反转Y坐标
      
      // 只绘制在画布范围内的标签
      if (y >= 5 && y <= height - 5) {
        ctx.fillText(price.toFixed(2), 55, y);
      }
    }
    
  }, [priceBins, priceRange]);

  const getOption = () => {
    if (!data || data.price_bins.length === 0) {
      return {};
    }

    // ECharts 始终使用完整价格数据（不筛选），这样才能与 Canvas 标签范围对齐
    const priceBins = data.price_bins;
    const profitVolumes = data.profit_volumes;
    const lossVolumes = data.loss_volumes;
    
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
          show: false
        }
      });
    }

    // 添加当前价格线（区分获利盘和套牢盘）
    if (data.current_price !== null && data.current_price !== undefined) {
      markLines.push({
        yAxis: findClosestIndex(data.current_price),
        lineStyle: { color: '#fbbf24', type: 'dashed', width: 1 },
        label: {
          show: false
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
      backgroundColor: '#1a1a2e',
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
      animation: false,
      // 使用 dataZoom 控制 Y轴显示范围，与主图同步
      ...(() => {
        if (!priceRange || priceRange.min === undefined || priceRange.max === undefined) {
          return {};
        }
        
        // 找到范围对应的索引
        let startIndex = priceBins.findIndex(p => p >= priceRange.min);
        if (startIndex === -1) startIndex = 0;
        
        let endIndex = priceBins.findIndex(p => p > priceRange.max);
        if (endIndex === -1) endIndex = priceBins.length - 1;
        else endIndex = Math.max(0, endIndex - 1);
        
        return {
          dataZoom: [
            {
              type: 'inside',
              yAxisIndex: 0,
              startValue: startIndex,
              endValue: endIndex,
              zoomLock: true  // 锁定缩放，只能通过主图控制
            }
          ]
        };
      })()
    };
  };

  return (
    <div className="w-full h-full flex">
      {/* 独立的Y轴 Canvas */}
      <div className="w-[60px] flex-shrink-0 h-full bg-[#1a1a2e] relative">
        <canvas 
          ref={yAxisCanvasRef} 
          className="w-full h-full"
        />
      </div>
      
      {/* 主图表区域 */}
      <div className="flex-1 h-full min-w-0">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center bg-[#1a1a2e]">
            <div className="text-gray-500 text-sm">加载中...</div>
          </div>
        ) : !data || data.price_bins.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center p-4 bg-[#1a1a2e]">
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
