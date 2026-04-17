import React, { useMemo, useState, useCallback } from 'react'
import type { EChartsOption } from 'echarts'
import * as echarts from 'echarts'
import { BaseChart } from './BaseChart'
import { chartTheme } from './theme'
import { formatPercent } from './utils'

interface ReturnDistributionChartProps {
  data: {
    dailyReturns: number[]
    weeklyReturns: number[]
    monthlyReturns: number[]
  }
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  onChartReady?: (chart: echarts.ECharts) => void
}

type ChartType = 'histogram' | 'density'
type PeriodType = 'daily' | 'weekly' | 'monthly'

export const ReturnDistributionChart: React.FC<ReturnDistributionChartProps> = ({
  data,
  loading = false,
  error = null,
  onRetry,
  height = 400,
  onChartReady,
}) => {
  const [chartType, setChartType] = useState<ChartType>('histogram')
  const [period, setPeriod] = useState<PeriodType>('daily')

  const getReturns = useCallback(() => {
    switch (period) {
      case 'daily':
        return data.dailyReturns
      case 'weekly':
        return data.weeklyReturns
      case 'monthly':
        return data.monthlyReturns
      default:
        return data.dailyReturns
    }
  }, [data, period])

  const calculateStats = useCallback((returns: number[]) => {
    if (returns.length === 0) {
      return { mean: 0, std: 0, skewness: 0, kurtosis: 0 }
    }

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
    const squaredDiffs = returns.map((r) => (r - mean) ** 2)
    const variance = squaredDiffs.reduce((sum, d) => sum + d, 0) / returns.length
    const std = Math.sqrt(variance)

    const cubedDiffs = returns.map((r) => (r - mean) ** 3)
    const skewness =
      std !== 0
        ? cubedDiffs.reduce((sum, d) => sum + d, 0) / returns.length / std ** 3
        : 0

    const fourthDiffs = returns.map((r) => (r - mean) ** 4)
    const kurtosis =
      std !== 0
        ? fourthDiffs.reduce((sum, d) => sum + d, 0) / returns.length / std ** 4 - 3
        : 0

    return { mean, std, skewness, kurtosis }
  }, [])

  const transformToHistogram = useCallback((returns: number[], binCount = 20) => {
    if (returns.length === 0) {
      return { bins: [], values: [] }
    }

    const minReturn = Math.min(...returns)
    const maxReturn = Math.max(...returns)
    const binWidth = (maxReturn - minReturn) / binCount

    const bins: string[] = []
    const values: number[] = new Array(binCount).fill(0)

    for (let i = 0; i < binCount; i++) {
      const binStart = minReturn + i * binWidth
      const binEnd = minReturn + (i + 1) * binWidth
      bins.push(`${((binStart + binEnd) / 2 * 100).toFixed(1)}%`)
    }

    returns.forEach((ret) => {
      let binIndex = Math.floor((ret - minReturn) / binWidth)
      binIndex = Math.max(0, Math.min(binIndex, binCount - 1))
      values[binIndex]++
    })

    return { bins, values }
  }, [])

  const transformToDensity = useCallback((returns: number[], binCount = 50) => {
    if (returns.length === 0) {
      return { bins: [], values: [] }
    }

    const minReturn = Math.min(...returns)
    const maxReturn = Math.max(...returns)
    const binWidth = (maxReturn - minReturn) / binCount

    const bins: string[] = []
    const values: number[] = new Array(binCount).fill(0)

    for (let i = 0; i < binCount; i++) {
      const binStart = minReturn + i * binWidth
      const binEnd = minReturn + (i + 1) * binWidth
      bins.push(`${((binStart + binEnd) / 2 * 100).toFixed(1)}%`)
    }

    const bandwidth = binWidth * 2

    for (let i = 0; i < binCount; i++) {
      const binCenter = minReturn + (i + 0.5) * binWidth
      let density = 0
      returns.forEach((ret) => {
        const diff = (ret - binCenter) / bandwidth
        density += Math.exp(-0.5 * diff * diff) / Math.sqrt(2 * Math.PI)
      })
      values[i] = density / (returns.length * bandwidth)
    }

    return { bins, values }
  }, [])

  const option = useMemo<EChartsOption>(() => {
    const returns = getReturns()
    const stats = calculateStats(returns)
    const chartData =
      chartType === 'histogram'
        ? transformToHistogram(returns)
        : transformToDensity(returns)

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const bin = params[0].axisValue
          const value = params[0].value
          return `<div class="font-medium mb-1">收益率区间: ${bin}</div>
            <div class="flex items-center gap-2">
              <span class="text-gray-400">${chartType === 'histogram' ? '频数' : '密度'}:</span>
              <span class="font-medium">${value.toFixed(chartType === 'histogram' ? 0 : 4)}</span>
            </div>`
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '20%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: chartData.bins,
        axisLabel: {
          rotate: 45,
          fontSize: 10,
        },
      },
      yAxis: {
        type: 'value',
        splitLine: {
          lineStyle: {
            color: chartTheme.gridColor,
          },
        },
      },
      series: [
        {
          name: chartType === 'histogram' ? '频数' : '密度',
          type: chartType === 'histogram' ? 'bar' : 'line',
          data: chartData.values,
          smooth: chartType === 'density',
          itemStyle: {
            color: chartTheme.accentCyan,
          },
          areaStyle:
            chartType === 'density'
              ? {
                  color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: `${chartTheme.accentCyan}33` },
                    { offset: 1, color: `${chartTheme.accentCyan}00` },
                  ]),
                }
              : undefined,
        },
      ],
      graphic: [
        {
          type: 'text',
          left: 10,
          top: 10,
          style: {
            text: `均值: ${formatPercent(stats.mean)}\n标准差: ${formatPercent(stats.std)}\n偏度: ${stats.skewness.toFixed(3)}\n峰度: ${stats.kurtosis.toFixed(3)}`,
            fill: chartTheme.textColorSecondary,
            fontSize: 12,
            lineHeight: 18,
          },
        },
      ],
    }
  }, [
    getReturns,
    calculateStats,
    transformToHistogram,
    transformToDensity,
    chartType,
  ])

  return (
    <div className="flex flex-col h-full">
      <div className="flex gap-2 mb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setChartType('histogram')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              chartType === 'histogram'
                ? 'bg-[#00d4ff] text-black'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            直方图
          </button>
          <button
            onClick={() => setChartType('density')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              chartType === 'density'
                ? 'bg-[#00d4ff] text-black'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            密度图
          </button>
        </div>
        <div className="flex items-center gap-2 ml-4">
          <button
            onClick={() => setPeriod('daily')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              period === 'daily'
                ? 'bg-[#6f61f1] text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            日
          </button>
          <button
            onClick={() => setPeriod('weekly')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              period === 'weekly'
                ? 'bg-[#6f61f1] text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            周
          </button>
          <button
            onClick={() => setPeriod('monthly')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              period === 'monthly'
                ? 'bg-[#6f61f1] text-white'
                : 'bg-white/10 text-gray-400 hover:bg-white/20'
            }`}
          >
            月
          </button>
        </div>
      </div>
      <div className="flex-1">
        <BaseChart
          option={option}
          loading={loading}
          error={error}
          onRetry={onRetry}
          height={height}
          onChartReady={onChartReady}
        />
      </div>
    </div>
  )
}
