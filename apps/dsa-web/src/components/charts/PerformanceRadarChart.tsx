import React, { useMemo } from 'react'
import type { EChartsOption } from 'echarts'
import { BaseChart } from './BaseChart'
import { chartTheme } from './theme'
import { formatPercent } from './utils'
import type { BacktestReport } from '../../types/backtest-charts'

interface PerformanceRadarChartProps {
  reports: Array<{ name: string; report: BacktestReport }>
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  onChartReady?: (chart: any) => void
}

export const PerformanceRadarChart: React.FC<PerformanceRadarChartProps> = ({
  reports,
  loading = false,
  error = null,
  onRetry,
  height = 400,
  onChartReady,
}) => {
  const option = useMemo<EChartsOption>(() => {
    const indicators = [
      { name: '总收益率', max: 100 },
      { name: '夏普比率', max: 5 },
      { name: '最大回撤', max: 100 },
      { name: '胜率', max: 100 },
      { name: '盈亏比', max: 5 },
      { name: '卡尔马比率', max: 10 },
    ]

    const seriesData = reports.map(({ name, report }) => ({
      name,
      value: [
        Math.min(Math.max(report.totalReturn * 100, 0), 100),
        Math.min(Math.max(report.sharpeRatio, 0), 5),
        Math.min(Math.max((1 - Math.abs(report.maxDrawdown)) * 100, 0), 100),
        Math.min(Math.max(report.winRate * 100, 0), 100),
        Math.min(Math.max(report.profitFactor, 0), 5),
        Math.min(Math.max(report.calmarRatio, 0), 10),
      ],
    }))

    const colors = [
      chartTheme.accentCyan,
      chartTheme.accentPurple,
      chartTheme.successColor,
      chartTheme.warningColor,
    ]

    return {
      tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (!params) return ''
        const report = reports.find((r) => r.name === params.seriesName)?.report
        if (!report) return params.seriesName

          return `<div class="font-medium mb-2">${params.seriesName}</div>
            <div class="space-y-1">
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">总收益率:</span>
                <span class="font-medium">${formatPercent(report.totalReturn)}</span>
              </div>
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">夏普比率:</span>
                <span class="font-medium">${report.sharpeRatio.toFixed(2)}</span>
              </div>
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">最大回撤:</span>
                <span class="font-medium">${formatPercent(report.maxDrawdown)}</span>
              </div>
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">胜率:</span>
                <span class="font-medium">${formatPercent(report.winRate)}</span>
              </div>
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">盈亏比:</span>
                <span class="font-medium">${report.profitFactor.toFixed(2)}</span>
              </div>
              <div class="flex justify-between gap-4">
                <span class="text-gray-400">卡尔马比率:</span>
                <span class="font-medium">${report.calmarRatio.toFixed(2)}</span>
              </div>
            </div>`
        },
      },
      legend: {
        top: 0,
        data: reports.map((r) => r.name),
        textStyle: {
          color: chartTheme.textColorSecondary,
        },
      },
      radar: {
        indicator: indicators,
        shape: 'polygon',
        splitNumber: 5,
        axisName: {
          color: chartTheme.textColorSecondary,
          fontSize: 12,
        },
        splitLine: {
          lineStyle: {
            color: chartTheme.borderColorDim,
          },
        },
        splitArea: {
          show: true,
          areaStyle: {
            color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.01)'],
          },
        },
        axisLine: {
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
      },
      series: [
        {
          name: '绩效指标',
          type: 'radar',
          data: seriesData.map((item, index) => ({
            name: item.name,
            value: item.value,
            symbol: 'circle',
            symbolSize: 6,
            lineStyle: {
              color: colors[index % colors.length],
              width: 2,
            },
            itemStyle: {
              color: colors[index % colors.length],
            },
            areaStyle: {
              color: `${colors[index % colors.length]}22`,
            },
          })),
        },
      ],
    }
  }, [reports])

  return (
    <BaseChart
      option={option}
      loading={loading}
      error={error}
      onRetry={onRetry}
      height={height}
      onChartReady={onChartReady}
    />
  )
}
