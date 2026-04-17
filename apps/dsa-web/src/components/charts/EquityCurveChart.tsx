import React, { useMemo, useRef, useCallback } from 'react'
import type { EChartsOption } from 'echarts'
import * as echarts from 'echarts'
import { BaseChart } from './BaseChart'
import { chartTheme } from './theme'
import { useChartInstance } from './hooks'
import { formatPercent, formatCurrency } from './utils'
import type { EquityChartData } from './utils'

interface EquityCurveChartProps {
  data: EquityChartData
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  onChartReady?: (chart: echarts.ECharts) => void
  onDataZoom?: (params: any) => void
}

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  loading = false,
  error = null,
  onRetry,
  height = 400,
  onChartReady,
  onDataZoom,
}) => {
  const { setChartInstance } = useChartInstance()
  const dataZoomStartEndRef = useRef<{ start: number; end: number }>({ start: 0, end: 100 })

  const handleChartReady = useCallback(
    (chart: echarts.ECharts) => {
      setChartInstance(chart)
      if (onChartReady) {
        onChartReady(chart)
      }
    },
    [setChartInstance, onChartReady]
  )

  const handleDataZoom = useCallback(
    (params: any) => {
      if (params.batch && params.batch.length > 0) {
        dataZoomStartEndRef.current = {
          start: params.batch[0].start,
          end: params.batch[0].end,
        }
      }
      if (onDataZoom) {
        onDataZoom(params)
      }
    },
    [onDataZoom]
  )

  const option = useMemo<EChartsOption>(() => {
    const series: any[] = [
      {
        name: '策略净值',
        type: 'line',
        data: data.equity,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2.5,
          color: chartTheme.accentCyan,
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: `${chartTheme.accentCyan}33` },
            { offset: 1, color: `${chartTheme.accentCyan}00` },
          ]),
        },
        markPoint: {
          data: [
            {
              type: 'max',
              name: '最大值',
              label: {
                formatter: (params: any) => formatCurrency(params.value as number),
              },
            },
            {
              type: 'min',
              name: '最小值',
              label: {
                formatter: (params: any) => formatCurrency(params.value as number),
              },
            },
          ],
        },
      },
    ]

    if (data.benchmark) {
      series.push({
        name: '基准净值',
        type: 'line',
        data: data.benchmark,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          color: chartTheme.accentPurple,
          type: 'dashed',
        },
      })
    }

    const startValue = data.equity[0]?.[1] || 0
    const endValue = data.equity[data.equity.length - 1]?.[1] || 0
    const totalReturn = startValue > 0 ? (endValue - startValue) / startValue : 0

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const date = params[0].axisValue
          let result = `<div class="font-medium mb-1">${date}</div>`
          params.forEach((param: any) => {
            const value = param.value as [string, number]
            const val = value[1]
            result += `<div class="flex items-center gap-2">
              <span class="inline-block w-2 h-2 rounded-full" style="background-color: ${param.color}"></span>
              <span class="text-gray-400">${param.seriesName}:</span>
              <span class="font-medium">${formatCurrency(val)}</span>
            </div>`
          })
          return result
        },
      },
      legend: {
        top: 0,
        textStyle: {
          color: chartTheme.textColorSecondary,
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: data.dates,
        axisLabel: {
          formatter: (value: string) => {
            return value
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => formatCurrency(value),
        },
        splitLine: {
          lineStyle: {
            color: chartTheme.gridColor,
          },
        },
      },
      dataZoom: [
        {
          type: 'inside',
          start: dataZoomStartEndRef.current.start,
          end: dataZoomStartEndRef.current.end,
        },
        {
          type: 'slider',
          start: dataZoomStartEndRef.current.start,
          end: dataZoomStartEndRef.current.end,
          bottom: 10,
        },
      ],
      series,
      graphic: [
        {
          type: 'text',
          left: 'center',
          top: 10,
          style: {
            text: `累计收益率: ${formatPercent(totalReturn)}`,
            fill: totalReturn >= 0 ? chartTheme.successColor : chartTheme.dangerColor,
            fontSize: 14,
            fontWeight: 'bold',
          },
        },
      ],
    }
  }, [data])

  return (
    <BaseChart
      option={option}
      loading={loading}
      error={error}
      onRetry={onRetry}
      height={height}
      onChartReady={handleChartReady}
      onEvents={onDataZoom ? { dataZoom: handleDataZoom } : undefined}
    />
  )
}
