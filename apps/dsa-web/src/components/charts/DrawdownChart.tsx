import React, { useMemo, useRef, useCallback } from 'react'
import type { EChartsOption } from 'echarts'
import * as echarts from 'echarts'
import { BaseChart } from './BaseChart'
import { chartTheme } from './theme'
import { useChartInstance } from './hooks'
import { formatPercent } from './utils'
import type { DrawdownChartData } from './utils'

interface DrawdownChartProps {
  data: DrawdownChartData
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  onChartReady?: (chart: echarts.ECharts) => void
  syncTimeRange?: { start: number; end: number }
  onDataZoom?: (params: any) => void
}

export const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  loading = false,
  error = null,
  onRetry,
  height = 250,
  onChartReady,
  syncTimeRange,
  onDataZoom,
}) => {
  const { setChartInstance } = useChartInstance()
  const dataZoomStartEndRef = useRef<{ start: number; end: number }>(syncTimeRange || { start: 0, end: 100 })

  React.useEffect(() => {
    if (syncTimeRange) {
      dataZoomStartEndRef.current = syncTimeRange
    }
  }, [syncTimeRange])

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

  const { maxDrawdown, maxDrawdownDuration } = useMemo(() => {
    let maxDD = 0
    let maxDDIndex = -1
    let peakIndex = -1

    data.drawdown.forEach((item, index) => {
      const dd = Math.abs(item[1])
      if (dd > maxDD) {
        maxDD = dd
        maxDDIndex = index
      }
    })

    for (let i = 0; i <= maxDDIndex; i++) {
      const dd = Math.abs(data.drawdown[i][1])
      if (dd === 0) {
        peakIndex = i
      }
    }

    const duration = maxDDIndex > peakIndex ? maxDDIndex - peakIndex : 0

    return {
      maxDrawdown: maxDD,
      maxDrawdownDuration: duration,
    }
  }, [data])

  const option = useMemo<EChartsOption>(() => {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
        },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const date = params[0].axisValue
          const value = params[0].value as [string, number]
          const dd = value[1]
          return `<div class="font-medium mb-1">${date}</div>
            <div class="flex items-center gap-2">
              <span class="inline-block w-2 h-2 rounded-full" style="background-color: ${chartTheme.dangerColor}"></span>
              <span class="text-gray-400">回撤:</span>
              <span class="font-medium text-red-400">${formatPercent(dd)}</span>
            </div>`
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
          formatter: (value: number) => formatPercent(value),
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
      series: [
        {
          name: '回撤',
          type: 'line',
          data: data.drawdown,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            width: 2,
            color: chartTheme.dangerColor,
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: `${chartTheme.dangerColor}33` },
              { offset: 1, color: `${chartTheme.dangerColor}00` },
            ]),
          },
          markPoint: {
            data: [
              {
                type: 'min',
                name: '最大回撤',
                label: {
                  formatter: (params: any) => formatPercent(params.value as number),
                },
              },
            ],
          },
        },
      ],
      graphic: [
        {
          type: 'text',
          left: 'center',
          top: 10,
          style: {
            text: `最大回撤: ${formatPercent(maxDrawdown)} (持续 ${maxDrawdownDuration} 天)`,
            fill: chartTheme.dangerColor,
            fontSize: 14,
            fontWeight: 'bold',
          },
        },
      ],
    }
  }, [data, maxDrawdown, maxDrawdownDuration])

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
