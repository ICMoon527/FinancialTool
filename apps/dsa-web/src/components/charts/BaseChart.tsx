import React, { useRef, useEffect, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import * as echarts from 'echarts'
import type { EChartsOption } from 'echarts'
import { Loading } from '../common/Loading'
import { Button } from '../common/Button'
import { registerEChartsTheme } from './theme'

interface BaseChartProps {
  option: EChartsOption
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  width?: number | string
  className?: string
  theme?: string
  notMerge?: boolean
  lazyUpdate?: boolean
  onChartReady?: (chart: echarts.ECharts) => void
  onEvents?: Record<string, (params: any) => void>
}

export const BaseChart: React.FC<BaseChartProps> = ({
  option,
  loading = false,
  error = null,
  onRetry,
  height = '100%',
  width = '100%',
  className = '',
  theme = 'dark-financial',
  notMerge = false,
  lazyUpdate = false,
  onChartReady,
  onEvents,
}) => {
  const chartRef = useRef<ReactECharts>(null)

  useEffect(() => {
    registerEChartsTheme()
  }, [])

  const handleChartReady = useCallback(
    (chart: echarts.ECharts) => {
      if (onChartReady) {
        onChartReady(chart)
      }
    },
    [onChartReady]
  )

  const renderContent = () => {
    if (loading) {
      return (
        <div className="flex justify-center items-center h-full">
          <Loading />
        </div>
      )
    }

    if (error) {
      return (
        <div className="flex flex-col justify-center items-center h-full gap-4">
          <div className="text-center">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-red-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <p className="text-gray-400 mb-2">图表加载失败</p>
            <p className="text-sm text-gray-500 mb-4">{error}</p>
            {onRetry && (
              <Button variant="outline" size="sm" onClick={onRetry}>
                重试
              </Button>
            )}
          </div>
        </div>
      )
    }

    return (
      <ReactECharts
        ref={chartRef}
        option={option}
        theme={theme}
        notMerge={notMerge}
        lazyUpdate={lazyUpdate}
        style={{ height, width }}
        onChartReady={handleChartReady}
        onEvents={onEvents}
      />
    )
  }

  return (
    <div
      className={`relative bg-[#08080c] rounded-lg border border-white/10 ${className}`}
      style={{ height, width }}
    >
      {renderContent()}
    </div>
  )
}
