import React, { useRef, useCallback } from 'react'
import * as echarts from 'echarts'
import { ChartExportButton } from './ChartExportButton'
import { useChartInstance } from './hooks'

interface ResponsiveChartContainerProps {
  title?: string
  children: React.ReactNode
  loading?: boolean
  error?: string | null
  className?: string
  filename?: string
  onChartReady?: (chart: echarts.ECharts) => void
}

export const ResponsiveChartContainer: React.FC<ResponsiveChartContainerProps> = ({
  title,
  children,
  loading = false,
  error = null,
  className = '',
  filename,
  onChartReady,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const { setChartInstance, getChartInstance } = useChartInstance()

  const handleChartReady = useCallback(
    (chart: echarts.ECharts) => {
      setChartInstance(chart)
      if (onChartReady) {
        onChartReady(chart)
      }
    },
    [setChartInstance, onChartReady]
  )

  const childrenWithProps = React.Children.map(children, (child) => {
    if (React.isValidElement(child)) {
      return React.cloneElement(child as React.ReactElement<any>, {
        onChartReady: handleChartReady,
      })
    }
    return child
  })

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`}
    >
      {(title || !loading) && (
        <div className="flex items-center justify-between mb-3">
          {title && (
            <h3 className="text-sm font-semibold text-white">{title}</h3>
          )}
          {!loading && !error && (
            <ChartExportButton
              chartInstance={getChartInstance()}
              filename={filename || title?.toLowerCase().replace(/\s+/g, '-') || 'chart'}
            />
          )}
        </div>
      )}

      {childrenWithProps}
    </div>
  )
}
