import React, { useState } from 'react'
import { Card } from '../common'
import {
  ChartTypeSelector,
  type ChartType,
  ChartLayout,
  type LayoutMode,
} from './'

interface BacktestChartsContainerProps {
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  className?: string
}

export const BacktestChartsContainer: React.FC<BacktestChartsContainerProps> = ({
  loading = false,
  error = null,
  onRetry,
  className = '',
}) => {
  const [selectedChartType, setSelectedChartType] = useState<ChartType>('equity')
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('grid')

  const renderCharts = () => {
    if (loading) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <Card key={i} padding="md" className="h-80">
              <div className="animate-pulse">
                <div className="h-4 bg-white/10 rounded w-24 mb-4"></div>
                <div className="h-64 bg-white/5 rounded"></div>
              </div>
            </Card>
          ))}
        </div>
      )
    }

    if (error) {
      return (
        <Card padding="lg" className="text-center">
          <div className="flex flex-col items-center gap-4">
            <svg
              className="w-16 h-16 text-red-500"
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
            <div>
              <p className="text-gray-400 mb-2">图表加载失败</p>
              <p className="text-sm text-gray-500 mb-4">{error}</p>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-md text-sm font-medium transition-colors"
                >
                  重试
                </button>
              )}
            </div>
          </div>
        </Card>
      )
    }

    return (
      <Card padding="lg" className="text-center">
        <div className="flex flex-col items-center gap-4">
          <svg
            className="w-16 h-16 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <div>
            <p className="text-gray-400 mb-2">暂无图表数据</p>
            <p className="text-sm text-gray-500">运行回测后将显示图表</p>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h2 className="text-lg font-semibold text-white">回测图表</h2>
        <div className="flex flex-wrap items-center gap-3">
          <ChartTypeSelector
            selectedType={selectedChartType}
            onChange={setSelectedChartType}
          />
        </div>
      </div>
      <ChartLayout
        layoutMode={layoutMode}
        onLayoutChange={setLayoutMode}
      >
        {renderCharts()}
      </ChartLayout>
    </div>
  )
}
