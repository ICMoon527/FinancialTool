import React, { useState, useMemo } from 'react'
import { Card } from '../common'
import {
  ChartTypeSelector,
  type ChartType,
  ChartLayout,
  type LayoutMode,
  EquityCurveChart,
  DrawdownChart,
  ReturnDistributionChart,
  PerformanceRadarChart,
  ReturnHeatmapChart,
} from './'
import {
  transformEquityData,
  transformDrawdownData,
} from './utils'

interface BacktestChartsContainerProps {
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  className?: string
  results?: any
  metrics?: any
}

export const BacktestChartsContainer: React.FC<BacktestChartsContainerProps> = ({
  loading = false,
  error = null,
  onRetry,
  className = '',
  results,
  metrics,
}) => {
  const [selectedChartType, setSelectedChartType] = useState<ChartType>('equity')
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('grid')

  // 调试信息
  console.log('BacktestChartsContainer - results:', results)
  console.log('BacktestChartsContainer - metrics:', metrics)

  // 转换数据转换
  const hasData = results && results.equity_history
  console.log('BacktestChartsContainer - hasData:', hasData)

  // 准备图表数据
  const chartData = useMemo(() => {
    if (!hasData || !results || !metrics) return null

    // 1. 净值曲线数据
    const dates = results.equity_history.map((item: any) => item.date)
    const equityCurve = results.equity_history.map((item: any) => item.equity)

    // 2. 计算最大回撤
    let maxEquity = equityCurve[0]
    const drawdownCurve = equityCurve.map((equity: number) => {
      maxEquity = Math.max(maxEquity, equity)
      return (maxEquity - equity) / maxEquity
    })

    // 3. 计算日收益率
    const returns: number[] = []
    for (let i = 1; i < equityCurve.length; i++) {
      returns.push((equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1])
    }

    // 4. 月度收益率热力图
    const monthlyReturns: Array<{ year: string; month: string; return: number }> = []
    const monthlyData: Record<string, Record<string, number[]>> = {}
    
    results.equity_history.forEach((item: any) => {
      const date = new Date(item.date)
      const year = date.getFullYear().toString()
      const month = (date.getMonth() + 1).toString()
      
      if (!monthlyData[year]) {
        monthlyData[year] = {}
      }
      if (!monthlyData[year][month]) {
        monthlyData[year][month] = []
      }
      monthlyData[year][month].push(item.equity)
    })

    Object.entries(monthlyData).forEach(([year, months]) => {
      Object.entries(months).forEach(([month, values]) => {
        if (values.length >= 2) {
          const monthlyReturn = (values[values.length - 1] - values[0]) / values[0]
          monthlyReturns.push({ year, month, return: monthlyReturn })
        }
      })
    })

    return {
      dates,
      equityCurve,
      drawdownCurve,
      returns,
      monthlyReturns,
      equityData: transformEquityData({
        dates,
        equityCurve,
        benchmarkCurve: undefined,
      }),
      drawdownData: transformDrawdownData({
        dates,
        equityCurve,
        drawdownCurve,
      }),
    }
  }, [results, metrics, hasData])

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

    if (!hasData || !chartData || !results || !metrics) {
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

    // 准备热力图数据
    const returnHeatmapData = {
      monthly: chartData.monthlyReturns.map(mr => ({
        year: mr.year,
        month: mr.month,
        return: mr.return
      })),
      annual: [] // 暂时留空
    }

    // 根据选择的图表类型渲染
    const charts = (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <EquityCurveChart data={chartData.equityData} />
        <DrawdownChart data={chartData.drawdownData} />
        <ReturnDistributionChart 
          data={{
            dailyReturns: chartData.returns,
            weeklyReturns: [],
            monthlyReturns: []
          }}
        />
        <PerformanceRadarChart 
          reports={[{
            name: '策略绩效',
            report: {
              totalReturn: (chartData.equityCurve[chartData.equityCurve.length - 1] - chartData.equityCurve[0]) / chartData.equityCurve[0],
              annualizedReturn: metrics.annualized_return || 0,
              maxDrawdown: Math.max(...chartData.drawdownCurve),
              sharpeRatio: metrics.sharpe_ratio || 0,
              winRate: metrics.win_rate || 0.5,
              profitFactor: metrics.profit_factor || 1,
              totalTrades: results.total_trades || 0,
              winningTrades: Math.floor((results.total_trades || 0) * (metrics.win_rate || 0.5)),
              losingTrades: Math.ceil((results.total_trades || 0) * (1 - (metrics.win_rate || 0.5))),
              avgWin: 0,
              avgLoss: 0,
              riskRewardRatio: 0,
              volatility: 0,
              sortinoRatio: 0,
              calmarRatio: metrics.calmar_ratio || 0
            }
          }]}
        />
        <ReturnHeatmapChart data={returnHeatmapData} />
      </div>
    )

    return charts
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
