import type { BacktestChartData, BacktestReport } from '../../types/backtest-charts'

export interface EquityChartData {
  dates: string[]
  equity: [string, number][]
  benchmark?: [string, number][]
}

export interface DrawdownChartData {
  dates: string[]
  drawdown: [string, number][]
}

export interface ReturnDistributionData {
  bins: string[]
  values: number[]
}

export interface MetricsRadarChartData {
  indicators: Array<{ name: string; max: number }>
  values: number[]
}

export interface ReturnHeatmapData {
  months: string[]
  years: string[]
  data: Array<{ year: string; month: string; value: number }>
}

export const transformEquityData = (data: BacktestChartData): EquityChartData => {
  const equity = data.dates.map((date, index) => [date, data.equityCurve[index]] as [string, number])
  
  let benchmark: [string, number][] | undefined
  if (data.benchmarkCurve) {
    benchmark = data.dates.map((date, index) => [date, data.benchmarkCurve![index]] as [string, number])
  }

  return {
    dates: data.dates,
    equity,
    benchmark,
  }
}

export const transformDrawdownData = (data: BacktestChartData): DrawdownChartData => {
  const drawdown = data.dates.map((date, index) => [
    date,
    data.drawdownCurve ? data.drawdownCurve[index] : 0,
  ] as [string, number])

  return {
    dates: data.dates,
    drawdown,
  }
}

export const transformReturnDistributionData = (
  returns: number[],
  binCount = 20
): ReturnDistributionData => {
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
    bins.push(`${(binStart * 100).toFixed(1)}% - ${(binEnd * 100).toFixed(1)}%`)
  }

  returns.forEach((ret) => {
    let binIndex = Math.floor((ret - minReturn) / binWidth)
    binIndex = Math.max(0, Math.min(binIndex, binCount - 1))
    values[binIndex]++
  })

  return { bins, values }
}

export const transformMetricsRadarData = (
  report: BacktestReport
): MetricsRadarChartData => {
  const indicators = [
    { name: '总收益率', max: 100 },
    { name: '年化收益率', max: 50 },
    { name: '夏普比率', max: 5 },
    { name: '胜率', max: 100 },
    { name: '盈亏比', max: 5 },
    { name: '卡玛比率', max: 10 },
  ]

  const values = [
    Math.min(report.totalReturn * 100, 100),
    Math.min(report.annualizedReturn * 100, 50),
    Math.min(report.sharpeRatio, 5),
    Math.min(report.winRate * 100, 100),
    Math.min(report.profitFactor, 5),
    Math.min(report.calmarRatio, 10),
  ]

  return { indicators, values }
}

export const transformReturnHeatmapData = (
  monthlyReturns: Array<{ year: string; month: string; return: number }>
): ReturnHeatmapData => {
  const years = Array.from(new Set(monthlyReturns.map((r) => r.year))).sort()
  const months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

  const data = monthlyReturns.map((r) => ({
    year: r.year,
    month: r.month,
    value: r.return,
  }))

  return { months, years, data }
}

export const getHeatmapColor = (value: number): string => {
  if (value > 0) {
    const intensity = Math.min(value / 0.1, 1)
    return `rgba(0, 255, 136, ${0.3 + intensity * 0.7})`
  } else if (value < 0) {
    const intensity = Math.min(Math.abs(value) / 0.1, 1)
    return `rgba(255, 68, 102, ${0.3 + intensity * 0.7})`
  }
  return 'rgba(255, 255, 255, 0.1)'
}

export const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`
}

export const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: 'CNY',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}
