export interface BacktestChartData {
  dates: string[]
  equityCurve: number[]
  benchmarkCurve?: number[]
  drawdownCurve?: number[]
  positionCurve?: number[]
}

export interface TradePoint {
  date: string
  type: 'buy' | 'sell'
  price: number
  quantity: number
  profit?: number
}

export interface MetricsRadarData {
  name: string
  value: number
  max: number
}

export interface MetricsHeatmapData {
  name: string
  value: number
  color: string
}

export interface BacktestReport {
  totalReturn: number
  annualizedReturn: number
  maxDrawdown: number
  sharpeRatio: number
  winRate: number
  profitFactor: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
  avgWin: number
  avgLoss: number
  riskRewardRatio: number
  volatility: number
  sortinoRatio: number
  calmarRatio: number
}

export interface ChartTheme {
  backgroundColor: string
  textColor: string
  textColorSecondary: string
  textColorMuted: string
  borderColor: string
  borderColorDim: string
  accentCyan: string
  accentCyanDim: string
  accentPurple: string
  accentPurpleDim: string
  successColor: string
  warningColor: string
  dangerColor: string
  gridColor: string
  tooltipBackgroundColor: string
}
