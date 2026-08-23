/**
 * Backtest API type definitions
 * Mirrors api/v1/schemas/backtest.py and api/v1/schemas/strategy_backtest.py
 */

// ============ 历史AI分析回测 (已废弃) ============

export interface BacktestRunRequest {
  code?: string;
  force?: boolean;
  evalWindowDays?: number;
  minAgeDays?: number;
  limit?: number;
}

export interface BacktestRunResponse {
  processed: number;
  saved: number;
  completed: number;
  insufficient: number;
  errors: number;
}

export interface BacktestResultItem {
  analysisHistoryId: number;
  code: string;
  analysisDate?: string;
  evalWindowDays: number;
  engineVersion: string;
  evalStatus: string;
  evaluatedAt?: string;
  operationAdvice?: string;
  positionRecommendation?: string;
  startPrice?: number;
  endClose?: number;
  maxHigh?: number;
  minLow?: number;
  stockReturnPct?: number;
  directionExpected?: string;
  directionCorrect?: boolean;
  outcome?: string;
  stopLoss?: number;
  takeProfit?: number;
  hitStopLoss?: boolean;
  hitTakeProfit?: boolean;
  firstHit?: string;
  firstHitDate?: string;
  firstHitTradingDays?: number;
  simulatedEntryPrice?: number;
  simulatedExitPrice?: number;
  simulatedExitReason?: string;
  simulatedReturnPct?: number;
}

export interface BacktestResultsResponse {
  total: number;
  page: number;
  limit: number;
  items: BacktestResultItem[];
}

export interface PerformanceMetrics {
  scope: string;
  code?: string;
  evalWindowDays: number;
  engineVersion: string;
  computedAt?: string;

  totalEvaluations: number;
  completedCount: number;
  insufficientCount: number;
  longCount: number;
  cashCount: number;
  winCount: number;
  lossCount: number;
  neutralCount: number;

  directionAccuracyPct?: number;
  winRatePct?: number;
  neutralRatePct?: number;
  avgStockReturnPct?: number;
  avgSimulatedReturnPct?: number;

  stopLossTriggerRate?: number;
  takeProfitTriggerRate?: number;
  ambiguousRate?: number;
  avgDaysToFirstHit?: number;

  adviceBreakdown: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
}

// ============ 策略回测 ============

export interface StrategyInfo {
  id: string;
  name: string;
  description: string;
  type: string;
}

export interface StrategyListResponse {
  strategies: StrategyInfo[];
}

export interface StrategyBacktestRunRequest {
  strategyId?: string;  // 已废弃，向后兼容
  strategyIds?: string[];  // 多策略支持
  startDate?: string;
  endDate?: string;
  stockPool?: string[];
  maxPositions?: number;
}

export interface StrategyBacktestRunResponse {
  success: boolean;
  message: string;
  results?: Record<string, any>;
  metrics?: Record<string, any>;
  reports?: Record<string, any>;
}

export interface StrategyBacktestRunAsyncRequest {
  strategyId?: string;  // 已废弃，向后兼容
  strategyIds?: string[];  // 多策略支持
  startDate: string;
  endDate: string;
  maxPositions: number;
  exitStrategy?: ExitStrategyConfig;  // 退出策略配置
}

export interface ExitStrategyConfig {
  strategy: string;  // 策略类型: "simple" | "tiered"
  name?: string;  // 止盈止损显示名（前端 preset 的中文名），用于结果子目录命名
  params: Record<string, number>;  // 策略参数
}

export interface ExitStrategyInfo {
  key: string;
  name: string;
  description: string;
}

export interface ExitStrategyPreset {
  name: string;
  strategy: string;
  params: Record<string, number>;
}

export interface ExitStrategiesResponse {
  success: boolean;
  strategies: ExitStrategyInfo[];
  presets: Record<string, ExitStrategyPreset>;
  active: string;
}

export interface StrategyBacktestRunAsyncResponse {
  success: boolean;
  message: string;
  task_id?: string;
  taskId?: string;
}

export interface StrategyBacktestTaskStatusResponse {
  success: boolean;
  message: string;
  task?: {
    task_id?: string;
    taskId?: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
    created_at?: string;
    createdAt?: string;
    started_at?: string;
    startedAt?: string;
    completed_at?: string;
    completedAt?: string;
    result?: Record<string, any>;
    error?: string;
  };
}

export interface StrategyBacktestStopByTaskIdRequest {
  task_id: string;
}
