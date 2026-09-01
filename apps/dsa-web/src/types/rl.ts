/**
 * RL 训练模块类型定义
 */

// ============ 训练任务 ============

export interface TrainRequest {
  algorithm?: 'dqn' | 'ppo';
  episodes?: number;
  batchSize?: number;
  learningRate?: number;
  resumeFrom?: string;
  useSignalScores?: boolean;
}

export interface TrainResponse {
  taskId: string;
  status: string;
  message: string;
}

export type RLTaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped' | 'stopping';

export interface TaskStatusResponse {
  taskId: string;
  status: RLTaskStatus;
  progress: number;
  message: string;
  createdAt: string;
}

/** 训练指标历史（与后端 TrainingMetrics.to_dict 对应） */
export interface RLMetricsHistory {
  episodeRewards: number[];
  episodeLengths: number[];
  losses: number[];
  tdErrors: number[];
  epsilons: number[];
  entropies: number[];
  valSharpeRatios: number[];
  valReturns: number[];
  valWinRates: number[];
}

export interface TrainingProgressResponse {
  currentEpisode: number;
  latestReward: number;
  metrics: Partial<RLMetricsHistory>;
  status: RLTaskStatus;
  progress: number;
  totalEpisodes: number;
  message: string;
  paused: boolean;
}

// ============ 模型 ============

export interface ModelInfo {
  modelId: string;
  algorithm: string;
  createdAt: string;
  metrics?: Record<string, unknown>;
}

export interface ModelListResponse {
  models: ModelInfo[];
}

// ============ 评估 ============

export interface EvaluateRequest {
  modelId: string;
  stockCodes?: string[];
  /** 抽样评估的最大交易日数；0 表示全量评估，不传默认抽样 100 */
  maxDays?: number;
}

export interface EvaluateTaskResponse {
  taskId: string;
  status: string;
  message: string;
}

export interface EvaluateProgressResponse {
  taskId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  /** 进度百分比 0-100 */
  progress: number;
  /** 已完成样本数 */
  done: number;
  /** 总样本数 */
  total: number;
  /** 当前阶段说明（当前回放股票/日期、做T收益等） */
  message: string;
  /** 是否已暂停 */
  paused: boolean;
  /** 对比评估：当前正在评估的模型序号（0-based）；单模型评估为 null */
  currentModelIdx?: number | null;
  /** 对比评估：当前模型已完成的交易日数 */
  modelDone?: number | null;
  /** 对比评估：每个模型的交易日总数（= 抽样天数） */
  modelTotal?: number | null;
  /** 评估结果（仅终态返回） */
  result: EvaluateResultResponse | EvaluateCompareResultResponse | null;
}

export interface DailySummary {
  date: string;
  stockCode: string;
  dailyReturn: number;
  tradeCount: number;
  avgReward: number;
  trades: Record<string, unknown>[];
}

export interface SummaryMetrics {
  sharpeRatio: number;
  totalReturn: number;
  winRate: number;
  maxDrawdown: number;
  totalTrades: number;
}

export interface EvaluateResultResponse {
  cumulativeReturns: number[];
  benchmarkReturns: number[];
  dailySummaries: DailySummary[];
  summaryMetrics: SummaryMetrics;
}

// ============ 多模型对比评估 ============

export interface EvaluateCompareRequest {
  modelIds: string[];
  /** 抽样评估的最大交易日数；0 表示全量评估，不传默认抽样 100 */
  maxDays?: number;
}

/** 对比评估中单个模型的结果（所有模型共用同一批样本/基准） */
export interface CompareModelResult {
  modelId: string;
  cumulativeReturns: number[];
  summaryMetrics: SummaryMetrics;
}

/** 多模型对比评估结果 */
export interface EvaluateCompareResultResponse {
  /** 实际评估的样本列表 */
  samples: { stockCode: string; date: string }[];
  /** 同一批样本的买入持有基准（所有模型共用） */
  benchmarkReturns: number[];
  models: CompareModelResult[];
}

// ============ 单日回放 ============

export interface DailyDecision {
  step: number;
  action: 'HOLD' | 'BUY' | 'SELL';
  reward: number;
  position: number;
}

export interface ReplayKline {
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  timestamp?: string;
  time?: string;
}

export interface DailyReplayResponse {
  stockCode: string;
  date: string;
  klines: ReplayKline[];
  decisions: DailyDecision[];
  rewardHeatmap: number[];
  trades: Record<string, unknown>[];
}
