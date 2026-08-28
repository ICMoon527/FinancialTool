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
  /** 评估结果（仅终态返回） */
  result: EvaluateResultResponse | null;
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
