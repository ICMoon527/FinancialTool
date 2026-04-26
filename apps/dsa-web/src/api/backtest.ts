import apiClient, { strategyBacktestApiClient } from './index';
import { toCamelCase } from './utils';
import type {
  BacktestRunRequest,
  BacktestRunResponse,
  BacktestResultsResponse,
  BacktestResultItem,
  PerformanceMetrics,
  StrategyInfo,
  StrategyListResponse,
  StrategyBacktestRunRequest,
  StrategyBacktestRunResponse,
  StrategyBacktestRunAsyncRequest,
  StrategyBacktestRunAsyncResponse,
  StrategyBacktestTaskStatusResponse,
  StrategyBacktestStopByTaskIdRequest,
} from '../types/backtest';

// ============ API ============

export const backtestApi = {
    /**
     * 获取回测默认配置
     */
    getBacktestConfig: async (): Promise<{
        start_date?: string;
        end_date?: string;
        max_positions?: number;
    }> => {
        const response = await apiClient.get<{
            success: boolean;
            config: {
                start_date?: string;
                end_date?: string;
                max_positions?: number;
            };
        }>('/api/v1/backtest/config');
        if (response.data.success) {
            return response.data.config;
        }
        return {};
    },

    /**
     * 获取策略列表
     */
    getStrategies: async (): Promise<StrategyInfo[]> => {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/backtest/strategies',
    );
    const data = toCamelCase<StrategyListResponse>(response.data);
    return data.strategies || [];
  },

  /**
   * 运行策略回测（同步，废弃）
   */
  runStrategyBacktest: async (params: StrategyBacktestRunRequest): Promise<StrategyBacktestRunResponse> => {
    const requestData: Record<string, unknown> = {};
    if (params.strategyIds) {
      requestData.strategy_ids = params.strategyIds;
    } else if (params.strategyId) {
      requestData.strategy_id = params.strategyId;
    }
    if (params.startDate) requestData.start_date = params.startDate;
    if (params.endDate) requestData.end_date = params.endDate;
    if (params.stockPool) requestData.stock_pool = params.stockPool;
    if (params.maxPositions != null) requestData.max_positions = params.maxPositions;

    const response = await strategyBacktestApiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/run',
      requestData,
    );
    return toCamelCase<StrategyBacktestRunResponse>(response.data);
  },

  /**
   * 异步运行策略回测
   */
  runStrategyBacktestAsync: async (params: StrategyBacktestRunAsyncRequest): Promise<StrategyBacktestRunAsyncResponse> => {
    const requestData: Record<string, unknown> = {};
    if (params.strategyIds) {
      requestData.strategy_ids = params.strategyIds;
    } else if (params.strategyId) {
      requestData.strategy_id = params.strategyId;
    }
    requestData.start_date = params.startDate;
    requestData.end_date = params.endDate;
    requestData.max_positions = params.maxPositions;

    const response = await strategyBacktestApiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/strategy/run-async',
      requestData,
    );
    return toCamelCase<StrategyBacktestRunAsyncResponse>(response.data);
  },

  /**
   * 获取回测任务状态
   */
  getBacktestTaskStatus: async (taskId: string): Promise<StrategyBacktestTaskStatusResponse> => {
    if (!taskId || taskId === 'undefined' || taskId === 'null') {
      throw new Error('无效的任务ID');
    }
    const response = await strategyBacktestApiClient.get<Record<string, unknown>>(
      `/api/v1/backtest/strategy/task/${encodeURIComponent(taskId)}`,
    );
    return toCamelCase<StrategyBacktestTaskStatusResponse>(response.data);
  },

  /**
   * 通过任务ID停止策略回测
   */
  stopStrategyBacktestByTaskId: async (taskId: string): Promise<void> => {
    const requestData: StrategyBacktestStopByTaskIdRequest = {
      task_id: taskId,
    };
    await strategyBacktestApiClient.post(
      '/api/v1/backtest/strategy/stop-by-task-id',
      requestData,
    );
  },

  /**
   * Trigger backtest evaluation (已废弃)
   */
  run: async (params: BacktestRunRequest = {}): Promise<BacktestRunResponse> => {
    const requestData: Record<string, unknown> = {};
    if (params.code) requestData.code = params.code;
    if (params.force) requestData.force = params.force;
    if (params.evalWindowDays) requestData.eval_window_days = params.evalWindowDays;
    if (params.minAgeDays != null) requestData.min_age_days = params.minAgeDays;
    if (params.limit) requestData.limit = params.limit;

    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/backtest/run',
      requestData,
    );
    return toCamelCase<BacktestRunResponse>(response.data);
  },

  /**
   * Get paginated backtest results (已废弃)
   */
  getResults: async (params: {
    code?: string;
    evalWindowDays?: number;
    page?: number;
    limit?: number;
  } = {}): Promise<BacktestResultsResponse> => {
    const { code, evalWindowDays, page = 1, limit = 20 } = params;

    const queryParams: Record<string, string | number> = { page, limit };
    if (code) queryParams.code = code;
    if (evalWindowDays) queryParams.eval_window_days = evalWindowDays;

    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/backtest/results',
      { params: queryParams },
    );

    const data = toCamelCase<BacktestResultsResponse>(response.data);
    return {
      total: data.total,
      page: data.page,
      limit: data.limit,
      items: (data.items || []).map(item => toCamelCase<BacktestResultItem>(item)),
    };
  },

  /**
   * Get overall performance metrics (已废弃)
   */
  getOverallPerformance: async (evalWindowDays?: number): Promise<PerformanceMetrics | null> => {
    try {
      const params: Record<string, number> = {};
      if (evalWindowDays) params.eval_window_days = evalWindowDays;
      const response = await apiClient.get<Record<string, unknown>>(
        '/api/v1/backtest/performance',
        { params },
      );
      return toCamelCase<PerformanceMetrics>(response.data);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } };
        if (axiosErr.response?.status === 404) return null;
      }
      throw err;
    }
  },

  /**
   * Get per-stock performance metrics (已废弃)
   */
  getStockPerformance: async (code: string, evalWindowDays?: number): Promise<PerformanceMetrics | null> => {
    try {
      const params: Record<string, number> = {};
      if (evalWindowDays) params.eval_window_days = evalWindowDays;
      const response = await apiClient.get<Record<string, unknown>>(
        `/api/v1/backtest/performance/${encodeURIComponent(code)}`,
        { params },
      );
      return toCamelCase<PerformanceMetrics>(response.data);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { status?: number } };
        if (axiosErr.response?.status === 404) return null;
      }
      throw err;
    }
  },
};
