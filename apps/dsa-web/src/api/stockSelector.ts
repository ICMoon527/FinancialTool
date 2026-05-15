import apiClient from './index';
import type {
  StockSelectorRequest,
  StockSelectorResponse,
  StrategiesResponse,
  StockSelectorConfigResponse,
  ScreenProgressStatus,
} from '../types/stockSelector';

export const stockSelectorApi = {
  async getStrategies(): Promise<StrategiesResponse> {
    const response = await apiClient.get<StrategiesResponse>('/api/v1/stock-selector/strategies');
    return response.data;
  },

  async getConfig(): Promise<StockSelectorConfigResponse> {
    const response = await apiClient.get<StockSelectorConfigResponse>('/api/v1/stock-selector/config');
    return response.data;
  },

  async activateStrategy(strategyId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post<{ success: boolean }>('/api/v1/stock-selector/strategies/activate', {
      strategy_id: strategyId,
    });
    return response.data;
  },

  async deactivateStrategy(strategyId: string): Promise<{ success: boolean }> {
    const response = await apiClient.post<{ success: boolean }>('/api/v1/stock-selector/strategies/deactivate', {
      strategy_id: strategyId,
    });
    return response.data;
  },

  async screenStocks(request: StockSelectorRequest): Promise<StockSelectorResponse> {
    const response = await apiClient.post<StockSelectorResponse>('/api/v1/stock-selector/screen', request, {
      timeout: 1200000,
    });
    return response.data;
  },

  async screenStocksAsync(request: StockSelectorRequest): Promise<ScreenProgressStatus> {
    const response = await apiClient.post<ScreenProgressStatus>(
      '/api/v1/stock-selector/screen-async',
      request,
      { timeout: 1200000 },
    );
    return response.data;
  },

  async getScreenAsyncStatus(taskId?: string): Promise<ScreenProgressStatus> {
    const params = taskId ? `?task_id=${encodeURIComponent(taskId)}` : '';
    const response = await apiClient.get<ScreenProgressStatus>(
      `/api/v1/stock-selector/screen-async/status${params}`,
    );
    return response.data;
  },

  async cancelScreenAsync(taskId: string): Promise<ScreenProgressStatus> {
    const response = await apiClient.post<ScreenProgressStatus>(
      `/api/v1/stock-selector/screen-async/cancel?task_id=${encodeURIComponent(taskId)}`,
    );
    return response.data;
  },
};
