import apiClient from './index';
import type {
  StockSelectorRequest,
  StockSelectorResponse,
  StrategiesResponse,
} from '../types/stockSelector';

export const stockSelectorApi = {
  async getStrategies(): Promise<StrategiesResponse> {
    const response = await apiClient.get<StrategiesResponse>('/api/v1/stock-selector/strategies');
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
      timeout: 120000,
    });
    return response.data;
  },
};
