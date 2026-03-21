import apiClient from './index';

export type KLineDataPoint = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  amount?: number;
  pct_chg?: number;
};

export type IndicatorDataPoint = {
  [key: string]: any;
  date?: string;
};

export type IndicatorData = {
  indicator_type: string;
  data: IndicatorDataPoint[];
};

export type VisualizationResponse = {
  stock_code: string;
  stock_name?: string;
  kline_data: KLineDataPoint[];
  indicators: IndicatorData[];
};

export type VisualizationSearchHistoryItem = {
  id: number;
  stock_code: string;
  stock_name?: string;
  searched_at: string;
  selected_indicators: string[];
  days?: number;
};

export type VisualizationSearchHistoryResponse = {
  items: VisualizationSearchHistoryItem[];
};

export type VisualizationSearchRequest = {
  stock_code: string;
  stock_name?: string;
  days?: number;
  indicator_types?: string[];
};

export const visualizationApi = {
  async getVisualizationData(
    stockCode: string,
    days: number = 3650,
    indicatorTypes?: string[]
  ): Promise<VisualizationResponse> {
    const params: Record<string, any> = { days };
    if (indicatorTypes && indicatorTypes.length > 0) {
      params.indicator_types = indicatorTypes.join(',');
    }

    const response = await apiClient.get(`/api/v1/visualization/${encodeURIComponent(stockCode)}`, {
      params,
    });

    return response.data as VisualizationResponse;
  },

  async getSearchHistory(limit: number = 20): Promise<VisualizationSearchHistoryResponse> {
    const response = await apiClient.get('/api/v1/visualization/history', {
      params: { limit },
    });

    return response.data as VisualizationSearchHistoryResponse;
  },

  async saveSearchHistory(request: VisualizationSearchRequest): Promise<VisualizationSearchHistoryItem> {
    const response = await apiClient.post('/api/v1/visualization/history', request);

    return response.data as VisualizationSearchHistoryItem;
  },

  async deleteSearchHistory(recordId: number): Promise<void> {
    await apiClient.delete(`/api/v1/visualization/history/${recordId}`);
  },
};
