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
  metadata?: {
    main_net_inflow?: number;
    super_net_inflow?: number;
    big_net_inflow?: number;
    medium_net_inflow?: number;
    small_net_inflow?: number;
    [key: string]: any;
  };
};

export type VisualizationResponse = {
  stock_code: string;
  stock_name?: string;
  kline_data: KLineDataPoint[];
  indicators: IndicatorData[];
  chip_distribution?: {
    latest?: ChipDistributionResponse;
    history?: Array<{ date: string; data: ChipDistributionResponse }>;
  };
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

export type ChipDistributionResponse = {
  stock_code: string;
  price_bins: number[];
  chip_volumes: number[];
  profit_volumes: number[];
  loss_volumes: number[];
  avg_cost?: number;
  max_chip_price?: number;
  current_price?: number;
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
    indicatorTypes?: string[],
    startDate?: string
  ): Promise<VisualizationResponse> {
    const params: Record<string, any> = {};
    if (startDate) {
      params.start_date = startDate;
    } else {
      params.days = days;
    }
    if (indicatorTypes && indicatorTypes.length > 0) {
      params.indicator_types = indicatorTypes.join(',');
    }

    const response = await apiClient.get(`/api/v1/visualization/${encodeURIComponent(stockCode)}`, {
      params,
    });

    return response.data as VisualizationResponse;
  },

  async getSearchHistory(limit: number = 100): Promise<VisualizationSearchHistoryResponse> {
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

  async updateSearchHistoryTimestamp(recordId: number): Promise<VisualizationSearchHistoryItem> {
    const response = await apiClient.patch(`/api/v1/visualization/history/${recordId}/timestamp`);
    return response.data as VisualizationSearchHistoryItem;
  },

  async getChipDistribution(
    stockCode: string,
    days: number = 365,
    startDate?: string,
    endDate?: string,
    endDateIdx?: number
  ): Promise<ChipDistributionResponse> {
    const params: Record<string, any> = { days };
    if (startDate) {
      params.start_date = startDate;
    }
    if (endDate) {
      params.end_date = endDate;
    }
    if (endDateIdx !== undefined) {
      params.end_date_idx = endDateIdx;
    }

    const response = await apiClient.get(`/api/v1/visualization/chip-distribution/${encodeURIComponent(stockCode)}`, {
      params,
    });

    return response.data as ChipDistributionResponse;
  },
};
