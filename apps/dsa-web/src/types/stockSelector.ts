/**
 * Stock Selector Type Definitions
 */

export interface StrategyInfo {
  id: string;
  name: string;
  display_name: string;
  description: string;
  strategy_type: 'NATURAL_LANGUAGE' | 'PYTHON';
  category: string;
  source: string;
  version: string;
  created_at: string;
  is_active: boolean;
}

export interface StrategyMatchInfo {
  strategy_id: string;
  strategy_name: string;
  matched: boolean;
  score: number;
  reason?: string;
  match_details: Record<string, any>;
}

export interface StockCandidateInfo {
  stock_code: string;
  stock_name: string;
  current_price: number;
  overall_score: number;
  strategy_matches: StrategyMatchInfo[];
  created_at: string;
  extra_data: Record<string, any>;
}

export interface StockSelectorRequest {
  stock_codes?: string[];
  strategy_ids?: string[];
  top_n?: number;
  update_data?: boolean;
}

export interface StockSelectorConfig {
  success: boolean;
  default_top_n: number;
  error?: string;
}

export interface StockSelectorConfigResponse {
  success: boolean;
  default_top_n: number;
  error?: string;
}

export interface StockSelectorResponse {
  success: boolean;
  candidates: StockCandidateInfo[];
  total_screened: number;
  execution_time_ms: number;
  error?: string;
}

export interface StrategiesResponse {
  success: boolean;
  strategies: StrategyInfo[];
  active_strategy_ids: string[];
}

export interface ActivateStrategyRequest {
  strategy_id: string;
}

export interface DeactivateStrategyRequest {
  strategy_id: string;
}
