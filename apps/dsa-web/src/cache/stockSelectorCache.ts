/**
 * 选股页面数据缓存模块
 *
 * 在内存中缓存选股页面的关键状态数据（选中标的、候选列表、策略列表等），
 * 确保用户切换页面后再次返回时无需重新加载即可恢复之前的状态。
 * 缓存生命周期：仅在浏览器标签页存活期间有效，刷新页面或关闭标签页后自动清除。
 */

import type { StockCandidateInfo, StrategyInfo } from '../types/stockSelector';

/** 选股页面缓存数据结构 */
export interface StockSelectorCacheData {
  /** 当前选中的标的（用于K线图展示） */
  selectedStock: StockCandidateInfo | null;
  /** 候选标的列表 */
  candidates: StockCandidateInfo[];
  /** 已加载的策略列表 */
  strategies: StrategyInfo[];
  /** 用户选中的策略ID列表 */
  selectedStrategyIds: string[];
  /** 用户输入的股票代码过滤 */
  stockCodes: string;
  /** 策略类型过滤 */
  strategyTypeFilter: 'ALL' | 'NATURAL_LANGUAGE' | 'PYTHON';
  /** 是否更新数据 */
  updateData: boolean;
  /** 是否更新实时数据 */
  updateRealtime: boolean;
}

/** 模块级缓存变量 */
let _cache: StockSelectorCacheData | null = null;

/**
 * 获取当前缓存的选股页面数据
 * @returns 缓存数据，无缓存时返回 null
 */
export function getCachedStockSelector(): StockSelectorCacheData | null {
  return _cache;
}

/**
 * 设置选股页面数据缓存（替换已有缓存）
 * @param data 要缓存的完整数据
 */
export function setCachedStockSelector(data: StockSelectorCacheData): void {
  _cache = data;
  console.log('[选股缓存] 已更新缓存: 候选数', data.candidates.length, '选中标的', data.selectedStock?.stock_code);
}

/**
 * 清除选股页面数据缓存
 */
export function clearStockSelectorCache(): void {
  if (_cache) {
    console.log('[选股缓存] 已清除缓存');
  }
  _cache = null;
}

/**
 * 检查是否有缓存数据可用
 * @returns 是否有缓存
 */
export function hasStockSelectorCache(): boolean {
  return _cache !== null;
}