/**
 * 可视化数据缓存模块
 *
 * 在内存中缓存最近查看的一只标的的可视化数据，
 * 确保用户切换页面后再次返回时无需重新搜索即可查看之前的标的信息。
 * 缓存生命周期：仅在浏览器标签页存活期间有效，刷新页面或关闭标签页后自动清除。
 */

import type { VisualizationResponse } from '../api/visualization';

/** 缓存数据结构 */
export interface VisualizationCacheData {
  /** 标的代码 */
  stockCode: string;
  /** 标的名称 */
  stockName: string;
  /** 可视化数据（包含K线、指标等） */
  visualizationData: VisualizationResponse;
  /** 用户选中的指标类型列表 */
  selectedIndicators: string[];
  /** 日期范围 */
  selectedDateRange: string;
}

/** 模块级缓存变量，仅保留最近一只标的 */
let _cache: VisualizationCacheData | null = null;

/**
 * 获取当前缓存的可视化数据
 * @returns 缓存数据，无缓存时返回 null
 */
export function getCachedVisualization(): VisualizationCacheData | null {
  return _cache;
}

/**
 * 设置可视化数据缓存（替换已有缓存）
 * @param data 要缓存的完整数据
 */
export function setCachedVisualization(data: VisualizationCacheData): void {
  _cache = data;
  console.log('[可视化缓存] 已更新缓存:', data.stockCode, data.stockName);
}

/**
 * 清除可视化数据缓存
 */
export function clearVisualizationCache(): void {
  if (_cache) {
    console.log('[可视化缓存] 已清除缓存:', _cache.stockCode);
  }
  _cache = null;
}

/**
 * 检查是否有缓存数据可用
 * @returns 是否有缓存
 */
export function hasVisualizationCache(): boolean {
  return _cache !== null;
}