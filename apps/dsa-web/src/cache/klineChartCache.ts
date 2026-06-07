/**
 * K线图缓存模块（独立于 VisualizationPage 的缓存）
 *
 * 在内存中缓存 KlineChart 组件最近查看的一只标的的可视化数据，
 * 与 VisualizationPage 的缓存完全隔离，避免跨页面"串台"。
 * 缓存生命周期：仅在浏览器标签页存活期间有效，刷新页面或关闭标签页后自动清除。
 */

import type { VisualizationResponse } from '../api/visualization';

/** 缓存数据结构 */
export interface KlineChartCacheData {
  /** 标的代码 */
  stockCode: string;
  /** 标的名称 */
  stockName: string;
  /** 可视化数据（包含K线、指标等） */
  visualizationData: VisualizationResponse;
}

/** 模块级缓存变量，仅保留最近一只标的 */
let _cache: KlineChartCacheData | null = null;

/**
 * 获取缓存
 * @returns 缓存数据，无缓存时返回 null
 */
export function getCachedKlineChart(): KlineChartCacheData | null {
  return _cache;
}

/**
 * 设置缓存（替换已有缓存）
 * @param data 要缓存的完整数据
 */
export function setCachedKlineChart(data: KlineChartCacheData): void {
  _cache = data;
  console.log('[KlineChart缓存] 已更新缓存:', data.stockCode, data.stockName);
}

/**
 * 清除缓存
 */
export function clearKlineChartCache(): void {
  if (_cache) {
    console.log('[KlineChart缓存] 已清除缓存:', _cache.stockCode);
  }
  _cache = null;
}

/**
 * 检查是否有缓存数据可用
 * @returns 是否有缓存
 */
export function hasKlineChartCache(): boolean {
  return _cache !== null;
}