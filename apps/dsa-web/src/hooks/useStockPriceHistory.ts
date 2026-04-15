import { useState, useEffect, useCallback } from 'react';

// 股价记录数据结构
export interface StockPriceRecord {
  stockCode: string;
  firstSearchedAt: string;
  firstPrice: number;
  firstPriceDate: string;
}

const STORAGE_KEY = 'stock_price_history';

/**
 * 股价记录管理 Hook
 * 用于存储和管理股票首次搜索时的股价数据
 */
export const useStockPriceHistory = () => {
  const [priceRecords, setPriceRecords] = useState<Record<string, StockPriceRecord>>({});

  // 从 localStorage 加载数据
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setPriceRecords(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load stock price history from localStorage:', error);
    }
  }, []);

  // 保存数据到 localStorage
  const saveToStorage = useCallback((records: Record<string, StockPriceRecord>) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
    } catch (error) {
      console.error('Failed to save stock price history to localStorage:', error);
    }
  }, []);

  // 记录或更新股价记录
  const recordPrice = useCallback((stockCode: string, price: number, priceDate: string) => {
    setPriceRecords(prev => {
      // 如果已存在记录，不更新，保持首次记录
      if (prev[stockCode]) {
        return prev;
      }

      const newRecord: StockPriceRecord = {
        stockCode,
        firstSearchedAt: new Date().toISOString(),
        firstPrice: price,
        firstPriceDate: priceDate,
      };

      const newRecords = {
        ...prev,
        [stockCode]: newRecord,
      };

      saveToStorage(newRecords);
      return newRecords;
    });
  }, [saveToStorage]);

  // 获取股价记录
  const getPriceRecord = useCallback((stockCode: string): StockPriceRecord | undefined => {
    return priceRecords[stockCode];
  }, [priceRecords]);

  // 删除股价记录
  const deletePriceRecord = useCallback((stockCode: string) => {
    setPriceRecords(prev => {
      const newRecords = { ...prev };
      delete newRecords[stockCode];
      saveToStorage(newRecords);
      return newRecords;
    });
  }, [saveToStorage]);

  // 计算涨跌幅
  const calculatePriceChange = useCallback((stockCode: string, currentPrice: number): {
    changePercent: number;
    hasRecord: boolean;
    firstPrice?: number;
    firstPriceDate?: string;
  } => {
    const record = priceRecords[stockCode];
    
    if (!record) {
      return {
        changePercent: 0,
        hasRecord: false,
      };
    }

    const changePercent = ((currentPrice - record.firstPrice) / record.firstPrice) * 100;

    return {
      changePercent,
      hasRecord: true,
      firstPrice: record.firstPrice,
      firstPriceDate: record.firstPriceDate,
    };
  }, [priceRecords]);

  return {
    priceRecords,
    recordPrice,
    getPriceRecord,
    deletePriceRecord,
    calculatePriceChange,
  };
};
