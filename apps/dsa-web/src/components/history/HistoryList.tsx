import type React from 'react';
import { useRef, useCallback, useEffect } from 'react';
import type { HistoryItem } from '../../types/analysis';
import { getSentimentColor } from '../../types/analysis';
import { formatDateTime } from '../../utils/format';

interface HistoryListProps {
  items: HistoryItem[];
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  selectedId?: number;
  onItemClick: (recordId: number) => void;
  onLoadMore: () => void;
  onDelete?: (id: number) => void;
  className?: string;
}

/**
 * 历史记录列表组件
 * 显示最近的股票分析历史，支持点击查看详情、滚动加载更多和删除
 */
export const HistoryList: React.FC<HistoryListProps> = ({
  items,
  isLoading,
  isLoadingMore,
  hasMore,
  selectedId,
  onItemClick,
  onLoadMore,
  onDelete,
  className = '',
}) => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const loadMoreTriggerRef = useRef<HTMLDivElement>(null);

  const handleObserver = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const target = entries[0];
      if (target.isIntersecting && hasMore && !isLoading && !isLoadingMore) {
        const container = scrollContainerRef.current;
        if (container && container.scrollHeight > container.clientHeight) {
          onLoadMore();
        }
      }
    },
    [hasMore, isLoading, isLoadingMore, onLoadMore]
  );

  useEffect(() => {
    const trigger = loadMoreTriggerRef.current;
    const container = scrollContainerRef.current;
    if (!trigger || !container) return;

    const observer = new IntersectionObserver(handleObserver, {
      root: container,
      rootMargin: '20px',
      threshold: 0.1,
    });

    observer.observe(trigger);

    return () => {
      observer.disconnect();
    };
  }, [handleObserver]);

  return (
    <aside className={`glass-card overflow-hidden flex flex-col ${className}`}>
      <div ref={scrollContainerRef} className="p-3 flex-1 overflow-y-auto">
        <h2 className="text-xs font-medium text-purple uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          历史记录
        </h2>

        {isLoading ? (
          <div className="flex justify-center py-6">
            <div className="w-5 h-5 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-6 text-muted text-xs">
            暂无历史记录
          </div>
        ) : (
          <div className="space-y-1.5">
            {items.map((item) => (
              <div key={item.id} className="group flex items-center">
                <button
                  type="button"
                  onClick={() => onItemClick(item.id)}
                  className={`history-item w-full text-left flex-1 min-w-0 ${selectedId === item.id ? 'active' : ''
                    }`}
                >
                  <div className="flex items-center gap-2 w-full">
                    {item.sentimentScore !== undefined && (
                      <span
                        className="w-0.5 h-8 rounded-full flex-shrink-0"
                        style={{
                          backgroundColor: getSentimentColor(item.sentimentScore),
                          boxShadow: `0 0 6px ${getSentimentColor(item.sentimentScore)}40`
                        }}
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="font-medium text-white truncate text-xs">
                          {item.stockName || item.stockCode}
                        </span>
                        {item.sentimentScore !== undefined && (
                          <span
                            className="text-xs font-mono font-semibold px-1 py-0.5 rounded"
                            style={{
                              color: getSentimentColor(item.sentimentScore),
                              backgroundColor: `${getSentimentColor(item.sentimentScore)}15`
                            }}
                          >
                            {item.sentimentScore}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-xs text-muted font-mono">
                          {item.stockCode}
                        </span>
                        <span className="text-xs text-muted/50">·</span>
                        <span className="text-xs text-muted">
                          {formatDateTime(item.createdAt)}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
                {onDelete && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(item.id!);
                    }}
                    className="p-0.5 ml-0.5 text-muted/40 hover:text-danger transition-colors flex-shrink-0 opacity-0 group-hover:opacity-100"
                    title="删除"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                )}
              </div>
            ))}

            <div ref={loadMoreTriggerRef} className="h-4" />

            {isLoadingMore && (
              <div className="flex justify-center py-3">
                <div className="w-4 h-4 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
              </div>
            )}

            {!hasMore && items.length > 0 && (
              <div className="text-center py-2 text-muted/50 text-xs">
                已加载全部
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};