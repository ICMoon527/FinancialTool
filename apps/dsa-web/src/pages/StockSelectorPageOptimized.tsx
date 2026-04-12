import type React from 'react';
import { useState, useEffect, useCallback, useRef, useMemo, memo, Suspense } from 'react';
import { stockSelectorApi } from '../api/stockSelector';
import { Card, Badge } from '../components/common';
import type {
  StrategyInfo,
  StockCandidateInfo,
  StrategyMatchInfo,
} from '../types/stockSelector';

// 骨架屏组件
const StockCandidateCardSkeleton = memo(() => (
  <Card variant="gradient" padding="md" className="opacity-70">
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-white/10 animate-pulse" />
        <div className="space-y-2">
          <div className="h-5 w-24 bg-white/10 rounded animate-pulse" />
          <div className="h-3 w-32 bg-white/10 rounded animate-pulse" />
        </div>
      </div>
      <div className="h-7 w-16 bg-white/10 rounded animate-pulse" />
    </div>
    
    <div className="grid grid-cols-4 gap-2 mb-3">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="h-4 w-full bg-white/10 rounded animate-pulse mb-1" />
          <div className="h-2 w-8 bg-white/10 rounded animate-pulse mx-auto" />
        </div>
      ))}
    </div>
  </Card>
));

StockCandidateCardSkeleton.displayName = 'StockCandidateCardSkeleton';

// 骨架屏列表组件
const StockListSkeleton = memo(({ count = 10 }: { count?: number }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
    {Array.from({ length: count }).map((_, i) => (
      <StockCandidateCardSkeleton key={i} />
    ))}
  </div>
));

StockListSkeleton.displayName = 'StockListSkeleton';

// 策略项组件（使用 memo 优化）
const StrategyItem = memo(({
  strategy,
  onToggle,
  isLoading
}: {
  strategy: StrategyInfo;
  onToggle: (strategyId: string, active: boolean) => void;
  isLoading: boolean;
}) => {
  const strategyTypeBadge = useMemo(() => {
    switch (strategy.strategy_type) {
      case 'NATURAL_LANGUAGE':
        return <Badge variant="info">NL</Badge>;
      case 'PYTHON':
        return <Badge variant="success">PY</Badge>;
      default:
        return <Badge variant="default">{strategy.strategy_type}</Badge>;
    }
  }, [strategy.strategy_type]);

  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
      <div className="flex items-center gap-2">
        {strategyTypeBadge}
        <div>
          <div className="text-sm font-medium text-white">{strategy.display_name}</div>
          <div className="text-xs text-muted">{strategy.description}</div>
        </div>
      </div>
      <button
        type="button"
        onClick={() => onToggle(strategy.id, !strategy.is_active)}
        disabled={isLoading}
        className={`
          px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer
          ${strategy.is_active
            ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
            : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
          }
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
      >
        {strategy.is_active ? 'Active' : 'Inactive'}
      </button>
    </div>
  );
});

StrategyItem.displayName = 'StrategyItem';

// 匹配徽章组件（使用 memo 优化）
const MatchBadge = memo(({ matched }: { matched: boolean }) => {
  return matched ? (
    <Badge variant="success" glow>✓</Badge>
  ) : (
    <Badge variant="danger">✗</Badge>
  );
});

MatchBadge.displayName = 'MatchBadge';

// 股票候选卡片组件（使用 memo 优化）
const StockCandidateCard = memo(({
  candidate,
  rank,
  isSelected,
  onClick
}: {
  candidate: StockCandidateInfo;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}) => {
  const changePct = candidate.extra_data?.change_pct;
  const controlDegree = candidate.extra_data?.control_degree;
  const purpleDays = candidate.extra_data?.purple_days;
  const momentum2PrevColor = candidate.extra_data?.momentum2_prev_color;
  const momentum2HeightChangePct = candidate.extra_data?.momentum2_height_change_pct;

  // 使用 useMemo 优化计算
  const getChangePctColor = useCallback((pct: number | undefined | null) => {
    if (pct === undefined || pct === null) return 'text-muted';
    if (pct > 0) return 'text-red-400';
    if (pct < 0) return 'text-green-400';
    return 'text-secondary';
  }, []);

  const formatChangePct = useCallback((pct: number | undefined | null) => {
    if (pct === undefined || pct === null) return '-';
    const sign = pct > 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
  }, []);

  const renderMomentum2Display = useMemo(() => {
    if (!momentum2PrevColor) return <span>-</span>;
    
    const colorMap: Record<string, string> = {
      '红': 'text-red-400',
      '黄': 'text-yellow-400',
      '绿': 'text-green-400',
      '蓝': 'text-blue-400',
    };
    
    if (momentum2PrevColor === '红') {
      const heightChange = momentum2HeightChangePct !== undefined && momentum2HeightChangePct !== null 
        ? `+${momentum2HeightChangePct.toFixed(0)}%` 
        : '';
      return (
        <span>
          <span className={colorMap['红']}>红</span>
          <span className={colorMap['红']}>红</span>
          {heightChange && <span className="text-red-400">{heightChange}</span>}
        </span>
      );
    } else if (momentum2PrevColor === '黄') {
      return (
        <span>
          <span className={colorMap['黄']}>黄</span>
          <span className={colorMap['红']}>红</span>
        </span>
      );
    } else if (momentum2PrevColor === '绿') {
      return (
        <span>
          <span className={colorMap['绿']}>绿</span>
          <span className={colorMap['红']}>红</span>
        </span>
      );
    } else if (momentum2PrevColor === '蓝') {
      return (
        <span>
          <span className={colorMap['蓝']}>蓝</span>
          <span className={colorMap['红']}>红</span>
        </span>
      );
    } else {
      return <span className={colorMap['红']}>红</span>;
    }
  }, [momentum2PrevColor, momentum2HeightChangePct]);

  const changePctColor = useMemo(() => getChangePctColor(changePct), [getChangePctColor, changePct]);
  const formattedChangePct = useMemo(() => formatChangePct(changePct), [formatChangePct, changePct]);

  return (
    <Card 
      variant="gradient" 
      padding="md" 
      className={`animate-fade-in cursor-pointer transition-all duration-200 ${isSelected ? 'ring-2 ring-cyan/50 shadow-[0_0_20px_rgba(0,212,255,0.2)]' : 'hover:ring-1 hover:ring-cyan/30'}`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-cyan/20 flex items-center justify-center text-cyan font-bold text-sm">
            #{rank}
          </div>
          <div>
            <div className="text-lg font-bold text-white">{candidate.stock_code}</div>
            <div className="text-xs text-muted">{candidate.stock_name || '-'}</div>
            {candidate.sectors && candidate.sectors.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {candidate.sectors.map((sector, i) => (
                  <span key={i} className="text-xs px-1.5 py-0.5 rounded-full bg-cyan/15 text-cyan">
                    {sector}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
        <div className="text-right">
          <div className="text-lg font-bold text-cyan">{candidate.overall_score.toFixed(2)}</div>
          <div className="text-xs text-muted">Score</div>
        </div>
      </div>
      
      <div className="grid grid-cols-4 gap-2 mb-3">
        <div className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className={`text-sm font-bold ${changePctColor}`}>
            {formattedChangePct}
          </div>
          <div className="text-[9px] text-muted">涨跌幅</div>
        </div>
        <div className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="text-sm font-bold text-yellow-400">
            {controlDegree !== undefined && controlDegree !== null ? controlDegree.toFixed(1) : '-'}
          </div>
          <div className="text-[9px] text-muted">控盘度</div>
        </div>
        <div className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="text-sm font-bold text-purple-400">
            {purpleDays !== undefined ? purpleDays : '-'}
          </div>
          <div className="text-[9px] text-muted">连紫数</div>
        </div>
        <div className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="text-sm font-bold">
            {renderMomentum2Display}
          </div>
          <div className="text-[9px] text-muted">动能二号</div>
        </div>
      </div>
      
      <div className="space-y-2">
        {candidate.strategy_matches.map((match: StrategyMatchInfo) => (
          <div key={match.strategy_id} className="flex items-center justify-between py-1">
            <div className="flex items-center gap-2">
              <MatchBadge matched={match.matched} />
              <span className="text-sm text-secondary">{match.strategy_name}</span>
            </div>
            <div className="text-xs text-muted">
              {match.score !== undefined ? match.score.toFixed(2) : '-'}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
});

StockCandidateCard.displayName = 'StockCandidateCard';

// 进度条组件
const ProgressBar = memo(({ progress, label }: { progress: number; label?: string }) => (
  <div className="w-full">
    {label && <div className="text-sm text-muted mb-1">{label}</div>}
    <div className="w-full bg-white/10 rounded-full h-2">
      <div 
        className="bg-cyan rounded-full h-2 transition-all duration-300 ease-out"
        style={{ width: `${Math.min(progress, 100)}%` }}
      />
    </div>
    <div className="text-xs text-cyan mt-1 text-right">{Math.round(progress)}%</div>
  </div>
));

ProgressBar.displayName = 'ProgressBar';

// 主页面组件
const StockSelectorPageOptimized: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [candidates, setCandidates] = useState<StockCandidateInfo[]>([]);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [screeningProgress, setScreeningProgress] = useState(0);
  const [screeningStatus, setScreeningStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // 使用 useCallback 优化回调
  const handleToggleStrategy = useCallback((strategyId: string, active: boolean) => {
    setStrategies(prev =>
      prev.map(s =>
        s.id === strategyId ? { ...s, is_active: active } : s
      )
    );
  }, []);

  const handleScreenStocks = useCallback(async () => {
    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);
    setScreeningProgress(0);
    setScreeningStatus('准备中...');
    setCandidates([]);

    try {
      setScreeningStatus('加载策略...');
      const activeStrategyIds = strategies.filter(s => s.is_active).map(s => s.id);
      setScreeningProgress(10);

      setScreeningStatus('筛选股票中...');
      const result = await stockSelectorApi.screenStocks(
        {
          strategy_ids: activeStrategyIds,
          top_n: 20
        }
      );
      
      setScreeningProgress(100);
      setScreeningStatus('完成！');
      setCandidates(result.candidates);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setScreeningStatus('已取消');
        setError('选股操作已取消');
      } else {
        setScreeningStatus('失败');
        setError(err.message || '选股失败');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [strategies]);

  const handleCancelScreening = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const handleStockClick = useCallback((stockCode: string) => {
    setSelectedStock(stockCode === selectedStock ? null : stockCode);
  }, [selectedStock]);

  // 加载策略列表
  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const data = await stockSelectorApi.getStrategies();
        setStrategies(data.strategies);
      } catch (err) {
        console.error('Failed to load strategies:', err);
      }
    };

    loadStrategies();
  }, []);

  // 使用 useMemo 优化渲染
  const activeStrategiesCount = useMemo(() => 
    strategies.filter(s => s.is_active).length,
    [strategies]
  );

  const sortedCandidates = useMemo(() => 
    [...candidates].sort((a, b) => b.overall_score - a.overall_score),
    [candidates]
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            智能选股系统
          </h1>
          <p className="text-muted">基于多策略的智能股票筛选</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* 左侧策略面板 */}
          <div className="lg:col-span-1">
            <Card variant="glass" padding="lg">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white">选股策略</h2>
                <span className="text-xs text-muted">
                  {activeStrategiesCount} / {strategies.length} 激活
                </span>
              </div>
              
              <div className="space-y-1 max-h-96 overflow-y-auto">
                {strategies.map(strategy => (
                  <StrategyItem
                    key={strategy.id}
                    strategy={strategy}
                    onToggle={handleToggleStrategy}
                    isLoading={isLoading}
                  />
                ))}
              </div>

              <div className="mt-4 space-y-2">
                {!isLoading ? (
                  <button
                    type="button"
                    onClick={handleScreenStocks}
                    disabled={activeStrategiesCount === 0}
                    className="
                      w-full py-3 rounded-lg font-semibold text-white
                      bg-gradient-to-r from-cyan-500 to-blue-500
                      hover:from-cyan-400 hover:to-blue-400
                      transition-all duration-200 shadow-lg shadow-cyan/25
                      disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-cyan-500 disabled:hover:to-blue-500
                    "
                  >
                    开始选股
                  </button>
                ) : (
                  <div className="space-y-3">
                    <ProgressBar progress={screeningProgress} label={screeningStatus} />
                    <button
                      type="button"
                      onClick={handleCancelScreening}
                      className="
                        w-full py-2 rounded-lg font-medium text-white
                        bg-red-500/20 border border-red-500/30
                        hover:bg-red-500/30
                        transition-all duration-200
                      "
                    >
                      取消选股
                    </button>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* 右侧候选股票面板 */}
          <div className="lg:col-span-3">
            {error && (
              <div className="mb-4 p-4 bg-red-500/20 border border-red-500/30 rounded-lg">
                <div className="flex items-center gap-2">
                  <span className="text-red-400">⚠️</span>
                  <span className="text-red-300">{error}</span>
                </div>
              </div>
            )}

            {isLoading ? (
              <Suspense fallback={<StockListSkeleton count={12} />}>
                <StockListSkeleton count={12} />
              </Suspense>
            ) : candidates.length > 0 ? (
              <>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">
                    选股结果 ({candidates.length} 只)
                  </h2>
                  <div className="text-sm text-muted">
                    按综合评分降序排列
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {sortedCandidates.map((candidate, index) => (
                    <StockCandidateCard
                      key={candidate.stock_code}
                      candidate={candidate}
                      rank={index + 1}
                      isSelected={selectedStock === candidate.stock_code}
                      onClick={() => handleStockClick(candidate.stock_code)}
                    />
                  ))}
                </div>
              </>
            ) : (
              <Card variant="glass" padding="xl" className="text-center">
                <div className="text-6xl mb-4">📈</div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  开始选股
                </h3>
                <p className="text-muted">
                  选择策略后点击"开始选股"来筛选优质股票
                </p>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockSelectorPageOptimized;
