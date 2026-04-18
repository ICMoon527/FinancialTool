import type React from 'react';
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { stockSelectorApi } from '../api/stockSelector';
import { Card, Badge, KlineChart } from '../components/common';
import type {
  StrategyInfo,
  StockCandidateInfo,
  StrategyMatchInfo,
} from '../types/stockSelector';

const strategyTypeBadge = (type: string) => {
  switch (type) {
    case 'NATURAL_LANGUAGE':
      return <Badge variant="info">NL</Badge>;
    case 'PYTHON':
      return <Badge variant="success">PY</Badge>;
    default:
      return <Badge variant="default">{type}</Badge>;
  }
};

const matchBadge = (matched: boolean) => {
  return matched ? (
    <Badge variant="success" glow>✓</Badge>
  ) : (
    <Badge variant="danger">✗</Badge>
  );
};

const StrategyItem: React.FC<{
  strategy: StrategyInfo;
  onToggle: (strategyId: string, active: boolean) => void;
  isLoading: boolean;
}> = ({ strategy, onToggle, isLoading }) => {
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
      <div className="flex items-center gap-2">
        {strategyTypeBadge(strategy.strategy_type)}
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
};

const StockCandidateCard: React.FC<{
  candidate: StockCandidateInfo;
  rank: number;
  isSelected: boolean;
  onClick: () => void;
}> = ({ candidate, rank, isSelected, onClick }) => {
  const changePct = candidate.extra_data?.change_pct;
  const controlDegree = candidate.extra_data?.control_degree;
  const purpleDays = candidate.extra_data?.purple_days;
  const momentum2PrevColor = candidate.extra_data?.momentum2_prev_color;
  const momentum2HeightChangePct = candidate.extra_data?.momentum2_height_change_pct;

  const getChangePctColor = (pct: number | undefined | null) => {
    if (pct === undefined || pct === null) return 'text-muted';
    if (pct > 0) return 'text-red-400';
    if (pct < 0) return 'text-green-400';
    return 'text-secondary';
  };

  const formatChangePct = (pct: number | undefined | null) => {
    if (pct === undefined || pct === null) return '-';
    const sign = pct > 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
  };

  const renderMomentum2Display = () => {
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
  };

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
          <div className={`text-sm font-bold ${getChangePctColor(changePct)}`}>
            {formatChangePct(changePct)}
          </div>
          <div className="text-[9px] text-muted">涨跌幅</div>
        </div>
        <div className="bg-white/5 rounded-lg p-1.5 text-center">
          <div className="text-sm font-bold text-yellow-400">
            {controlDegree !== undefined && controlDegree !== null ? controlDegree.toFixed(2) : '-'}
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
            {renderMomentum2Display()}
          </div>
          <div className="text-[9px] text-muted">动能二号</div>
        </div>
      </div>
      
      <div className="space-y-2">
        {candidate.strategy_matches.map((match: StrategyMatchInfo) => (
          <div key={match.strategy_id} className="flex items-center justify-between py-1">
            <div className="flex items-center gap-2">
              {matchBadge(match.matched)}
              <span className="text-sm text-secondary">{match.strategy_name}</span>
            </div>
            <span className="text-xs font-mono text-muted">{match.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
};

const StockSelectorPage: React.FC = () => {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);
  const [isTogglingStrategy, setIsTogglingStrategy] = useState(false);

  const [candidates, setCandidates] = useState<StockCandidateInfo[]>(() => {
    const saved = localStorage.getItem('stockSelector_candidates');
    return saved ? JSON.parse(saved) : [];
  });
  const [isScreening, setIsScreening] = useState(false);
  const [screeningError, setScreeningError] = useState<string | null>(null);
  const [screeningStage, setScreeningStage] = useState('');
  const [selectedStock, setSelectedStock] = useState<StockCandidateInfo | null>(null);

  const [stockCodes, setStockCodes] = useState('');
  const [strategyTypeFilter, setStrategyTypeFilter] = useState<'ALL' | 'NATURAL_LANGUAGE' | 'PYTHON'>('ALL');
  
  const [updateData, setUpdateData] = useState(false);
  const [updateRealtime, setUpdateRealtime] = useState(false);
  const [selectedStrategyIds, setSelectedStrategyIds] = useState<string[]>([]);
  const [isStrategyDropdownOpen, setIsStrategyDropdownOpen] = useState(false);

  const [sortField, setSortField] = useState<'score' | 'purpleDays'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterEnabled, setFilterEnabled] = useState(false);
  
  const strategyDropdownRef = useRef<HTMLDivElement>(null);

  const fetchStrategies = useCallback(async () => {
    try {
      const response = await stockSelectorApi.getStrategies();
      console.log('Strategies response:', response);
      if (response && response.strategies) {
        return response.strategies;
      } else {
        console.error('Invalid strategies response:', response);
        return [];
      }
    } catch (err) {
      console.error('Failed to fetch strategies:', err);
      return [];
    }
  }, []);

  const toggleStrategy = useCallback(async (strategyId: string, active: boolean) => {
    setIsTogglingStrategy(true);
    try {
      if (active) {
        await stockSelectorApi.activateStrategy(strategyId);
      } else {
        await stockSelectorApi.deactivateStrategy(strategyId);
      }
      await fetchStrategies();
    } catch (err) {
      console.error('Failed to toggle strategy:', err);
    } finally {
      setIsTogglingStrategy(false);
    }
  }, [fetchStrategies]);

  const handleScreen = useCallback(async () => {
    setIsScreening(true);
    setScreeningError(null);
    setScreeningStage('正在选股，请稍候...');

    try {
      const codes = stockCodes.trim() ? stockCodes.trim().split(/[,\s]+/).filter(Boolean) : undefined;
      const strategyIds = selectedStrategyIds.length > 0 ? selectedStrategyIds : undefined;
      const response = await stockSelectorApi.screenStocks({
        stock_codes: codes,
        update_data: updateData,
        update_realtime: updateRealtime,
        strategy_ids: strategyIds,
      });
      
      setScreeningStage('完成！');
      
      await new Promise(resolve => setTimeout(resolve, 300));
      setCandidates(response.candidates);
      localStorage.setItem('stockSelector_candidates', JSON.stringify(response.candidates));
    } catch (err) {
      setScreeningError(err instanceof Error ? err.message : 'Screening failed');
    } finally {
      setIsScreening(false);
    }
  }, [stockCodes, updateData, updateRealtime, selectedStrategyIds]);

  useEffect(() => {
    const initPage = async () => {
      setIsLoadingStrategies(true);
      try {
        const strategiesResult = await fetchStrategies();
        setStrategies(strategiesResult);
      } catch (err) {
        console.error('Failed to init page:', err);
        setStrategies([]);
      } finally {
        setIsLoadingStrategies(false);
      }
    };
    initPage();
  }, [fetchStrategies]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (strategyDropdownRef.current && !strategyDropdownRef.current.contains(event.target as Node)) {
        setIsStrategyDropdownOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // 移除固定高度设置，保持自然的 flexbox 布局
  // 右侧 section 已经有 overflow-y-auto，配合 flex-1 会自动处理滚动

  const filteredStrategies = strategies.filter(s => {
    if (strategyTypeFilter === 'ALL') return true;
    return s.strategy_type === strategyTypeFilter;
  });

  const processedCandidates = useMemo(() => {
    let result = [...candidates];

    if (filterEnabled) {
      result = result.filter(candidate => {
        const changePct = candidate.extra_data?.change_pct;
        return changePct !== undefined && changePct !== null && Math.abs(changePct) <= 3;
      });
    }

    result.sort((a, b) => {
      if (sortField === 'score') {
        return sortOrder === 'desc' ? b.overall_score - a.overall_score : a.overall_score - b.overall_score;
      } else {
        const purpleDaysA = a.extra_data?.purple_days;
        const purpleDaysB = b.extra_data?.purple_days;

        if (purpleDaysA === undefined || purpleDaysA === null) {
          return 1;
        }
        if (purpleDaysB === undefined || purpleDaysB === null) {
          return -1;
        }

        return sortOrder === 'desc' ? purpleDaysB - purpleDaysA : purpleDaysA - purpleDaysB;
      }
    });

    return result;
  }, [candidates, sortField, sortOrder, filterEnabled]);

  useEffect(() => {
    if (processedCandidates.length > 0 && (!selectedStock || !processedCandidates.find(c => c.stock_code === selectedStock.stock_code))) {
      setSelectedStock(processedCandidates[0]);
    }
  }, [processedCandidates]);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2 max-w-6xl flex-wrap">
          <div className="flex-1 relative min-w-[200px]">
            <input
              type="text"
              value={stockCodes}
              onChange={(e) => setStockCodes(e.target.value.toUpperCase())}
              placeholder="Stock codes (comma/space separated, leave empty for all)"
              disabled={isScreening}
              className="input-terminal w-full"
            />
          </div>
          
          <div className="relative" ref={strategyDropdownRef}>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsStrategyDropdownOpen(!isStrategyDropdownOpen);
              }}
              disabled={isScreening}
              className="input-terminal flex items-center gap-2 whitespace-nowrap"
            >
              <span className="text-xs">
                {selectedStrategyIds.length === 0 
                  ? 'All Strategies' 
                  : `${selectedStrategyIds.length} Selected`}
              </span>
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {isStrategyDropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-72 bg-elevated border border-white/15 rounded-xl shadow-2xl z-[9999] max-h-80 overflow-y-auto">
                {strategies.length === 0 ? (
                  <div className="px-3 py-4 text-center text-xs text-muted">
                    Loading strategies...
                  </div>
                ) : (
                  strategies.map((strategy) => (
                    <label
                      key={strategy.id}
                      className="flex items-center gap-2 px-3 py-2.5 hover:bg-white/10 cursor-pointer transition-colors border-b border-white/5 last:border-0"
                    >
                      <input
                        type="checkbox"
                        checked={selectedStrategyIds.includes(strategy.id)}
                        onChange={(e) => {
                          e.stopPropagation();
                          if (e.target.checked) {
                            setSelectedStrategyIds([...selectedStrategyIds, strategy.id]);
                          } else {
                            setSelectedStrategyIds(selectedStrategyIds.filter(id => id !== strategy.id));
                          }
                        }}
                        disabled={isScreening}
                        className="rounded w-4 h-4"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white truncate">{strategy.display_name}</div>
                        <div className="text-xs text-muted truncate">{strategy.description}</div>
                      </div>
                    </label>
                  ))
                )}
              </div>
            )}
          </div>
          
          <button
            type="button"
            onClick={handleScreen}
            disabled={isScreening}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
          >
            {isScreening ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Screening...
              </>
            ) : (
              'Screen Stocks'
            )}
          </button>
          
          <label className="flex items-center gap-2 whitespace-nowrap cursor-pointer">
            <input
              type="checkbox"
              checked={updateData}
              onChange={(e) => {
                setUpdateData(e.target.checked);
                if (e.target.checked) {
                  setUpdateRealtime(false);
                }
              }}
              disabled={isScreening}
              className="rounded"
            />
            <span className="text-xs text-secondary">Update Data</span>
          </label>
          <label className="flex items-center gap-2 whitespace-nowrap cursor-pointer">
            <input
              type="checkbox"
              checked={updateRealtime}
              onChange={(e) => {
                setUpdateRealtime(e.target.checked);
                if (e.target.checked) {
                  setUpdateData(false);
                }
              }}
              disabled={isScreening}
              className="rounded"
            />
            <span className="text-xs text-secondary">Update Realtime</span>
          </label>
        </div>
        {screeningError && (
          <p className="mt-2 text-xs text-danger">{screeningError}</p>
        )}
      </header>

      <main className="flex-1 flex overflow-hidden p-3 gap-3">
        <div className="w-80 flex-shrink-0 rounded-2xl terminal-card p-4 overflow-hidden flex flex-col min-h-0 h-[1400px]">
          <div className="mb-3 flex-shrink-0">
            <span className="label-uppercase">Strategies</span>
          </div>
          <div className="flex gap-2 mb-3 flex-shrink-0">
            <button
              type="button"
              onClick={() => setStrategyTypeFilter('ALL')}
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${
                strategyTypeFilter === 'ALL'
                  ? 'bg-cyan/20 text-cyan border border-cyan/30'
                  : 'bg-transparent text-muted border border-white/10 hover:border-white/20'
              }`}
            >
              All
            </button>
            <button
              type="button"
              onClick={() => setStrategyTypeFilter('NATURAL_LANGUAGE')}
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${
                strategyTypeFilter === 'NATURAL_LANGUAGE'
                  ? 'bg-purple/20 text-purple border border-purple/30'
                  : 'bg-transparent text-muted border border-white/10 hover:border-white/20'
              }`}
            >
              NL
            </button>
            <button
              type="button"
              onClick={() => setStrategyTypeFilter('PYTHON')}
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-all ${
                strategyTypeFilter === 'PYTHON'
                  ? 'bg-emerald/20 text-emerald border border-emerald/30'
                  : 'bg-transparent text-muted border border-white/10 hover:border-white/20'
              }`}
            >
              PY
            </button>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0">
            {isLoadingStrategies ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
              </div>
            ) : filteredStrategies.length === 0 ? (
              <p className="text-xs text-muted text-center py-4">
                No strategies available
              </p>
            ) : (
              <div className="space-y-1">
                {filteredStrategies.map((strategy) => (
                  <StrategyItem
                    key={strategy.id}
                    strategy={strategy}
                    onToggle={toggleStrategy}
                    isLoading={isTogglingStrategy}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        <section className="flex-1 overflow-hidden flex gap-3 h-[1400px]">
          {isScreening ? (
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-full max-w-md">
                <div className="flex items-center justify-center gap-3 mb-6">
                  <div className="w-10 h-10 border-3 border-cyan/20 border-t-cyan rounded-full animate-spin" />
                </div>
                
                <div className="mb-3 text-center">
                  <p className="text-white text-sm font-medium">
                    {screeningStage || '正在选股，请稍候...'}
                  </p>
                </div>
                
                <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-cyan/30 via-cyan to-cyan/30 rounded-full animate-pulse" style={{ width: '100%' }} />
                </div>
                
                <div className="mt-2 text-center">
                  <span className="text-xs text-muted">数据处理中...</span>
                </div>
              </div>
            </div>
          ) : candidates.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <div className="w-12 h-12 mb-3 rounded-xl bg-elevated flex items-center justify-center">
                <svg className="w-6 h-6 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-white mb-1.5">No Results</h3>
              <p className="text-xs text-muted max-w-xs">
                Click "Screen Stocks" to find top candidates matching your active strategies
              </p>
            </div>
          ) : (
            <>
              <div className="w-1/2 overflow-y-auto pr-2 py-1 pl-1">
                <div className="mb-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setSortField('score');
                      setSortOrder('desc');
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer ${
                      sortField === 'score'
                        ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                        : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                    }`}
                  >
                    <span>综合评分</span>
                    {sortField === 'score' && (
                      <span className="ml-1">{sortOrder === 'desc' ? '↓' : '↑'}</span>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (sortField !== 'purpleDays') {
                        setSortField('purpleDays');
                        setSortOrder('asc');
                      } else {
                        setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer ${
                      sortField === 'purpleDays'
                        ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                        : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                    }`}
                  >
                    <span>连紫数</span>
                    {sortField === 'purpleDays' && (
                      <span className="ml-1">{sortOrder === 'desc' ? '↓' : '↑'}</span>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => setFilterEnabled(!filterEnabled)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer ${
                      filterEnabled
                        ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                        : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                    }`}
                  >
                    筛选3%以内涨跌幅
                  </button>
                </div>
                <div className="grid gap-4">
                  {processedCandidates.map((candidate, index) => (
                    <StockCandidateCard
                      key={candidate.stock_code}
                      candidate={candidate}
                      rank={index + 1}
                      isSelected={selectedStock?.stock_code === candidate.stock_code}
                      onClick={() => setSelectedStock(candidate)}
                    />
                  ))}
                </div>
              </div>
              
              <div className="w-1/2 overflow-y-auto pl-2">
                {selectedStock && (
                  <div className="terminal-card rounded-2xl p-4">
                    <KlineChart 
                      stockCode={selectedStock.stock_code} 
                      stockName={selectedStock.stock_name} 
                    />
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
};

export default StockSelectorPage;
