import type React from 'react';
import { useState, useEffect, useCallback, useRef, useMemo, useLayoutEffect } from 'react';
import { stockSelectorApi } from '../api/stockSelector';
import { getIntradayConfig } from '../api/intraday';
import { Card, Badge, KlineChart } from '../components/common';
import type {
  StrategyInfo,
  StockCandidateInfo,
  StrategyMatchInfo,
  ScreenProgressStatus,
} from '../types/stockSelector';
import { getCachedStockSelector, setCachedStockSelector } from '../cache/stockSelectorCache';

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
    // 优先使用同步缓存，避免 useEffect 时序问题导致选中标的被覆盖
    const cached = getCachedStockSelector();
    if (cached?.candidates && cached.candidates.length > 0) return cached.candidates;
    const saved = localStorage.getItem('stockSelector_candidates');
    return saved ? JSON.parse(saved) : [];
  });
  const [screeningError, setScreeningError] = useState<string | null>(null);
  const [selectedStock, setSelectedStock] = useState<StockCandidateInfo | null>(() => {
    const cached = getCachedStockSelector();
    return cached?.selectedStock || null;
  });

  const [stockCodes, setStockCodes] = useState('');
  const [strategyTypeFilter, setStrategyTypeFilter] = useState<'ALL' | 'NATURAL_LANGUAGE' | 'PYTHON'>('ALL');
  
  const [updateData, setUpdateData] = useState(false);
  const [updateRealtime, setUpdateRealtime] = useState(false);
  const [selectedStrategyIds, setSelectedStrategyIds] = useState<string[]>([]);
  const [isStrategyDropdownOpen, setIsStrategyDropdownOpen] = useState(false);

  const [sortField, setSortField] = useState<'score' | 'purpleDays' | 'controlDegree'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterEnabled, setFilterEnabled] = useState(false);
  
  // 收藏策略状态管理
  const [favoriteStrategyIds, setFavoriteStrategyIds] = useState<string[]>(() => {
    const saved = localStorage.getItem('favorite_strategies');
    return saved ? JSON.parse(saved) : [];
  });
  
  const strategyDropdownRef = useRef<HTMLDivElement>(null);

  // 异步选股进度追踪
  const [screenTask, setScreenTask] = useState<ScreenProgressStatus | null>(null);
  const [showScreenModal, setShowScreenModal] = useState(false);
  const [isStartingScreen, setIsStartingScreen] = useState(false);
  const screenTaskRef = useRef<ScreenProgressStatus | null>(null);
  screenTaskRef.current = screenTask;
  const screenPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const screenPollingIntervalRef = useRef(1000);  // 从后端配置动态获取

  // Canvas 进度条
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawProgressBar = useCallback((progress: number, barStatus: string, barStage: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) {
      requestAnimationFrame(() => drawProgressBar(progress, barStatus, barStage));
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const radius = h / 2;
    const pct = Math.max(0, progress) / 100;
    const progressW = w * pct;

    // 阶段颜色
    let barColor = '#06b6d4';
    if (barStatus === 'completed') {
      barColor = '#34d399';
    } else if (barStatus === 'cancelled') {
      barColor = 'rgba(234,179,8,0.6)';
    } else if (barStatus === 'failed') {
      barColor = 'rgba(239,68,68,0.6)';
    } else if (barStage === 'update_realtime') {
      barColor = '#06b6d4';
    } else if (barStage === 'update_data') {
      barColor = '#a855f7';
    } else if (barStage === 'screening') {
      barColor = '#f59e0b';
    }

    // 背景
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.moveTo(radius, 0);
    ctx.arcTo(w, 0, w, h, radius);
    ctx.arcTo(w, h, 0, h, radius);
    ctx.arcTo(0, h, 0, 0, radius);
    ctx.arcTo(0, 0, w, 0, radius);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fill();

    // 进度填充
    if (progressW > 0) {
      ctx.beginPath();
      ctx.moveTo(radius, 0);
      ctx.arcTo(progressW, 0, progressW, h, radius);
      ctx.arcTo(progressW, h, 0, h, radius);
      ctx.arcTo(0, h, 0, 0, radius);
      ctx.arcTo(0, 0, progressW, 0, radius);
      ctx.closePath();
      ctx.fillStyle = barColor;
      ctx.fill();
    }
  }, []);

  useLayoutEffect(() => {
    if (screenTask && screenTask.total_stocks > 0) {
      drawProgressBar(screenTask.stage_progress, screenTask.status, screenTask.stage);
    }
  }, [screenTask?.stage_progress, screenTask?.status, screenTask?.stage, screenTask?.total_stocks, drawProgressBar]);
  
  // 切换收藏状态
  const toggleFavorite = (strategyId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavoriteStrategyIds(prev => {
      const newFavorites = prev.includes(strategyId)
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId];
      localStorage.setItem('favorite_strategies', JSON.stringify(newFavorites));
      return newFavorites;
    });
  };
  
  // 星星图标组件
  const StarIcon = ({ isFavorited, onClick }: { isFavorited: boolean; onClick: (e: React.MouseEvent) => void }) => (
    <button
      type="button"
      onClick={onClick}
      className="flex-shrink-0 w-5 h-5 p-0.5 hover:bg-white/10 rounded transition-colors cursor-pointer"
    >
      {isFavorited ? (
        <svg className="w-full h-full text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
        </svg>
      ) : (
        <svg className="w-full h-full text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118L12 16.055l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.783-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
        </svg>
      )}
    </button>
  );

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
    setIsStartingScreen(true);
    setScreeningError(null);

    const codes = stockCodes.trim() ? stockCodes.trim().split(/[,\s]+/).filter(Boolean) : undefined;
    const strategyIds = selectedStrategyIds.length > 0 ? selectedStrategyIds : undefined;

    setShowScreenModal(true);
    try {
      const status = await stockSelectorApi.screenStocksAsync({
        stock_codes: codes,
        update_data: updateData,
        update_realtime: updateRealtime,
        strategy_ids: strategyIds,
      });
      setScreenTask(status);
    } catch (err) {
      setScreeningError(err instanceof Error ? err.message : '选股启动失败');
      setShowScreenModal(false);
    } finally {
      setIsStartingScreen(false);
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

  // 从缓存恢复选股页面状态（切换页面后返回时快速恢复）
  useEffect(() => {
    const cached = getCachedStockSelector();
    if (cached) {
      console.log('[选股缓存] 从缓存恢复数据: 候选数', cached.candidates.length, '选中标的', cached.selectedStock?.stock_code);
      if (cached.candidates.length > 0) {
        setCandidates(cached.candidates);
        localStorage.setItem('stockSelector_candidates', JSON.stringify(cached.candidates));
      }
      if (cached.selectedStock) {
        setSelectedStock(cached.selectedStock);
      }
      if (cached.strategies.length > 0) {
        setStrategies(cached.strategies);
      }
      if (cached.selectedStrategyIds.length > 0) {
        setSelectedStrategyIds(cached.selectedStrategyIds);
      }
      if (cached.stockCodes) {
        setStockCodes(cached.stockCodes);
      }
      setStrategyTypeFilter(cached.strategyTypeFilter);
      setUpdateData(cached.updateData);
      setUpdateRealtime(cached.updateRealtime);
    }
  }, []);

  // 关键状态变更时自动保存到缓存
  useEffect(() => {
    setCachedStockSelector({
      selectedStock,
      candidates,
      strategies,
      selectedStrategyIds,
      stockCodes,
      strategyTypeFilter,
      updateData,
      updateRealtime,
    });
  }, [selectedStock, candidates, strategies, selectedStrategyIds, stockCodes, strategyTypeFilter, updateData, updateRealtime]);

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

  // 从后端获取轮询配置（页面加载时调用一次）
  useEffect(() => {
    getIntradayConfig().then(cfg => {
      screenPollingIntervalRef.current = cfg.screen_async_polling_interval_ms;
    }).catch(() => {
      // 降级使用默认值，已在 ref 初始化时设置
    });
  }, []);

  // 异步选股进度轮询
  useEffect(() => {
    if (screenTask?.status === 'running') {
      if (screenPollingRef.current) return;
      screenPollingRef.current = setInterval(async () => {
        try {
          const status = await stockSelectorApi.getScreenAsyncStatus(screenTaskRef.current?.task_id);
          setScreenTask(status);
          if (status.status !== 'running') {
            if (screenPollingRef.current) {
              clearInterval(screenPollingRef.current);
              screenPollingRef.current = null;
            }
            // 完成后自动更新结果
            if (status.result?.success && status.result.candidates) {
              setCandidates(status.result.candidates);
              localStorage.setItem('stockSelector_candidates', JSON.stringify(status.result.candidates));
            }
          }
        } catch {
          // ignore polling errors
        }
      }, screenPollingIntervalRef.current);
    } else {
      if (screenPollingRef.current) {
        clearInterval(screenPollingRef.current);
        screenPollingRef.current = null;
      }
    }
    return () => {
      if (screenPollingRef.current) {
        clearInterval(screenPollingRef.current);
        screenPollingRef.current = null;
      }
    };
  }, [screenTask?.status, screenTask?.task_id]);

  // 移除固定高度设置，保持自然的 flexbox 布局
  // 右侧 section 已经有 overflow-y-auto，配合 flex-1 会自动处理滚动

  const filteredStrategies = useMemo(() => {
    const filtered = strategies.filter(s => {
      if (strategyTypeFilter === 'ALL') return true;
      return s.strategy_type === strategyTypeFilter;
    });
    // 收藏的策略置顶
    return [...filtered].sort((a, b) => {
      const aFav = favoriteStrategyIds.includes(a.id);
      const bFav = favoriteStrategyIds.includes(b.id);
      if (aFav && !bFav) return -1;
      if (!aFav && bFav) return 1;
      return 0;
    });
  }, [strategies, strategyTypeFilter, favoriteStrategyIds]);
  
  // 下拉框用的策略列表（同样收藏置顶）
  const dropdownStrategies = useMemo(() => {
    return [...strategies].sort((a, b) => {
      const aFav = favoriteStrategyIds.includes(a.id);
      const bFav = favoriteStrategyIds.includes(b.id);
      if (aFav && !bFav) return -1;
      if (!aFav && bFav) return 1;
      return 0;
    });
  }, [strategies, favoriteStrategyIds]);

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
      } else if (sortField === 'purpleDays') {
        const purpleDaysA = a.extra_data?.purple_days;
        const purpleDaysB = b.extra_data?.purple_days;

        if (purpleDaysA === undefined || purpleDaysA === null) {
          return 1;
        }
        if (purpleDaysB === undefined || purpleDaysB === null) {
          return -1;
        }

        return sortOrder === 'desc' ? purpleDaysB - purpleDaysA : purpleDaysA - purpleDaysB;
      } else {
        // controlDegree
        const cdA = a.extra_data?.control_degree;
        const cdB = b.extra_data?.control_degree;

        if (cdA === undefined || cdA === null) {
          return 1;
        }
        if (cdB === undefined || cdB === null) {
          return -1;
        }

        return sortOrder === 'desc' ? cdB - cdA : cdA - cdB;
      }
    });

    return result;
  }, [candidates, sortField, sortOrder, filterEnabled]);

  useEffect(() => {
    if (processedCandidates.length > 0 && (!selectedStock || !processedCandidates.find(c => c.stock_code === selectedStock.stock_code))) {
      // 检查缓存中是否有选中的标的，如果有则优先使用缓存的选择而非锁定为第一个
      const cached = getCachedStockSelector();
      if (cached?.selectedStock && processedCandidates.find(c => c.stock_code === cached.selectedStock!.stock_code)) {
        setSelectedStock(cached.selectedStock);
      } else {
        setSelectedStock(processedCandidates[0]);
      }
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
              disabled={screenTask?.status === 'running'}
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
              disabled={screenTask?.status === 'running'}
              className="input-terminal text-xs py-2 px-3 min-w-64 flex items-center justify-between"
            >
              <span>
                {selectedStrategyIds.length === 0 
                  ? 'All Strategies' 
                  : `${selectedStrategyIds.length} Selected`}
              </span>
              <svg 
                className={`w-4 h-4 transition-transform ${isStrategyDropdownOpen ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {isStrategyDropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-72 bg-elevated border border-white/15 rounded-xl shadow-2xl z-[9999] max-h-80 overflow-y-auto">
                {dropdownStrategies.length === 0 ? (
                  <div className="px-3 py-4 text-center text-xs text-muted">
                    Loading strategies...
                  </div>
                ) : (
                  dropdownStrategies.map((strategy) => (
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
                        disabled={screenTask?.status === 'running'}
                        className="rounded w-4 h-4"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white truncate">{strategy.display_name}</div>
                        <div className="text-xs text-muted truncate">{strategy.description}</div>
                      </div>
                      <StarIcon
                        isFavorited={favoriteStrategyIds.includes(strategy.id)}
                        onClick={(e) => toggleFavorite(strategy.id, e)}
                      />
                    </label>
                  ))
                )}
              </div>
            )}
          </div>
          
          <button
            type="button"
            onClick={handleScreen}
            disabled={isStartingScreen || screenTask?.status === 'running'}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
          >
            {(isStartingScreen || screenTask?.status === 'running') ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                选股中...
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
              disabled={screenTask?.status === 'running'}
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
              disabled={screenTask?.status === 'running'}
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
          {candidates.length === 0 ? (
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
                    onClick={() => {
                      if (sortField !== 'controlDegree') {
                        setSortField('controlDegree');
                        setSortOrder('desc');
                      } else {
                        setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                      }
                    }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 border cursor-pointer ${
                      sortField === 'controlDegree'
                        ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                        : 'border-white/10 bg-transparent text-muted hover:border-white/20 hover:text-secondary'
                    }`}
                  >
                    <span>控盘度</span>
                    {sortField === 'controlDegree' && (
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

      {/* 异步选股进度弹窗 */}
      {showScreenModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60" onClick={() => {
            if (!isStartingScreen && screenTask?.status !== 'running') setShowScreenModal(false);
          }} />
          <div className="relative terminal-card border border-white/10 rounded-xl shadow-2xl p-5 w-full max-w-md mx-4">
            {screenTask ? (
              <>
            {/* 标题 + 阶段图标 */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                {screenTask.status === 'running' && screenTask.stage === 'update_realtime' && (
                  <span className="text-cyan text-lg">⏱</span>
                )}
                {screenTask.status === 'running' && screenTask.stage === 'update_data' && (
                  <span className="text-purple-400 text-lg">📊</span>
                )}
                {screenTask.status === 'running' && screenTask.stage === 'screening' && (
                  <span className="text-amber text-lg">🔍</span>
                )}
                {screenTask.status === 'running' && (screenTask.stage === 'preparing' || !screenTask.stage) && (
                  <span className="text-muted text-lg">⌛</span>
                )}
                {screenTask.status === 'completed' && <span className="text-emerald text-lg">✅</span>}
                {screenTask.status === 'cancelled' && <span className="text-yellow-400 text-lg">⏹</span>}
                {screenTask.status === 'failed' && <span className="text-red-400 text-lg">❌</span>}
                <h3 className={`text-sm font-semibold ${
                  screenTask.status === 'completed' ? 'text-emerald' :
                  screenTask.status === 'cancelled' ? 'text-yellow-400' :
                  screenTask.status === 'failed' ? 'text-red-400' :
                  'text-white'
                }`}>
                  {screenTask.status === 'running' && screenTask.stage === 'update_realtime' && '更新实时数据'}
                  {screenTask.status === 'running' && screenTask.stage === 'update_data' && '更新历史数据'}
                  {screenTask.status === 'running' && screenTask.stage === 'screening' && '策略筛选'}
                  {screenTask.status === 'running' && (screenTask.stage === 'preparing' || !screenTask.stage) && '准备中...'}
                  {screenTask.status === 'completed' && '选股完成'}
                  {screenTask.status === 'cancelled' && '已取消'}
                  {screenTask.status === 'failed' && '选股失败'}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (screenTask.status !== 'running') setShowScreenModal(false);
                }}
                className="text-muted hover:text-white p-0.5"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Canvas 进度条 */}
            {screenTask.total_stocks > 0 ? (
              <div className="mb-3">
                <canvas
                  ref={canvasRef}
                  className="w-full h-3 block"
                  style={{ borderRadius: '9999px' }}
                />
                <div className="text-xs text-muted mt-1 text-right">
                  {screenTask.status === 'completed' || screenTask.status === 'failed' || screenTask.status === 'cancelled'
                    ? `${screenTask.stage_progress.toFixed(1)}%`
                    : `${screenTask.processed_stocks} / ${screenTask.total_stocks} (${screenTask.stage_progress.toFixed(1)}%)`}
                </div>
              </div>
            ) : (
              <div className="text-xs text-muted mb-3 text-center py-2">
                {screenTask.status === 'running' ? '正在获取股票列表...' : '\u00A0'}
              </div>
            )}

            {/* 阶段相关统计卡片 */}
            {screenTask.total_stocks > 0 && screenTask.status === 'running' && (
              <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
                <div className="bg-white/[0.03] rounded px-2 py-1.5">
                  <div className="text-muted">总股票数</div>
                  <div className="text-white font-mono">{screenTask.total_stocks}</div>
                </div>
                <div className="bg-white/[0.03] rounded px-2 py-1.5">
                  <div className="text-muted">
                    {screenTask.stage === 'screening' ? '已筛选' : '已处理'}
                  </div>
                  <div className={`font-mono ${
                    screenTask.stage === 'update_realtime' ? 'text-cyan' :
                    screenTask.stage === 'update_data' ? 'text-purple-400' :
                    screenTask.stage === 'screening' ? 'text-amber' :
                    'text-cyan'
                  }`}>
                    {screenTask.processed_stocks}
                  </div>
                </div>
                <div className="bg-white/[0.03] rounded px-2 py-1.5">
                  <div className="text-muted">耗时</div>
                  <div className="text-white font-mono">
                    {screenTask.elapsed_seconds > 0
                      ? `${Math.floor(screenTask.elapsed_seconds / 60)}分${Math.floor(screenTask.elapsed_seconds % 60)}秒`
                      : '-'}
                  </div>
                </div>
              </div>
            )}

            {/* 完成/取消/失败时显示总数 */}
            {screenTask.total_stocks > 0 && screenTask.status !== 'running' && (
              <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                <div className="bg-white/[0.03] rounded px-2 py-1.5">
                  <div className="text-muted">总股票数</div>
                  <div className="text-white font-mono">{screenTask.total_stocks}</div>
                </div>
                <div className="bg-white/[0.03] rounded px-2 py-1.5">
                  <div className="text-muted">耗时</div>
                  <div className="text-white font-mono">
                    {screenTask.elapsed_seconds > 0
                      ? `${Math.floor(screenTask.elapsed_seconds / 60)}分${Math.floor(screenTask.elapsed_seconds % 60)}秒`
                      : '-'}
                  </div>
                </div>
              </div>
            )}

            {/* 当前处理 */}
            {screenTask.status === 'running' && screenTask.current_code && (
              <div className="text-xs text-muted mb-3">
                当前: <span className="text-white font-mono">{screenTask.current_code}</span>
                {screenTask.current_name && <span className="ml-1 text-white/70">{screenTask.current_name}</span>}
              </div>
            )}

            {/* 错误列表 */}
            {screenTask.errors && screenTask.errors.length > 0 && (
              <div className="mb-3">
                <div className="text-xs text-red-400 mb-1">错误:</div>
                <div className="max-h-24 overflow-y-auto text-xs font-mono bg-black/20 rounded px-2 py-1">
                  {screenTask.errors.map((e, i) => (
                    <div key={i} className="text-red-400/80 truncate">{e}</div>
                  ))}
                </div>
              </div>
            )}

            {/* 失败时显示错误信息 */}
            {screenTask.status === 'failed' && screenTask.result?.error && (
              <div className="mb-3 text-xs text-red-400 bg-red-400/10 rounded px-2 py-1.5">
                {screenTask.result.error}
              </div>
            )}

            {/* 按钮 */}
            <div className="flex items-center gap-2">
              {screenTask.status !== 'running' && (
                <button
                  type="button"
                  onClick={() => setShowScreenModal(false)}
                  className="px-3 py-1.5 text-xs font-medium rounded border border-white/10 text-muted hover:text-white hover:border-white/30 transition-colors"
                >
                  关闭
                </button>
              )}
              {screenTask.status === 'completed' && screenTask.result && (
                <div className="text-xs text-emerald">
                  已筛选 {screenTask.result.candidates.length} 只股票
                </div>
              )}
              {screenTask.status === 'running' && (
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      if (screenTask.task_id) {
                        await stockSelectorApi.cancelScreenAsync(screenTask.task_id);
                      }
                    } catch {}
                  }}
                  className="px-3 py-1.5 text-xs font-medium rounded border border-red-500/30 text-red-400 hover:text-red-300 hover:border-red-500/60 transition-colors"
                >
                  取消
                </button>
              )}
            </div>
                </>
            ) : (
              <div className="flex flex-col items-center justify-center py-8">
                <svg className="w-8 h-8 animate-spin text-white/40 mb-3" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <p className="text-sm text-white/70 font-medium mb-1">正在启动选股任务</p>
                <p className="text-xs text-muted">获取股票列表...</p>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
};

export default StockSelectorPage;
