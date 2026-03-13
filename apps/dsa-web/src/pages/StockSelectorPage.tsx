import type React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { stockSelectorApi } from '../api/stockSelector';
import { Card, Badge } from '../components/common';
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
}> = ({ candidate, rank }) => {
  return (
    <Card variant="gradient" padding="md" className="animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-cyan/20 flex items-center justify-center text-cyan font-bold text-sm">
            #{rank}
          </div>
          <div>
            <div className="text-lg font-bold text-white">{candidate.stock_code}</div>
            <div className="text-xs text-muted">{candidate.stock_name || '-'}</div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-lg font-bold text-cyan">{candidate.overall_score.toFixed(2)}</div>
          <div className="text-xs text-muted">Score</div>
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

  const [candidates, setCandidates] = useState<StockCandidateInfo[]>([]);
  const [isScreening, setIsScreening] = useState(false);
  const [screeningError, setScreeningError] = useState<string | null>(null);

  const [topN, setTopN] = useState(5);
  const [stockCodes, setStockCodes] = useState('');
  const [strategyTypeFilter, setStrategyTypeFilter] = useState<'ALL' | 'NATURAL_LANGUAGE' | 'PYTHON'>('ALL');

  const fetchStrategies = useCallback(async () => {
    setIsLoadingStrategies(true);
    try {
      const response = await stockSelectorApi.getStrategies();
      setStrategies(response.strategies);
    } catch (err) {
      console.error('Failed to fetch strategies:', err);
    } finally {
      setIsLoadingStrategies(false);
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
    try {
      const codes = stockCodes.trim() ? stockCodes.trim().split(/[,\s]+/).filter(Boolean) : undefined;
      const response = await stockSelectorApi.screenStocks({
        top_n: topN,
        stock_codes: codes,
      });
      setCandidates(response.candidates);
    } catch (err) {
      setScreeningError(err instanceof Error ? err.message : 'Screening failed');
    } finally {
      setIsScreening(false);
    }
  }, [stockCodes, topN]);

  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  const filteredStrategies = strategies.filter(s => {
    if (strategyTypeFilter === 'ALL') return true;
    return s.strategy_type === strategyTypeFilter;
  });

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2 max-w-4xl flex-wrap">
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
          <div className="flex items-center gap-1 whitespace-nowrap">
            <span className="text-xs text-muted">Top</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topN}
              onChange={(e) => setTopN(parseInt(e.target.value, 10) || 5)}
              disabled={isScreening}
              className="input-terminal w-14 text-center text-xs py-2"
            />
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
        </div>
        {screeningError && (
          <p className="mt-2 text-xs text-danger">{screeningError}</p>
        )}
      </header>

      <main className="flex-1 flex overflow-hidden p-3 gap-3">
        <div className="flex flex-col gap-3 w-80 flex-shrink-0 overflow-y-auto">
          <Card padding="md">
            <div className="mb-3">
              <span className="label-uppercase">Strategies</span>
            </div>
            <div className="flex gap-2 mb-3">
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
          </Card>
        </div>

        <section className="flex-1 overflow-y-auto">
          {isScreening ? (
            <div className="flex flex-col items-center justify-center h-64">
              <div className="w-10 h-10 border-3 border-cyan/20 border-t-cyan rounded-full animate-spin" />
              <p className="mt-3 text-secondary text-sm">Screening stocks...</p>
            </div>
          ) : candidates.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
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
            <div className="grid gap-3 max-w-4xl">
              {candidates.map((candidate, index) => (
                <StockCandidateCard
                  key={candidate.stock_code}
                  candidate={candidate}
                  rank={index + 1}
                />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default StockSelectorPage;
