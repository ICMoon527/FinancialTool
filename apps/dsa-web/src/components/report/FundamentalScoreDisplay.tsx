import type React from 'react';
import { useState } from 'react';
import type { FundamentalScore } from '../../types/analysis';
import { Card } from '../common';

interface FundamentalScoreDisplayProps {
  data: FundamentalScore;
}

const DIMENSION_LABELS: Record<string, string> = {
  '三表勾稽真实性': '勾稽真实性',
  '盈利与现金流质量': '盈利与现金流',
  '营运效率与议价能力': '营运效率',
  '杜邦驱动质量': '杜邦驱动',
  '估值合理性': '估值合理性',
};

const DIMENSION_MAX: Record<string, number> = {
  '三表勾稽真实性': 20,
  '盈利与现金流质量': 30,
  '营运效率与议价能力': 15,
  '杜邦驱动质量': 20,
  '估值合理性': 15,
};

const getScoreColor = (score: number, max: number): string => {
  const ratio = score / max;
  if (ratio >= 0.7) return 'bg-emerald-500';
  if (ratio >= 0.5) return 'bg-yellow-500';
  if (ratio >= 0.3) return 'bg-orange-500';
  return 'bg-red-500';
};

const getRatingColor = (rating: string): string => {
  switch (rating) {
    case '优秀': return 'text-emerald-400';
    case '良好': return 'text-cyan-400';
    case '一般': return 'text-yellow-400';
    default: return 'text-red-400';
  }
};

const RAW_DATA_LABELS: Record<string, string> = {
  roe: 'ROE (%)',
  roa: 'ROA (%)',
  net_profit_margin: '净利率 (%)',
  deducted_net_profit_margin: '扣非净利率 (%)',
  gross_profit_margin: '毛利率 (%)',
  operating_profit_margin: '营业利润率 (%)',
  revenue_growth_yoy: '营收增速 (%)',
  net_profit_growth_yoy: '净利增速 (%)',
  debt_to_asset_ratio: '资产负债率 (%)',
  asset_turnover: '资产周转率',
  equity_multiplier: '权益乘数',
  operating_cashflow_to_revenue: '经营现金流/营收 (%)',
  roic: 'ROIC (%)',
  pb: 'PB',
  pe: 'PE',
  eps: 'EPS',
};

export const FundamentalScoreDisplay: React.FC<FundamentalScoreDisplayProps> = ({ data }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showReasons, setShowReasons] = useState(false);
  const [showRawData, setShowRawData] = useState(false);

  const reasons = data.reasons || [];
  const rawData = data.rawData || {};
  const hasRawData = Object.keys(rawData).length > 0;

  return (
    <Card variant="bordered" padding="md" className="text-left">
      <div className="mb-3 flex items-baseline gap-2">
        <span className="label-uppercase">FUNDAMENTAL</span>
        <h3 className="text-base font-semibold text-white mt-0.5">基本面评分</h3>
      </div>

      {/* 行业标签 */}
      {data.industry && (
        <div className="flex items-center gap-2 text-xs text-muted mb-3 pb-3 border-b border-white/5">
          <span>行业:</span>
          <code className="font-mono text-xs text-cyan bg-cyan/10 px-1.5 py-0.5 rounded">
            {data.industry}
          </code>
        </div>
      )}

      {/* 总分 + 评级 */}
      <div className="flex items-center gap-4 mb-4">
        <div className="flex-shrink-0 w-16 h-16 rounded-full bg-elevated flex items-center justify-center border-2 border-white/10">
          <span className="text-2xl font-bold text-white">{Math.round(data.totalScore)}</span>
        </div>
        <div>
          <div className={`text-lg font-semibold ${getRatingColor(data.rating)}`}>
            {data.rating}
          </div>
          <div className="text-xs text-muted">总分 100</div>
        </div>
      </div>

      {/* 五个维度得分 */}
      <div className="space-y-2 mb-3">
        {Object.entries(data.dimensionScores).map(([dim, score]) => {
          const max = DIMENSION_MAX[dim] || 20;
          const pct = Math.min((score / max) * 100, 100);
          const label = DIMENSION_LABELS[dim] || dim;
          return (
            <div key={dim} className="flex items-center gap-2">
              <span className="text-xs text-secondary w-24 flex-shrink-0">{label}</span>
              <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${getScoreColor(score, max)}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-muted w-10 text-right">
                {score.toFixed(1)}/{max}
              </span>
            </div>
          );
        })}
      </div>

      {/* 折叠区域 */}
      <div className="space-y-2">
        {/* 评分详情 */}
        <div>
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="w-full flex items-center justify-between p-2.5 rounded-lg bg-elevated hover:bg-hover transition-colors"
          >
            <span className="text-xs text-white">评分详情</span>
            <svg
              className={`w-3.5 h-3.5 text-muted transition-transform ${showDetails ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showDetails && (
            <div className="mt-2 animate-fade-in">
              {/* 评分理由 */}
              {reasons.length > 0 && (
                <div className="mb-2">
                  <button
                    type="button"
                    onClick={() => setShowReasons(!showReasons)}
                    className="w-full flex items-center justify-between p-2 rounded bg-base hover:bg-hover transition-colors"
                  >
                    <span className="text-xs text-secondary">
                      评分理由 ({reasons.length} 项)
                    </span>
                    <svg
                      className={`w-3 h-3 text-muted transition-transform ${showReasons ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  {showReasons && (
                    <ul className="mt-1 space-y-1 max-h-60 overflow-y-auto">
                      {reasons.map((reason, idx) => {
                        const isWarn = reason.includes('0分') || reason.includes('风险') || reason.includes('异常');
                        const isGood = reason.includes('满分') || reason.includes('优秀') || reason.includes('良好');
                        return (
                          <li
                            key={idx}
                            className={`text-xs px-2 py-1 rounded ${
                              isWarn ? 'text-red-400 bg-red-400/5' : isGood ? 'text-emerald-400 bg-emerald-400/5' : 'text-secondary'
                            }`}
                          >
                            {reason}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              )}

              {/* 原始财报数据 */}
              {hasRawData && (
                <div>
                  <button
                    type="button"
                    onClick={() => setShowRawData(!showRawData)}
                    className="w-full flex items-center justify-between p-2 rounded bg-base hover:bg-hover transition-colors"
                  >
                    <span className="text-xs text-secondary">
                      原始财报数据 ({Object.keys(rawData).length} 项)
                    </span>
                    <svg
                      className={`w-3 h-3 text-muted transition-transform ${showRawData ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  {showRawData && (
                    <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 max-h-60 overflow-y-auto">
                      {Object.entries(rawData).map(([key, value]) => {
                        const label = RAW_DATA_LABELS[key] || key;
                        const displayValue = value !== null && value !== undefined
                          ? (typeof value === 'number' ? value.toFixed(2) : String(value))
                          : '--';
                        return (
                          <div key={key} className="flex justify-between text-xs py-1 border-b border-white/5">
                            <span className="text-muted">{label}</span>
                            <span className="text-white font-mono">{displayValue}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};