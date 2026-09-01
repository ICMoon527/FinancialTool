import React from 'react';
import type { EChartsOption } from 'echarts';
import { BaseChart } from '../charts/BaseChart';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';
import type { CompareModelResult } from '../../types/rl';

/**
 * 多模型对比评估面板（同一批抽样数据 / 同基准）
 * 评估中：显示总体进度 + 暂停/恢复/终止控制
 * 完成后：模型指标对比表格 + 所有模型累积收益曲线（vs 共用基准）
 */

// 模型曲线配色（与基准区分开）
const MODEL_COLORS = [
  '#00d4ff',
  '#ff9f43',
  '#a29bfe',
  '#00cec9',
  '#fd79a8',
  '#fdcb6e',
  '#74b9ff',
  '#55efc4',
  '#e17055',
  '#6c5ce7',
];

const fmtPct = (v: number) => `${(v * 100).toFixed(2)}%`;

const MetricCell: React.FC<{ label: string; value: string; color?: string }> = ({
  label,
  value,
  color = 'text-gray-100',
}) => (
  <td className="px-2 py-1.5 text-center">
    <p className={`text-sm font-semibold tabular-nums ${color}`}>{value}</p>
    <p className="text-[10px] text-gray-500">{label}</p>
  </td>
);

export const CompareResultPanel: React.FC = () => {
  const compareEvaluating = useRLStore((s) => s.compareEvaluating);
  const compareResult = useRLStore((s) => s.compareResult);
  const compareDone = useRLStore((s) => s.compareDone);
  const compareTotal = useRLStore((s) => s.compareTotal);
  const compareProgress = useRLStore((s) => s.compareProgress);
  const compareMessage = useRLStore((s) => s.compareMessage);
  const comparePaused = useRLStore((s) => s.comparePaused);
  const compareModelIds = useRLStore((s) => s.compareModelIds);
  const compareCurrentModelIdx = useRLStore((s) => s.compareCurrentModelIdx);
  const compareModelDone = useRLStore((s) => s.compareModelDone);
  const compareModelTotal = useRLStore((s) => s.compareModelTotal);
  const pauseCompare = useRLStore((s) => s.pauseCompare);
  const resumeCompare = useRLStore((s) => s.resumeCompare);
  const stopCompare = useRLStore((s) => s.stopCompare);

  // 评估中：进度 + 控制
  if (compareEvaluating) {
    // 按模型拆分进度：已完成的模型满格，当前模型显示已做天数，未开始的为 0
    const perModel = compareModelIds.map((modelId, idx) => {
      let done: number;
      if (compareCurrentModelIdx == null) {
        done = 0;
      } else if (idx < compareCurrentModelIdx) {
        done = compareModelTotal;
      } else if (idx === compareCurrentModelIdx) {
        done = compareModelDone;
      } else {
        done = 0;
      }
      return { modelId, done, total: compareModelTotal, idx };
    });

    return (
      <Card title="模型对比评估" variant="bordered">
        <div className="py-6 px-2">
          <div className="text-center text-gray-400 text-sm mb-4">
            <div className="inline-block w-5 h-5 border-2 border-purple-500/30 border-t-purple-400 rounded-full animate-spin mr-2 align-middle" />
            <span className="align-middle">
              {comparePaused
                ? '对比评估已暂停'
                : '正在同基准对比评估多个模型（逐日回放验证集交易日）'}
            </span>
          </div>

          {/* 按模型拆分的进度列表 */}
          <div className="space-y-2 max-w-2xl mx-auto mb-4">
            {perModel.map((pm) => {
              const isDone = compareCurrentModelIdx != null && pm.idx < compareCurrentModelIdx;
              const isCurrent = pm.idx === compareCurrentModelIdx;
              const pct = pm.total > 0 ? Math.min((pm.done / pm.total) * 100, 100) : 0;
              return (
                <div key={pm.modelId} className="flex items-center gap-2 text-xs">
                  <span
                    className={`w-2 h-2 rounded-full shrink-0 ${
                      isDone ? 'bg-green-400' : isCurrent ? 'bg-purple-400 animate-pulse' : 'bg-slate-600'
                    }`}
                  />
                  <span className="font-mono text-gray-300 truncate shrink-0 max-w-[180px]" title={pm.modelId}>
                    模型{pm.idx + 1}: {pm.modelId}
                  </span>
                  <span className="text-gray-500 shrink-0">{pm.done}/{pm.total}</span>
                  <div className="h-1.5 rounded-full bg-slate-700/60 flex-1 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isDone ? 'bg-green-500' : isCurrent ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'bg-slate-600/40'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-10 text-right text-gray-500 shrink-0">
                    {isDone ? '完成' : isCurrent ? `${pct.toFixed(0)}%` : '等待'}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mb-2">
            <div className="h-2 rounded-full bg-slate-700/60 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  comparePaused ? 'bg-amber-500' : 'bg-gradient-to-r from-purple-500 to-pink-500'
                }`}
                style={{ width: `${Math.min(Math.max(compareProgress, 0), 100)}%` }}
              />
            </div>
            <div className="flex justify-between mt-1.5 text-xs text-gray-400">
              <span className="font-mono">
                总进度 {compareDone} / {compareTotal}
                {comparePaused && <span className="text-amber-400 ml-2">已暂停</span>}
              </span>
              <span className={`font-mono ${comparePaused ? 'text-amber-400' : 'text-purple-400'}`}>
                {compareProgress.toFixed(1)}%
              </span>
            </div>
          </div>

          <p className="font-mono text-[11px] text-gray-500 truncate" title={compareMessage}>
            ▸ {compareMessage || '准备中...'}
          </p>

          <div className="flex justify-center gap-2 mt-4">
            {comparePaused ? (
              <Button variant="outline" size="sm" onClick={() => void resumeCompare()}>
                继续评估
              </Button>
            ) : (
              <Button variant="outline" size="sm" onClick={() => void pauseCompare()}>
                暂停评估
              </Button>
            )}
            <Button variant="danger" size="sm" onClick={() => void stopCompare()}>
              终止评估
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  if (!compareResult) {
    return (
      <Card title="模型对比评估" variant="bordered">
        <div className="py-10 text-center text-gray-500 text-sm">
          在左侧模型列表中勾选 2 个及以上模型，点击「对比评估（同基准）」查看对比结果
          <span className="block mt-2 text-xs text-gray-600">
            所有模型将在同一批抽样数据（同一基准）上评估，保证结果可比
          </span>
        </div>
      </Card>
    );
  }

  const { models, benchmarkReturns } = compareResult;
  const n = Math.max(
    benchmarkReturns.length,
    ...models.map((m) => m.cumulativeReturns.length)
  );
  const xData = Array.from({ length: n }, (_, i) => i + 1);

  // 模型曲线 series（每个模型一条线，标注模型 ID）
  const modelSeries = models.map((m, idx) => ({
    name: m.modelId,
    type: 'line' as const,
    data: m.cumulativeReturns,
    symbol: 'none',
    lineStyle: { width: 2 },
    color: MODEL_COLORS[idx % MODEL_COLORS.length],
  }));

  const benchmarkSeries = {
    name: '基准(买入持有)',
    type: 'line' as const,
    data: benchmarkReturns,
    symbol: 'none',
    lineStyle: { type: 'dashed' as const, width: 1.5 },
    color: '#ffffff',
    opacity: 0.7,
  };

  const option: EChartsOption = {
    title: { text: '累积收益对比（同一批数据 / 同基准）', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => `${((v as number) * 100).toFixed(2)}%` },
    legend: {
      type: 'scroll',
      top: 5,
      right: 10,
      textStyle: { fontSize: 10 },
      data: [...models.map((m) => m.modelId), '基准(买入持有)'],
    },
    grid: { left: 60, right: 20, top: 48, bottom: 30 },
    xAxis: { type: 'category', data: xData, name: '交易日', axisLabel: { color: '#a0a0b0' } },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#a0a0b0', formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
    },
    series: [...modelSeries, benchmarkSeries],
  };

  // 按总收益降序排列，便于比较
  const sorted = [...models].sort((a, b) => b.summaryMetrics.totalReturn - a.summaryMetrics.totalReturn);

  return (
    <div className="space-y-3">
      {/* 指标对比表格 */}
      <Card title="指标对比" variant="bordered" padding="sm">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-white/10">
                <th className="text-left px-2 py-1.5 font-medium">模型</th>
                <th className="px-2 py-1.5 text-center font-medium">总收益</th>
                <th className="px-2 py-1.5 text-center font-medium">夏普比率</th>
                <th className="px-2 py-1.5 text-center font-medium">胜率</th>
                <th className="px-2 py-1.5 text-center font-medium">最大回撤</th>
                <th className="px-2 py-1.5 text-center font-medium">交易次数</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((m: CompareModelResult, idx) => {
                const s = m.summaryMetrics;
                return (
                  <tr key={m.modelId} className="border-b border-white/5 hover:bg-slate-800/40">
                    <td className="px-2 py-1.5">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span
                          className="inline-block w-2 h-2 rounded-full shrink-0"
                          style={{ background: MODEL_COLORS[idx % MODEL_COLORS.length] }}
                        />
                        <span className="font-mono text-gray-200 truncate" title={m.modelId}>
                          {m.modelId}
                        </span>
                      </div>
                    </td>
                    <MetricCell
                      label="总收益"
                      value={fmtPct(s.totalReturn)}
                      color={s.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}
                    />
                    <MetricCell
                      label="夏普"
                      value={s.sharpeRatio.toFixed(2)}
                      color={s.sharpeRatio >= 1 ? 'text-green-400' : s.sharpeRatio >= 0 ? 'text-yellow-400' : 'text-red-400'}
                    />
                    <MetricCell label="胜率" value={`${(s.winRate * 100).toFixed(1)}%`} />
                    <MetricCell label="回撤" value={fmtPct(s.maxDrawdown)} color="text-red-400" />
                    <MetricCell label="交易次数" value={String(s.totalTrades)} />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 累积收益对比曲线 */}
      <div className="rounded-xl border border-white/10 bg-[#08080c] p-3">
        <BaseChart option={option} height={320} />
      </div>

      <p className="text-[11px] text-gray-600 px-1">
        评估样本数 {compareResult.samples.length} 天 · 所有模型共用同一批数据与买入持有基准，
        保证结果可横向对比
      </p>
    </div>
  );
};

export default CompareResultPanel;
