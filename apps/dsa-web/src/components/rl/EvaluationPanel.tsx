import React from 'react';
import type { EChartsOption } from 'echarts';
import { BaseChart } from '../charts/BaseChart';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';

/**
 * 评估结果面板：性能指标卡片 + 累积收益 vs 基准曲线
 */

const MetricCard: React.FC<{ label: string; value: string; sub?: string; color?: string }> = ({
  label,
  value,
  sub,
  color = 'text-gray-100',
}) => (
  <div className="rounded-xl border border-white/10 bg-slate-800/40 p-3 text-center">
    <p className="text-[11px] text-gray-500 mb-1">{label}</p>
    <p className={`text-lg font-semibold tabular-nums ${color}`}>{value}</p>
    {sub && <p className="text-[10px] text-gray-500 mt-0.5">{sub}</p>}
  </div>
);

export const EvaluationPanel: React.FC = () => {
  const evaluateResult = useRLStore((s) => s.evaluateResult);
  const evaluating = useRLStore((s) => s.evaluating);
  const selectedModelId = useRLStore((s) => s.selectedModelId);
  const evalDone = useRLStore((s) => s.evalDone);
  const evalTotal = useRLStore((s) => s.evalTotal);
  const evalProgress = useRLStore((s) => s.evalProgress);
  const evalMessage = useRLStore((s) => s.evalMessage);
  const evalPaused = useRLStore((s) => s.evalPaused);
  const pauseEvaluate = useRLStore((s) => s.pauseEvaluate);
  const resumeEvaluate = useRLStore((s) => s.resumeEvaluate);
  const stopEvaluate = useRLStore((s) => s.stopEvaluate);

  if (evaluating) {
    return (
      <Card title="评估结果" variant="bordered">
        <div className="py-8 px-2">
          <div className="text-center text-gray-400 text-sm mb-4">
            <div className="inline-block w-5 h-5 border-2 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin mr-2 align-middle" />
            <span className="align-middle">
              {evalPaused
                ? `评估已暂停（模型 ${selectedModelId}）`
                : `正在评估模型 ${selectedModelId}，逐日回放验证集交易日`}
            </span>
          </div>

          {/* 进度条 */}
          <div className="mb-2">
            <div className="h-2 rounded-full bg-slate-700/60 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  evalPaused
                    ? 'bg-amber-500'
                    : 'bg-gradient-to-r from-cyan-500 to-blue-500'
                }`}
                style={{ width: `${Math.min(Math.max(evalProgress, 0), 100)}%` }}
              />
            </div>
            <div className="flex justify-between mt-1.5 text-xs text-gray-400">
              <span className="font-mono">
                {evalDone} / {evalTotal} 个交易日
                {evalPaused && <span className="text-amber-400 ml-2">已暂停</span>}
              </span>
              <span className={`font-mono ${evalPaused ? 'text-amber-400' : 'text-cyan-400'}`}>
                {evalProgress.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* 当前阶段日志 */}
          <p className="font-mono text-[11px] text-gray-500 truncate" title={evalMessage}>
            ▸ {evalMessage || '准备中...'}
          </p>

          {/* 暂停 / 恢复 / 终止控制 */}
          <div className="flex justify-center gap-2 mt-4">
            {evalPaused ? (
              <Button variant="outline" size="sm" onClick={() => void resumeEvaluate()}>
                继续评估
              </Button>
            ) : (
              <Button variant="outline" size="sm" onClick={() => void pauseEvaluate()}>
                暂停评估
              </Button>
            )}
            <Button variant="danger" size="sm" onClick={() => void stopEvaluate()}>
              终止评估
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  if (!evaluateResult) {
    return (
      <Card title="评估结果" variant="bordered">
        <div className="py-10 text-center text-gray-500 text-sm">
          在左侧模型列表中选择模型并点击「评估」查看结果
        </div>
      </Card>
    );
  }

  const { summaryMetrics: m, cumulativeReturns, benchmarkReturns } = evaluateResult;
  const n = Math.max(cumulativeReturns.length, benchmarkReturns.length);
  const xData = Array.from({ length: n }, (_, i) => i + 1);

  const option: EChartsOption = {
    title: { text: '累积收益 vs 买入持有基准', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => `${((v as number) * 100).toFixed(2)}%` },
    legend: { data: ['策略', '基准'], top: 5, right: 10 },
    grid: { left: 60, right: 20, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: xData, name: '交易日', axisLabel: { color: '#a0a0b0' } },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#a0a0b0', formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
    },
    series: [
      { name: '策略', type: 'line', data: cumulativeReturns, symbol: 'none', color: '#00d4ff' },
      {
        name: '基准',
        type: 'line',
        data: benchmarkReturns,
        symbol: 'none',
        lineStyle: { type: 'dashed' },
        color: '#6f61f1',
      },
    ],
  };

  const fmtPct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="space-y-3">
      {/* 指标卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <MetricCard
          label="夏普比率"
          value={m.sharpeRatio.toFixed(2)}
          color={m.sharpeRatio >= 1 ? 'text-green-400' : m.sharpeRatio >= 0 ? 'text-yellow-400' : 'text-red-400'}
        />
        <MetricCard
          label="总收益"
          value={fmtPct(m.totalReturn)}
          color={m.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <MetricCard label="胜率" value={`${(m.winRate * 100).toFixed(1)}%`} />
        <MetricCard label="最大回撤" value={fmtPct(m.maxDrawdown)} color="text-red-400" />
        <MetricCard label="交易次数" value={String(m.totalTrades)} />
      </div>

      {/* 累积收益曲线 */}
      <div className="rounded-xl border border-white/10 bg-[#08080c] p-3">
        <BaseChart option={option} height={280} />
      </div>
    </div>
  );
};

export default EvaluationPanel;
