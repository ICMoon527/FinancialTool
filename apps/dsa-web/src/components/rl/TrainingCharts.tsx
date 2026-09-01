import React from 'react';
import type { EChartsOption } from 'echarts';
import { BaseChart } from '../charts/BaseChart';
import type { RLMetricsHistory } from '../../types/rl';

/**
 * RL 实时监控图表区：奖励曲线 / 损失+TD误差 / 探索率衰减 / 验证Sharpe
 * 指标历史由父组件从 store 传入，echarts 增量更新（lazyUpdate）
 */

const EMPTY: Partial<RLMetricsHistory> = {};

const EpisodeXAxis = (length: number) => ({
  type: 'category' as const,
  data: Array.from({ length }, (_, i) => i + 1),
  name: 'Episode',
  nameTextStyle: { color: '#a0a0b0' },
  axisLabel: { color: '#a0a0b0' },
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
});

const chartCard = 'rounded-xl border border-white/10 bg-[#08080c] p-3';

/** 奖励曲线（含滑动平均） */
const RewardChart: React.FC<{ data: Partial<RLMetricsHistory> }> = ({ data }) => {
  const rewards = data.episodeRewards ?? [];
  // 滑动平均（窗口10）平滑曲线
  const ma: number[] = [];
  const win = 10;
  for (let i = 0; i < rewards.length; i++) {
    const s = rewards.slice(Math.max(0, i - win + 1), i + 1);
    ma.push(Number((s.reduce((a, b) => a + b, 0) / s.length).toFixed(4)));
  }
  const option: EChartsOption = {
    title: { text: 'Episode 奖励', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    legend: { data: ['单集奖励', '滑动平均(10)'], top: 5, right: 10 },
    grid: { left: 45, right: 15, top: 40, bottom: 30 },
    xAxis: EpisodeXAxis(rewards.length),
    yAxis: { type: 'value', axisLabel: { color: '#a0a0b0' } },
    series: [
      {
        name: '单集奖励',
        type: 'line',
        data: rewards,
        symbol: 'none',
        lineStyle: { width: 1, opacity: 0.45 },
        color: '#6f61f1',
      },
      {
        name: '滑动平均(10)',
        type: 'line',
        data: ma,
        symbol: 'none',
        lineStyle: { width: 2 },
        color: '#00d4ff',
      },
    ],
  };
  return (
    <div className={chartCard}>
      <BaseChart option={option} height={230} lazyUpdate />
    </div>
  );
};

/** 损失 + TD误差 双轴 */
const LossChart: React.FC<{ data: Partial<RLMetricsHistory> }> = ({ data }) => {
  const losses = data.losses ?? [];
  const tdErrors = data.tdErrors ?? [];
  const option: EChartsOption = {
    title: { text: '损失 / TD误差', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    legend: { data: ['Loss', 'TD误差'], top: 5, right: 10 },
    grid: { left: 65, right: 65, top: 40, bottom: 30 },
    xAxis: EpisodeXAxis(Math.max(losses.length, tdErrors.length)),
    yAxis: [
      {
        type: 'value',
        name: 'Loss',
        nameLocation: 'middle',
        nameGap: 42,
        nameTextStyle: { color: '#ffaa00' },
        axisLabel: { color: '#a0a0b0' },
        splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
      },
      {
        type: 'value',
        name: 'TD',
        nameLocation: 'middle',
        nameGap: 42,
        nameTextStyle: { color: '#ff4466' },
        axisLabel: { color: '#a0a0b0' },
        splitLine: { show: false },
      },
    ],
    series: [
      { name: 'Loss', type: 'line', data: losses, symbol: 'none', color: '#ffaa00' },
      { name: 'TD误差', type: 'line', yAxisIndex: 1, data: tdErrors, symbol: 'none', color: '#ff4466' },
    ],
  };
  return (
    <div className={chartCard}>
      <BaseChart option={option} height={230} lazyUpdate />
    </div>
  );
};

/** 探索率 ε 衰减 */
const EpsilonChart: React.FC<{ data: Partial<RLMetricsHistory> }> = ({ data }) => {
  const eps = data.epsilons ?? [];
  const option: EChartsOption = {
    title: { text: '探索率 (Epsilon)', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 55, right: 15, top: 40, bottom: 30 },
    xAxis: EpisodeXAxis(eps.length),
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#a0a0b0' } },
    series: [
      { name: 'Epsilon', type: 'line', data: eps, symbol: 'none', areaStyle: { opacity: 0.15 }, color: '#00ff88' },
    ],
  };
  return (
    <div className={chartCard}>
      <BaseChart option={option} height={230} lazyUpdate />
    </div>
  );
};

/** 验证集 Sharpe / 收益率 */
const ValChart: React.FC<{ data: Partial<RLMetricsHistory> }> = ({ data }) => {
  const sharpes = data.valSharpeRatios ?? [];
  const returns = data.valReturns ?? [];
  const option: EChartsOption = {
    title: { text: '验证集 Sharpe / 收益率', left: 10, top: 5, textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    legend: { data: ['Sharpe', '收益率'], top: 5, right: 10 },
    grid: { left: 65, right: 65, top: 40, bottom: 30 },
    xAxis: EpisodeXAxis(Math.max(sharpes.length, returns.length)),
    yAxis: [
      {
        type: 'value',
        name: 'Sharpe',
        nameLocation: 'middle',
        nameGap: 42,
        nameTextStyle: { color: '#00d4ff' },
        axisLabel: { color: '#a0a0b0' },
        splitLine: { lineStyle: { type: 'dashed', opacity: 0.3 } },
      },
      {
        type: 'value',
        name: '收益率',
        nameLocation: 'middle',
        nameGap: 42,
        nameTextStyle: { color: '#00ff88' },
        axisLabel: {
          color: '#a0a0b0',
          formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
        },
        splitLine: { show: false },
      },
    ],
    series: [
      { name: 'Sharpe', type: 'line', data: sharpes, symbol: 'circle', symbolSize: 5, color: '#00d4ff' },
      {
        name: '收益率',
        type: 'line',
        yAxisIndex: 1,
        data: returns,
        symbol: 'none',
        lineStyle: { type: 'dashed' },
        color: '#00ff88',
      },
    ],
  };
  return (
    <div className={chartCard}>
      <BaseChart option={option} height={230} lazyUpdate />
    </div>
  );
};

const EmptyHint: React.FC<{ text: string }> = ({ text }) => (
  <div className={`${chartCard} h-[270px] flex items-center justify-center`}>
    <div className="text-center text-gray-500">
      <div className="text-3xl mb-2">📈</div>
      <p className="text-sm">{text}</p>
    </div>
  </div>
);

/** 4 张监控图网格（无数据时显示占位提示） */
export const TrainingCharts: React.FC<{ metrics: Partial<RLMetricsHistory>; running: boolean }> = ({
  metrics = EMPTY,
  running,
}) => {
  const hasData = (metrics.episodeRewards?.length ?? 0) > 0;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {hasData ? (
        <>
          <RewardChart data={metrics} />
          <LossChart data={metrics} />
          <EpsilonChart data={metrics} />
          <ValChart data={metrics} />
        </>
      ) : (
        <>
          <EmptyHint text={running ? '等待第一个 Episode 完成...' : '启动训练后此处显示实时监控图表'} />
          <EmptyHint text={running ? '等待数据...' : '配置参数后点击「开始训练」'} />
        </>
      )}
    </div>
  );
};

export default TrainingCharts;
