import React from 'react';
import type { EChartsOption } from 'echarts';
import { BaseChart } from '../charts/BaseChart';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { rlApi } from '../../api/rl';
import { useRLStore } from '../../stores/rlStore';
import type { DailyReplayResponse } from '../../types/rl';

/**
 * 单日回放面板：选择模型/股票/交易日 → 分时K线 + 买卖点标注 + 每步 reward 柱状图
 */

export const DailyReplayPanel: React.FC = () => {
  const selectedModelId = useRLStore((s) => s.selectedModelId);
  const dailySummaries = useRLStore((s) => s.evaluateResult?.dailySummaries);

  const [stockCode, setStockCode] = React.useState('');
  const [dateStr, setDateStr] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [replay, setReplay] = React.useState<DailyReplayResponse | null>(null);

  // 从评估结果中带入第一个有交易的交易日，方便快速回放
  React.useEffect(() => {
    if (dailySummaries && dailySummaries.length > 0 && !replay) {
      const withTrades = dailySummaries.find((d) => d.tradeCount > 0) ?? dailySummaries[0];
      setStockCode(withTrades.stockCode);
      setDateStr(withTrades.date);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dailySummaries]);

  const handleLoad = async () => {
    if (!selectedModelId) {
      setError('请先在模型列表中选择模型');
      return;
    }
    if (!stockCode.trim() || !dateStr.trim()) {
      setError('请填写股票代码和日期（YYYY-MM-DD）');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await rlApi.getDailyReplay(selectedModelId, stockCode.trim(), dateStr.trim());
      setReplay(data);
    } catch (err) {
      setError(`加载回放失败: ${(err as Error).message}`);
      setReplay(null);
    } finally {
      setLoading(false);
    }
  };

  const inputCls =
    'rounded-lg bg-slate-800/60 border border-slate-600 px-3 py-1.5 text-sm text-gray-200 ' +
    'focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 disabled:opacity-50';

  // K线 + 买卖点 option
  const buildKlineOption = (data: DailyReplayResponse): EChartsOption => {
    const categories = data.klines.map((_, i) => String(i));
    const kdata = data.klines.map((k) => [k.Open, k.Close, k.Low, k.High]);
    const buys: Array<[number, number]> = [];
    const sells: Array<[number, number]> = [];
    data.decisions.forEach((d) => {
      const k = data.klines[d.step];
      if (!k) return;
      if (d.action === 'BUY') buys.push([String(d.step), k.Close] as unknown as [number, number]);
      if (d.action === 'SELL') sells.push([String(d.step), k.Close] as unknown as [number, number]);
    });

    return {
      title: { text: `分时K线 ${data.stockCode} @ ${data.date}`, left: 10, top: 5, textStyle: { fontSize: 13 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      grid: { left: 55, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: categories, axisLabel: { color: '#a0a0b0' } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: '#a0a0b0' } },
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: kdata,
          itemStyle: { color: '#00ff88', color0: '#ff4466', borderColor: '#00ff88', borderColor0: '#ff4466' },
          markPoint: {
            symbolSize: 14,
            data: [
              ...buys.map((p) => ({
                name: 'BUY',
                coord: p,
                symbol: 'triangle',
                symbolRotate: 0,
                itemStyle: { color: '#00ff88' },
                label: { show: false },
              })),
              ...sells.map((p) => ({
                name: 'SELL',
                coord: p,
                symbol: 'triangle',
                symbolRotate: 180,
                itemStyle: { color: '#ff4466' },
                label: { show: false },
              })),
            ],
          },
        },
      ],
    };
  };

  // reward 热力柱状图 option
  const buildRewardOption = (data: DailyReplayResponse): EChartsOption => {
    const rewards = data.rewardHeatmap;
    return {
      title: { text: '每步 Reward 热力', left: 10, top: 5, textStyle: { fontSize: 13 } },
      tooltip: { trigger: 'axis' },
      grid: { left: 55, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: rewards.map((_, i) => String(i)), axisLabel: { color: '#a0a0b0' } },
      yAxis: { type: 'value', axisLabel: { color: '#a0a0b0' } },
      visualMap: {
        show: false,
        min: -Math.max(0.5, ...rewards.map((r) => Math.abs(r))),
        max: Math.max(0.5, ...rewards.map((r) => Math.abs(r))),
        inRange: { color: ['#ff4466', '#333344', '#00ff88'] },
      },
      series: [
        { name: 'Reward', type: 'bar', data: rewards, barWidth: '60%' },
      ],
    };
  };

  return (
    <Card title="单日回放" variant="bordered">
      <div className="space-y-3">
        {/* 查询条件 */}
        <div className="flex flex-wrap items-center gap-2">
          <input
            className={`${inputCls} w-28`}
            placeholder="股票代码"
            value={stockCode}
            onChange={(e) => setStockCode(e.target.value)}
          />
          <input
            className={`${inputCls} w-40`}
            type="date"
            value={dateStr}
            onChange={(e) => setDateStr(e.target.value)}
          />
          <Button
            variant="primary"
            size="sm"
            disabled={!selectedModelId || loading}
            isLoading={loading}
            onClick={() => void handleLoad()}
          >
            加载回放
          </Button>
          {!selectedModelId && <span className="text-xs text-gray-500">（需先在模型列表选择模型并评估）</span>}
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        {/* 回放图表 */}
        {replay && replay.klines.length > 0 ? (
          <div className="space-y-3">
            <div className="rounded-xl border border-white/10 bg-[#08080c] p-3">
              <BaseChart option={buildKlineOption(replay)} height={300} notMerge />
            </div>
            {replay.rewardHeatmap.length > 0 && (
              <div className="rounded-xl border border-white/10 bg-[#08080c] p-3">
                <BaseChart option={buildRewardOption(replay)} height={200} notMerge />
              </div>
            )}
            {/* 交易明细摘要 */}
            {replay.trades.length > 0 && (
              <div className="text-xs text-gray-400">
                当日交易 {replay.trades.length} 笔
                <span className="ml-2 text-gray-500">
                  BUY {replay.decisions.filter((d) => d.action === 'BUY').length} 次 /
                  SELL {replay.decisions.filter((d) => d.action === 'SELL').length} 次
                </span>
              </div>
            )}
          </div>
        ) : replay ? (
          <p className="text-sm text-gray-500 py-6 text-center">该日无K线数据</p>
        ) : null}
      </div>
    </Card>
  );
};

export default DailyReplayPanel;
