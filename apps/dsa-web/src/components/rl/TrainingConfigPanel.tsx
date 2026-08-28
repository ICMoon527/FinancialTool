import React from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';

/**
 * 模型参数配置面板
 *
 * 核心训练参数（算法/轮数/批次/学习率）在此配置，随 POST /train 下发；
 * 高级参数（奖励函数权重、折扣因子、交易成本等 29 项 RL_* 配置）
 * 已注册到设置页「RL Training」分类，此处展示当前值并跳转设置页修改。
 */

interface Props {
  disabled: boolean; // 训练运行中禁用
}

export const TrainingConfigPanel: React.FC<Props> = ({ disabled }) => {
  const startTraining = useRLStore((s) => s.startTraining);
  const totalEpisodes = useRLStore((s) => s.totalEpisodes);

  const [algorithm, setAlgorithm] = React.useState<'dqn' | 'ppo'>('dqn');
  const [episodes, setEpisodes] = React.useState(500);
  const [batchSize, setBatchSize] = React.useState(128);
  const [learningRate, setLearningRate] = React.useState(0.001);
  const [resumeEnabled, setResumeEnabled] = React.useState(false);
  const [starting, setStarting] = React.useState(false);

  const handleStart = async () => {
    setStarting(true);
    try {
      await startTraining({
        algorithm,
        episodes,
        batchSize,
        learningRate,
        resumeFrom: resumeEnabled ? 'latest' : undefined,
      });
    } finally {
      setStarting(false);
    }
  };

  const inputCls =
    'w-full rounded-lg bg-slate-800/60 border border-slate-600 px-3 py-2 text-sm text-gray-200 ' +
    'focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 disabled:opacity-50';

  return (
    <Card title="模型参数配置" variant="bordered" padding="md">
      <div className="space-y-3">
        {/* 算法 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">算法</label>
          <select
            className={inputCls}
            value={algorithm}
            disabled={disabled}
            onChange={(e) => setAlgorithm(e.target.value as 'dqn' | 'ppo')}
          >
            <option value="dqn">DQN（含 Double / Dueling）</option>
            <option value="ppo" disabled>
              PPO（Phase B 实现）
            </option>
          </select>
        </div>

        {/* 迭代次数 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">
            训练轮数 (Episodes){totalEpisodes > 0 && <span className="ml-1 text-cyan-400">当前目标: {totalEpisodes}</span>}
          </label>
          <input
            type="number"
            min={10}
            max={10000}
            step={10}
            className={inputCls}
            value={episodes}
            disabled={disabled}
            onChange={(e) => setEpisodes(Number(e.target.value) || 100)}
          />
        </div>

        {/* 批次大小 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">批次大小 (Batch Size)</label>
          <input
            type="number"
            min={16}
            max={1024}
            step={16}
            className={inputCls}
            value={batchSize}
            disabled={disabled}
            onChange={(e) => setBatchSize(Number(e.target.value) || 64)}
          />
        </div>

        {/* 学习率 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">学习率 (Learning Rate)</label>
          <input
            type="number"
            min={0.00001}
            max={0.1}
            step={0.0001}
            className={inputCls}
            value={learningRate}
            disabled={disabled}
            onChange={(e) => setLearningRate(Number(e.target.value) || 0.001)}
          />
        </div>

        {/* 断点续训 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/50"
            checked={resumeEnabled}
            disabled={disabled}
            onChange={(e) => setResumeEnabled(e.target.checked)}
          />
          <span className="text-xs text-gray-300">
            从断点续训
            <span className="block text-gray-500 text-[11px]">恢复 dqn_latest 中的权重/优化器/经验池</span>
          </span>
        </label>

        {/* 开始按钮 */}
        <Button
          variant="primary"
          glow
          className="w-full"
          disabled={disabled}
          isLoading={starting}
          onClick={() => void handleStart()}
        >
          ▶ 开始训练
        </Button>

        {/* 高级参数提示 */}
        <div className="rounded-lg bg-slate-800/40 border border-slate-700 p-3 text-[11px] text-gray-400 leading-relaxed">
          <p className="text-gray-300 font-medium mb-1">高级参数（奖励函数、折扣因子、交易成本等）</p>
          <p>
            已在「设置 → RL Training」中统一管理，修改后自动生效。
            <a href="/settings" className="text-cyan-400 hover:underline ml-1">
              前往设置 →
            </a>
          </p>
        </div>
      </div>
    </Card>
  );
};

export default TrainingConfigPanel;
