import React from 'react';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';
import type { RLTaskStatus } from '../../types/rl';

/**
 * 训练控制栏：开始/暂停/恢复/停止/重置 + 状态徽章 + 进度条
 */

const STATUS_META: Record<string, { label: string; cls: string }> = {
  pending: { label: '排队中', cls: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40' },
  running: { label: '训练中', cls: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' },
  completed: { label: '已完成', cls: 'bg-green-500/20 text-green-300 border-green-500/40' },
  failed: { label: '失败', cls: 'bg-red-500/20 text-red-300 border-red-500/40' },
  stopped: { label: '已停止', cls: 'bg-slate-500/20 text-slate-300 border-slate-500/40' },
  stopping: { label: '停止中', cls: 'bg-orange-500/20 text-orange-300 border-orange-500/40' },
};

const isBusy = (status: RLTaskStatus | null) =>
  status === 'running' || status === 'pending' || status === 'stopping';

export const TrainingControlBar: React.FC = () => {
  const taskStatus = useRLStore((s) => s.taskStatus);
  const taskMessage = useRLStore((s) => s.taskMessage);
  const paused = useRLStore((s) => s.paused);
  const currentEpisode = useRLStore((s) => s.currentEpisode);
  const totalEpisodes = useRLStore((s) => s.totalEpisodes);
  const progressPercent = useRLStore((s) => s.progressPercent);
  const taskId = useRLStore((s) => s.taskId);
  const pauseTraining = useRLStore((s) => s.pauseTraining);
  const resumeTraining = useRLStore((s) => s.resumeTraining);
  const stopTraining = useRLStore((s) => s.stopTraining);
  const reset = useRLStore((s) => s.reset);

  const busy = isBusy(taskStatus);
  const meta = taskStatus ? STATUS_META[taskStatus] : null;

  // 当前 episode 奖励
  const rewards = useRLStore((s) => s.metrics.episodeRewards);
  const latest = rewards && rewards.length > 0 ? rewards[rewards.length - 1] : null;

  return (
    <div className="rounded-xl border border-white/10 bg-[#0d0d14] p-4 space-y-3">
      {/* 第一行：按钮 + 状态 */}
      <div className="flex flex-wrap items-center gap-2">
        {/* 暂停/恢复 */}
        {busy && taskStatus === 'running' && (
          paused ? (
            <Button variant="primary" size="sm" onClick={() => void resumeTraining()}>
              ▶ 恢复
            </Button>
          ) : (
            <Button variant="secondary" size="sm" onClick={() => void pauseTraining()}>
              ⏸ 暂停
            </Button>
          )
        )}

        {/* 停止 */}
        {busy && (
          <Button variant="danger" size="sm" onClick={() => void stopTraining()}>
            ⏹ 停止
          </Button>
        )}

        {/* 重置 */}
        {!busy && taskId && (
          <Button variant="outline" size="sm" onClick={reset}>
            ↺ 重置
          </Button>
        )}

        <div className="flex-1" />

        {/* 状态徽章 */}
        {meta && (
          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${meta.cls}`}>
            {meta.label}
            {paused && taskStatus === 'running' ? ' (暂停)' : ''}
          </span>
        )}

        {/* Episode 进度数字 */}
        {totalEpisodes > 0 && (
          <span className="text-sm text-gray-300 tabular-nums">
            E{currentEpisode}/{totalEpisodes}
          </span>
        )}

        {/* 最新奖励 */}
        {latest != null && (
          <span className="text-sm text-cyan-300 tabular-nums">
            reward: {latest >= 0 ? '+' : ''}
            {latest.toFixed(2)}
          </span>
        )}
      </div>

      {/* 第二行：进度条 */}
      {totalEpisodes > 0 && (
        <div>
          <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300 transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }}
            />
          </div>
          {taskMessage && <p className="mt-1.5 text-xs text-gray-500">{taskMessage}</p>}
        </div>
      )}
    </div>
  );
};

export default TrainingControlBar;
