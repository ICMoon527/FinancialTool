import React from 'react';
import { useRLStore } from '../stores/rlStore';
import TrainingConfigPanel from '../components/rl/TrainingConfigPanel';
import TrainingControlBar from '../components/rl/TrainingControlBar';
import TrainingCharts from '../components/rl/TrainingCharts';
import ModelListPanel from '../components/rl/ModelListPanel';
import EvaluationPanel from '../components/rl/EvaluationPanel';
import DailyReplayPanel from '../components/rl/DailyReplayPanel';

/**
 * RL 训练页面
 * 布局：左侧（参数配置 + 模型列表）| 右侧（控制栏 + 监控图表 / 评估 / 回放 Tab）
 * 响应式：<1024px 单列堆叠
 */

type ResultTab = 'monitor' | 'evaluation' | 'replay';

const TABS: Array<{ key: ResultTab; label: string }> = [
  { key: 'monitor', label: '训练监控' },
  { key: 'evaluation', label: '评估结果' },
  { key: 'replay', label: '单日回放' },
];

const RLTrainingPage: React.FC = () => {
  const taskStatus = useRLStore((s) => s.taskStatus);
  const error = useRLStore((s) => s.error);
  const notice = useRLStore((s) => s.notice);
  const metrics = useRLStore((s) => s.metrics);
  const evaluating = useRLStore((s) => s.evaluating);
  const setError = useRLStore((s) => s.setError);
  const setNotice = useRLStore((s) => s.setNotice);

  const [activeTab, setActiveTab] = React.useState<ResultTab>('monitor');

  // 评估启动时自动切换到「评估结果」Tab，让用户看到进度条
  React.useEffect(() => {
    if (evaluating) {
      setActiveTab('evaluation');
    }
  }, [evaluating]);

  // 错误/通知 5 秒自动消失
  React.useEffect(() => {
    if (error) {
      const t = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(t);
    }
  }, [error, setError]);

  React.useEffect(() => {
    if (notice) {
      const t = setTimeout(() => setNotice(null), 5000);
      return () => clearTimeout(t);
    }
  }, [notice, setNotice]);

  const running = taskStatus === 'running' || taskStatus === 'pending' || taskStatus === 'stopping';

  return (
    <div className="min-h-screen p-4 md:p-6 space-y-4">
      {/* 页头 */}
      <header>
        <h1 className="text-xl font-semibold text-gray-100 flex items-center gap-2">
          <span aria-hidden>🤖</span> RL 强化学习训练
        </h1>
        <p className="text-xs text-gray-500 mt-1">
          分时做T策略 · DQN（Double / Dueling）· 基于 CPU/GPU 的离线训练与评估
        </p>
      </header>

      {/* 通知条 */}
      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2.5 text-sm text-red-300 flex items-center justify-between">
          <span>{error.message}</span>
          <button type="button" onClick={() => setError(null)} className="text-red-400 hover:text-red-200 ml-3">
            ✕
          </button>
        </div>
      )}
      {notice && (
        <div className="rounded-lg border border-green-500/40 bg-green-500/10 px-4 py-2.5 text-sm text-green-300 flex items-center justify-between">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice(null)} className="text-green-400 hover:text-green-200 ml-3">
            ✕
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
        {/* 左侧栏 */}
        <div className="space-y-4 order-2 lg:order-1">
          <TrainingConfigPanel disabled={running} />
          <ModelListPanel />
        </div>

        {/* 右侧主区 */}
        <div className="space-y-4 order-1 lg:order-2 min-w-0">
          <TrainingControlBar />

          {/* Tab 切换 */}
          <div className="flex gap-1 rounded-xl border border-white/10 bg-[#0d0d14] p-1 w-fit">
            {TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => setActiveTab(t.key)}
                className={`px-4 py-1.5 rounded-lg text-sm transition-colors ${
                  activeTab === t.key
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab 内容 */}
          {activeTab === 'monitor' && <TrainingCharts metrics={metrics} running={running} />}
          {activeTab === 'evaluation' && <EvaluationPanel />}
          {activeTab === 'replay' && <DailyReplayPanel />}
        </div>
      </div>
    </div>
  );
};

export default RLTrainingPage;
