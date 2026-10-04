import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';
import { systemConfigApi } from '../../api/systemConfig';
import { rlApi } from '../../api/rl';
import { formatModelId, isPriorModel, isCnnModel } from '../../utils/format';

/**
 * 模型参数配置面板
 *
 * 核心训练参数（算法/网络结构/轮数/批次/学习率/先验开关）在挂载时从系统配置接口读取，
 * 与「设置 → RL Training」及 .env 共用同一数据源，随 POST /train 下发；
 * 其余高级参数（奖励函数权重、折扣因子、交易成本等）在设置页「RL Training」统一管理，
 * 此处仅展示当前值并跳转设置页修改。
 */

interface Props {
  disabled: boolean; // 训练运行中禁用
}

export const TrainingConfigPanel: React.FC<Props> = ({ disabled }) => {
  const startTraining = useRLStore((s) => s.startTraining);
  const totalEpisodes = useRLStore((s) => s.totalEpisodes);
  const models = useRLStore((s) => s.models);

  // 核心参数（算法/网络结构/轮数/批次/学习率/先验开关）从系统配置接口读取，与「设置 → RL Training」
  // 及 .env 共用同一数据源；以下常量仅在接口不可用或字段缺失时作为兜底
  const [algorithm, setAlgorithm] = React.useState<'dqn' | 'ppo'>('dqn');
  const [episodes, setEpisodes] = React.useState(300);
  const [batchSize, setBatchSize] = React.useState(128);
  const [learningRate, setLearningRate] = React.useState(0.0003);
  const [resumeEnabled, setResumeEnabled] = React.useState(false);
  const [resumeFromModel, setResumeFromModel] = React.useState('latest');
  const [useSignalScores, setUseSignalScores] = React.useState(true);
  const [useCnnEncoder, setUseCnnEncoder] = React.useState(false);
  const [maxSamples, setMaxSamples] = React.useState(0); // 采样样本数上限：0=全量（不限制）
  // 数据集规模（滑块上限）：从索引缓存读取，totalSamples=0 表示尚未建立索引
  const [totalSamples, setTotalSamples] = React.useState(0);
  const [totalStocks, setTotalStocks] = React.useState(0);
  const [datasetIndexed, setDatasetIndexed] = React.useState(true);
  const [showSampleTip, setShowSampleTip] = React.useState(false); // 悬停显示样本口径提示
  const [starting, setStarting] = React.useState(false);

  // 挂载时拉取 RL 配置，替换本地兜底默认，避免训练页与设置页出现两套值
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const resp = await systemConfigApi.getConfig(false);
        if (cancelled) return;
        const configMap = new Map(resp.items.map((item) => [item.key, item.value]));

        const readString = (key: string): string | undefined => {
          const raw = configMap.get(key);
          return raw === undefined || raw.trim() === '' ? undefined : raw.trim();
        };
        const readNumber = (key: string, fallback: number): number => {
          const raw = readString(key);
          if (raw === undefined) return fallback;
          const parsed = Number(raw);
          return Number.isFinite(parsed) ? parsed : fallback;
        };
        const readBoolean = (key: string, fallback: boolean): boolean => {
          const raw = readString(key)?.toLowerCase();
          if (raw === undefined) return fallback;
          return raw === 'true' || raw === '1' || raw === 'yes';
        };

        const algo = readString('RL_DEFAULT_ALGORITHM');
        if (algo === 'dqn' || algo === 'ppo') setAlgorithm(algo);
        setEpisodes(readNumber('RL_TRAINING_EPISODES', 300));
        setBatchSize(readNumber('RL_BATCH_SIZE', 128));
        setLearningRate(readNumber('RL_LEARNING_RATE', 0.0003));
        setUseSignalScores(readBoolean('RL_USE_SIGNAL_SCORES', true));
        setUseCnnEncoder(readBoolean('RL_USE_CNN_ENCODER', false));
        setMaxSamples(readNumber('RL_MAX_SAMPLES', 0));
      } catch {
        // 配置接口不可用时保留本地兜底默认，不阻塞训练流程
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 挂载时拉取数据集规模，作为滑块上限（读取元数据索引缓存，毫秒级）
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const info = await rlApi.getDatasetInfo();
        if (cancelled) return;
        setTotalSamples(info.totalSamples);
        setTotalStocks(info.totalStocks);
        setDatasetIndexed(info.cacheExists);
      } catch {
        // 接口不可用时保留兜底上限，不阻塞训练流程
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 滑块上限 = 数据集真实样本总数（未建索引时用兜底值）；步长随规模自适应
  const sliderMax = totalSamples > 0 ? totalSamples : 50000;
  const sliderStep = Math.max(1, Math.round(sliderMax / 200));
  // 超过真实总数即等价「全量」，统一归零，保证显示值与下发票值一致
  const effectiveMaxSamples = maxSamples > sliderMax ? 0 : maxSamples;

  const handleStart = async () => {
    setStarting(true);
    try {
      await startTraining({
        algorithm,
        episodes,
        batchSize,
        learningRate,
        resumeFrom: resumeEnabled ? resumeFromModel : undefined,
        useSignalScores,
        useCnnEncoder,
        maxSamples: effectiveMaxSamples,
      });
    } finally {
      setStarting(false);
    }
  };

  // 依据所选续训模型的标记自动同步先验/网络结构，避免权重维度不匹配
  const handleResumeModelChange = (id: string) => {
    setResumeFromModel(id);
    if (id !== 'latest') {
      setUseSignalScores(isPriorModel(id));
      setUseCnnEncoder(isCnnModel(id));
    }
  };

  const resumePrior = resumeFromModel !== 'latest' && isPriorModel(resumeFromModel);
  const resumeCnn = resumeFromModel !== 'latest' && isCnnModel(resumeFromModel);

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

        {/* 网络结构 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">网络结构</label>
          <select
            className={inputCls}
            value={useCnnEncoder ? 'cnn' : 'mlp'}
            disabled={disabled}
            onChange={(e) => setUseCnnEncoder(e.target.value === 'cnn')}
          >
            <option value="mlp">纯 MLP（Dueling DQN）</option>
            <option value="cnn">CNN 形态编码器 + MLP</option>
          </select>
          <p className="text-[11px] text-gray-500 mt-1">
            两种结构权重不兼容，切换后需重新训练，不可续训旧结构模型
          </p>
        </div>

        {/* 迭代次数 */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">
            训练轮数 (Episodes)
            {resumeEnabled && (
              <span className="ml-1 text-amber-400">续训时在现有进度上再训练 {episodes} 轮</span>
            )}
            {totalEpisodes > 0 && !resumeEnabled && (
              <span className="ml-1 text-cyan-400">当前目标: {totalEpisodes}</span>
            )}
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

        {/* 采样样本数上限（滑块）：完整说明改为悬停显示，避免长数字换行 */}
        <div>
          <div className="flex items-center justify-between gap-2 mb-1">
            <label className="text-xs text-gray-400 whitespace-nowrap">
              采样样本数上限
              <span
                className="ml-1 inline-block cursor-help rounded px-1 text-gray-500 hover:text-cyan-400"
                onMouseEnter={() => setShowSampleTip(true)}
                onMouseLeave={() => setShowSampleTip(false)}
              >
                ⓘ
              </span>
            </label>
            <span className="shrink-0 whitespace-nowrap rounded-md border border-slate-700 bg-slate-800/80 px-2 py-0.5 text-[11px] font-mono tabular-nums text-cyan-300">
              {effectiveMaxSamples === 0 ? '全量' : effectiveMaxSamples.toLocaleString()}
            </span>
          </div>

          <div className="relative">
            <input
              type="range"
              min={0}
              max={sliderMax}
              step={sliderStep}
              className="w-full accent-cyan-500 cursor-pointer disabled:opacity-50"
              value={effectiveMaxSamples}
              disabled={disabled}
              onChange={(e) => setMaxSamples(Number(e.target.value))}
            />
            <div className="mt-0.5 flex justify-between font-mono text-[10px] text-gray-600">
              <span>0</span>
              <span>{sliderMax.toLocaleString()}</span>
            </div>

            {/* 悬停提示：数据集规模与口径说明 */}
            <div className={`pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-max max-w-[280px] -translate-x-1/2 rounded-lg border border-slate-600 bg-slate-900/95 px-3 py-2 text-[11px] leading-relaxed text-gray-300 shadow-xl ${
              showSampleTip ? 'block' : 'hidden'
            }`}>
              {datasetIndexed && totalSamples > 0 ? (
                <>
                  <div>
                    数据集共
                    <span className="mx-1 font-mono text-cyan-300">{totalSamples.toLocaleString()}</span>
                    个样本
                  </div>
                  <div className="text-gray-500">{totalStocks.toLocaleString()} 只股票 × 交易日</div>
                  <div className="mt-1 text-gray-400">
                    0 = 全量；设为 N 时最多用 N 个（超出随机下采样）
                  </div>
                </>
              ) : (
                <div>尚未建立数据索引，无法获取样本总数（可先运行一次训练生成）</div>
              )}
            </div>
          </div>
        </div>

        {/* 断点续训 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/50"
            checked={resumeEnabled}
            disabled={disabled}
            onChange={(e) => {
              setResumeEnabled(e.target.checked);
              if (!e.target.checked) setResumeFromModel('latest');
            }}
          />
          <span className="text-xs text-gray-300">
            从断点续训
            <span className="block text-gray-500 text-[11px]">
              从所选模型的断点恢复权重/优化器/经验池
            </span>
          </span>
        </label>

        {resumeEnabled && (
          <div>
            <label className="block text-xs text-gray-400 mb-1">续训模型</label>
            <select
              className={inputCls}
              value={resumeFromModel}
              disabled={disabled}
              onChange={(e) => handleResumeModelChange(e.target.value)}
            >
              <option value="latest">latest（最近断点，自动匹配当前先验设置）</option>
              {models.map((m) => (
                <option key={m.modelId} value={m.modelId}>
                  {formatModelId(m.modelId)}
                </option>
              ))}
            </select>
            {(resumePrior || resumeCnn) && (
              <p className="text-[11px] text-cyan-400 mt-1">
                已自动匹配该模型结构：
                {[resumePrior ? '规则先验买卖点' : '', resumeCnn ? 'CNN 形态编码器' : '']
                  .filter(Boolean)
                  .join(' + ')}
              </p>
            )}
          </div>
        )}

        {/* 启用先验买卖点 */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/50"
            checked={useSignalScores}
            disabled={disabled}
            onChange={(e) => setUseSignalScores(e.target.checked)}
          />
          <span className="text-xs text-gray-300">
            启用规则先验买卖点
            <span className="block text-gray-500 text-[11px]">
              将规则买卖点评分接入状态特征（state_dim 8→10），需重新训练，不可续训旧维度模型
            </span>
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
            <Link to="/settings" className="text-cyan-400 hover:underline ml-1">
              前往设置 →
            </Link>
          </p>
        </div>
      </div>
    </Card>
  );
};

export default TrainingConfigPanel;
