import React from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { useRLStore } from '../../stores/rlStore';
import { rlApi } from '../../api/rl';

/**
 * 模型列表面板：展示已训练模型，支持评估 / 断点续训选中 / 删除
 */

export const ModelListPanel: React.FC = () => {
  const models = useRLStore((s) => s.models);
  const modelsLoading = useRLStore((s) => s.modelsLoading);
  const selectedModelId = useRLStore((s) => s.selectedModelId);
  const busy = useRLStore((s) => ['running', 'pending', 'stopping'].includes(s.taskStatus ?? ''));
  const evaluating = useRLStore((s) => s.evaluating);
  const evalDone = useRLStore((s) => s.evalDone);
  const evalTotal = useRLStore((s) => s.evalTotal);
  const evalProgress = useRLStore((s) => s.evalProgress);
  const evalMessage = useRLStore((s) => s.evalMessage);
  const evaluateModel = useRLStore((s) => s.evaluateModel);
  const selectModel = useRLStore((s) => s.selectModel);
  const fetchModels = useRLStore((s) => s.fetchModels);
  const setError = useRLStore((s) => s.setError);
  // 多模型对比评估（同一批数据/同基准）
  const compareEvaluate = useRLStore((s) => s.compareEvaluate);
  const compareEvaluating = useRLStore((s) => s.compareEvaluating);
  const compareDone = useRLStore((s) => s.compareDone);
  const compareTotal = useRLStore((s) => s.compareTotal);
  const compareProgress = useRLStore((s) => s.compareProgress);
  const compareMessage = useRLStore((s) => s.compareMessage);
  const compareCurrentModelIdx = useRLStore((s) => s.compareCurrentModelIdx);
  const compareModelDone = useRLStore((s) => s.compareModelDone);
  const compareModelTotal = useRLStore((s) => s.compareModelTotal);
  const [deleting, setDeleting] = React.useState<string | null>(null);
  // 待确认删除的模型 id：非空时弹出应用内确认框。
  // 用自绘弹窗替代 window.confirm，避免内嵌预览环境对原生对话框的处理缺陷
  // （原生 confirm 会被跳过直接返回、并触发预览运行时崩溃）
  const [pendingDelete, setPendingDelete] = React.useState<string | null>(null);
  // 全量评估开关：默认抽样 100 个交易日（约 3 分钟），勾选后全量（数万交易日，耗时极长）
  const [fullEval, setFullEval] = React.useState(false);
  // 勾选用于多模型对比评估的模型 ID 集合
  const [selectedForCompare, setSelectedForCompare] = React.useState<string[]>([]);

  React.useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  const handleDelete = (modelId: string) => {
    setPendingDelete(modelId);
  };

  const confirmDelete = async () => {
    const modelId = pendingDelete;
    if (!modelId) return;
    setDeleting(modelId);
    try {
      await rlApi.deleteModel(modelId);
      if (selectedModelId === modelId) selectModel(null);
      await fetchModels();
      setPendingDelete(null);
    } catch (err) {
      setError(`删除失败: ${(err as Error).message}`);
    } finally {
      setDeleting(null);
    }
  };

  return (
    <>
      <Card title="模型列表" variant="bordered" padding="md">
      {/* 评估模式开关 */}
      {models.length > 0 && (
        <label
          className="flex items-center gap-1.5 text-[11px] text-gray-400 mb-2 cursor-pointer select-none"
          title="默认随机抽样 100 个交易日快速评估（约 3 分钟）；勾选后评估全部验证集（数万交易日，耗时极长）"
        >
          <input
            type="checkbox"
            checked={fullEval}
            onChange={(e) => setFullEval(e.target.checked)}
            disabled={evaluating}
            className="accent-cyan-500"
          />
          全量评估（默认抽样 100 天）
        </label>
      )}
      {modelsLoading && models.length === 0 ? (
        <p className="text-xs text-gray-500 py-4 text-center">加载中...</p>
      ) : models.length === 0 ? (
        <p className="text-xs text-gray-500 py-4 text-center">
          暂无模型
          <span className="block mt-1 text-gray-600">完成一次训练后模型将出现在这里</span>
        </p>
      ) : (
        <ul className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {models.map((m) => (
            <li
              key={m.modelId}
              className={`rounded-lg border p-2.5 text-xs transition-colors ${
                selectedModelId === m.modelId
                  ? 'border-cyan-500/60 bg-cyan-500/10'
                  : 'border-slate-700 bg-slate-800/40 hover:border-slate-500'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <input
                    type="checkbox"
                    checked={selectedForCompare.includes(m.modelId)}
                    onChange={(e) => {
                      e.stopPropagation();
                      setSelectedForCompare((prev) =>
                        e.target.checked
                          ? [...prev, m.modelId]
                          : prev.filter((id) => id !== m.modelId)
                      );
                    }}
                    disabled={evaluating || compareEvaluating || busy}
                    title="勾选用于多模型对比评估（同一批数据/同基准）"
                    className="rounded w-3.5 h-3.5 shrink-0 accent-cyan-500"
                  />
                  <button
                    type="button"
                    className="font-mono text-gray-200 truncate hover:text-cyan-300 text-left"
                    title={m.modelId}
                    onClick={() => selectModel(selectedModelId === m.modelId ? null : m.modelId)}
                  >
                    {m.modelId}
                  </button>
                </div>
                <span className="shrink-0 px-1.5 py-0.5 rounded bg-slate-700 text-[10px] uppercase text-gray-300">
                  {m.algorithm}
                </span>
              </div>
              <p className="text-gray-500 mt-0.5">{m.createdAt.slice(0, 19).replace('T', ' ')}</p>

              {/* 操作按钮 */}
              <div className="flex gap-1.5 mt-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={busy || m.algorithm === 'ppo'}
                  isLoading={evaluating && selectedModelId === m.modelId}
                  onClick={() => void evaluateModel(m.modelId, fullEval ? 0 : 100)}
                  className="!px-2 !py-1 text-[11px]"
                >
                  评估
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  isLoading={deleting === m.modelId}
                  onClick={() => void handleDelete(m.modelId)}
                  className="!px-2 !py-1 text-[11px] !text-red-400 hover:!bg-red-500/10"
                >
                  删除
                </Button>
              </div>

              {/* 评估中：按钮下方迷你进度条与当前样本 */}
              {evaluating && selectedModelId === m.modelId && (
                <div className="mt-1.5">
                  <div className="h-1 rounded-full bg-slate-700/60 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
                      style={{ width: `${Math.min(Math.max(evalProgress, 0), 100)}%` }}
                    />
                  </div>
                  <p
                    className="font-mono text-[10px] text-gray-400 mt-1 truncate"
                    title={evalMessage}
                  >
                    {evalDone}/{evalTotal} · {evalProgress.toFixed(1)}% · {evalMessage || '准备中...'}
                  </p>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {/* 多模型对比评估工具栏（同一批数据/同基准） */}
      {models.length > 0 && (
        <div className="mt-3 pt-3 border-t border-white/10">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-gray-400">
              已选 <span className="text-cyan-400 font-mono">{selectedForCompare.length}</span> 个模型
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={
                selectedForCompare.length === 0 || busy || evaluating || compareEvaluating
              }
              isLoading={compareEvaluating}
              onClick={() => void compareEvaluate(selectedForCompare, fullEval ? 0 : 100)}
              className="!px-2.5 !py-1 text-[11px]"
            >
              对比评估（同基准）
            </Button>
          </div>
          {compareEvaluating && (
            <div className="mt-2">
              <div className="h-1 rounded-full bg-slate-700/60 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500"
                  style={{ width: `${Math.min(Math.max(compareProgress, 0), 100)}%` }}
                />
              </div>
              <p
                className="font-mono text-[10px] text-gray-400 mt-1 truncate"
                title={compareMessage}
              >
                {compareDone}/{compareTotal} · {compareProgress.toFixed(1)}% ·{' '}
                {compareCurrentModelIdx != null
                  ? `模型${compareCurrentModelIdx + 1} ${compareModelDone}/${compareModelTotal} · `
                  : ''}
                {compareMessage || '准备中...'}
              </p>
            </div>
          )}
        </div>
      )}
      </Card>

      {/* 删除确认弹窗（应用内自绘，替代 window.confirm） */}
      {pendingDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={() => {
            if (deleting === null) setPendingDelete(null);
          }}
        >
          <div
            className="bg-slate-800 border border-slate-600 rounded-xl p-5 w-80 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-white font-semibold mb-2">确认删除模型</h3>
            <p className="text-sm text-gray-400 mb-4 break-all">
              确定要删除 <span className="text-red-400 font-mono">{pendingDelete}</span> 吗？
              <span className="block mt-1 text-gray-500">
                该操作会删除对应模型文件夹，且不可恢复。
              </span>
            </p>
            <div className="flex justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                disabled={deleting !== null}
                onClick={() => setPendingDelete(null)}
              >
                取消
              </Button>
              <Button
                variant="danger"
                size="sm"
                isLoading={deleting !== null}
                onClick={() => void confirmDelete()}
              >
                确认删除
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ModelListPanel;
