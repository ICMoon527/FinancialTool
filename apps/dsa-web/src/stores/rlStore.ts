import { create } from 'zustand';
import { rlApi } from '../api/rl';
import type {
  RLMetricsHistory,
  RLTaskStatus,
  ModelInfo,
  EvaluateResultResponse,
} from '../types/rl';

/**
 * RL 训练全局状态（zustand）
 * 包含轮询控制逻辑，组件只需调用 action，无需自行管理定时器
 */

export interface RLError {
  message: string;
  at: number;
}

interface RLState {
  // 任务状态
  taskId: string | null;
  taskStatus: RLTaskStatus | null;
  taskMessage: string;
  paused: boolean;

  // 进度
  currentEpisode: number;
  totalEpisodes: number;
  progressPercent: number;

  // 指标历史（增量更新）
  metrics: Partial<RLMetricsHistory>;

  // 模型列表
  models: ModelInfo[];
  modelsLoading: boolean;

  // 评估结果
  selectedModelId: string | null;
  evaluating: boolean;
  evaluateResult: EvaluateResultResponse | null;
  // 评估进度（轮询更新）
  evalTaskId: string | null;
  evalDone: number;
  evalTotal: number;
  evalProgress: number;
  evalMessage: string;
  evalPaused: boolean;

  // 错误/提示
  error: RLError | null;
  notice: string | null;

  // 内部
  _pollTimer: ReturnType<typeof setInterval> | null;
  _failCount: number;
  _evalPollTimer: ReturnType<typeof setInterval> | null;
  _evalFailCount: number;

  // actions
  startTraining: (params: {
    algorithm?: 'dqn' | 'ppo';
    episodes?: number;
    batchSize?: number;
    learningRate?: number;
    resumeFrom?: string;
  }) => Promise<void>;
  pauseTraining: () => Promise<void>;
  resumeTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  reset: () => void;
  startPolling: (taskId: string) => void;
  stopPolling: () => void;
  fetchModels: () => Promise<void>;
  evaluateModel: (modelId: string, maxDays?: number) => Promise<void>;
  pauseEvaluate: () => Promise<void>;
  resumeEvaluate: () => Promise<void>;
  stopEvaluate: () => Promise<void>;
  startEvalPolling: (taskId: string) => void;
  stopEvalPolling: () => void;
  selectModel: (modelId: string | null) => void;
  setError: (message: string | null) => void;
  setNotice: (message: string | null) => void;
}

export const useRLStore = create<RLState>((set, get) => ({
  taskId: null,
  taskStatus: null,
  taskMessage: '',
  paused: false,
  currentEpisode: 0,
  totalEpisodes: 0,
  progressPercent: 0,
  metrics: {},
  models: [],
  modelsLoading: false,
  selectedModelId: null,
  evaluating: false,
  evaluateResult: null,
  evalTaskId: null,
  evalDone: 0,
  evalTotal: 0,
  evalProgress: 0,
  evalMessage: '',
  evalPaused: false,
  error: null,
  notice: null,
  _pollTimer: null,
  _failCount: 0,
  _evalPollTimer: null,
  _evalFailCount: 0,

  startTraining: async (params) => {
    try {
      get().setError(null);
      const resp = await rlApi.startTraining(params);
      set({
        taskId: resp.taskId,
        taskStatus: 'pending',
        taskMessage: resp.message,
        paused: false,
        currentEpisode: 0,
        progressPercent: 0,
        metrics: {},
        evaluateResult: null,
      });
      get().startPolling(resp.taskId);
    } catch (err) {
      set({ error: { message: `启动训练失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  pauseTraining: async () => {
    const { taskId } = get();
    if (!taskId) return;
    try {
      await rlApi.pauseTraining(taskId);
      set({ paused: true, taskMessage: '暂停中（等待当前 episode 结束）...' });
    } catch (err) {
      set({ error: { message: `暂停失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  resumeTraining: async () => {
    const { taskId } = get();
    if (!taskId) return;
    try {
      await rlApi.resumeTraining(taskId);
      set({ paused: false, taskMessage: '训练中...' });
    } catch (err) {
      set({ error: { message: `恢复失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  stopTraining: async () => {
    const { taskId } = get();
    if (!taskId) return;
    try {
      await rlApi.stopTraining(taskId);
      set({ taskStatus: 'stopping', taskMessage: '正在停止（保存断点）...' });
    } catch (err) {
      set({ error: { message: `停止失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  reset: () => {
    get().stopPolling();
    get().stopEvalPolling();
    set({
      taskId: null,
      taskStatus: null,
      taskMessage: '',
      paused: false,
      currentEpisode: 0,
      totalEpisodes: 0,
      progressPercent: 0,
      metrics: {},
      evaluateResult: null,
      evalTaskId: null,
      evalDone: 0,
      evalTotal: 0,
      evalProgress: 0,
      evalMessage: '',
      evaluating: false,
      error: null,
      notice: null,
    });
  },

  startPolling: (taskId) => {
    get().stopPolling();
    const timer = setInterval(async () => {
      try {
        const progress = await rlApi.getProgress(taskId);
        set({
          taskStatus: progress.status,
          taskMessage: progress.message,
          paused: progress.paused,
          currentEpisode: progress.currentEpisode,
          totalEpisodes: progress.totalEpisodes,
          progressPercent: progress.progress,
          metrics: progress.metrics ?? {},
          _failCount: 0,
        });

        // 终态：停止轮询
        const st = get();
        if (
          progress.status === 'completed' ||
          progress.status === 'failed' ||
          progress.status === 'stopped'
        ) {
          st.stopPolling();
          if (progress.status === 'completed') {
            st.setNotice('训练完成');
            void st.fetchModels();
          } else if (progress.status === 'failed') {
            st.setError(`训练失败: ${progress.message}`);
          }
        }
      } catch {
        // 轮询失败重试计数，连续 3 次失败停止轮询
        const fail = get()._failCount + 1;
        set({ _failCount: fail });
        if (fail >= 3) {
          get().stopPolling();
          set({
            error: { message: '进度轮询连续失败，已停止轮询（任务可能已结束）', at: Date.now() },
          });
        }
      }
    }, 2000);
    set({ _pollTimer: timer });
  },

  stopPolling: () => {
    const timer = get()._pollTimer;
    if (timer) {
      clearInterval(timer);
      set({ _pollTimer: null });
    }
  },

  fetchModels: async () => {
    set({ modelsLoading: true });
    try {
      const resp = await rlApi.getModels();
      set({ models: resp.models, modelsLoading: false });
    } catch (err) {
      set({
        modelsLoading: false,
        error: { message: `获取模型列表失败: ${(err as Error).message}`, at: Date.now() },
      });
    }
  },

  evaluateModel: async (modelId, maxDays) => {
    set({
      evaluating: true,
      selectedModelId: modelId,
      evaluateResult: null,
      evalTaskId: null,
      evalDone: 0,
      evalTotal: 0,
      evalProgress: 0,
      evalMessage: '提交评估任务...',
      evalPaused: false,
    });
    try {
      const resp = await rlApi.startEvaluate({ modelId, maxDays });
      set({ evalTaskId: resp.taskId });
      get().startEvalPolling(resp.taskId);
    } catch (err) {
      set({
        evaluating: false,
        evalMessage: '',
        error: { message: `启动评估失败: ${(err as Error).message}`, at: Date.now() },
      });
    }
  },

  pauseEvaluate: async () => {
    const taskId = get().evalTaskId;
    if (!taskId) return;
    try {
      await rlApi.pauseEvaluate(taskId);
      set({ evalPaused: true });
    } catch (err) {
      set({ error: { message: `暂停评估失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  resumeEvaluate: async () => {
    const taskId = get().evalTaskId;
    if (!taskId) return;
    try {
      await rlApi.resumeEvaluate(taskId);
      set({ evalPaused: false });
    } catch (err) {
      set({ error: { message: `恢复评估失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  stopEvaluate: async () => {
    const taskId = get().evalTaskId;
    if (!taskId) return;
    try {
      await rlApi.stopEvaluate(taskId);
      set({ evalMessage: '终止指令已发送，等待当前交易日回放结束...' });
    } catch (err) {
      set({ error: { message: `终止评估失败: ${(err as Error).message}`, at: Date.now() } });
    }
  },

  startEvalPolling: (taskId) => {
    get().stopEvalPolling();
    const timer = setInterval(async () => {
      try {
        const progress = await rlApi.getEvaluateProgress(taskId);
        set({
          evalTaskId: taskId,
          evalDone: progress.done,
          evalTotal: progress.total,
          evalProgress: progress.progress,
          evalMessage: progress.message,
          evalPaused: progress.paused,
          _evalFailCount: 0,
        });

        // 终态：停止轮询并处理结果
        if (
          progress.status === 'completed' ||
          progress.status === 'failed' ||
          progress.status === 'stopped'
        ) {
          const st = get();
          st.stopEvalPolling();
          if (progress.status === 'completed' && progress.result) {
            set({
              evaluateResult: progress.result,
              evaluating: false,
              evalProgress: 100,
              evalMessage: '评估完成',
              evalPaused: false,
              notice: '评估完成',
            });
          } else if (progress.status === 'stopped') {
            set({
              evaluating: false,
              evalPaused: false,
              evalMessage: `评估已终止（完成 ${progress.done}/${progress.total}）`,
              notice: '评估已终止',
            });
          } else {
            set({
              evaluating: false,
              evalMessage: '',
              evalPaused: false,
              error: {
                message: `评估失败: ${progress.message || '未知错误'}`,
                at: Date.now(),
              },
            });
          }
        }
      } catch {
        // 轮询失败重试计数，连续 3 次失败停止轮询
        const fail = get()._evalFailCount + 1;
        set({ _evalFailCount: fail });
        if (fail >= 3) {
          get().stopEvalPolling();
          set({
            evaluating: false,
            evalMessage: '',
            error: {
              message: '评估进度轮询连续失败，已停止（任务可能已结束）',
              at: Date.now(),
            },
          });
        }
      }
    }, 1500);
    set({ _evalPollTimer: timer });
  },

  stopEvalPolling: () => {
    const timer = get()._evalPollTimer;
    if (timer) {
      clearInterval(timer);
      set({ _evalPollTimer: null });
    }
  },

  selectModel: (modelId) => set({ selectedModelId: modelId }),

  setError: (message) =>
    set({ error: message ? { message, at: Date.now() } : null }),

  setNotice: (message) =>
    set({ notice: message ? message : null }),
}));
