import apiClient from './index';
import { toCamelCase } from './utils';
import type {
  TrainRequest,
  TrainResponse,
  TaskStatusResponse,
  TrainingProgressResponse,
  ModelListResponse,
  EvaluateRequest,
  EvaluateTaskResponse,
  EvaluateProgressResponse,
  DailyReplayResponse,
} from '../types/rl';

/**
 * RL 训练 API 封装
 * 后端端点: /api/v1/rl/*
 */
export const rlApi = {
  /** 启动训练任务（支持 resumeFrom 断点续训） */
  startTraining: async (request: TrainRequest): Promise<TrainResponse> => {
    const payload: Record<string, unknown> = {};
    if (request.algorithm) payload.algorithm = request.algorithm;
    if (request.episodes != null) payload.episodes = request.episodes;
    if (request.batchSize != null) payload.batch_size = request.batchSize;
    if (request.learningRate != null) payload.learning_rate = request.learningRate;
    if (request.resumeFrom) payload.resume_from = request.resumeFrom;
    const response = await apiClient.post('/api/v1/rl/train', payload);
    return toCamelCase(response.data);
  },

  /** 查询任务状态 */
  getStatus: async (taskId: string): Promise<TaskStatusResponse> => {
    const response = await apiClient.get(`/api/v1/rl/train/${taskId}/status`);
    return toCamelCase(response.data);
  },

  /** 轮询训练进度（含完整指标历史） */
  getProgress: async (taskId: string): Promise<TrainingProgressResponse> => {
    const response = await apiClient.get(`/api/v1/rl/train/${taskId}/progress`);
    return toCamelCase(response.data);
  },

  /** 暂停训练（当前 episode 结束后生效） */
  pauseTraining: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/train/${taskId}/pause`);
    return response.data;
  },

  /** 恢复已暂停的训练 */
  resumeTraining: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/train/${taskId}/resume`);
    return response.data;
  },

  /** 停止训练（保存断点，可通过 resumeFrom 续训） */
  stopTraining: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/train/${taskId}/stop`);
    return response.data;
  },

  /** 获取已训练模型列表 */
  getModels: async (): Promise<ModelListResponse> => {
    const response = await apiClient.get('/api/v1/rl/models');
    return toCamelCase(response.data);
  },

  /** 启动异步评估任务（返回 taskId；maxDays 抽样评估天数，0/不传=全量默认抽样100） */
  startEvaluate: async (request: EvaluateRequest): Promise<EvaluateTaskResponse> => {
    const payload: Record<string, unknown> = { model_id: request.modelId };
    if (request.stockCodes) payload.stock_codes = request.stockCodes;
    if (request.maxDays != null) payload.max_days = request.maxDays;
    const response = await apiClient.post('/api/v1/rl/evaluate', payload);
    return toCamelCase(response.data);
  },

  /** 轮询评估任务进度（任务完成时附带完整评估结果） */
  getEvaluateProgress: async (taskId: string): Promise<EvaluateProgressResponse> => {
    const response = await apiClient.get(`/api/v1/rl/evaluate/${taskId}/progress`);
    return toCamelCase(response.data);
  },

  /** 暂停评估任务（当前交易日回放完成后生效） */
  pauseEvaluate: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/evaluate/${taskId}/pause`);
    return response.data;
  },

  /** 恢复已暂停的评估任务 */
  resumeEvaluate: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/evaluate/${taskId}/resume`);
    return response.data;
  },

  /** 终止评估任务（在下一个交易日回放前中断） */
  stopEvaluate: async (taskId: string): Promise<{ message: string }> => {
    const response = await apiClient.post(`/api/v1/rl/evaluate/${taskId}/stop`);
    return response.data;
  },

  /** 获取单日逐笔决策回放数据 */
  getDailyReplay: async (
    modelId: string,
    stockCode: string,
    date: string
  ): Promise<DailyReplayResponse> => {
    const response = await apiClient.get(
      `/api/v1/rl/evaluate/${modelId}/daily/${stockCode}/${date}`
    );
    return toCamelCase(response.data);
  },

  /** 删除模型 */
  deleteModel: async (modelId: string): Promise<{ message: string }> => {
    const response = await apiClient.delete(`/api/v1/rl/models/${modelId}`);
    return response.data;
  },
};

export default rlApi;
