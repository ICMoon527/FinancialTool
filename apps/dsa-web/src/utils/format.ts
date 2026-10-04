export const formatDateTime = (value?: string): string => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

export const formatDate = (value?: string): string => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(date);
};

export const toDateInputValue = (date: Date): string => {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, '0');
  const day = `${date.getDate()}`.padStart(2, '0');
  return `${year}-${month}-${day}`;
};

/**
 * Returns the date N days ago as YYYY-MM-DD in Asia/Shanghai timezone.
 * Consistent with getTodayInShanghai() so both ends of the date range
 * are expressed in the same timezone as the backend.
 */
export const getRecentStartDate = (days: number): string => {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' }).format(date);
};

/**
 * Returns today's date as YYYY-MM-DD in Asia/Shanghai timezone.
 * Use this instead of browser-local date to stay consistent with the backend,
 * which stores and filters timestamps in server local time (Asia/Shanghai).
 */
export const getTodayInShanghai = (): string =>
  new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' }).format(new Date());

export const formatReportType = (value?: string): string => {
  if (!value) return '—';
  if (value === 'simple') return '普通';
  if (value === 'detailed') return '标准';
  return value;
};

/**
 * 解析 RL model_id
 *
 * 后端 model_id = 相对 rl/models 的 checkpoint 路径，各段用 "__" 连接：
 * - 嵌套布局（训练用 RL_MODEL_DIR 指向实验子目录）：`{实验名}__{checkpoint名}`
 * - 扁平布局（checkpoint 直接位于 rl/models 下）：仅 `{checkpoint名}`
 */
export const parseModelId = (
  modelId: string
): { experiment: string | null; checkpoint: string } => {
  const sepIndex = modelId.indexOf('__');
  if (sepIndex === -1) return { experiment: null, checkpoint: modelId };
  return {
    experiment: modelId.slice(0, sepIndex),
    checkpoint: modelId.slice(sepIndex + 2),
  };
};

/** 由 model_id 判断是否为「开启先验买卖点」的模型（checkpoint 名第 2 段为 prior） */
export const isPriorModel = (modelId: string): boolean =>
  parseModelId(modelId).checkpoint.split('_')[1] === 'prior';

/** 由 model_id 判断是否为「启用 CNN 形态编码器」的模型（checkpoint 名含 cnn 标记段） */
export const isCnnModel = (modelId: string): boolean =>
  parseModelId(modelId).checkpoint.split('_').includes('cnn');

/** model_id 的展示名：嵌套布局显示为「实验名 / checkpoint 名」，扁平布局仅显示 checkpoint 名 */
export const formatModelId = (modelId: string): string => {
  const { experiment, checkpoint } = parseModelId(modelId);
  return experiment ? `${experiment} / ${checkpoint}` : checkpoint;
};
