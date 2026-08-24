import type React from 'react';
import { useState, useEffect, useCallback, useRef, memo, useMemo } from 'react';
import { backtestApi } from '../api/backtest';
import { Card, Badge } from '../components/common';
import { BacktestChartsContainer } from '../components/charts';
import type {
  StrategyInfo,
  StrategyBacktestTaskStatusResponse,
  StrategyBacktestRunAsyncRequest,
  ExitStrategiesResponse,
} from '../types/backtest';

// 星星图标组件
const StarIcon = ({ isFavorited, onClick }: { isFavorited: boolean; onClick: (e: React.MouseEvent) => void }) => (
  <button
    type="button"
    onClick={onClick}
    className="flex-shrink-0 w-5 h-5 p-0.5 hover:bg-white/10 rounded transition-colors cursor-pointer"
  >
    {isFavorited ? (
      <svg className="w-full h-full text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
      </svg>
    ) : (
      <svg className="w-full h-full text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118L12 16.055l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.783-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
      </svg>
    )}
  </button>
);

// 策略项组件
const StrategyItem = memo(({
  strategy,
  isSelected,
  onToggle,
  isLoading,
  isFavorited,
  onToggleFavorite
}: {
  strategy: StrategyInfo;
  isSelected: boolean;
  onToggle: (strategyId: string, selected: boolean) => void;
  isLoading: boolean;
  isFavorited: boolean;
  onToggleFavorite: (strategyId: string, e: React.MouseEvent) => void;
}) => {
  const strategyTypeBadge = (() => {
    switch (strategy.type) {
      case 'NATURAL_LANGUAGE':
        return <Badge variant="info">NL</Badge>;
      case 'PYTHON':
        return <Badge variant="success">PY</Badge>;
      default:
        return <Badge variant="default">{strategy.type}</Badge>;
    }
  })();

  return (
    <label className="flex items-center gap-2 px-3 py-2.5 hover:bg-white/10 cursor-pointer transition-colors border-b border-white/5 last:border-0">
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => {
          e.stopPropagation();
          onToggle(strategy.id, e.target.checked);
        }}
        disabled={isLoading}
        className="rounded w-4 h-4"
      />
      {strategyTypeBadge}
      <div className="flex-1 min-w-0">
        <div className="text-sm text-white truncate">{strategy.name}</div>
        <div className="text-xs text-muted truncate">{strategy.description}</div>
      </div>
      <StarIcon
        isFavorited={isFavorited}
        onClick={(e) => onToggleFavorite(strategy.id, e)}
      />
    </label>
  );
});

StrategyItem.displayName = 'StrategyItem';

// ============ 格式化函数 ============

function formatNumber(value?: number | null): string {
  if (value == null) return '--';
  return value.toFixed(4);
}

function formatPercent(value?: number | null): string {
  if (value == null) return '--';
  return `${(value * 100).toFixed(2)}%`;
}

// ============ 绩效指标卡片 ============

const MetricsCard: React.FC<{ metrics: Record<string, unknown>; title: string }> = ({ metrics, title }) => {
  return (
    <Card variant="gradient" padding="md" className="animate-fade-in">
      <div className="mb-3">
        <span className="label-uppercase">{title}</span>
      </div>
      {Object.entries(metrics).map(([key, value]) => (
        <div key={key} className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
          <span className="text-xs text-secondary">{key}</span>
          <span className="text-sm font-mono font-semibold text-white">
            {typeof value === 'number' 
              ? (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('ratio'))
                ? formatPercent(value)
                : formatNumber(value)
              : String(value)
            }
          </span>
        </div>
      ))}
    </Card>
  );
};

// ============ 主页面 ============

const BacktestPage: React.FC = () => {
  // 策略状态
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategyIds, setSelectedStrategyIds] = useState<string[]>([]);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);
  const [isStrategyListOpen, setIsStrategyListOpen] = useState(false);
  const strategyContainerRef = useRef<HTMLDivElement>(null);

  // 收藏策略状态
  const [favoriteStrategyIds, setFavoriteStrategyIds] = useState<string[]>(() => {
    const saved = localStorage.getItem('favorite_strategies');
    return saved ? JSON.parse(saved) : [];
  });

  // 日期状态
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // 最高持仓数状态
  const [maxPositions, setMaxPositions] = useState<number | ''>(3);

  // 退出策略状态
  const [exitStrategiesData, setExitStrategiesData] = useState<ExitStrategiesResponse | null>(null);
  const [selectedExitPreset, setSelectedExitPreset] = useState<string>('');
  const [editedExitParams, setEditedExitParams] = useState<Record<string, number>>({});
  const [isExitStrategyListOpen, setIsExitStrategyListOpen] = useState(false);
  const exitStrategyContainerRef = useRef<HTMLDivElement>(null);

  // 运行状态
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<StrategyBacktestTaskStatusResponse['task'] | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // 回测结果状态
  const [backtestImages, setBacktestImages] = useState<{
    [key: string]: string;
  }>({});
  const [latestBacktestData, setLatestBacktestData] = useState<any>(null);
  
  // QuantStats 报告状态
  const [quantstatsHtml, setQuantstatsHtml] = useState<string | null>(null);
  const [quantstatsMetrics, setQuantstatsMetrics] = useState<any>(null);

  // 历史回测选择状态
  const [historyItems, setHistoryItems] = useState<{ name: string; modifiedAt: number }[]>([]);
  const [selectedHistoryDir, setSelectedHistoryDir] = useState<string>(''); // 空串表示最新
  const [historyDropdownOpen, setHistoryDropdownOpen] = useState<boolean>(false); // 历史回测下拉是否展开

  // 轮询定时器引用
  const pollTimerRef = useRef<number | null>(null);
  const isMountedRef = useRef<boolean>(true);

  // 切换收藏状态
  const toggleFavorite = useCallback((strategyId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavoriteStrategyIds(prev => {
      const newFavorites = prev.includes(strategyId)
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId];
      localStorage.setItem('favorite_strategies', JSON.stringify(newFavorites));
      return newFavorites;
    });
  }, []);

  // 下拉框用的策略列表（收藏置顶）
  const orderedStrategies = useMemo(() => {
    return [...strategies].sort((a, b) => {
      const aFav = favoriteStrategyIds.includes(a.id);
      const bFav = favoriteStrategyIds.includes(b.id);
      if (aFav && !bFav) return -1;
      if (!aFav && bFav) return 1;
      return 0;
    });
  }, [strategies, favoriteStrategyIds]);

  // 策略切换
  const handleStrategyToggle = useCallback((strategyId: string, selected: boolean) => {
    setSelectedStrategyIds(prev => {
      if (selected) {
        return [...prev, strategyId];
      } else {
        return prev.filter(id => id !== strategyId);
      }
    });
  }, []);

  // 获取策略列表
  const fetchStrategies = useCallback(async () => {
    setIsLoadingStrategies(true);
    try {
      const data = await backtestApi.getStrategies();
      setStrategies(data);
    } catch (err) {
      console.error('获取策略列表失败:', err);
    } finally {
      setIsLoadingStrategies(false);
    }
  }, []);

  // 获取最近回测结果的函数
  const fetchLatestBacktestResults = useCallback(async (dir?: string) => {
    console.log('=== fetchLatestBacktestResults 被调用 ===', { dir });
    console.log('⏳ 调用时间:', new Date().toISOString());
    
    if (!isMountedRef.current) {
      console.warn('组件已卸载，跳过');
      return;
    }
    
    try {
      console.log('📡 正在请求API获取最近回测结果...');
      const response = await backtestApi.getLatestBacktestResults(dir);
      
      if (!isMountedRef.current) {
        console.warn('收到响应时组件已卸载，跳过');
        return;
      }
      
      console.log('✅ 收到API响应:', response);
      console.log('  - success:', response.success);
      console.log('  - images keys:', Object.keys(response.images || {}));
      console.log('  - quantstatsHtml:', response.quantstatsHtml);
      console.log('  - quantstatsMetrics:', response.quantstatsMetrics ? '有' : '无');
      
      if (response.success) {
        console.log('🔄 更新前端状态...');
        
        // 更新图片
        console.log('  - 设置 backtestImages:', response.images);
        setBacktestImages(response.images || {});
        
        // 更新数据
        if (response.data) {
          console.log('  - 设置 latestBacktestData');
          setLatestBacktestData(response.data);
        }
        
        // 更新 QuantStats HTML
        if (response.quantstatsHtml) {
          console.log('  - 设置 quantstatsHtml:', response.quantstatsHtml);
          setQuantstatsHtml(response.quantstatsHtml);
        }
        
        // 更新 QuantStats Metrics
        if (response.quantstatsMetrics) {
          console.log('  - 设置 quantstatsMetrics');
          setQuantstatsMetrics(response.quantstatsMetrics);
        }
        
        console.log('✅ 所有状态已更新！');
      } else {
        console.warn('API返回success=false:', response);
      }
    } catch (err) {
      console.error('❌ 获取最近回测结果失败:', err);
    }
  }, []); // 移除所有依赖项！

  // 加载历史回测列表
  const loadHistoryList = useCallback(async () => {
    try {
      const response = await backtestApi.listBacktestResults();
      if (response.success && response.items) {
        const items = response.items.map(item => ({
          name: item.name,
          modifiedAt: item.modifiedAt ?? item.modified_at ?? 0,
        }));
        setHistoryItems(items);
      }
    } catch (err) {
      console.error('❌ 获取历史回测列表失败:', err);
    }
  }, []);

  // 将回测文件夹名解析为友好展示标签（策略名 · 止盈名 · 最大持仓 · 日期区间）
  const formatHistoryLabel = useCallback((name: string) => {
    // 命名格式：选股策略_止盈止损_最大持仓N_起始日期_结束日期
    // 兼容旧格式：选股策略_止盈止损_起始日期_结束日期（无"最大持仓"段）
    const parts = name.split('_').filter(Boolean);
    if (parts.length >= 4) {
      // 定位日期段（倒数第2、倒数第1），识别可选的"最大持仓N"段
      const start = parts[parts.length - 2];
      const end = parts[parts.length - 1];
      let maxHold = '';
      const prefix = parts.slice(0, -2);
      for (let i = prefix.length - 1; i >= 0; i--) {
        if (/^最大持仓\d+$/.test(prefix[i])) {
          maxHold = prefix[i];
          prefix.splice(i, 1);
          break;
        }
      }
      const strategy = prefix.slice(0, -1).join('_');
      const exit = prefix[prefix.length - 1] || '';
      let label = `${strategy} · ${exit}`;
      if (maxHold) label += ` · ${maxHold}`;
      label += ` · ${start} 至 ${end}`;
      return label;
    }
    return name;
  }, []);

  // 切换历史回测视图
  const handleHistoryChange = useCallback((dir: string) => {
    setSelectedHistoryDir(dir);
    // 空串表示最新结果
    fetchLatestBacktestResults(dir ? dir : undefined);
  }, [fetchLatestBacktestResults]);

  // 删除指定的历史回测文件夹
  const handleDeleteHistory = useCallback(async (dir: string) => {
    const label = formatHistoryLabel(dir);
    if (!window.confirm(`确定要删除历史回测「${label}」吗？\n该操作会删除对应的结果文件夹，且不可恢复。`)) {
      return;
    }
    try {
      const response = await backtestApi.deleteBacktestResults(dir);
      if (response.success) {
        // 若删除的是当前选中的回测，切回最新
        if (selectedHistoryDir === dir) {
          setSelectedHistoryDir('');
          await fetchLatestBacktestResults();
        }
        await loadHistoryList();
      } else {
        window.alert(response.message || '删除失败');
      }
    } catch (err) {
      console.error('❌ 删除历史回测失败:', err);
      window.alert('删除历史回测失败，请稍后重试');
    }
  }, [selectedHistoryDir, fetchLatestBacktestResults, loadHistoryList, formatHistoryLabel]);

  // 获取任务状态的函数
  const fetchTaskStatus = useCallback(async (currentTaskId: string) => {
    console.log('=== fetchTaskStatus 被调用 ===');
    console.log('currentTaskId:', currentTaskId);
    console.log('isMountedRef.current:', isMountedRef.current);
    
    if (!isMountedRef.current) {
      console.log('组件已卸载，跳过');
      return;
    }
    
    if (!currentTaskId) {
      console.log('taskId 无效，跳过');
      return;
    }
    
    try {
      console.log('正在获取任务状态:', currentTaskId);
      const response = await backtestApi.getBacktestTaskStatus(currentTaskId);
      
      console.log('收到任务状态响应:', response);
      
      if (!isMountedRef.current) {
        console.log('收到响应时组件已卸载，跳过');
        return;
      }
      
      console.log('任务状态响应:', response);
      setTaskStatus(response.task);

      if (response.task) {
        const status = response.task.status;
        console.log('任务状态:', status);
        
        // 任务结束
        if (status === 'completed' || status === 'failed' || status === 'stopped') {
          console.log('任务结束，停止轮询');
          setIsRunning(false);
          setIsStopping(false);
          
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }

          if (status === 'failed') {
            const errorMsg = response.task.error || '回测失败';
            setRunError(errorMsg);
          }
          
          // 如果任务成功完成，刷新最新的回测结果
          if (status === 'completed') {
            console.log('✅ 回测完成！等待报告文件完全写入磁盘...');
            // 延迟2.5秒，确保所有文件完全写入（包括QuantStats的HTML、图片和JSON）
            console.log('⌛ 等待 2.5 秒...');
            await new Promise(resolve => setTimeout(resolve, 2500));
            console.log('🔄 开始刷新最新回测结果...');
            // 回测完成后切回最新结果，并刷新历史列表
            setSelectedHistoryDir('');
            await fetchLatestBacktestResults();
            await loadHistoryList();
            console.log('✅ 刷新完成！');
          }
        }
      }
    } catch (err) {
      console.error('获取任务状态失败:', err);
      // 如果出错，清理掉无效的 taskId 和轮询
      setTaskId(null);
      setIsRunning(false);
      setIsStopping(false);
      setTaskStatus(null);
      // 清理 localStorage
      localStorage.removeItem('backtest_task_id');
      // 清理轮询定时器
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    }
  }, [fetchLatestBacktestResults, loadHistoryList]);

  // 加载默认配置和策略列表
  useEffect(() => {
    const loadDefaults = async () => {
      try {
        console.log('正在加载回测配置...');
        const config = await backtestApi.getBacktestConfig();
        console.log('获取到的配置:', config);
        
        // 使用配置文件的值，或回退到默认值
        if (config.start_date) {
          setStartDate(config.start_date);
        }
        if (config.end_date) {
          setEndDate(config.end_date);
        }
        if (config.max_positions !== undefined) {
          setMaxPositions(config.max_positions);
        }
        console.log('配置加载完成');
      } catch (error) {
        console.error('加载默认配置失败:', error);
        // 加载失败时使用默认值
        const today = new Date();
        const oneYearAgo = new Date();
        oneYearAgo.setFullYear(today.getFullYear() - 1);
        setEndDate(today.toISOString().split('T')[0]);
        setStartDate(oneYearAgo.toISOString().split('T')[0]);
        setMaxPositions(3);
      }
    };

    const loadExitStrategies = async () => {
      try {
        const data = await backtestApi.getExitStrategies();
        setExitStrategiesData(data);
        // 设置默认选中
        if (data.active && data.presets[data.active]) {
          setSelectedExitPreset(data.active);
          setEditedExitParams({ ...data.presets[data.active].params });
        }
      } catch (error) {
        console.error('加载退出策略失败:', error);
      }
    };
    
    // 尝试从localStorage读取之前的taskId
    const savedTaskId = localStorage.getItem('backtest_task_id');
    if (savedTaskId) {
      console.log('发现保存的taskId:', savedTaskId);
      setTaskId(savedTaskId);
    }
    
    // 获取最近回测结果
    fetchLatestBacktestResults();
    loadHistoryList();
    
    loadDefaults();
    fetchStrategies();
    loadExitStrategies();
  }, [fetchLatestBacktestResults, fetchStrategies, loadHistoryList]);

  // 当有taskId时，获取任务状态
  useEffect(() => {
    if (taskId && !isRunning) {
      fetchTaskStatus(taskId);
    }
  }, [taskId, fetchTaskStatus, isRunning]);

  // 点击其他地方关闭策略列表
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        strategyContainerRef.current &&
        !strategyContainerRef.current.contains(event.target as Node)
      ) {
        setIsStrategyListOpen(false);
      }
      if (
        exitStrategyContainerRef.current &&
        !exitStrategyContainerRef.current.contains(event.target as Node)
      ) {
        setIsExitStrategyListOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // 组件挂载
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      // 清理所有资源
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, []);

  // 轮询任务状态
  const pollTaskStatus = fetchTaskStatus;

  // 运行策略回测
  const handleRun = async () => {
    if (selectedStrategyIds.length === 0) {
      setRunError('请选择策略');
      return;
    }

    setIsRunning(true);
    setIsStopping(false);
    setTaskId(null);
    setTaskStatus(null);
    setRunError(null);

    // 清理之前的轮询
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
    }

    try {
      const requestBody: StrategyBacktestRunAsyncRequest = {
        strategyIds: selectedStrategyIds,
        startDate: startDate,
        endDate: endDate,
        maxPositions: typeof maxPositions === 'number' ? maxPositions : 3,
      };
      
      // 添加退出策略配置
      if (selectedExitPreset && exitStrategiesData?.presets[selectedExitPreset]) {
        const preset = exitStrategiesData.presets[selectedExitPreset];
        requestBody.exitStrategy = {
          strategy: preset.strategy,
          name: preset.name,  // 止盈止损中文名，用于结果子目录命名
          params: editedExitParams,
        };
      }
      
      const response = await backtestApi.runStrategyBacktestAsync(requestBody);
      
      console.log('=== API 响应完整数据 ===');
      console.log('response:', response);
      console.log('response.taskId:', response.taskId);
      console.log('response.task_id:', (response as any).task_id);
      console.log('Object.keys(response):', Object.keys(response));
      
      const actualTaskId = (response.taskId || (response as any).task_id) as string;
      
      if (!actualTaskId) {
        throw new Error('API 没有返回有效的 taskId');
      }
      
      setTaskId(actualTaskId);
      // 保存taskId到localStorage
      localStorage.setItem('backtest_task_id', actualTaskId);
      console.log(`回测任务已提交: ${actualTaskId}`);
      
      // 清理之前的轮询
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      
      // 开始轮询任务状态
      pollTimerRef.current = setInterval(() => {
        pollTaskStatus(actualTaskId);
      }, 3000);
      
      // 立即查询一次
      pollTaskStatus(actualTaskId);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '回测任务提交失败';
      setRunError(errorMessage);
      setIsRunning(false);
    }
  };

  // 终止回测
  const handleStop = async () => {
    if (!taskId) return;

    setIsStopping(true);
    
    try {
      await backtestApi.stopStrategyBacktestByTaskId(taskId);
    } catch (err) {
      console.error('停止回测失败:', err);
      setIsStopping(false);
    }
  };

  // 获取结果数据（备用数据）
  const resultData = taskStatus?.result;
  const metrics = resultData?.metrics as Record<string, unknown> | undefined;
  
  // 调试信息
  console.log('=== BacktestPage 调试 ===');
  console.log('  - taskStatus:', taskStatus);
  console.log('  - taskStatus 完整类型:', typeof taskStatus);
  console.log('  - taskStatus 所有键:', taskStatus ? Object.keys(taskStatus) : '无');
  console.log('  - latestBacktestData:', latestBacktestData ? '有数据' : '无数据');
  console.log('  - latestBacktestData 类型:', typeof latestBacktestData);
  console.log('  - latestBacktestData 所有键:', latestBacktestData ? Object.keys(latestBacktestData) : '无');
  console.log('  - backtestImages keys:', Object.keys(backtestImages).join(', '));
  console.log('  - quantstatsHtml:', quantstatsHtml ? '有数据' : '无数据');
  console.log('  - quantstatsMetrics:', quantstatsMetrics ? '有数据' : '无数据');

  // 获取已选策略的显示文本
  const getSelectedStrategiesText = useCallback(() => {
    if (selectedStrategyIds.length === 0) {
      return '请选择策略';
    }
    const selected = strategies.filter(s => selectedStrategyIds.includes(s.id));
    if (selected.length === 1) {
      return selected[0].name;
    }
    return `${selected.length} 个策略`;
  }, [selectedStrategyIds, strategies]);

  // 获取已选退出策略的显示文本
  const getSelectedExitStrategyText = useCallback(() => {
    if (!selectedExitPreset || !exitStrategiesData?.presets[selectedExitPreset]) {
      return '止盈止损策略';
    }
    return exitStrategiesData.presets[selectedExitPreset].name;
  }, [selectedExitPreset, exitStrategiesData]);

  // 切换退出策略预设
  const handleExitPresetSelect = useCallback((presetKey: string) => {
    setSelectedExitPreset(presetKey);
    setIsExitStrategyListOpen(false);
    if (exitStrategiesData?.presets[presetKey]) {
      setEditedExitParams({ ...exitStrategiesData.presets[presetKey].params });
    }
  }, [exitStrategiesData]);

  // 更新退出策略参数
  const handleExitParamChange = useCallback((paramKey: string, value: string) => {
    setEditedExitParams(prev => ({
      ...prev,
      [paramKey]: parseFloat(value) || 0,
    }));
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* 页面头部 */}
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-start gap-2 w-full flex-wrap">
          {/* 策略选择 */}
          <div ref={strategyContainerRef} className="relative">
            <button
              type="button"
              onClick={() => setIsStrategyListOpen(!isStrategyListOpen)}
              disabled={isRunning || isLoadingStrategies}
              className="input-terminal text-xs py-2 px-3 min-w-64 flex items-center justify-between"
            >
              <span>{getSelectedStrategiesText()}</span>
              <svg
                className={`w-4 h-4 transition-transform ${isStrategyListOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {isStrategyListOpen && (
              <div className="absolute top-full left-0 mt-1 w-72 bg-elevated border border-white/15 rounded-xl shadow-2xl z-[9999] max-h-80 overflow-y-auto">
                {orderedStrategies.length === 0 ? (
                  <div className="px-3 py-4 text-center text-xs text-muted">
                    加载策略中...
                  </div>
                ) : (
                  orderedStrategies.map((strategy) => (
                    <StrategyItem
                      key={strategy.id}
                      strategy={strategy}
                      isSelected={selectedStrategyIds.includes(strategy.id)}
                      onToggle={handleStrategyToggle}
                      isLoading={isRunning}
                      isFavorited={favoriteStrategyIds.includes(strategy.id)}
                      onToggleFavorite={toggleFavorite}
                    />
                  ))
                )}
              </div>
            )}
          </div>

          {/* 退出策略选择 */}
          <div ref={exitStrategyContainerRef} className="relative">
            <button
              type="button"
              onClick={() => setIsExitStrategyListOpen(!isExitStrategyListOpen)}
              disabled={isRunning}
              className="input-terminal text-xs py-2 px-3 min-w-48 flex items-center justify-between"
            >
              <span>{getSelectedExitStrategyText()}</span>
              <svg
                className={`w-4 h-4 transition-transform ${isExitStrategyListOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {isExitStrategyListOpen && exitStrategiesData && (
              <div className="absolute top-full left-0 mt-1 w-56 bg-elevated border border-white/15 rounded-xl shadow-2xl z-[9999] max-h-80 overflow-y-auto">
                {Object.entries(exitStrategiesData.presets).map(([key, preset]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleExitPresetSelect(key)}
                    className={`w-full text-left px-3 py-2.5 hover:bg-white/10 transition-colors border-b border-white/5 last:border-0 ${
                      selectedExitPreset === key ? 'bg-white/10' : ''
                    }`}
                  >
                    <div className="text-sm text-white">{preset.name}</div>
                    <div className="text-xs text-muted">
                      {preset.strategy === 'simple' ? '固定止盈止损' : '动态分级止盈'}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* 开始日期 */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">开始</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              disabled={isRunning}
              className="input-terminal text-xs py-2"
            />
          </div>

          {/* 结束日期 */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">结束</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              disabled={isRunning}
              className="input-terminal text-xs py-2"
            />
          </div>

          {/* 最高持仓数 */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">最高持仓</span>
            <input
              type="number"
              min="1"
              value={maxPositions}
              onChange={(e) => {
                const val = e.target.value;
                setMaxPositions(val ? parseInt(val, 10) : '');
              }}
              disabled={isRunning}
              placeholder="不限制"
              className="input-terminal text-xs py-2 w-24"
            />
          </div>

          {/* 运行按钮 */}
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning || selectedStrategyIds.length === 0}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
          >
            {isRunning ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                {taskStatus ? `运行中 (${taskStatus.status})` : '提交中...'}
              </>
            ) : (
              '运行回测'
            )}
          </button>

          {/* 终止按钮 */}
          <button
            type="button"
            onClick={handleStop}
            disabled={!isRunning || isStopping || !taskId}
            className="btn-secondary flex items-center gap-1.5 whitespace-nowrap border-red-500/30 hover:border-red-500/50 text-red-400 hover:text-red-300"
          >
            {isStopping ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                </svg>
                停止中...
              </>
            ) : (
              '终止'
            )}
          </button>
        </div>

        {/* 任务状态显示 */}
        {taskStatus && (
          <div className="mt-2 flex items-center gap-2 text-xs">
            <span className="text-muted">任务ID:</span>
            <span className="font-mono text-white">{taskId}</span>
            <span className="text-muted mx-2">|</span>
            <span className="text-muted">状态:</span>
            <span className={`font-mono ${
              taskStatus.status === 'completed' ? 'text-green-400' :
              taskStatus.status === 'failed' || taskStatus.status === 'stopped' ? 'text-red-400' :
              'text-yellow-400'
            }`}>
              {taskStatus.status}
            </span>
          </div>
        )}

        {/* 退出策略参数编辑面板 */}
        {selectedExitPreset && exitStrategiesData?.presets[selectedExitPreset] && Object.keys(editedExitParams).length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted">参数:</span>
            {Object.entries(editedExitParams).map(([key, value]) => (
              <div key={key} className="flex items-center gap-1">
                <span className="text-xs text-secondary">{key.replace(/_/g, ' ')}</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={value}
                  onChange={(e) => handleExitParamChange(key, e.target.value)}
                  disabled={isRunning}
                  className="input-terminal text-xs py-1 px-2 w-20"
                />
              </div>
            ))}
          </div>
        )}
      </header>

      {/* 页面主体 */}
      <main className="flex-1 p-4 overflow-y-auto">
        <div className="w-full">
          {/* 错误提示 */}
          {runError && (
            <Card padding="md" className="mb-4 border-red-500/30 bg-red-500/10">
              <div className="text-red-400 text-sm">
                <strong>错误:</strong> {runError}
              </div>
            </Card>
          )}

          {/* 终端日志 - 暂时隐藏 */}
          {/* <TerminalLog logs={logs} /> */}

          {/* 调试信息 */}
          <div className="mb-4">
            <Card padding="sm">
              <div className="text-xs text-muted h-[200px] overflow-y-auto">
                <p>调试信息:</p>
                <p>taskStatus.status: {JSON.stringify(taskStatus?.status)}</p>
                <p>taskStatus 完整对象: {JSON.stringify(taskStatus, null, 2)}</p>
                <p>latestBacktestData: {latestBacktestData ? '有数据' : '无数据'}</p>
                <p>backtestImages keys: {Object.keys(backtestImages).join(', ')}</p>
                <p>quantstatsHtml: {quantstatsHtml ? quantstatsHtml : '无数据'}</p>
                <p>quantstatsMetrics: {quantstatsMetrics ? '有数据' : '无数据'}</p>
                {quantstatsMetrics && (
                  <p>quantstatsMetrics keys: {Object.keys(quantstatsMetrics).join(', ')}</p>
                )}
              </div>
            </Card>
          </div>

          {/* 回测结果 */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">回测结果</h2>
              {historyItems.length > 0 && (
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted whitespace-nowrap">历史回测:</label>
                  <div className="relative">
                    <button
                      type="button"
                      className="input-terminal text-xs py-2 px-3 min-w-[432px] flex items-center justify-between gap-2"
                      onClick={() => setHistoryDropdownOpen(prev => !prev)}
                    >
                      <span className="truncate">
                        {selectedHistoryDir
                          ? formatHistoryLabel(selectedHistoryDir)
                          : '（最新一次回测）'}
                      </span>
                      <svg className={`w-4 h-4 transition-transform ${historyDropdownOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {historyDropdownOpen && (
                      <>
                        {/* 点击空白处关闭下拉 */}
                        <div className="fixed inset-0 z-10" onClick={() => setHistoryDropdownOpen(false)} />
                        <div className="absolute top-full left-0 right-0 mt-1 max-h-72 overflow-y-auto rounded-lg border border-white/10 bg-[#1a1f2e] shadow-lg z-20">
                          <button
                            type="button"
                            className={`w-full text-left text-xs px-3 py-2 hover:bg-white/5 ${selectedHistoryDir === '' ? 'text-blue-400' : 'text-gray-200'}`}
                            onClick={() => {
                              handleHistoryChange('');
                              setHistoryDropdownOpen(false);
                            }}
                          >
                            （最新一次回测）
                          </button>
                          {historyItems.map(item => (
                            <div
                              key={item.name}
                              className={`flex items-center justify-between px-3 py-2 hover:bg-white/5 ${selectedHistoryDir === item.name ? 'text-blue-400' : 'text-gray-200'}`}
                            >
                              <button
                                type="button"
                                className="flex-1 text-left text-xs"
                                onClick={() => {
                                  handleHistoryChange(item.name);
                                  setHistoryDropdownOpen(false);
                                }}
                              >
                                <span className="truncate block">{formatHistoryLabel(item.name)}</span>
                              </button>
                              <button
                                type="button"
                                title="删除该历史回测"
                                className="ml-2 p-1 rounded text-red-400 hover:text-red-300 hover:bg-red-500/10"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteHistory(item.name);
                                }}
                              >
                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </button>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {/* QuantStats 专业分析报告 - 优先显示 */}
            {quantstatsHtml && (
              <div className="space-y-4">
                <Card padding="md">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-white">QuantStats 专业分析报告</h3>
                    <a
                      href={`/${quantstatsHtml}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-400 hover:text-blue-300 underline"
                    >
                      在新标签页打开
                    </a>
                  </div>
                  <iframe
                    src={`/${quantstatsHtml}`}
                    title="QuantStats Report"
                    className="w-full h-[800px] rounded-lg border border-white/10"
                    sandbox="allow-scripts allow-same-origin"
                  />
                </Card>
                
                {/* QuantStats 指标显示 */}
                {quantstatsMetrics && (
                  <Card padding="md">
                    <h3 className="text-sm font-semibold text-white mb-4">QuantStats 关键指标</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(quantstatsMetrics).map(([key, value]) => (
                        <div key={key} className="p-3 rounded-lg bg-white/5">
                          <div className="text-xs text-muted truncate">{key}</div>
                          <div className="text-sm font-semibold text-white mt-1">
                            {typeof value === 'number' 
                              ? (key.toLowerCase().includes('pct') || key.toLowerCase().includes('rate') || key.toLowerCase().includes('ratio'))
                                ? `${(value * 100).toFixed(2)}%`
                                : value.toFixed(4)
                              : String(value)
                            }
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
                
                {/* QuantStats 图表 */}
                <div className="space-y-4">
                  {backtestImages['qs_equity_curve'] && (
                    <Card padding="md">
                      <h3 className="text-sm font-semibold text-white mb-2">净值曲线</h3>
                      <img
                        src={`/${backtestImages['qs_equity_curve']}`}
                        alt="净值曲线"
                        className="w-full rounded-lg"
                      />
                    </Card>
                  )}
                  
                  {backtestImages['qs_drawdown_curve'] && (
                    <Card padding="md">
                      <h3 className="text-sm font-semibold text-white mb-2">回撤曲线</h3>
                      <img
                        src={`/${backtestImages['qs_drawdown_curve']}`}
                        alt="回撤曲线"
                        className="w-full rounded-lg"
                      />
                    </Card>
                  )}
                  
                  {backtestImages['qs_monthly_heatmap'] && (
                    <Card padding="md">
                      <h3 className="text-sm font-semibold text-white mb-2">月度收益热力图</h3>
                      <img
                        src={`/${backtestImages['qs_monthly_heatmap']}`}
                        alt="月度收益热力图"
                        className="w-full rounded-lg"
                      />
                    </Card>
                  )}
                </div>
              </div>
            )}
            
            {/* 传统图表 - 如果没有 QuantStats 或有额外图表 */}
            {(!quantstatsHtml || (backtestImages['equity_curve'] || backtestImages['drawdown_curve'] || backtestImages['metrics_heatmap'] || backtestImages['metrics_radar'])) && (
              <div className="space-y-4">
                {/* 净值曲线 */}
                {backtestImages['equity_curve'] && (
                  <Card padding="md">
                    <h3 className="text-sm font-semibold text-white mb-2">净值曲线 (原始)</h3>
                    <img
                      src={`/${backtestImages['equity_curve']}`}
                      alt="净值曲线"
                      className="w-full rounded-lg"
                    />
                  </Card>
                )}
                
                {/* 回撤曲线 */}
                {backtestImages['drawdown_curve'] && (
                  <Card padding="md">
                    <h3 className="text-sm font-semibold text-white mb-2">回撤曲线 (原始)</h3>
                    <img
                      src={`/${backtestImages['drawdown_curve']}`}
                      alt="回撤曲线"
                      className="w-full rounded-lg"
                    />
                  </Card>
                )}
                
                {/* 指标热力图 */}
                {backtestImages['metrics_heatmap'] && (
                  <Card padding="md">
                    <h3 className="text-sm font-semibold text-white mb-2">指标热力图 (原始)</h3>
                    <img
                      src={`/${backtestImages['metrics_heatmap']}`}
                      alt="指标热力图"
                      className="w-full rounded-lg"
                    />
                  </Card>
                )}
                
                {/* 指标雷达图 */}
                {backtestImages['metrics_radar'] && (
                  <Card padding="md">
                    <h3 className="text-sm font-semibold text-white mb-2">指标雷达图 (原始)</h3>
                    <img
                      src={`/${backtestImages['metrics_radar']}`}
                      alt="指标雷达图"
                      className="w-full rounded-lg"
                    />
                  </Card>
                )}
              </div>
            )}
            
            {/* 或者，用最新保存的数据显示绩效指标和图表 */}
            {Object.keys(backtestImages).length === 0 && (
              <>
                {/* 绩效指标 */}
                {(metrics || latestBacktestData?.metrics) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <MetricsCard
                      metrics={metrics || latestBacktestData?.metrics}
                      title="策略绩效"
                    />
                  </div>
                )}

                {/* 图表 */}
                <div className="mt-6">
                  <BacktestChartsContainer
                    loading={isRunning && !resultData && !latestBacktestData}
                    error={runError}
                    results={resultData?.results || latestBacktestData?.results}
                    metrics={resultData?.metrics || latestBacktestData?.metrics}
                    onRetry={() => {
                      if (taskId) {
                        pollTaskStatus(taskId);
                      }
                    }}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default BacktestPage;
