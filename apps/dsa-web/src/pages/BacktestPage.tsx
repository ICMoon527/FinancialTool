import type React from 'react';
import { useState, useEffect, useCallback, useRef } from 'react';
import { backtestApi } from '../api/backtest';
import { Card } from '../components/common';
import { BacktestChartsContainer } from '../components/charts';
import type {
  StrategyInfo,
  StrategyBacktestTaskStatusResponse,
} from '../types/backtest';

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

// ============ 终端日志显示框 ============

const TerminalLog: React.FC<{ logs: string[] }> = ({ logs }) => {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Card padding="md" className="mt-3">
      <div className="mb-2">
        <span className="label-uppercase">终端日志</span>
      </div>
      <div 
        ref={logRef}
        className="bg-black/50 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs"
      >
        {logs.length === 0 ? (
          <p className="text-muted">等待回测开始...</p>
        ) : (
          logs.map((log, index) => (
            <div key={index} className="py-0.5">
              <span className="text-cyan">{log}</span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};

// ============ 主页面 ============

const BacktestPage: React.FC = () => {
  // 策略状态
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);

  // 日期状态
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // 最高持仓数状态
  const [maxPositions, setMaxPositions] = useState<number | ''>(3);

  // 运行状态
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<StrategyBacktestTaskStatusResponse['task'] | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // 终端日志
  const [logs, setLogs] = useState<string[]>([]);

  // 轮询定时器引用
  const pollTimerRef = useRef<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const isMountedRef = useRef<boolean>(true);

  // 添加日志
  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  }, []);

  // 获取策略列表
  const fetchStrategies = useCallback(async () => {
    setIsLoadingStrategies(true);
    try {
      const data = await backtestApi.getStrategies();
      setStrategies(data);
      if (data.length > 0 && !selectedStrategy) {
        setSelectedStrategy(data[0].id);
      }
    } catch (err) {
      console.error('获取策略列表失败:', err);
      addLog('获取策略列表失败');
    } finally {
      setIsLoadingStrategies(false);
    }
  }, [selectedStrategy, addLog]);

  // 初始化日期
  useEffect(() => {
    const today = new Date();
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(today.getFullYear() - 1);

    setEndDate(today.toISOString().split('T')[0]);
    setStartDate(oneYearAgo.toISOString().split('T')[0]);
  }, []);

  // 加载策略列表
  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  // 组件挂载
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      // 彻底清理所有资源
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  // 轮询任务状态
  const pollTaskStatus = useCallback(async (currentTaskId: string) => {
    // 防护1：检查组件是否已卸载
    if (!isMountedRef.current) {
      return;
    }
    
    // 防护2：检查 taskId 是否有效
    if (!currentTaskId || currentTaskId === 'undefined' || currentTaskId === 'null') {
      console.warn('pollTaskStatus 被调用了无效的 taskId:', currentTaskId);
      return;
    }
    
    try {
      const response = await backtestApi.getBacktestTaskStatus(currentTaskId);
      
      // 再次检查组件是否已卸载，避免在组件卸载后设置状态
      if (!isMountedRef.current) {
        return;
      }
      
      setTaskStatus(response.task);

      if (response.task) {
        const status = response.task.status;
        
        // 任务结束
        if (status === 'completed' || status === 'failed' || status === 'stopped') {
          setIsRunning(false);
          setIsStopping(false);
          
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          
          if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
          }

          if (status === 'completed') {
            addLog('回测完成');
          } else if (status === 'failed') {
            const errorMsg = response.task.error || '回测失败';
            setRunError(errorMsg);
            addLog(`错误: ${errorMsg}`);
          } else if (status === 'stopped') {
            addLog('回测已终止');
          }
        }
      }
    } catch (err) {
      console.error('获取任务状态失败:', err);
    }
  }, [addLog]);

  // 运行策略回测
  const handleRun = async () => {
    if (!selectedStrategy) {
      setRunError('请选择策略');
      addLog('错误: 请选择策略');
      return;
    }

    const strategyInfo = strategies.find(s => s.id === selectedStrategy);
    if (!strategyInfo) {
      setRunError('策略不存在');
      addLog('错误: 策略不存在');
      return;
    }

    setIsRunning(true);
    setIsStopping(false);
    setTaskId(null);
    setTaskStatus(null);
    setRunError(null);
    setLogs([]);

    // 清理之前的轮询
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // 连接SSE获取实时日志
    const eventSource = new EventSource('/api/v1/backtest/strategy/logs');
    eventSourceRef.current = eventSource;
    
    eventSource.onmessage = (event) => {
      if (event.data === '[DONE]') {
        return;
      }
      // 直接添加后端传来的日志（已经包含时间戳）
      setLogs(prev => [...prev, event.data]);
    };

    eventSource.onerror = (error) => {
      console.error('SSE连接错误:', error);
    };

    try {
      const response = await backtestApi.runStrategyBacktestAsync({
        strategyId: strategyInfo.id,
        startDate: startDate,
        endDate: endDate,
        maxPositions: typeof maxPositions === 'number' ? maxPositions : 3,
      });
      
      setTaskId(response.task_id);
      addLog(`回测任务已提交: ${response.task_id}`);
      
      // 清理之前的轮询（双重保险）
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      
      // 开始轮询任务状态
      pollTimerRef.current = setInterval(() => {
        // 再次检查 taskId 是否有效
        if (response.task_id && response.task_id !== 'undefined' && response.task_id !== 'null') {
          pollTaskStatus(response.task_id);
        } else {
          // 如果 taskId 无效，清除定时器
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
        }
      }, 3000);
      
      // 立即查询一次
      pollTaskStatus(response.task_id);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '回测任务提交失败';
      setRunError(errorMessage);
      addLog(`错误: ${errorMessage}`);
      setIsRunning(false);
      
      if (eventSource.readyState !== EventSource.CLOSED) {
        eventSource.close();
      }
    }
  };

  // 终止回测
  const handleStop = async () => {
    if (!taskId) {
      return;
    }

    setIsStopping(true);
    addLog('正在终止回测...');
    
    try {
      await backtestApi.stopStrategyBacktestByTaskId(taskId);
      addLog('已发送停止信号');
    } catch (err) {
      console.error('停止回测失败:', err);
      addLog('停止回测失败');
      setIsStopping(false);
    }
  };

  // 获取结果数据
  const resultData = taskStatus?.result;
  const metrics = resultData?.metrics as Record<string, unknown> | undefined;

  return (
    <div className="min-h-screen flex flex-col">
      {/* 页面头部 */}
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2 max-w-6xl flex-wrap">
          {/* 策略选择 */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">策略</span>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              disabled={isRunning || isLoadingStrategies}
              className="input-terminal text-xs py-2 min-w-48"
            >
              <option value="">-- 选择策略 --</option>
              {strategies.map((strategy) => (
                <option key={strategy.id} value={strategy.id}>
                  {strategy.name} ({strategy.type})
                </option>
              ))}
            </select>
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
            disabled={isRunning || !selectedStrategy}
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
      </header>

      {/* 页面主体 */}
      <main className="flex-1 p-4 overflow-y-auto">
        <div className="max-w-6xl">
          {/* 错误提示 */}
          {runError && (
            <Card padding="md" className="mb-4 border-red-500/30 bg-red-500/10">
              <div className="text-red-400 text-sm">
                <strong>错误:</strong> {runError}
              </div>
            </Card>
          )}

          {/* 终端日志 */}
          <TerminalLog logs={logs} />

          {/* 回测结果 */}
          {resultData && (
            <div className="mt-6 space-y-6">
              <h2 className="text-lg font-semibold text-white">回测结果</h2>
              
              {/* 绩效指标 */}
              {metrics && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(metrics).map(([category, categoryMetrics]) => (
                    <MetricsCard
                      key={category}
                      metrics={categoryMetrics as Record<string, unknown>}
                      title={category}
                    />
                  ))}
                </div>
              )}

              {/* 图表 */}
              <div className="mt-6">
                <BacktestChartsContainer
                  loading={isRunning && !resultData}
                  error={runError}
                  onRetry={() => {
                    if (taskId) {
                      pollTaskStatus(taskId);
                    }
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default BacktestPage;
