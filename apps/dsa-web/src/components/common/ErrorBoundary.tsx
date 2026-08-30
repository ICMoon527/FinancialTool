import React from 'react';

declare const __BUILD_TIME__: string;

interface ErrorBoundaryState {
  error: Error | null;
  info: string;
}

/**
 * 全局错误边界
 * 捕获渲染期崩溃（如 Maximum update depth exceeded），在页面上直接显示
 * 可读的错误信息与组件栈，避免只看到白屏或预览壳的压缩报错。
 */
export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null, info: '' };

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error('❌ [ErrorBoundary] 渲染崩溃:', error, info.componentStack);
    this.setState({ info: info.componentStack || '' });
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: 24,
            color: '#e2e8f0',
            background: '#0f172a',
            fontFamily: 'monospace',
            fontSize: 13,
            minHeight: '100vh',
          }}
        >
          <h2 style={{ color: '#f87171' }}>页面渲染崩溃</h2>
          <p style={{ margin: '12px 0' }}>
            <strong>错误信息：</strong>
            {this.state.error.message}
          </p>
          <p style={{ margin: '12px 0' }}>
            <strong>前端构建时间：</strong>
            {__BUILD_TIME__}
            （如果这个时间不是最近一次构建，说明浏览器加载的是旧缓存，请 Ctrl+F5 强制刷新）
          </p>
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              color: '#94a3b8',
              maxHeight: '50vh',
              overflow: 'auto',
              background: '#020617',
              padding: 12,
              borderRadius: 8,
            }}
          >
            {this.state.info}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 12,
              padding: '8px 16px',
              background: '#0891b2',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              cursor: 'pointer',
            }}
          >
            刷新页面
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
