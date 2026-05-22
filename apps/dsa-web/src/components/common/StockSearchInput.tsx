import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from 'react';
import { createPortal } from 'react-dom';
import { searchStocks } from '../../api/stockSearch';
import type { StockSearchResult, MatchSegment } from '../../api/stockSearch';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface StockSearchInputProps {
  /** 选择股票后的回调 */
  onSelect: (code: string, name: string, market: string) => void;
  /** 输入内容变动时的回调（用于父组件同步 stockCode） */
  onChange?: (query: string) => void;
  /** 无下拉结果时按 Enter 键的回调（传入当前输入内容） */
  onSubmit?: (query: string) => void;
  /** 是否禁用输入 */
  disabled?: boolean;
  /** 是否有输入错误（显示红色边框） */
  error?: string | null;
  /** 输入框占位文字 */
  placeholder?: string;
  /** 自定义样式类名 */
  className?: string;
  /** 搜索结果最大数量 */
  maxResults?: number;
}

// ---------------------------------------------------------------------------
// 高亮工具函数
// ---------------------------------------------------------------------------

function highlightText(
  text: string,
  segments: MatchSegment[],
): React.ReactNode {
  if (!segments || segments.length === 0) return text;

  const sorted = [...segments].sort((a, b) => a.start - b.start);

  const parts: React.ReactNode[] = [];
  let lastEnd = 0;

  for (const seg of sorted) {
    if (seg.start > lastEnd) {
      parts.push(text.slice(lastEnd, seg.start));
    }
    parts.push(
      <mark
        key={`${seg.start}-${seg.end}`}
        className="bg-cyan-500/30 text-cyan-200 rounded-sm px-0.5"
      >
        {text.slice(seg.start, seg.end)}
      </mark>,
    );
    lastEnd = seg.end;
  }
  if (lastEnd < text.length) {
    parts.push(text.slice(lastEnd));
  }

  return parts.length > 0 ? <>{parts}</> : text;
}

function marketBadge(market: string): { bg: string; text: string } {
  switch (market.toUpperCase()) {
    case 'SH':
      return { bg: 'bg-red-500/20', text: 'text-red-400' };
    case 'SZ':
      return { bg: 'bg-green-500/20', text: 'text-green-400' };
    default:
      return { bg: 'bg-slate-500/20', text: 'text-slate-400' };
  }
}

// ---------------------------------------------------------------------------
// 组件
// ---------------------------------------------------------------------------

const StockSearchInput: React.FC<StockSearchInputProps> = ({
  onSelect,
  onChange,
  onSubmit,
  disabled = false,
  error,
  placeholder = '搜索股票代码/名称/拼音...',
  className = '',
  maxResults = 20,
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<StockSearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [selectedName, setSelectedName] = useState('');

  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 防抖搜索
  const doSearch = useCallback(
    (q: string) => {
      if (!q.trim()) {
        setResults([]);
        setIsOpen(false);
        return;
      }
      setIsLoading(true);
      searchStocks(q, maxResults)
        .then((data) => {
          setResults(data);
          setIsOpen(true);
          setSelectedIndex(-1);
        })
        .catch(() => {
          setResults([]);
          setIsOpen(false);
        })
        .finally(() => setIsLoading(false));
    },
    [maxResults],
  );

  // 输入变更
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setQuery(value);
      setSelectedName('');

      // 同步到父组件，使按钮等依赖 stockCode 的控件能正常工作
      onChange?.(value);

      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => doSearch(value), 200);
    },
    [doSearch, onChange],
  );

  // 选择结果
  const handleSelect = useCallback(
    (result: StockSearchResult) => {
      setQuery('');
      setSelectedName(`${result.name} (${result.code})`);
      setResults([]);
      setIsOpen(false);
      setSelectedIndex(-1);
      onSelect(result.code, result.name, result.market);
    },
    [onSelect],
  );

  // 键盘导航
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (isOpen && results.length > 0) {
        switch (e.key) {
          case 'ArrowDown':
            e.preventDefault();
            setSelectedIndex((prev) => (prev + 1 >= results.length ? 0 : prev + 1));
            return;
          case 'ArrowUp':
            e.preventDefault();
            setSelectedIndex((prev) => (prev - 1 < 0 ? results.length - 1 : prev - 1));
            return;
          case 'Enter':
            e.preventDefault();
            if (selectedIndex >= 0 && selectedIndex < results.length) {
              handleSelect(results[selectedIndex]);
            } else {
              setIsOpen(false);
              if (onSubmit && query.trim()) onSubmit(query.trim());
            }
            return;
          case 'Escape':
            e.preventDefault();
            setIsOpen(false);
            setSelectedIndex(-1);
            return;
        }
      }

      if (e.key === 'Enter' && !isOpen && onSubmit && query.trim()) {
        e.preventDefault();
        onSubmit(query.trim());
      }
    },
    [isOpen, results, selectedIndex, handleSelect, onSubmit, query],
  );

  // 点击外部关闭（注意：排除 Portal 下拉框自身）
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      const isInsideWrapper = wrapperRef.current?.contains(target);
      const isInsideDropdown = dropdownRef.current?.contains(target);
      if (!isInsideWrapper && !isInsideDropdown) {
        setIsOpen(false);
        setSelectedIndex(-1);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 滚动选中项到可见区域
  useEffect(() => {
    if (selectedIndex >= 0 && isOpen) {
      const el = document.querySelector(
        `[data-search-index="${selectedIndex}"]`,
      );
      el?.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex, isOpen]);

  // 输入框聚焦事件
  const handleFocus = useCallback(() => {
    if (results.length > 0) {
      setIsOpen(true);
    }
  }, [results]);

  // 输入框样式
  const inputClassName = useMemo(() => {
    const base = `
      w-full px-4 py-2.5 rounded-lg
      bg-slate-800/50 border
      text-gray-200 placeholder-gray-500
      focus:outline-none focus:ring-2
      transition-all duration-200
    `;
    if (error) {
      return `${base} border-red-500/50 focus:ring-red-500/40 focus:border-red-500/40 hover:border-red-500/40`;
    }
    return `${base} border-cyan-500/20 focus:ring-cyan-500/40 focus:border-cyan-500/40 hover:border-cyan-500/30`;
  }, [error]);

  // 下拉框内容
  const dropdownContent = useMemo(() => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-4 text-sm text-gray-400">
          <svg
            className="animate-spin h-4 w-4 mr-2 text-cyan-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          搜索中...
        </div>
      );
    }

    if (!isLoading && results.length === 0 && query.trim()) {
      return (
        <div className="py-3 text-center text-sm text-gray-400">
          未找到相关股票
        </div>
      );
    }

    if (results.length === 0) return null;

    return (
      <ul className="py-1">
        {results.map((result, idx) => {
          const isSelected = idx === selectedIndex;
          const marketStyle = marketBadge(result.market);

          const codeSegments = result.match_segments.filter((s) => s.field === 'code');
          const nameSegments = result.match_segments.filter((s) => s.field === 'name');

          return (
            <li
              key={`${result.code}-${idx}`}
              data-search-index={idx}
              className={`
                px-3 py-2 cursor-pointer transition-colors duration-100 flex items-center gap-3
                ${isSelected ? 'bg-cyan-500/15' : 'hover:bg-slate-700/50'}
              `}
              onClick={() => handleSelect(result)}
              onMouseEnter={() => setSelectedIndex(idx)}
            >
              <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${marketStyle.bg} ${marketStyle.text}`}>
                {result.market || '-'}
              </span>
              <span className="text-sm font-mono text-gray-300 w-20 shrink-0">
                {codeSegments.length > 0
                  ? highlightText(result.code, codeSegments)
                  : result.code}
              </span>
              <span className="text-sm text-gray-100 truncate">
                {nameSegments.length > 0
                  ? highlightText(result.name, nameSegments)
                  : result.name}
              </span>
              <span className="ml-auto text-xs text-gray-500 shrink-0">
                {result.match_type}
              </span>
            </li>
          );
        })}
      </ul>
    );
  }, [results, isLoading, query, selectedIndex, handleSelect]);

  // ---------------------------------------------------------------------------
  // Portal 定位
  // ---------------------------------------------------------------------------

  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>({});

  useEffect(() => {
    if (isOpen && inputRef.current) {
      const rect = inputRef.current.getBoundingClientRect();
      setDropdownStyle({
        position: 'fixed',
        top: `${rect.bottom + 4}px`,
        left: `${rect.left}px`,
        width: `${rect.width}px`,
      });
    }
  }, [isOpen, query]);

  // ---------------------------------------------------------------------------
  // 渲染
  // ---------------------------------------------------------------------------

  const dropdownElement = isOpen && (
    <div
      ref={dropdownRef}
      style={dropdownStyle}
      className={`
        z-[9999] rounded-lg
        bg-slate-800/95 border border-cyan-500/20
        shadow-lg shadow-black/30
        max-h-64 overflow-y-auto
        backdrop-blur-sm
      `}
    >
      {dropdownContent}
    </div>
  );

  return (
    <div ref={wrapperRef} className={`relative ${className}`}>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={selectedName || query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          placeholder={placeholder}
          disabled={disabled}
          className={inputClassName}
        />
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* 下拉框通过 Portal 渲染到 body 层级，避免被父容器遮挡 */}
      {createPortal(dropdownElement, document.body)}
    </div>
  );
};

export default StockSearchInput;