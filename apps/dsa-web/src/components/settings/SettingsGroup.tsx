import type React from 'react';
import { useState } from 'react';
import type { SystemConfigItem } from '../../types/systemConfig';
import { SettingsField } from './SettingsField';
import type { ConfigValidationIssue } from '../../types/systemConfig';

interface SettingsGroupProps {
  name: string;
  items: SystemConfigItem[];
  issueByKey: Record<string, ConfigValidationIssue[]>;
  disabled: boolean;
  onChange: (key: string, value: string) => void;
}

export const SettingsGroup: React.FC<SettingsGroupProps> = ({
  name,
  items,
  issueByKey,
  disabled,
  onChange,
}) => {
  const [expanded, setExpanded] = useState(false);

  if (items.length === 0) {
    return null;
  }

  // 统计已配置的项数（有实际值的项）
  const configuredCount = items.filter((item) => item.rawValueExists && item.value !== '').length;

  return (
    <div className="rounded-xl border border-white/8 overflow-hidden">
      <button
        type="button"
        className="w-full flex items-center justify-between px-4 py-3 bg-elevated/40 hover:bg-elevated/60 transition text-left"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <svg
            className={`w-4 h-4 text-muted transition-transform ${expanded ? 'rotate-90' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <span className="text-sm font-medium text-white">{name}</span>
        </div>
        <span className="text-xs text-muted">
          {configuredCount}/{items.length}
        </span>
      </button>

      {expanded ? (
        <div className="border-t border-white/8 px-4 pb-3 pt-2 space-y-3">
          {items.map((item) => (
            <SettingsField
              key={item.key}
              item={item}
              value={item.value}
              disabled={disabled}
              onChange={onChange}
              issues={issueByKey[item.key] || []}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
};