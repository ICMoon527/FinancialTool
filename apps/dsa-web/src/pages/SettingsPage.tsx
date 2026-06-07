import type React from 'react';
import { useEffect, useMemo } from 'react';
import { useAuth, useSystemConfig } from '../hooks';
import {
  ChangePasswordCard,
  ImageStockExtractor,
  SettingsAlert,
  SettingsGroup,
  SettingsLoading,
} from '../components/settings';
import { getCategoryDescriptionZh, getCategoryTitleZh } from '../utils/systemConfigI18n';
import type { SystemConfigItem } from '../types/systemConfig';

const SettingsPage: React.FC = () => {
  const { passwordChangeable } = useAuth();
  const {
    categories,
    itemsByCategory,
    issueByKey,
    activeCategory,
    setActiveCategory,
    hasDirty,
    dirtyCount,
    toast,
    clearToast,
    isLoading,
    isSaving,
    loadError,
    saveError,
    retryAction,
    load,
    retry,
    save,
    setDraftValue,
    configVersion,
    maskToken,
    configOutdated,
    confirmReload,
  } = useSystemConfig();

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!toast) {
      return;
    }

    const timer = window.setTimeout(() => {
      clearToast();
    }, 3200);

    return () => {
      window.clearTimeout(timer);
    };
  }, [clearToast, toast]);

  const activeItems = itemsByCategory[activeCategory] || [];

  // 按 group 分组当前分类下的配置项
  const activeGroups = useMemo(() => {
    const groupMap = new Map<string, SystemConfigItem[]>();
    // 没有 group 的配置项归入"其他"组
    groupMap.set('其他', []);
    for (const item of activeItems) {
      const group = item.schema?.group || '其他';
      if (!groupMap.has(group)) {
        groupMap.set(group, []);
      }
      groupMap.get(group)!.push(item);
    }
    // 移除空的"其他"组
    if (groupMap.get('其他')!.length === 0) {
      groupMap.delete('其他');
    }
    return Array.from(groupMap.entries()).map(([name, items]) => ({ name, items }));
  }, [activeItems]);

  return (
    <div className="min-h-screen px-4 pb-6 pt-4 md:px-6">
      <header className="mb-4 rounded-2xl border border-white/8 bg-card/80 p-4 backdrop-blur-sm">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-xl font-semibold text-white">系统设置</h1>
            <p className="text-sm text-secondary">
              默认使用 .env 中的配置
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className="btn-secondary" onClick={() => void load()} disabled={isLoading || isSaving}>
              重置
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={() => void save()}
              disabled={!hasDirty || isSaving || isLoading}
            >
              {isSaving ? '保存中...' : `保存配置${dirtyCount ? ` (${dirtyCount})` : ''}`}
            </button>
          </div>
        </div>

        {saveError ? (
          <SettingsAlert
            className="mt-3"
            title="保存失败"
            message={saveError}
            actionLabel={retryAction === 'save' ? '重试保存' : undefined}
            onAction={retryAction === 'save' ? () => void retry() : undefined}
          />
        ) : null}
      </header>

      {configOutdated ? (
        <div className="mb-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/>
              </svg>
              <span className="text-sm text-amber-200">检测到配置文件已更新，点击刷新加载最新配置。</span>
            </div>
            <button
              type="button"
              className="btn-secondary text-sm"
              onClick={() => void confirmReload()}
              disabled={isLoading || isSaving}
            >
              刷新配置
            </button>
          </div>
        </div>
      ) : null}

      {loadError ? (
        <SettingsAlert
          title="加载设置失败"
          message={loadError}
          actionLabel={retryAction === 'load' ? '重试加载' : '重新加载'}
          onAction={() => void retry()}
          className="mb-4"
        />
      ) : null}

      {isLoading ? (
        <SettingsLoading />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[260px_1fr]">
          <aside className="rounded-2xl border border-white/8 bg-card/60 p-3 backdrop-blur-sm">
            <p className="mb-2 text-xs uppercase tracking-wide text-muted">配置分类</p>
            <div className="space-y-2">
              {categories.map((category) => {
                const isActive = category.category === activeCategory;
                const count = (itemsByCategory[category.category] || []).length;
                const title = getCategoryTitleZh(category.category, category.title);
                const description = getCategoryDescriptionZh(category.category, category.description);

                return (
                  <button
                    key={category.category}
                    type="button"
                    className={`w-full rounded-lg border px-3 py-2 text-left transition ${
                      isActive
                        ? 'border-accent bg-cyan/10 text-white'
                        : 'border-white/8 bg-elevated/40 text-secondary hover:border-white/16 hover:text-white'
                    }`}
                    onClick={() => setActiveCategory(category.category)}
                  >
                    <span className="flex items-center justify-between text-sm font-medium">
                      {title}
                      <span className="text-xs text-muted">{count}</span>
                    </span>
                    {description ? <span className="mt-1 block text-xs text-muted">{description}</span> : null}
                  </button>
                );
              })}
            </div>
          </aside>

          <section className="space-y-3 rounded-2xl border border-white/8 bg-card/60 p-4 backdrop-blur-sm">
            {activeCategory === 'home' ? (
              <div className="space-y-3">
                <ImageStockExtractor
                  stockListValue={
                    (activeItems.find((i) => i.key === 'STOCK_LIST')?.value as string) ?? ''
                  }
                  configVersion={configVersion}
                  maskToken={maskToken}
                  onMerged={() => void load()}
                  disabled={isSaving || isLoading}
                />
              </div>
            ) : null}
            {activeCategory === 'settings' && passwordChangeable ? (
              <div className="space-y-3">
                <ChangePasswordCard />
              </div>
            ) : null}
            {activeItems.length ? (
              activeGroups.map((group) => (
                <SettingsGroup
                  key={group.name}
                  name={group.name}
                  items={group.items}
                  issueByKey={issueByKey}
                  disabled={isSaving}
                  onChange={setDraftValue}
                />
              ))
            ) : (
              <div className="rounded-xl border border-white/8 bg-elevated/40 p-5 text-sm text-secondary">
                当前分类下暂无配置项。
              </div>
            )}
          </section>
        </div>
      )}

      {toast ? (
        <div className="fixed bottom-5 right-5 z-50 w-[320px] max-w-[calc(100vw-24px)]">
          <SettingsAlert
            title={toast.type === 'success' ? '操作成功' : '操作失败'}
            message={toast.message}
            variant={toast.type === 'success' ? 'success' : 'error'}
          />
        </div>
      ) : null}
    </div>
  );
};

export default SettingsPage;
