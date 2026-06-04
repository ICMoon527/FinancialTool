import type { Time, IChartApi, ISeriesApi } from 'lightweight-charts';

interface ChartEntry {
  chart: IChartApi;
  primarySeries: ISeriesApi<any>;
  data: { time: number; value: number }[];
  lastValueSeries?: ISeriesApi<any>;
  lastValueVisibleOrig?: boolean;
}

export interface CrosshairCallbacks {
  onMove?: (time: Time, sourceId: string) => void;
  onLeave?: (sourceId: string) => void;
}

export interface RegisterOptions {
  lastValueSeries?: ISeriesApi<any>;
}

export class CrosshairSyncEngine {
  private entries = new Map<string, ChartEntry>();
  private syncDepth = 0;
  private callbacks: CrosshairCallbacks = {};
  private crosshairActive = false;

  register(
    id: string,
    chart: IChartApi,
    primarySeries: ISeriesApi<any>,
    data: { time: number; value: number }[],
    options?: RegisterOptions,
  ): void {
    const entry: ChartEntry = {
      chart,
      primarySeries,
      data,
      lastValueSeries: options?.lastValueSeries,
    };
    if (options?.lastValueSeries) {
      try {
        const opts = (options.lastValueSeries as any).options?.();
        entry.lastValueVisibleOrig = opts?.lastValueVisible ?? false;
      } catch {
        /* ignore */
      }
    }
    this.entries.set(id, entry);
  }

  unregister(id: string): void {
    this.entries.delete(id);
  }

  clear(): void {
    this.syncDepth = 0;
    this.crosshairActive = false;
    this.callbacks = {};
    this.entries.clear();
  }

  setCallbacks(callbacks: CrosshairCallbacks): void {
    this.callbacks = callbacks;
  }

  handleMove = (sourceId: string, param: any): void => {
    if (this.syncDepth > 0) {
      if (param.time) {
        this.callbacks.onMove?.(param.time, sourceId);
      }
      return;
    }

    this.syncDepth++;
    try {
      if (param.time) {
        if (!this.crosshairActive) {
          this.crosshairActive = true;
          this._toggleLastValueVisible(false);
        }
        this.entries.forEach((entry, id) => {
          if (id === sourceId) return;
          this._setCrosshair(entry, param.time, id);
        });
        // 同时更新源图表自身：对于主图（mode=0），使其水平线锁定到分时白线值，而非跟随鼠标Y坐标
        const sourceEntry = this.entries.get(sourceId);
        if (sourceEntry) {
          this._setCrosshair(sourceEntry, param.time, sourceId);
        }
        this.callbacks.onMove?.(param.time, sourceId);
      } else {
        this.crosshairActive = false;
        this._restoreLastValueVisible();
        this.callbacks.onLeave?.(sourceId);
      }
    } finally {
      this.syncDepth--;
    }
  };

  setCrosshairAtTime(time: Time): void {
    this.syncDepth++;
    try {
      if (!this.crosshairActive) {
        this.crosshairActive = true;
        this._toggleLastValueVisible(false);
      }
      this.entries.forEach((entry, id) => {
        this._setCrosshair(entry, time, id);
      });
      this.callbacks.onMove?.(time, '__external__');
    } finally {
      this.syncDepth--;
    }
  }

  syncTimeRange(range: { from: Time; to: Time }): void {
    const fromStr = new Date(Number(range.from) * 1000).toLocaleTimeString();
    const toStr = new Date(Number(range.to) * 1000).toLocaleTimeString();
    console.log(`[Engine] syncTimeRange from=${fromStr} to=${toStr}`, range);
    console.trace('[Engine] syncTimeRange stack');
    this.entries.forEach((entry, id) => {
      try {
        entry.chart.timeScale().setVisibleRange(range);
      } catch (e) {
        console.warn(`[Engine] syncTimeRange FAILED id=${id} error=`, e);
      }
    });
  }

  clearCrosshair(): void {
    this.crosshairActive = false;
    this._restoreLastValueVisible();
    this.entries.forEach((entry) => {
      try {
        entry.chart.clearCrosshairPosition();
      } catch {
        /* ignore */
      }
    });
  }

  /** 用于 mode 0 图表在 RAF 回调中重设 crosshair，抵消内置 crosshair 的覆盖 */
  reapplyCrosshair(id: string, time: Time): void {
    const entry = this.entries.get(id);
    if (entry) {
      this._setCrosshair(entry, time, id);
    }
  }

  private _setCrosshair(entry: ChartEntry, time: Time, targetId: string): void {
    try {
      const { chart, primarySeries, data } = entry;
      const timeNum = Number(time);
      const dataPt = data.find((d) => d.time === timeNum);
      let value: number | null = null;
      let matchTime: number; // 实际匹配到的数据点时间，用于 setCrosshairPosition
      if (dataPt && dataPt.value != null && !isNaN(dataPt.value)) {
        value = dataPt.value;
        matchTime = dataPt.time;
      } else {
        let bestDiff = Infinity;
        let bestVal: number | null = null;
        let bestTime = timeNum;
        for (const d of data) {
          if (d.value == null || isNaN(d.value)) continue;
          const diff = Math.abs(d.time - timeNum);
          if (diff < bestDiff) {
            bestDiff = diff;
            bestVal = d.value;
            bestTime = d.time;
          }
        }
        value = bestVal;
        matchTime = bestTime;
      }
      if (value != null && isFinite(value) && !isNaN(value)) {
        // 使用 matchTime（实际数据点时间）而非传入的 time（可能超出数据范围），
        // 确保 crosshair 始终定位在有数据的点上，避免空白区域定位不准
        chart.setCrosshairPosition(value, matchTime as Time, primarySeries);
      }
    } catch (e) {
      console.error(`[Engine] _setCrosshair error id=${targetId}`, e);
    }
  }

  private _toggleLastValueVisible(visible: boolean): void {
    this.entries.forEach((entry) => {
      if (entry.lastValueSeries) {
        try {
          entry.lastValueSeries.applyOptions({ lastValueVisible: visible });
        } catch {
          /* ignore */
        }
      }
    });
  }

  private _restoreLastValueVisible(): void {
    this.entries.forEach((entry) => {
      if (entry.lastValueSeries && entry.lastValueVisibleOrig !== undefined) {
        try {
          entry.lastValueSeries.applyOptions({ lastValueVisible: entry.lastValueVisibleOrig });
        } catch {
          /* ignore */
        }
      }
    });
  }
}