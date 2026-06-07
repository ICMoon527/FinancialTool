/**
 * 天道 DRAWBAND 自定义绘制组件
 * 使用 ISeriesPrimitive API 在金钻趋势和金牛2之间绘制红绿带状填充区域
 */

interface BandDataPoint {
  date: string;
  td_jinzuan: number | null;
  td_jinniu2: number | null;
}

export class TiandaoBandPrimitive {
  private _chart: any = null;
  private _series: any = null;
  public data: BandDataPoint[] = [];

  attached({ chart, series }: any) {
    this._chart = chart;
    this._series = series;
  }

  detached() {
    // cleanup
  }

  paneViews() {
    const self = this;
    return [
      {
        zOrder() {
          return 'bottom' as const;
        },
        renderer() {
          return {
            draw(target: any) {
              if (!self._chart || !self._series || self.data.length === 0) return;
              const timeScale = self._chart.timeScale();
              const series = self._series;

              target.useMediaCoordinateSpace(({ context: ctx }: { context: CanvasRenderingContext2D }) => {
                // 分别收集红色和绿色区域的点坐标
                const redPoints: { x: number; yTop: number; yBottom: number }[] = [];
                const greenPoints: { x: number; yTop: number; yBottom: number }[] = [];

                for (const item of self.data) {
                  if (item.td_jinzuan == null || item.td_jinniu2 == null) continue;

                  const x = timeScale.timeToCoordinate(item.date as any);
                  if (x === null) continue;

                  const yJin = series.priceToCoordinate(item.td_jinzuan);
                  const yNiu = series.priceToCoordinate(item.td_jinniu2);
                  if (yJin === null || yNiu === null) continue;

                  if (item.td_jinzuan > item.td_jinniu2) {
                    redPoints.push({ x, yTop: yJin, yBottom: yNiu });
                  } else if (item.td_jinniu2 > item.td_jinzuan) {
                    greenPoints.push({ x, yTop: yNiu, yBottom: yJin });
                  }
                }

                // 绘制红色填充区域
                if (redPoints.length >= 2) {
                  ctx.fillStyle = 'rgba(55, 0, 0, 0.2)';
                  ctx.beginPath();
                  ctx.moveTo(redPoints[0].x, redPoints[0].yTop);
                  for (let i = 1; i < redPoints.length; i++) {
                    ctx.lineTo(redPoints[i].x, redPoints[i].yTop);
                  }
                  for (let i = redPoints.length - 1; i >= 0; i--) {
                    ctx.lineTo(redPoints[i].x, redPoints[i].yBottom);
                  }
                  ctx.closePath();
                  ctx.fill();
                }

                // 绘制绿色填充区域
                if (greenPoints.length >= 2) {
                  ctx.fillStyle = 'rgba(0, 91, 0, 0.2)';
                  ctx.beginPath();
                  ctx.moveTo(greenPoints[0].x, greenPoints[0].yTop);
                  for (let i = 1; i < greenPoints.length; i++) {
                    ctx.lineTo(greenPoints[i].x, greenPoints[i].yTop);
                  }
                  for (let i = greenPoints.length - 1; i >= 0; i--) {
                    ctx.lineTo(greenPoints[i].x, greenPoints[i].yBottom);
                  }
                  ctx.closePath();
                  ctx.fill();
                }
              });
            },
          };
        },
      },
    ];
  }
}