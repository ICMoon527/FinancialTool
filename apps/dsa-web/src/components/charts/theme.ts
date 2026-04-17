import * as echarts from 'echarts'
import type { ChartTheme } from '../../types/backtest-charts'

export const chartTheme: ChartTheme = {
  backgroundColor: '#08080c',
  textColor: '#ffffff',
  textColorSecondary: '#a0a0b0',
  textColorMuted: '#606070',
  borderColor: 'rgba(255, 255, 255, 0.1)',
  borderColorDim: 'rgba(255, 255, 255, 0.06)',
  accentCyan: '#00d4ff',
  accentCyanDim: '#00a8cc',
  accentPurple: '#6f61f1',
  accentPurpleDim: '#5a4ed4',
  successColor: '#00ff88',
  warningColor: '#ffaa00',
  dangerColor: '#ff4466',
  gridColor: 'rgba(255, 255, 255, 0.05)',
  tooltipBackgroundColor: 'rgba(13, 13, 20, 0.95)',
}

export const registerEChartsTheme = () => {
  echarts.registerTheme('dark-financial', {
    color: [
      chartTheme.accentCyan,
      chartTheme.accentPurple,
      chartTheme.successColor,
      chartTheme.warningColor,
      chartTheme.dangerColor,
    ],
    backgroundColor: chartTheme.backgroundColor,
    textStyle: {
      color: chartTheme.textColor,
    },
    title: {
      textStyle: {
        color: chartTheme.textColor,
      },
      subtextStyle: {
        color: chartTheme.textColorSecondary,
      },
    },
    line: {
      itemStyle: {
        borderWidth: 1,
      },
      lineStyle: {
        width: 2,
      },
      symbolSize: 4,
      symbol: 'circle',
      smooth: false,
    },
    radar: {
      itemStyle: {
        borderWidth: 1,
      },
      lineStyle: {
        width: 2,
      },
      symbolSize: 4,
      symbol: 'circle',
      smooth: false,
    },
    bar: {
        itemStyle: {
          barBorderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    pie: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    scatter: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    boxplot: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    parallel: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    sankey: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    funnel: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    gauge: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
    },
    candlestick: {
        itemStyle: {
          color: chartTheme.successColor,
          color0: chartTheme.dangerColor,
          borderColor: chartTheme.successColor,
          borderColor0: chartTheme.dangerColor,
        },
    },
    graph: {
        itemStyle: {
          borderWidth: 0,
          barBorderColor: '#ccc',
        },
        lineStyle: {
          width: 1,
          color: '#aaa',
        },
        symbolSize: 4,
        symbol: 'circle',
        smooth: false,
        color: [
          chartTheme.accentCyan,
          chartTheme.accentPurple,
          chartTheme.successColor,
          chartTheme.warningColor,
          chartTheme.dangerColor,
        ],
    },
    map: {
        itemStyle: {
          areaColor: '#eee',
          color: '#eee',
          borderColor: '#444',
          borderWidth: 0.5,
        },
        label: {
          color: '#333',
        },
        emphasis: {
          itemStyle: {
            areaColor: 'rgba(255,215,0,0.8)',
            color: 'rgba(255,215,0,0.8)',
            borderColor: '#444',
            borderWidth: 1,
          },
          label: {
            color: 'rgb(100,0,0)',
          },
        },
    },
    geo: {
        itemStyle: {
          areaColor: '#eee',
          color: '#eee',
          borderColor: '#444',
          borderWidth: 0.5,
        },
        label: {
          color: '#333',
        },
        emphasis: {
          itemStyle: {
            areaColor: 'rgba(255,215,0,0.8)',
            color: 'rgba(255,215,0,0.8)',
            borderColor: '#444',
            borderWidth: 1,
          },
          label: {
            color: 'rgb(100,0,0)',
          },
        },
    },
    categoryAxis: {
        axisLine: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisTick: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisLabel: {
          show: true,
          color: chartTheme.textColorSecondary,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: chartTheme.gridColor,
            type: 'dashed',
          },
        },
        splitArea: {
          show: false,
          areaStyle: {
            color: chartTheme.gridColor,
          },
        },
    },
    valueAxis: {
        axisLine: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisTick: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisLabel: {
          show: true,
          color: chartTheme.textColorSecondary,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: chartTheme.gridColor,
            type: 'dashed',
          },
        },
        splitArea: {
          show: false,
          areaStyle: {
            color: chartTheme.gridColor,
          },
        },
    },
    logAxis: {
        axisLine: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisTick: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisLabel: {
          show: true,
          color: chartTheme.textColorSecondary,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: chartTheme.gridColor,
            type: 'dashed',
          },
        },
        splitArea: {
          show: false,
          areaStyle: {
            color: chartTheme.gridColor,
          },
        },
    },
    timeAxis: {
        axisLine: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisTick: {
          show: true,
          lineStyle: {
            color: chartTheme.borderColor,
          },
        },
        axisLabel: {
          show: true,
          color: chartTheme.textColorSecondary,
        },
        splitLine: {
          show: true,
          lineStyle: {
            color: chartTheme.gridColor,
            type: 'dashed',
          },
        },
        splitArea: {
          show: false,
          areaStyle: {
            color: chartTheme.gridColor,
          },
        },
    },
    toolbox: {
        iconStyle: {
          borderColor: chartTheme.textColorSecondary,
        },
        emphasis: {
          iconStyle: {
            borderColor: chartTheme.accentCyan,
          },
        },
    },
    legend: {
        textStyle: {
          color: chartTheme.textColorSecondary,
        },
    },
    tooltip: {
        backgroundColor: chartTheme.tooltipBackgroundColor,
        borderColor: chartTheme.borderColor,
        borderWidth: 1,
        textStyle: {
          color: chartTheme.textColor,
        },
    },
    timeline: {
        lineStyle: {
          color: chartTheme.borderColor,
          width: 1,
        },
        itemStyle: {
          color: chartTheme.accentCyan,
          borderWidth: 1,
        },
        label: {
          color: chartTheme.textColorSecondary,
        },
        controlStyle: {
          color: chartTheme.accentCyan,
          borderColor: chartTheme.borderColor,
          borderWidth: 0.5,
        },
        checkpointStyle: {
          color: chartTheme.accentCyan,
          borderColor: chartTheme.borderColor,
        },
        emphasis: {
          itemStyle: {
            color: chartTheme.accentCyan,
          },
          label: {
            color: chartTheme.textColorSecondary,
          },
          controlStyle: {
            color: chartTheme.accentCyan,
            borderColor: chartTheme.borderColor,
            borderWidth: 0.5,
          },
        },
    },
    visualMap: {
        color: [chartTheme.accentCyan, chartTheme.accentCyanDim],
        textStyle: {
          color: chartTheme.textColorSecondary,
        },
    },
    dataZoom: {
        backgroundColor: 'rgba(255,255,255,0)',
        dataBackgroundColor: 'rgba(255,255,255,0.1)',
        fillerColor: 'rgba(0,212,255,0.2)',
        handleColor: chartTheme.accentCyan,
        handleSize: '100%',
        textStyle: {
          color: chartTheme.textColorSecondary,
        },
    },
    markPoint: {
        label: {
          color: chartTheme.textColor,
        },
        emphasis: {
          label: {
            color: chartTheme.textColor,
          },
        },
    },
    markLine: {
        lineStyle: {
          width: 1,
          color: chartTheme.borderColor,
        },
        label: {
          color: chartTheme.textColor,
        },
        emphasis: {
          label: {
            color: chartTheme.textColor,
          },
        },
    },
    markArea: {
        itemStyle: {
          color: chartTheme.borderColor,
        },
        label: {
          color: chartTheme.textColor,
        },
        emphasis: {
          label: {
            color: chartTheme.textColor,
          },
        },
    },
  })
}
