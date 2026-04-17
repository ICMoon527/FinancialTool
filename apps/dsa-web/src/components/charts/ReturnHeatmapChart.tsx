import React, { useMemo, useState } from 'react'
import type { EChartsOption, VisualMapComponentOption } from 'echarts'
import { BaseChart } from './BaseChart'
import { chartTheme } from './theme'
import { formatPercent } from './utils'

interface MonthlyReturn {
  year: string
  month: string
  return: number
}

interface AnnualReturn {
  year: string
  return: number
}

interface ReturnHeatmapChartProps {
  data: {
    monthly: MonthlyReturn[]
    annual: AnnualReturn[]
  }
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  height?: number | string
  onChartReady?: (chart: any) => void
}

type ViewType = 'monthly' | 'annual'

export const ReturnHeatmapChart: React.FC<ReturnHeatmapChartProps> = ({
  data,
  loading = false,
  error = null,
  onRetry,
  height = 400,
  onChartReady,
}) => {
  const [viewType, setViewType] = useState<ViewType>('monthly')

  const option = useMemo<EChartsOption>(() => {
    if (viewType === 'monthly') {
      const years = Array.from(new Set(data.monthly.map((r) => r.year))).sort()
      const months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

      const heatmapData: [number, number, number][] = []
      const monthMap = {
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
        '7': 6, '8': 7, '9': 8, '10': 9, '11': 10, '12': 11,
      }

      data.monthly.forEach((r) => {
        const yearIndex = years.indexOf(r.year)
        const monthIndex = monthMap[r.month as keyof typeof monthMap]
        if (yearIndex !== -1 && monthIndex !== undefined) {
          heatmapData.push([yearIndex, monthIndex, r.return])
        }
      })

      return {
        tooltip: {
          position: 'top',
          formatter: (params: any) => {
            if (!params) return ''
            const [yearIndex, monthIndex, value] = params.value as [number, number, number]
            return `<div class="font-medium mb-1">${years[yearIndex]}年 ${months[monthIndex]}</div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400">收益率:</span>
                <span class="font-medium ${value >= 0 ? 'text-green-400' : 'text-red-400'}">${formatPercent(value)}</span>
              </div>`
          },
        },
        grid: {
          left: '10%',
          right: '10%',
          top: '15%',
          bottom: '15%',
        },
        xAxis: {
          type: 'category',
          data: years,
          splitArea: {
            show: true,
          },
          axisLabel: {
            color: chartTheme.textColorSecondary,
          },
        },
        yAxis: {
          type: 'category',
          data: months,
          splitArea: {
            show: true,
          },
          axisLabel: {
            color: chartTheme.textColorSecondary,
          },
        },
        visualMap: {
          min: -0.3,
          max: 0.3,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '0%',
          inRange: {
            color: [chartTheme.dangerColor, '#333333', chartTheme.successColor],
          },
          textStyle: {
            color: chartTheme.textColorSecondary,
          },
          formatter: (value: any) => formatPercent(value as number),
        } as VisualMapComponentOption,
        series: [
          {
            name: '月度收益',
            type: 'heatmap',
            data: heatmapData,
            label: {
              show: true,
              formatter: (params: any) => {
                const value = params.value[2] as number
                return formatPercent(value)
              },
              color: '#ffffff',
              fontSize: 10,
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)',
              },
            },
          },
        ],
      }
    } else {
      const years = data.annual.map((r) => r.year)
      const values = data.annual.map((r) => r.return)

      return {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow',
          },
          formatter: (params: any) => {
            if (!Array.isArray(params) || params.length === 0) return ''
            const year = params[0].axisValue
            const value = params[0].value as number
            return `<div class="font-medium mb-1">${year}年</div>
              <div class="flex items-center gap-2">
                <span class="text-gray-400">年化收益率:</span>
                <span class="font-medium ${value >= 0 ? 'text-green-400' : 'text-red-400'}">${formatPercent(value)}</span>
              </div>`
          },
        },
        grid: {
          left: '10%',
          right: '10%',
          top: '15%',
          bottom: '15%',
          containLabel: true,
        },
        xAxis: {
          type: 'category',
          data: years,
          axisLabel: {
            color: chartTheme.textColorSecondary,
          },
        },
        yAxis: {
          type: 'value',
          axisLabel: {
            formatter: (value: number) => formatPercent(value),
            color: chartTheme.textColorSecondary,
          },
          splitLine: {
            lineStyle: {
              color: chartTheme.gridColor,
            },
          },
        },
        visualMap: {
          show: false,
          min: -0.5,
          max: 0.5,
          inRange: {
            color: [chartTheme.dangerColor, chartTheme.successColor],
          },
        },
        series: [
          {
            name: '年度收益',
            type: 'bar',
            data: values.map((value) => ({
              value,
              itemStyle: {
                color: value >= 0 ? chartTheme.successColor : chartTheme.dangerColor,
              },
            })),
            label: {
              show: true,
              position: 'top',
              formatter: (params: any) => formatPercent(params.value as number),
              color: chartTheme.textColor,
              fontSize: 11,
            },
          },
        ],
      }
    }
  }, [data, viewType])

  return (
    <div className="flex flex-col h-full">
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => setViewType('monthly')}
          className={`px-3 py-1 text-xs rounded-md transition-colors ${
            viewType === 'monthly'
              ? 'bg-[#00d4ff] text-black'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          月度视图
        </button>
        <button
          onClick={() => setViewType('annual')}
          className={`px-3 py-1 text-xs rounded-md transition-colors ${
            viewType === 'annual'
              ? 'bg-[#00d4ff] text-black'
              : 'bg-white/10 text-gray-400 hover:bg-white/20'
          }`}
        >
          年度视图
        </button>
      </div>
      <div className="flex-1">
        <BaseChart
          option={option}
          loading={loading}
          error={error}
          onRetry={onRetry}
          height={height}
          onChartReady={onChartReady}
        />
      </div>
    </div>
  )
}
