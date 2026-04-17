import { useState, useRef, useEffect, useCallback } from 'react'
import * as echarts from 'echarts'

interface ChartSize {
  width: number
  height: number
}

export const useChartSize = (
  containerRef: React.RefObject<HTMLElement>,
  chart?: echarts.ECharts | null
): ChartSize => {
  const [size, setSize] = useState<ChartSize>({ width: 0, height: 0 })
  const resizeObserverRef = useRef<ResizeObserver | null>(null)

  const handleResize = useCallback(() => {
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current
      setSize({ width: clientWidth, height: clientHeight })
      
      if (chart) {
        chart.resize()
      }
    }
  }, [containerRef, chart])

  useEffect(() => {
    if (!containerRef.current) return

    resizeObserverRef.current = new ResizeObserver(handleResize)
    resizeObserverRef.current.observe(containerRef.current)

    handleResize()

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
        resizeObserverRef.current = null
      }
    }
  }, [containerRef, handleResize])

  return size
}

export const useChartInstance = () => {
  const chartRef = useRef<echarts.ECharts | null>(null)

  const setChartInstance = useCallback((chart: echarts.ECharts | null) => {
    chartRef.current = chart
  }, [])

  const getChartInstance = useCallback(() => {
    return chartRef.current
  }, [])

  return {
    chartRef,
    setChartInstance,
    getChartInstance,
  }
}

export const useChartAutoResize = (chart: echarts.ECharts | null) => {
  useEffect(() => {
    if (!chart) return

    const handleResize = () => {
      chart.resize()
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [chart])
}
