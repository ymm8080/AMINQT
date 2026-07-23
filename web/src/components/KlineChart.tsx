import { useEffect, useRef } from 'react'
import { createChart, ColorType, IChartApi } from 'lightweight-charts'
import type { OhlcBar } from '../api'

/** K线图 (lightweight-charts candlestick). */
export function KlineChart({ data, height = 380 }: { data: OhlcBar[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#8b949e',
      },
      grid: {
        vertLines: { color: '#21262d' },
        horzLines: { color: '#21262d' },
      },
      timeScale: { borderColor: '#30363d' },
    })
    chartRef.current = chart
    const series = chart.addCandlestickSeries({
      upColor: '#e54545',
      downColor: '#26a69a',
      borderUpColor: '#e54545',
      borderDownColor: '#26a69a',
      wickUpColor: '#e54545',
      wickDownColor: '#26a69a',
    })
    series.setData(
      data.map((b) => ({
        time: b.date.split(' ')[0] as string,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })) as never,
    )
    chart.timeScale().fitContent()
    const onResize = () => chart.applyOptions({ width: ref.current!.clientWidth })
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.remove()
    }
  }, [data, height])

  return <div ref={ref} />
}
