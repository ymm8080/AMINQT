import { useEffect, useRef } from 'react'
import { createChart, ColorType, IChartApi } from 'lightweight-charts'
import type { OhlcBar } from '../api'

/** K线图 (lightweight-charts candlestick) + MA5/10/20 + 成交量. */
export function KlineChart({
  data,
  height = 420,
  mas = [5, 10, 20],
  showMaLines = true,
  onPriceRangeChange,
}: {
  data: OhlcBar[]
  height?: number
  mas?: number[]
  showMaLines?: boolean
  onPriceRangeChange?: (range: { min: number; max: number }) => void
}) {
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
      rightPriceScale: { borderColor: '#30363d' },
    })
    chartRef.current = chart

    const candle = chart.addCandlestickSeries({
      upColor: '#e54545',
      downColor: '#26a69a',
      borderUpColor: '#e54545',
      borderDownColor: '#26a69a',
      wickUpColor: '#e54545',
      wickDownColor: '#26a69a',
    })
    candle.setData(
      data.map((b) => ({
        time: b.date.split(' ')[0],
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })) as never,
    )

    // 均线
    const colors = ['#facc15', '#4f8ef7', '#a855f7']
    mas.forEach((w, idx) => {
      if (data.length < w) return
      const ma = data.map((b, i) => {
        if (i < w - 1) return null
        const sum = data.slice(i - w + 1, i + 1).reduce((s, x) => s + x.close, 0)
        return { time: b.date.split(' ')[0], value: sum / w }
      }).filter(Boolean) as { time: string; value: number }[]
      if (showMaLines) {
        const line = chart.addLineSeries({
          color: colors[idx % colors.length],
          lineWidth: 2,
          title: `MA${w}`,
          priceScaleId: 'right',
          lastValueVisible: true,
          priceLineVisible: false,
        })
        line.setData(ma as never)
      }
    })

    // 成交量
    const volSeries = chart.addHistogramSeries({
      color: '#4f8ef7',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } })
    volSeries.setData(
      data.map((b) => ({
        time: b.date.split(' ')[0],
        value: b.volume,
        color: b.close >= b.open ? '#e54545' : '#26a69a',
      })) as never,
    )

    chart.timeScale().fitContent()

    const notifyPriceRange = () => {
      if (!onPriceRangeChange || !ref.current) return
      const h = ref.current.clientHeight
      const top = candle.coordinateToPrice(0)
      const bottom = candle.coordinateToPrice(h)
      if (top == null || bottom == null) return
      onPriceRangeChange({
        min: Math.min(Number(top), Number(bottom)),
        max: Math.max(Number(top), Number(bottom)),
      })
    }
    notifyPriceRange()
    chart.timeScale().subscribeVisibleLogicalRangeChange(notifyPriceRange)

    const onResize = () => chart.applyOptions({ width: ref.current!.clientWidth })
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(notifyPriceRange)
      chart.remove()
    }
  }, [data, height, mas, showMaLines])

  return <div ref={ref} />
}
