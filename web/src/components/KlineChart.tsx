import { useEffect, useRef, useState } from 'react'
import {
  Bar,
  Brush,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { OhlcBar } from '../api'

const MA_COLORS = ['#facc15', '#4f8ef7', '#a855f7']

/** Custom candlestick shape for recharts Bar (range bar [low, high]). */
function candleShape(props: any) {
  const { x = 0, y = 0, width = 0, height = 0, payload } = props
  if (!payload) return <g />

  const open = payload.open as number
  const close = payload.close as number
  const high = payload.high as number
  const low = payload.low as number

  const isUp = close >= open
  const color = isUp ? '#e54545' : '#26a69a'

  const range = high - low
  const yOpen = range > 0 ? y + ((high - open) / range) * height : y
  const yClose = range > 0 ? y + ((high - close) / range) * height : y

  const bodyTop = Math.min(yOpen, yClose)
  const bodyHeight = Math.max(Math.abs(yClose - yOpen), 1)
  const wickX = x + width / 2
  const bodyWidth = Math.max(width * 0.6, 1)
  const bodyX = x + (width - bodyWidth) / 2

  return (
    <g>
      <line x1={wickX} x2={wickX} y1={y} y2={y + height} stroke={color} strokeWidth={1} />
      <rect x={bodyX} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} />
    </g>
  )
}

/** K线图 (recharts candlestick) + MA + 成交量, syncId 与副图联动. */
export function KlineChart({
  data,
  height = 420,
  mas = [5, 10, 20],
  showMaLines = true,
  syncId = 'stock-detail',
  priceDomain,
}: {
  data: OhlcBar[]
  height?: number
  mas?: number[]
  showMaLines?: boolean
  syncId?: string
  priceDomain?: [number, number]
}) {
  const chartData = data.map((b, i) => {
    const row: Record<string, unknown> = {
      date: b.date,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume,
      range: [b.low, b.high],
    }
    mas.forEach((w) => {
      if (i >= w - 1) {
        const sum = data.slice(i - w + 1, i + 1).reduce((s, x) => s + x.close, 0)
        row[`ma${w}`] = sum / w
      }
    })
    // 成交量 MA20
    if (i >= 19) {
      const volSum = data.slice(i - 19, i + 1).reduce((s, x) => s + x.volume, 0)
      row.volMa20 = volSum / 20
    }
    return row
  })

  const maxVol = Math.max(...data.map((d) => d.volume), 1)

  // --- 受控 Brush 范围 + 鼠标滚轮缩放 ---
  const [range, setRange] = useState<[number, number]>([0, 0])
  const containerRef = useRef<HTMLDivElement>(null)

  // 数据变化（切换股票）时重置为全范围
  useEffect(() => {
    if (data.length > 0) setRange([0, data.length - 1])
  }, [data])

  // 非被动 wheel 监听，阻止页面滚动并实现缩放
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setRange(([start, end]) => {
        const total = data.length
        if (total === 0) return [0, 0]
        const center = (start + end) / 2
        const currentSpan = end - start
        // deltaY > 0 (向下滚) → 放大（缩小可见范围）
        // deltaY < 0 (向上滚) → 缩小（扩大可见范围）
        const factor = e.deltaY > 0 ? 0.8 : 1.25
        let newSpan = Math.round(currentSpan * factor)
        newSpan = Math.max(10, Math.min(total - 1, newSpan))
        let newStart = Math.round(center - newSpan / 2)
        let newEnd = newStart + newSpan
        if (newStart < 0) {
          newStart = 0
          newEnd = newSpan
        }
        if (newEnd > total - 1) {
          newEnd = total - 1
          newStart = Math.max(0, newEnd - newSpan)
        }
        return [newStart, newEnd]
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [data.length])

  return (
    <div ref={containerRef} style={{ width: '100%', cursor: 'crosshair' }}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} syncId={syncId}>
          <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={40} />
          <YAxis
            yAxisId="price"
            tick={{ fill: '#8b949e', fontSize: 10 }}
            domain={priceDomain ?? ['auto', 'auto']}
            orientation="right"
            width={55}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <YAxis yAxisId="volume" orientation="left" hide width={0} domain={[0, maxVol * 5]} />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
            labelStyle={{ color: '#8b949e' }}
            formatter={(value: unknown, name: string, item: any) => {
              if (name === 'K线' && Array.isArray(value)) {
                const p = item?.payload
                if (p) {
                return [
                  `开:${Number(p.open).toFixed(2)} 高:${Number(p.high).toFixed(2)} 低:${Number(p.low).toFixed(2)} 收:${Number(p.close).toFixed(2)}`,
                  'K线',
                ]
                }
              }
            if (name === '成交量') {
              return [Number(value).toLocaleString(), name]
            }
            if (name === 'VolMA20') {
              return [Number(value).toLocaleString(), name]
            }
            return [Number(value).toFixed(2), name]
            }}
          />
          <Bar yAxisId="price" dataKey="range" shape={candleShape} name="K线" isAnimationActive={false} />
          <Bar yAxisId="volume" dataKey="volume" fill="#4f8ef755" name="成交量" isAnimationActive={false} />
          <Line
            yAxisId="volume"
            type="monotone"
            dataKey="volMa20"
            stroke="#ff9800"
            dot={false}
            strokeWidth={1}
            connectNulls
            name="VolMA20"
          />
          {showMaLines &&
            mas.map((w, idx) => (
              <Line
                key={w}
                yAxisId="price"
                type="monotone"
                dataKey={`ma${w}`}
                stroke={MA_COLORS[idx % MA_COLORS.length]}
                dot={false}
                strokeWidth={1}
                connectNulls
                name={`MA${w}`}
              />
            ))}
          <Brush
            dataKey="close"
            height={30}
            stroke="#4f8ef7"
            fill="#161b22"
            travellerWidth={8}
            startIndex={range[0]}
            endIndex={range[1]}
            onChange={(e: { startIndex?: number; endIndex?: number }) => {
              if (e.startIndex != null && e.endIndex != null) {
                setRange([e.startIndex, e.endIndex])
              }
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
