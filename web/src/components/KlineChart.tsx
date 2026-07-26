import {
  Bar,
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
    return row
  })

  const maxVol = Math.max(...data.map((d) => d.volume), 1)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} syncId={syncId}>
        <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={40} />
        <YAxis
          yAxisId="price"
          tick={{ fill: '#8b949e', fontSize: 10 }}
          domain={priceDomain ?? ['auto', 'auto']}
          orientation="right"
          width={55}
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
                  `开:${p.open} 高:${p.high} 低:${p.low} 收:${p.close}`,
                  'K线',
                ]
              }
            }
            if (name === '成交量') {
              return [Number(value).toLocaleString(), name]
            }
            return [String(value), name]
          }}
        />
        <Bar yAxisId="price" dataKey="range" shape={candleShape} name="K线" isAnimationActive={false} />
        <Bar yAxisId="volume" dataKey="volume" fill="#4f8ef755" name="成交量" isAnimationActive={false} />
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
      </ComposedChart>
    </ResponsiveContainer>
  )
}
