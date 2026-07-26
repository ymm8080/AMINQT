import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { OhlcBar } from '../api'

/** 筹码分布近似图 (收盘价分布). */
export function ChipDistributionChart({
  data,
  height = 260,
  priceRange,
}: {
  data: OhlcBar[]
  height?: number
  priceRange?: { min: number; max: number }
}) {
  const close = data.map((d) => d.close)
  const min = priceRange?.min ?? Math.min(...close)
  const max = priceRange?.max ?? Math.max(...close)
  const bins = 20
  const step = (max - min) / bins || 1
  const counts = new Array(bins).fill(0)
  close.forEach((v) => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1)
    counts[idx]++
  })
  const chartData = counts.map((c, i) => ({
    price: (min + (i + 0.5) * step).toFixed(2),
    count: c,
  }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} layout="vertical">
        <XAxis type="number" tick={{ fill: '#8b949e', fontSize: 10 }} />
        <YAxis
          dataKey="price"
          type="category"
          domain={[min, max]}
          tick={{ fill: '#8b949e', fontSize: 10 }}
          width={50}
        />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Bar dataKey="count" fill="#4f8ef7" name="筹码量" />
      </BarChart>
    </ResponsiveContainer>
  )
}
