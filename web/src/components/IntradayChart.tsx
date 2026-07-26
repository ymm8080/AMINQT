import {
  Bar,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { IntradayPoint } from '../api'

/** 分时图: 价格 + VWAP + 成交量. */
export function IntradayChart({
  data,
  prevClose,
  height = 320,
}: {
  data: IntradayPoint[]
  prevClose?: number
  height?: number
}) {
  const chartData = data.map((d, i) => {
    const vwap =
      i === 0
        ? d.price
        : data.slice(0, i + 1).reduce((s, x) => s + x.price * x.volume, 0) /
          data.slice(0, i + 1).reduce((s, x) => s + x.volume, 0)
    const volMa5 =
      i < 4
        ? d.volume
        : data.slice(i - 4, i + 1).reduce((s, x) => s + x.volume, 0) / 5
    return { ...d, vwap, volMa5 }
  })
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData}>
        <XAxis dataKey="time" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={30} />
        <YAxis
          yAxisId="price"
          domain={['auto', 'auto']}
          tick={{ fill: '#8b949e', fontSize: 10 }}
          orientation="right"
        />
        <YAxis
          yAxisId="vol"
          orientation="left"
          tick={{ fill: '#8b949e', fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Bar
          yAxisId="vol"
          dataKey="volume"
          fill="#4f8ef755"
          name="成交量"
        />
        <Line
          yAxisId="vol"
          type="monotone"
          dataKey="volMa5"
          stroke="#ff7f0e"
          dot={false}
          strokeWidth={1}
          name="量MA5"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="price"
          stroke="#1f77b4"
          dot={false}
          strokeWidth={1.5}
          name="价格"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="vwap"
          stroke="#ff7f0e"
          dot={false}
          strokeDasharray="4 4"
          strokeWidth={1}
          name="VWAP"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
