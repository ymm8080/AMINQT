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

/** MACD 副图 (DIF/DEA/BAR). */
export function MacdChart({ data, height = 200 }: { data: OhlcBar[]; height?: number }) {
  const close = data.map((d) => d.close)
  const ema = (arr: number[], span: number) => {
    const k = 2 / (span + 1)
    const out: number[] = []
    arr.forEach((v, i) => {
      out.push(i === 0 ? v : v * k + out[i - 1] * (1 - k))
    })
    return out
  }
  const ema12 = ema(close, 12)
  const ema26 = ema(close, 26)
  const dif = ema12.map((v, i) => v - ema26[i])
  const dea = ema(dif, 9)
  const bar = dif.map((v, i) => (v - dea[i]) * 2)
  const chartData = data.map((d, i) => ({
    date: d.date,
    dif: dif[i],
    dea: dea[i],
    bar: bar[i],
  }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData}>
        <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={40} />
        <YAxis tick={{ fill: '#8b949e', fontSize: 10 }} />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Bar dataKey="bar" fill="#8884d8" name="MACD" />
        <Line type="monotone" dataKey="dif" stroke="#fff" dot={false} name="DIF" />
        <Line type="monotone" dataKey="dea" stroke="#facc15" dot={false} name="DEA" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
