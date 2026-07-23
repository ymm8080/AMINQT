import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

/** 净值曲线 (recharts area). */
export function EquityChart({
  data,
  height = 300,
}: {
  data: { date: string; nav: number }[]
  height?: number
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data}>
        <CartesianGrid stroke="#21262d" />
        <XAxis dataKey="date" tick={{ fill: '#8b949e', fontSize: 11 }} minTickGap={40} />
        <YAxis
          tick={{ fill: '#8b949e', fontSize: 11 }}
          domain={['auto', 'auto']}
          tickFormatter={(v: number) => (v / 10000).toFixed(0) + '万'}
        />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Area type="monotone" dataKey="nav" stroke="#e54545" fill="#e5454533" name="净值" />
      </AreaChart>
    </ResponsiveContainer>
  )
}
