import {
  Bar,
  ComposedChart,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { IntradayPoint, SignalItem } from '../api'

function timeToMinutes(t: string) {
  const [h, m] = t.split(':').map(Number)
  return h * 60 + m
}

function nearestTime(target: string, times: string[]) {
  if (times.length === 0) return target
  const tm = timeToMinutes(target)
  return times.reduce((best, cur) =>
    Math.abs(timeToMinutes(cur) - tm) < Math.abs(timeToMinutes(best) - tm) ? cur : best,
  )
}

/** 分时图: 价格 + VWAP 与成交量上下分开展示, 支持标注买卖信号. */
export function IntradayChart({
  data,
  prevClose,
  signals,
  priceHeight = 220,
  volHeight = 100,
}: {
  data: IntradayPoint[]
  prevClose?: number
  signals?: SignalItem[]
  priceHeight?: number
  volHeight?: number
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
  const timeKeys = chartData.map((d) => d.time)
  const signalDots =
    signals?.map((sig) => ({
      ...sig,
      time: nearestTime(sig.time, timeKeys),
    })) ?? []
  const totalHeight = priceHeight + volHeight + 8
  return (
    <div style={{ height: totalHeight }}>
      <ResponsiveContainer width="100%" height={priceHeight}>
        <LineChart data={chartData}>
          <XAxis dataKey="time" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={30} />
          <YAxis
            yAxisId="price"
            domain={['auto', 'auto']}
            tick={{ fill: '#8b949e', fontSize: 10 }}
            orientation="right"
          />
          <Tooltip
            contentStyle={{ background: '#161b22', border: '1px solid #30363d' }}
            labelStyle={{ color: '#8b949e' }}
          />
          {prevClose != null && (
            <ReferenceLine
              yAxisId="price"
              y={prevClose}
              stroke="#8b949e"
              strokeDasharray="4 4"
              label={{ value: '昨收', fill: '#8b949e', fontSize: 10, position: 'insideTopLeft' }}
            />
          )}
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
          {signalDots.map((sig, i) => {
            const isBuy = sig.side === 'buy'
            const color = isBuy ? '#e54545' : '#26a69a'
            return (
              <ReferenceDot
                key={i}
                x={sig.time}
                y={sig.price}
                yAxisId="price"
                r={5}
                fill={sig.executed ? color : 'transparent'}
                stroke={color}
                strokeDasharray={sig.executed ? undefined : '2 2'}
                label={{
                  value: isBuy ? '买' : '卖',
                  fill: '#fff',
                  fontSize: 9,
                  position: 'top',
                }}
              />
            )
          })}
        </LineChart>
      </ResponsiveContainer>

      <ResponsiveContainer width="100%" height={volHeight}>
        <ComposedChart data={chartData}>
          <XAxis dataKey="time" tick={{ fill: '#8b949e', fontSize: 10 }} minTickGap={30} />
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
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
