import {
  Bar,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { OhlcBar } from '../api'

function ema(arr: number[], span: number): number[] {
  const k = 2 / (span + 1)
  const out: number[] = []
  arr.forEach((v, i) => {
    out.push(i === 0 ? v : v * k + out[i - 1] * (1 - k))
  })
  return out
}

function sma(arr: number[], window: number): number[] {
  const out: number[] = []
  for (let i = 0; i < arr.length; i++) {
    if (i < window - 1) {
      out.push(arr[i])
    } else {
      out.push(arr.slice(i - window + 1, i + 1).reduce((s, x) => s + x, 0) / window)
    }
  }
  return out
}

function absArr(arr: number[]): number[] {
  return arr.map((v) => Math.abs(v))
}

function rollMax(arr: number[], window: number): number[] {
  return arr.map((_, i) => {
    if (i < window - 1) return Math.max(...arr.slice(0, i + 1))
    return Math.max(...arr.slice(i - window + 1, i + 1))
  })
}

function rollMin(arr: number[], window: number): number[] {
  return arr.map((_, i) => {
    if (i < window - 1) return Math.min(...arr.slice(0, i + 1))
    return Math.min(...arr.slice(i - window + 1, i + 1))
  })
}

/** 1. 主力筹码指标: 主力轨迹 + 主力平均线. */
export function MainForceChipsChart({ data, height = 200 }: { data: OhlcBar[]; height?: number }) {
  const close = data.map((d) => d.close)
  const mtm = close.map((v, i) => (i === 0 ? 0 : v - close[i - 1]))
  const mainTraj = ema(ema(mtm, 9), 9).map((v, i) => {
    const denom = ema(ema(absArr(mtm), 9), 9)[i]
    return denom === 0 ? 0 : 100 * v / denom
  })
  const mainAvg = sma(mainTraj, 5)

  const chartData = data.map((d, i) => ({
    date: d.date,
    mainTraj: mainTraj[i],
    mainAvg: mainAvg[i],
  }))

  return (
    <div className="panel" style={{ padding: '12px 0' }}>
      <h4 style={{ margin: '0 0 8px' }}>主力筹码指标</h4>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} syncId="stock-detail">
          <XAxis dataKey="date" tick={{ fill: '#636c76', fontSize: 10 }} minTickGap={40} />
          <YAxis tick={{ fill: '#636c76', fontSize: 10 }} orientation="right" width={55} />
          <Tooltip
            contentStyle={{ background: '#ffffff', border: '1px solid #30363d' }}
            labelStyle={{ color: '#636c76' }}
            formatter={(value: unknown, name: string) => [Number(value).toFixed(2), name]}
          />
          <ReferenceLine y={0} stroke="#636c76" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="mainTraj" stroke="#e6edf3" dot={false} strokeWidth={1.2} name="主力轨迹" />
          <Line type="monotone" dataKey="mainAvg" stroke="#facc15" dot={false} strokeWidth={1.2} name="主力平均" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 2. 主力筹码控盘程度 N. */
export function ChipControlChart({ data, height = 200 }: { data: OhlcBar[]; height?: number }) {
  const close = data.map((d) => d.close)
  const open = data.map((d) => d.open)
  const high = data.map((d) => d.high)
  const low = data.map((d) => d.low)
  const a01 = close.map((c, i) => (c + open[i] + low[i] + high[i]) / 4)
  const hh = rollMax(high, 30)
  const ll = rollMin(low, 30)

  const a04 = a01.map((v, i) => {
    const denom = hh[i] - ll[i]
    return denom === 0 ? 0 : 100 * (v - ll[i]) / denom
  })
  const a02 = a01.map((v, i) => {
    const denom = hh[i] - ll[i]
    return denom === 0 ? 0 : 100 * (v * 1.04 - ll[i]) / denom
  })
  const a06 = a02.map((v) => 100 - v)
  const a08 = a02.map((v, i) => v - a04[i])

  const chartData = data.map((d, i) => ({
    date: d.date,
    a04: a04[i],
    a02: a02[i],
    a06: a06[i],
    a08: a08[i],
  }))

  return (
    <div className="panel" style={{ padding: '12px 0' }}>
      <h4 style={{ margin: '0 0 8px' }}>主力筹码控盘程度 N</h4>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} syncId="stock-detail">
          <XAxis dataKey="date" tick={{ fill: '#636c76', fontSize: 10 }} minTickGap={40} />
          <YAxis domain={[0, 100]} tick={{ fill: '#636c76', fontSize: 10 }} orientation="right" width={55} />
          <Tooltip
            contentStyle={{ background: '#ffffff', border: '1px solid #30363d' }}
            labelStyle={{ color: '#636c76' }}
            formatter={(value: unknown, name: string) => [Number(value).toFixed(2), name]}
          />
          <ReferenceLine y={50} stroke="#636c76" strokeDasharray="4 4" />
          <Bar dataKey="a04" fill="#e54545" name="获利盘(近)" />
          <Bar dataKey="a02" fill="#26a69a" name="获利盘(远)" />
          <Bar dataKey="a08" fill="#facc15" name="筹码差" />
          <Line type="monotone" dataKey="a06" stroke="#00ffff" dot={false} strokeWidth={1} name="套牢盘" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/** 3. 同花顺益盟趋势顶底. */
export function TrendTopBottomChart({ data, height = 200 }: { data: OhlcBar[]; height?: number }) {
  const close = data.map((d) => d.close)
  const high = data.map((d) => d.high)
  const low = data.map((d) => d.low)

  const hh14 = rollMax(high, 14)
  const ll14 = rollMin(low, 14)
  const b = close.map((c, i) => {
    const denom = hh14[i] - ll14[i]
    return denom === 0 ? 0 : 100 * (c - hh14[i]) / denom
  })
  const shortLine = b.map((v) => v + 100)

  const hh34 = rollMax(high, 34)
  const ll34 = rollMin(low, 34)
  const raw34 = close.map((c, i) => {
    const denom = hh34[i] - ll34[i]
    return denom === 0 ? 0 : 100 * (c - hh34[i]) / denom
  })
  const midLine = ema(raw34, 4).map((v) => v + 100)
  const longLine = sma(raw34, 19).map((v) => v + 100)

  const chartData = data.map((d, i) => ({
    date: d.date,
    short: shortLine[i],
    mid: midLine[i],
    long: longLine[i],
  }))

  return (
    <div className="panel" style={{ padding: '12px 0' }}>
      <h4 style={{ margin: '0 0 8px' }}>趋势顶底</h4>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} syncId="stock-detail">
          <XAxis dataKey="date" tick={{ fill: '#636c76', fontSize: 10 }} minTickGap={40} />
          <YAxis domain={[0, 100]} tick={{ fill: '#636c76', fontSize: 10 }} orientation="right" width={55} />
          <Tooltip
            contentStyle={{ background: '#ffffff', border: '1px solid #30363d' }}
            labelStyle={{ color: '#636c76' }}
            formatter={(value: unknown, name: string) => [Number(value).toFixed(2), name]}
          />
          <ReferenceLine y={20} stroke="#26a69a" strokeDasharray="4 4" />
          <ReferenceLine y={80} stroke="#26a69a" strokeDasharray="4 4" />
          <ReferenceLine y={90} stroke="#e54545" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="short" stroke="#e54545" dot={false} strokeWidth={1} name="短期线" />
          <Line type="monotone" dataKey="mid" stroke="#26a69a" dot={false} strokeWidth={2} name="中期线" />
          <Line type="monotone" dataKey="long" stroke="#4f8ef7" dot={false} strokeWidth={1} name="长期线" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
