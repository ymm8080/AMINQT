import {
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'

/** 表格内迷你走势 sparkline (SVG). */
export function Sparkline({ data, height = 32, color = '#4f8ef7' }: { data: number[]; height?: number; color?: string }) {
  if (!data.length) return <span>-</span>
  const width = 100
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - ((v - min) / range) * height
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        points={points}
      />
    </svg>
  )
}
