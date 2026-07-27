import { useEffect, useState } from 'react'
import { api, type SectorItem } from '../api'
import { Sparkline } from '../components/Sparkline'

export function SectorPage() {
  const [sectors, setSectors] = useState<SectorItem[]>([])

  useEffect(() => {
    api.sectors().then((r) => setSectors(r.items)).catch(() => {})
  }, [])

  return (
    <>
      <h2>板块行情</h2>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>板块</th>
              <th>涨跌幅</th>
              <th>日内走势</th>
              <th>上涨家数</th>
              <th>下跌家数</th>
            </tr>
          </thead>
          <tbody>
            {sectors.map((s) => (
              <tr key={s.板块}>
                <td>{s.板块}</td>
                <td className={s.涨跌幅 >= 0 ? 'up' : 'down'}>{(s.涨跌幅 * 100).toFixed(2)}%</td>
                <td><Sparkline data={s.intraday} /></td>
                <td>{s.上涨家数}</td>
                <td>{s.下跌家数}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {sectors.length === 0 && <p className="dim">暂无板块数据</p>}
      </div>
    </>
  )
}
