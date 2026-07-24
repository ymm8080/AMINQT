import { useEffect, useState } from 'react'
import { api, LatestList } from '../api'
import { KlineChart } from '../components/KlineChart'

const pct = (v: number) => (
  <span className={v >= 0 ? 'up' : 'down'}>{(v * 100).toFixed(2)}%</span>
)

/** 选股页: V3.5 清单 + K线详情 + 关注. */
export function SelectionPage() {
  const [data, setData] = useState<LatestList | null>(null)
  const [error, setError] = useState('')
  const [detail, setDetail] = useState<string | null>(null)
  const [ohlc, setOhlc] = useState<import('../api').OhlcBar[]>([])
  const [watched, setWatched] = useState<Set<string>>(new Set())

  useEffect(() => {
    api.latestList().then(setData).catch((e) => setError(String(e)))
    api.watchlist().then((r) => setWatched(new Set(r.items.map((i) => i.symbol)))).catch(() => {})
  }, [])

  useEffect(() => {
    if (detail) api.ohlc(detail).then((r) => setOhlc(r.items)).catch(() => setOhlc([]))
  }, [detail])

  if (error) return <div className="panel">API 错误: {error} (确认 uvicorn app.main:app 已启动)</div>
  if (!data) return <div className="panel">加载中…</div>

  return (
    <>
      <h2>
        选股池 · Pipeline 1 (V3.5)
        {data.demo && <span className="badge">演示数据</span>}
      </h2>
      <p className="dim">
        清单日期 {data.date} · schema {data.schema_version} · Top {data.items.length}
      </p>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>代码</th><th>行业</th><th>评分</th><th>概率</th>
              <th>1日</th><th>3日</th><th>5日</th><th>动量</th><th>冲突</th><th>关注</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((it) => (
              <tr key={it.symbol} onClick={() => setDetail(it.symbol)}>
                <td>{it.symbol}</td>
                <td className="dim">{it.industry ?? '-'}</td>
                <td>{it.score.toFixed(4)}</td>
                <td>{it.prob_up.toFixed(3)}</td>
                <td>{pct(it.pred_ret_1d)}</td>
                <td>{pct(it.pred_ret_3d)}</td>
                <td>{pct(it.pred_ret_5d)}</td>
                <td>{it.momentum}</td>
                <td>{it.signal_conflict ? '⚠' : ''}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={() =>
                      api.toggleWatch(it.symbol, it.name ?? '').then((r) => {
                        setWatched((w) => {
                          const n = new Set(w)
                          r.watched ? n.add(it.symbol) : n.delete(it.symbol)
                          return n
                        })
                      })
                    }
                  >
                    {watched.has(it.symbol) ? '⭐' : '☆'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {detail && (
        <div className="panel">
          <h3>{detail} 日K</h3>
          <KlineChart data={ohlc} />
        </div>
      )}
    </>
  )
}
