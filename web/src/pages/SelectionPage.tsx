import { useEffect, useMemo, useState } from 'react'
import { api, type ListItem, type OhlcBar, type IntradayPoint, type SectorItem } from '../api'
import { ChipDistributionChart } from '../components/ChipDistributionChart'
import { IntradayChart } from '../components/IntradayChart'
import { KlineChart } from '../components/KlineChart'
import { MacdChart } from '../components/MacdChart'
import { Sparkline } from '../components/Sparkline'

function pct(v: number) {
  return (
    <span className={v >= 0 ? 'up' : 'down'}>{(v * 100).toFixed(2)}%</span>
  )
}

export function SelectionPage({ onJumpToTrading }: { onJumpToTrading?: (symbol: string) => void }) {
  type PoolData = { date: string; demo: boolean; schema_version: string; items: ListItem[] } | null
  const [data, setData] = useState<PoolData>(null)
  const [error, setError] = useState('')
  const [detailIdx, setDetailIdx] = useState(0)
  const [ohlc, setOhlc] = useState<OhlcBar[]>([])
  const [intraday, setIntraday] = useState<IntradayPoint[]>([])
  const [sectors, setSectors] = useState<SectorItem[]>([])
  const [priority, setPriority] = useState<Set<string>>(new Set())
  const [addSymbol, setAddSymbol] = useState('')
  const [tab, setTab] = useState<'kline' | 'intraday'>('kline')

  useEffect(() => {
    api.latestList()
      .then((r) => {
        setData(r)
        // 同步 priority 状态
        const pri = new Set(r.items.filter((i) => i.priority).map((i) => i.symbol))
        setPriority(pri)
      })
      .catch((e) => setError(String(e)))
    api.sectors().then((r) => setSectors(r.items)).catch(() => {})
  }, [])

  const items = useMemo(() => {
    if (!data) return []
    return data.items.map((it) => ({ ...it, priority: priority.has(it.symbol) }))
  }, [data, priority])

  const detail = items[detailIdx] ?? null

  useEffect(() => {
    if (!detail) return
    api.ohlc(detail.symbol).then((r) => setOhlc(r.items)).catch(() => setOhlc([]))
    api.intraday(detail.symbol).then((r) => setIntraday(r.items)).catch(() => setIntraday([]))
  }, [detail])

  useEffect(() => {
    api.priority().then((r) => setPriority(new Set(r.symbols))).catch(() => {})
  }, [])

  const togglePriority = async (symbol: string, name = '') => {
    const r = await api.togglePriority(symbol, name)
    setPriority((prev) => {
      const n = new Set(prev)
      r.priority ? n.add(symbol) : n.delete(symbol)
      return n
    })
  }

  const addStock = () => {
    const sym = addSymbol.trim()
    if (!sym || !data) return
    if (data.items.some((i) => i.symbol === sym)) return
    const row: ListItem = {
      symbol: sym,
      name: sym,
      board: '-',
      industry: '-',
      pred_ret_1d: 0,
      pred_ret_3d: 0,
      pred_ret_5d: 0,
      prob_up: 0.5,
      momentum: '-',
      consensus_score: 0,
      signal_conflict: 0,
      market_state: '-',
      score: 0,
      schema_version: data.schema_version,
      priority: true,
    }
    setData({ ...data, items: [...data.items, row] })
    setAddSymbol('')
    togglePriority(sym, sym)
  }

  const prev = () => setDetailIdx((i) => Math.max(0, i - 1))
  const next = () => setDetailIdx((i) => Math.min(items.length - 1, i + 1))

  if (error) return <div className="panel">API 错误: {error} (确认 uvicorn app.main:app 已启动)</div>
  if (!data) return <div className="panel">加载中…</div>

  return (
    <>
      <h2>
        选股池 · Pipeline 1 (V3.5)
        {data.demo && <span className="badge">演示数据</span>}
      </h2>
      <p className="dim">
        清单日期 {data.date} · schema {data.schema_version} · Top {items.length}
      </p>

      <div className="panel" style={{ display: 'flex', gap: 8 }}>
        <input
          placeholder="输入股票代码, 如 600519"
          value={addSymbol}
          onChange={(e) => setAddSymbol(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addStock()}
        />
        <button className="primary" onClick={addStock}>➕ 添加</button>
      </div>

      <div className="panel" style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>代码</th>
              <th>名称</th>
              <th>日内走势</th>
              <th>评分</th>
              <th>概率</th>
              <th>1日</th>
              <th>3日</th>
              <th>5日</th>
              <th>行业</th>
              <th>日内买入</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, idx) => (
              <tr
                key={it.symbol}
                className={idx === detailIdx ? 'selected-row' : ''}
                onClick={() => setDetailIdx(idx)}
              >
                <td>{it.symbol}</td>
                <td className="dim">{it.name ?? '-'}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <IntradayMini symbol={it.symbol} />
                </td>
                <td>{it.score.toFixed(4)}</td>
                <td>{it.prob_up.toFixed(3)}</td>
                <td>{pct(it.pred_ret_1d)}</td>
                <td>{pct(it.pred_ret_3d)}</td>
                <td>{pct(it.pred_ret_5d)}</td>
                <td className="dim">{it.industry ?? '-'}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button onClick={() => togglePriority(it.symbol, it.name)}>
                    {it.priority ? '✅' : '⬜'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {detail && (
        <div className="panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>{detail.symbol} {detail.name} 详情</h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={prev}>⬆ 上一个</button>
              <button onClick={next}>⬇ 下一个</button>
              <button onClick={() => togglePriority(detail.symbol, detail.name)}>
                {detail.priority ? '取消日内买入' : '标记日内买入'}
              </button>
              {onJumpToTrading && (
                <button className="primary" onClick={() => onJumpToTrading(detail.symbol)}>
                  📈 去交易看板
                </button>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <button className={tab === 'kline' ? 'primary' : ''} onClick={() => setTab('kline')}>日K</button>
            <button className={tab === 'intraday' ? 'primary' : ''} onClick={() => setTab('intraday')}>分时</button>
          </div>
          {tab === 'kline' ? (
            <>
              <KlineChart data={ohlc} />
              <MacdChart data={ohlc} />
              <div className="grid grid-2">
                <ChipDistributionChart data={ohlc} />
                <div className="panel" style={{ minHeight: 260 }}>
                  <h4>参考指标</h4>
                  <p className="dim">主力筹码 / 控盘 / 牛股 / 趋势顶底 (复刻中)</p>
                </div>
              </div>
            </>
          ) : (
            <IntradayChart data={intraday} prevClose={ohlc[ohlc.length - 1]?.close} />
          )}
        </div>
      )}

      <div className="panel">
        <h3>板块行情</h3>
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
      </div>
    </>
  )
}

/** 日内走势 sparkline (异步加载). */
function IntradayMini({ symbol }: { symbol: string }) {
  const [data, setData] = useState<number[]>([])
  useEffect(() => {
    api.intraday(symbol)
      .then((r) => {
        const p0 = r.items[0]?.price ?? 1
        setData(r.items.map((d) => d.price / p0 - 1))
      })
      .catch(() => setData([]))
  }, [symbol])
  return <Sparkline data={data} color="#4f8ef7" />
}
