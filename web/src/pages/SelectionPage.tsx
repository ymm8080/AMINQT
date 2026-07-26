import { useEffect, useMemo, useState } from 'react'
import { api, type ListItem, type OhlcBar, type IntradayPoint, type SectorItem } from '../api'
import { ChipDistributionChart } from '../components/ChipDistributionChart'
import {
  ChipControlChart,
  FindBullChart,
  MainForceChipsChart,
  TrendTopBottomChart,
} from '../components/IndicatorCharts'
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
  const [priceRange, setPriceRange] = useState<{ min: number; max: number } | undefined>(undefined)
  const [intraday, setIntraday] = useState<IntradayPoint[]>([])
  const [priority, setPriority] = useState<Set<string>>(new Set())
  const [sectors, setSectors] = useState<SectorItem[]>([])
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
  }, [])

  const items = useMemo(() => {
    if (!data) return []
    return data.items
      .map((it) => ({ ...it, priority: priority.has(it.symbol) }))
      .sort((a, b) => {
        // 1. 日内买入指标优先
        if (a.priority && !b.priority) return -1
        if (!a.priority && b.priority) return 1
        // 2. 同一优先级内：新插入的排在最前
        const aAdded = a.added_at ?? 0
        const bAdded = b.added_at ?? 0
        if (aAdded !== bAdded) return bAdded - aAdded
        // 3. 再按评分降序
        return b.score - a.score
      })
  }, [data, priority])

  const detail = items[detailIdx] ?? null

  useEffect(() => {
    if (!detail) return
    api.ohlc(detail.symbol).then((r) => setOhlc(r.items)).catch(() => setOhlc([]))
    api.intraday(detail.symbol).then((r) => setIntraday(r.items)).catch(() => setIntraday([]))
  }, [detail])

  useEffect(() => {
    api.priority().then((r) => setPriority(new Set(r.symbols))).catch(() => {})
    api.sectors().then((r) => setSectors(r.items)).catch(() => {})
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
      day_change: 0,
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
      added_at: Date.now(),
    }
    setData({ ...data, items: [row, ...data.items] })
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
              <th>当日涨跌幅</th>
              <th>日内走势</th>
              <th>板块</th>
              <th>评分</th>
              <th>概率</th>
              <th>1日</th>
              <th>3日</th>
              <th>5日</th>
              <th>日内买入</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, idx) => {
              const sector = sectors.find((s) => s.板块 === it.industry)
              return (
                <tr
                  key={it.symbol}
                  className={idx === detailIdx ? 'selected-row' : ''}
                  onClick={() => setDetailIdx(idx)}
                >
                  <td>{it.symbol}</td>
                  <td className="dim">{it.name ?? '-'}</td>
                  <td>{pct(it.day_change ?? 0)}</td>
                  <td>
                    <IntradayMini symbol={it.symbol} />
                  </td>
                  <td>
                    {sector ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span className="dim">{it.industry}</span>
                        <span className={sector.涨跌幅 >= 0 ? 'up' : 'down'}>
                          {(sector.涨跌幅 * 100).toFixed(2)}%
                        </span>
                        <div style={{ width: 80 }}>
                          <Sparkline data={sector.intraday} />
                        </div>
                      </div>
                    ) : (
                      <span className="dim">{it.industry ?? '-'}</span>
                    )}
                  </td>
                  <td>{it.score.toFixed(4)}</td>
                  <td>{it.prob_up.toFixed(3)}</td>
                  <td>{pct(it.pred_ret_1d)}</td>
                  <td>{pct(it.pred_ret_3d)}</td>
                  <td>{pct(it.pred_ret_5d)}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <button onClick={() => togglePriority(it.symbol, it.name)}>
                      {it.priority ? '✅' : '⬜'}
                    </button>
                  </td>
                </tr>
              )
            })}
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
              <div style={{ display: 'flex', gap: 16, marginBottom: 12, fontSize: 13 }}>
                {[5, 10, 20].map((w) => {
                  if (ohlc.length < w) return null
                  const value = ohlc.slice(-w).reduce((s, b) => s + b.close, 0) / w
                  return (
                    <span key={w} className="dim">
                      MA{w}: <span style={{ color: '#e6edf3' }}>{value.toFixed(2)}</span>
                    </span>
                  )
                })}
              </div>
              <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
                <div style={{ flex: 1 }}>
                  <KlineChart data={ohlc} showMaLines={false} onPriceRangeChange={setPriceRange} />
                </div>
                <div style={{ width: 320, flexShrink: 0 }}>
                  <ChipDistributionChart data={ohlc} priceRange={priceRange} height={420} />
                </div>
              </div>
              <MacdChart data={ohlc} />
              <MainForceChipsChart data={ohlc} />
              <ChipControlChart data={ohlc} />
              <FindBullChart data={ohlc} />
              <TrendTopBottomChart data={ohlc} />
            </>
          ) : (
            <IntradayChart data={intraday} prevClose={ohlc[ohlc.length - 1]?.close} />
          )}
        </div>
      )}
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
