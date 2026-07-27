import { useEffect, useMemo, useState } from 'react'
import { api, type ListItem, type OhlcBar, type IntradayPoint, type SectorItem } from '../api'
import { ChipDistributionChart } from '../components/ChipDistributionChart'
import {
  ChipControlChart,
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

function QualityBadge({ predRun }: { predRun: Record<string, unknown> | null }) {
  if (!predRun?.stocks) return null
  const stocks = predRun.stocks as Record<string, unknown>[]
  const withOutcome = stocks.filter((s) => s.actual_ret_1d != null)
  if (!withOutcome.length) return <span className="dim">待收盘回填实际收益</span>
  const dirCorrect = withOutcome.filter((s) => s.direction_correct_1d === 1).length
  const dirAcc = dirCorrect / withOutcome.length
  const biases = withOutcome.map((s) => s.pred_error_1d as number).filter((v) => !isNaN(v))
  const mae = biases.length ? biases.reduce((a, b) => a + Math.abs(b), 0) / biases.length : 0
  const bias = biases.length ? biases.reduce((a, b) => a + b, 0) / biases.length : 0
  return (
    <span style={{ fontSize: 13 }}>
      🎯 方向准确率 <b style={{ color: dirAcc >= 0.5 ? '#2a2' : '#e44' }}>{(dirAcc * 100).toFixed(0)}%</b>
      {' · '}MAE <b>{mae.toFixed(4)}</b>
      {' · '}BIAS <b style={{ color: Math.abs(bias) > 0.005 ? '#e44' : '#2a2' }}>{bias >= 0 ? '+' : ''}{bias.toFixed(4)}</b>
      {' · '}N={<b>{withOutcome.length}</b>}
    </span>
  )
}

export function SelectionPage({ onJumpToTrading }: { onJumpToTrading?: (symbol: string) => void }) {
  type PoolData = { date: string; demo: boolean; schema_version: string; items: ListItem[] } | null
  const [data, setData] = useState<PoolData>(null)
  const [error, setError] = useState('')
  const [detailIdx, setDetailIdx] = useState(0)
  const [ohlc, setOhlc] = useState<OhlcBar[]>([])
  const [intraday, setIntraday] = useState<IntradayPoint[]>([])
  const [priority, setPriority] = useState<Set<string>>(new Set())
  const [sectors, setSectors] = useState<SectorItem[]>([])
  const [addSymbol, setAddSymbol] = useState('')
  const [tab, setTab] = useState<'kline' | 'intraday'>('kline')
  // 历史日期
  const [runs, setRuns] = useState<{ date: string; n_stocks: number }[]>([])
  const [selectedDate, setSelectedDate] = useState('')
  const [predRun, setPredRun] = useState<Record<string, unknown> | null>(null)
  const [histLoading, setHistLoading] = useState(false)

  useEffect(() => {
    api.latestList()
      .then((r) => { setData(r); if (!selectedDate) setSelectedDate(r.date) })
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    api.priority().then((r) => setPriority(new Set(r.symbols))).catch(() => {})
    api.sectors().then((r) => setSectors(r.items)).catch(() => {})
    // 加载历史日期列表
    api.predictionRuns().then((r) => {
      if (r.runs?.length) {
        setRuns(r.runs)
        // 默认最近一次预测日期
        if (!selectedDate) setSelectedDate(r.runs[0].date)
      }
    }).catch(() => {})
  }, [])

  // 切换日期 → 加载历史预测
  useEffect(() => {
    if (!selectedDate) return
    // 如果是当前最新清单日期, 用 latestList 数据
    if (data && selectedDate === data.date) {
      setPredRun(null)
      return
    }
    setHistLoading(true)
    api.predictionRun(selectedDate)
      .then((r) => setPredRun(r))
      .catch(() => setPredRun(null))
      .finally(() => setHistLoading(false))
  }, [selectedDate, data?.date])

  const items = useMemo(() => {
    if (!data) return []
    const listSyms = new Set(data.items.map((it) => it.symbol))
    // 合并 priority.json 中不在清单里的股票
    const extra: ListItem[] = [...priority]
      .filter((s) => !listSyms.has(s))
      .map((s) => ({
        symbol: s, name: s, board: '-', industry: '-',
        day_change: 0, pred_ret_1d: 0, pred_ret_3d: 0, pred_ret_5d: 0,
        prob_up: 0.5, momentum: '-', consensus_score: 0, signal_conflict: 0,
        market_state: '-', score: 0, schema_version: data.schema_version,
        priority: true, added_at: 0,
      }))
    return [...data.items, ...extra]
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


  const priceDomain = useMemo<[number, number] | undefined>(() => {
    if (ohlc.length === 0) return undefined
    const minLow = Math.min(...ohlc.map((d) => d.low))
    const maxHigh = Math.max(...ohlc.map((d) => d.high))
    const padding = (maxHigh - minLow) * 0.05
    return [minLow - padding, maxHigh + padding]
  }, [ohlc])

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
    if (priority.has(sym)) return
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

      {/* ── 日期选择 + 预测质量 ── */}
      <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <label style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>📅 预测日期:</label>
        <select
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          style={{ width: 160 }}
        >
          {runs.length === 0 && data.date && (
            <option value={data.date}>{data.date}</option>
          )}
          {runs.map((r) => (
            <option key={r.date} value={r.date}>{r.date} ({r.n_stocks}只)</option>
          ))}
        </select>
        {histLoading && <span className="dim">加载中...</span>}
        <QualityBadge predRun={predRun} />
      </div>

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
                  <td>{it.score.toFixed(2)}</td>
                  <td>{it.prob_up.toFixed(2)}</td>
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
                      MA{w}: <span style={{ color: '#1f2328' }}>{value.toFixed(2)}</span>
                    </span>
                  )
                })}
              </div>
              <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <KlineChart data={ohlc} showMaLines={true} priceDomain={priceDomain} />
                  <MacdChart data={ohlc} />
                  <MainForceChipsChart data={ohlc} />
                  <ChipControlChart data={ohlc} />
                  <TrendTopBottomChart data={ohlc} />
                </div>
                <div style={{ width: 160, flexShrink: 0 }}>
                  <ChipDistributionChart data={ohlc} priceDomain={priceDomain} height={420} />
                </div>
              </div>
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
