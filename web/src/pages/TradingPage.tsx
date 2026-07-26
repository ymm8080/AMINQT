import { useEffect, useState } from 'react'
import { api, type OhlcBar, type IntradayPoint, type SectorItem, type SignalItem } from '../api'
import { IntradayChart } from '../components/IntradayChart'
import { Sparkline } from '../components/Sparkline'

const DEMO_NAMES: Record<string, string> = {
  '600519': '贵州茅台', '300750': '宁德时代', '601318': '中国平安',
  '600000': '浦发银行', '000001': '平安银行', '002594': '比亚迪',
  '688981': '中芯国际', '600036': '招商银行', '000858': '五粮液',
  '601899': '紫金矿业',
}

export function TradingPage({ initialSymbol }: { initialSymbol?: string }) {
  const [symbol, setSymbol] = useState(initialSymbol ?? '')
  const [ohlc, setOhlc] = useState<OhlcBar[]>([])
  const [intraday, setIntraday] = useState<IntradayPoint[]>([])
  const [sectors, setSectors] = useState<SectorItem[]>([])
  const [signals, setSignals] = useState<SignalItem[]>([])
  const [autoBuy, setAutoBuy] = useState(false)
  const [autoSell, setAutoSell] = useState(false)
  const [prioritySymbols, setPrioritySymbols] = useState<string[]>([])

  useEffect(() => {
    if (initialSymbol) setSymbol(initialSymbol)
  }, [initialSymbol])

  useEffect(() => {
    if (!symbol) return
    api.ohlc(symbol).then((r) => setOhlc(r.items)).catch(() => setOhlc([]))
    api.intraday(symbol).then((r) => setIntraday(r.items)).catch(() => setIntraday([]))
    api.signals(symbol).then((r) => setSignals(r.items)).catch(() => setSignals([]))
  }, [symbol])

  useEffect(() => {
    api.sectors().then((r) => setSectors(r.items)).catch(() => {})
    api.priority().then((r) => {
      const syms = r.symbols
      setPrioritySymbols(syms)
      setSymbol((prev) => (prev && syms.includes(prev) ? prev : syms[0] ?? ''))
    }).catch(() => setPrioritySymbols([]))
  }, [])

  const last = ohlc[ohlc.length - 1]
  const first = ohlc[0]
  const change = last && first ? last.close / first.close - 1 : 0

  return (
    <>
      <h2>交易看板 · Pipeline 2</h2>

      <div className="panel" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <div>
          <strong>状态:</strong>{' '}
          <span className={autoBuy || autoSell ? 'up' : 'dim'}>
            {autoBuy && autoSell ? '全自动' : autoBuy ? '仅自动买入' : autoSell ? '仅自动卖出' : '手动'}
          </span>
        </div>
        <button className={autoBuy ? 'primary' : ''} onClick={() => setAutoBuy((v) => !v)}>
          {autoBuy ? '⏹ 停止买入' : '▶️ 启动买入'}
        </button>
        <button className={autoSell ? 'primary' : ''} onClick={() => setAutoSell((v) => !v)}>
          {autoSell ? '⏹ 停止卖出' : '▶️ 启动卖出'}
        </button>
        <button onClick={() => { setAutoBuy(false); setAutoSell(false) }}>⏹ 全部停止</button>
      </div>

      <div className="panel grid grid-2">
        <div>
          <label>标的（选股看板日内买入标记股）</label>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)} disabled={prioritySymbols.length === 0}>
            {prioritySymbols.length > 0 ? (
              prioritySymbols.map((s) => (
                <option key={s} value={s}>{s} {DEMO_NAMES[s] ?? ''}</option>
              ))
            ) : (
              <option value="">暂无日内买入标的</option>
            )}
          </select>
          {prioritySymbols.length === 0 && <p className="dim">请先在选股看板标记“日内买入”股票。</p>}
          <div style={{ marginTop: 12, fontSize: 24, fontWeight: 700 }}>
            {last?.close.toFixed(2)}{' '}
            <span className={change >= 0 ? 'up' : 'down'}>{(change * 100).toFixed(2)}%</span>
          </div>
          <p className="dim">最新价 / 相对首根K线</p>
        </div>
        <div className="panel">
          <h4>信号列表</h4>
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>方向</th>
                <th>价格</th>
                <th>数量</th>
                <th>级别</th>
                <th>状态</th>
                <th>原因</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s) => (
                <tr key={`${s.time}-${s.side}`}>
                  <td>{s.time}</td>
                  <td className={s.side === 'buy' ? 'up' : 'down'}>{s.side === 'buy' ? '买入' : '卖出'}</td>
                  <td>{s.price.toFixed(2)}</td>
                  <td>{s.qty}</td>
                  <td className="dim">{s.priority}</td>
                  <td className={s.executed ? 'up' : 'dim'}>{s.executed ? '已执行' : '未执行'}</td>
                  <td className="dim">{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {signals.length === 0 && <p className="dim">暂无信号</p>}
        </div>
      </div>

      {symbol && (
        <div className="panel">
          <h3>{symbol} 分时</h3>
          <IntradayChart data={intraday} prevClose={first?.close} signals={signals} />
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
