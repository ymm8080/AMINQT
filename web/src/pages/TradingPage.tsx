import { useEffect, useMemo, useState } from 'react'
import { api, type OhlcBar, type IntradayPoint } from '../api'
import { IntradayChart } from '../components/IntradayChart'
import { KlineChart } from '../components/KlineChart'

const DEMO_SYMBOLS = [
  '600519', '300750', '601318', '600000', '000001',
  '002594', '688981', '600036', '000858', '601899',
]
const DEMO_NAMES: Record<string, string> = {
  '600519': '贵州茅台', '300750': '宁德时代', '601318': '中国平安',
  '600000': '浦发银行', '000001': '平安银行', '002594': '比亚迪',
  '688981': '中芯国际', '600036': '招商银行', '000858': '五粮液',
  '601899': '紫金矿业',
}

export function TradingPage({ initialSymbol }: { initialSymbol?: string }) {
  const [symbol, setSymbol] = useState(initialSymbol ?? DEMO_SYMBOLS[0])
  const [ohlc, setOhlc] = useState<OhlcBar[]>([])
  const [intraday, setIntraday] = useState<IntradayPoint[]>([])
  const [autoBuy, setAutoBuy] = useState(false)
  const [autoSell, setAutoSell] = useState(false)

  useEffect(() => {
    if (initialSymbol) setSymbol(initialSymbol)
  }, [initialSymbol])

  useEffect(() => {
    api.ohlc(symbol).then((r) => setOhlc(r.items)).catch(() => setOhlc([]))
    api.intraday(symbol).then((r) => setIntraday(r.items)).catch(() => setIntraday([]))
  }, [symbol])

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
          <label>标的</label>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {DEMO_SYMBOLS.map((s) => (
              <option key={s} value={s}>{s} {DEMO_NAMES[s]}</option>
            ))}
          </select>
          <div style={{ marginTop: 12, fontSize: 24, fontWeight: 700 }}>
            {last?.close.toFixed(2)}{' '}
            <span className={change >= 0 ? 'up' : 'down'}>{(change * 100).toFixed(2)}%</span>
          </div>
          <p className="dim">最新价 / 相对首根K线</p>
        </div>
        <div className="panel">
          <h4>信号列表</h4>
          <p className="dim">演示: 09:44 600519 买入 | 10:12 300750 卖出</p>
        </div>
      </div>

      <div className="panel">
        <h3>{symbol} 日K</h3>
        <KlineChart data={ohlc} height={480} />
      </div>

      <div className="panel">
        <h3>{symbol} 分时</h3>
        <IntradayChart data={intraday} prevClose={first?.close} />
      </div>
    </>
  )
}
