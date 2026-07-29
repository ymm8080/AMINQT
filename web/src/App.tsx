import { useState } from 'react'
import { BacktestPage } from './pages/BacktestPage'
import { ConfigPage } from './pages/ConfigPage'
import { DataPipelinePage } from './pages/DataPipelinePage'
import { SelectionPage } from './pages/SelectionPage'
import { TradingPage } from './pages/TradingPage'

const PAGES: Record<string, (props: Record<string, unknown>) => JSX.Element> = {
  选股看板: SelectionPage,
  交易看板: TradingPage,
  回测中心: BacktestPage,
  数据管道: DataPipelinePage,
  配置中心: ConfigPage,
}

export default function App() {
  const [page, setPage] = useState<keyof typeof PAGES>('选股看板')
  const [tradingSymbol, setTradingSymbol] = useState<string | undefined>(undefined)
  const Page = PAGES[page]

  const jumpToTrading = (symbol: string) => {
    setTradingSymbol(symbol)
    setPage('交易看板')
  }

  return (
    <div className="app">
      <nav>
        <div className="logo">📈 AMINQT</div>
        {Object.keys(PAGES).map((k) => (
          <button key={k} className={k === page ? 'active' : ''} onClick={() => setPage(k as keyof typeof PAGES)}>
            {k}
          </button>
        ))}
        <div style={{ marginTop: 'auto', fontSize: 11, color: '#8b949e' }}>
          Pipeline-1 V3.5
          <br />
          LightGBM 双轨 · 规则引擎 v2
        </div>
      </nav>
      <main>
        {page === '选股看板' ? (
          <SelectionPage onJumpToTrading={jumpToTrading} />
        ) : page === '交易看板' ? (
          <TradingPage initialSymbol={tradingSymbol} />
        ) : (
          <Page />
        )}
      </main>
    </div>
  )
}
