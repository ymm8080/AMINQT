import { useState } from 'react'
import { BacktestPage } from './pages/BacktestPage'
import { ConfigPage } from './pages/ConfigPage'
import { SelectionPage } from './pages/SelectionPage'
import { TradingPage } from './pages/TradingPage'

const PAGES: Record<string, () => JSX.Element> = {
  选股看板: SelectionPage,
  交易看板: TradingPage,
  回测中心: BacktestPage,
  配置中心: ConfigPage,
}

export default function App() {
  const [page, setPage] = useState<keyof typeof PAGES>('选股看板')
  const Page = PAGES[page]
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
        <Page />
      </main>
    </div>
  )
}
