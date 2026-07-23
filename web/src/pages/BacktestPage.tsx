import { useState } from 'react'
import { api, BacktestResult } from '../api'
import { EquityChart } from '../components/EquityChart'

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

/** 回测页: 参数表单 + V3.5 协议回测 + 网格调参. */
export function BacktestPage() {
  const [params, setParams] = useState({
    top_n: 15, max_hold_days: 3, hard_stop: -0.04,
    trailing_drawdown: 0.04, prob_exit: 0.5, initial_capital: 1000000,
  })
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [tune, setTune] = useState<Record<string, unknown> | null>(null)
  const [tuneParams, setTuneParams] = useState('max_hold_days,prob_exit')
  const [busy, setBusy] = useState(false)

  const run = () => {
    setBusy(true)
    api.runBacktest(params).then(setResult).finally(() => setBusy(false))
  }
  const runTune = () => {
    setBusy(true)
    api.runTune(tuneParams.split(',').map((s) => s.trim()).filter(Boolean))
      .then(setTune).finally(() => setBusy(false))
  }

  const num = (k: keyof typeof params, label: string, step = 1) => (
    <div key={k}>
      <label>{label}</label>
      <input
        type="number"
        step={step}
        value={params[k]}
        onChange={(e) => setParams({ ...params, [k]: Number(e.target.value) })}
      />
    </div>
  )

  return (
    <>
      <h2>回测中心 · V3.5 协议</h2>
      <p className="dim">T+1 open + 滑点0.05% · 佣金万2.5 + 印花税0.05% · 等权1/N 单票≤10% · 验收=扣费后净超额</p>
      <div className="panel grid grid-3">
        {num('top_n', 'Top N')}
        {num('max_hold_days', '最大持仓天数')}
        {num('prob_exit', '概率衰减退出', 0.05)}
        {num('hard_stop', '硬止损', 0.005)}
        {num('trailing_drawdown', '移动止盈回撤', 0.005)}
        {num('initial_capital', '初始资金', 100000)}
      </div>
      <button className="primary" onClick={run} disabled={busy}>▶ 执行回测</button>
      {result && (
        <>
          <div className="panel grid grid-3" style={{ marginTop: 16 }}>
            <div><div className="metric-label">总收益</div><div className="metric">{fmtPct(result.metrics.total_return)}</div></div>
            <div><div className="metric-label">年化</div><div className="metric">{fmtPct(result.metrics.annual_return)}</div></div>
            <div><div className="metric-label">净超额(年化)</div><div className="metric">{fmtPct(result.metrics.net_excess_annual)}</div></div>
            <div><div className="metric-label">最大回撤</div><div className="metric">{(result.metrics.max_drawdown * 100).toFixed(1)}%</div></div>
            <div><div className="metric-label">夏普</div><div className="metric">{result.metrics.sharpe.toFixed(2)}</div></div>
          </div>
          <div className="panel">
            <h3>净值曲线 {result.demo && <span className="badge">演示面板</span>}</h3>
            <EquityChart data={result.nav_curve} />
          </div>
        </>
      )}
      <div className="panel">
        <h3>参数调优 (网格搜索 + OOS 复验)</h3>
        <label>调参目标 (逗号分隔, ≤4 维)</label>
        <input style={{ width: 400 }} value={tuneParams} onChange={(e) => setTuneParams(e.target.value)} />
        <button className="primary" style={{ marginLeft: 12 }} onClick={runTune} disabled={busy}>🔍 调优</button>
        {tune && <pre style={{ fontSize: 12, marginTop: 12 }}>{JSON.stringify(tune, null, 2)}</pre>}
      </div>
    </>
  )
}
