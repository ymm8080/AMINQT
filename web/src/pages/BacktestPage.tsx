import { useState } from 'react'
import { api, BacktestResult } from '../api'
import { EquityChart } from '../components/EquityChart'

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

export function BacktestPage() {
  const [params, setParams] = useState({
    top_n: 15,
    max_hold_days: 3,
    hard_stop: -0.04,
    trailing_drawdown: 0.04,
    prob_exit: 0.5,
    initial_capital: 1000000,
    window_days: 180,
  })
  const [objective, setObjective] = useState('net_excess_annual')
  const [maxDdLimit, setMaxDdLimit] = useState(-0.1)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [tune, setTune] = useState<Record<string, unknown> | null>(null)
  const [tuneParams, setTuneParams] = useState('max_hold_days,prob_exit')
  const [tuneRanges, setTuneRanges] = useState('')
  const [busy, setBusy] = useState(false)

  const run = () => {
    setBusy(true)
    api.runBacktest({ ...params, objective, max_dd_limit: maxDdLimit })
      .then(setResult)
      .finally(() => setBusy(false))
  }

  const runTune = () => {
    setBusy(true)
    const names = tuneParams.split(',').map((s) => s.trim()).filter(Boolean)
    const ranges: Record<string, [number, number, number]> = {}
    if (tuneRanges.trim()) {
      tuneRanges.split(',').forEach((part) => {
        const [name, lo, hi, step] = part.split(':')
        if (name && lo && hi && step) {
          ranges[name.trim()] = [Number(lo), Number(hi), Number(step)]
        }
      })
    }
    api.runTune(names, objective, maxDdLimit, ranges)
      .then(setTune)
      .finally(() => setBusy(false))
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
        <div>
          <label>回测窗口</label>
          <select
            value={params.window_days}
            onChange={(e) => setParams({ ...params, window_days: Number(e.target.value) })}
          >
            <option value={120}>最近 6 个月</option>
            <option value={750}>过去三年</option>
          </select>
        </div>
        {num('top_n', 'Top N')}
        {num('max_hold_days', '最大持仓天数')}
        {num('prob_exit', '概率衰减退出', 0.05)}
        {num('hard_stop', '硬止损', 0.005)}
        {num('trailing_drawdown', '移动止盈回撤', 0.005)}
        {num('initial_capital', '初始资金', 100000)}
      </div>

      <div className="panel grid grid-2">
        <div>
          <label>目标函数</label>
          <select value={objective} onChange={(e) => setObjective(e.target.value)}>
            <option value="net_excess_annual">净超额(年化)</option>
            <option value="sharpe">夏普</option>
            <option value="total_return">总收益</option>
          </select>
        </div>
        <div>
          <label>约束: 最大回撤限制 %</label>
          <input
            type="number"
            step={1}
            value={maxDdLimit * 100}
            onChange={(e) => setMaxDdLimit(Number(e.target.value) / 100)}
          />
        </div>
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
        <label>自定义范围 (可选, 格式 name:lo:hi:step,...)</label>
        <input
          style={{ width: 400 }}
          placeholder="max_hold_days:2:5:1,prob_exit:0.4:0.6:0.05"
          value={tuneRanges}
          onChange={(e) => setTuneRanges(e.target.value)}
        />
        <button className="primary" style={{ marginLeft: 12 }} onClick={runTune} disabled={busy}>🔍 调优</button>
        {tune && <pre style={{ fontSize: 12, marginTop: 12 }}>{JSON.stringify(tune, null, 2)}</pre>}
      </div>
    </>
  )
}
