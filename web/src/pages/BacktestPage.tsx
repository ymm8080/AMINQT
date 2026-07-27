import { useEffect, useState } from 'react'
import { api, BacktestResult, ForecastQuality } from '../api'
import { EquityChart } from '../components/EquityChart'

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`
const fmtNum = (v: number | null | undefined, decimals = 4) =>
  v != null ? v.toFixed(decimals) : '—'
const qualityTrafficLight = (q: ForecastQuality | null) => {
  if (!q) return '⚪'
  const biasBigDown = q.bias_big_down ?? 0
  const bias1d = q.bias_1d ?? 0
  const dirAcc = q.direction_accuracy ?? 0.55
  if (biasBigDown > 0.02) return '🔴'
  if (Math.abs(bias1d) > 0.01) return '🔴'
  if (dirAcc < 0.50) return '🔴'
  if (Math.abs(bias1d) > 0.005) return '🟡'
  return '🟢'
}

export function BacktestPage() {
  const today = new Date().toISOString().slice(0, 10)
  const ago180 = new Date(Date.now() - 180 * 86400000).toISOString().slice(0, 10)
  const [params, setParams] = useState({
    top_n: 15,
    max_hold_days: 3,
    hard_stop: -0.04,
    trailing_drawdown: 0.04,
    prob_exit: 0.5,
    initial_capital: 1000000,
    window_days: 180,
  })
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [objective, setObjective] = useState('net_excess_annual')
  const [maxDdLimit, setMaxDdLimit] = useState(-0.1)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [quality, setQuality] = useState<ForecastQuality | null>(null)
  const [tune, setTune] = useState<Record<string, unknown> | null>(null)
  const [tuneParams, setTuneParams] = useState('max_hold_days,prob_exit')
  const [tuneRanges, setTuneRanges] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.tuningReport()
      .then((r) => {
        if (!r.exists || !r.best_params || typeof r.best_params !== 'object') return
        const best = r.best_params as Record<string, number>
        setParams((prev) => ({
          ...prev,
          ...Object.fromEntries(
            Object.entries(best).map(([k, v]) => [k, Number(v)])
          ),
        }))
        setTune(r)
      })
      .catch(() => {})
    api.forecastQuality().then(setQuality).catch(() => {})
  }, [])

  const run = () => {
    setBusy(true)
    api.runBacktest({
      ...params,
      objective,
      max_dd_limit: maxDdLimit,
      start_date: startDate || undefined,
      end_date: endDate || undefined,
    })
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
        {tune && tune.exists === true && (
          <p className="dim" style={{ gridColumn: '1 / -1', margin: 0 }}>
            已加载参数调优最优解 (目标: {String(tune.objective)})
            {tune.fallback_to_default === true && ' · 已回退默认值'}
          </p>
        )}
        {num('window_days', '回测窗口')}
        {num('initial_capital', '初始资金', 100000)}
        {num('top_n', 'Top N')}
        {num('max_hold_days', '最大持仓天数')}
        {num('prob_exit', '概率衰减退出', 0.05)}
        {num('hard_stop', '硬止损', 0.005)}
        {num('trailing_drawdown', '移动止盈回撤', 0.005)}
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

      <div className="panel grid grid-4" style={{ marginTop: 12 }}>
        <div>
          <label>起始日期 (留空=自动)</label>
          <input
            type="date"
            value={startDate}
            max={endDate || today}
            onChange={(e) => setStartDate(e.target.value)}
            placeholder={ago180}
          />
        </div>
        <div>
          <label>结束日期 (留空=今天)</label>
          <input
            type="date"
            value={endDate}
            min={startDate || undefined}
            max={today}
            onChange={(e) => setEndDate(e.target.value)}
            placeholder={today}
          />
        </div>
        <div>
          <label>或: 回测天数</label>
          <input
            type="number"
            value={params.window_days}
            onChange={(e) => setParams({ ...params, window_days: Number(e.target.value) })}
            disabled={!!startDate && !!endDate}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button className="primary" onClick={run} disabled={busy} style={{ width: '100%' }}>▶ 执行回测</button>
        </div>
      </div>

      {result && (
        <>
          {/* ── 赚钱 ── */}
          <div className="panel" style={{ marginTop: 16 }}>
            <h3>💰 赚钱</h3>
            <div className="grid grid-4">
              <div><div className="metric-label">总收益</div><div className="metric" style={{ color: result.metrics.total_return >= 0 ? '#2a2' : '#e44' }}>{fmtPct(result.metrics.total_return)}</div></div>
              <div><div className="metric-label">年化收益</div><div className="metric">{fmtPct(result.metrics.annual_return)}</div></div>
              <div><div className="metric-label">净超额(年化)</div><div className="metric" style={{ color: result.metrics.net_excess_annual >= 0 ? '#2a2' : '#e44' }}>{fmtPct(result.metrics.net_excess_annual)}</div></div>
              <div><div className="metric-label">交易笔数</div><div className="metric">{result.metrics.total_trades}</div></div>
              <div><div className="metric-label">胜率</div><div className="metric" style={{ color: result.metrics.win_rate >= 0.45 ? '#2a2' : result.metrics.win_rate < 0.35 ? '#e44' : undefined }}>{(result.metrics.win_rate * 100).toFixed(1)}%</div></div>
              <div><div className="metric-label">盈亏比</div><div className="metric" style={{ color: result.metrics.pl_ratio >= 1.5 ? '#2a2' : result.metrics.pl_ratio < 1.0 ? '#e44' : undefined }}>{result.metrics.pl_ratio.toFixed(2)}</div></div>
              <div><div className="metric-label">期望/笔</div><div className="metric" style={{ color: result.metrics.expectancy >= 0 ? '#2a2' : '#e44' }}>{fmtPct(result.metrics.expectancy)}</div></div>
              <div><div className="metric-label">平均持仓(天)</div><div className="metric">{result.metrics.avg_holding_days}</div></div>
            </div>
          </div>

          {/* ── 风险 ── */}
          <div className="panel">
            <h3>⚠️ 风险</h3>
            <div className="grid grid-4">
              <div><div className="metric-label">最大回撤</div><div className="metric" style={{ color: result.metrics.max_drawdown < -0.15 ? '#e44' : undefined }}>{(result.metrics.max_drawdown * 100).toFixed(1)}%</div></div>
              <div><div className="metric-label">最大连亏</div><div className="metric">{result.metrics.max_consecutive_loss} 笔</div></div>
              <div><div className="metric-label">Sortino</div><div className="metric" style={{ color: result.metrics.sortino >= 1.0 ? '#2a2' : undefined }}>{result.metrics.sortino.toFixed(2)}</div></div>
              <div><div className="metric-label">夏普</div><div className="metric">{result.metrics.sharpe.toFixed(2)}</div></div>
            </div>
          </div>

          {/* ── OOS IC ── */}
          <div className="panel">
            <h3>🎯 OOS Rank IC (模型有效性)</h3>
            <div className="grid grid-3">
              <div>
                <div className="metric-label">OOS Rank IC</div>
                <div className="metric" style={{ color: result.metrics.oos_rank_ic >= 0.03 ? '#2a2' : result.metrics.oos_rank_ic > 0 ? '#e90' : '#e44', fontSize: 28 }}>
                  {result.metrics.oos_rank_ic.toFixed(4)}
                </div>
                <div className="dim" style={{ fontSize: 11 }}>
                  {result.metrics.oos_rank_ic >= 0.03 ? '🟢 有效 (>0.03)' : result.metrics.oos_rank_ic > 0 ? '🟡 弱效' : '🔴 失效'}
                </div>
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <div className="metric-label">含义</div>
                <p className="dim" style={{ fontSize: 12, margin: '4px 0' }}>
                  Rank IC = 预测排序 与 实际收益 的 Spearman 相关系数。<br />
                  <b>&gt; 0.03</b> = 模型对"哪些股票会涨更多"有稳定的排序能力<br />
                  <b>0 ~ 0.03</b> = 排序能力弱，检查特征/标签<br />
                  <b>&lt; 0</b> = 模型反向预测，立即停止实盘
                </p>
              </div>
            </div>
          </div>

          <div className="panel">
            <h3>净值曲线 {result.demo && <span className="badge">演示面板</span>}</h3>
            <EquityChart data={result.nav_curve} />
          </div>
        </>
      )}

      {/* P25 预测质量 */}
      <div className="panel" style={{ marginTop: 16 }}>
        <h3>
          预测质量 D-23~D-26 {quality?.demo && <span className="badge">演示</span>}
          {quality?.date && <span className="dim" style={{ fontSize: 12, marginLeft: 8 }}>{quality.date}</span>}
        </h3>
        <div className="grid grid-4" style={{ marginBottom: 12 }}>
          <div>
            <div className="metric-label">MAE 1d</div>
            <div className="metric">{fmtNum(quality?.mae_1d)}</div>
          </div>
          <div>
            <div className="metric-label">BIAS 1d</div>
            <div className="metric" style={{ color: (quality?.bias_1d ?? 0) > 0.005 ? '#e44' : (quality?.bias_1d ?? 0) < -0.005 ? '#e90' : undefined }}>
              {fmtNum(quality?.bias_1d)}
            </div>
          </div>
          <div>
            <div className="metric-label">方向准确率</div>
            <div className="metric" style={{ color: (quality?.direction_accuracy ?? 0.55) < 0.50 ? '#e44' : undefined }}>
              {quality?.direction_accuracy != null ? `${(quality.direction_accuracy * 100).toFixed(1)}%` : '—'}
            </div>
          </div>
          <div>
            <div className="metric-label">红灯状态</div>
            <div className="metric">{qualityTrafficLight(quality)}</div>
          </div>
        </div>

        <div className="grid grid-4">
          <div>
            <div className="metric-label">BIAS 大涨 (&gt;3%)</div>
            <div className="metric dim">{fmtNum(quality?.bias_big_up)}</div>
          </div>
          <div>
            <div className="metric-label">BIAS 小涨 (0~3%)</div>
            <div className="metric dim">{fmtNum(quality?.bias_small_up)}</div>
          </div>
          <div>
            <div className="metric-label">BIAS 小跌 (-3%~0)</div>
            <div className="metric dim">{fmtNum(quality?.bias_small_down)}</div>
          </div>
          <div>
            <div className="metric-label">BIAS 大跌 (&lt;-3%)</div>
            <div className="metric dim" style={{ color: (quality?.bias_big_down ?? 0) > 0.02 ? '#e44' : undefined }}>
              {fmtNum(quality?.bias_big_down)}
            </div>
          </div>
        </div>
        <p className="dim" style={{ fontSize: 11, marginTop: 8, marginBottom: 0 }}>
          MAE/BIAS/方向准确率仅用于监控与校准执行参数，不得用于优化模型目标函数。<br />
          BIAS_big_down &gt; +0.02 → 触发 E4-L1 模型降级 (大跌日模型高估=攻击档风险)。
        </p>
      </div>

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
