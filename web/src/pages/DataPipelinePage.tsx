import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api,
  type PipelineStatus,
  type PipelineTaskStatusResponse,
  type AppendDailyResult,
} from '../api'

const PIPELINE_LABELS: Record<string, string> = {
  daily_fetch: 'Daily Fetch',
  announcement: 'Announcement Fetch',
  predict: 'Predict',
}

const PIPELINE_ORDER: Record<string, string> = {
  daily_fetch: '①',
  announcement: '②',
  predict: '③',
}

export function DataPipelinePage() {
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [tasks, setTasks] = useState<PipelineTaskStatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [result, setResult] = useState<AppendDailyResult | null>(null)
  const [tradeDate, setTradeDate] = useState('')
  const [marketState, setMarketState] = useState('range')
  const [savePanel, setSavePanel] = useState(true)
  const [mtimeNotice, setMtimeNotice] = useState(false)
  const prevMtime = useRef<number | undefined>(undefined)

  // ── Refresh status ──────────────────────────────────────────
  const refreshStatus = useCallback(async () => {
    setLoading(true)
    try {
      const [s, t] = await Promise.all([api.pipelineStatus(), api.taskStatus()])
      setStatus(s)
      setTasks(t)

      // Detect panel mtime change
      const m = s.panel?.panel_mtime_epoch
      if (m !== undefined && prevMtime.current !== undefined && m !== prevMtime.current) {
        setMtimeNotice(true)
        setTimeout(() => setMtimeNotice(false), 8000)
      }
      if (m !== undefined) prevMtime.current = m
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshStatus()
  }, [refreshStatus])

  // ── 10-second polling ───────────────────────────────────────
  useEffect(() => {
    const iv = setInterval(refreshStatus, 10_000)
    return () => clearInterval(iv)
  }, [refreshStatus])

  // ── Trigger pipeline (daily_fetch / announcement / predict) ─
  const handleTrigger = async (script: string) => {
    setBusy(script)
    try {
      await api.triggerPipeline(script, tradeDate || undefined)
      await refreshStatus()
    } catch (e) {
      alert(`Failed to trigger ${script}: ${e}`)
    } finally {
      setBusy(null)
    }
  }

  // ── Run prediction (append-daily endpoint) ──────────────────
  const handleRun = async () => {
    setBusy('append-daily')
    setResult(null)
    try {
      const r = await api.appendDaily({
        trade_date: tradeDate || undefined,
        market_state: marketState,
        save_panel: savePanel,
      })
      setResult(r)
      if (r.success) refreshStatus()
    } catch (e) {
      setResult({
        success: false,
        trade_date: tradeDate || '',
        error: String(e),
        logs: [],
      })
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <h2>数据管道</h2>

      {/* mtime change notice */}
      {mtimeNotice && (
        <div className="panel" style={{ background: '#e8f5e9', border: '1px solid #4caf50' }}>
          ✅ V3 面板已更新 — {status?.panel?.panel_mtime}
        </div>
      )}

      {/* 状态概览 */}
      <div className="panel">
        <h3>面板 & 模型状态 {loading && <span className="dim"> (加载中...)</span>}</h3>
        {status && (
          <div className="grid grid-3">
            <div>
              <div className="metric-label">历史面板</div>
              {status.panel.exists ? (
                <>
                  <div className="metric">{status.panel.n_stocks}</div>
                  <div className="dim">
                    {status.panel.n_rows?.toLocaleString()} 行 · {status.panel.size_mb}MB
                    <br />
                    {status.panel.first_date} ~ {status.panel.last_date}
                    <br />
                    <span style={{ fontSize: 11 }}>mtime: {status.panel.panel_mtime}</span>
                  </div>
                </>
              ) : (
                <div className="up">未找到面板文件</div>
              )}
            </div>
            <div>
              <div className="metric-label">模型包</div>
              {Object.keys(status.models).length > 0 ? (
                Object.entries(status.models).map(([board, m]) => (
                  <div key={board} style={{ marginBottom: 8 }}>
                    <strong>{board}</strong>: {m.file}
                    <br />
                    <span className="dim">更新: {m.modified}</span>
                  </div>
                ))
              ) : (
                <div className="up">无模型包</div>
              )}
            </div>
            <div>
              <div className="metric-label">最新清单</div>
              <div className="metric">{status.latest_list_date || '—'}</div>
              <div className="dim">共 {status.list_count} 份历史清单</div>
            </div>
          </div>
        )}
        <button className="primary" onClick={refreshStatus} disabled={loading} style={{ marginTop: 12 }}>
          刷新状态
        </button>
      </div>

      {/* Pipeline 触发 */}
      <div className="panel">
        <h3>管道触发</h3>
        <p className="dim">
          按顺序执行：① Daily Fetch (22:00) → ② Announcement Fetch (22:40) → ③ Predict (23:00)
        </p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <label className="dim">交易日期 (留空=今天)</label>
          <input
            type="text"
            placeholder="YYYYMMDD"
            value={tradeDate}
            onChange={(e) => setTradeDate(e.target.value)}
            style={{ width: 120 }}
          />
        </div>
        <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
          {(['daily_fetch', 'announcement', 'predict'] as const).map((key) => (
            <button
              key={key}
              className="primary"
              onClick={() => handleTrigger(key)}
              disabled={busy !== null}
            >
              {busy === key
                ? `${PIPELINE_LABELS[key]} 运行中...`
                : `${PIPELINE_ORDER[key]} ${PIPELINE_LABELS[key]}`}
            </button>
          ))}
        </div>
      </div>

      {/* 任务状态表 */}
      {tasks && Object.keys(tasks.tasks).length > 0 && (
        <div className="panel">
          <h3>任务状态</h3>
          <table>
            <thead>
              <tr>
                <th>脚本</th>
                <th>状态</th>
                <th>开始时间</th>
                <th>完成时间</th>
                <th>Return Code</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(tasks.tasks).map(([id, t]) => (
                <tr key={id}>
                  <td>{t.script}</td>
                  <td>
                    {t.status === 'running' && <span className="dim">🔄 running</span>}
                    {t.status === 'done' && <span className="down">✅ done</span>}
                    {t.status === 'failed' && <span className="up">❌ failed</span>}
                  </td>
                  <td>{t.started_at || '—'}</td>
                  <td>{t.finished_at || '—'}</td>
                  <td>{t.returncode ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* 展开日志 */}
          {Object.entries(tasks.tasks).map(([id, t]) =>
            t.stderr ? (
              <details key={id} style={{ marginTop: 8 }}>
                <summary className="dim">{t.script} — stderr</summary>
                <pre
                  style={{
                    fontSize: 12,
                    background: '#fff3e0',
                    padding: 12,
                    borderRadius: 8,
                    overflowX: 'auto',
                    marginTop: 8,
                  }}
                >
                  {t.stderr.slice(-2000)}
                </pre>
              </details>
            ) : null
          )}
        </div>
      )}

      {/* 触发每日追加 + 推理预测 */}
      <div className="panel">
        <h3>每日数据追加 + 推理预测</h3>
        <p className="dim">
          拉取当日 OHLCV + 融资融券/北向资金/龙虎榜 → 追加到 V3 面板 (WORM 备份) → 加载模型推理 → 生成清单
        </p>
        <div className="grid grid-3">
          <div>
            <label>市场状态</label>
            <select value={marketState} onChange={(e) => setMarketState(e.target.value)}>
              <option value="range">震荡 (range)</option>
              <option value="bull">牛市 (bull)</option>
              <option value="bear">熊市 (bear)</option>
            </select>
          </div>
          <div>
            <label>保存面板 (WORM 备份)</label>
            <select value={savePanel ? '1' : '0'} onChange={(e) => setSavePanel(e.target.value === '1')}>
              <option value="1">是 — 追加后保存</option>
              <option value="0">否 — 仅推理不保存</option>
            </select>
          </div>
        </div>
        <button
          className="primary"
          onClick={handleRun}
          disabled={busy !== null}
          style={{ marginTop: 16 }}
        >
          {busy === 'append-daily' ? '运行中...' : '运行预测'}
        </button>
      </div>

      {/* 运行结果 */}
      {result && (
        <div className="panel">
          <h3>
            运行结果 — {result.trade_date}{' '}
            {result.success ? (
              <span className="down">✅ 成功</span>
            ) : (
              <span className="up">❌ 失败</span>
            )}
          </h3>

          {result.error && (
            <p className="up" style={{ fontWeight: 600 }}>
              错误: {result.error}
            </p>
          )}

          {result.success && (
            <>
              <div className="grid grid-3" style={{ marginBottom: 16 }}>
                <div>
                  <div className="metric-label">模式</div>
                  <div style={{ fontSize: 18, fontWeight: 600 }}>{result.mode}</div>
                </div>
                <div>
                  <div className="metric-label">入选股票</div>
                  <div className="metric">{result.n_stocks}</div>
                </div>
                <div>
                  <div className="metric-label">仓位上限</div>
                  <div style={{ fontSize: 18, fontWeight: 600 }}>
                    {((result.cap_position ?? 0) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              {result.list_preview && result.list_preview.length > 0 && (
                <table>
                  <thead>
                    <tr>
                      <th>代码</th>
                      <th>板块</th>
                      <th>prob_up</th>
                      <th>pred_1d</th>
                      <th>pred_3d</th>
                      <th>pred_5d</th>
                      <th>score</th>
                      <th>weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.list_preview.map((s) => (
                      <tr key={s.symbol}>
                        <td>{s.symbol}</td>
                        <td>{s.board}</td>
                        <td>{(s.prob_up * 100).toFixed(1)}%</td>
                        <td className={s.pred_ret_1d >= 0 ? 'up' : 'down'}>
                          {(s.pred_ret_1d * 100).toFixed(2)}%
                        </td>
                        <td className={s.pred_ret_3d >= 0 ? 'up' : 'down'}>
                          {(s.pred_ret_3d * 100).toFixed(2)}%
                        </td>
                        <td className={s.pred_ret_5d >= 0 ? 'up' : 'down'}>
                          {(s.pred_ret_5d * 100).toFixed(2)}%
                        </td>
                        <td>{s.score.toFixed(4)}</td>
                        <td>{(s.weight * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {result.empty && (
                <p className="dim">
                  空清单 — {result.valve_state === 'empty' ? '流动性安全阀触发' : '无候选通过阈值'}
                </p>
              )}
            </>
          )}

          {/* 执行日志 */}
          {result.logs.length > 0 && (
            <details style={{ marginTop: 16 }}>
              <summary className="dim">执行日志 ({result.logs.length} 行)</summary>
              <pre
                style={{
                  fontSize: 12,
                  background: '#f0f3f6',
                  padding: 12,
                  borderRadius: 8,
                  overflowX: 'auto',
                  marginTop: 8,
                }}
              >
                {result.logs.join('\n')}
              </pre>
            </details>
          )}
        </div>
      )}
    </>
  )
}
