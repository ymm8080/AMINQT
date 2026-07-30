import { useCallback, useEffect, useState } from 'react'
import { api, type PipelineStatus, type AppendDailyResult } from '../api'

export function DataPipelinePage() {
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<AppendDailyResult | null>(null)
  const [tradeDate, setTradeDate] = useState('')
  const [marketState, setMarketState] = useState('range')
  const [savePanel, setSavePanel] = useState(true)

  const refreshStatus = useCallback(async () => {
    setLoading(true)
    try {
      const s = await api.pipelineStatus()
      setStatus(s)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshStatus()
  }, [refreshStatus])

  const handleRun = async () => {
    setRunning(true)
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
      setRunning(false)
    }
  }

  return (
    <>
      <h2>数据管道</h2>

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

      {/* 触发每日追加 */}
      <div className="panel">
        <h3>每日数据追加 + 推理预测</h3>
        <p className="dim">
          拉取当日 OHLCV + 融资融券/北向资金/龙虎榜 → 追加到 V3 面板 (WORM 备份) → 加载模型推理 → 生成清单
        </p>
        <div className="grid grid-3">
          <div>
            <label>交易日期 (留空=今天)</label>
            <input
              type="text"
              placeholder="YYYYMMDD"
              value={tradeDate}
              onChange={(e) => setTradeDate(e.target.value)}
            />
          </div>
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
          disabled={running}
          style={{ marginTop: 16 }}
        >
          {running ? '运行中...' : '运行管道'}
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
