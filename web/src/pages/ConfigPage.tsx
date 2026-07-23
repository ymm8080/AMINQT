import { useEffect, useState } from 'react'
import { api } from '../api'

/** 配置页: 规则 [TUNABLE] 参数 + 调参报告. */
export function ConfigPage() {
  const [rules, setRules] = useState<Record<string, { value: number; bounds: number[] }>>({})
  const [report, setReport] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    api.ruleConfig().then((r) => setRules(r.tunable)).catch(() => {})
    api.tuningReport().then((r) => setReport(r.exists ? r : null)).catch(() => {})
  }, [])

  return (
    <>
      <h2>配置中心</h2>
      <div className="panel">
        <h3>规则引擎参数 ([TUNABLE] 可回测调优)</h3>
        <p className="dim">在线写回在 Pipeline-2 定稿后开放; 当前经 回测中心→调参 写回</p>
        <table>
          <thead><tr><th>参数</th><th>当前值</th><th>边界</th></tr></thead>
          <tbody>
            {Object.entries(rules).map(([name, v]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{v.value}</td>
                <td className="dim">[{v.bounds.join(' ~ ')}]</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <h3>调参报告</h3>
        {report
          ? <pre style={{ fontSize: 12 }}>{JSON.stringify(report, null, 2)}</pre>
          : <p className="dim">暂无 — 在回测中心执行参数调优后生成</p>}
      </div>
    </>
  )
}
