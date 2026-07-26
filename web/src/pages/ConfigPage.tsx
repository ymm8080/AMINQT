import { useEffect, useState } from 'react'
import { api } from '../api'

export function ConfigPage() {
  const [rules, setRules] = useState<Record<string, { value: number; bounds: number[] }>>({})
  const [report, setReport] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    api.ruleConfig().then((r) => setRules(r.tunable)).catch(() => {})
    api.tuningReport().then((r) => setReport(r.exists ? r : null)).catch(() => {})
  }, [])

  const factorDims = [
    ['① 价量动能', 'MACD/RSI/KDJ/60日乖离/量价背离'],
    ['② 波动率', 'ATR_pct / 布林带宽'],
    ['③ 基本面', 'PE_log/PB/净利营收增速 (announce_date PIT)'],
    ['④ 板块效应', '板块涨停家数/板块收益 (历史快照)'],
    ['⑤ 筹码分布', '集中度/获利盘 (shift 1)'],
    ['⑥ 个股公告因子', 'announce_score: 公告/业绩预告/解禁/分红等事件评分'],
    ['⑦ 涨停基因', '10/20日涨停天数/炸板率/连板高度0-4'],
    ['⑧ 日历-月份', '月份分类'],
    ['⑨ 自定义公式', '4 同花顺公式 (已审计, NECESSARY INDICATOR 复刻)'],
    ['⑩ 资金流', '主力净流入/超大单 (shift 1, 单一数据源)'],
    ['⑪ 连板/清单', 'is_in_yesterday_list (Holding Bonus)'],
    ['⑫ 均线系统', '5/10/20/60/120/250 距离 + 排列'],
    ['⑬ 日历-长假', 'days_to/after_holiday, is_pre/post'],
    ['⑭ 全市场情绪', '两市成交额 + 5d/20d 比值 + 涨跌停家数'],
  ]

  return (
    <>
      <h2>配置中心</h2>
      <div className="panel">
        <h3>规则引擎参数 ([TUNABLE] 可回测调优)</h3>
        <p className="dim">在线写回在 Pipeline-2 定稿后开放; 当前经 回测中心→调参 写回</p>
        <table>
          <thead>
            <tr>
              <th>参数</th>
              <th>当前值</th>
              <th>边界</th>
            </tr>
          </thead>
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
        {report ? (
          <TuningReport report={report} />
        ) : (
          <p className="dim">暂无 — 在回测中心执行参数调优后生成</p>
        )}
      </div>

      <div className="panel">
        <h3>V3.5 特征维度 (14 维 + 公告因子)</h3>
        <table>
          <thead>
            <tr>
              <th>维度</th>
              <th>组成</th>
            </tr>
          </thead>
          <tbody>
            {factorDims.map(([dim, comp]) => (
              <tr key={dim}>
                <td>{dim}</td>
                <td className="dim">{comp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function TuningReport({ report }: { report: Record<string, unknown> }) {
  const best = (report.best_params as Record<string, unknown>) ?? {}
  const fallback = report.fallback_to_default as boolean
  const train = report.train_score as number
  const oos = report.oos_score as number
  const leaderboard = (report.leaderboard as [Record<string, unknown>, number][] | undefined) ?? []

  return (
    <div>
      {fallback ? (
        <p className="up" style={{ fontWeight: 600 }}>
          ⚠️ 调参结果在样本外 (OOS) 表现不如默认参数, 系统已自动回退到默认值。
        </p>
      ) : (
        <p className="up" style={{ fontWeight: 600 }}>✅ 调参结果在样本外 (OOS) 验证通过。</p>
      )}
      <p>
        <strong>推荐参数:</strong>
      </p>
      <ul>
        {Object.entries(best).length ? (
          Object.entries(best).map(([k, v]) => <li key={k}>{k}: {String(v)}</li>)
        ) : (
          <li>使用默认值</li>
        )}
      </ul>
      <p className="dim">训练段评分: {fmtPct(train)} | OOS 评分: {fmtPct(oos)}</p>
      {leaderboard.length > 0 && (
        <>
          <h4>TOP 参数组合</h4>
          <ul>
            {leaderboard.slice(0, 3).map(([p, s], i) => (
              <li key={i}>第 {i + 1} 名: {fmtPct(s)} — {JSON.stringify(p)}</li>
            ))}
          </ul>
        </>
      )}
      <pre style={{ fontSize: 12, marginTop: 12 }}>{JSON.stringify(report, null, 2)}</pre>
    </div>
  )
}

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
