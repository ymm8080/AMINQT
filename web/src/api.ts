// API client — FastAPI /api/frontier/*
const BASE = '/api/frontier'

export interface ListItem {
  symbol: string
  board: string
  pred_ret_1d: number
  pred_ret_3d: number
  pred_ret_5d: number
  prob_up: number
  momentum: string
  consensus_score: number
  signal_conflict: number
  market_state: string
  score: number
  schema_version: string
  name?: string
  industry?: string
}

export interface LatestList {
  date: string
  demo: boolean
  schema_version: string
  items: ListItem[]
}

export interface OhlcBar {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface BacktestMetrics {
  total_return: number
  annual_return: number
  net_excess_annual: number
  max_drawdown: number
  sharpe: number
  n_days: number
}

export interface BacktestResult {
  demo: boolean
  metrics: BacktestMetrics
  nav_curve: { date: string; nav: number }[]
  trades: Record<string, unknown>[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const api = {
  latestList: () => req<LatestList>('/list/latest'),
  ohlc: (symbol: string, days = 120) =>
    req<{ items: OhlcBar[] }>(`/ohlc/${symbol}?days=${days}`),
  watchlist: () => req<{ items: { symbol: string; name?: string; note?: string }[] }>('/watchlist'),
  toggleWatch: (symbol: string, name = '') =>
    req<{ watched: boolean }>('/watchlist/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, name }),
    }),
  runBacktest: (params: Record<string, number>) =>
    req<BacktestResult>('/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  runTune: (params: string[]) =>
    req<Record<string, unknown>>('/backtest/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    }),
  ruleConfig: () =>
    req<{ tunable: Record<string, { value: number; bounds: number[] }> }>('/config/rules'),
  tuningReport: () => req<Record<string, unknown> & { exists: boolean }>('/tuning/report'),
}
