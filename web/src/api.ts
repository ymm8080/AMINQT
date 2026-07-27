// API client — FastAPI /api/frontier/*
const BASE = '/api/frontier'

export interface ListItem {
  symbol: string
  board: string
  day_change: number
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
  priority?: boolean
  added_at?: number
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

export interface IntradayPoint {
  time: string
  price: number
  volume: number
}

export interface SectorItem {
  板块: string
  涨跌幅: number
  上涨家数: number
  下跌家数: number
  intraday: number[]
}

export interface SignalItem {
  time: string
  symbol: string
  side: 'buy' | 'sell'
  price: number
  qty: number
  priority?: string
  reason?: string
  executed?: boolean
}

export interface BacktestMetrics {
  total_return: number
  annual_return: number
  net_excess_annual: number
  max_drawdown: number
  sharpe: number
  sortino: number
  n_days: number
  // trade-level profitability
  win_rate: number
  pl_ratio: number
  expectancy: number
  total_trades: number
  max_consecutive_loss: number
  avg_holding_days: number
  // OOS IC — model predictive validity
  oos_rank_ic: number
  ic_daily: number[]
}

export interface BacktestResult {
  demo: boolean
  metrics: BacktestMetrics
  nav_curve: { date: string; nav: number }[]
  trades: Record<string, unknown>[]
}

export interface ForecastQuality {
  exists: boolean
  demo: boolean
  date?: string
  mae_1d: number | null
  bias_1d: number | null
  direction_accuracy: number | null
  n_samples: number
  bias_big_up: number | null
  bias_small_up: number | null
  bias_small_down: number | null
  bias_big_down: number | null
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { cache: 'no-store', ...init })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const api = {
  latestList: () => req<LatestList>('/list/latest'),
  ohlc: (symbol: string, days = 120) =>
    req<{ items: OhlcBar[] }>(`/ohlc/${symbol}?days=${days}`),
  intraday: (symbol: string) =>
    req<{ items: IntradayPoint[] }>(`/intraday/${symbol}`),
  sectors: () => req<{ demo: boolean; items: SectorItem[] }>('/sectors'),
  signals: (symbol: string) =>
    req<{ demo: boolean; items: SignalItem[] }>(`/signals/${symbol}`),
  priority: () => req<{ symbols: string[] }>('/priority'),
  togglePriority: (symbol: string, name = '') =>
    req<{ priority: boolean }>('/priority/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, name }),
    }),
  watchlist: () => req<{ items: { symbol: string; name?: string; note?: string }[] }>('/watchlist'),
  toggleWatch: (symbol: string, name = '') =>
    req<{ watched: boolean }>('/watchlist/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, name }),
    }),
  runBacktest: (params: Record<string, unknown>) =>
    req<BacktestResult>('/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    }),
  runTune: (params: string[], objective: string, maxDdLimit: number | null, ranges: Record<string, [number, number, number]>) =>
    req<Record<string, unknown>>('/backtest/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params, objective, max_dd_limit: maxDdLimit, ranges }),
    }),
  ruleConfig: () =>
    req<{ tunable: Record<string, { value: number; bounds: number[] }> }>('/config/rules'),
  tuningReport: () => req<Record<string, unknown> & { exists: boolean }>('/tuning/report'),
  forecastQuality: () => req<ForecastQuality>('/forecast/quality'),
  // prediction pool
  predictionRuns: () => req<{ runs: { date: string; n_stocks: number; schema_version: string; created_at: string }[] }>('/prediction/runs'),
  predictionRun: (date: string) => req<{
    date: string
    meta: Record<string, unknown>
    stocks: (ListItem & {
      actual_ret_1d?: number; actual_ret_3d?: number; actual_ret_5d?: number
      direction_correct_1d?: number; pred_error_1d?: number
    })[]
  }>('/prediction/run/' + date),
  predictionQuality: () => req<{ items: { date: string; n: number; direction_accuracy: number; bias_1d: number; mae_1d: number }[] }>('/prediction/quality'),
}
