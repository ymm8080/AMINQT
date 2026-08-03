export const meta = {
  name: 'v3-merge-verify-fanout',
  description: 'Parallel read-only verification before the LHB/SW merge into the production V3 panel',
  phases: [
    { title: 'Verify', detail: 'LHB cache integrity + merge script review + zero-fill sweep' },
  ],
}

const LHB_SCHEMA = {
  type: 'object',
  properties: {
    checks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          pass: { type: 'boolean' },
          detail: { type: 'string' },
        },
        required: ['name', 'pass', 'detail'],
      },
    },
    missing_dates: { type: 'array', items: { type: 'string' } },
    all_zero_rows: { type: 'integer' },
    notes: { type: 'string' },
  },
  required: ['checks', 'missing_dates', 'all_zero_rows', 'notes'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'warning', 'info'] },
          file: { type: 'string' },
          line: { type: 'string' },
          issue: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'file', 'line', 'issue', 'fix'],
      },
    },
    verdict: { type: 'string', enum: ['safe', 'fix-before-merge'] },
    summary: { type: 'string' },
  },
  required: ['findings', 'verdict', 'summary'],
}

const SWEEP_SCHEMA = {
  type: 'object',
  properties: {
    flagged: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          col: { type: 'string' },
          frac_zero: { type: 'number' },
          frac_nan: { type: 'number' },
          distinct: { type: 'integer' },
          reason: { type: 'string' },
        },
        required: ['col', 'frac_zero', 'frac_nan', 'distinct', 'reason'],
      },
    },
    clean: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['flagged', 'clean', 'notes'],
}

phase('Verify')

const lhbTask = () => agent(
  'You are verifying the integrity of an LHB (dragon-tiger list) dataset that will be merged into a production quant panel. Read the parquet file at D:/AMINQT/AMINQT CODES/data/supply_cache/alt_data/lhb/all_20240102_20260727.parquet using Python + pandas/pyarrow (use forward slashes in paths). It has 44,854 rows fetched from Tushare top_list, expected schema [symbol, date, lhb_net_buy, lhb_buy_amt, lhb_sell_amt], covering trade dates 2024-01-02 to 2026-07-27. This data will OVERWRITE currently-all-zero LHB cells in the production panel, so correctness matters. Run these checks and report: (a) schema and dtypes; (b) date coverage: distinct dates vs the expected ~620 SSE trading days in that range; list the trade dates that have NO rows — holidays with no dragon-tiger activity are legitimate, but large unexplained gaps are suspicious (you may assume the full list of SSE trade dates is obtainable via tushare trade_cal if tushare is installed, otherwise just list the actual min/max and a sensible gap analysis); (c) duplicate (symbol, date) pairs; (d) value sanity: lhb_net_buy may be negative (net sell); lhb_buy_amt and lhb_sell_amt should be positive; count any row where ALL THREE lhb columns are exactly 0 (a fake-zero that would propagate into the panel); count all-NaN rows; (e) print 5 sample rows from each of 2024, 2025, 2026. Report each check as pass/fail with a concrete detail. Under 400 words.',
  { label: 'lhb-cache-integrity', phase: 'Verify', schema: LHB_SCHEMA }
)

const reviewTask = () => agent(
  'You are doing a pre-flight code review of two Python scripts that will MUTATE a 765MB production parquet panel at D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet (~3.15M rows, 97 cols). A bug here corrupts the production source of truth, so review rigorously. Read BOTH scripts fully: (1) D:/AMINQT/AMINQT CODES/scripts/_refill_lhb_2024_26.py — loads the panel, WORM-backups it, then overwrites cells where the 3 LHB columns are all-zero with real values from a cache (unmatched becomes NaN). (2) D:/AMINQT/AMINQT CODES/scripts/add_sw_cols_to_panel.py — removes sw_l1/l2/l3_code and sw_index_* columns, ADDS sw_l1/l2/l3_name string columns, by streaming row-group-by-row-group rewrite to a .tmp then atomic rename. Hunt for bugs that corrupt data or lose columns: (1) zero_mask logic correctness (does eq(0) treat NaN as not-zero — is that right for 2023 rows where LHB was never present?); (2) merge suffix collisions (suffixes ("", "_lhb") — could a real LHB column get clobbered by the merge); (3) the assignment v3.loc[zero_mask, c] = cand.values — does .values align correctly with the boolean mask after a filtered merge, or could NaN or 0 land on the wrong rows; (4) dtype coercion in add_sw_cols (string cols, PA.NA handling, the junk all-NaN CSV row being skipped by the "if tc:" guard); (5) streaming rewrite: does it preserve ALL other columns and row order; any column-order mismatch vs new_schema; does rg_df[new_col_names] raise on missing cols; (6) atomic rename failure modes on Windows (os.remove then os.rename). For each finding give severity (blocker/warning/info), file, approximate line, issue, and a concrete fix. Conclude with verdict: "safe" or "fix-before-merge". Under 450 words.',
  { label: 'merge-script-review', phase: 'Verify', schema: REVIEW_SCHEMA }
)

const sweepTask = () => agent(
  'You are sweeping a production quant panel for data-quality pathologies similar to a known bug: the LHB columns were all-zero because a fetch script used wrong column names and wrote scalar 0.0 everywhere. Now hunt for OTHER columns in the panel with the same pathology. Read D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet (97 cols, ~3.15M rows, READ-ONLY — do not write; another process may be mid-rewrite, so just read whatever is there). Use pyarrow.parquet row-group reading (the file has 3 row groups of ~1M rows) to keep memory low. For EACH numeric column compute: fraction of exactly-zero values, fraction of NaN, and distinct count. Flag any column that is: (a) over 90% exactly zero, (b) over 90% NaN, (c) constant (1 distinct), or (d) zero-heavy AND its name suggests money or volume (name contains _amt, _buy, _sell, _net, _amount, _vol, _ret, _close, _mv, or _flow). For flagged columns report col, frac_zero, frac_nan, distinct, and a one-line reason. Also list a sample of ~10 columns you verified as CLEAN (healthy nonzero distribution). Under 350 words. IMPORTANT: this is a read-only sweep — do not modify the panel.',
  { label: 'panel-zero-fill-sweep', phase: 'Verify', schema: SWEEP_SCHEMA }
)

const [lhb, review, sweep] = await parallel([lhbTask, reviewTask, sweepTask])

return { lhb, review, sweep }
