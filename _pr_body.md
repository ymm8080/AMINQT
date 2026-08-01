## Changes

1. **setup_scheduled_tasks.ps1**: Pipeline 1 ScriptPath changed to _daily_fetch.py; Pipeline 2 trigger time changed to 22:40; added $TriggerTime parameter
2. **run_announcement_pipeline.py**: added update_v3_panel_holdertrade() function — fetches holdertrade, aggregates by (symbol, announce_date), updates V3 panel today rows
3. **feature_engine_v35.py**: fixed pct_90_con fallback formula (denominator from cost_50pct to cost_95pct+cost_5pct); added pct_70_con fallback
4. Deleted run_daily_market_pipeline.py (no longer used)
5. V3 panel: deleted holder_count column (98 cols remaining)
