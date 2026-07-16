# AMINQT — A-Share Graphic-Factor Quant Trading Platform

Selects China A-shares from daily K-line graphic factors, predicts returns
with LSTM/LightGBM, learns intraday patterns into trading rules, and
executes trades in **AUTO** (granted) or **MANUAL** (recommendation pop-up)
mode.

## Architecture — 3 modules

| Module | Purpose | Location |
| :--- | :--- | :--- |
| **M1** Factor capture + prediction | K-line → factor matrix → LSTM/LGB → return forecast | `app/core/factor_engine.py`, `app/models/` |
| **M2** Intraday pattern → trading rules | Learn within-day patterns, emit executable rules | `app/pattern/`, `app/rules/` |
| **M3** Execution (auto/manual) | Buy/sell via miniQMT or SIM; mode toggle | `services/` |

Data flow:
`iFinD/akshare → data/adapters → CSV → data_loader → factor_engine → model → risk_filter → API → executor`

## Data source (iFinD primary, akshare fallback)

- **iFinD (同花顺)** — primary / production. Needs the iFinD terminal +
  `iFinDPy` (NOT on pip) + `IFIND_USER` / `IFIND_PASSWORD` env vars.
- **akshare** — free fallback / dev. Used automatically when iFinD is
  unavailable. Default for local development.

```bash
export AMINQT_DATA_SOURCE=ifind   # or akshare (default)
```

## Broker (miniQMT + simulator)

- **miniQMT (xtquant)** — real trading. Needs the miniQMT client +
  `xtquant` (NOT on pip). A-share T+1 enforced in `sync_portfolio`.
- **SIM** — prints orders only. Default.

```bash
export AMINQT_BROKER=xt            # or sim (default)
export AMINQT_EXEC_MODE=manual     # or auto (granted)
```

## Install

```bash
pip install -r requirements.txt
# iFinDPy and xtquant are NOT on pip — install via their terminals.
cp config/.env.example .env        # fill IFIND_USER / IFIND_PASSWORD
```

## Run

```bash
python scripts/download_data.py            # Phase 1: fetch K-line → data/raw/
uvicorn app.main:app --reload              # Phase 4: API at http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py         # Phase 5: research dashboard
```

## Test

```bash
pytest                                     # collected from tests/ (stubs skip until phases land)
```

## Phased roadmap

- **Phase 1** ✅ scaffold + data download (this commit)
- **Phase 2** — `factor_engine.build_features` (MACD/KDJ/BOLL/RSI + derived)
- **Phase 3** — LSTM + LightGBM training (strict time split)
- **Phase 4** — FastAPI `/select` + hard-constraint risk filter + scheduler
- **Phase 5** — Streamlit dashboard (Plotly K-line + signals)
- **Phase 6 / M3** — miniQMT execution (auto/manual)
- **M2** — intraday pattern learner + rule engine
