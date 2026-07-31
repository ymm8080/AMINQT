# Pipeline1 Three-Layer Feature Pipeline Design

**Date**: 2026-07-30
**Author**: Claude Code (ABC Test + Architecture Design)
**Status**: Implemented

---

## 1. Overview

Three layers with distinct frequencies and data windows:

| | Layer 1: Build | Layer 2: Select | Training |
|---|---|---|---|
| **Frequency** | Monthly / On-demand | Monthly (after Layer1) | Weekly / Daily |
| **Duration** | ~3-5 min | ~1-2 min | ~30s |
| **Trigger** | Panel update / Manual | After Layer1 / Manual | Cron scheduled |

### Data Windows per Board

| Board | Data Window | Reason |
|-------|------------|--------|
| **MAIN** | **3 years** (rolling, no cutoff) | Cross-regime training produces positive Sharpe |
| **DUAL** | **1 year** (rolling, last 365 days) | Gate D overfits on longer data |

```
Layer1 (Monthly)  -->  features_{board}_{ts}.parquet + registry_{ts}.json
Layer2 (Monthly)  -->  selected_{board}_{ts}.json + selected_{board}_current.json
Training (Weekly) -->  Read current.json --> LightGBM --> Model bundle
```

---

## 2. Layer 1: Feature Build (per board)

### Step 1: Update Registry (sync with panel)

- **Add**: Discover new panel columns not in registry. Register with `status=active`.
- **Remove**: Detect registry columns no longer in panel. Mark `status=removed` with timestamp.
- **No IC gate** on adoption. Pure registration/de-registration.
- Output: `data/factor_registry/registry_{timestamp}.json` (one file, full market)

### Step 2A: MAIN Board (3-year data)

- Load 3 years of panel data (no date cutoff)
- Select main board stocks (60/00/002)
- Generate ~3,200 brute-force columns from ALL ~100 eligible numeric raw columns
- Transforms: pct_change(1,2,3,5,10,20,40,60), rolling_mean(5,10,20,40,60), rolling_std(5,10,20,40), rolling_max/min(10,20,40), diff(1,5,20), momentum(5,20,40), EMA(5,20,40)
- Output: `features_main_{timestamp}.parquet`

### Step 2B: DUAL Board (1-year data)

- Load 1 year of panel data (last 365 days)
- `FeatureEngineV35.feature_columns()` to get curated features (~420)
- Select dual board stocks (300/68)
- Output: `features_dual_{timestamp}.parquet`

### CLI

```bash
python scripts/build_features.py                    # Full build both boards (auto 3Y/1Y)
python scripts/build_features.py --board main        # MAIN only
python scripts/build_features.py --board dual        # DUAL only
python scripts/build_features.py --adoption-only     # Only sync registry
python scripts/build_features.py --data-window 1Y    # Override window
```

### Dim Gating

`config/feature_selector.json` controls per-dim on/off. `false` = completely excluded from pipeline.

### Auto-Adoption
Bidirectional: add + remove. No IC gate. Removed columns marked `status=removed` with `removed_at`.

---

## 3. Layer 2: Feature Select (per board)

Reads Layer1 latest outputs. Board-specific selection. User controls KEEP vs UPDATE.

### MAIN Pipeline

```
features_main_{latest}.parquet (~3,200 cols)
  --> NaN filter (>95% NaN drop)
  --> Dedup L2 (|r| > 0.7 within same base column group)
  --> ~1,069 selected features
```

### DUAL Pipeline

```
features_dual_{latest}.parquet (~420 cols)
  --> NaN filter (>95% NaN drop)
  --> Gate D (importance forward ablation, 95% saturation, min=30)
  --> ~30 selected features
```

### Outputs
- `selected_{board}_{timestamp}.json` — immutable version record
- `selected_{board}_current.json` — pointer to active version

### CLI

```bash
python scripts/select_features.py --board main --update        # Run + diff + confirm
python scripts/select_features.py --board main --update --dry-run  # Preview only
python scripts/select_features.py --board main --keep           # Keep current
python scripts/select_features.py --board main --status         # View status
python scripts/select_features.py --board main --history        # Version history
python scripts/select_features.py --board main --rollback <ts>  # Revert
```

### KEEP vs UPDATE

On `--update`, shows diff vs current version. User types `y` to activate, `n` to save as draft.

---

## 4. Training Layer

Reads `selected_{board}_current.json` directly. No feature computation.

```bash
python scripts/train_and_predict.py --train                    # Default: read current
python scripts/train_and_predict.py --train --feature-version 20260729T200000  # Specific version
```

---

## 5. File Structure

```
data/factor_registry/
  registry_{timestamp}.json                    # Layer1: full market registry
  features_main_{timestamp}.parquet             # Layer1: MAIN (~3,200 cols)
  features_dual_{timestamp}.parquet             # Layer1: DUAL (~420 cols)
  selected_main_{timestamp}.json               # Layer2: MAIN selection
  selected_main_current.json                    # Layer2: MAIN pointer
  selected_dual_{timestamp}.json               # Layer2: DUAL selection
  selected_dual_current.json                    # Layer2: DUAL pointer
```

Layer1 and Layer2 timestamps NOT required to match. Layer2 reads latest Layer1.

---

## 6. Verified Results (ABC Testing 2026-07-29/30)

| | MAIN (3Y) | DUAL (1Y) |
|---|---|---|
| **Pipeline** | NaN -> 3,200 brute -> Dedup L2 | NaN -> 420 curated -> Gate D |
| **Pool -> Selected** | 3,243 -> 1,069 | 429 -> 30 |
| **OOS IC** | +0.004 | +0.007 |
| **ICIR** | 0.017 | 0.045 |
| **Top10 Sharpe** | **+0.32** | -0.41 |
| **Top10 WinRate** | **49.8%** | 48.1% |
| **Composite** | **+0.243** | -0.005 |

### Decisions Tested and Rejected

- IC screening (pre/post): always degrades
- Var filter: always removes useful signals
- Gate D on MAIN: all configs < Dedup L2
- 5000-level transforms (+z/skew/kurt): degrades MAIN from +0.243 to -0.028
- Gate D + Dedup combo: worse than solo

---

## 7. Implementation Files

| File | Status |
|------|--------|
| `app/pipeline1/feature_selector.py` | Created |
| `scripts/build_features.py` | Created |
| `scripts/select_features.py` | Created |
| `config/feature_selector.json` | Created |
| `app/pipeline1/train_runner.py` | Modified (feature_list_path param) |
