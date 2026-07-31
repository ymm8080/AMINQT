"""Extra Gate D tests: min_features, sat threshold, label horizon variants."""
import json, os, sys, time, warnings
import pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb

TESTS = [
    # (label, board, min_feats, sat_pct)
    ('Main_min50', 'main', 50, 0.95),
    ('Main_sat80', 'main', 5, 0.80),
    ('Dual_min30', 'dual', 30, 0.95),
    ('Dual_5d_label', 'dual', 5, 0.95),  # use label_5d_net
]

OUT_DIR = 'data/abc_test_results'
results = {}

for test_name, board, min_feats, sat_pct in TESTS:
    label = 'label_5d_net' if '5d' in test_name else 'label_pm_1d_net'
    prebuilt = os.path.join(OUT_DIR, f'prebuilt_{board}.parquet')
    t0 = time.time()
    print(f'\n{"="*60}')
    print(f'Test: {test_name} | board={board} | min_feats={min_feats} | sat={sat_pct} | label={label}')
    print(f'{"="*60}')

    # Load
    df = pd.read_parquet(prebuilt)
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    all_feats = FeatureEngineV35.feature_columns(df)

    # NaN+Var base
    good = [c for c in all_feats if c in df.columns and df[c].isna().mean() < 0.95]
    base = [c for c in good if df[c].var() > 1e-8]
    print(f'Base (NaN+Var): {len(base)}/{len(all_feats)}')

    # Label & split
    if label not in df.columns:
        label = 'label_1d_net'
    dates = sorted(df['date'].unique())
    split = int(len(dates) * 0.75)
    train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
    test_df  = df[df['date'].isin(dates[split:])].dropna(subset=[label])

    X_tr = train_df[base].fillna(0); y_tr = train_df[label]
    X_te = test_df[base].fillna(0)

    # Train full model
    full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    full.fit(X_tr, y_tr)
    imp = pd.DataFrame({'feature': base, 'gain': full.booster_.feature_importance(importance_type='gain')})
    imp = imp.sort_values('gain', ascending=False)

    # Ablation
    def eval_icir(preds, df_te, lab):
        df_e = df_te.copy(); df_e['pred'] = preds
        ics = [spearmanr(g['pred'], g[lab])[0] for _, g in df_e.groupby('date') if len(g)>=10]
        a = np.array([x for x in ics if not np.isnan(x)])
        return float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

    ns = sorted(set([5, 10, 20, 30, 50, 75, 100, 150, 200, len(base), min_feats]))
    ablation_log = []
    best_n, best_icir = len(base), 0
    for n in ns:
        if n > len(base): continue
        top = imp.head(n)['feature'].tolist()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr[top], y_tr)
        ir = eval_icir(m.predict(X_te[top]), test_df, label)
        ablation_log.append({'n': n, 'icir': ir})
        if ir > best_icir: best_n, best_icir = n, ir
        print(f'  abl n={n:>3}: ICIR={ir:.4f}')

    # Saturation at specified pct
    sat_n = ns[0]
    for log in ablation_log:
        if log['icir'] >= best_icir * sat_pct:
            sat_n = log['n']; break
    sat_n = max(sat_n, min_feats)

    top_feats = imp.head(sat_n)['feature'].tolist()
    print(f'Best ICIR={best_icir:.4f} at n={best_n}, sat({sat_pct:.0%}) at n={sat_n} (clamped ≥{min_feats})')

    # Final train with selected features
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_tr[top_feats], y_tr)
    preds = model.predict(X_te[top_feats])

    # Evaluate
    df_e = test_df.copy(); df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[label])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    ic = float(round(a.mean(), 5))
    icir_final = float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

    daily_top = [g.nlargest(10, 'pred') for _, g in df_e.groupby('date')]
    top_df = pd.concat(daily_top)
    rets = top_df[label].dropna()
    sharpe = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    winrate = float((rets>0).mean())
    composite = round(icir_final*0.40 + sharpe*0.35 + winrate*0.25, 4)

    result = {
        'test': test_name, 'board': board, 'min_feats': min_feats, 'sat_pct': sat_pct,
        'label': label, 'n_base': len(base), 'n_selected': len(top_feats),
        'best_icir_abl': best_icir, 'sat_n': sat_n,
        'oos_ic': ic, 'oos_icir': icir_final,
        'top10_sharpe': round(sharpe,4), 'top10_win_rate': round(winrate,4),
        'composite': composite, 'elapsed_s': round(time.time()-t0, 1),
        'ablation': ablation_log,
    }
    results[test_name] = result
    print(f'FINAL: IC={ic:+.5f} ICIR={icir_final:.4f} Sharpe={sharpe:.2f} WinRate={winrate:.1%} Composite={composite:.4f}')
    print(f'DONE in {result["elapsed_s"]:.0f}s')

# Summary
print(f'\n{"="*70}')
print(f'GATE D EXTRA TESTS — SUMMARY')
print(f'{"="*70}')
print(f'{"Test":<20} {"Board":<6} {"Label":<16} {"Base":>5} {"Sel":>5} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"Comp":>8}')
print('-'*70)
for name, r in sorted(results.items(), key=lambda x: -x[1]['composite']):
    print(f'{name:<20} {r["board"]:<6} {r["label"]:<16} {r["n_base"]:>5} {r["n_selected"]:>5} '
          f'{r["oos_ic"]:>+9.5f} {r["oos_icir"]:>7.4f} {r["top10_sharpe"]:>7.2f} {r["composite"]:>8.4f}')

out = os.path.join(OUT_DIR, f'gate_extra_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nSaved: {out}')
