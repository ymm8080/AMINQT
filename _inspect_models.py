"""Inspect model tuple internals."""
import pickle, os

models_dir = 'D:\\AMINQT\\AMINQT CODES\\models\\pipeline1'
candidates = sorted([f for f in os.listdir(models_dir) if f.startswith('main_20260728')])

for fname in candidates:
    path = os.path.join(models_dir, fname)
    with open(path, 'rb') as fp:
        m = pickle.load(fp)
    sz_mb = os.path.getsize(path) / 1e6
    print(f'=== {fname} ({sz_mb:.1f}MB) ===')
    print(f'  feature_cols: {len(m["feature_cols"])}')

    # Inspect the tuple models
    for mk, mv in m['models'].items():
        print(f'  {mk}: tuple(len={len(mv)})')
        if isinstance(mv, tuple):
            for i, v in enumerate(mv):
                vt = type(v).__name__
                if hasattr(v, 'n_features_in_'):
                    extra = f' nf={v.n_features_in_}'
                elif hasattr(v, '__len__'):
                    extra = f' len={len(v)}'
                else:
                    extra = ''
                print(f'    [{i}]: {vt}{extra}')

    # rank_model
    rm = m['rank_model']
    print(f'  rank_model: tuple(len={len(rm)})')
    if isinstance(rm, tuple):
        for i, rmi in enumerate(rm):
            rmt = type(rmi).__name__
            rmnf = getattr(rmi, 'n_features_in_', getattr(rmi, '__len__', lambda: None)() if hasattr(rmi, '__len__') else '?')
            print(f'    [{i}]: {rmt} nf={rmnf}')
            if i == 0 and hasattr(rmi, 'n_features_in_'):
                print(f'           coef_={hasattr(rmi, "coef_")}, feature_importances_={hasattr(rmi, "feature_importances_")}')
    print()
