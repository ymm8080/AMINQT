import importlib

mods = {
    "talib": "TA-Lib",
    "shap": "shap",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "stable_baselines3": "SB3",
    "gymnasium": "gymnasium",
    "tushare": "tushare",
    "playwright": "playwright",
    "statsmodels": "statsmodels",
    "onnxruntime": "onnxruntime",
    "qlib": "pyqlib",
    "alphalens": "alphalens",
    "pypfopt": "pyportfolioopt",
    "optuna": "optuna",
    "mlflow": "mlflow",
    "backtrader": "backtrader",
    "pyfolio": "pyfolio",
    "empyrical": "empyrical",
    "transformers": "transformers",
    "peft": "peft",
}
for m, pkg in mods.items():
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"{pkg:18s} OK   {ver}")
    except Exception as e:
        print(f"{pkg:18s} FAIL {type(e).__name__}: {e}")
