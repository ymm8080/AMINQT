"""_train_parallel_prob_head.py — 并行概率头训练 (每日自动化, 自判断新鲜度).

定案 (memory parallel-gbm-wf-verdict): 每 refit_every_days 交易日扩窗重训全局
LGBM 概率头 (mfe_3d >= abs_target 二分类), WORM bundle 落盘 data/prob_head/.
训练读 _diag_stage_{board}_3y.parquet (parallel 步骤当日检查点产物, 与短名单同源);
全史扩窗 = 面板全部行 (mfe_3d 尾段 NaN 行自动排除), 预测目标是次日及以后的决策日
→ 无前瞻. trailing 242d 训练=数据饥饿退化, 勿用.

半衰期集成 (2026-09-03 用户定案 ensB3): PROB_GATE["half_lives"] 逐档训练
(<board>_prob_hl<hl>_<ts>.joblib), 新鲜度逐档判断, serving 侧概率取均值.

用法: python scripts/_train_parallel_prob_head.py [--force]
自判断: 某档最新 bundle 距今 < refit_every_days 交易日 → skip 该档 (--force 强制重训).
WORM: <board>_prob_hl<hl>_<ts>.joblib, 旧 bundle 不覆盖不删除.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel import prob_head
from config.settings import DATA_DIR, PROB_GATE


def _load_board(board: str) -> pd.DataFrame | None:
    """同回测载入: 全部特征 + mfe_3d + label_pain (行序任意, 训练按行)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    need = [
        c
        for c in schema
        if not c.startswith("label_")
        and not c.startswith("pred_")
        and c not in prob_head.META
    ]
    need += ["symbol", "date", "label_pain"]
    t = pq.read_table(str(fp), columns=list(dict.fromkeys(need))).to_pandas()
    if t.empty:
        return None
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    # prob 头 extras (面板外合成列) 与 serving 同源合成 (8格 0912); 宽集 feature_cols
    # 自动收录. 全史扩窗 = per-date 截面 rank 在全训练史上合成, 与 gate_probabilities
    # 单日截面合成同公式 (add_quality_factor groupby date).
    t = prob_head.synthesize_prob_extras(t, board)
    t = prob_head._add_mfe_3d(t)
    return t


def _missing_feat_cols(bundle: dict | None, columns) -> list[str]:
    """[0912 夜] bundle 特征已不在面板的列 (schema 漂移检测, 供自愈重训)."""
    if bundle is None:
        return []
    have = set(columns)
    return [c for c in bundle.get("feat_cols", []) if c not in have]


def _shadow_prob(
    board: str, t: pd.DataFrame, eval_rows: pd.DataFrame, eval_lo
) -> np.ndarray:
    """purged 影子概率 (0913 purged_v1): eval 窗前推 RANK_SOURCE_PURGE_DAYS 交易日
    截断面板重拟合各半衰期档, 影子集预测 eval_rows.

    动机: 原口径用在役全史 bundle 评 eval 窗 → 尾部 eval 日 100% in-sample →
    prob 系统性虚高 (0912 首评双板选 prob, walk-forward #11 实测 mag 碾压).
    purge = buy_lag 1 + 10d 视界 = 11 交易日, 同 MAG10D_CAL realized_drop
    (calibration.py) 无前瞻口径; label_pain 3d 窗 < 11 亦全覆盖.
    任一档 fit raise → 向上抛 (调用方 fail-open).
    """
    from app.pipeline_parallel.config import RANK_SOURCE_PURGE_DAYS

    uniq = np.unique(t["date"].values)
    pos = int(np.searchsorted(uniq, pd.Timestamp(eval_lo).to_datetime64()))
    purge = int(RANK_SOURCE_PURGE_DAYS)
    if pos < purge:
        raise ValueError(
            f"[{board}] 面板日期不足以前推 {purge} 交易日 purge "
            f"(eval_lo={eval_lo}) -> 影子评估放弃 (fail-open)"
        )
    cut = uniq[pos - purge]
    train = t.loc[t["date"] <= cut]
    if train.empty:
        raise ValueError(f"[{board}] purged 影子训练集为空 (cutoff={cut})")
    shadow: list[dict] = []
    for hl in PROB_GATE["half_lives"]:
        print(
            f"[{board}] purged 影子拟合 hl={hl} "
            f"(train<={pd.Timestamp(cut):%Y-%m-%d}, {len(train):,} 行)",
            flush=True,
        )
        model, cols = prob_head._fit_cls_model(board, train, hl)
        shadow.append({"feat_cols": cols, "model": model})
    return prob_head.ensemble_predict(shadow, eval_rows).to_numpy()


def _eval_rank_source(board: str, t: pd.DataFrame, trained_through: str) -> None:
    """[0912 夜用户令] 重训后评估 mag/prob/blend 三键 → WORM json + 头选择台账行.

    mag = 服务同款 calibrate_mag10d (both 口径 score, 只用已实现标签, 无前瞻);
    prob = [0913 purged_v1] purged 影子重评 (_shadow_prob): eval 窗 cutoff 前推
    11 交易日截断重拟合, 消除在役全史 bundle 对 eval 尾段的 in-sample 虚高;
    指标 = trailing RANK_SOURCE_EVAL_DAYS 个已实现决策日逐视界 (3d/5d/10d,
    用户令) Spearman + TOP10 实得。argmax 加权 IC 自选 (平局→blend)。
    任何异常 → fail-open 跳过 (serving 维持 blend, 不杀链)。
    头名映射 (台账列共用): mag≈reg 幅度头, prob≈cls 概率头。
    """
    from app.pipeline_parallel import rank_source
    from app.pipeline_parallel.calibration import calibrate_mag10d
    from app.pipeline_parallel.config import RANK_SOURCE_EVAL_DAYS
    from scripts._head_choice_ledger import append_head_choice_row
    from scripts._shortlist_t5_t10 import _panel_per_stock

    try:
        bundles = prob_head.load_all_tiers(board)
        if bundles is None:
            print(
                f"[{board}] rank_source 评估跳过: 概率 bundle 不全 (fail-open)",
                flush=True,
            )
            return
        panel = _panel_per_stock().get((board, "both"))
        if panel is None or panel.empty:
            print(f"[{board}] rank_source 评估跳过: both 面板缺失", flush=True)
            return
        work = panel[["symbol", "date", "score", "label_pm_10d_net"]].copy()
        work["board"] = board
        mag = calibrate_mag10d(work)  # → [symbol, date, board, mag] (已实现边界内)
        # prob 只在评估窗行上预测 (trailing 已实现日 + 成熟余量), 特征取自训练帧 t
        feat_union = sorted(
            {c for b in bundles.values() for c in b.get("feat_cols", [])}
        )
        missing = [c for c in feat_union if c not in t.columns]
        if missing:
            raise ValueError(f"概率头特征缺 {len(missing)} 列: {missing[:5]}")
        use_dates = sorted(mag["date"].unique())[-(int(RANK_SOURCE_EVAL_DAYS) + 15) :]
        # [0913 purged_v1] prob 换 purged 影子重评: 在役 bundle 见过全部历史,
        # eval 尾段 in-sample → prob 虚高; 影子 = cutoff 前推 11 交易日重拟合.
        rows = t.loc[t["date"].isin(use_dates), ["symbol", "date"] + feat_union].copy()
        rows["prob"] = _shadow_prob(board, t, rows, use_dates[0])
        labels = panel[
            ["symbol", "date"] + [f"label_pm_{h}_net" for h in rank_source.HORIZONS]
        ]
        frame = mag.merge(
            rows[["symbol", "date", "prob"]], on=["symbol", "date"], how="left"
        ).merge(labels, on=["symbol", "date"], how="left")
        evaluation = rank_source.evaluate_keys(frame, eval_days=RANK_SOURCE_EVAL_DAYS)
        chosen = rank_source.choose_rank_key(evaluation)
        prev = rank_source.load_latest_rank_source(board)
        payload = {
            "board": board,
            "trained_through": trained_through,
            "chosen": chosen,
            "eval_mode": "purged_v1",
            "eval_days": int(RANK_SOURCE_EVAL_DAYS),
            "metrics": evaluation,
            "n_rows": int(len(frame)),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = rank_source.save_rank_source(board, payload)
        ics = {
            f"{h}_{s}": evaluation[k]["per_horizon"][h]["ic"]
            for h in rank_source.HORIZONS
            for s, k in (("reg", "mag"), ("cls", "prob"))
        }
        wrote = append_head_choice_row(
            "parallel",
            board,
            time.strftime("%Y%m%d"),
            weighted_ic_reg=evaluation["mag"]["weighted_ic"],
            weighted_ic_cls=evaluation["prob"]["weighted_ic"],
            chosen=chosen,
            gate_pass=True,
            switched=bool(prev is not None and prev.get("chosen") != chosen),
            ics=ics,
            n_features=len(feat_union),
        )
        print(
            f"[{board}] rank_source: chosen={chosen} "
            f"(mag={evaluation['mag']['weighted_ic']}, prob={evaluation['prob']['weighted_ic']}) "
            f"-> {path.name} (台账{'写入' if wrote else '已有跳过'})",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[{board}] rank_source 评估失败 (fail-open, serving 维持 blend): {exc}",
            flush=True,
        )


def main() -> int:
    force = "--force" in sys.argv[1:]
    ok = True
    for board in ("main", "dual"):
        t = _load_board(board)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            ok = False
            continue
        dates = np.unique(t["date"].values)
        latest = pd.Timestamp(dates[-1])
        for hl in PROB_GATE["half_lives"]:
            b = prob_head.load_latest_tier(board, hl)
            age = (
                None
                if b is None
                else prob_head.bundle_age_trading_days(dates, str(b["trained_through"]))
            )
            # [0912 夜] schema 漂移自愈: bundle 特征列被面板删列 (ths_* 8列清除后
            # 0910 批 bundle 5 列悬空) → serving predict() 直接 raise, 且新鲜度
            # 判据 (age<21 skip) 看不见 → 必须无视年龄强制重训.
            missing = _missing_feat_cols(b, t.columns)
            if missing:
                print(
                    f"[{board}/hl{hl}] 面板已缺 bundle 特征 {len(missing)} 列 "
                    f"(如 {missing[:3]}) → 重训 (schema 漂移自愈)",
                    flush=True,
                )
            if (
                not force
                and not missing
                and b is not None
                and age is not None
                and age < PROB_GATE["refit_every_days"]
            ):
                print(
                    f"[{board}/hl{hl}] skip: bundle 距今 {age} 交易日 < "
                    f"{PROB_GATE['refit_every_days']} (面板最新 {latest:%Y-%m-%d})",
                    flush=True,
                )
                continue
            path = prob_head.train_bundle(board, t, str(latest.date()), hl)
            n_pos = int((t["mfe_3d"] >= PROB_GATE["abs_target"]).sum())
            print(
                f"[{board}/hl{hl}] 训练 {len(t):,} 行 (正样本 {n_pos:,}, "
                f"特征 {len(prob_head.feature_cols(t.drop(columns=['mfe_3d'])))} 列) "
                f"-> {path.name}",
                flush=True,
            )
        # [0912 夜用户令] 每次运行都评估三键 (bundle 未重训日也可用新鲜面板复核)
        _eval_rank_source(board, t, str(latest.date()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
