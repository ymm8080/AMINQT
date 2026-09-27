# -*- coding: utf-8 -*-
"""首板点名三头复活 P0 (0924) — 扩窗重训 + ensB3 半衰期集成 + 停训看门狗.

病理 (b614cd85 入库备注): _firstboard_pages --train 手动无调度, 生产三头
train_end=2025-12-31, 停训 ~9 个月 (~180 交易日) 无告警 — promo_yd/板块结构等
情绪生态漂移无捕获, 首板点名页 p_k3 读数实际基于陈旧分布.

配方移植 (照抄 app/pipeline_parallel/prob_head.py 2026-09-03 ensB3 定案):
  - 时间衰减样本权重 decay_sample_weights (单一来源 import): w = 0.5**(age_days/hl),
    自然日, age 相对 fit 集最新日;
  - 多档半衰期逐档训练, serving 概率 = 各档算术均值 (ensemble_predict 语义);
  - 全史扩窗 (parallel 定案: trailing 242d = 数据饥饿退化, 勿用);
  - 隔离验证尾早停: LEGACY_PROB_GATE 配方 (val_days=30 交易日 + es_patience=50
    + 地板 50 轮).

两阶段 (赢了才晋级 — 绝不动生产 models/firstboard/):
  Phase A 配方 AB (公平: 所有候选只见 <=2025-12-31 数据 = 生产 v1 同 cutoff,
    TE = 2026-01-01..面板最新, 指标口径逐行复刻 v1 train_and_save 的 TE 报告):
    候选 = OLD(生产 booster) / 单档 hl∈{None,7,15,30,60,120} / 集成 {7,15,30}
    {15,30,60} {30,60,120} (集成成员复用单档, 零额外训练).
  Phase B 落 staging models/firstboard_v2/ (默认仅当胜出 k3@1 且 auc_k3 双超 OLD,
    --force 可越过): winner 配方全史重训 (标签尾自然 NaN → 无前瞻), reload 复现
    断言 (磁盘=内存), meta.json 带 ensemble 块 — _firstboard_pages.load_models
    向后兼容 (无 ensemble 键 = v1 单文件路径, 零行为变化).

晋级 (用户令后才执行): 拷 v2 四类文件覆盖 models/firstboard/ 后跑
  _firstboard_pages --train 校验? 否 — v1 验收线 _EXPECT_TE 属旧口径, 晋级后
  需以新 TE 重锚验收线 (同 0923 rank_calib 先例); 本脚本 meta 已登记 Phase A
  TE 证据 (te_top1_k3 / te_auc_k3 字段语义保留).

看门狗 (防再停训无人知, 本脚本不负责 — 已进 _firstboard_pages.serve_firstboard):
  train_end 落后面板 > FB_MAX_STALE_DAYS=42 交易日 → log.error 大声告警,
  fail-open 页照出; GENIOUS 首板 banner 动态尾注 (scripts/_genious_excel.py).

Usage:
    python scripts/_firstboard_retrain.py                   # Phase A AB (+胜出则 B)
    python scripts/_firstboard_retrain.py --ab-only         # 只报告不落 staging
    python scripts/_firstboard_retrain.py --tiers 15,30,60  # 手动档位 (跳过自动胜者)
    python scripts/_firstboard_retrain.py --force           # 未胜出/已存在也落/覆盖
    python scripts/_firstboard_retrain.py --from-report tmp_t/firstboard_retrain_<ts>.json
        # 复用归档 Phase A 证据只跑 Phase B — 超时中断恢复 / 面板未动时省 8 分钟重跑
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline_parallel.prob_head import decay_sample_weights  # noqa: E402
from config.settings import PANEL_V3_PATH, PROJECT_ROOT  # noqa: E402
from scripts import _firstboard_pages as fbp  # noqa: E402

CUTOFF = pd.Timestamp(fbp.VA_END)  # 2025-12-31 — 与生产 v1 完全同界 (公平 AB 的根)
HEADS = ("k2", "k3", "next_board")  # 顺序 = _firstboard_pages._predict 的三头顺序
SINGLE_HLS: list[int | None] = [None, 7, 15, 30, 60, 120]
ENSEMBLES = [(7, 15, 30), (15, 30, 60), (30, 60, 120)]  # ensB3 镜像 + 放宽两档
VAL_DAYS = 30  # 隔离验证尾 (交易日) — LEGACY_PROB_GATE es 配方
ES_PATIENCE = 50  # 同 v1 train_and_save 早停耐心
MIN_ROUNDS = 50  # 早停地板 (LEGACY_PROB_GATE ③+④: 地板 50)
STAGE_DIR = Path(PROJECT_ROOT) / "models" / "firstboard_v2"
TMP_T = Path(PROJECT_ROOT) / "tmp_t"


def _prepare(df: pd.DataFrame):
    """全史事件 + 特征矩阵 (逐行复刻 v1 train_and_save 的 A/X/cats 口径).

    A = hist_ok ∩ mature10 ∩ k2/next 非 NaN ∩ need6 非 NaN (v1 同滤);
    cats 建在全量 A (含 TE 行, v1 line 334 同源); X 带 Categorical sw_l2."""
    ev = fbp.build_event_features(df)
    need6 = ["wr1", "ret60", "gap", "seal_hard", "cmv", "ftr"]
    A = ev[
        ev["hist_ok"]
        & ev["mature10"]
        & ev["k2"].notna()
        & ev["next_board"].notna()
        & ev[need6].notna().all(axis=1)
    ].copy()
    cats = sorted(A["sw_l2"].unique().tolist())
    A["sw_l2"] = pd.Categorical(A["sw_l2"], categories=cats)
    X = A[fbp.FEATS + ["sw_l2"]].copy()
    for c in fbp.FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return A, X, cats


def _fit_head(
    A: pd.DataFrame,
    X: pd.DataFrame,
    tgt: str,
    fit_mask: pd.Series,
    hl: int | None,
) -> object:
    """单头单档 LGBM (native lgb.train, 参数映射逐项复刻 v1 0923 修复A).

    fit 行内隔离验证尾 VAL_DAYS 个交易日做早停 (验证行不进训练);
    hl=None 不加权 (原 v1 行为), 否则 decay_sample_weights (age 相对 fit 集最新日,
    与 prob_head._fit_cls_model 同口径 — 权重数组按 fit 序计算后取训练子集)."""
    import lightgbm as lgb

    fit_idx = A.index[fit_mask & A[tgt].notna()]
    dates_fit = A.loc[fit_idx, "date"]
    uniq = sorted(dates_fit.unique())
    if len(uniq) <= VAL_DAYS:
        raise ValueError(f"[{tgt}] fit 交易日不足 ({len(uniq)} ≤ {VAL_DAYS})")
    val_start = uniq[-VAL_DAYS]
    m_va = (dates_fit >= val_start).to_numpy()
    m_tr = ~m_va
    Xf = X.loc[fit_idx]
    yf = A.loc[fit_idx, tgt]
    w = None
    if hl is not None:
        w = decay_sample_weights(dates_fit, hl)[m_tr]
    dtr = lgb.Dataset(
        Xf[m_tr],
        yf[m_tr],
        weight=w,
        categorical_feature=["sw_l2"],
        free_raw_data=False,
    )
    dva = lgb.Dataset(
        Xf[m_va],
        yf[m_va],
        reference=dtr,
        categorical_feature=["sw_l2"],
        free_raw_data=False,
    )
    native_params = dict(
        objective=fbp.PARAMS["objective"],
        num_leaves=fbp.PARAMS["num_leaves"],
        learning_rate=fbp.PARAMS["learning_rate"],
        min_child_samples=fbp.PARAMS["min_child_samples"],
        seed=fbp.PARAMS["random_state"],
        metric="binary_logloss",
        verbosity=fbp.PARAMS["verbose"],
    )
    mdl = lgb.train(
        native_params,
        dtr,
        num_boost_round=fbp.PARAMS["n_estimators"],
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)],
    )
    if mdl.best_iteration and mdl.best_iteration < MIN_ROUNDS:
        # 地板: 早停过狠 (<50 轮) → 定轮重训, 防欠拟合候选混进 AB
        mdl = lgb.train(
            native_params,
            dtr,
            num_boost_round=MIN_ROUNDS,
            callbacks=[],
        )
    return mdl


def _te_metrics(ev_te: pd.DataFrame, p_k2, p_k3, p_nb) -> dict:
    """TE 指标 — 逐行复刻 v1 train_and_save 的报告块 (k2@1/k2@3/k3@1/auc_next/auc_k3)."""
    from sklearn.metrics import roc_auc_score

    ev_te = ev_te.assign(p_k2=p_k2, p_k3=p_k3, p_nb=p_nb)
    hit1 = hit3 = tot1 = k3_hit1 = 0
    for _, grp in ev_te.groupby("date"):
        top = grp.sort_values("p_k2", ascending=False).head(min(3, len(grp)))
        hit3 += top["k2"].sum()
        tot1 += len(top)
        hit1 += grp.sort_values("p_k2", ascending=False).head(1)["k2"].sum()
        k3_hit1 += grp.sort_values("p_k3", ascending=False).head(1)["k3"].sum()
    k2_1 = 100 * hit1 / max(ev_te["date"].nunique(), 1)
    k2_3 = 100 * hit3 / tot1
    k3_1 = 100 * k3_hit1 / max(ev_te["date"].nunique(), 1)
    auc_next = roc_auc_score(ev_te["next_board"], p_nb)
    auc_k3 = (
        roc_auc_score(ev_te["k3"], p_k3) if ev_te["k3"].nunique() > 1 else float("nan")
    )
    return {
        "k2@1": k2_1,
        "k2@3": k2_3,
        "k3@1": k3_1,
        "auc_next": auc_next,
        "auc_k3": auc_k3,
        "te_days": int(ev_te["date"].nunique()),
        "te_n": int(len(ev_te)),
    }


def _cand_preds(models_by_head: dict, Xte: pd.DataFrame) -> dict:
    """候选三头 TE 概率 — 各头档位算术均值 (prob_head.ensemble_predict 语义)."""
    return {
        h: np.mean([m.predict(Xte) for m in models_by_head[h]], axis=0) for h in HEADS
    }


def _load_old() -> tuple[dict | None, dict | None]:
    """生产三头 (走 _firstboard_pages.load_models 新契约 = 服务同源; 缺 → (None, None))."""
    try:
        boosters, meta = fbp.load_models(fbp.MODELS_DIR)
    except FileNotFoundError:
        return None, None
    return (
        {"k2": boosters[0], "k3": boosters[1], "next_board": boosters[2]},
        meta,
    )


def _nan_safe(v: float) -> float:
    return v if v == v else -1.0  # NaN → -1 (TE 单类时排序不炸)


def _rank_key(m: dict) -> tuple:
    """胜者排序键: k3@1 (页主排序键) > auc_k3 > k2@1 (同 v1 meta 登记的两线优先)."""
    return (_nan_safe(m["k3@1"]), _nan_safe(m["auc_k3"]), _nan_safe(m["k2@1"]))


def run_phase_a(A: pd.DataFrame, X: pd.DataFrame) -> tuple[dict, str, dict | None]:
    """Phase A: 同 cutoff 公平 AB → (全候选指标表, 胜者名, OLD 指标)."""
    te_mask = A["date"] > CUTOFF
    fit_mask = A["date"] <= CUTOFF
    Xte = X[te_mask]
    ev_te = A[te_mask]
    print(
        f"[retrain] Phase A: fit≤{CUTOFF.date()} n={int(fit_mask.sum())} | "
        f"TE>{CUTOFF.date()} n={int(te_mask.sum())} ({ev_te['date'].nunique()} 交易日)"
    )

    singles: dict[int | None, dict] = {}
    for hl in SINGLE_HLS:
        singles[hl] = {tgt: _fit_head(A, X, tgt, fit_mask, hl) for tgt in HEADS}
        tag = "hlNone" if hl is None else f"hl{hl}"
        print(f"[retrain]   单档 {tag} 训练完成")

    cands: dict[str, dict] = {}
    old_models, old_meta = _load_old()
    if old_models is not None:
        p = _cand_preds(old_models, Xte)
        cands["OLD(生产)"] = _te_metrics(ev_te, p["k2"], p["k3"], p["next_board"])
    for hl in SINGLE_HLS:
        tag = "hlNone" if hl is None else f"hl{hl}"
        p = _cand_preds({t: [m] for t, m in singles[hl].items()}, Xte)
        cands[tag] = _te_metrics(ev_te, p["k2"], p["k3"], p["next_board"])
    for tiers in ENSEMBLES:
        name = "ens" + "_".join(str(h) for h in tiers)
        merged = {tgt: [singles[h][tgt] for h in tiers] for tgt in HEADS}
        p = _cand_preds(merged, Xte)
        cands[name] = _te_metrics(ev_te, p["k2"], p["k3"], p["next_board"])

    winner = max(
        (n for n in cands if n != "OLD(生产)"), key=lambda n: _rank_key(cands[n])
    )
    old_m = cands.get("OLD(生产)")
    return cands, winner, old_m


def run_phase_b(
    A: pd.DataFrame,
    X: pd.DataFrame,
    cats: list,
    tiers: list[int],
    phase_a: dict,
    stage_name: str,
    te_ev: dict,
    out_dir: Path,
    force: bool,
) -> dict:
    """Phase B: winner 配方全史重训 → staging (WORM: 已存在非 force 不覆盖).

    tiers=[] → v1 格式 (单文件无 ensemble 键, 即 v1 配方扩窗版 — v1 --train 的
    TR/VA 窗口是硬编码常量, 重跑不会扩窗, 扩窗只能走这里);
    tiers 非空 → ensB3 格式 (hl 后缀多文件 + ensemble 块)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "meta.json").exists():
        if not force:
            raise SystemExit(
                f"[retrain] staging 已存在 ({out_dir}) — 晋级请人工拷贝; "
                "重跑加 --force (旧文件转同目录 backup_<ts>/)"
            )
        bak = out_dir / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
        bak.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            if f.is_file():
                f.rename(bak / f.name)
        print(f"[retrain] 旧 staging → {bak.name}/")

    fit_all = pd.Series(True, index=A.index)
    saved: dict[str, list] = {}
    if not tiers:  # 无衰减单档 → v1 单文件命名
        for tgt, fname in (
            ("k2", "booster_k2.txt"),
            ("k3", "booster_k3.txt"),
            ("next_board", "booster_next.txt"),
        ):
            mdl = _fit_head(A, X, tgt, fit_all, None)
            mdl.save_model(str(out_dir / fname))
            saved[tgt] = [mdl]
        print("[retrain]   Phase B 无衰减单档 (v1 格式扩窗) 三头落盘")
    else:
        for hl in tiers:
            for tgt in HEADS:
                mdl = _fit_head(A, X, tgt, fit_all, hl)
                mdl.save_model(str(out_dir / f"booster_{tgt}_hl{hl}.txt"))
                saved.setdefault(tgt, []).append(mdl)
            print(f"[retrain]   Phase B 档 hl{hl} 三头落盘")

    meta = {
        "features": fbp.FEATS,
        "sw_l2_categories": cats,
        "train_end": str(A["date"].max().date()),
        "params": {k: v for k, v in fbp.PARAMS.items() if k != "verbose"},
        "recipe": {
            "tiers": list(tiers),
            "val_days": VAL_DAYS,
            "es_patience": ES_PATIENCE,
            "min_rounds": MIN_ROUNDS,
            "decay": "w=0.5**(age_days/hl) 自然日 (app.pipeline_parallel.prob_head)"
            if tiers
            else "无衰减 (v1 配方扩窗)",
            "phase_a_cutoff": str(CUTOFF.date()),
        },
        "phase_a": phase_a,
        "winner": stage_name,
        "te_top1_k3": round(_nan_safe((te_ev or {}).get("k3@1", float("nan"))), 1),
        "te_auc_k3": round(_nan_safe((te_ev or {}).get("auc_k3", float("nan"))), 4),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if tiers:
        meta["ensemble"] = {
            tgt: [f"booster_{tgt}_hl{h}.txt" for h in tiers] for tgt in HEADS
        }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # reload 复现断言 (v1 同款防两源分裂: 落盘口径 ≠ 内存口径即 fail)
    boosters, _meta2 = fbp.load_models(out_dir)
    idx = X.index[-200:]
    for i, tgt in enumerate(HEADS):
        p_mem = np.mean([m.predict(X.loc[idx]) for m in saved[tgt]], axis=0)
        p_disk = np.mean([m.predict(X.loc[idx]) for m in boosters[i]], axis=0)
        d = float(np.max(np.abs(p_mem - p_disk)))
        if d > 1e-9:
            raise RuntimeError(
                f"[retrain] {tgt} reload 复现分裂 max|Δ|={d:.2e} — 禁上线"
            )
    print(f"[retrain] reload 复现 OK (三头 max|Δ|<1e-9) → {out_dir}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ab-only", action="store_true", help="只跑 Phase A 报告, 不落 staging"
    )
    ap.add_argument(
        "--tiers", default=None, help="手动档位 (如 15,30,60); 默认 Phase A 胜者"
    )
    ap.add_argument("--force", action="store_true", help="未胜出/已有 staging 也落")
    ap.add_argument(
        "--out-dir", default=None, help="staging 目录 (默认 models/firstboard_v2)"
    )
    ap.add_argument(
        "--from-report",
        default=None,
        help="复用归档 Phase A 报告 (tmp_t/firstboard_retrain_*.json) 只跑 Phase B",
    )
    args = ap.parse_args()

    df = fbp.load_mainboard(PANEL_V3_PATH, tail_dates=None)
    # 开场哨兵: 生产模型停训状态 (与 serve_firstboard 看门狗同源)
    age = fbp.model_stale_trading_days(fbp.MODELS_DIR, df)
    print(
        f"[retrain] 面板最新 {df['date'].max().date()} | 生产模型停训 "
        f"{'?' if age is None else age} 交易日 (阈值 {fbp.FB_MAX_STALE_DAYS})"
    )

    A, X, cats = _prepare(df)
    if args.from_report:
        if args.ab_only:
            raise SystemExit("[retrain] --from-report 与 --ab-only 互斥 (报告已归档)")
        rpt = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
        cands, winner = rpt["phase_a"], rpt["winner"]
        old_m = cands.get("OLD(生产)")
        print(
            f"[retrain] Phase A 复用归档报告 {args.from_report} (winner={winner}) — "
            "面板若已更新则证据过期, 勿复用"
        )
    else:
        cands, winner, old_m = run_phase_a(A, X)

    cols = ["k3@1", "auc_k3", "k2@1", "k2@3", "auc_next", "te_days", "te_n"]
    tbl = pd.DataFrame(cands).T[cols].sort_values(["k3@1", "auc_k3"], ascending=False)
    print("\n===== Phase A 同 cutoff 公平 AB (TE 2026, 指标口径=v1) =====")
    print(tbl.round(4).to_string())

    # WORM 报告落盘 (tmp_t 约定, 决策证据可回溯; 复用归档时不重写 — 原报告即证据)
    if not args.from_report:
        TMP_T.mkdir(exist_ok=True)
        rpt_path = TMP_T / f"firstboard_retrain_{datetime.now():%Y%m%d_%H%M%S}.json"
        rpt_path.write_text(
            json.dumps(
                {"phase_a": cands, "winner": winner}, ensure_ascii=False, indent=1
            ),
            encoding="utf-8",
        )
        print(f"[retrain] 报告 → {rpt_path}")

    if args.ab_only:
        return 0

    # ── 选实际落盘配方 (自动胜者或 --tiers) + 其 Phase A 证据键 ──
    if args.tiers:
        tiers = [int(x) for x in args.tiers.split(",")]
        stage_name = f"manual({args.tiers})"
    else:
        stage_name = winner
        if winner == "hlNone":
            tiers = []
        elif winner.startswith("hl"):
            tiers = [int(winner[2:])]
        else:  # ensA_B_C
            tiers = [int(x) for x in winner[3:].split("_")]
    if not tiers:
        cand_key = "hlNone"
    elif len(tiers) == 1:
        cand_key = f"hl{tiers[0]}"
    else:
        cand_key = "ens" + "_".join(str(t) for t in tiers)
    te_ev = cands.get(cand_key)  # 未测组合 → None: 无 Phase A 证据, 须 --force

    beats_old = old_m is None or (
        te_ev is not None
        and _nan_safe(te_ev["k3@1"]) > _nan_safe(old_m["k3@1"])
        and _nan_safe(te_ev["auc_k3"]) > _nan_safe(old_m["auc_k3"])
    )
    if not beats_old and not args.force:
        print(
            f"[retrain] 配方 {stage_name} 未双超 OLD (k3@1 & auc_k3) — 按铁律不落 "
            "staging; 数字翻上面的表, 强行落加 --force"
        )
        return 1
    if te_ev is None and args.force:
        print(f"[retrain] ⚠ {stage_name} 无 Phase A 证据 (未测组合), --force 落盘")

    out_dir = Path(args.out_dir) if args.out_dir else STAGE_DIR
    meta = run_phase_b(A, X, cats, tiers, cands, stage_name, te_ev, out_dir, args.force)
    print(
        f"\n[retrain] Phase B 完成: {out_dir} (tiers={tiers or '无衰减'}, "
        f"train_end={meta['train_end']}) — serve 冒烟:"
    )
    smoke = fbp.serve_firstboard(df, models_dir=out_dir)
    print(smoke.head(5).to_string(index=False))
    print(
        "\n[retrain] 晋级 (用户令后): 拷 v2 全部文件覆盖 models/firstboard/ "
        "(git 可回滚); 验收线 _EXPECT_TE 需按新 TE 重锚 (0923 rank_calib 先例)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
