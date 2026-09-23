# -*- coding: utf-8 -*-
"""0923S hazard 头部反选拆解 — 回答「模型认得出再封板, 为什么还是亏钱」。

背景: 0923P A/B 三档全否, 但顶部十分位的**板率是提高的**(T+5 22.7% vs 基线 14.8%, 1.5x),
收益却是负的且不如全池。两个读数同时成立 ⇒ 只能是「失败尾巴被打得更狠」。
本脚本把头部的期望收益按 A/B 拆开, 定位亏损到底来自哪一半。

期望收益恒等式 (无假设, 纯分解):
    E[ret | head] = P(A|head)*E[ret|A,head] + (1-P(A|head))*E[ret|B,head]
三项分别在 head 内和全池内各算一次, 即可看出是「板率不够」还是「A/B 本身的收益被选差了」。

用法: python tmp_t/_0923S_hazard_head_decomp.py
"""

import gc
import json
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import _0923P_hazard_head as H
from app.pipeline1.ram_guard import check_startup_gate
from config.settings import RETRAIN_RAM_GUARD_MIN_FREE_GB, data_others_path
from scripts._run_guard import find_conflicts

TAG = "hazard_head_decomp_0923S"
DEC = 0.9          # 头部十分位阈值 (与 0923P econ 读数同口径)

log = logging.getLogger(TAG)


def _setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(os.path.join(HERE, "_0923S_hazard_head_decomp.log"),
                                  mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def decomp(ret: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> dict:
    """在 mask 内按 lab 拆期望收益 (恒等式, 不建模型)。"""
    r, y = ret[mask], lab[mask]
    ok = np.isfinite(r)
    r, y = r[ok], y[ok]
    if not len(r):
        return dict(n=0)
    a, b = y == 1, y == 0
    return dict(n=int(len(r)), a_rate=float(a.mean()), ret=float(r.mean()),
                a_n=int(a.sum()), a_ret=float(r[a].mean()) if a.any() else float("nan"),
                b_n=int(b.sum()), b_ret=float(r[b].mean()) if b.any() else float("nan"))


def main() -> int:
    _setup_logging()
    conflicts = find_conflicts()
    if conflicts:
        c = conflicts[0]
        log.error("[guard] 重活冲突: %s (PID %d) — 退出", c["sentinel"], c["pid"])
        return 3
    check_startup_gate(int(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024 ** 3))

    ev = H.load_events()
    P = H.load_panel(set(ev["symbol"]))
    M = H.build_frame(ev, P)
    del P
    gc.collect()

    X_d0, X_all, lab, uni, fill_ratio = H.build_features(M)
    d0 = M["d0"]
    log.info("[uni] n=%d  fill=%.1f%%  head=pred_te 分位>%.1f", int(uni.sum()), 100 * fill_ratio, DEC)

    payload = dict(tag=TAG, decile=DEC, cost=H.COST_ROUND_TRIP, horizons={})
    for hname, (hlab, hok) in lab.items():
        n_out = H.N_OF[hname]
        m = uni & hok
        tr_m = (d0 <= H.TR_END).to_numpy() & m
        te_m = ((d0 >= H.TE_START) & (d0 <= H.TE_END)).to_numpy() & m
        splitdf = pd.DataFrame({"date": d0.to_numpy()})[tr_m]
        segs = H.DualTrackTrainer.split_window(splitdf, H.WINDOW)
        idx_tr, idx_va, idx_te = segs["train"].index, segs["es"].index, np.where(te_m)[0]
        y = hlab.astype(int)
        y_te = np.asarray(y)[idx_te]
        te_ret = H.fwd_ret(M, n_out)[idx_te]

        log.info("=" * 96)
        log.info("[%s] TE n=%d 基础率=%.1f%% | 全池 期望收益=%+.2f%%", hname, len(idx_te),
                 100 * y_te.mean(), 100 * np.nanmean(te_ret))
        univ = decomp(te_ret, y_te, np.ones(len(y_te), bool))
        log.info("[%s][全池] A率=%.1f%% E[ret|A]=%+.2f%% (n=%d) | E[ret|B]=%+.2f%% (n=%d)",
                 hname, 100 * univ["a_rate"], 100 * univ["a_ret"], univ["a_n"],
                 100 * univ["b_ret"], univ["b_n"])

        rows = {}
        for arm, X in (("D0-only", X_d0), ("D0+D1", X_all)):
            _, pred_te, _, _ = H.train_arm(X, y, idx_tr, idx_va, idx_te)
            q = pd.Series(pred_te).rank(pct=True).to_numpy()
            dec = q > DEC
            d = decomp(te_ret, y_te, dec)
            # 盈亏平衡所需板率: 保持 head 的 A/B 收益不变, 需要多高的 P(A|head) 才追平全池
            brk = ((univ["ret"] - d["b_ret"]) / (d["a_ret"] - d["b_ret"])
                   if np.isfinite(d["a_ret"]) and d["a_ret"] != d["b_ret"] else float("nan"))
            log.info("[%s][%s 头部] n=%d 板率=%.1f%% (全池%.1f%%, lift %.2fx) E[ret]=%+.2f%% "
                     "| E[ret|A,head]=%+.2f%% (n=%d) | E[ret|B,head]=%+.2f%% (n=%d)",
                     hname, arm, d["n"], 100 * d["a_rate"], 100 * univ["a_rate"],
                     d["a_rate"] / univ["a_rate"] if univ["a_rate"] else float("nan"),
                     100 * d["ret"], 100 * d["a_ret"], d["a_n"], 100 * d["b_ret"], d["b_n"])
            log.info("[%s][%s 拆解] 亏损来源: 板率贡献 %+.2fpp / A档收益差 %+.2fpp / B档收益差 %+.2fpp "
                     "| 追平全池所需板率=%.1f%% (现%.1f%%)",
                     hname, arm,
                     100 * (d["a_rate"] - univ["a_rate"]) * (univ["a_ret"] - univ["b_ret"]),
                     100 * d["a_rate"] * (d["a_ret"] - univ["a_ret"]),
                     100 * (1 - d["a_rate"]) * (d["b_ret"] - univ["b_ret"]),
                     100 * brk, 100 * d["a_rate"])
            # 尾巴归属: head 的 B 腿 (=没再封板) 是不是「高波动/高位」选出来的?
            # 若 head-B 的 D+1 振幅/涨幅显著高于全池-B, 则这条尾巴就是波动马甲, 不是选股失误。
            desc = {}
            # 三列在 build_features 的派生框里, M 上只有原料 ⇒ 就地按定义算 (口径同 0923P)
            tail_cols = {
                "amp1": ((M["high_1"] - M["low_1"]) / M["pre_1"]).to_numpy(),
                "pct1": pd.to_numeric(M["pct_1"], errors="coerce").to_numpy(),
                "wr1c": pd.to_numeric(M["wr_p1"], errors="coerce").to_numpy(),
            }
            for nm, full in tail_cols.items():
                v = full[idx_te]
                hb, ub = v[dec & (y_te == 0)], v[y_te == 0]
                desc[nm] = dict(head_b=float(np.nanmean(hb)), univ_b=float(np.nanmean(ub)),
                                head=float(np.nanmean(v[dec])), univ=float(np.nanmean(v)))
            log.info("[%s][%s 尾巴归属] head-B vs 全池-B: D+1振幅 %.4f vs %.4f (+%.1f%%) | "
                     "D+1涨幅 %+.2f%% vs %+.2f%% | D+1获利盘 %.4f vs %.4f",
                     hname, arm, desc["amp1"]["head_b"], desc["amp1"]["univ_b"],
                     100 * (desc["amp1"]["head_b"] / desc["amp1"]["univ_b"] - 1),
                     desc["pct1"]["head_b"], desc["pct1"]["univ_b"],
                     desc["wr1c"]["head_b"], desc["wr1c"]["univ_b"])
            rows[arm] = dict(head=d, break_even_a_rate=float(brk), tail_desc=desc)
            gc.collect()
        payload["horizons"][hname] = dict(universe=univ, arms=rows, te_n=int(len(idx_te)))

    payload["rows"] = dict(events=int(len(M)), uni=int(uni.sum()))
    out = data_others_path("diag")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", p)
    log.info("[done] %s", time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
