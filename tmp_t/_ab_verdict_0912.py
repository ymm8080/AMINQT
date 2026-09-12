# -*- coding: utf-8 -*-
"""0912 逐特征 A/B 判词汇总: 读两板 WORM JSON → Δ表 + 留/摘建议.

输入: data_others/diag/cls_top10_0912_{main,dual}_*.json (由 tmp_t/_cls_top10_0912.py 产出)
规则 (与 _cls_top10_0912.py 判词一致):
  主判 = cls 头 top10 实得 (prob10_pull 清单排名键): 两板 Δ≥0 → 留; 任一板降 >1pp → 摘
  边界带 (0 > Δ > -1pp) → 留观, 表里标明
参考列 = cls IC / reg top10 / reg IC 一并展示:
  dual reg 头单特征臂大多 Δ=0 (deterministic LightGBM 零分裂, 见 dual-reg-zero-split
  记忆) — Δ=0 不代表有益, cls 头才是区分头; reg 数字仅作 main 侧参考.

用法: python tmp_t/_ab_verdict_0912.py  (轻量: 只读 JSON, 无重活)
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from config.settings import data_others_path

TAG = "ab_verdict_0912"
DIAG = data_others_path("diag")
PP = 100.0  # 小数 → pp


def _latest(board: str) -> dict:
    pats = sorted(glob.glob(str(DIAG / f"cls_top10_0912_{board}_*.json")))
    if not pats:
        raise SystemExit(f"[FAIL] 缺 cls_top10_0912_{board}_*.json (WORM 未产出?)")
    with open(pats[-1], encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) else "  — "


def main() -> int:
    rows = []
    for board in ("main", "dual"):
        rep = _latest(board)
        arms = rep["arms"]
        base = arms["base"]
        for feat in rep["features"]:
            a = arms.get(feat)
            if a is None:
                continue
            rows.append({
                "board": board, "feat": feat,
                "d_cls_top10": PP * (a["cls"]["top10_real_net"] - base["cls"]["top10_real_net"]),
                "d_cls_ic": PP * (a["cls"]["ic_mean"] - base["cls"]["ic_mean"]),
                "d_reg_top10": PP * (a["reg"]["top10_real_net"] - base["reg"]["top10_real_net"]),
                "d_reg_ic": PP * (a["reg"]["ic_mean"] - base["reg"]["ic_mean"]),
            })

    by_feat = {}
    for r in rows:
        by_feat.setdefault(r["feat"], {})[r["board"]] = r

    verdicts = {}
    print("\n=== 逐特征 Δ vs base (cls top10 实得 pp 为主判 | 参考: cls IC / reg top10 / reg IC) ===")
    print(f"{'特征':<14} {'板':<5} {'Δcls_top10':>10} {'Δcls_IC':>8} {'Δreg_top10':>10} {'Δreg_IC':>8}")
    for feat, brd in by_feat.items():
        for board in ("main", "dual"):
            r = brd.get(board)
            if r:
                print(f"{feat:<14} {board:<5} {_fmt(r['d_cls_top10']):>10} "
                      f"{_fmt(r['d_cls_ic']):>8} {_fmt(r['d_reg_top10']):>10} {_fmt(r['d_reg_ic']):>8}")

    print("\n=== 判词 (主判 cls top10: 两板 Δ≥0 留 / 任一板 <-1pp 摘 / 其间留观) ===")
    for feat, brd in by_feat.items():
        ds = {b: r["d_cls_top10"] for b, r in brd.items() if r}
        if not ds:
            verdicts[feat] = "无数据"
            continue
        if all(d >= 0 for d in ds.values()):
            v = "留"
        elif any(d < -1.0 for d in ds.values()):
            v = "摘"
        else:
            v = "留观(边界带)"
        verdicts[feat] = v
        detail = " ".join(f"{b}:{d:+.2f}pp" for b, d in ds.items())
        print(f"  {feat:<14} {v:<10} [{detail}]")

    print("\n注: dual reg 臂 Δ=0 = 零分裂 (非收益证据); 获利盘斜率20/出货密度为干净因果口径"
          " (网格修复 818bd386 落地后跑的, 可直接采信; 勿与昨夜 Phase A 旧口径对比).")

    out = DIAG / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdicts": verdicts}, fh, ensure_ascii=False, indent=2)
    print(f"[WORM] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
