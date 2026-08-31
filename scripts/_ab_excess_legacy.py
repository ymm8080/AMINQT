"""超额标签 A/B (legacy, 2026-08-29): 回归头训练目标 label_pm_{k}d_net → 板内当日超额.

背景: parallel 臂已判结构性无效 (池分=横截面排名, 标签去均值不改排序,
_ab_excess_calib_20260829). legacy 的幅度来自 LGBM 回归头本体 (200 特征学绝对
涨幅), 牛市权重灌入高贝塔特征 — 标签中性化可真改模型.

设计 (单变量):
- 只 demean 回归头目标 label_pm_{3,5,10}d_net (板×日 全池等权, prepare_board_frame
  按板调用 → groupby(date) 即板内); cls/prob 头的 *_cls 标签不动 (排名键
  pred_ret_10d = 回归头输出).
- 特征选择 (IC 筛) 在 demean 后帧上跑 — 属于处理的一部分.
- IC 可比性: 按日去均值不改变当日标签截面排名 → validate_oos 的逐日 rank IC
  与绝对口径完全等价, 两臂数字直接对表.
- 基线 = 昨晚 08-28 批次生产 run 的 oos (同面板同配置, 种子固定可复现).

判定 (预登记): 双板 weighted_IC 均不低于基线, 且无单视界 IC 崩塌 (<基线-0.02)
→ 进 Tier-2 (OOS top10 实测); 否则 REJECT 归档.
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.train_runner import run_training
from config.settings import PANEL_V3_PATH

TAG = "abexcess0829"
HORIZONS = (3, 5, 10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ab_excess_legacy")


def main() -> int:
    t0 = time.time()
    orig_build = LabelEngine.build_labels

    def build_labels_demeaned(df, *a, **kw):
        df = orig_build(df, *a, **kw)
        for k in HORIZONS:
            c = f"label_pm_{k}d_net"
            if c in df.columns:
                df[c] = df[c] - df.groupby("date")[c].transform("mean")
        return df

    LabelEngine.build_labels = staticmethod(build_labels_demeaned)
    log.info("[ab] LabelEngine.build_labels 已包裹: label_pm_{3,5,10}d_net 板内按日去均值")

    results = run_training(panel_path=PANEL_V3_PATH, tag=TAG, boards=("main", "dual"))
    LabelEngine.build_labels = staticmethod(orig_build)

    report = {"ts": datetime.now().isoformat(timespec="seconds"), "tag": TAG,
              "boards": {}}
    for board, r in results.items():
        report["boards"][board] = {
            "path": str(r.get("path")),
            "oos": r.get("oos"),
            "n_features": r.get("n_features"),
        }
        log.info("[ab] %s oos=%s", board, json.dumps(r.get("oos"), default=str))

    out = Path(f"data/others/_ab_excess_legacy_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    log.info("[ab] 结果 WORM 落盘 %s (%.0fs)", out, time.time() - t0)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
