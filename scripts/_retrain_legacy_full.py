"""全量周频重训 (V3 面板直读, 含当日) — 2026-08-05 用户确认跑全量.

与 run_weekly_retrain.py 的区别: 训练面板直接读 PANEL_V3_PATH (已含当日, 免 akshare
网络装配), 对齐 3 年周频窗口. OOS IC 过闸才把 bundle 发布为 current (镜像 weekly 语义),
current_meta.json 同步更新.

用法: python scripts/_retrain_legacy_full.py [tag] [--board main|dual]
      --board 只重训指定板块 (默认双板), 单板省内存省时.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from app.pipeline1.train_runner import run_training
from config.settings import (
    LEGACY_TOP10_SECOND_VOTE,
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

MODEL_DIR = "models/pipeline1"
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FINALTOP_TOOL = os.path.join(ROOT, "tmp_t", "_dual_pkg_finaltop_compare.py")


def _recent_draw_bundles(board: str, exclude: set, cfg: dict) -> list:
    """近 N 日同板 WORM 构建包 (构建间稳定性检查的额外抽签臂), 新→旧.

    只认文件名日期在 stability_window_days 内的包 (更老的包是不同数据/代码
    环境的产物, 不是同过程抽签); current/backup 无日期段自然被 regex 排除.
    """
    from datetime import datetime

    max_extra = int(cfg.get("stability_max_extra", 2))
    window = int(cfg.get("stability_window_days", 3))
    t0 = datetime.strptime(time.strftime("%Y%m%d"), "%Y%m%d")
    out: list = []
    for p in sorted(
        glob.glob(os.path.join(MODEL_DIR, f"{board}_*.pkl")),
        key=os.path.getmtime,
        reverse=True,
    ):
        if os.path.abspath(p) in exclude:
            continue
        m = re.search(r"_((20\d{6})\w*)\.pkl$", os.path.basename(p))
        if not m:
            continue
        try:
            age = (t0 - datetime.strptime(m.group(2), "%Y%m%d")).days
        except ValueError:
            continue
        if not (0 <= age <= window):
            continue
        out.append(p)
        if len(out) >= max_extra:
            break
    return out


def _finaltop_gate(board: str, new_path: str, cfg: dict) -> bool:
    """[09-02] final_list_tool 口径第二票: IC 过闸后调终榜回放工具对拍
    current(A) vs 新包(B), 套新判词 (全窗≥0 且双半≥tol_half 且胜率≥min) 才放行.

    防坏签稳定性 (stability_enable): 近 N 日同板构建包作额外臂 (C, D...) 一起
    对拍, 晋升 = 新包自身 PASS 且 可判臂严格多数 PASS — 单抽幸运签不再晋升.
    fail-safe: 工具失败/超时/无产出/无判词 → 保留旧包 (False). 无 current 包
    → IC 闸独裁放行. 判据实现见 app/pipeline1/finaltop_verdict.py.
    """
    from app.pipeline1.finaltop_verdict import verdict_from_payload

    cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
    if not os.path.exists(cur):
        print(f"[{board}] finaltop: 无 current 包可比, IC 闸独裁 -> 放行", flush=True)
        return True
    arms = [cur, new_path]
    extras: list = []
    if cfg.get("stability_enable"):
        extras = _recent_draw_bundles(
            board,
            exclude={os.path.abspath(cur), os.path.abspath(new_path)},
            cfg=cfg,
        )
        arms += extras
    diag_dir = data_others_path("diag")
    before = set(glob.glob(str(diag_dir / "_dual_pkg_finaltop_compare_*.json")))
    flag = "--main-bundles" if board == "main" else "--dual-bundles"
    cmd = [
        sys.executable,
        FINALTOP_TOOL,
        "--boards",
        board,
        "--bundles",
        ",".join(chr(ord("a") + i) for i in range(len(arms))),
        flag,
        ",".join(arms),
        "--eval-days",
        str(int(cfg.get("eval_days", 48))),
        "--guard-exclude-pid",
        str(os.getpid()),
    ]
    print(
        f"[{board}] finaltop: 终榜回放对拍 current vs "
        f"{os.path.basename(new_path)} (eval_days={cfg.get('eval_days', 48)}, "
        f"臂={len(arms)} 含稳定性臂 {len(extras)}) ...",
        flush=True,
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5400,
        )
    except subprocess.TimeoutExpired:
        print(f"[{board}] finaltop: 工具超时 (>5400s), 保留旧包", flush=True)
        return False
    new_json = sorted(
        set(glob.glob(str(diag_dir / "_dual_pkg_finaltop_compare_*.json"))) - before
    )
    if proc.returncode != 0 or not new_json:
        tail = (proc.stdout or "").strip().splitlines()[-5:]
        print(
            f"[{board}] finaltop: 工具失败 rc={proc.returncode}, 保留旧包; tail={tail}",
            flush=True,
        )
        return False
    payload = json.loads(Path(new_json[-1]).read_text(encoding="utf-8"))
    kw = dict(
        tol_half=float(cfg.get("tol_half", -0.005)),
        win_rate_min=float(cfg.get("win_rate_min", 0.5)),
        min_days=int(cfg.get("min_days", 10)),
    )
    v = verdict_from_payload(payload, board, **kw)
    if not v.get("ok"):
        print(f"[{board}] finaltop: 无判词 ({v.get('reason')}), 保留旧包", flush=True)
        return False
    print(
        f"[{board}] finaltop 判词: Δ={v['d3_full']:+.5f}/日 双半 "
        f"{v['d3_h1']:+.5f}/{v['d3_h2']:+.5f} 胜率 {v['win_rate']:.3f} "
        f"({v['win_days']}/{v['win_days'] + v['lose_days']}) checks={v['checks']}",
        flush=True,
    )
    usable = []
    for i, p in enumerate(extras):
        xv = verdict_from_payload(payload, board, arm=chr(ord("C") + i), **kw)
        tag = os.path.basename(p)
        if xv.get("ok"):
            usable.append(xv)
            print(
                f"[{board}] finaltop 稳定性臂 {tag}: Δ={xv['d3_full']:+.5f}/日 "
                f"胜率 {xv['win_rate']:.3f} pass={xv['pass']}",
                flush=True,
            )
        else:
            print(
                f"[{board}] finaltop 稳定性臂 {tag}: 无判词 ({xv.get('reason')}), "
                f"不计入多数",
                flush=True,
            )
    n = 1 + len(usable)
    passing = (1 if v["pass"] else 0) + sum(1 for x in usable if x["pass"])
    if len(usable) and passing * 2 <= n:
        print(
            f"[{board}] finaltop: 稳定性 FAIL ({passing}/{n} 抽签过判词), 保留旧模型",
            flush=True,
        )
        return False
    if not v["pass"]:
        print(f"[{board}] finaltop: FAIL, 保留旧模型", flush=True)
    return v["pass"]


def _record_canary_state(board: str, tag: str, prev_tag: str, cfg: dict) -> None:
    """晋升成功 → 记 canary state (scripts/_finaltop_canary.py 晋升后真 OOS 复核).

    backup 指向被替换 tag 的 WORM 原包 (f"{board}_{prev_tag}.pkl", 不复制零额外
    磁盘); 找不到才退回 _retrain_backup. 无可回退目标 → 不记 state.
    """
    tag_file = os.path.abspath(os.path.join(MODEL_DIR, f"{board}_{prev_tag}.pkl"))
    bak = os.path.abspath(
        os.path.join(MODEL_DIR, f"{board}_current_retrain_backup.pkl")
    )
    target = (
        tag_file
        if (prev_tag and os.path.exists(tag_file))
        else (bak if os.path.exists(bak) else None)
    )
    if target is None:
        return
    p = data_others_path("diag") / "finaltop_canary_state.json"
    state: dict = {}
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    state[board] = {
        "tag": tag,
        "prev_tag": prev_tag,
        "backup": target,
        "promoted": time.strftime("%Y-%m-%d"),
        "days": 10,
        "ran": {},
        "status": "active",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{board}] canary state 已记 (backup={os.path.basename(target)}, "
        f"prev={prev_tag})",
        flush=True,
    )


def main() -> int:
    # 并发守卫 (2026-09-01): 已有重训/预测进程在跑 → 退出, 防页交换卡死/OOM
    # (08-17/08-24 事故). 哨兵不含 run_daily_automation.py: 链作为父进程存活是
    # 合法场景, 否则链调起本脚本时会被自己的父进程误杀.
    from scripts._run_guard import find_conflicts

    others = find_conflicts()
    if others:
        for c in others:
            print(
                f"[guard] 冲突进程: {c['sentinel']} (PID {c['pid']}) {c['cmdline']}",
                flush=True,
            )
        print(
            f"[guard] 已有 {len(others)} 个重训/预测进程在跑, 本实例退出 (rc=3). "
            f"等其结束后再启动.",
            flush=True,
        )
        return 3

    # 子模块全部用 logging.getLogger(__name__) 传播到 root, 无 handler 时 info 被丢弃
    # (Python last-resort handler 只放 WARNING+), 重训会"看似卡住". 这里挂一个 handler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description="legacy 周频重训 (V3 面板直读, 含当日)")
    ap.add_argument(
        "tag", nargs="?", default=None, help="模型包标签 (默认今天 YYYYMMDD)"
    )
    ap.add_argument(
        "--board",
        choices=("main", "dual"),
        default=None,
        help="只重训指定板块 (默认双板)",
    )
    args = ap.parse_args()
    tag = args.tag or time.strftime("%Y%m%d")
    boards = (args.board,) if args.board else ("main", "dual")
    # LEGACY_FORCE_FALLBACK=1 → 只对 main 跳过 FeatureSelector (bruteforce_dedup 选择
    # 过大必 OOM, 直用 FeatureEngine 全量 316 特征), dual 仍走 gate_d (38). 两段式发布:
    # 先落 cls 修复, 再单独验证 cap 选择过 OOS 门.
    fallback_boards = (
        {"main"} if os.environ.get("LEGACY_FORCE_FALLBACK", "0") == "1" else None
    )
    t0 = time.time()
    # 内存独占闸 (2026-08-15 用户定案): 启动时可用内存不足 → 拒绝启动; 运行期
    # 每 30s 采样, 被其他重活挤兑 → WARNING (不杀进程, 训练有 per-model 检查点).
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)
    # 面板由 run_training 内部直读并持有 (panel_path 模式): 若本脚本持有 panel 引用,
    # run_training 内 `del panel` 失效 → 特征 build 阶段 (dim17) 峰值贴 commit 上限
    # 偶发 OOM (2026-08-13 r2/r4 同一崩溃点). 直读预过滤 (amount>=5000万 且 非停牌)
    # 同口径少读 ~20% 行, 输出与整表读取后 run_train 完全一致 (2026-08-10).
    results = run_training(
        panel=None,
        panel_path=PANEL_V3_PATH,
        tag=tag,
        model_dir=MODEL_DIR,
        use_ic_screen=True,
        fallback_boards=fallback_boards,
        boards=boards,
    )

    from app.pipeline1.model_meta import load_modules, save_modules

    # LEGACY_FORCE_FALLBACK=1 (20260811c 安全发布): 双创 gate_d 本轮抽到 208 特征
    # 过拟合抽签 (r3=38, 本轮=208, 输入完全一致 → LGBM n_jobs 线程非确定性), 不让
    # 该抽签发布, 保留 proven 的 20260811b (38 特征). 主板 cls 修复照常发布.
    skip_dual_switch = os.environ.get("LEGACY_FORCE_FALLBACK", "0") == "1"
    mods = load_modules()
    for board, res in results.items():
        if skip_dual_switch and board == "dual" and res["switched"]:
            print(
                "[dual] gate_d 非确定性漂移 (38→208 特征), 保留 20260811b, 跳过切换",
                flush=True,
            )
            continue
        if res["switched"]:
            cfg = LEGACY_TOP10_SECOND_VOTE
            if cfg.get("enable") and cfg.get("caliber") == "final_list_tool":
                if not _finaltop_gate(board, res["path"], cfg):
                    continue
            cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
            prev_tag = (mods.get(board) or {}).get("tag", "")
            bak = os.path.join(MODEL_DIR, f"{board}_current_retrain_backup.pkl")
            if os.path.exists(cur) and not os.path.exists(bak):
                shutil.copy(cur, bak)
                print(f"[{board}] 旧 current 备份 -> {bak}", flush=True)
            shutil.copy(res["path"], cur)
            mods[board] = {
                "tag": tag,
                "file": os.path.basename(res["path"]),
                "updated": time.strftime("%Y-%m-%d %H:%M"),
            }
            print(f"[{board}] switched -> current = {res['path']}", flush=True)
            _record_canary_state(board, tag, prev_tag, LEGACY_TOP10_SECOND_VOTE)
        else:
            print(
                f"[{board}] OOS weighted_IC={res['oos'].get('weighted_ic'):.4f} "
                f"< {res['oos'].get('threshold', '?')}, 保留旧模型",
                flush=True,
            )
    save_modules(mods)
    print(f"[meta] current_meta.json = {mods}", flush=True)
    print(f"[done] 全部完成 ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
