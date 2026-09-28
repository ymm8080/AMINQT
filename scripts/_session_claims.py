# -*- coding: utf-8 -*-
"""跨会话 OPEN TASK 认领登记 — 防同机多个 Claude Code 会话重复认领同一件事.

`scripts/_run_guard.py` 只挡"重活已在跑"(进程级 cmdline 哨兵), 挡不住"还没起进程
时的意图撞车": 两个会话各自只读 memory 索引, 于是同时认领同一个 OPEN TASK 各跑一遍.

数据全在 .claude/coordination/ (机器本地, 不入库):
  items.json                  item_id -> {title, aliases[], source} — 人工维护, 量小
  sessions/<session_id>.json  心跳 {cwd, transcript_path, started_at, last_seen}
  claims/<item_id>.json       认领 {item_id, session_id, claimed_at, note}

存活判据 = max(心跳 last_seen, transcript_path 的 mtime) 距今 <= LIVE_TTL_S.
取 transcript mtime 是因为它由 Claude Code 每次写消息时自己刷新, 与我们的 hook 是否
触发无关. 不用 pid (无 session-id 环境变量, hook 的 ppid 是转瞬即逝的 bash); 也不用
SessionEnd (崩溃时不触发, 会留永久死锁) — 所以 TTL 是主要的回收路径, 不是兜底.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

COORD = (
    Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parent.parent)
    / ".claude"
    / "coordination"
)
SESSIONS = COORD / "sessions"
CLAIMS = COORD / "claims"
ITEMS = COORD / "items.json"

# 认领保鲜期. 会话空闲超过它即视为可回收 (用户思考/离开时 hook 不触发, 属预期).
LIVE_TTL_S = 45 * 60
# 纯记录用的会话文件保留期, 由 SessionStart 顺手清理.
SESSION_GC_S = 24 * 3600

# 只有"起进程"的命令才可能撞车. grep/read 类命令文案里恰含 item 关键词是常态
# (用户天天 grep 脚本名), 一律放行 — 这正是 A/B 里"读不拦"的那一档.
RUNNERS = frozenset(
    {
        "python",
        "python.exe",
        "python3",
        "python3.exe",
        "py",
        "py.exe",
        "pytest",
        "pytest.exe",
        "uv",
        "poetry",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "cmd",
        "cmd.exe",
        "start-process",
        "invoke-expression",
    }
)
_SEG_SPLIT = re.compile(r"&&|\|\||[;|]")


class ClaimHeldError(RuntimeError):
    """目标 item 已被另一个存活会话认领."""

    def __init__(self, item_id: str, holder: dict):
        self.item_id = item_id
        self.holder = holder
        sid = holder.get("session_id", "?")
        super().__init__(f"item '{item_id}' 已被会话 {sid} 认领")


def _now() -> float:
    return time.time()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mtime(path_str: str | None) -> float:
    if not path_str:
        return 0.0
    try:
        return Path(path_str).stat().st_mtime
    except OSError:
        return 0.0


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _session_path(session_id: str) -> Path:
    return SESSIONS / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)}.json"


def is_alive(info: dict | None) -> bool:
    """心跳与 transcript mtime 取新者 — transcript 由 CLI 自己刷, 不受 hook 触发与否影响."""
    if not info:
        return False
    seen = max(float(info.get("last_seen") or 0), _mtime(info.get("transcript_path")))
    return (_now() - seen) <= LIVE_TTL_S


def touch(
    session_id: str,
    *,
    cwd: str = "",
    transcript_path: str = "",
) -> dict:
    """写/刷本会话心跳. 任何 hook 事件都调它."""
    path = _session_path(session_id)
    info = _read_json(path) or {}
    info.update(
        {
            "session_id": session_id,
            "last_seen": _now(),
            "started_at": info.get("started_at") or _now(),
        }
    )
    if cwd:
        info["cwd"] = cwd
    if transcript_path:
        info["transcript_path"] = transcript_path
    _write_json(path, info)
    return info


def gc_sessions() -> int:
    """清掉过期会话文件. 认领由 is_alive 判废, 不依赖文件是否还在."""
    if not SESSIONS.is_dir():
        return 0
    n = 0
    for p in SESSIONS.glob("*.json"):
        info = _read_json(p)
        if not is_alive(info) and (_now() - _mtime(str(p))) > SESSION_GC_S:
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


def live_sessions(exclude: str | None = None) -> dict[str, dict]:
    if not SESSIONS.is_dir():
        return {}
    out: dict[str, dict] = {}
    for p in SESSIONS.glob("*.json"):
        info = _read_json(p)
        sid = (info or {}).get("session_id")
        if sid and sid != exclude and is_alive(info):
            out[sid] = info
    return out


def load_items() -> dict[str, dict]:
    return _read_json(ITEMS) or {}


def item_for_text(text: str) -> str | None:
    """文本里命中的第一个 item_id. 用于 UserPromptSubmit 自动认领与命令匹配."""
    low = (text or "").lower()
    for item_id, spec in load_items().items():
        for alias in spec.get("aliases") or ():
            if alias.lower() in low:
                return item_id
    return None


def _claim_path(item_id: str) -> Path:
    return CLAIMS / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', item_id)}.json"


def read_claim(item_id: str) -> dict | None:
    return _read_json(_claim_path(item_id))


def claim(
    item_id: str, session_id: str, *, note: str = "", steal: bool = False
) -> dict:
    """认领 item. 已被别人持有且其会话存活 -> ClaimHeldError.

    首发用 O_CREAT|O_EXCL 拿原子性 (两个人同时抢只会有一个人成功); 续期与抢占走
    原子替换. 抢占是显式动作, 调用方负责说明理由 — 死会话的认领靠 is_alive 判废,
    而不是靠文件是否还在.
    """
    path = _claim_path(item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "item_id": item_id,
        "session_id": session_id,
        "claimed_at": _now(),
        "note": note,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        prev = _read_json(path) or {}
        holder = prev.get("session_id")
        if (
            holder == session_id
            or steal
            or not is_alive(_read_json(_session_path(holder or "")))
        ):
            payload["claimed_at"] = (
                prev.get("claimed_at") if holder == session_id else _now()
            )
            _write_json(path, payload)
            return payload
        raise ClaimHeldError(item_id, prev) from None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def release(item_id: str, session_id: str) -> bool:
    """只释放自己持有的 — 抢别人的认领必须先显式 steal."""
    path = _claim_path(item_id)
    info = _read_json(path)
    if not info or info.get("session_id") != session_id:
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def held_by_others(session_id: str) -> list[dict]:
    """别的存活会话持有的认领 [{item_id, title, holder, age_s}]."""
    if not CLAIMS.is_dir():
        return []
    items = load_items()
    out: list[dict] = []
    for p in CLAIMS.glob("*.json"):
        info = _read_json(p)
        if not info:
            continue
        holder = info.get("session_id")
        if not holder or holder == session_id:
            continue
        if not is_alive(_read_json(_session_path(holder))):
            continue
        item_id = info.get("item_id") or p.stem
        out.append(
            {
                "item_id": item_id,
                "title": (items.get(item_id) or {}).get("title", ""),
                "holder": info,
                "age_s": _now() - float(info.get("claimed_at") or _now()),
            }
        )
    return sorted(out, key=lambda d: d["age_s"])


def _runners_in(command: str) -> list[str]:
    """命令里每一段的头 token, 判断这是"起进程"还是"只读".

    `cd X && python y.py` 这类复合命令要按段看, 否则会漏判.
    """
    found: list[str] = []
    for seg in _SEG_SPLIT.split(command or ""):
        toks = [t for t in seg.strip().split() if "=" not in t or t.startswith("-")]
        if not toks:
            continue
        head = os.path.basename(toks[0]).lower().strip("\"'")
        if head in RUNNERS:
            found.append(head)
    return found


def deny_reason(tool_name: str, tool_input: dict, session_id: str) -> str | None:
    """两档: 只拦"起进程跑该 item"与"写该 item 的产物"; 读一律放行.

    返回给用户看的拒绝理由, 或 None=放行.
    """
    if tool_name == "Bash":
        command = tool_input.get("command") or ""
        item_id = item_for_text(command)
        if not item_id or not _runners_in(command):
            return None
        target = f"命令 `{command.strip()[:160]}`"
    elif tool_name in ("Write", "Edit", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        item_id = item_for_text(path)
        if not item_id:
            return None
        target = f"写入 `{path}`"
    else:
        return None

    for held in held_by_others(session_id):
        if held["item_id"] != item_id:
            continue
        holder = held["holder"]
        sid = holder.get("session_id", "?")
        age = int(held["age_s"] // 60)
        note = holder.get("note") or ""
        return (
            f"{target} 命中 OPEN TASK `{item_id}`"
            f"{' (' + held['title'] + ')' if held['title'] else ''}, "
            f"但该 item 已被另一个存活会话 {sid} 认领 ({age} 分钟前"
            f"{', 备注: ' + note if note else ''}).\n"
            f"两个会话跑同一件事会重复烧机时并可能互相覆盖产物. "
            f"若是只读/复核请改用 Read/Grep (不受此拦); "
            f"确认对方已放弃则显式抢占: "
            f"python scripts/_session_claims.py steal {item_id}"
        )
    return None


def banner(session_id: str, include_processes: bool = True) -> str | None:
    """给模型的每轮横幅: 别人持有的 item + 正在跑的重活. 无内容返回 None."""
    lines: list[str] = []
    held = held_by_others(session_id)
    if held:
        for h in held:
            age = int(h["age_s"] // 60)
            lines.append(
                f"  CLAIMED: {h['item_id']}"
                f"{' — ' + h['title'] if h['title'] else ''}"
                f"  <- {h['holder'].get('session_id', '?')} ({age}m ago)"
            )
    if include_processes:
        try:
            from scripts._run_guard import find_conflicts

            for c in find_conflicts():
                lines.append(f"  RUNNING: {c['sentinel']} (PID {c['pid']})")
        except Exception:
            pass
    if not lines:
        return None
    return (
        "[跨会话协调] 其他 Claude Code 会话当前占用:\n"
        + "\n".join(lines)
        + "\n认领中的 OPEN TASK 不要重复开工 — 只读复核可直接做;"
        " 确需接手先 `python scripts/_session_claims.py steal <item_id>`."
    )


def _fmt_status() -> str:
    items = load_items()
    alive = live_sessions()
    out = [f"live sessions ({len(alive)}):"]
    for sid, info in alive.items():
        age = int((_now() - float(info.get("last_seen") or 0)) // 60)
        out.append(f"  {sid}  {info.get('cwd', '')}  (seen {age}m ago)")
    out.append("claims:")
    for item_id in items:
        info = read_claim(item_id)
        if not info:
            out.append(f"  {item_id}: free")
            continue
        holder = info.get("session_id")
        holder_alive = is_alive(_read_json(_session_path(holder or "")))
        out.append(
            f"  {item_id}: {holder}"
            f"{' (live)' if holder_alive else ' (STALE — stealable)'}"
            f"{' — ' + info['note'] if info.get('note') else ''}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "status":
        print(_fmt_status())
        return 0
    cmd, item_id = args[0], (args[1] if len(args) > 1 else "")
    if cmd not in ("claim", "release", "steal"):
        print(__doc__)
        return 2
    if cmd == "steal":
        sid = args[2] if len(args) > 2 else "manual-steal"
        prev = read_claim(item_id) or {}
        print(f"抢占 {item_id} (原持有者 {prev.get('session_id', 'none')})")
        claim(item_id, sid, note=" ".join(args[3:]) or "explicit steal", steal=True)
        return 0
    print(
        "claim/release 需要真实 session_id — 认领由 UserPromptSubmit hook 自动完成; "
        "手动接手用 `steal`."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
