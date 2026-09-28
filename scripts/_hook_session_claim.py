# -*- coding: utf-8 -*-
"""Claude Code hook 入口 — 跨会话 OPEN TASK 协调 (见 scripts/_session_claims.py).

一个入口按 stdin 的 hook_event_name 分派, 在 .claude/settings.json 里对四个事件注册:
  SessionStart     注册会话 + 顺手清过期会话文件
  UserPromptSubmit 刷心跳 + 按提示词自动认领 + 输出"别人占用了什么"横幅
  PreToolUse       刷心跳 + 两档拦截 (只拦起进程跑该 item / 写该 item 产物)
  Stop             只刷心跳, 绝不释放 — Stop 是"一轮答完", 不是"任务结束"

失败一律放行 (exit 0): 这是协调辅助而非安全边界, hook 崩了不该让所有会话瘫痪.
异常写 .claude/coordination/hook_errors.log.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(
    os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parent.parent
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import _session_claims as sc  # noqa: E402


def _log_error(exc: BaseException) -> None:
    try:
        sc.COORD.mkdir(parents=True, exist_ok=True)
        with open(sc.COORD / "hook_errors.log", "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {type(exc).__name__}: {exc}\n")
    except Exception:
        pass


def _emit(payload: dict) -> None:
    """直接写 UTF-8 字节: Windows 下 sys.stdout 的编码是 locale (cp936),
    走文本层会把中文横幅编成 GBK 传给 CLI 解析坏掉."""
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()


def _on_prompt(payload: dict, session_id: str) -> None:
    """本会话的提示词命中某个 OPEN TASK 时自动认领; 已被别人认领则报警不抢."""
    prompt = payload.get("prompt") or ""
    item_id = sc.item_for_text(prompt)
    parts: list[str] = []
    if item_id:
        try:
            sc.claim(item_id, session_id, note=prompt.strip()[:80])
            parts.append(f"[跨会话协调] 本会话已认领 OPEN TASK `{item_id}`。")
        except sc.ClaimHeldError as exc:
            holder = exc.holder
            note = holder.get("note") or ""
            parts.append(
                f"[跨会话协调] ⚠ 本提示命中的 OPEN TASK `{item_id}` "
                f"已被另一个存活会话 {holder.get('session_id')} 认领"
                f"{' (备注: ' + note + ')' if note else ''}。"
                f"请勿重复开工 — 同一件事两个会话各跑一遍会重复烧机时并可能互相覆盖产物。"
                f"先向用户确认; 确需接手再 "
                f"`python scripts/_session_claims.py steal {item_id}`。"
            )
    b = sc.banner(session_id)
    if b:
        parts.append(b)
    if parts:
        _emit(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(parts),
                }
            }
        )


def _on_tool(payload: dict, session_id: str) -> None:
    reason = sc.deny_reason(
        payload.get("tool_name") or "", payload.get("tool_input") or {}, session_id
    )
    if reason:
        _emit(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )


def main() -> int:
    raw = sys.stdin.buffer.read().decode("utf-8", "replace")
    payload = json.loads(raw) if raw.strip() else {}
    event = payload.get("hook_event_name") or ""
    session_id = payload.get("session_id") or ""
    if not session_id:
        return 0

    if event == "SessionStart":
        sc.touch(
            session_id,
            cwd=payload.get("cwd") or "",
            transcript_path=payload.get("transcript_path") or "",
        )
        sc.gc_sessions()
    elif event == "UserPromptSubmit":
        sc.touch(
            session_id,
            cwd=payload.get("cwd") or "",
            transcript_path=payload.get("transcript_path") or "",
        )
        _on_prompt(payload, session_id)
    elif event == "PreToolUse":
        sc.touch(
            session_id,
            cwd=payload.get("cwd") or "",
            transcript_path=payload.get("transcript_path") or "",
        )
        _on_tool(payload, session_id)
    elif event == "Stop":
        sc.touch(
            session_id,
            cwd=payload.get("cwd") or "",
            transcript_path=payload.get("transcript_path") or "",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # 协调辅助: 崩了放行, 不拦任何会话
        _log_error(exc)
        raise SystemExit(0) from None
