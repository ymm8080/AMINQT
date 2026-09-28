# -*- coding: utf-8 -*-
"""Tests for scripts/_session_claims — 跨会话 OPEN TASK 认领 (2026-09-28).

存活性判定用"心跳 / transcript mtime"两信号, 不依赖 pid 与 SessionEnd (崩溃不触发).
所有用例走 tmp_path, 不碰真实 .claude/coordination/.
"""

import json
import os
import time

import pytest

import scripts._session_claims as sc

ITEM = "firstboard-retrain"


@pytest.fixture
def coord(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "COORD", tmp_path)
    monkeypatch.setattr(sc, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(sc, "CLAIMS", tmp_path / "claims")
    monkeypatch.setattr(sc, "ITEMS", tmp_path / "items.json")
    (tmp_path / "sessions").mkdir()
    (tmp_path / "claims").mkdir()
    (tmp_path / "items.json").write_text(
        json.dumps(
            {
                ITEM: {
                    "title": "首板重训",
                    "aliases": ["_firstboard_retrain", "首板重训"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


def _age_session(session_id, seconds):
    """把某会话的心跳推回过去, 模拟空闲/崩溃."""
    path = sc._session_path(session_id)
    info = json.loads(path.read_text(encoding="utf-8"))
    info["last_seen"] = time.time() - seconds
    info.pop("transcript_path", None)
    path.write_text(json.dumps(info), encoding="utf-8")


# ── 存活判定 ────────────────────────────────────────────────────────────────


def test_fresh_heartbeat_is_alive(coord):
    sc.touch("s1")
    assert sc.is_alive(sc._read_json(sc._session_path("s1"))) is True


def test_idle_beyond_ttl_is_not_alive(coord):
    sc.touch("s1")
    _age_session("s1", sc.LIVE_TTL_S + 60)
    assert sc.is_alive(sc._read_json(sc._session_path("s1"))) is False


def test_transcript_mtime_keeps_idle_session_alive(coord, tmp_path):
    """心跳旧了但 transcript 刚被 CLI 写过 -> 仍算存活 (hook 未触发不代表会话死了)."""
    tr = tmp_path / "transcript.jsonl"
    tr.write_text("{}", encoding="utf-8")
    sc.touch("s1", transcript_path=str(tr))
    _age_session("s1", sc.LIVE_TTL_S + 60)
    info = json.loads(sc._session_path("s1").read_text(encoding="utf-8"))
    info["transcript_path"] = str(tr)
    sc._session_path("s1").write_text(json.dumps(info), encoding="utf-8")
    assert sc.is_alive(sc._read_json(sc._session_path("s1"))) is True


def test_missing_session_file_is_not_alive(coord):
    assert sc.is_alive(None) is False


# ── 认领 / 释放 / 抢占 ──────────────────────────────────────────────────────


def test_second_live_session_cannot_claim(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    sc.touch("s2")
    with pytest.raises(sc.ClaimHeldError) as ei:
        sc.claim(ITEM, "s2")
    assert ei.value.holder["session_id"] == "s1"


def test_same_session_renews(coord):
    sc.touch("s1")
    first = sc.claim(ITEM, "s1")
    again = sc.claim(ITEM, "s1", note="still me")
    assert again["claimed_at"] == first["claimed_at"]
    assert again["note"] == "still me"


def test_stale_holder_claim_is_takeable(coord):
    """持有者会话已过期 -> 直接认领, 不需要 steal (TTL 是主要回收路径)."""
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    _age_session("s1", sc.LIVE_TTL_S + 60)
    sc.touch("s2")
    assert sc.claim(ITEM, "s2")["session_id"] == "s2"


def test_steal_overrides_live_holder(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    sc.touch("s2")
    assert sc.claim(ITEM, "s2", steal=True)["session_id"] == "s2"
    assert sc.read_claim(ITEM)["session_id"] == "s2"


def test_release_only_by_holder(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    assert sc.release(ITEM, "s2") is False
    assert sc.read_claim(ITEM) is not None
    assert sc.release(ITEM, "s1") is True
    assert sc.read_claim(ITEM) is None


def test_held_by_others_ignores_own_and_stale(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    assert sc.held_by_others("s1") == []
    assert [h["item_id"] for h in sc.held_by_others("s2")] == [ITEM]
    _age_session("s1", sc.LIVE_TTL_S + 60)
    assert sc.held_by_others("s2") == []


# ── item 匹配 ───────────────────────────────────────────────────────────────


def test_item_for_text_matches_alias_case_insensitively(coord):
    assert sc.item_for_text("python scripts/_Firstboard_Retrain.py") == ITEM
    assert sc.item_for_text("跑一下首板重训") == ITEM
    assert sc.item_for_text("别的活") is None


# ── 两档拦截: 只拦"起进程跑该 item"与"写该 item 产物" ────────────────────────


def test_deny_runner_command_on_claimed_item(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    reason = sc.deny_reason(
        "Bash", {"command": "python scripts/_firstboard_retrain.py"}, "s2"
    )
    assert reason and "s1" in reason and ITEM in reason


def test_deny_runner_command_inside_compound(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    reason = sc.deny_reason(
        "Bash", {"command": "cd tmp_t && python _firstboard_retrain.py"}, "s2"
    )
    assert reason is not None


def test_readonly_bash_is_allowed_even_when_matching(coord):
    """用户天天 grep 脚本名 — 只读命令文案命中 item 不得被拦."""
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    for cmd in (
        "grep -rn _firstboard_retrain scripts/",
        "git log --oneline -- _firstboard_retrain.py",
        "ls -la tmp_t | grep firstboard",
    ):
        assert sc.deny_reason("Bash", {"command": cmd}, "s2") is None, cmd


def test_runner_command_on_unrelated_item_is_allowed(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    assert sc.deny_reason("Bash", {"command": "python other_thing.py"}, "s2") is None


def test_write_to_matching_path_is_denied(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    reason = sc.deny_reason("Write", {"file_path": "tmp_t/_firstboard_retrain.py"}, "s2")
    assert reason is not None


def test_read_tools_are_never_denied(coord):
    """复核别人的结果是常态 — Read/Grep 不在 matcher 里, 这里再钉一次."""
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    assert sc.deny_reason("Read", {"file_path": "tmp_t/_firstboard_retrain.py"}, "s2") is None


def test_holder_is_not_denied_on_own_item(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1")
    assert (
        sc.deny_reason("Bash", {"command": "python scripts/_firstboard_retrain.py"}, "s1")
        is None
    )


def test_no_claim_means_no_deny(coord):
    assert sc.deny_reason("Bash", {"command": "python scripts/_firstboard_retrain.py"}, "s2") is None


# ── 横幅 ────────────────────────────────────────────────────────────────────


def test_banner_lists_other_claim(coord):
    sc.touch("s1")
    sc.claim(ITEM, "s1", note="扩窗重训")
    b = sc.banner("s2", include_processes=False)
    assert b and ITEM in b and "s1" in b and "首板重训" in b


def test_banner_none_when_nothing_held(coord):
    assert sc.banner("s2", include_processes=False) is None


# ── 会话文件回收 ────────────────────────────────────────────────────────────


def test_gc_removes_old_session_files_but_keeps_live(coord):
    sc.touch("live")
    sc.touch("dead")
    _age_session("dead", sc.SESSION_GC_S + 60)
    os.utime(sc._session_path("dead"), (0, 0))
    assert sc.gc_sessions() == 1
    assert sc._session_path("live").exists()


# ── 输出编码 ────────────────────────────────────────────────────────────────


def test_hook_emit_writes_utf8_bytes(monkeypatch):
    """Windows 下 sys.stdout 是 locale 编码 (cp936), 走文本层会把中文横幅编成 GBK
    传给 CLI 解析坏掉 — 必须直写 UTF-8 字节."""
    import io

    import scripts._hook_session_claim as hook

    buf = io.BytesIO()

    class _FakeStdout:
        buffer = buf

    monkeypatch.setattr("sys.stdout", _FakeStdout())
    hook._emit({"hookSpecificOutput": {"additionalContext": "已认领 OPEN TASK"}})

    raw = buf.getvalue()
    assert "已认领 OPEN TASK" in raw.decode("utf-8")
