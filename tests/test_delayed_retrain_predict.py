"""延迟重训+预测触发器 (scripts/_delayed_retrain_predict_0916.py): 闸判断真
("管线有新改动 → 挂任务"), 幂等 (已挂不重挂), --run-now 分支直接 spawn."""

import os
import subprocess

import pytest

from scripts import _delayed_retrain_predict_0916 as dly


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    # 最老 mtime 的管线文件树 + 空的 stamp
    for rel in dly.PIPELINE_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    monkeypatch.setattr(dly, "ROOT", tmp_path)
    monkeypatch.setattr(dly, "STAMP_PATH", tmp_path / "logs" / "pipe.stamp")
    return tmp_path


def test_no_stale_no_defer(fake_pipeline):
    dly._mark_stamp()
    assert dly._deferred_needed() is False


def test_pipeline_edited_defers(fake_pipeline):
    dly._mark_stamp()
    # 模拟管线修改: mtime 往后戳
    f = fake_pipeline / dly.PIPELINE_FILES[0]
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 100))
    assert dly._deferred_needed() is True


def test_no_stamp_at_all_defers_once(fake_pipeline):
    assert dly._deferred_needed() is True


def test_schedule_query_short_circuit(fake_pipeline, monkeypatch, capsys):
    monkeypatch.setattr(
        dly,
        "_run_schtasks",
        lambda args: subprocess.CompletedProcess(
            args, 0, stdout=dly.TASK_NAME, stderr=""
        ),
    )
    assert dly._schedule_delayed_run(0.01) is False  # 已挂 → 不重挂
    out = capsys.readouterr().out
    assert "Task" in out or "TASK" in out or "已挂" in out


def test_schedule_creates_task(fake_pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr(
        dly,
        "_run_schtasks",
        lambda args: (
            calls.append(args),
            subprocess.CompletedProcess(args, 0, stdout="ok", stderr=""),
        )[1],
    )
    ok = dly._schedule_delayed_run(0.01)
    assert ok is True
    assert calls and calls[0] == ["schtasks", "/query", "/tn", dly.TASK_NAME]
    # 二次调用是 /create
    assert len(calls) == 2 and "/create" in calls[1]
    assert dly.TASK_NAME in " ".join(calls[1])
