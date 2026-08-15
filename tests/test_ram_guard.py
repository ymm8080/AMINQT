"""重训内存独占闸单测 (2026-08-15).

规则 (用户定案): 重训期间机器须独占, 其他重活抢内存 → 08-14 训练 8.4h
零模型完成. 代码强制两层: 启动闸 (内存不足拒绝启动) + 运行期警报 (daemon
线程持续 WARNING, 永不杀进程).

纯判定函数 should_block 可离线测试; check_startup_gate 通过 monkeypatch
free_physical_bytes 注入假内存值, 不依赖真实机器状态.
"""

from __future__ import annotations

import pytest

from app.pipeline1.ram_guard import check_startup_gate, should_block

_GB = 1024**3


class TestShouldBlock:
    def test_below_threshold_blocks(self):
        assert should_block(1_999, 2_000) is True

    def test_equal_threshold_blocks(self):
        # 低于下限才拒 (>= 放行): 边界语义固定
        assert should_block(2_000, 2_000) is False

    def test_above_threshold_passes(self):
        assert should_block(10 * _GB, 2 * _GB) is False


class TestStartupGate:
    def test_starved_machine_refuses_to_start(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "app.pipeline1.ram_guard.free_physical_bytes", lambda: 1 * _GB
        )
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as e:
                check_startup_gate(2 * _GB)
        assert e.value.code == 2
        assert "重训启动被拒" in caplog.text

    def test_healthy_machine_passes(self, monkeypatch):
        monkeypatch.setattr(
            "app.pipeline1.ram_guard.free_physical_bytes", lambda: 10 * _GB
        )
        check_startup_gate(2 * _GB)  # 不抛即过
