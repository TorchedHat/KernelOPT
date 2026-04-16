"""Unit tests for pipeline HUD helpers (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


def test_plain_from_rich_string_strips_markup():
    from kernelopt.pipeline_ui import plain_from_rich_string

    s = "[bold green]Hello[/] [dim]world[/]"
    out = plain_from_rich_string(s)
    assert out.strip() == "Hello world"


def test_pipeline_config_has_hud_tty_default():
    from kernelopt.state import PipelineConfig

    c = PipelineConfig()
    assert c.hud_tty is None


def test_pipeline_state_has_pre_run_snapshot_default():
    from kernelopt.state import PipelineState

    s = PipelineState()
    assert s.pre_run_system_snapshot == {}
