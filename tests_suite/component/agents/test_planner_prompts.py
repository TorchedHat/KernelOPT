# kernelopt/tests/agents/test_planner_prompts.py
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from kernelopt.agents.planner.prompts import (
    SYSTEM_PROMPT,
    build_diversity_hint,
    build_initial_message,
)


SAMPLE_RULES = [
    {"name": "TheoreticalOccupancy", "speedup_pct": 24.0},
    {"name": "MemoryBound", "speedup_pct": 8.0},
]


def test_system_prompt_contains_role():
    assert "Planner Agent" in SYSTEM_PROMPT
    assert "submit_plan" in SYSTEM_PROMPT


def test_system_prompt_has_no_step_by_step_algorithm():
    # Must NOT contain hardcoded reasoning steps
    assert "Step 1" not in SYSTEM_PROMPT
    assert "Step 2" not in SYSTEM_PROMPT
    assert "Step 3" not in SYSTEM_PROMPT


def test_build_diversity_hint_first_plan():
    hint = build_diversity_hint(0, SAMPLE_RULES)
    assert "TheoreticalOccupancy" in hint
    assert "24.0" in hint


def test_build_diversity_hint_second_plan():
    hint = build_diversity_hint(1, SAMPLE_RULES)
    assert "MemoryBound" in hint
    assert "8.0" in hint


def test_build_diversity_hint_beyond_rules():
    hint = build_diversity_hint(5, SAMPLE_RULES)
    assert "raw metrics" in hint.lower()


def test_build_diversity_hint_empty_rules():
    hint = build_diversity_hint(0, [])
    assert "raw metrics" in hint.lower() or "no" in hint.lower()


def test_build_initial_message_is_human_message():
    hint = build_diversity_hint(0, SAMPLE_RULES)
    msg = build_initial_message(hint, [])
    assert isinstance(msg, HumanMessage)


def test_build_initial_message_contains_hint():
    hint = "Target rule 'TheoreticalOccupancy' (estimated speedup: 24.0%)"
    msg = build_initial_message(hint, [])
    assert "TheoreticalOccupancy" in msg.content


def test_build_initial_message_contains_tried_directions():
    msg = build_initial_message("hint", ["reduce_register_pressure", "increase_tile_k"])
    assert "reduce_register_pressure" in msg.content
    assert "increase_tile_k" in msg.content


def test_build_initial_message_empty_directions():
    msg = build_initial_message("hint", [])
    assert isinstance(msg.content, str)
    assert len(msg.content) > 0
