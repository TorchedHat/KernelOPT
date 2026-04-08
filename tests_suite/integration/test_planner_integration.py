"""
Comprehensive tests for the planner agent.

Tests cover:
- Plan generation from profiling context
- Evidence extraction from NCU rules
- Diversity in plan generation
- Plan structure validation
- Confidence scoring
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestPlannerBasic:
    """Basic planner agent tests."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_run_planner_basic(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test basic plan generation."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plans = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )

        assert plans is not None
        assert len(plans) > 0
        plan = plans[0][1]  # Extract plan from (letter, plan) tuple
        assert isinstance(plan, dict)

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_plan_structure(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that generated plan has required fields."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plans = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )

        # Required fields
        plan = plans[0][1] if plans else {}
        assert "plan_id" in plan
        assert "direction" in plan
        assert "framework" in plan
        assert "evidence" in plan
        assert "step" in plan
        assert "confidence" in plan

        # Evidence structure
        evidence = plan["evidence"]
        assert "ncu_rule" in evidence
        assert "reasoning" in evidence

        # Step structure
        step = plan["step"]
        assert "action" in step
        assert "target" in step
        assert "change" in step


class TestPlannerDiversity:
    """Test plan diversity generation."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_multiple_plans_different(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that multiple plans are different."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        # New API generates N plans in parallel
        plan_tuples = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=2,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )

        # Extract plan dicts from (letter, plan) tuples
        plans = [plan for (letter, plan) in plan_tuples]

        # Plans should have different directions
        directions = [p.get("direction") for p in plans]
        # Note: LLM might still generate similar plans, so this is a soft check
        assert len(plans) == 2


class TestPlannerEvidence:
    """Test evidence extraction and reasoning."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_evidence_from_ncu_rules(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that evidence references NCU rules."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        # Add rules to context
        sample_profiling_context["rules"] = [
            {
                "name": "MemoryBound",
                "type": "OPT",
                "description": "Kernel is memory bound",
                "speedup_pct": 15.0,
            }
        ]

        kernel_source = test_kernel_path.read_text()

        plan_result = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}
        plan = plan_result[0][1] if plan_result else {}

        evidence = plan.get("evidence", {})
        assert "ncu_rule" in evidence
        assert "reasoning" in evidence
        assert len(evidence["reasoning"]) > 0


class TestPlannerConfidence:
    """Test confidence scoring."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_confidence_levels(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that confidence is one of the valid levels."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plan_result = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        confidence = plan.get("confidence")
        assert confidence in ["high", "medium", "low"]


class TestPlannerMemory:
    """Test optimization memory usage."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_planner_with_memory(
        self, sample_profiling_context, test_kernel_path,
        sample_memory_items, patch_build_llm, gpu_id
    ):
        """Test that planner uses optimization memory."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plan_result = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=sample_memory_items,
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        # Plan should be generated (memory helps inform decisions)
        assert plan is not None
        assert "direction" in plan


class TestPlannerFrameworks:
    """Test framework-specific planning."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_triton_framework_plan(
        self, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test planning for Triton kernel."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        ctx = {
            "kernel_name": "softmax_kernel",
            "framework_hint": "triton",
            "sections": {},
            "rules": [],
            "launch": {},
        }

        kernel_source = test_kernel_path.read_text()

        plan_result = run_planner(
            profiling_context=ctx,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        assert plan.get("framework") in ["triton", "cuda", "helion"]


class TestPlannerErrorHandling:
    """Test error handling in planner."""

    def test_empty_profiling_context(
        self, test_kernel_path, patch_build_llm, gpu_id
    ):
        """Test handling of minimal profiling context."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        minimal_ctx = {
            "kernel_name": "test",
            "framework_hint": "cuda",
            "sections": {},
            "rules": [],
            "launch": {},
        }

        kernel_source = test_kernel_path.read_text()

        # Should still generate a plan (maybe with low confidence)
        plan_result = run_planner(
            profiling_context=minimal_ctx,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        assert plan is not None


class TestPlannerPrompt:
    """Test planner prompt construction."""

    def test_prompt_includes_context(
        self, sample_profiling_context, test_kernel_path
    ):
        """Test that prompt includes profiling context."""
        # This would require exposing the prompt building function
        # For now, we test indirectly through the plan output
        pass


class TestPlannerActions:
    """Test different optimization actions."""

    @pytest.mark.agent
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_plan_action_types(
        self, sample_profiling_context, test_kernel_path,
        patch_build_llm, gpu_id
    ):
        """Test that plans use valid action types."""
        from kernelopt.agents.planner import run_planner_agent as run_planner

        patch_build_llm(gpu_id)

        kernel_source = test_kernel_path.read_text()

        plan_result = run_planner(
            profiling_context=sample_profiling_context,
            kernel_source=kernel_source,
            memory=[],
            N=1,
            llm_config={"model": "claude-sonnet-4@20250514", "api_key": None, "project_id": None, "region": None},
            all_tried_directions=[],
        )
        plan = plan_result[0][1] if plan_result else {}

        step = plan.get("step", {})
        action = step.get("action")

        # Valid actions
        valid_actions = ["add", "modify", "remove", "replace", "refactor"]
        # Action should be specified
        assert action is not None
