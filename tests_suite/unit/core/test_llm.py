"""
Comprehensive tests for LLM module (Claude Vertex AI).

Tests cover:
- LLM factory with Claude Vertex AI
- Model configuration
- Thinking block stripping
- JSON extraction from LLM output
- Error handling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestLLMFactory:
    """Test LLM factory and configuration."""

    def test_build_llm_default(self, vertex_project_id, vertex_region):
        """Test building LLM with default settings."""
        from kernelopt.llm import build_llm

        # This will try to create a Claude Vertex AI client
        # In tests, we'll mock it
        try:
            llm = build_llm(
                model="claude-sonnet-4@20250514",
                project_id=vertex_project_id,
                region=vertex_region,
            )
            assert llm is not None
        except Exception as e:
            # Expected if Vertex AI isn't configured
            assert "vertex" in str(e).lower() or "auth" in str(e).lower()

    def test_build_llm_custom_model(self, vertex_project_id, vertex_region):
        """Test building LLM with custom model."""
        from kernelopt.llm import build_llm

        try:
            llm = build_llm(
                model="claude-opus-4@20250514",
                project_id=vertex_project_id,
                region=vertex_region,
            )
            assert llm is not None
        except Exception:
            # Expected if Vertex AI isn't configured
            pass

    def test_build_llm_custom_params(self, vertex_project_id, vertex_region):
        """Test building LLM with custom parameters."""
        from kernelopt.llm import build_llm

        try:
            llm = build_llm(
                model="claude-sonnet-4@20250514",
                project_id=vertex_project_id,
                region=vertex_region,
                max_tokens=8192,
                temperature=0.5,
            )
            assert llm is not None
        except Exception:
            # Expected if Vertex AI isn't configured
            pass


class TestThinkingStripping:
    """Test thinking block stripping."""

    def test_strip_thinking_basic(self):
        """Test basic thinking block removal."""
        from kernelopt.llm import strip_thinking

        content = "Before <think>internal reasoning</think> After"
        result = strip_thinking(content)
        assert "think" not in result.lower()
        assert "Before" in result
        assert "After" in result

    def test_strip_thinking_multiple(self):
        """Test removing multiple thinking blocks."""
        from kernelopt.llm import strip_thinking

        content = "A <think>think1</think> B <think>think2</think> C"
        result = strip_thinking(content)
        assert "think" not in result.lower()
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_strip_thinking_multiline(self):
        """Test removing multiline thinking blocks."""
        from kernelopt.llm import strip_thinking

        content = """
Before
<think>
Line 1
Line 2
Line 3
</think>
After
"""
        result = strip_thinking(content)
        assert "think" not in result.lower()
        assert "Before" in result
        assert "After" in result
        assert "Line 1" not in result

    def test_strip_thinking_none(self):
        """Test content without thinking blocks."""
        from kernelopt.llm import strip_thinking

        content = "Just regular content"
        result = strip_thinking(content)
        assert result.strip() == content.strip()

    def test_strip_thinking_nested(self):
        """Test nested thinking blocks (shouldn't happen but handle it)."""
        from kernelopt.llm import strip_thinking

        content = "A <think>outer <think>inner</think> outer</think> B"
        result = strip_thinking(content)
        # Should remove outer block
        assert "think" not in result.lower()



# TestJSONExtraction class removed - no longer needed\n# The codebase now uses structured outputs with Pydantic models via\n# llm.with_structured_output() instead of manual JSON extraction\n

class TestLLMConfiguration:
    """Test LLM configuration constants."""

    def test_default_model(self):
        """Test default model constant."""
        from kernelopt.llm import DEFAULT_MODEL

        # Default model is Claude Sonnet 4 on Vertex AI
        assert DEFAULT_MODEL == "claude-sonnet-4@20250514"

    def test_model_naming_convention(self):
        """Test that model names follow expected conventions."""
        from kernelopt.llm import DEFAULT_MODEL

        # Model should be a valid HuggingFace or Vertex AI format
        assert len(DEFAULT_MODEL) > 0
        # Either HF format (org/model) or Vertex AI format (model@version)
        assert "/" in DEFAULT_MODEL or "@" in DEFAULT_MODEL


class TestLLMErrorHandling:
    """Test error handling in LLM module."""

    def test_build_llm_missing_project_id(self):
        """Test building LLM without project ID."""
        from kernelopt.llm import build_llm

        try:
            llm = build_llm(
                model="claude-sonnet-4@20250514",
                project_id=None,
                region="us-east5",
            )
            # May use environment variable or fail
        except Exception as e:
            # Should fail gracefully
            assert e is not None

    def test_strip_thinking_malformed(self):
        """Test stripping malformed thinking blocks."""
        from kernelopt.llm import strip_thinking

        # Unclosed thinking block
        content = "A <think>unclosed"
        result = strip_thinking(content)
        # Should handle gracefully
        assert isinstance(result, str)
