"""Shared LLM factory — uses Claude on Vertex AI via langchain-anthropic."""

from __future__ import annotations

import os
import re
from functools import cached_property

from langchain_core.language_models.chat_models import BaseChatModel


DEFAULT_MODEL = "claude-sonnet-4@20250514"
DEFAULT_PROJECT_ID = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
DEFAULT_REGION = "us-east5"


def _build_vertex_llm(
    model: str,
    project_id: str,
    region: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    """Return a ChatAnthropic subclass backed by AnthropicVertex."""
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
    from langchain_anthropic import ChatAnthropic

    class _ChatAnthropicVertex(ChatAnthropic):
        _vertex_project_id: str = project_id
        _vertex_region: str = region

        @cached_property
        def _client(self):
            return AnthropicVertex(
                project_id=self._vertex_project_id,
                region=self._vertex_region,
            )

        @cached_property
        def _async_client(self):
            return AsyncAnthropicVertex(
                project_id=self._vertex_project_id,
                region=self._vertex_region,
            )

    return _ChatAnthropicVertex(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        anthropic_api_key="not-needed",
    )


def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    project_id: str | None = None,
    region: str | None = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model backed by Claude on Vertex AI.

    Uses ChatAnthropic with AnthropicVertex client underneath.
    """
    return _build_vertex_llm(
        model=model,
        project_id=project_id or DEFAULT_PROJECT_ID,
        region=region or DEFAULT_REGION,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from thinking model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
