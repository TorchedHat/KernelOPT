"""Shared LLM factory — uses Claude on Vertex AI via langchain-anthropic."""

from __future__ import annotations

import os
import re
from functools import cached_property

from langchain_core.language_models.chat_models import BaseChatModel

from kernelopt.constants import VERTEX_API_TIMEOUT_S


DEFAULT_MODEL = "claude-sonnet-4-6"
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
                timeout=VERTEX_API_TIMEOUT_S,  # 15 min hard timeout — prevents infinite hangs on Vertex API
            )

        @cached_property
        def _async_client(self):
            return AsyncAnthropicVertex(
                project_id=self._vertex_project_id,
                region=self._vertex_region,
                timeout=VERTEX_API_TIMEOUT_S,
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
    """
    Strip <think>...</think> blocks from thinking model output.

    Handles nested blocks by repeatedly removing innermost blocks first.
    Pattern uses negative lookahead (?!<think>) to match blocks with no
    nested <think> tags inside them.

    Examples:
        - "A <think>B</think> C" → "A  C"
        - "A <think>1</think> B <think>2</think> C" → "A  B  C"
        - "A <think>outer <think>inner</think> outer</think> B" → "A  B"
    """
    # Repeatedly remove innermost blocks (blocks with no nested <think>)
    while '<think>' in content:
        prev = content
        # (?:(?!<think>)[\s\S])*? matches any char that's NOT start of <think>
        content = re.sub(r"<think>(?:(?!<think>)[\s\S])*?</think>", "", content)
        if prev == content:  # Safety: prevent infinite loop on malformed input
            break
    return content.strip()
