"""Shared LLM factory — uses a local vLLM server with OpenAI-compatible API."""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel


DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    base_url: str = DEFAULT_BASE_URL,
) -> BaseChatModel:
    """
    Return a LangChain chat model pointing at a local vLLM server.

    Uses ChatOpenAI with a custom base_url (OpenAI-compatible API).
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from Qwen3 thinking model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()


def extract_json(content: str) -> str:
    """Extract JSON from LLM output, stripping thinking blocks and markdown."""
    content = strip_thinking(content)

    # Strip markdown fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content[: content.rfind("```")]
    content = content.strip()

    # If content doesn't start with {, try to find JSON in it
    if not content.startswith("{") and not content.startswith("["):
        match = re.search(r"(\{[\s\S]*\})", content)
        if match:
            content = match.group(1)

    return content
