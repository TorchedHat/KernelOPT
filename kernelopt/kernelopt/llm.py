"""Shared LLM factory — Vertex, Anthropic API, OpenAI-compatible, Gemini, Hugging Face, local."""

from __future__ import annotations

import os
import re
from functools import cached_property
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from kernelopt.constants import VERTEX_API_TIMEOUT_S

if TYPE_CHECKING:
    from kernelopt.state import PipelineConfig

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_PROJECT_ID = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
DEFAULT_REGION = "us-east5"

PROVIDERS: tuple[tuple[str, str], ...] = (
    ("vertex", "Claude on Google Vertex AI"),
    ("anthropic_api", "Anthropic API (direct, API key)"),
    ("openai", "OpenAI-compatible APIs (OpenAI, Azure, Groq, etc.)"),
    ("google_genai", "Google Gemini API"),
    ("huggingface", "Hugging Face Inference"),
    ("local", "Local model server (vLLM, SGLang, Ollama, etc.)"),
)


def pipeline_llm_base_kwargs(cfg: "PipelineConfig") -> dict:
    """Kwargs merged into every agent `llm_config` (plus per-agent `model`)."""
    return {
        "api_key": cfg.api_key,
        "project_id": cfg.project_id or None,
        "region": cfg.region or None,
        "provider": cfg.llm_provider,
        "hf_inference_provider": cfg.hf_inference_provider,
        "openai_base_url": cfg.openai_base_url,
    }


# ---------------------------------------------------------------------------
# Provider builders (lazy imports — only the chosen provider's deps load)
# ---------------------------------------------------------------------------

def _build_vertex_llm(
    model: str,
    project_id: str,
    region: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
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
                timeout=VERTEX_API_TIMEOUT_S,
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


def _build_anthropic_api_llm(
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API: set ANTHROPIC_API_KEY or pass --api-key")
    return ChatAnthropic(
        model=model, api_key=key, max_tokens=max_tokens, temperature=temperature,
    )


def _build_openai_llm(
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    base_url: str | None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI-compatible API: set OPENAI_API_KEY or pass --api-key")
    kwargs: dict = {
        "model": model, "api_key": key,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    url = base_url or os.environ.get("OPENAI_BASE_URL")
    if url:
        kwargs["base_url"] = url.strip()
    return ChatOpenAI(**kwargs)


def _build_google_genai_llm(
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google Gemini: set GOOGLE_API_KEY or GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=key,
        max_output_tokens=max_tokens, temperature=temperature,
    )


def _build_huggingface_llm(
    model: str,
    api_key: str | None,
    hf_inference_provider: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    token = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Hugging Face: set HF_TOKEN or pass --api-key")
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError:
        from langchain_community.chat_models import ChatHuggingFace
        from langchain_community.llms import HuggingFaceEndpoint

    base_kw: dict = {
        "repo_id": model, "huggingfacehub_api_token": token,
        "temperature": temperature, "max_new_tokens": max_tokens,
    }
    if hf_inference_provider and hf_inference_provider != "auto":
        base_kw["provider"] = hf_inference_provider
    try:
        endpoint = HuggingFaceEndpoint(**base_kw)
    except TypeError:
        base_kw.pop("provider", None)
        try:
            endpoint = HuggingFaceEndpoint(**base_kw)
        except TypeError:
            endpoint = HuggingFaceEndpoint(
                repo_id=model, huggingfacehub_api_token=token,
                model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
            )
    return ChatHuggingFace(llm=endpoint)


def _build_local_llm(
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: str | None,
    api_key: str | None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    url = base_url or os.environ.get("LOCAL_LLM_BASE_URL") or "http://localhost:8000/v1"
    key = api_key or os.environ.get("LOCAL_LLM_API_KEY") or "not-needed"
    return ChatOpenAI(
        model=model, api_key=key, base_url=url.strip(),
        max_tokens=max_tokens, temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    project_id: str | None = None,
    region: str | None = None,
    provider: str = "vertex",
    hf_inference_provider: str = "auto",
    openai_base_url: str | None = None,
    **_: object,
) -> BaseChatModel:
    """Return a LangChain chat model for the given provider.

    Extra keyword arguments are silently ignored so callers can pass full
    llm_config dicts without filtering provider-specific keys.
    """
    pid = (provider or "vertex").strip().lower()
    if pid == "vertex":
        return _build_vertex_llm(
            model=model,
            project_id=project_id or DEFAULT_PROJECT_ID,
            region=region or DEFAULT_REGION,
            max_tokens=max_tokens, temperature=temperature,
        )
    if pid == "anthropic_api":
        return _build_anthropic_api_llm(
            model=model, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
        )
    if pid == "openai":
        return _build_openai_llm(
            model=model, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
            base_url=openai_base_url,
        )
    if pid in ("google_genai", "gemini"):
        return _build_google_genai_llm(
            model=model, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
        )
    if pid in ("huggingface", "hf"):
        return _build_huggingface_llm(
            model=model, api_key=api_key,
            hf_inference_provider=hf_inference_provider,
            max_tokens=max_tokens, temperature=temperature,
        )
    if pid == "local":
        return _build_local_llm(
            model=model, max_tokens=max_tokens, temperature=temperature,
            base_url=openai_base_url, api_key=api_key,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        f"Use: vertex, anthropic_api, openai, google_genai, huggingface, local"
    )


def strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from thinking model output.

    No-op when content has no <think> tags.
    """
    while '<think>' in content:
        prev = content
        content = re.sub(r"<think>(?:(?!<think>)[\s\S])*?</think>", "", content)
        if prev == content:
            break
    return content.strip()
