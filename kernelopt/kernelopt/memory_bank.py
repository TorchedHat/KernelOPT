"""Persistent cross-run optimization memory bank with semantic retrieval.

Stores optimization experiences (successes, failures, constraints) across
kernelopt runs. Grows organically: each run adds entries, future runs query
them for strategy decisions and plan guidance.

Features:
  - Structural matching by kernel_type + hardware + outcome
  - Semantic retrieval via sentence-transformers/all-MiniLM-L6-v2 embeddings
  - Automatic constraint extraction from repeated failure patterns
  - Differential profiling deltas (NCU metric before/after)
  - Confidence weighting by speedup magnitude and recency
  - Code pattern templates as reusable diffs

Storage: JSON file at ~/.kernelopt/memory_bank.json (or custom path via
KERNELOPT_MEMORY_BANK env var). Embeddings cached alongside as .npy.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BANK_PATH = Path.home() / ".kernelopt" / "memory_bank.json"
MAX_ENTRIES = 500
CONSTRAINT_THRESHOLD = 3  # failures before auto-extracting a constraint


class MemoryEntry:
    """A single optimization experience."""

    def __init__(
        self,
        kernel_structure: dict,
        hardware: dict,
        strategy: dict,
        outcome: str,
        speedup: float = 0.0,
        failure_reason: str = "",
        constraint: str = "",
        optimization_insight: str = "",
        code_pattern: str = "",
        code_diff: str = "",
        template_applicable: str = "",
        metric_deltas: dict | None = None,
        entry_id: str | None = None,
        timestamp: float | None = None,
    ):
        self.id = entry_id or str(uuid.uuid4())[:12]
        self.timestamp = timestamp or time.time()
        self.kernel_structure = kernel_structure
        self.hardware = hardware
        self.strategy = strategy
        self.outcome = outcome
        self.speedup = speedup
        self.failure_reason = failure_reason
        self.constraint = constraint
        self.optimization_insight = optimization_insight
        self.code_pattern = code_pattern
        self.code_diff = code_diff
        self.template_applicable = template_applicable
        self.metric_deltas = metric_deltas or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "kernel_structure": self.kernel_structure,
            "hardware": self.hardware,
            "strategy": self.strategy,
            "outcome": self.outcome,
            "speedup": self.speedup,
            "failure_reason": self.failure_reason,
            "constraint": self.constraint,
            "optimization_insight": self.optimization_insight,
            "code_pattern": self.code_pattern[:500] if self.code_pattern else "",
            "code_diff": self.code_diff[:2000] if self.code_diff else "",
            "template_applicable": self.template_applicable,
            "metric_deltas": self.metric_deltas,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MemoryEntry:
        return cls(
            entry_id=d.get("id"),
            timestamp=d.get("timestamp"),
            kernel_structure=d.get("kernel_structure", {}),
            hardware=d.get("hardware", {}),
            strategy=d.get("strategy", {}),
            outcome=d.get("outcome", "unknown"),
            speedup=d.get("speedup", 0.0),
            failure_reason=d.get("failure_reason", ""),
            constraint=d.get("constraint", ""),
            optimization_insight=d.get("optimization_insight", ""),
            code_pattern=d.get("code_pattern", ""),
            code_diff=d.get("code_diff", ""),
            template_applicable=d.get("template_applicable", ""),
            metric_deltas=d.get("metric_deltas", {}),
        )


def classify_kernel_structure(kernel_source: str, profiling_context: dict) -> dict:
    """Classify a kernel's structure for memory matching."""
    source_lower = kernel_source.lower()
    multi_kernel = profiling_context.get("multi_kernel_summary", "")

    kernel_type = "unknown"
    if "torch.mm" in kernel_source or "extern_kernels.mm" in kernel_source:
        if "@triton.jit" in kernel_source:
            kernel_type = "gemm_epilogue"
        else:
            kernel_type = "gemm"
    elif "softmax" in source_lower:
        kernel_type = "softmax"
    elif "layernorm" in source_lower or "layer_norm" in source_lower:
        kernel_type = "layernorm"
    elif "rmsnorm" in source_lower or "rms_norm" in source_lower:
        kernel_type = "rmsnorm"
    elif "attention" in source_lower:
        kernel_type = "attention"
    elif "conv" in source_lower:
        kernel_type = "conv_epilogue" if "@triton.jit" in kernel_source else "conv"
    elif "@triton.jit" in kernel_source:
        if "tl.sum" in kernel_source or "tl.max" in kernel_source:
            kernel_type = "reduction"
        else:
            kernel_type = "pointwise"
    elif "__global__" in kernel_source:
        kernel_type = "cuda"

    has_library = any(p in kernel_source for p in ["torch.mm", "extern_kernels", "torch.addmm"])
    has_triton = "@triton.jit" in kernel_source

    return {
        "type": kernel_type,
        "has_library_calls": has_library,
        "has_triton": has_triton,
        "multi_kernel": bool(multi_kernel and "single" not in multi_kernel.lower()),
    }


def get_hardware_info() -> dict:
    """Get current GPU hardware info for memory matching."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            shmem = getattr(props, "shared_memory_per_block_optin",
                            getattr(props, "shared_memory_per_block", 0))
            return {
                "gpu": props.name,
                "cc": f"{props.major}.{props.minor}",
                "shared_memory_kb": shmem // 1024,
                "sm_count": props.multi_processor_count,
            }
    except Exception:
        pass
    return {"gpu": "unknown", "cc": "unknown", "shared_memory_kb": 0, "sm_count": 0}


def compute_metric_deltas(baseline_ctx: dict, candidate_ctx: dict) -> dict:
    """Compare NCU metrics between baseline and candidate for differential profiling.

    Returns a dict of metric_name -> {before, after, delta_pct} for metrics
    that changed significantly. This provides causal evidence: did the
    intended NCU metric actually improve?
    """
    deltas = {}
    sections_to_compare = [
        "GPU Speed Of Light Throughput",
        "Memory Workload Analysis",
        "Occupancy",
    ]
    for section in sections_to_compare:
        base_metrics = baseline_ctx.get(section, {})
        cand_metrics = candidate_ctx.get(section, {})
        for metric_name in base_metrics:
            if metric_name not in cand_metrics:
                continue
            try:
                b = float(str(base_metrics[metric_name]).replace("%", "").strip())
                c = float(str(cand_metrics[metric_name]).replace("%", "").strip())
                if b > 0:
                    delta_pct = round(100 * (c - b) / b, 1)
                    if abs(delta_pct) > 1.0:
                        deltas[metric_name] = {
                            "before": round(b, 2),
                            "after": round(c, 2),
                            "delta_pct": delta_pct,
                        }
            except (ValueError, TypeError):
                pass
    return deltas


class MemoryBank:
    """Persistent cross-run optimization memory with semantic retrieval."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(
            os.environ.get("KERNELOPT_MEMORY_BANK", str(DEFAULT_BANK_PATH))
        )
        self.entries: list[MemoryEntry] = []
        self._embeddings: np.ndarray | None = None
        self._embed_model = None
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.entries = [MemoryEntry.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self.entries = []

        emb_path = self.path.with_suffix(".embeddings.npy")
        if emb_path.exists():
            try:
                self._embeddings = np.load(str(emb_path))
                if len(self._embeddings) != len(self.entries):
                    self._embeddings = None
            except Exception:
                self._embeddings = None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        self.path.write_text(json.dumps(data, indent=2))

        if self._embeddings is not None and len(self._embeddings) == len(self.entries):
            emb_path = self.path.with_suffix(".embeddings.npy")
            np.save(str(emb_path), self._embeddings)

    # Embedding models in priority order, selected for code/technical content
    # retrieval on CPU. Override with KERNELOPT_EMBED_MODEL env var.
    EMBED_MODELS = [
        "codefuse-ai/F2LLM-v2-330M",
        "ibm-granite/granite-embedding-311m-multilingual-r2",
        "jinaai/jina-embeddings-v3",
        "nomic-ai/nomic-embed-text-v1.5",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    def _get_embed_model(self):
        """Lazy-load sentence-transformers model.

        Tries models in priority order (best for code retrieval first).
        Falls back gracefully if a model can't be loaded.
        Override with KERNELOPT_EMBED_MODEL env var for a specific model.
        """
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic retrieval disabled. Install with: "
                    "pip install sentence-transformers"
                )
                return None

            override = os.environ.get("KERNELOPT_EMBED_MODEL")
            candidates = [override] if override else self.EMBED_MODELS

            for model_name in candidates:
                try:
                    self._embed_model = SentenceTransformer(
                        model_name,
                        device="cpu",
                        trust_remote_code=True,
                    )
                    logger.info(f"Loaded embedding model: {model_name}")
                    break
                except Exception as e:
                    logger.debug(f"Could not load {model_name}: {e}")
                    continue

            if self._embed_model is None:
                logger.warning(
                    f"No embedding model could be loaded from: {candidates}. "
                    f"Semantic retrieval disabled."
                )
        return self._embed_model

    def _embed_entry(self, entry: MemoryEntry) -> np.ndarray | None:
        """Create a semantic embedding for a memory entry."""
        model = self._get_embed_model()
        if model is None:
            return None
        text = (
            f"kernel_type:{entry.kernel_structure.get('type', '')} "
            f"has_library:{entry.kernel_structure.get('has_library_calls', '')} "
            f"multi_kernel:{entry.kernel_structure.get('multi_kernel', '')} "
            f"strategy:{entry.strategy.get('type', '')} "
            f"direction:{entry.strategy.get('direction', '')} "
            f"outcome:{entry.outcome} "
            f"gpu:{entry.hardware.get('gpu', '')} "
            f"{entry.optimization_insight[:200]} "
            f"{entry.code_pattern[:200]}"
        )
        return model.encode(text, normalize_embeddings=True)

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)
        if len(self.entries) > MAX_ENTRIES:
            self.entries = self.entries[-MAX_ENTRIES:]
            self._embeddings = None

        emb = self._embed_entry(entry)
        if emb is not None:
            if self._embeddings is not None and len(self._embeddings) == len(self.entries) - 1:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            else:
                self._embeddings = None

        self.save()

    def query(
        self,
        kernel_structure: dict | None = None,
        hardware: dict | None = None,
        outcome: str | None = None,
        limit: int = 10,
        use_confidence: bool = True,
    ) -> list[MemoryEntry]:
        """Query by structural matching with confidence weighting."""
        results = []
        for entry in self.entries:
            score = 0.0

            if kernel_structure:
                if entry.kernel_structure.get("type") == kernel_structure.get("type"):
                    score += 10
                if entry.kernel_structure.get("has_library_calls") == kernel_structure.get("has_library_calls"):
                    score += 3
                if entry.kernel_structure.get("multi_kernel") == kernel_structure.get("multi_kernel"):
                    score += 2

            if hardware and entry.hardware.get("gpu") == hardware.get("gpu"):
                score += 5

            if outcome and entry.outcome == outcome:
                score += 3

            if use_confidence:
                if entry.outcome == "success" and entry.speedup > 1.0:
                    score *= entry.speedup
                age_days = (time.time() - entry.timestamp) / 86400
                score *= max(0.5, 1.0 - age_days / 365)

            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [e for _, e in results[:limit]]

    def query_semantic(
        self,
        query_text: str,
        limit: int = 5,
        outcome: str | None = None,
    ) -> list[MemoryEntry]:
        """Cosine similarity search over embedded experiences.

        query_text should be a concatenation of kernel source snippet + NCU
        summary. Uses all-MiniLM-L6-v2 embeddings for semantic matching
        (matmul+relu matches matmul+gelu).
        """
        model = self._get_embed_model()
        if model is None or not self.entries:
            return []

        if self._embeddings is None or len(self._embeddings) != len(self.entries):
            self._rebuild_embeddings()

        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_emb = model.encode(query_text[:1000], normalize_embeddings=True)
        scores = np.dot(self._embeddings, query_emb)

        indices = np.argsort(-scores)
        results = []
        for idx in indices:
            entry = self.entries[idx]
            if outcome and entry.outcome != outcome:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def _rebuild_embeddings(self):
        """Rebuild all embeddings from scratch."""
        model = self._get_embed_model()
        if model is None or not self.entries:
            return
        texts = []
        for e in self.entries:
            texts.append(
                f"kernel_type:{e.kernel_structure.get('type', '')} "
                f"has_library:{e.kernel_structure.get('has_library_calls', '')} "
                f"strategy:{e.strategy.get('type', '')} "
                f"direction:{e.strategy.get('direction', '')} "
                f"outcome:{e.outcome} "
                f"gpu:{e.hardware.get('gpu', '')} "
                f"{e.optimization_insight[:200]} "
                f"{e.code_pattern[:200]}"
            )
        self._embeddings = model.encode(texts, normalize_embeddings=True)
        self.save()

    def get_failures(self, kernel_structure: dict, hardware: dict | None = None) -> list[MemoryEntry]:
        """Get failure entries for similar kernels."""
        return self.query(kernel_structure=kernel_structure, hardware=hardware, outcome="failure", limit=5)

    def get_successes(self, kernel_structure: dict, hardware: dict | None = None) -> list[MemoryEntry]:
        """Get success entries for similar kernels."""
        return self.query(kernel_structure=kernel_structure, hardware=hardware, outcome="success", limit=5)

    def get_templates(self, kernel_source: str, ncu_summary: str, limit: int = 3) -> list[MemoryEntry]:
        """Get reusable code templates from semantically similar successful optimizations."""
        query_text = f"{kernel_source[:500]} {ncu_summary[:200]}"
        successes = self.query_semantic(query_text, limit=limit * 2, outcome="success")
        return [e for e in successes if e.code_diff][:limit]

    def extract_constraints(self) -> list[dict]:
        """Auto-generate constraint rules from repeated failure patterns.

        When the same (kernel_type, strategy_type, hardware) combination fails
        >= CONSTRAINT_THRESHOLD times with 0 successes, emit a structured
        constraint that the strategy LLM can interpret.
        """
        pattern_counts: dict[tuple, list[MemoryEntry]] = {}
        for entry in self.entries:
            if entry.outcome != "failure":
                continue
            key = (
                entry.kernel_structure.get("type"),
                entry.strategy.get("type"),
                entry.hardware.get("gpu"),
            )
            pattern_counts.setdefault(key, []).append(entry)

        constraints = []
        for (ktype, strategy, gpu), failures in pattern_counts.items():
            if len(failures) < CONSTRAINT_THRESHOLD:
                continue
            successes = [
                e for e in self.entries
                if e.kernel_structure.get("type") == ktype
                and e.strategy.get("type") == strategy
                and e.outcome == "success"
            ]
            if len(successes) > 0:
                continue
            common_reasons = {}
            for f in failures:
                reason = f.failure_reason[:100] if f.failure_reason else "unknown"
                common_reasons[reason] = common_reasons.get(reason, 0) + 1
            top_reason = max(common_reasons, key=common_reasons.get)

            constraints.append({
                "kernel_type": ktype,
                "strategy": strategy,
                "hardware": gpu,
                "failures": len(failures),
                "successes": 0,
                "common_reason": top_reason,
                "rule": (
                    f"Strategy '{strategy}' on {ktype} kernels has failed "
                    f"{len(failures)} times with 0 successes on {gpu}. "
                    f"Most common failure: {top_reason}. Avoid this approach."
                ),
            })
        return constraints

    def format_for_llm(self, entries: list[MemoryEntry], max_entries: int = 3) -> str:
        """Format memory entries as context for LLM prompts."""
        if not entries:
            return ""

        lines = ["## Relevant past optimization experiences:"]
        for i, e in enumerate(entries[:max_entries]):
            status = "SUCCESS" if e.outcome == "success" else "FAILURE"
            lines.append(
                f"\n### Experience {i+1} [{status}] (kernel type: {e.kernel_structure.get('type', '?')}, "
                f"GPU: {e.hardware.get('gpu', '?')})"
            )
            lines.append(f"Strategy: {e.strategy.get('type', '?')} -- {e.strategy.get('direction', '?')}")
            if e.outcome == "success":
                lines.append(f"Speedup: {e.speedup:.2f}x")
                if e.optimization_insight:
                    lines.append(f"Insight: {e.optimization_insight}")
                if e.code_diff:
                    lines.append(f"Reusable code diff:\n```diff\n{e.code_diff[:500]}\n```")
                elif e.code_pattern:
                    lines.append(f"Code pattern:\n```python\n{e.code_pattern}\n```")
                if e.metric_deltas:
                    delta_lines = []
                    for metric, delta in list(e.metric_deltas.items())[:5]:
                        delta_lines.append(
                            f"  {metric}: {delta['before']}% → {delta['after']}% "
                            f"({'+' if delta['delta_pct'] > 0 else ''}{delta['delta_pct']}%)"
                        )
                    if delta_lines:
                        lines.append("NCU metric changes:\n" + "\n".join(delta_lines))
            else:
                lines.append(f"Failure: {e.failure_reason}")
                if e.constraint:
                    lines.append(f"Constraint: {e.constraint}")

        return "\n".join(lines)

    def format_constraints_for_llm(self) -> str:
        """Format auto-extracted constraints as rules for the strategy LLM."""
        constraints = self.extract_constraints()
        if not constraints:
            return ""
        lines = ["## Optimization constraints (auto-extracted from past failures):"]
        for c in constraints:
            lines.append(
                f"- **{c['kernel_type']}** + **{c['strategy']}** on {c['hardware']}: "
                f"{c['rule']}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)
