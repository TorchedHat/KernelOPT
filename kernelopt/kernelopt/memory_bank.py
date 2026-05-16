"""Persistent cross-run optimization memory bank.

Stores optimization experiences (successes, failures, constraints) across
kernelopt runs. Grows organically: each run adds entries, future runs query
them for strategy decisions and plan guidance.

Storage: JSON file at ~/.kernelopt/memory_bank.json (or custom path).
Retrieval: structural matching by kernel_type + bottleneck + hardware.

Ships with a starter bank from KernelBench/SOL-ExecBench benchmarks.
Users' local banks diverge and grow independently.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_BANK_PATH = Path.home() / ".kernelopt" / "memory_bank.json"
MAX_ENTRIES = 500


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
        )


def classify_kernel_structure(kernel_source: str, profiling_context: dict) -> dict:
    """Classify a kernel's structure for memory matching.

    Returns a dict with type, characteristics, and dimensions.
    """
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


class MemoryBank:
    """Persistent cross-run optimization memory."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_BANK_PATH
        self.entries: list[MemoryEntry] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.entries = [MemoryEntry.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self.entries = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)
        if len(self.entries) > MAX_ENTRIES:
            self.entries = self.entries[-MAX_ENTRIES:]
        self.save()

    def query(
        self,
        kernel_structure: dict | None = None,
        hardware: dict | None = None,
        outcome: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Query the memory bank by structural matching.

        Returns entries sorted by relevance (most relevant first).
        """
        results = []
        for entry in self.entries:
            score = 0

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

            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [e for _, e in results[:limit]]

    def get_failures(self, kernel_structure: dict, hardware: dict | None = None) -> list[MemoryEntry]:
        """Get failure entries for similar kernels -- used to avoid repeating mistakes."""
        return self.query(
            kernel_structure=kernel_structure,
            hardware=hardware,
            outcome="failure",
            limit=5,
        )

    def get_successes(self, kernel_structure: dict, hardware: dict | None = None) -> list[MemoryEntry]:
        """Get success entries for similar kernels -- used to replicate patterns."""
        return self.query(
            kernel_structure=kernel_structure,
            hardware=hardware,
            outcome="success",
            limit=5,
        )

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
                if e.code_pattern:
                    lines.append(f"Code pattern:\n```python\n{e.code_pattern}\n```")
            else:
                lines.append(f"Failure: {e.failure_reason}")
                if e.constraint:
                    lines.append(f"Constraint: {e.constraint}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)
