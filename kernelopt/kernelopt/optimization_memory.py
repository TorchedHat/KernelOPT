"""Optimization memory that improves with every pipeline run.

Tracks which optimization strategies work on which kernel types,
and stores concrete before/after examples for the executor.

Design:
  - One JSON file (strategies.json) — maps kernel_type → ranked strategies
  - One directory (references/) — before/after kernel pairs on disk
  - Auto-updates after every plan execution
  - Planner gets ranked strategies; executor gets a concrete example
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from kernelopt.constants import MAX_SOURCE_CHARS


@dataclass
class StrategyRecord:
    name: str
    successes: int = 0
    failures: int = 0
    total_speedup: float = 0.0
    best_speedup: float = 0.0
    reference_id: str = ""

    @property
    def attempts(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def avg_speedup(self) -> float:
        return self.total_speedup / self.attempts if self.attempts else 1.0

    @property
    def score(self) -> float:
        return self.success_rate * self.avg_speedup


class OptimizationMemory:
    """Learns which strategies work on which kernels across runs."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "references").mkdir(exist_ok=True)

        self._strategies_path = self.root / "strategies.json"
        self._data: dict[str, dict[str, dict]] = {}  # kernel_type → strategy_name → record

        if self._strategies_path.exists():
            raw = json.loads(self._strategies_path.read_text())
            for kt, strategies in raw.items():
                self._data[kt] = {
                    name: StrategyRecord(**rec) if isinstance(rec, dict) else rec
                    for name, rec in strategies.items()
                }

    def record(
        self,
        kernel_type: str,
        strategy: str,
        speedup: float,
        passed: bool,
        before_kernel: str = "",
        after_kernel: str = "",
        change_summary: str = "",
    ) -> None:
        """Record one optimization attempt. Call after every plan execution."""
        if kernel_type not in self._data:
            self._data[kernel_type] = {}

        if strategy not in self._data[kernel_type]:
            self._data[kernel_type][strategy] = StrategyRecord(name=strategy)

        rec = self._data[kernel_type][strategy]
        if isinstance(rec, dict):
            rec = StrategyRecord(**rec)
            self._data[kernel_type][strategy] = rec

        if passed and speedup > 1.0:
            rec.successes += 1
        else:
            rec.failures += 1
        rec.total_speedup += speedup

        if passed and speedup > rec.best_speedup and before_kernel and after_kernel:
            rec.best_speedup = speedup
            ref_id = f"{kernel_type}__{strategy}"[:80]
            ref_id = re.sub(r'[^\w]', '_', ref_id)
            rec.reference_id = ref_id
            ref_path = self.root / "references" / f"{ref_id}.json"
            ref_path.write_text(json.dumps({
                "kernel_type": kernel_type,
                "strategy": strategy,
                "speedup": speedup,
                "change_summary": change_summary,
                "before": before_kernel[:MAX_SOURCE_CHARS],
                "after": after_kernel[:MAX_SOURCE_CHARS],
            }, indent=2))

        self._save()

    def get_strategies(self, kernel_type: str) -> list[dict]:
        """Get strategies ranked by score for a kernel type.

        Falls back to all-type aggregate if no data for this kernel type.
        """
        strategies = self._data.get(kernel_type, {})
        if not strategies:
            strategies = self._aggregate_all()
        if not strategies:
            return []

        ranked = []
        for name, rec in strategies.items():
            if isinstance(rec, dict):
                rec = StrategyRecord(**rec)
            ranked.append({
                "name": rec.name,
                "success_rate": round(rec.success_rate, 2),
                "avg_speedup": round(rec.avg_speedup, 2),
                "best_speedup": round(rec.best_speedup, 2),
                "attempts": rec.attempts,
                "score": round(rec.score, 3),
                "avoid": rec.success_rate < 0.3 and rec.attempts >= 3,
                "reference_id": rec.reference_id,
            })

        return sorted(ranked, key=lambda s: s["score"], reverse=True)

    def get_reference(self, ref_id: str) -> dict | None:
        """Load a before/after kernel pair."""
        ref_path = self.root / "references" / f"{ref_id}.json"
        if ref_path.exists():
            return json.loads(ref_path.read_text())
        return None

    def format_for_planner(self, kernel_type: str) -> str:
        """Inject into the planner prompt."""
        strategies = self.get_strategies(kernel_type)
        if not strategies:
            return ""

        lines = [f"## Past Results ({kernel_type} kernels)\n"]
        for i, s in enumerate(strategies[:6], 1):
            if s["avoid"]:
                lines.append(
                    f"{i}. ~~{s['name']}~~ — AVOID ({s['success_rate']:.0%} success, "
                    f"{s['attempts']} attempts, avg {s['avg_speedup']:.2f}x)"
                )
            else:
                lines.append(
                    f"{i}. **{s['name']}** — {s['success_rate']:.0%} success, "
                    f"avg {s['avg_speedup']:.2f}x, best {s['best_speedup']:.2f}x "
                    f"({s['attempts']} attempts)"
                )
        return "\n".join(lines)

    def format_for_executor(self, kernel_type: str) -> str:
        """Inject the best reference pair into the executor prompt."""
        strategies = self.get_strategies(kernel_type)
        for s in strategies:
            if s["reference_id"] and not s["avoid"]:
                ref = self.get_reference(s["reference_id"])
                if ref and ref.get("before") and ref.get("after"):
                    return (
                        f"## Reference: {ref['kernel_type']} kernel optimized "
                        f"{ref['speedup']:.2f}x\n"
                        f"Strategy: {ref['strategy']}\n"
                        f"Key change: {ref.get('change_summary', '')}\n\n"
                        f"### Before:\n```python\n{ref['before'][:1200]}\n```\n\n"
                        f"### After ({ref['speedup']:.2f}x faster):\n"
                        f"```python\n{ref['after'][:1200]}\n```"
                    )
        return ""

    @staticmethod
    def classify_kernel(kernel_source: str) -> str:
        """Classify a kernel into a pattern type."""
        src = kernel_source.lower()
        if "tl.dot" in src or "block_m" in src or "block_n" in src or "block_k" in src:
            return "matmul"
        if "tl.sum" in src or "tl.max" in src or "rblock" in src:
            return "reduction"
        fused = re.findall(r'fused_(\w+?)_\d', src)
        if fused:
            return f"fused_{fused[0]}"
        if "conv" in src:
            return "convolution"
        return "pointwise"

    def _aggregate_all(self) -> dict[str, StrategyRecord]:
        """Aggregate strategies across all kernel types as fallback."""
        agg: dict[str, StrategyRecord] = {}
        for strategies in self._data.values():
            for name, rec in strategies.items():
                if isinstance(rec, dict):
                    rec = StrategyRecord(**rec)
                if name not in agg:
                    agg[name] = StrategyRecord(name=name)
                a = agg[name]
                a.successes += rec.successes
                a.failures += rec.failures
                a.total_speedup += rec.total_speedup
                if rec.best_speedup > a.best_speedup:
                    a.best_speedup = rec.best_speedup
                    a.reference_id = rec.reference_id
        return agg

    def _save(self) -> None:
        out = {}
        for kt, strategies in self._data.items():
            out[kt] = {}
            for name, rec in strategies.items():
                if isinstance(rec, StrategyRecord):
                    out[kt][name] = asdict(rec)
                else:
                    out[kt][name] = rec
        self._strategies_path.write_text(json.dumps(out, indent=2))
