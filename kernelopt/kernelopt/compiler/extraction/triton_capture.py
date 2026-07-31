"""Extract metadata from Triton's MLIR compilation pipeline.

Uses triton.knobs.compilation.listener to intercept every kernel
compiled by Triton (including Inductor-triggered compilations).
Captures metadata, file paths to IR stages, and per-stage timing.

After compilation, reads TTIR/TTGIR text and parses structural
metrics (dot count, load/store count, async copy, warp specialization).

References:
    - CompilationListener protocol: triton/knobs.py lines 261-265
    - CompileTimes: triton/knobs.py lines 234-258
    - CompiledKernel.asm: triton/compiler/compiler.py lines 424-429
    - NVIDIA backend passes: triton/third_party/nvidia/backend/compiler.py
"""

from __future__ import annotations

import re
import threading
from typing import Any


_triton_store = threading.local()


def _get_store():
    if not hasattr(_triton_store, "compilations"):
        _triton_store.compilations = []
    return _triton_store


def install_triton_listener() -> bool:
    """Install Triton compilation listener.

    Call BEFORE torch.compile() in the extraction subprocess.
    Returns True if listener was installed.
    """
    try:
        import triton
    except ImportError:
        return False

    if not hasattr(triton.knobs.compilation, "listener"):
        return False

    store = _get_store()

    def _on_compilation(*, src, metadata, metadata_group, times, cache_hit):
        entry = {
            "kernel_name": metadata.get("name", ""),
            "cache_hit": cache_hit,
            "num_warps": metadata.get("num_warps", 4),
            "num_stages": metadata.get("num_stages", 1),
            "num_ctas": metadata.get("num_ctas", 1),
            "shared_bytes": metadata.get("shared", 0),
            "hash": metadata.get("hash", ""),
            "ir_files": dict(metadata_group) if metadata_group else {},
            "compilation_time_us": 0.0,
        }

        if times:
            total_us = getattr(times, "ir_initialization", 0)
            for stage_name, duration_us in getattr(times, "lowering_stages", []):
                total_us += duration_us
            total_us += getattr(times, "store_results", 0)
            entry["compilation_time_us"] = total_us

        store.compilations.append(entry)

    triton.knobs.compilation.listener = _on_compilation
    return True


def extract_triton_compilation_info() -> list[dict]:
    """After compilation, process captured Triton metadata.

    Reads TTIR/TTGIR text from file paths captured by the listener,
    parses structural metrics, and returns TritonCompilationInfo-compatible dicts.
    """
    store = _get_store()
    results = []

    for entry in store.compilations:
        if entry.get("cache_hit"):
            continue

        ttir_text = _read_ir_file(entry.get("ir_files", {}), "ttir")
        ttgir_text = _read_ir_file(entry.get("ir_files", {}), "ttgir")

        ir_metrics = _parse_ttir_metrics(ttir_text)
        layout_info = _parse_ttgir_layout(ttgir_text)

        results.append({
            "kernel_name": entry["kernel_name"],
            "num_warps": entry["num_warps"],
            "num_stages": entry["num_stages"],
            "shared_bytes": entry["shared_bytes"],
            "n_regs": 0,
            "n_spills": 0,
            "n_max_threads": 0,
            "ttir_text": ttir_text[:5000] if ttir_text else "",
            "ttgir_text": ttgir_text[:5000] if ttgir_text else "",
            "compilation_time_us": entry["compilation_time_us"],
            **ir_metrics,
            **layout_info,
        })

    return results


def _read_ir_file(ir_files: dict, stage: str) -> str:
    """Read IR text from a file path in the metadata_group dict."""
    for key, path in ir_files.items():
        if key.endswith(f".{stage}"):
            try:
                with open(path) as f:
                    return f.read()
            except (OSError, IOError):
                pass
    return ""


def _parse_ttir_metrics(ttir_text: str) -> dict:
    """Parse structural metrics from TTIR text."""
    if not ttir_text:
        return {
            "dot_count": 0, "load_count": 0, "store_count": 0,
            "has_async_copy": False, "has_warp_specialize": False,
            "mma_type": "",
        }

    return {
        "dot_count": len(re.findall(r"tt\.dot\b", ttir_text)),
        "load_count": len(re.findall(r"tt\.load\b", ttir_text)),
        "store_count": len(re.findall(r"tt\.store\b", ttir_text)),
        "has_async_copy": "async_copy" in ttir_text or "cp.async" in ttir_text,
        "has_warp_specialize": "warp_specialize" in ttir_text,
        "mma_type": _detect_mma_type(ttir_text),
    }


def _parse_ttgir_layout(ttgir_text: str) -> dict:
    """Parse layout information from TTGIR text."""
    if not ttgir_text:
        return {}

    info = {}
    if "#ttg.blocked" in ttgir_text:
        info["ttgir_has_blocked_layout"] = True
    if "#ttg.nvidia_mma" in ttgir_text or "#ttg.amd_mfma" in ttgir_text:
        info["ttgir_has_mma_layout"] = True
    if "#ttg.shared" in ttgir_text:
        info["ttgir_has_shared_layout"] = True
    if "async_copy_global_to_local" in ttgir_text:
        info["has_async_copy"] = True
    if "warp_specialize" in ttgir_text:
        info["has_warp_specialize"] = True

    return info


def _detect_mma_type(ir_text: str) -> str:
    """Detect which MMA instruction type is used."""
    if "nvidia_mma" in ir_text or "wgmma" in ir_text.lower():
        return "wgmma" if "wgmma" in ir_text.lower() else "mma"
    if "amd_mfma" in ir_text:
        return "mfma"
    if "tt.dot" in ir_text:
        return "dot"
    return ""
