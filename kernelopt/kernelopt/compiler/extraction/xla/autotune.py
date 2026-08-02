"""Parse XLA autotuning results.

XLA's autotuner dumps results as text-proto (``AutotuneResults``) via::

    --xla_gpu_dump_autotune_results_to=<path>

or per-fusion cache files via::

    --xla_gpu_per_fusion_autotune_cache_dir=<dir>

The text-proto structure (xla/autotune_results.proto) is::

    AutotuneResults {
      version: int
      results {                         # repeated Entry
        device: str
        hlo: str                        # fusion HLO text
        result {                        # AutotuneResult
          key oneof:  conv | gemm | triton (TritonGemmKey) | ...
          run_time { seconds, nanos }   # google.protobuf.Duration
          failure { kind, msg }
        }
      }
    }

``TritonGemmKey`` contains: block_m, block_n, block_k, num_stages,
num_warps, num_ctas, etc.

References:
    - AutotuneResult proto: xla/autotuning.proto
    - AutotuneResults proto: xla/autotune_results.proto
    - AutotunerCache: xla/service/gpu/autotuning/autotuner_cache.h
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_INT_FIELDS = frozenset({
    "block_m", "block_n", "block_k", "split_k",
    "num_stages", "num_warps", "num_ctas",
    "waves_per_eu", "algorithm", "scratch_bytes",
    "kernel_index", "version", "buffer_address",
    "major_version", "minor_version", "patch_version",
    "autotune_workspace_size", "fusion_count",
})
_BOOL_FIELDS = frozenset({
    "tensor_ops_enabled", "is_tma_allowed",
    "is_warp_specialization_allowed",
})


def _parse_textproto_to_tree(content: str) -> dict[str, Any]:
    """Parse text-proto content into a nested dict/list tree.

    Handles repeated sub-messages by collecting them into lists.
    """
    root: dict[str, Any] = {}
    stack: list[dict[str, Any]] = [root]

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.endswith("{"):
            key = line[:-1].strip().rstrip(":")
            child: dict[str, Any] = {}
            parent = stack[-1]
            if key in parent:
                existing = parent[key]
                if isinstance(existing, list):
                    existing.append(child)
                else:
                    parent[key] = [existing, child]
            else:
                parent[key] = child
            stack.append(child)
        elif line == "}":
            if len(stack) > 1:
                stack.pop()
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().rstrip(";").strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif key in _INT_FIELDS:
                try:
                    val = int(val)
                except ValueError:
                    pass
            elif key in _BOOL_FIELDS:
                val = val.lower() in ("true", "1")
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            stack[-1][key] = val

    return root


def _flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one AutotuneResults.Entry into a uniform result dict."""
    result_msg = entry.get("result", {})
    if isinstance(result_msg, list):
        result_msg = result_msg[0] if result_msg else {}

    flat: dict[str, Any] = {
        "device": entry.get("device", ""),
        "hlo": entry.get("hlo", ""),
        "fusion_name": "",
        "block_m": 0, "block_n": 0, "block_k": 0,
        "num_stages": 0, "num_warps": 0, "num_ctas": 0,
        "algorithm_type": "",
        "run_time_us": 0.0,
        "scratch_bytes": 0,
        "failure": None,
    }

    hlo = flat["hlo"]
    if hlo:
        m = re.match(r"(\S+)", hlo)
        if m:
            flat["fusion_name"] = m.group(1).split(".")[-1]

    triton = result_msg.get("triton", {})
    if isinstance(triton, list):
        triton = triton[0] if triton else {}
    if triton:
        flat["algorithm_type"] = "triton_gemm"
        for fld in ("block_m", "block_n", "block_k", "num_stages",
                     "num_warps", "num_ctas"):
            flat[fld] = triton.get(fld, 0)

    if not triton:
        for alt_key, alt_name in [
            ("gemm", "gemm"), ("conv", "conv"),
            ("cuda_conv_plan", "cudnn"), ("algorithm", "algorithm"),
            ("custom_kernel_fusion", "custom_kernel"),
        ]:
            alt = result_msg.get(alt_key)
            if alt:
                flat["algorithm_type"] = alt_name
                break

    run_time = result_msg.get("run_time", {})
    if isinstance(run_time, list):
        run_time = run_time[0] if run_time else {}
    if isinstance(run_time, dict):
        secs = run_time.get("seconds", 0)
        nanos = run_time.get("nanos", 0)
        flat["run_time_us"] = (int(secs) * 1_000_000) + (int(nanos) / 1000.0)

    flat["scratch_bytes"] = result_msg.get("scratch_bytes", 0)

    failure = result_msg.get("failure")
    if failure:
        if isinstance(failure, list):
            failure = failure[0]
        flat["failure"] = {
            "kind": failure.get("kind", "UNKNOWN"),
            "msg": failure.get("msg", ""),
        }

    return flat


def parse_autotune_results(results_path: str) -> list[dict]:
    """Parse XLA autotune results from a dump file.

    Supports text-proto format (.textproto / .txt / .pbtxt) matching
    the ``AutotuneResults`` proto in ``xla/autotune_results.proto``.

    Args:
        results_path: Path to the autotune results file.

    Returns:
        List of dicts with keys: device, hlo, fusion_name, block_m,
        block_n, block_k, num_stages, num_warps, num_ctas,
        algorithm_type, run_time_us, scratch_bytes, failure.
    """
    path = Path(results_path)
    if not path.exists():
        return []

    try:
        content = path.read_text()
    except (OSError, IOError) as e:
        logger.debug("Failed to read autotune results %s: %s", path, e)
        return []

    if path.suffix in (".textproto", ".txt", ".pbtxt", ""):
        return _parse_text_proto(content)

    return []


def _parse_text_proto(content: str) -> list[dict]:
    """Parse text-proto autotune results (AutotuneResults or AutotuningLog)."""
    tree = _parse_textproto_to_tree(content)

    entries_raw = tree.get("results", [])
    if isinstance(entries_raw, dict):
        entries_raw = [entries_raw]

    if entries_raw and "result" in (entries_raw[0] if entries_raw else {}):
        return [_flatten_entry(e) for e in entries_raw]

    logs_raw = tree.get("logs", [])
    if isinstance(logs_raw, dict):
        logs_raw = [logs_raw]
    if logs_raw:
        return _parse_autotuning_logs(logs_raw)

    if any(k in tree for k in ("triton", "gemm", "conv", "run_time")):
        return [_flatten_entry({"result": tree})]

    return []


def _parse_autotuning_logs(logs: list[dict]) -> list[dict]:
    """Parse AutotuningLogs entries (--xla_gpu_dump_autotuning_log)."""
    out = []
    for log in logs:
        fusion_name = log.get("fusion_name", "")
        device_pci = log.get("device_pci_bus_id", "")
        results_raw = log.get("results", [])
        if isinstance(results_raw, dict):
            results_raw = [results_raw]
        for r in results_raw:
            flat = _flatten_entry({"result": r})
            flat["fusion_name"] = flat["fusion_name"] or fusion_name
            flat["device"] = flat["device"] or device_pci
            out.append(flat)
    return out


def parse_per_fusion_cache(cache_dir: str) -> list[dict]:
    """Parse per-fusion autotune cache directory.

    Each file in the directory contains autotune results for one fusion.

    Args:
        cache_dir: Path to the per-fusion cache directory.

    Returns:
        List of dicts (same schema as ``parse_autotune_results``).
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    results = []
    for f in sorted(cache_path.iterdir()):
        if not f.is_file():
            continue
        try:
            content = f.read_text()
            parsed = _parse_text_proto(content)
            for entry in parsed:
                entry["fusion_name"] = entry.get("fusion_name") or f.stem
            results.extend(parsed)
        except (OSError, IOError, ValueError) as e:
            logger.debug("Failed to read cache file %s: %s", f, e)
            continue

    return results


def setup_autotune_dump_flags(dump_path: str) -> None:
    """Set XLA flags for autotune result dumping.

    Must be called BEFORE importing JAX.
    """
    existing = os.environ.get("XLA_FLAGS", "")
    flag = f"--xla_gpu_dump_autotune_results_to={dump_path}"
    if flag not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()
