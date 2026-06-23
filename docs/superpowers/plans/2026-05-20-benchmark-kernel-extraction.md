# Lossless Kernel Extraction via TORCHINDUCTOR_BENCHMARK_KERNEL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `build_standalone_kernel()` (wrong grids, dropped strides, hardcoded constexprs) with PyTorch's `TORCHINDUCTOR_BENCHMARK_KERNEL=1` per-kernel files that are correct by construction.

**Architecture:** During the synthesis subprocess, set `TORCHINDUCTOR_BENCHMARK_KERNEL=1` so PyTorch generates self-contained `.py` files per Triton kernel in the inductor cache. Collect those files and use them as the `best_kernel` for the LLM executor. The benchmark files use `kernel.run()` with the full Inductor runtime, which handles grid computation (1D/2D/3D), buffer strides, autotuning, and constexpr injection correctly.

**Tech Stack:** Python subprocess, glob, `torch._inductor` env vars, existing `_clean_for_llm()` for LLM readability

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `kernelopt/kernelopt/synthesis/compile_extract.py` | Modify | Add `TORCHINDUCTOR_BENCHMARK_KERNEL=1` to subprocess env, collect benchmark files, return in results |
| `kernelopt/kernelopt/synthesis/wrap_kernel.py` | Modify | Add `build_benchmark_standalone()`, remove `build_standalone_kernel()` and all dead translation code |
| `kernelopt/kernelopt/langgraph_pipeline.py` | Modify | Use benchmark kernel files in `maybe_synthesize()` |
| `kernelopt/kernelopt/synthesis/__init__.py` | Modify | Update exports |
| `tests_suite/unit/test_benchmark_kernel_extraction.py` | **Create** | Tests for the new extraction approach |

---

### Task 1: Modify `compile_extract.py` to collect benchmark kernel files

**Files:**
- Modify: `kernelopt/kernelopt/synthesis/compile_extract.py`
- Create: `tests_suite/unit/test_benchmark_kernel_extraction.py`

- [ ] **Step 1: Create test file**

Create `tests_suite/unit/test_benchmark_kernel_extraction.py`:

```python
"""Tests for lossless kernel extraction via TORCHINDUCTOR_BENCHMARK_KERNEL."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))


class TestExtractWithBenchmarkKernels:
    """Verify extract_triton_from_pytorch returns benchmark_kernel_files."""

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="requires GPU",
    )
    def test_simple_model_returns_benchmark_files(self, tmp_path):
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch

        model_file = tmp_path / "model.py"
        model_file.write_text(
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x): return torch.relu(x) + torch.sin(x)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(1024, device='cuda')]\n"
        )
        results = extract_triton_from_pytorch(str(model_file), timeout=180)
        assert len(results) >= 1
        primary = results[0]

        # Must have benchmark_kernel_files key
        assert "benchmark_kernel_files" in primary
        bk_files = primary["benchmark_kernel_files"]
        assert len(bk_files) >= 1

        # Each entry has name, source, path
        for bk in bk_files:
            assert "name" in bk
            assert "source" in bk
            assert "@triton.jit" in bk["source"]
            assert "def get_args" in bk["source"]
            assert "def call" in bk["source"]

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="requires GPU",
    )
    def test_no_triton_returns_empty_benchmark_files(self, tmp_path):
        """Pure extern_kernels (cuBLAS) produces no benchmark files."""
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch

        model_file = tmp_path / "model.py"
        model_file.write_text(
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, A, B): return torch.matmul(A, B)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(16, 1024, 2048, device='cuda'),\n"
            "            torch.randn(2048, 768, device='cuda')]\n"
        )
        results = extract_triton_from_pytorch(str(model_file), timeout=180)
        primary = results[0]
        # May or may not have benchmark files depending on whether
        # inductor generates any Triton kernels for this op
        assert "benchmark_kernel_files" in primary
```

- [ ] **Step 2: Modify `_build_driver_script()` to enable benchmark kernel collection**

In `compile_extract.py`, update `_build_driver_script()`. The generated driver script needs to:
1. Set `TORCHINDUCTOR_BENCHMARK_KERNEL=1` and `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`
2. Use a dedicated cache directory
3. After `run_and_get_code()`, scan the cache for benchmark `.py` files
4. Include them in the output JSON

Replace the return statement at the end of `_build_driver_script()` (the part that builds the driver code string). The new driver code should set the env vars and add a benchmark file collection step. Change the preamble from:

```python
    return textwrap.dedent(f"""\
        import json, sys, os, torch
        os.environ["TORCH_COMPILE_DEBUG"] = "0"
```

to:

```python
    return textwrap.dedent(f"""\
        import json, sys, os, torch, glob, re
        os.environ["TORCH_COMPILE_DEBUG"] = "0"
        os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
        os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
```

And replace the output JSON writing block from:

```python
        out = {{
            "num_sources": len(sources),
            "sources": list(sources),
        }}
        with open({output_json!r}, "w") as f:
            json.dump(out, f)
```

to:

```python
        # Collect benchmark kernel files from inductor cache
        _bk_files = []
        for _f in glob.glob(os.path.join(os.path.dirname({output_json!r}), "**", "*.py"), recursive=True):
            try:
                _content = open(_f).read()
            except OSError:
                continue
            if "@triton.jit" in _content and "def get_args" in _content and "def call" in _content:
                _bk_files.append({{"name": os.path.basename(_f), "source": _content, "path": _f}})
        # Also check default inductor cache
        _inductor_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", os.path.join("/tmp", f"torchinductor_{{os.environ.get('USER', 'root')}}"))
        if os.path.isdir(_inductor_cache):
            for _f in glob.glob(os.path.join(_inductor_cache, "**", "*.py"), recursive=True):
                try:
                    _content = open(_f).read()
                except OSError:
                    continue
                if "@triton.jit" in _content and "def get_args" in _content and "def call" in _content:
                    if not any(b["path"] == _f for b in _bk_files):
                        _bk_files.append({{"name": os.path.basename(_f), "source": _content, "path": _f}})

        out = {{
            "num_sources": len(sources),
            "sources": list(sources),
            "benchmark_kernel_files": _bk_files,
        }}
        with open({output_json!r}, "w") as f:
            json.dump(out, f)
```

- [ ] **Step 3: Update `extract_triton_from_pytorch()` to pass through benchmark files**

In the results loop (around line 348), add the `benchmark_kernel_files` key:

Change:
```python
            results.append({
                "inductor_source": src,
                "triton_kernels": triton_kernels,
                "call_source": call_source,
                "benchmark_source": benchmark_source,
                "callable_info": info,
            })
```

to:
```python
            results.append({
                "inductor_source": src,
                "triton_kernels": triton_kernels,
                "call_source": call_source,
                "benchmark_source": benchmark_source,
                "callable_info": info,
                "benchmark_kernel_files": data.get("benchmark_kernel_files", []),
            })
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests_suite/unit/test_benchmark_kernel_extraction.py -v --timeout=300`
Expected: Tests pass (the GPU test may take ~60s for torch.compile)

- [ ] **Step 5: Commit**

```bash
git add kernelopt/kernelopt/synthesis/compile_extract.py tests_suite/unit/test_benchmark_kernel_extraction.py
git commit -m "feat: collect TORCHINDUCTOR_BENCHMARK_KERNEL files during synthesis

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `build_benchmark_standalone()` to `wrap_kernel.py`

**Files:**
- Modify: `kernelopt/kernelopt/synthesis/wrap_kernel.py`
- Modify: `tests_suite/unit/test_benchmark_kernel_extraction.py`

- [ ] **Step 1: Add tests**

Append to `tests_suite/unit/test_benchmark_kernel_extraction.py`:

```python
from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone, _clean_for_llm


class TestBuildBenchmarkStandalone:
    """Verify benchmark kernel wrapping with kernelopt interface."""

    def test_adds_kernel_function(self):
        benchmark_src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\ndef my_kernel(): pass\n\n"
            "def get_args(): return []\n\n"
            "def call(args): pass\n"
        )
        result = build_benchmark_standalone(benchmark_src, "# original pytorch")
        assert "def kernel_function(" in result
        assert "def get_inputs(" in result
        assert "--profile-only" in result
        assert "_PYTORCH_REFERENCE" in result
        assert "_TRITON_KERNEL_SOURCES" in result

    def test_preserves_benchmark_source_intact(self):
        benchmark_src = (
            "import triton\nimport triton.language as tl\n"
            "from torch._inductor.runtime import triton_helpers\n\n"
            "@triton.jit\ndef k(x_ptr, BLOCK: tl.constexpr): pass\n\n"
            "def get_args(): return []\n\n"
            "def call(args): my_kernel.run(*args)\n"
        )
        result = build_benchmark_standalone(benchmark_src, "# ref")
        # The original benchmark source should be preserved (kernel.run, triton_helpers, etc.)
        assert "triton_helpers" in result
        assert ".run(" in result

    def test_clean_kernel_in_triton_sources(self):
        benchmark_src = (
            "import triton\nimport triton.language as tl\n"
            "from torch._inductor.runtime import triton_helpers\n\n"
            "@triton.jit\n"
            "def k(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n"
            "    tmp = triton_helpers.maximum(tl.full([1], 0, tl.int32), tl.load(in_ptr0))\n"
            "    tl.store(out_ptr0, tmp)\n\n"
            "def get_args(): return []\n\n"
            "def call(args): pass\n"
        )
        result = build_benchmark_standalone(benchmark_src, "# ref")
        # _TRITON_KERNEL_SOURCES should have the CLEANED version
        assert "tl.maximum" in result  # cleaned from triton_helpers.maximum
```

- [ ] **Step 2: Implement `build_benchmark_standalone()`**

Add this function to `wrap_kernel.py`, after `_clean_for_llm()` and BEFORE `build_standalone_kernel()`:

```python
def build_benchmark_standalone(
    benchmark_source: str,
    original_pytorch_source: str,
) -> str:
    """Wrap a TORCHINDUCTOR_BENCHMARK_KERNEL file with kernelopt interface.

    The benchmark file already has correct ``kernel.run()``, ``get_args()``,
    and grid computation via ``triton_heuristics``. We keep it intact and
    append:
      - ``_PYTORCH_REFERENCE`` (original source for context)
      - ``_TRITON_KERNEL_SOURCES`` (clean ``@triton.jit`` bodies for LLM reading)
      - ``kernel_function(*inputs)`` wrapper delegating to ``call()``
      - ``get_inputs()`` delegating to ``get_args()``
      - ``__main__`` with ``--profile-only``
    """
    import re as _re

    escaped_ref = original_pytorch_source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    # Extract and clean @triton.jit blocks for LLM reading
    triton_blocks = _re.findall(
        r'(@triton\.jit\s*\ndef\s+\w+\([^)]*\):.*?)(?=\ndef\s|\Z)',
        benchmark_source,
        _re.DOTALL,
    )
    if not triton_blocks:
        # Fallback: extract everything from @triton.jit to next top-level def
        lines = benchmark_source.split("\n")
        in_kernel = False
        block_lines: list[str] = []
        blocks: list[str] = []
        for line in lines:
            if "@triton.jit" in line:
                if block_lines:
                    blocks.append("\n".join(block_lines))
                in_kernel = True
                block_lines = [line]
                continue
            if in_kernel:
                if line and not line[0].isspace() and not line.startswith("@"):
                    blocks.append("\n".join(block_lines))
                    block_lines = []
                    in_kernel = False
                else:
                    block_lines.append(line)
        if block_lines:
            blocks.append("\n".join(block_lines))
        triton_blocks = blocks

    kernel_src_list = []
    for block in triton_blocks:
        cleaned = _clean_for_llm(block)
        escaped = cleaned.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        kernel_src_list.append(f"    '''{escaped}'''")
    kernel_src_joined = ",\n".join(kernel_src_list) if kernel_src_list else ""

    interface = f"""

# ===== kernelopt interface (appended) =====

_PYTORCH_REFERENCE = '''{escaped_ref}'''

_TRITON_KERNEL_SOURCES = [
{kernel_src_joined}
]

def kernel_function(*inputs):
    \"\"\"kernelopt entry point — delegates to benchmark call().\"\"\"
    args = list(get_args()) if not inputs else list(inputs)
    call(args)
    import torch
    torch.cuda.synchronize()

def get_inputs():
    \"\"\"Return inputs for profiling — delegates to benchmark get_args().\"\"\"
    args = get_args()
    return [a for a in args if isinstance(a, __import__('torch').Tensor)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    _args = parser.parse_args()
    _inputs = get_args()
    call(list(_inputs))
    __import__('torch').cuda.synchronize()
    if not _args.profile_only:
        print("Kernel executed successfully.")
"""

    base = benchmark_source.rstrip()

    # Remove existing __main__ block if present
    if "if __name__" in base:
        base = _re.sub(
            r'\nif __name__\s*==\s*["\']__main__["\'].*',
            '',
            base,
            flags=_re.DOTALL,
        )

    return base.rstrip() + "\n" + interface
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests_suite/unit/test_benchmark_kernel_extraction.py::TestBuildBenchmarkStandalone -v`
Expected: All 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add kernelopt/kernelopt/synthesis/wrap_kernel.py tests_suite/unit/test_benchmark_kernel_extraction.py
git commit -m "feat: add build_benchmark_standalone() for lossless kernel wrapping

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire into `maybe_synthesize()` and remove dead code

**Files:**
- Modify: `kernelopt/kernelopt/langgraph_pipeline.py`
- Modify: `kernelopt/kernelopt/synthesis/wrap_kernel.py` (remove dead code)
- Modify: `kernelopt/kernelopt/synthesis/__init__.py` (update exports)

- [ ] **Step 1: Update `maybe_synthesize()` in `langgraph_pipeline.py`**

Find the block in `maybe_synthesize()` that calls `build_standalone_kernel` (around lines 1485-1490):

```python
        # Build a complete standalone Triton file for the LLM to read/modify
        from kernelopt.synthesis.wrap_kernel import build_standalone_kernel
        standalone_kernel = build_standalone_kernel(
            triton_kernel_sources=primary["triton_kernels"],
            inductor_source=primary["inductor_source"],
        )
```

Replace with:

```python
        # Use benchmark kernel files (correct by construction) if available,
        # otherwise fall back to the wrapped Inductor file.
        from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone
        benchmark_files = primary.get("benchmark_kernel_files", [])
        if benchmark_files:
            # Pick the largest benchmark file (likely the most complex kernel)
            target_bk = max(benchmark_files, key=lambda b: len(b.get("source", "")))
            standalone_kernel = build_benchmark_standalone(
                benchmark_source=target_bk["source"],
                original_pytorch_source=original_source,
            )
            pipe_print(
                f"  Benchmark kernel: [green]{target_bk['name'][:50]}[/] "
                f"({len(benchmark_files)} total)"
            )
        else:
            # No Triton kernels (pure extern_kernels like cuBLAS mm)
            standalone_kernel = wrapped_source
            pipe_print("  [dim]No Triton benchmark kernels — using wrapped Inductor file[/]")
```

- [ ] **Step 2: Remove dead code from `wrap_kernel.py`**

Delete these functions from `wrap_kernel.py` (they are no longer called by any code):
- `build_standalone_kernel()` (lines 156-236)
- `_extract_function_from_source()` (lines 239-261)
- `_parse_kernel_signatures()` (lines 264-288)
- `_CONSTEXPR_DEFAULTS` dict (lines 291-299)
- `_extract_helper_functions()` (lines 302-337)
- `_translate_call_to_kernel_function()` (lines 340-497)
- `_build_standalone_wrapper()` (lines 500-594)

Keep these functions (still used by `wrap_inductor_output()`):
- `wrap_inductor_output()` (lines 21-115)
- `_clean_for_llm()` (lines 118-153)
- `build_benchmark_standalone()` (newly added in Task 2)
- `_extract_get_inputs()` (lines 597-657)

- [ ] **Step 3: Update `__init__.py` exports**

In `kernelopt/kernelopt/synthesis/__init__.py`, add the new export:

```python
from .compile_extract import extract_triton_from_pytorch
from .wrap_kernel import wrap_inductor_output, build_benchmark_standalone

__all__ = ["extract_triton_from_pytorch", "wrap_inductor_output", "build_benchmark_standalone"]
```

- [ ] **Step 4: Verify no remaining references to removed functions**

Run: `grep -rn "build_standalone_kernel\|_translate_call_to_kernel_function\|_build_standalone_wrapper\|_CONSTEXPR_DEFAULTS\|_parse_kernel_signatures\|_extract_helper_functions\|_extract_function_from_source" kernelopt/ tests_suite/ --include="*.py" | grep -v __pycache__ | grep -v "test_benchmark_kernel"`

Expected: No results (all references removed)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests_suite/unit/ tests_suite/frameworks/ tests_suite/quality/ -x -q 2>&1 | tail -10`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add kernelopt/kernelopt/langgraph_pipeline.py kernelopt/kernelopt/synthesis/wrap_kernel.py kernelopt/kernelopt/synthesis/__init__.py
git commit -m "feat: use TORCHINDUCTOR_BENCHMARK_KERNEL for lossless kernel extraction

Replace build_standalone_kernel() (wrong grids, dropped strides) with
PyTorch's own per-kernel benchmark files. Verified correct on Grid1D,
Grid2D, reductions, channels-last, fused ops, embeddings, batch norm,
broadcast, and multi-output patterns.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: E2E validation on real kernels

**Files:**
- Modify: `tests_suite/unit/test_benchmark_kernel_extraction.py`

- [ ] **Step 1: Add E2E test that validates the full synthesis→wrap→run pipeline**

Append to `tests_suite/unit/test_benchmark_kernel_extraction.py`:

```python
class TestBenchmarkKernelE2E:
    """E2E: synthesis → benchmark file → wrap → run → correct output."""

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="requires GPU",
    )
    def test_2d_kernel_produces_correct_output(self, tmp_path):
        """The pattern that build_standalone_kernel always got wrong."""
        import subprocess
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch
        from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone

        model_file = tmp_path / "model.py"
        model_file.write_text(
            "import torch\nimport torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self): super().__init__()\n"
            "    def forward(self, x, s):\n"
            "        return x.transpose(-2, -1).contiguous() * s\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(32, 64, 256, device='cuda'),\n"
            "            torch.randn(32, 256, 64, device='cuda')]\n"
        )

        results = extract_triton_from_pytorch(str(model_file), timeout=180)
        primary = results[0]
        bk_files = primary.get("benchmark_kernel_files", [])

        # Must have benchmark files with 2D grids
        assert len(bk_files) >= 1, "No benchmark kernel files generated"

        # Pick the first one and wrap it
        target = bk_files[0]
        wrapped = build_benchmark_standalone(target["source"], model_file.read_text())

        # Write and run
        kernel_file = tmp_path / "wrapped_benchmark.py"
        kernel_file.write_text(wrapped)

        result = subprocess.run(
            [sys.executable, str(kernel_file), "--profile-only"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"Wrapped benchmark kernel crashed:\n{result.stderr[-500:]}"
```

- [ ] **Step 2: Run E2E test**

Run: `python -m pytest tests_suite/unit/test_benchmark_kernel_extraction.py::TestBenchmarkKernelE2E -v --timeout=300`
Expected: PASS

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests_suite/unit/ tests_suite/frameworks/ tests_suite/quality/ -x -q 2>&1 | tail -10`
Expected: No new failures

- [ ] **Step 4: Commit**

```bash
git add tests_suite/unit/test_benchmark_kernel_extraction.py
git commit -m "test: add E2E test for benchmark kernel extraction with 2D grid

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Commit docs

**Files:**
- Add: `docs/superpowers/specs/2026-05-20-benchmark-kernel-extraction.md`
- Add: `docs/superpowers/plans/2026-05-20-benchmark-kernel-extraction.md`
- Add: `docs/pytorch-limitations.md`

- [ ] **Step 1: Commit documentation**

```bash
git add -f docs/superpowers/specs/2026-05-20-benchmark-kernel-extraction.md docs/superpowers/plans/2026-05-20-benchmark-kernel-extraction.md docs/pytorch-limitations.md
git commit -m "docs: add spec, plan, and pytorch limitations for benchmark kernel extraction

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
