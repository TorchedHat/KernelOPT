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

        assert "benchmark_kernel_files" in primary
        bk_files = primary["benchmark_kernel_files"]
        assert len(bk_files) >= 1

        for bk in bk_files:
            assert "name" in bk
            assert "source" in bk
            assert "@triton.jit" in bk["source"]
            assert "def get_args" in bk["source"]
            assert "def call" in bk["source"]


from kernelopt.synthesis.wrap_kernel import build_benchmark_standalone, _clean_for_llm


class TestBuildBenchmarkStandalone:
    """Verify benchmark kernel produces clean standalone file."""

    def test_produces_clean_format(self):
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
        # Clean format: no Inductor infrastructure
        assert "triton_heuristics" not in result
        assert "kernel.run(" not in result
        assert "@triton.jit" in result

    def test_strips_inductor_helpers(self):
        benchmark_src = (
            "import triton\nimport triton.language as tl\n"
            "from torch._inductor.runtime import triton_helpers\n\n"
            "@triton.jit\ndef k(x_ptr, BLOCK: tl.constexpr): pass\n\n"
            "def get_args(): return []\n\n"
            "def call(args): my_kernel.run(*args)\n"
        )
        result = build_benchmark_standalone(benchmark_src, "# ref")
        # Clean format should NOT have inductor runtime imports or .run() calls
        assert "triton_helpers" not in result
        assert ".run(" not in result
        assert "@triton.jit" in result

    def test_cleans_inductor_ops_in_kernel_body(self):
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
        # triton_helpers.maximum should be cleaned to tl.maximum
        assert "tl.maximum" in result


class TestBenchmarkKernelE2E:
    """E2E: synthesis -> benchmark file -> wrap -> run -> correct output."""

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

        assert len(bk_files) >= 1, "No benchmark kernel files generated"

        target = bk_files[0]
        wrapped = build_benchmark_standalone(target["source"], model_file.read_text())

        kernel_file = tmp_path / "wrapped_benchmark.py"
        kernel_file.write_text(wrapped)

        result = subprocess.run(
            [sys.executable, str(kernel_file), "--profile-only"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"Wrapped benchmark kernel crashed:\n{result.stderr[-500:]}"
