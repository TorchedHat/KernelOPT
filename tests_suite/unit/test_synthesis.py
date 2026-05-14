"""Unit tests for the synthesis module (Feature 4).

No GPU or LLM required — tests regex/parsing logic only.
"""

import pytest
import re
import textwrap

# ---------------------------------------------------------------------------
# Import helpers directly (no torch dependency at import time)
# ---------------------------------------------------------------------------
from kernelopt.synthesis.compile_extract import (
    _detect_pytorch_pattern,
    _extract_function,
    _extract_triton_jit_blocks,
)
from kernelopt.synthesis.wrap_kernel import (
    _clean_triton_kernel,
    _replace_triton_helpers,
    _extract_constexpr_params,
    _build_get_inputs,
    _find_all_run_kernel_names,
    _find_matching_paren,
    _build_launch_replacement,
    _convert_run_to_launch,
)


# ===== compile_extract.py =====


class TestDetectPytorchPattern:
    def test_kernelbench_pattern(self):
        src = "class Model(nn.Module):\n    pass\ndef get_inputs():\n    pass"
        assert _detect_pytorch_pattern(src) == "kernelbench"

    def test_torch_nn_module_pattern(self):
        src = "class Foo(torch.nn.Module):\n    pass\ndef get_inputs():\n    pass"
        assert _detect_pytorch_pattern(src) == "kernelbench"

    def test_nn_module_without_get_inputs(self):
        src = "import torch\nclass Model(nn.Module):\n    pass"
        assert _detect_pytorch_pattern(src) == "kernelbench"

    def test_run_function_pattern(self):
        src = "import torch\n@torch.no_grad\ndef run():\n    pass"
        assert _detect_pytorch_pattern(src) == "run_function"

    def test_no_pattern(self):
        src = "import triton\n@triton.jit\ndef kernel(): pass"
        assert _detect_pytorch_pattern(src) == "none"


class TestExtractFunction:
    def test_single_line_signature(self):
        src = "def call(args):\n    x = args[0]\n    return x\n\ndef other(): pass"
        result = _extract_function(src, "call")
        assert "x = args[0]" in result
        assert "other" not in result

    def test_multi_line_signature(self):
        src = textwrap.dedent("""\
            def call(
                arg0_1, arg1_1,
                arg2_1,
            ):
                buf0 = alloc()
                return (buf0, )

            def next_func():
                pass
        """)
        result = _extract_function(src, "call")
        assert "buf0 = alloc()" in result
        assert "return (buf0, )" in result
        assert "next_func" not in result

    def test_function_not_found(self):
        assert _extract_function("def foo(): pass", "bar") == ""

    def test_return_type_annotation(self):
        src = "def call(args) -> tuple:\n    return (1,)\n\ndef other(): pass"
        result = _extract_function(src, "call")
        assert "return (1,)" in result
        assert "other" not in result


class TestExtractTritonJitBlocks:
    SAMPLE_INDUCTOR = textwrap.dedent("""\
        triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
        import triton
        @triton.jit
        def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            pass
        ''', device_str='cuda')

        triton_tem_fused_mm_0 = async_compile.triton_template('triton_tem_fused_mm_0', '''
        import triton
        @triton.jit
        def triton_tem_fused_mm_0(A, B, C, BLOCK_M: tl.constexpr):
            pass
        ''', device_str='cuda')
    """)

    def test_extracts_regular_kernel(self):
        blocks = _extract_triton_jit_blocks(self.SAMPLE_INDUCTOR)
        names = [b["name"] for b in blocks]
        assert "triton_poi_fused_add_0" in names

    def test_extracts_template_kernel(self):
        blocks = _extract_triton_jit_blocks(self.SAMPLE_INDUCTOR)
        templates = [b for b in blocks if b["is_template"]]
        assert len(templates) == 1
        assert templates[0]["name"] == "triton_tem_fused_mm_0"

    def test_returns_both_kernels(self):
        blocks = _extract_triton_jit_blocks(self.SAMPLE_INDUCTOR)
        assert len(blocks) == 2

    def test_no_duplicates(self):
        doubled = self.SAMPLE_INDUCTOR + self.SAMPLE_INDUCTOR
        blocks = _extract_triton_jit_blocks(doubled)
        names = [b["name"] for b in blocks]
        assert len(names) == len(set(names))

    def test_empty_source(self):
        assert _extract_triton_jit_blocks("") == []

    def test_no_triton_kernels(self):
        src = "def call(args):\n    extern_kernels.mm(a, b, out=c)\n"
        assert _extract_triton_jit_blocks(src) == []


# ===== wrap_kernel.py =====


class TestReplaceTritonHelpers:
    @pytest.mark.parametrize("inp,expected", [
        ("triton_helpers.promote_to_tensor(tl.sum(tmp, 0))", "tl.sum(tmp, 0)"),
        ("triton_helpers.maximum(a, b)", "tl.maximum(a, b)"),
        ("triton_helpers.minimum(a, b)", "tl.minimum(a, b)"),
        ("triton_helpers.device_assert(x < 100)", ""),
        ("libdevice.sqrt(tmp5)", "tl.sqrt(tmp5)"),
        ("libdevice.rsqrt(tmp5)", "tl.math.rsqrt(tmp5)"),
        ("libdevice.exp(tmp3)", "tl.exp(tmp3)"),
        ("libdevice.log(tmp2)", "tl.log(tmp2)"),
        ("libdevice.abs(tmp1)", "tl.abs(tmp1)"),
        ("libdevice.sin(tmp0)", "tl.sin(tmp0)"),
        ("libdevice.cos(tmp0)", "tl.cos(tmp0)"),
        ("libdevice.tanh(tmp4)", "tl.math.tanh(tmp4)"),
        ("libdevice.pow(tmp1, 0.5)", "tl.math.pow(tmp1, 0.5)"),
        ("tl_math.exp(tmp3)", "tl.exp(tmp3)"),
        ("tl_math.log(tmp2)", "tl.log(tmp2)"),
        ("tl_math.abs(tmp1)", "tl.abs(tmp1)"),
        ("tl_math.sqrt(tmp0)", "tl.sqrt(tmp0)"),
        ("tl_math.sin(tmp0)", "tl.sin(tmp0)"),
        ("tl_math.cos(tmp0)", "tl.cos(tmp0)"),
        ("tl_math.tanh(tmp0)", "tl.math.tanh(tmp0)"),
    ])
    def test_replacement(self, inp, expected):
        assert _replace_triton_helpers(inp).strip() == expected.strip()

    def test_no_false_positives(self):
        src = "tmp0 = tl.load(in_ptr0 + x0)"
        assert _replace_triton_helpers(src) == src


class TestExtractConstexprParams:
    def test_single_param(self):
        body = "def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):\n    pass"
        assert _extract_constexpr_params(body) == ["XBLOCK"]

    def test_multiple_params(self):
        body = "def kernel(in_ptr0, xnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):\n    pass"
        assert _extract_constexpr_params(body) == ["XBLOCK", "RBLOCK"]

    def test_multiline_signature(self):
        body = textwrap.dedent("""\
            def kernel(
                in_ptr0,
                out_ptr0,
                xnumel,
                XBLOCK: tl.constexpr,
                RBLOCK: tl.constexpr,
            ):
                pass
        """)
        params = _extract_constexpr_params(body)
        assert "XBLOCK" in params
        assert "RBLOCK" in params

    def test_no_constexpr(self):
        body = "def kernel(in_ptr0, out_ptr0, xnumel):\n    pass"
        assert _extract_constexpr_params(body) == []


class TestFindMatchingParen:
    def test_simple(self):
        s = "foo(a, b, c)"
        assert _find_matching_paren(s, 3) == 11

    def test_nested(self):
        s = "foo(a, grid=(1024,), stream=s)"
        assert _find_matching_paren(s, 3) == len(s) - 1

    def test_multiline(self):
        s = "foo(\n  a,\n  b\n)"
        assert _find_matching_paren(s, 3) == len(s) - 1

    def test_no_match(self):
        assert _find_matching_paren("foo(a, b", 3) == -1


class TestFindAllRunKernelNames:
    def test_single_kernel(self):
        src = "kernel_0.run(x, y, stream=s)"
        assert _find_all_run_kernel_names(src) == ["kernel_0"]

    def test_multiple_kernels(self):
        src = "kernel_0.run(x)\nkernel_1.run(y)\nkernel_0.run(z)"
        names = _find_all_run_kernel_names(src)
        assert names == ["kernel_0", "kernel_1"]

    def test_no_kernels(self):
        assert _find_all_run_kernel_names("x = 1") == []


class TestBuildGetInputs:
    def test_standard_format(self):
        bsrc = textwrap.dedent("""\
            def benchmark_compiled_module(times=10, repeat=10):
                from torch._dynamo.testing import rand_strided
                arg0_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
                arg1_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
                fn = lambda: call([arg0_1, arg1_1])
                return print_performance(fn, times=times, repeat=repeat)
        """)
        result = _build_get_inputs(bsrc)
        assert "def get_inputs():" in result
        assert "arg0_1" in result
        assert "arg1_1" in result
        assert "torch.randn" in result

    def test_double_quoted_device(self):
        bsrc = 'arg0_1 = rand_strided((64, 64), (64, 1), device="cuda:0", dtype=torch.float16)'
        result = _build_get_inputs(bsrc)
        assert "arg0_1" in result

    def test_unquoted_device(self):
        bsrc = "arg0_1 = rand_strided((64, 64), (64, 1), device=cuda, dtype=torch.float32)"
        result = _build_get_inputs(bsrc)
        assert "arg0_1" in result

    def test_empty_source(self):
        result = _build_get_inputs("")
        assert "return []" in result


class TestConvertRunToLaunch:
    def test_single_line_run(self):
        src = "    kernel_0.run(arg0, arg1, buf0, 1024, stream=stream0)"
        result = _convert_run_to_launch(src, "kernel_0", ["XBLOCK"])
        assert "kernel_0[grid]" in result
        assert ".run(" not in result
        assert "XBLOCK" in result

    def test_multiline_run(self):
        src = textwrap.dedent("""\
                kernel_0.run(
                    arg0, arg1, buf0,
                    1024,
                    stream=stream0
                )
        """)
        result = _convert_run_to_launch(src, "kernel_0", ["XBLOCK"])
        assert "kernel_0[grid]" in result
        assert ".run(" not in result

    def test_nested_parens_in_args(self):
        src = "    kernel_0.run(arg0, arg1, buf0, 1024, grid=(128,), stream=s)"
        result = _convert_run_to_launch(src, "kernel_0", ["XBLOCK"])
        assert "kernel_0[grid]" in result
        assert ".run(" not in result

    def test_multiple_kernels(self):
        src = textwrap.dedent("""\
            kernel_0.run(arg0, buf0, 64, stream=s)
            kernel_1.run(buf0, buf1, 64, stream=s)
        """)
        result = _convert_run_to_launch(src, "kernel_0", ["XBLOCK"])
        assert "kernel_0[grid]" in result
        assert "kernel_1[grid]" in result
        assert ".run(" not in result


class TestCleanTritonKernel:
    def test_strips_inductor_decorators(self):
        raw = textwrap.dedent("""\
            import triton
            import triton.language as tl
            from torch._inductor.runtime import triton_helpers

            @triton_heuristics.pointwise(size_hints={'x': 1024})
            @triton.jit
            def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
                xoffset = tl.program_id(0) * XBLOCK
                xindex = xoffset + tl.arange(0, XBLOCK)[:]
                tl.store(out_ptr0 + xindex, tl.load(in_ptr0 + xindex))
        """)
        name, body, params = _clean_triton_kernel(raw)
        assert name == "triton_poi_fused_add_0"
        assert "@triton.jit" in body
        assert "@triton_heuristics" not in body
        assert "import triton" not in body
        assert "in_ptr0" in params
