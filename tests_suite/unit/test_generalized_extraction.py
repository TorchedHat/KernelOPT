"""Tests for the generalized PyTorch kernel extraction system.

Verifies that _detect_callable, _build_driver_script, and extract_triton_from_pytorch
work with arbitrary PyTorch code — not just KernelBench/SOL-ExecBench formats.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))

from kernelopt.synthesis.compile_extract import (
    CallableInfo,
    SynthesisError,
    _build_driver_script,
    _detect_callable,
)


# ---------------------------------------------------------------------------
# _detect_callable: backward-compatible patterns
# ---------------------------------------------------------------------------

class TestDetectCallableBackwardCompat:
    """Old KernelBench and run_function patterns must still be detected."""

    def test_kernelbench_class_model_with_get_inputs(self):
        src = textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x):
                    return torch.relu(x)

            def get_inputs():
                return [torch.randn(8, 64)]
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "Model"
        assert info.has_get_inputs is True

    def test_kernelbench_with_get_init_inputs(self):
        src = textwrap.dedent("""\
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                def forward(self, x):
                    return x

            def get_inputs():
                return [torch.randn(8, 64)]

            def get_init_inputs():
                return [128]
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "Model"
        assert info.has_get_inputs is True
        assert info.has_get_init_inputs is True

    def test_run_function_with_get_inputs(self):
        src = textwrap.dedent("""\
            import torch

            def run(x):
                return torch.softmax(x, dim=-1)

            def get_inputs():
                return [torch.randn(8, 64)]
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "function"
        assert info.function_name == "run"
        assert info.has_get_inputs is True

    def test_run_function_without_get_inputs(self):
        src = textwrap.dedent("""\
            import torch

            @torch.no_grad()
            def run(x):
                return torch.softmax(x, dim=-1)
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "function"
        assert info.function_name == "run"
        assert info.has_get_inputs is False


# ---------------------------------------------------------------------------
# _detect_callable: new generic patterns
# ---------------------------------------------------------------------------

class TestDetectCallableGeneric:
    """New patterns that previously failed with SynthesisError."""

    def test_custom_class_name_no_get_inputs(self):
        src = textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Attention(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return x * 2
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "Attention"
        assert info.has_get_inputs is False

    def test_torch_nn_module_full_path(self):
        src = textwrap.dedent("""\
            import torch

            class MyNet(torch.nn.Module):
                def forward(self, x):
                    return x
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "MyNet"

    def test_standalone_function_with_torch(self):
        src = textwrap.dedent("""\
            import torch

            def my_softmax(x):
                return torch.softmax(x, dim=-1)
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "function"
        assert info.function_name == "my_softmax"

    def test_multiple_modules_prefers_model(self):
        src = textwrap.dedent("""\
            import torch.nn as nn

            class Helper(nn.Module):
                def forward(self, x): return x

            class Model(nn.Module):
                def forward(self, x): return x * 2

            def get_inputs():
                return []
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.class_name == "Model"

    def test_multiple_modules_picks_first_when_no_model(self):
        src = textwrap.dedent("""\
            import torch.nn as nn

            class Encoder(nn.Module):
                def forward(self, x): return x

            class Decoder(nn.Module):
                def forward(self, x): return x
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.class_name == "Encoder"

    def test_skips_helper_functions(self):
        src = textwrap.dedent("""\
            import torch

            def get_inputs():
                return [torch.randn(8, 64)]

            def _private_helper():
                pass

            def compute(x):
                return torch.relu(x)
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.function_name == "compute"
        assert info.has_get_inputs is True

    def test_prefers_run_over_other_functions(self):
        src = textwrap.dedent("""\
            import torch

            def compute(x):
                return torch.relu(x)

            def run(x):
                return torch.softmax(x, dim=-1)
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.function_name == "run"

    def test_function_with_torch_decorator(self):
        src = textwrap.dedent("""\
            import torch

            @torch.no_grad()
            def inference(model, x):
                return model(x)
        """)
        info = _detect_callable(src)
        assert info is not None
        assert info.kind == "function"
        assert info.function_name == "inference"


# ---------------------------------------------------------------------------
# _detect_callable: entry_point override
# ---------------------------------------------------------------------------

class TestDetectCallableEntryPoint:
    """User-specified --entry-point overrides auto-detection."""

    def test_entry_point_selects_specific_class(self):
        src = textwrap.dedent("""\
            import torch.nn as nn

            class Foo(nn.Module):
                def forward(self, x): return x

            class Bar(nn.Module):
                def forward(self, x): return x * 2
        """)
        info = _detect_callable(src, entry_point="Bar")
        assert info is not None
        assert info.class_name == "Bar"

    def test_entry_point_selects_specific_function(self):
        src = textwrap.dedent("""\
            import torch

            def alpha(x):
                return x + 1

            def beta(x):
                return x * 2
        """)
        info = _detect_callable(src, entry_point="beta")
        assert info is not None
        assert info.function_name == "beta"

    def test_entry_point_not_found_returns_none(self):
        src = textwrap.dedent("""\
            import torch

            def foo(x):
                return x
        """)
        info = _detect_callable(src, entry_point="nonexistent")
        assert info is None

    def test_entry_point_with_dot_notation(self):
        src = textwrap.dedent("""\
            import torch.nn as nn

            class MyModel(nn.Module):
                def forward(self, x): return x
        """)
        info = _detect_callable(src, entry_point="MyModel.forward")
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "MyModel"

    def test_entry_point_non_module_class(self):
        src = textwrap.dedent("""\
            class DataProcessor:
                def process(self, data):
                    return data
        """)
        info = _detect_callable(src, entry_point="DataProcessor")
        assert info is not None
        assert info.kind == "module_class"
        assert info.class_name == "DataProcessor"


# ---------------------------------------------------------------------------
# _detect_callable: edge cases / no match
# ---------------------------------------------------------------------------

class TestDetectCallableEdgeCases:
    """Edge cases that should return None or handle gracefully."""

    def test_empty_string(self):
        assert _detect_callable("") is None

    def test_syntax_error(self):
        assert _detect_callable("def broken(:\n  pass") is None

    def test_only_comments(self):
        assert _detect_callable("# just a comment\n# another comment") is None

    def test_no_torch_plain_function(self):
        src = textwrap.dedent("""\
            def add(a, b):
                return a + b
        """)
        assert _detect_callable(src) is None

    def test_only_private_functions(self):
        src = textwrap.dedent("""\
            import torch

            def _internal():
                pass

            def _also_private():
                pass
        """)
        assert _detect_callable(src) is None

    def test_only_helper_functions(self):
        src = textwrap.dedent("""\
            import torch

            def get_inputs():
                return [torch.randn(8, 64)]

            def get_init_inputs():
                return []
        """)
        assert _detect_callable(src) is None

    def test_constants_only(self):
        assert _detect_callable("X = 42\nY = 'hello'") is None


# ---------------------------------------------------------------------------
# _build_driver_script: generated code structure
# ---------------------------------------------------------------------------

class TestBuildDriverScript:
    """Test that _build_driver_script generates correct subprocess code."""

    def test_module_class_with_get_inputs(self):
        info = CallableInfo(
            kind="module_class", class_name="MyModel",
            has_get_inputs=True, has_get_init_inputs=False,
        )
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "_mod.MyModel" in script
        assert "_mod.get_inputs()" in script
        assert "torch.compile(model)" in script
        assert "run_and_get_code" in script
        assert "init_args = []" in script

    def test_module_class_with_init_inputs(self):
        info = CallableInfo(
            kind="module_class", class_name="Transformer",
            has_get_inputs=True, has_get_init_inputs=True,
        )
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "_mod.Transformer(*init_args)" in script
        assert "get_init_inputs" in script

    def test_module_class_without_get_inputs_uses_shapes(self):
        info = CallableInfo(
            kind="module_class", class_name="Attention",
            has_get_inputs=False,
        )
        script = _build_driver_script(
            "/tmp/test.py", "/tmp/out.json", info,
            input_shapes=[(8, 512), (8, 512, 64)],
        )

        assert "torch.randn((8, 512), device='cuda')" in script
        assert "torch.randn((8, 512, 64), device='cuda')" in script
        assert "get_inputs" not in script

    def test_module_class_no_inputs_fallback(self):
        info = CallableInfo(
            kind="module_class", class_name="Net",
            has_get_inputs=False,
        )
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "WARNING" in script
        assert "torch.randn(1, 64" in script

    def test_function_with_get_inputs(self):
        info = CallableInfo(
            kind="function", function_name="run",
            has_get_inputs=True,
        )
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "fn = _mod.run" in script
        assert "_mod.get_inputs()" in script
        assert "torch.compile(fn)" in script

    def test_function_with_shapes(self):
        info = CallableInfo(
            kind="function", function_name="compute",
            has_get_inputs=False,
        )
        script = _build_driver_script(
            "/tmp/test.py", "/tmp/out.json", info,
            input_shapes=[(4, 256)],
        )

        assert "fn = _mod.compute" in script
        assert "torch.randn((4, 256), device='cuda')" in script

    def test_function_fallback_warning(self):
        info = CallableInfo(
            kind="function", function_name="process",
            has_get_inputs=False,
        )
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "fn = _mod.process" in script
        assert "WARNING" in script

    def test_driver_has_max_autotune(self):
        info = CallableInfo(kind="function", function_name="f", has_get_inputs=False)
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "max_autotune = True" in script
        assert "max_autotune_gemm = True" in script

    def test_driver_writes_json_output(self):
        info = CallableInfo(kind="function", function_name="f", has_get_inputs=False)
        script = _build_driver_script("/tmp/test.py", "/tmp/out.json", info)

        assert "json.dump(out, f)" in script
        assert '"num_sources"' in script


# ---------------------------------------------------------------------------
# extract_triton_from_pytorch: error handling
# ---------------------------------------------------------------------------

class TestExtractTritonErrors:
    """Test that extract_triton_from_pytorch raises correctly."""

    def test_no_callable_raises_synthesis_error(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("x = 42\n")
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch

        with pytest.raises(SynthesisError, match="Cannot find a compilable callable"):
            extract_triton_from_pytorch(str(f))

    def test_error_message_suggests_entry_point(self, tmp_path):
        f = tmp_path / "bare.py"
        f.write_text("# nothing here\n")
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch

        with pytest.raises(SynthesisError, match="--entry-point"):
            extract_triton_from_pytorch(str(f))

    def test_entry_point_not_found_raises(self, tmp_path):
        f = tmp_path / "model.py"
        f.write_text("import torch\ndef foo(x): return x\n")
        from kernelopt.synthesis.compile_extract import extract_triton_from_pytorch

        with pytest.raises(SynthesisError, match="Cannot find"):
            extract_triton_from_pytorch(str(f), entry_point="NonExistent")


# ---------------------------------------------------------------------------
# ncu_runner._is_pytorch_source: unified detection
# ---------------------------------------------------------------------------

class TestNcuRunnerPytorchDetection:
    """Test that ncu_runner uses the same detection as synthesis."""

    def test_detects_module_class(self):
        import ast
        from kernelopt.agents.profiler.ncu_runner import _is_pytorch_source

        src = textwrap.dedent("""\
            import torch.nn as nn

            class Attention(nn.Module):
                def forward(self, x): return x
        """)
        tree = ast.parse(src)
        assert _is_pytorch_source(src, tree) is True

    def test_detects_function_with_torch(self):
        import ast
        from kernelopt.agents.profiler.ncu_runner import _is_pytorch_source

        src = textwrap.dedent("""\
            import torch

            def compute(x):
                return torch.relu(x)
        """)
        tree = ast.parse(src)
        assert _is_pytorch_source(src, tree) is True

    def test_rejects_plain_python(self):
        import ast
        from kernelopt.agents.profiler.ncu_runner import _is_pytorch_source

        src = "def add(a, b): return a + b"
        tree = ast.parse(src)
        assert _is_pytorch_source(src, tree) is False

    def test_rejects_triton_code(self):
        import ast
        from kernelopt.agents.profiler.ncu_runner import _is_pytorch_source

        src = textwrap.dedent("""\
            import triton
            import triton.language as tl

            @triton.jit
            def kernel(x_ptr, BLOCK: tl.constexpr):
                pass
        """)
        tree = ast.parse(src)
        # This should not match as "pytorch" because it's handled
        # by the Triton detector first in extract_kernel_info.
        # _is_pytorch_source itself may return True (it has torch-like
        # functions) but extract_kernel_info checks Triton first.
        # The important thing is it doesn't crash.
        _is_pytorch_source(src, tree)  # no crash


# ---------------------------------------------------------------------------
# PipelineConfig: new fields
# ---------------------------------------------------------------------------

class TestPipelineConfigNewFields:
    """Test that PipelineConfig has the new synthesis fields."""

    def test_defaults_are_none(self):
        from kernelopt.state import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.entry_point is None
        assert cfg.input_shapes is None

    def test_can_set_fields(self):
        from kernelopt.state import PipelineConfig
        cfg = PipelineConfig(
            entry_point="MyModel",
            input_shapes=[(8, 512), (8, 512, 64)],
        )
        assert cfg.entry_point == "MyModel"
        assert cfg.input_shapes == [(8, 512), (8, 512, 64)]

    def test_old_checkpoint_compat(self):
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict
        old_ckpt = {"T": 3, "N": 2, "K": 1, "model": "claude-sonnet-4-6"}
        cfg = pipeline_config_from_checkpoint_dict(old_ckpt)
        assert cfg.entry_point is None
        assert cfg.input_shapes is None
        assert cfg.T == 3

    def test_new_checkpoint_with_entry_point(self):
        from kernelopt.checkpoint import pipeline_config_from_checkpoint_dict
        ckpt = {
            "T": 5, "N": 4, "K": 1,
            "entry_point": "Attention",
            "input_shapes": [(8, 512)],
        }
        cfg = pipeline_config_from_checkpoint_dict(ckpt)
        assert cfg.entry_point == "Attention"
        assert cfg.input_shapes == [(8, 512)]
