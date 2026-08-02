"""Tests for compiler passes: hook installer, capture, fusion patterns,
InductorChoices, autotune plugin, template registration, decompositions.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# =========================================================================
# Hook installer
# =========================================================================

class TestHookInstaller:
    def setup_method(self):
        from kernelopt.compiler.passes.hook_installer import (
            clear_captured_data,
            uninstall_hooks,
        )
        uninstall_hooks()
        clear_captured_data()

    def test_initial_state(self):
        from kernelopt.compiler.passes import is_installed
        assert not is_installed()

    def test_install_uninstall_cycle(self):
        from kernelopt.compiler.passes import install_hooks, uninstall_hooks, is_installed
        install_hooks()
        assert is_installed()
        uninstall_hooks()
        assert not is_installed()

    def test_double_install_warns(self, caplog):
        import logging
        from kernelopt.compiler.passes import install_hooks, uninstall_hooks
        install_hooks()
        with caplog.at_level(logging.WARNING):
            install_hooks()
        assert "already installed" in caplog.text
        uninstall_hooks()

    def test_uninstall_when_not_installed(self):
        from kernelopt.compiler.passes import uninstall_hooks
        uninstall_hooks()

    def test_captured_data_empty_initially(self):
        from kernelopt.compiler.passes import get_captured_data
        data = get_captured_data()
        assert data["pre_grad"] == []
        assert data["post_grad"] == []
        assert data["triton"] == []
        assert data["fusion_pre"] == []
        assert data["fusion_post"] == []
        assert data["fusion_rejections"] == []

    def test_clear_captured_data(self):
        from kernelopt.compiler.passes.hook_installer import (
            _captured_pre_grad,
            clear_captured_data,
        )
        _captured_pre_grad.append({"test": True})
        clear_captured_data()
        from kernelopt.compiler.passes import get_captured_data
        data = get_captured_data()
        assert data["pre_grad"] == []
        assert data["triton"] == []


# =========================================================================
# Capture context manager
# =========================================================================

class TestCaptureContextManager:
    def setup_method(self):
        from kernelopt.compiler.passes.hook_installer import (
            clear_captured_data,
            uninstall_hooks,
        )
        uninstall_hooks()
        clear_captured_data()

    def test_capture_installs_and_uninstalls(self):
        from kernelopt.compiler.passes import compiler_capture, is_installed
        with compiler_capture() as ctx:
            assert is_installed()
        assert not is_installed()

    def test_capture_result_attributes(self):
        from kernelopt.compiler.passes import compiler_capture
        with compiler_capture() as ctx:
            pass
        assert ctx.pre_grad_ops == []
        assert ctx.post_grad_nodes == []
        assert ctx.triton_kernels == []

    def test_capture_uninstalls_on_error(self):
        from kernelopt.compiler.passes import compiler_capture, is_installed
        try:
            with compiler_capture() as ctx:
                raise ValueError("test error")
        except ValueError:
            pass
        assert not is_installed()


# =========================================================================
# Fusion patterns
# =========================================================================

class TestFusionPatterns:
    def setup_method(self):
        from kernelopt.compiler.passes.fusion_patterns import clear_patterns
        clear_patterns()

    def test_register_pattern(self):
        from kernelopt.compiler.passes.fusion_patterns import (
            register_fusion_pattern,
            get_registered_patterns,
        )
        register_fusion_pattern(
            lambda x: x, lambda x: x,
            name="test_pattern",
            min_runtime_pct=5.0,
        )
        patterns = get_registered_patterns()
        assert len(patterns) == 1
        assert patterns[0]["name"] == "test_pattern"
        assert patterns[0]["min_runtime_pct"] == 5.0

    def test_clear_patterns(self):
        from kernelopt.compiler.passes.fusion_patterns import (
            register_fusion_pattern,
            clear_patterns,
            get_registered_patterns,
        )
        register_fusion_pattern(lambda x: x, lambda x: x, name="p1")
        clear_patterns()
        assert get_registered_patterns() == []

    def test_register_builtin_patterns(self):
        from kernelopt.compiler.passes.fusion_patterns import (
            register_builtin_patterns,
            get_registered_patterns,
        )
        register_builtin_patterns()
        patterns = get_registered_patterns()
        assert len(patterns) >= 1
        assert any("swiglu" in p["name"] for p in patterns)

    def test_auto_naming(self):
        from kernelopt.compiler.passes.fusion_patterns import (
            register_fusion_pattern,
            get_registered_patterns,
        )
        register_fusion_pattern(lambda x: x, lambda x: x)
        patterns = get_registered_patterns()
        assert patterns[0]["name"].startswith("pattern_")


# =========================================================================
# InductorChoices
# =========================================================================

class TestInductorChoices:
    def test_override_persistent_reduction(self):
        from kernelopt.compiler.passes.inductor_choices import ProfileGuidedChoices
        choices = ProfileGuidedChoices()
        choices.set_override("persistent_reduction", True)
        assert choices.should_use_persistent_reduction(1024, 100000) is True

    def test_profiling_data_force_streaming(self):
        from kernelopt.compiler.passes.inductor_choices import ProfileGuidedChoices
        choices = ProfileGuidedChoices(
            profiling_data={"reduction_analysis": {"force_streaming": True}}
        )
        assert choices.should_use_persistent_reduction(1, 1) is False

    def test_profiling_data_force_persistent(self):
        from kernelopt.compiler.passes.inductor_choices import ProfileGuidedChoices
        choices = ProfileGuidedChoices(
            profiling_data={"reduction_analysis": {"force_persistent": True}}
        )
        assert choices.should_use_persistent_reduction(1, 100000) is True

    def test_set_override(self):
        from kernelopt.compiler.passes.inductor_choices import ProfileGuidedChoices
        choices = ProfileGuidedChoices()
        choices.set_override("test_key", 42)
        assert choices._overrides["test_key"] == 42


# =========================================================================
# Autotune plugin
# =========================================================================

class TestAutotunePlugin:
    def test_plugin_creation(self):
        from kernelopt.compiler.passes.autotune_plugin import KernelOPTAutotunePlugin
        plugin = KernelOPTAutotunePlugin(
            profiling_data={"suggested_configs": {"k0": [{"BLOCK_M": 256}]}},
            avoid_configs=[{"BLOCK_M": 32}],
        )
        assert plugin.stats["injected"] == 0
        assert plugin.stats["pruned"] == 0

    def test_should_avoid_launcher(self):
        from kernelopt.compiler.passes.autotune_plugin import KernelOPTAutotunePlugin
        plugin = KernelOPTAutotunePlugin(
            avoid_configs=[{"BLOCK_M": 32, "num_warps": 1}]
        )

        class FakeConfig:
            def __init__(self, **kw):
                self.kwargs = kw

        class FakeLauncher:
            def __init__(self, **kw):
                self.config = FakeConfig(**kw)

        bad = FakeLauncher(BLOCK_M=32, num_warps=1)
        good = FakeLauncher(BLOCK_M=128, num_warps=4)
        assert plugin._should_avoid_launcher(bad) is True
        assert plugin._should_avoid_launcher(good) is False

    def test_install_uninstall(self):
        from kernelopt.compiler.passes.autotune_plugin import (
            install_autotune_plugin,
            uninstall_autotune_plugin,
        )
        plugin = install_autotune_plugin(profiling_data={"test": True})
        uninstall_autotune_plugin()

    def test_profiling_data_stored(self):
        from kernelopt.compiler.passes.autotune_plugin import KernelOPTAutotunePlugin
        data = {"suggested_configs": {"k0": [{"BLOCK_M": 256}]}}
        plugin = KernelOPTAutotunePlugin(profiling_data=data)
        assert plugin._profiling_data == data


# =========================================================================
# Template registration
# =========================================================================

class TestTemplateRegistration:
    def setup_method(self):
        from kernelopt.compiler.passes.template_registration import clear_registry
        clear_registry()

    def test_register_kernel(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            get_registered_kernel,
        )
        key = register_optimized_kernel(
            "aten.mm",
            "def kernel(): pass",
            lambda n: (n,),
            num_warps=4,
        )
        result = get_registered_kernel("aten.mm")
        assert result is not None
        assert result["num_warps"] == 4

    def test_register_with_shapes(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            get_registered_kernel,
        )
        register_optimized_kernel(
            "aten.mm",
            "def kernel(): pass",
            lambda n: (n,),
            input_shapes=[(1024, 1024), (1024, 1024)],
        )
        result = get_registered_kernel("aten.mm", [(1024, 1024), (1024, 1024)])
        assert result is not None

    def test_list_registered(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            list_registered_kernels,
        )
        register_optimized_kernel("op1", "src1", lambda n: (n,))
        register_optimized_kernel("op2", "src2", lambda n: (n,))
        kernels = list_registered_kernels()
        assert len(kernels) == 2

    def test_unregister(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            unregister_kernel,
            list_registered_kernels,
        )
        key = register_optimized_kernel("op1", "src1", lambda n: (n,))
        assert unregister_kernel(key)
        assert len(list_registered_kernels()) == 0

    def test_clear_registry(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            clear_registry,
            list_registered_kernels,
        )
        register_optimized_kernel("op1", "src1", lambda n: (n,))
        clear_registry()
        assert len(list_registered_kernels()) == 0

    def test_source_hash(self):
        from kernelopt.compiler.passes.template_registration import (
            register_optimized_kernel,
            get_registered_kernel,
        )
        register_optimized_kernel("op1", "def kernel(): pass", lambda n: (n,))
        result = get_registered_kernel("op1")
        assert len(result["source_hash"]) == 16


# =========================================================================
# Decompositions
# =========================================================================

class TestDecompositions:
    def setup_method(self):
        from kernelopt.compiler.passes.decompositions import clear_decompositions
        clear_decompositions()

    def test_register_decomposition(self):
        from kernelopt.compiler.passes.decompositions import (
            register_decomposition,
            get_custom_decompositions,
        )
        register_decomposition("test_op", lambda x: x, name="test")
        assert "test_op" in get_custom_decompositions()

    def test_clear_decompositions(self):
        from kernelopt.compiler.passes.decompositions import (
            register_decomposition,
            clear_decompositions,
            get_custom_decompositions,
        )
        register_decomposition("op1", lambda x: x)
        clear_decompositions()
        assert len(get_custom_decompositions()) == 0

    def test_get_merged_table(self):
        from kernelopt.compiler.passes.decompositions import (
            register_decomposition,
            get_decomposition_table,
        )
        register_decomposition("custom_op", lambda x: x)
        table = get_decomposition_table()
        assert "custom_op" in table


# =========================================================================
# StableHLO analysis (no torch-mlir required)
# =========================================================================

class TestStableHLOAnalysis:
    def test_analyze_empty(self):
        from kernelopt.compiler.extraction.mlir.stablehlo import analyze_stablehlo
        assert analyze_stablehlo("") == {}

    def test_analyze_with_ops(self):
        from kernelopt.compiler.extraction.mlir.stablehlo import analyze_stablehlo
        hlo_text = """
        func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) {
          %0 = stablehlo.dot_general %arg0, %arg1
          %1 = stablehlo.add %0, %bias
          %2 = stablehlo.reduce %1
          %3 = stablehlo.broadcast_in_dim %2
          %4 = stablehlo.multiply %3, %scale
        }
        """
        result = analyze_stablehlo(hlo_text)
        assert result["op_counts"]["dot_general"] == 1
        assert result["op_counts"]["add"] == 1
        assert result["op_counts"]["reduce"] == 1
        assert "compute_bound_matmul" in result["patterns"]
        assert "reduction" in result["patterns"]
        assert result["total_ops"] > 0

    def test_classify_elementwise_pattern(self):
        from kernelopt.compiler.extraction.mlir.stablehlo import analyze_stablehlo
        hlo_text = """
        %0 = stablehlo.add %a, %b
        %1 = stablehlo.multiply %0, %c
        %2 = stablehlo.subtract %1, %d
        """
        result = analyze_stablehlo(hlo_text)
        assert "memory_bound_elementwise" in result["patterns"]

    def test_export_no_torch_mlir(self):
        from kernelopt.compiler.extraction.mlir.stablehlo import export_to_stablehlo
        with patch.dict("sys.modules", {"torch_mlir": None, "torch_mlir.fx": None}):
            result = export_to_stablehlo(None, ())
            assert result is None


# =========================================================================
# Linalg analysis (no torch-mlir required)
# =========================================================================

class TestLinalgAnalysis:
    def test_analyze_empty(self):
        from kernelopt.compiler.extraction.mlir.linalg_analysis import analyze_linalg
        assert analyze_linalg("") == {}

    def test_extract_iterator_types(self):
        from kernelopt.compiler.extraction.mlir.linalg_analysis import analyze_linalg
        linalg_text = """
        linalg.generic {
          iterator_types = ["parallel", "parallel", "reduction"]
        }
        linalg.generic {
          iterator_types = ["parallel", "parallel"]
        }
        """
        result = analyze_linalg(linalg_text)
        assert result["generic_ops"] == 2
        assert result["iterator_types"][0]["classification"] == "reduction"
        assert result["iterator_types"][1]["classification"] == "elementwise"

    def test_access_patterns(self):
        from kernelopt.compiler.extraction.mlir.linalg_analysis import analyze_linalg
        linalg_text = """
        affine_map<(d0, d1) -> (d0, d1)>
        affine_map<(d0, d1) -> (d1, d0)>
        """
        result = analyze_linalg(linalg_text)
        assert "contiguous_2d" in result["access_patterns"]
        assert "transpose_2d" in result["access_patterns"]

    def test_export_no_torch_mlir(self):
        from kernelopt.compiler.extraction.mlir.linalg_analysis import export_to_linalg
        with patch.dict("sys.modules", {"torch_mlir": None, "torch_mlir.fx": None}):
            result = export_to_linalg(None, ())
            assert result is None
