"""
Pytest configuration and shared fixtures for the kernel optimization test suite.

This file provides:
- Common fixtures for test data, kernels, and profiling contexts
- Mocking utilities for Claude Vertex AI LLM calls
- Test configuration and helpers

Updated for Claude Sonnet 4 on Vertex AI.
"""

from __future__ import annotations

# Fix OpenBLAS thread initialization issues
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))

# Default model for tests (matches README)
DEFAULT_TEST_MODEL = "claude-sonnet-4@20250514"

# Test suite configuration for H200 GPUs
# These tests are optimized for NVIDIA H200 GPUs with:
# - 141 GB HBM3 memory
# - 16896 CUDA cores
# - Compute Capability 9.0
#
# For profiling tests, we require:
# - At least 40% free GPU memory (to accommodate NCU profiling overhead)
# - Less than 20% GPU utilization (for accurate timing measurements)


def pytest_addoption(parser):
    """Add custom command line options for GPU testing."""
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        type=int,
        help="GPU device ID to use for tests (default: auto-select or prompt user)",
    )
    parser.addoption(
        "--skip-gpu-check",
        action="store_true",
        default=False,
        help="Skip GPU resource checking (use with caution)",
    )


# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data and fixtures."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_kernel_path():
    """Path to the primary softmax test kernel."""
    return Path(__file__).parent / "fixtures" / "kernels" / "softmaxtri.py"


@pytest.fixture(scope="session")
def simple_kernel_path():
    """Path to the simple addition test kernel."""
    return Path(__file__).parent / "fixtures" / "kernels" / "simple_add.py"


@pytest.fixture(scope="session")
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Direct LLM Inference (for testing agents)
# ---------------------------------------------------------------------------

_LOADED_MODELS: dict[str, tuple] = {}
_MODEL_LOCK = threading.Lock()


def load_model(model_name: str, gpu_id: int = 2):
    """Load a model+tokenizer once, cache for reuse. Thread-safe."""
    with _MODEL_LOCK:
        if model_name in _LOADED_MODELS:
            return _LOADED_MODELS[model_name]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n  Loading {model_name} on GPU {gpu_id}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if "FP8" not in model_name else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map={"": f"cuda:{gpu_id}"},
            trust_remote_code=True,
        )
        model.eval()

        _LOADED_MODELS[model_name] = (model, tokenizer)
        return model, tokenizer


_INFERENCE_LOCK = threading.Lock()


def direct_inference(
    model_name: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.2,
    gpu_id: int = 2,
) -> str:
    """Mock inference - returns appropriate JSON based on agent type."""
    import json

    # Determine agent type from message content
    msg_content = str(messages).lower()

    # Planner agent response
    if "optimization" in msg_content and "plan" in msg_content:
        return json.dumps({
            "plan_id": "test_plan_001",
            "direction": "Increase thread block occupancy",
            "framework": "triton",
            "evidence": {
                "ncu_rule": "SOL_SMSP__ISSUE_ACTIVE_LOW",
                "reasoning": "Low SM activity indicates underutilization"
            },
            "step": {
                "action": "modify",
                "target": "BLOCK_SIZE",
                "change": "Increase from 128 to 256"
            },
            "confidence": "high"
        })

    # Executor agent response
    elif "apply" in msg_content or "execute" in msg_content or "kernel" in msg_content:
        return json.dumps({
            "status": "success",
            "optimized_source": "# Optimized kernel code\nimport triton\n@triton.jit\ndef kernel():\n    pass",
            "lines_changed": 5,
            "attempt": 1
        })

    # Summarizer agent response
    elif "summarize" in msg_content or "experience" in msg_content:
        return json.dumps({
            "experience_item": {
                "iteration": 1,
                "strategy": "Increase block size",
                "speedup": 1.5,
                "rewrite_type": "positive"
            },
            "memory_update": {
                "action": "append",
                "item": {
                    "direction": "increase_block_size",
                    "speedup": 1.5
                }
            }
        })

    # Default response
    return json.dumps({"status": "success", "result": "test"})


# ---------------------------------------------------------------------------
# LLM Mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predefined responses."""
    def _create_mock(response_text: str = "Mock LLM response"):
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_text
        mock.invoke = MagicMock(return_value=mock_response)
        return mock
    return _create_mock


@pytest.fixture
def patch_build_llm(monkeypatch):
    """Patch build_llm to use direct transformers inference or mock Claude."""
    def _patch(gpu_id: int = 2, use_mock: bool = True):
        import kernelopt.llm as llm_module
        import threading

        # Shared counter across all mock LLM instances in this test
        # (must be outside patched_build_llm so it persists across calls)
        call_counter = {"planner": 0, "executor": 0}
        counter_lock = threading.Lock()

        def patched_build_llm(model=None, project_id=None, region=None,
                              max_tokens=4096, temperature=0.2, **kwargs):
            model = model or DEFAULT_TEST_MODEL

            mock_llm = MagicMock()

            if use_mock:
                # Mock that simulates LangGraph agent behavior with tool calls
                # Uses shared counter to generate different plans for diversity testing
                def invoke(messages):
                    from langchain_core.messages import AIMessage

                    messages_str = str(messages).lower()

                    # Detect planner agent - should call submit_plan
                    # Look for system prompt indicators or tool names
                    if any(keyword in messages_str for keyword in ["profiling", "ncu", "query_profiling_context", "submit_plan"]):
                        # Increment counter for diverse plans (thread-safe)
                        with counter_lock:
                            call_counter["planner"] += 1
                            plan_num = call_counter["planner"]

                        # Generate different plans based on call number
                        directions = ["reduce_register_pressure", "increase_tile_size", "fuse_operations", "optimize_memory_access"]
                        targets = ["register_usage", "BLOCK_SIZE", "loop_fusion", "memory_coalescing"]

                        direction = directions[(plan_num - 1) % len(directions)]
                        target = targets[(plan_num - 1) % len(targets)]

                        # Return AIMessage with tool_calls for submit_plan
                        # Plan must match OptimizationPlan schema exactly
                        return AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "submit_plan",
                                "args": {
                                    "plan": {
                                        "plan_id": f"test_{plan_num:03d}",
                                        "direction": direction,
                                        "framework": "triton",
                                        "evidence": {
                                            "ncu_rule": "SOL_DRAM",
                                            "ncu_rule_speedup_pct": 15.0,
                                            "supporting_metrics": {"Memory Throughput": 45.0},
                                            "reasoning": "Memory throughput is at 45%, indicating memory bound kernel. Increasing block size can improve occupancy."
                                        },
                                        "step": {
                                            "action": "modify",
                                            "target": target,
                                            "change": f"Optimize {target} for better performance",
                                            "predicted_effect": "Should improve kernel performance by 10-20%",
                                            "implementation_hints": [
                                                f"Update {target} parameter in kernel code",
                                                "Verify correctness after change"
                                            ],
                                            "success_criteria": {
                                                "primary_metric": "Occupancy",
                                                "expected_direction": "increase",
                                                "expected_magnitude": "20-30%",
                                                "secondary_effects": "Memory throughput may also increase"
                                            }
                                        },
                                        "correctness_constraints": ["Must preserve output shape", "Must maintain numerical accuracy"],
                                        "architecture_notes": None,
                                        "memory_item_used": None,
                                        "confidence": "medium",
                                        "confidence_reason": "NCU rule suggests clear optimization path, but magnitude uncertain"
                                    }
                                },
                                "id": "call_test_001"
                            }]
                        )

                    # Detect executor agent - should call submit_kernel
                    elif any(keyword in messages_str for keyword in ["executor", "get_kernel_source", "get_optimization_plan", "submit_kernel"]):
                        # Return valid Triton kernel with @triton.jit marker (required for validation)
                        return AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "submit_kernel",
                                "args": {
                                    "kernel_source": """import triton
import triton.language as tl

@triton.jit
def optimized_kernel(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Optimized kernel implementation
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * N + offs, mask=offs < N)
    y = x * 2.0  # Simple optimization
    tl.store(y_ptr + pid * N + offs, y, mask=offs < N)
""",
                                    "change_summary": "Applied test optimization: doubled computation efficiency"
                                },
                                "id": "call_test_002"
                            }]
                        )

                    # Default: no tool calls, just text response
                    else:
                        return AIMessage(
                            content='{"status": "success"}',
                            tool_calls=[]
                        )

                mock_llm.invoke = invoke
            else:
                # Use direct inference for real tests
                def invoke(messages):
                    msg_dicts = []
                    for m in messages:
                        if hasattr(m, "type"):
                            role = {"system": "system", "human": "user", "ai": "assistant"}.get(m.type, "user")
                        else:
                            role = "user"
                        msg_dicts.append({"role": role, "content": m.content})

                    result = direct_inference(
                        "claude-sonnet-4@20250514",  # Uses Claude Vertex AI for tests
                        msg_dicts, max_tokens=max_tokens,
                        temperature=temperature, gpu_id=gpu_id,
                    )

                    response = MagicMock()
                    response.content = result
                    return response

                mock_llm.invoke = invoke

            # Add bind_tools method (LangGraph calls this to bind tools to LLM)
            def bind_tools(tools, **kwargs):
                # Return the same mock - tools are already "bound" via our invoke logic
                return mock_llm
            mock_llm.bind_tools = bind_tools

            return mock_llm

        monkeypatch.setattr(llm_module, "build_llm", patched_build_llm)

    return _patch


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profiling_context():
    """Sample profiling context for testing."""
    return {
        "kernel_name": "triton_softmax",
        "kernel_name_full": "triton__0d1d2d3_softmax_kernel",
        "kernel_id": 0,
        "framework_hint": "triton",
        "device": "NVIDIA H100 SXM5",
        "compute_capability": "9.0",
        "launch": {
            "block_size": 1024,
            "grid_size": 4096,
            "sm_count": 132,
            "waves_per_sm": 0.62,
            "registers_per_thread": 48,
            "dynamic_smem_per_block_bytes": 0,
            "static_smem_per_block_bytes": 0,
            "threads": 1048576,
        },
        "gpu_specs": {
            "gpu_model": "H100 SXM5",
            "gpu_model_detected": True,
            "peak_hbm_bandwidth_gbs": 3350.0,
            "peak_fp32_tflops": 67.0,
            "sm_count": 132,
        },
        "sections": {
            "GPU Speed Of Light Throughput": {
                "Compute (SM) Throughput": {"value": 18.4, "unit": "%"},
                "Memory Throughput": {"value": 72.1, "unit": "%"},
            },
            "Occupancy": {
                "Theoretical Occupancy": {"value": 62.5, "unit": "%"},
                "Achieved Occupancy": {"value": 50.5, "unit": "%"},
                "Block Limit Registers": {"value": 5, "unit": "block"},
                "Block Limit Warps": {"value": 8, "unit": "block"},
            },
            "Memory Workload Analysis": {
                "DRAM Throughput": {"value": 68.3, "unit": "%"},
            },
            "Scheduler Statistics": {
                "No Eligible": {"value": 41.2, "unit": "%"},
            },
            "Instruction Statistics": {
                "IPC Active": {"value": 1.2, "unit": "inst/cycle"},
            },
        },
        "rules": [
            {
                "name": "TheoreticalOccupancy",
                "type": "OPT",
                "description": "Register usage limits occupancy to 62.5%.",
                "speedup_type": "global",
                "speedup_pct": 24.0,
            },
            {
                "name": "MemoryBound",
                "type": "INF",
                "description": "Kernel is memory bound.",
                "speedup_type": "local",
                "speedup_pct": 8.0,
            },
        ],
    }


@pytest.fixture
def sample_optimization_plan():
    """Sample optimization plan for testing."""
    return {
        "plan_id": "test_plan_001",
        "direction": "improve_memory_coalescing",
        "framework": "triton",
        "evidence": {
            "ncu_rule": "UncoalescedGlobalAccesses",
            "ncu_rule_speedup_pct": 8.2,
            "supporting_metrics": {
                "Memory Throughput": 850.5,
            },
            "reasoning": "Improve memory access patterns to achieve better coalescing",
        },
        "step": {
            "action": "modify",
            "target": "memory access pattern",
            "change": "Use vectorized loads with proper alignment",
            "predicted_effect": "Reduced memory latency and improved bandwidth utilization",
        },
        "correctness_constraints": [
            "Output must match reference within 1e-4 tolerance",
        ],
        "confidence": "high",
        "confidence_reason": "NCU rule has strong evidence",
    }


@pytest.fixture
def sample_execution_result():
    """Sample execution result for testing."""
    return {
        "status": "success",
        "lines_changed": 5,
        "ready_for_profiling": True,
        "correctness": "passed",
    }


@pytest.fixture
def sample_memory_items():
    """Sample optimization memory items for testing."""
    return [
        {
            "item_id": "abc123",
            "iteration": 1,
            "speedup": 1.28,
            "rewrite_type": "positive",
            "framework": "triton",
            "direction": "reduce_tile_size_for_occupancy",
            "strategy_title": "Halve tile size to reduce register pressure",
            "strategy_description": (
                "When Block Limit Registers is the binding constraint and "
                "registers_per_thread > 32, reducing tile size lowers register usage."
            ),
            "applicable_when": (
                "Block Limit Registers < Block Limit Warps AND registers_per_thread > 32"
            ),
            "do_not_apply_when": "Kernel is compute-bound (high SM throughput).",
            "profiling_signal": {},
            "slow_pseudocode": "",
            "fast_pseudocode": "",
            "framework_notes": None,
        },
        {
            "item_id": "def456",
            "iteration": 2,
            "speedup": 0.91,
            "rewrite_type": "negative",
            "framework": "triton",
            "direction": "increase_num_warps",
            "strategy_title": "Increasing num_warps caused regression",
            "strategy_description": "More warps increased register pressure further.",
            "applicable_when": "N/A",
            "do_not_apply_when": "When register pressure is already high.",
            "profiling_signal": {},
            "slow_pseudocode": "",
            "fast_pseudocode": "",
            "framework_notes": None,
        },
    ]


@pytest.fixture
def sample_memory(sample_memory_items):
    """Alias for sample_memory_items for backwards compatibility."""
    return sample_memory_items


# ---------------------------------------------------------------------------
# GPU and CUDA Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for tests."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_id(request):
    """
    Get GPU ID for tests.

    Prompts user to select GPU if not specified via TEST_GPU_ID environment variable.
    Checks that GPU has sufficient resources for testing (40% free memory minimum).
    """
    from unit.utils.gpu_utils import (
        list_available_gpus,
        print_gpu_summary,
        select_best_gpu,
        is_gpu_free,
        check_gpu_for_profiling,
    )

    # Check if GPU ID is specified via environment variable or command line
    gpu_id_env = os.environ.get("TEST_GPU_ID")
    gpu_id_cli = request.config.getoption("--gpu", default=None)

    if gpu_id_cli is not None:
        selected_gpu = int(gpu_id_cli)
    elif gpu_id_env is not None:
        selected_gpu = int(gpu_id_env)
    else:
        # Show available GPUs
        print("\n" + "="*70)
        print("GPU SELECTION FOR TESTS (H200 Optimized)")
        print("="*70)
        print_gpu_summary()

        # Try to auto-select best GPU
        best_gpu = select_best_gpu(min_free_memory_pct=40.0)

        if best_gpu is not None:
            print(f"Recommended GPU: {best_gpu} (most free memory)")
            response = input(f"\nUse GPU {best_gpu}? [Y/n]: ").strip().lower()

            if response in ['', 'y', 'yes']:
                selected_gpu = best_gpu
            else:
                gpus = list_available_gpus()
                selected_gpu = int(input(f"Enter GPU ID to use {gpus}: "))
        else:
            gpus = list_available_gpus()
            if not gpus:
                pytest.skip("No CUDA GPUs available")

            print("Warning: No GPU with ≥40% free memory found")
            selected_gpu = int(input(f"Enter GPU ID to use {gpus}: "))

    # Validate selected GPU
    try:
        check_gpu_for_profiling(selected_gpu)
    except Exception as e:
        print(f"\nWarning: {e}")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            pytest.skip(f"GPU {selected_gpu} does not meet requirements")

    return selected_gpu


@pytest.fixture(scope="session")
def vertex_project_id():
    """Vertex AI project ID for tests."""
    return os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "test-project")


@pytest.fixture(scope="session")
def vertex_region():
    """Vertex AI region for tests."""
    return os.environ.get("VERTEX_REGION", "us-east5")


@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_gpu_busy(gpu_id):
    """Skip test if GPU doesn't have enough free resources."""
    from unit.utils.gpu_utils import is_gpu_free

    if not is_gpu_free(gpu_id, min_free_memory_pct=40.0, max_utilization_pct=20.0):
        pytest.skip(f"GPU {gpu_id} is too busy for profiling tests")


@pytest.fixture
def ensure_gpu_free_memory(gpu_id):
    """Ensure GPU has at least 40% free memory before test."""
    from unit.utils.gpu_utils import check_gpu_for_profiling, clear_gpu_memory

    # Clear cache first
    clear_gpu_memory(gpu_id)

    # Check resources
    try:
        check_gpu_for_profiling(gpu_id)
    except Exception as e:
        pytest.skip(f"Insufficient GPU resources: {e}")

    yield

    # Cleanup after test
    clear_gpu_memory(gpu_id)


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def create_temp_kernel():
    """Factory to create temporary kernel files for testing."""
    def _create(content: str, suffix: str = ".py") -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as f:
            f.write(content)
            path = Path(f.name)
        return path
    return _create


@pytest.fixture
def sample_ncu_csv():
    """Path to sample NCU CSV file (simple synthetic sample)."""
    return Path(__file__).parent / "fixtures" / "data" / "sample_ncu_output.csv"


@pytest.fixture
def h200_ncu_csv():
    """Path to real H200 NCU profiling output CSV."""
    return Path(__file__).parent / "fixtures" / "data" / "h200_sample_ncu_output.csv"


@pytest.fixture(scope="session")
def run_sanity_check_fixture():
    """Run sanity check before optimization pipeline tests."""
    import subprocess

    sanity_check_path = Path(__file__).parent / "unit" / "core" / "sanity_check.py"

    print("\n" + "="*70)
    print("🔍 Running Sanity Check before optimization pipeline tests...")
    print("="*70 + "\n")

    result = subprocess.call([sys.executable, str(sanity_check_path)])

    if result != 0:
        print("\n" + "="*70)
        print("⚠️  Sanity check failed!")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("✅ Sanity check passed!")
        print("="*70 + "\n")

    return result == 0


# ---------------------------------------------------------------------------
# Additional Fixtures from kernelopt/tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_kernel_source():
    """Sample Triton kernel source code for testing."""
    return """\
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE))
    y = tl.softmax(x, axis=0)
    tl.store(y_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), y)
"""
