import pytest


@pytest.fixture
def sample_profiling_context():
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
def sample_kernel_source():
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


@pytest.fixture
def sample_memory():
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
