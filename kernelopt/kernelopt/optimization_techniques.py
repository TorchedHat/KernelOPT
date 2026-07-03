"""Optimization technique library and deterministic selection for GPU kernels.

Layer 1: TECHNIQUE_LIBRARY — a registry of ~14 concrete GPU kernel optimization
         techniques, each with before/after Triton code patterns and hardware-
         aware applicability conditions.

Layer 2: select_techniques() — deterministically selects the top N applicable
         techniques based on NCU profiling metrics, kernel type, and GPU
         compute capability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TechniqueConditions:
    bottleneck_types: list[str]        # ["memory_bound", "compute_bound", "underutilized", "near_optimal"]
    kernel_types: list[str]            # ["matmul", "reduction", "pointwise", "attention"] — empty = all
    min_compute_capability: str        # "9.0" for Hopper, "8.0" for Ampere, "" for any
    source_contains: list[str]         # kernel source must contain at least one
    source_excludes: list[str]         # kernel source must NOT contain any (= already applied)


@dataclass
class CodePattern:
    before: str       # 5-15 lines of Triton code showing the unoptimized pattern
    after: str        # 5-15 lines showing the optimized pattern
    explanation: str   # WHY this works, not just what changed


@dataclass
class OptimizationTechnique:
    name: str
    category: str              # "memory" | "compute" | "parallelism" | "structural"
    description: str           # 1-2 sentences
    when_to_use: TechniqueConditions
    code_pattern: CodePattern
    expected_impact: str       # e.g. "1.5-3x for memory-bound matmul on Hopper"
    conflicts_with: list[str] = field(default_factory=list)  # technique names that shouldn't co-apply


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _classify_kernel(source: str) -> str:
    """Classify a kernel into a pattern type from its source code.

    Returns one of: "matmul", "reduction", "pointwise", "attention", "softmax".
    """
    src = source.lower()

    # Attention: must check before matmul since attention kernels often
    # contain tl.dot but also have softmax/score/mask patterns.
    if ("score" in src or "attn" in src or "attention" in src) and (
        "softmax" in src or "tl.exp" in src
    ):
        return "attention"

    # Softmax: standalone softmax without attention context
    if "softmax" in src and ("tl.exp" in src or "tl.max" in src):
        return "softmax"

    # Matmul: look for tl.dot or GEMM-style block tiling
    if "tl.dot" in src or (
        "block_m" in src and "block_n" in src and "block_k" in src
    ):
        return "matmul"

    # Reduction: sum, max, argmax over a dimension
    if any(op in src for op in ("tl.sum", "tl.max", "tl.argmax", "rblock")):
        return "reduction"

    return "pointwise"


def _parse_compute_capability(cc_str: str) -> float:
    """Parse a compute capability string into a comparable float.

    Handles formats like "9.0", "90", "8.9", "89", "".
    Returns 0.0 for empty or unparseable strings.
    """
    if not cc_str:
        return 0.0
    cc_str = cc_str.strip()

    # Already has a dot: "9.0", "8.9"
    if "." in cc_str:
        try:
            return float(cc_str)
        except ValueError:
            return 0.0

    # Integer form: "90" → 9.0, "89" → 8.9, "80" → 8.0
    try:
        val = int(cc_str)
        if val >= 10:
            return val / 10.0
        return float(val)
    except ValueError:
        return 0.0


def _get_ncu_tier(profiling_context: dict) -> str:
    """Determine the bottleneck type from NCU Speed Of Light percentages.

    Classification logic (matches NCU's own roofline analysis):
      - near_optimal: both memory% and compute% >= 60
      - memory_bound:  memory% > compute% and memory% >= 20
      - compute_bound: compute% > memory% and compute% >= 20
      - underutilized: both < 20
    """
    sections = profiling_context.get("sections", {})

    # NCU section name for Speed Of Light
    sol_section = sections.get("GPU Speed Of Light Throughput", {})

    mem_pct = 0.0
    compute_pct = 0.0

    # Try the standard metric names NCU emits
    for key, metric in sol_section.items():
        val = metric.get("value", 0) if isinstance(metric, dict) else 0
        if not isinstance(val, (int, float)):
            try:
                val = float(str(val).replace("%", "").replace(",", ""))
            except (ValueError, TypeError):
                val = 0.0

        key_lower = key.lower()
        if "mem" in key_lower and "%" in key_lower or "memory" in key_lower:
            mem_pct = max(mem_pct, float(val))
        elif "sm" in key_lower and "%" in key_lower or "compute" in key_lower:
            compute_pct = max(compute_pct, float(val))

    # Fallback: check for explicit SOL% metrics
    if mem_pct == 0 and compute_pct == 0:
        for key, metric in sol_section.items():
            val = metric.get("value", 0) if isinstance(metric, dict) else 0
            if not isinstance(val, (int, float)):
                continue
            if "dram" in key.lower() or "l2" in key.lower() or "mem" in key.lower():
                mem_pct = max(mem_pct, float(val))
            elif "sm" in key.lower() or "alu" in key.lower() or "fma" in key.lower():
                compute_pct = max(compute_pct, float(val))

    if mem_pct >= 60 and compute_pct >= 60:
        return "near_optimal"
    if mem_pct >= 20 and mem_pct > compute_pct:
        return "memory_bound"
    if compute_pct >= 20 and compute_pct > mem_pct:
        return "compute_bound"
    return "underutilized"


# ---------------------------------------------------------------------------
# Layer 1: Technique registry
# ---------------------------------------------------------------------------

TECHNIQUE_LIBRARY: list[OptimizationTechnique] = [

    # ── Memory (4) ────────────────────────────────────────────────────────

    OptimizationTechnique(
        name="tma_loads",
        category="memory",
        description=(
            "Replace scalar pointer arithmetic with Tensor Memory Accelerator "
            "(TMA) descriptors on Hopper GPUs. TMA offloads address generation "
            "to a dedicated hardware unit, freeing SM cycles and enabling "
            "asynchronous bulk copies."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="9.0",
            source_contains=["tl.load", "tl.dot"],
            source_excludes=["tl.make_tensor_descriptor", "tl._experimental_descriptor_load"],
        ),
        code_pattern=CodePattern(
            before="""\
# Standard pointer-arithmetic loads — SM computes every address
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
    a_ptrs += BLOCK_K""",
            after="""\
# TMA descriptor loads — hardware address generation unit handles
# all index math; SM issues a single descriptor load instruction
a_desc = tl.make_tensor_descriptor(
    a_ptr, shape=[M, K], strides=[K, 1],
    block_shape=[BLOCK_M, BLOCK_K],
)
for k in range(0, K, BLOCK_K):
    a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_M, k])""",
            explanation=(
                "Hopper's TMA engine is a dedicated address-generation unit on "
                "each SM. By expressing loads as tensor descriptors, the SM no "
                "longer spends ALU cycles computing pointer offsets for each "
                "element. The TMA unit also coalesces and pads automatically, "
                "eliminating manual mask logic. This frees the SM pipeline for "
                "compute and reduces register pressure from address variables."
            ),
        ),
        expected_impact="1.3-2x for memory-bound matmul on Hopper (cc>=9.0)",
        conflicts_with=["block_pointers"],
    ),

    OptimizationTechnique(
        name="block_pointers",
        category="memory",
        description=(
            "Replace scalar pointer arithmetic with block pointers "
            "(tl.make_block_ptr / tl.advance). Enables the compiler to prove "
            "contiguity and emit wider vector loads without manual bookkeeping."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="8.0",
            source_contains=["tl.load", "tl.dot"],
            source_excludes=["tl.make_block_ptr", "tl.make_tensor_descriptor"],
        ),
        code_pattern=CodePattern(
            before="""\
# Manual pointer offsets — compiler cannot prove contiguity
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
    a_ptrs += BLOCK_K * stride_ak""",
            after="""\
# Block pointer — compiler knows the tile shape and can emit
# widest possible vector loads (e.g. 128-bit LDG.128)
a_block_ptr = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))""",
            explanation=(
                "When the compiler sees scalar pointer arithmetic, it cannot "
                "statically prove that loads are contiguous in memory, so it "
                "falls back to narrow scalar loads (LDG.32). Block pointers "
                "encode shape, stride, and order metadata that lets the Triton "
                "compiler emit vectorized loads (LDG.128) and avoid redundant "
                "address math. The order=(1,0) parameter specifies that the "
                "innermost dimension is contiguous, enabling maximum coalescing."
            ),
        ),
        expected_impact="1.3-2x for memory-bound matmul on Ampere+ (cc>=8.0)",
        conflicts_with=["tma_loads"],
    ),

    OptimizationTechnique(
        name="eviction_policies",
        category="memory",
        description=(
            "Annotate tl.load/tl.store with cache eviction policies to control "
            "L2 residency. Streaming data that is used once should be marked "
            "evict_first to avoid polluting the L2 for reused data."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=[],  # any kernel type
            min_compute_capability="",
            source_contains=["tl.load"],
            source_excludes=["eviction_policy"],
        ),
        code_pattern=CodePattern(
            before="""\
# Default eviction — L2 treats all lines equally (LRU)
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
# x is reused across iterations, y is streaming
output = x * y
tl.store(out_ptr + offsets, output, mask=mask)""",
            after="""\
# Explicit eviction — keep reused data in L2, evict streaming data
x = tl.load(x_ptr + offsets, mask=mask, eviction_policy='evict_last')
y = tl.load(y_ptr + offsets, mask=mask, eviction_policy='evict_first')
output = x * y
tl.store(out_ptr + offsets, output, mask=mask,
         eviction_policy='evict_first')""",
            explanation=(
                "The GPU L2 cache uses an LRU-like policy by default. When a "
                "kernel mixes streaming data (accessed once) with reused data "
                "(accessed across loop iterations), the streaming lines thrash "
                "the reused lines out of cache. evict_first tells the L2 to "
                "place the line at the LRU end immediately so it gets evicted "
                "first, preserving cache space for evict_last lines that will "
                "be accessed again. This can significantly reduce L2 miss rate "
                "for mixed-access patterns."
            ),
        ),
        expected_impact="1.1-1.5x for memory-bound kernels with mixed access patterns",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="vectorized_loads",
        category="memory",
        description=(
            "Use tl.max_contiguous and tl.multiple_of hints to enable the "
            "compiler to emit 128-bit vectorized loads instead of scalar loads. "
            "Critical for pointwise and reduction kernels with contiguous access."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["pointwise", "reduction"],
            min_compute_capability="",
            source_contains=["tl.load"],
            source_excludes=["tl.max_contiguous", "tl.multiple_of"],
        ),
        code_pattern=CodePattern(
            before="""\
# Bare offsets — compiler assumes worst-case alignment
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)""",
            after="""\
# Annotated offsets — compiler can prove 128-bit alignment and
# emit LDG.128 (4x wider than scalar LDG.32)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)""",
            explanation=(
                "GPU memory controllers transfer data in 32-byte sectors. A "
                "single LDG.128 instruction loads 16 bytes per thread — when "
                "32 threads in a warp issue it together, the controller can "
                "serve them in one transaction. Without contiguity hints, the "
                "compiler must assume offsets could be strided or misaligned, "
                "forcing it to emit narrow LDG.32 loads that require 4x more "
                "memory transactions. tl.multiple_of proves alignment; "
                "tl.max_contiguous proves the range is dense."
            ),
        ),
        expected_impact="1.5-3x for memory-bound pointwise/reduction with contiguous access",
        conflicts_with=[],
    ),

    # ── Compute (4) ───────────────────────────────────────────────────────

    OptimizationTechnique(
        name="tf32_tensor_cores",
        category="compute",
        description=(
            "Enable TF32 (TensorFloat-32) precision in tl.dot to use Tensor "
            "Cores at roughly 8x the throughput of FP32 FFMA, at the cost of "
            "reducing mantissa precision from 23 bits to 10 bits."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["compute_bound", "underutilized", "memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="8.0",
            source_contains=["tl.dot"],
            source_excludes=["allow_tf32=True", "allow_tf32 = True"],
        ),
        code_pattern=CodePattern(
            before="""\
# FP32 dot — uses FFMA (scalar FP32 multiply-add), 1x throughput
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator += tl.dot(a, b)
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * stride_bk""",
            after="""\
# TF32 dot — uses Tensor Cores (HMMA), ~8x throughput on Ampere
# Truncates FP32 mantissa from 23 to 10 bits (still 8-bit exponent)
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator += tl.dot(a, b, allow_tf32=True)
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * stride_bk""",
            explanation=(
                "Ampere and later GPUs have Tensor Cores that operate on TF32: "
                "a 19-bit format with FP32's 8-bit exponent but only 10 bits "
                "of mantissa (vs 23 for FP32). This lets Tensor Cores perform "
                "matrix multiply-accumulate on 4x4 or 8x8 tiles in hardware, "
                "achieving ~8x the FP32 FLOPS. The accumulator stays in full "
                "FP32 so rounding errors don't cascade across the K dimension. "
                "For most ML workloads the precision loss is negligible."
            ),
        ),
        expected_impact="2-3x speedup for compute-bound matmul on Ampere+ (cc>=8.0)",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="fast_accumulation",
        category="compute",
        description=(
            "Enable FP16 accumulation in the matrix multiply inner loop when "
            "the final result tolerance permits it. Doubles Tensor Core "
            "throughput by avoiding the FP32 accumulate step."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["compute_bound"],
            kernel_types=["matmul"],
            min_compute_capability="8.0",
            source_contains=["tl.dot"],
            source_excludes=["USE_FAST_ACCUM", "input_precision"],
        ),
        code_pattern=CodePattern(
            before="""\
# FP32 accumulation — safe but slower; Tensor Core must widen each
# partial product to FP32 before adding to accumulator
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs).to(tl.float16)
    b = tl.load(b_ptrs).to(tl.float16)
    accumulator += tl.dot(a, b)""",
            after="""\
# FP16 accumulation — Tensor Core accumulates in FP16 natively,
# roughly doubling throughput (one fewer conversion per MAC)
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs).to(tl.float16)
    b = tl.load(b_ptrs).to(tl.float16)
    accumulator = tl.dot(a, b, acc=accumulator,
                         input_precision="ieee")
# Note: USE_FAST_ACCUM flag can also be passed via constexpr meta
# to let the autotuner toggle this on/off""",
            explanation=(
                "When Tensor Cores accumulate in FP32, each 4x4 tile result "
                "must be widened from FP16 to FP32 before the add — this extra "
                "conversion halves throughput. FP16 accumulation skips the "
                "widening and performs the add in half-precision directly. The "
                "trade-off is reduced numerical precision: over long K "
                "dimensions, FP16 accumulation can lose significant bits. Best "
                "for inference or short-K training steps where the optimizer "
                "corrects any drift."
            ),
        ),
        expected_impact="1.3-2x for compute-bound FP16 matmul on Ampere+ (cc>=8.0)",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="fp8_precision",
        category="compute",
        description=(
            "Cast operands to FP8 (E4M3 or E5M2) before tl.dot on Hopper GPUs. "
            "FP8 Tensor Cores deliver 2x the FLOPS of FP16 Tensor Cores at the "
            "cost of reduced dynamic range and precision."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["compute_bound"],
            kernel_types=["matmul"],
            min_compute_capability="9.0",
            source_contains=["tl.dot"],
            source_excludes=["float8", "tl.float8e4nv", "tl.float8e5"],
        ),
        code_pattern=CodePattern(
            before="""\
# FP16 Tensor Core matmul
a = tl.load(a_ptrs).to(tl.float16)
b = tl.load(b_ptrs).to(tl.float16)
accumulator += tl.dot(a, b)""",
            after="""\
# FP8 Tensor Core matmul — 2x FP16 FLOPS on Hopper
# E4M3: 4-bit exponent, 3-bit mantissa — good for forward pass
# E5M2: 5-bit exponent, 2-bit mantissa — good for gradients
a = tl.load(a_ptrs).to(tl.float8e4nv)
b = tl.load(b_ptrs).to(tl.float8e4nv)
accumulator += tl.dot(a, b)""",
            explanation=(
                "Hopper's 4th-gen Tensor Cores have native FP8 datapaths that "
                "process twice as many elements per cycle as FP16. The E4M3 "
                "format (4-bit exponent, 3-bit mantissa) covers a dynamic "
                "range of +/-448 with ~3 decimal digits of precision — "
                "sufficient for most inference workloads when combined with "
                "per-tensor or per-channel scaling factors. The memory "
                "bandwidth benefit is also significant: FP8 halves the bytes "
                "transferred compared to FP16."
            ),
        ),
        expected_impact="1.5-2x for compute-bound matmul on Hopper (cc>=9.0)",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="software_pipelining",
        category="compute",
        description=(
            "Increase the number of pipeline stages (num_stages) to overlap "
            "memory loads with compute. While one tile is being computed, the "
            "next tile's data is being prefetched asynchronously."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["matmul", "reduction"],
            min_compute_capability="",
            source_contains=["tl.load", "tl.dot"],
            source_excludes=["num_stages"],
        ),
        code_pattern=CodePattern(
            before="""\
# Single-stage pipeline — load stalls until data arrives
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # ... pid setup ...
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)     # stall: wait for DRAM
        b = tl.load(b_ptrs)     # stall: wait for DRAM
        acc += tl.dot(a, b)     # compute only starts after both loads
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk""",
            after="""\
# Multi-stage pipeline — loads overlap with compute via async copy
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr,
                  num_stages: tl.constexpr = 3):
    # ... pid setup ...
    # num_stages=3 means 3 tiles in flight: 1 computing, 2 prefetching
    for k in tl.range(0, K, BLOCK_K, num_stages=num_stages):
        a = tl.load(a_ptrs)     # async: returns immediately
        b = tl.load(b_ptrs)     # async: returns immediately
        acc += tl.dot(a, b)     # compute on tile from 2 iterations ago
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk""",
            explanation=(
                "DRAM latency is 400-800 cycles on modern GPUs. Without "
                "pipelining, the SM stalls on every tl.load waiting for data "
                "to arrive. With num_stages=N, the compiler emits async copy "
                "instructions (cp.async on Ampere, TMA on Hopper) that fill N "
                "buffers in shared memory. While the dot product processes "
                "buffer[i], the memory system is filling buffer[i+1] and "
                "buffer[i+2]. This hides memory latency at the cost of "
                "additional shared memory (N * tile_size bytes)."
            ),
        ),
        expected_impact="1.3-2x for memory-bound matmul/reduction, depends on K dimension",
        conflicts_with=[],
    ),

    # ── Parallelism (4) ──────────────────────────────────────────────────

    OptimizationTechnique(
        name="persistent_kernels",
        category="parallelism",
        description=(
            "Launch exactly one thread block per SM and have each block loop "
            "over multiple output tiles. Eliminates block scheduling overhead "
            "and enables cross-tile data reuse in shared memory or registers."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["underutilized"],
            kernel_types=["matmul"],
            min_compute_capability="",
            source_contains=["tl.dot"],
            source_excludes=["NUM_SMS", "num_sms", "tl.range"],
        ),
        code_pattern=CodePattern(
            before="""\
# Non-persistent: one block per output tile, many blocks launched
pid = tl.program_id(0)
pid_m = pid // num_n_tiles
pid_n = pid % num_n_tiles
# ... compute one tile and exit""",
            after="""\
# Persistent: one block per SM, loops over all assigned tiles
pid = tl.program_id(0)
num_tiles = num_m_tiles * num_n_tiles
for tile_id in tl.range(pid, num_tiles, NUM_SMS):
    pid_m = tile_id // num_n_tiles
    pid_n = tile_id % num_n_tiles
    # ... compute tile (pid_m, pid_n)
    # Block stays resident — no re-launch overhead per tile""",
            explanation=(
                "Standard Triton kernels launch one block per output tile, "
                "which can mean thousands of blocks for large problems. The GPU "
                "block scheduler has non-trivial overhead (~1us per wave) and "
                "blocks cannot share state across launches. Persistent kernels "
                "launch exactly NUM_SMS blocks (one per SM) that each loop over "
                "their assigned tiles. This eliminates scheduling overhead, "
                "allows the block to keep data in shared memory across tiles, "
                "and gives the programmer explicit control over tile-to-SM "
                "mapping for L2 locality."
            ),
        ),
        expected_impact="1.2-2x for underutilized matmul with many small tiles",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="split_k",
        category="parallelism",
        description=(
            "Partition the K (reduction) dimension across multiple thread blocks, "
            "then reduce partial results with atomic adds. Increases parallelism "
            "when M and N are small but K is large."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["underutilized"],
            kernel_types=["matmul"],
            min_compute_capability="",
            source_contains=["tl.dot"],
            source_excludes=["SPLIT_K", "split_k", "tl.atomic_add"],
        ),
        code_pattern=CodePattern(
            before="""\
# Standard matmul: each block computes full K reduction
# Problem: if M=N=64, only 1 tile → 1 SM busy, rest idle
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * stride_bk
tl.store(c_ptrs, acc)""",
            after="""\
# Split-K: partition K across SPLIT_K blocks, atomic reduce
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
pid_k = tl.program_id(2)   # grid z-dimension = SPLIT_K
k_start = pid_k * (K // SPLIT_K)
k_end = min(k_start + (K // SPLIT_K), K)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(k_start, k_end, BLOCK_K):
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * stride_bk
# Atomic add partial results — all SPLIT_K blocks contribute
tl.atomic_add(c_ptrs, acc)""",
            explanation=(
                "For GEMM with small M and N but large K, the output tile grid "
                "is tiny — often just a few blocks — leaving most SMs idle. "
                "Split-K creates SPLIT_K independent blocks per output tile, "
                "each responsible for K/SPLIT_K elements of the reduction. The "
                "partial results are combined with atomic adds. This trades "
                "extra memory traffic (the atomic reduce) for dramatically "
                "higher SM occupancy. SPLIT_K is typically chosen as "
                "num_SMs / (M_tiles * N_tiles), clamped to a power of 2."
            ),
        ),
        expected_impact="2-5x for underutilized matmul with small M,N and large K",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="l2_tile_grouping",
        category="parallelism",
        description=(
            "Reorder the program_id to tile mapping so that spatially adjacent "
            "thread blocks access adjacent rows/columns, maximizing L2 cache "
            "hit rate for the shared operand."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="",
            source_contains=["tl.dot", "tl.program_id"],
            source_excludes=["GROUP_M", "group_m", "swizzle"],
        ),
        code_pattern=CodePattern(
            before="""\
# Naive row-major tile order: pid 0,1,2,... map to (m=0,n=0),(m=0,n=1),...
# Adjacent blocks access completely different B-columns → L2 thrash
pid = tl.program_id(0)
pid_m = pid // num_n_tiles
pid_n = pid % num_n_tiles""",
            after="""\
# Grouped tile order: GROUP_M adjacent rows share the same B columns
# Blocks in the same group hit L2 for B; blocks across groups hit L2 for A
pid = tl.program_id(0)
num_pid_in_group = GROUP_M * num_n_tiles
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_m_tiles - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m""",
            explanation=(
                "In a standard row-major mapping, consecutive PIDs walk across "
                "the N dimension. If there are many N-tiles, by the time pid_m "
                "increments, the A-tile data loaded by the previous row of "
                "blocks has been evicted from L2. GROUP_M reorders the mapping "
                "so that GROUP_M rows of tiles are interleaved: blocks within a "
                "group share columns of B (so B stays in L2), while groups "
                "share rows of A. This is the same swizzle trick used by cuBLAS "
                "and reduces L2 miss rate by up to 50% on large matrices."
            ),
        ),
        expected_impact="1.2-1.5x for memory-bound matmul with large M and N",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="warp_specialization",
        category="parallelism",
        description=(
            "Split warps within a block into producer warps (that issue TMA "
            "loads) and consumer warps (that execute Tensor Core math). "
            "Producers and consumers synchronize via barriers, enabling true "
            "overlap of memory and compute pipelines."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="9.0",
            source_contains=["tl.dot"],
            source_excludes=["warp_specialize", "producer", "consumer"],
        ),
        code_pattern=CodePattern(
            before="""\
# Homogeneous warps — all warps do both loads and compute
# Memory and compute pipelines compete for the same resources
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)          # all warps issue loads
    b = tl.load(b_ptrs)          # all warps issue loads
    acc += tl.dot(a, b)          # all warps do math
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * stride_bk""",
            after="""\
# Warp-specialized kernel — producer/consumer split
# Producer warps: issue TMA loads into shared memory
# Consumer warps: execute tl.dot on shared memory buffers
# Triton handles this via num_consumer_groups and num_buffers_warp_spec
@triton.jit
def matmul_ws(a_ptr, b_ptr, c_ptr, M, N, K,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr,
              num_stages: tl.constexpr = 3,
              num_consumer_groups: tl.constexpr = 2,
              num_buffers_warp_spec: tl.constexpr = 1):
    # Triton compiler partitions warps into producers/consumers
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)""",
            explanation=(
                "On Hopper, memory (TMA) and compute (Tensor Core) are "
                "separate hardware pipelines that can run concurrently. In a "
                "homogeneous kernel, each warp alternates between load and "
                "compute instructions, creating pipeline bubbles. Warp "
                "specialization dedicates some warps entirely to issuing TMA "
                "loads (producers) and others entirely to Tensor Core math "
                "(consumers). The two groups communicate through shared memory "
                "buffers and barrier synchronization, achieving near-perfect "
                "overlap of the two pipelines."
            ),
        ),
        expected_impact="1.3-1.8x for memory-bound matmul on Hopper (cc>=9.0)",
        conflicts_with=[],
    ),

    # ── Structural (2) ───────────────────────────────────────────────────

    OptimizationTechnique(
        name="kernel_fusion",
        category="structural",
        description=(
            "Fuse multiple kernel launches into a single kernel to eliminate "
            "intermediate global memory round-trips. Each element is loaded "
            "once, transformed, and stored once."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound", "underutilized"],
            kernel_types=["pointwise"],
            min_compute_capability="",
            source_contains=["tl.load", "tl.store"],
            source_excludes=[],
        ),
        code_pattern=CodePattern(
            before="""\
# Two separate kernels — intermediate result round-trips through DRAM
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

@triton.jit
def scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * scale, mask=mask)
# relu_kernel(x, tmp, n); scale_kernel(tmp, y, s, n)  ← 2 launches""",
            after="""\
# Single fused kernel — no intermediate DRAM write/read
@triton.jit
def fused_relu_scale_kernel(x_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    x = tl.maximum(x, 0.0)   # relu in registers
    x = x * scale             # scale in registers
    tl.store(out_ptr + offs, x, mask=mask)
# One launch, one load, one store""",
            explanation=(
                "Each kernel launch reads inputs from DRAM and writes outputs "
                "back to DRAM. When two kernels are chained (output of kernel 1 "
                "= input of kernel 2), the intermediate tensor must make a full "
                "DRAM round-trip — consuming 2x the bandwidth for zero compute. "
                "Fusion keeps the intermediate value in registers, eliminating "
                "both the write and read. For pointwise operations this can cut "
                "memory traffic in half. Launch overhead (~5-10us per kernel) "
                "is also eliminated."
            ),
        ),
        expected_impact="1.5-3x for memory-bound pointwise chains (bandwidth halved)",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="memory_coalescing",
        category="structural",
        description=(
            "Reorder tile dimensions or loop iteration order so that the "
            "innermost memory access is contiguous. Prevents the memory "
            "controller from issuing multiple transactions per warp request."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound"],
            kernel_types=[],  # any kernel type
            min_compute_capability="",
            source_contains=["tl.load"],
            source_excludes=[],
        ),
        code_pattern=CodePattern(
            before="""\
# Column-major access in a row-major tensor — threads in a warp
# access addresses stride elements apart → N memory transactions
# instead of 1 coalesced transaction
for col in range(0, N, BLOCK_N):
    offs = row_offsets[:, None] * N + col_offsets[None, :]
    # row_offsets changes fastest → non-contiguous in memory
    x = tl.load(x_ptr + offs)""",
            after="""\
# Row-major access matches the tensor layout — threads in a warp
# access consecutive addresses → 1 coalesced transaction
for col in range(0, N, BLOCK_N):
    offs = row_offsets[:, None] * N + col_offsets[None, :]
    # Transpose the tile dims so col (contiguous dim) is innermost
    # Or restructure so that the fast-varying thread index maps to
    # the contiguous dimension of the tensor
    x = tl.load(x_ptr + offs)
    # Key: ensure tl.arange maps to the stride-1 dimension""",
            explanation=(
                "GPU memory is accessed in 32-byte sectors. When all 32 threads "
                "in a warp access consecutive addresses (coalesced), the memory "
                "controller serves them in a single transaction. When threads "
                "access addresses that are stride elements apart (non-coalesced), "
                "each thread's access may hit a different sector, requiring up "
                "to 32 separate transactions — a 32x bandwidth waste. Fixing "
                "coalescing means ensuring the fastest-varying thread index "
                "(threadIdx.x, or the innermost tl.arange) maps to the "
                "stride-1 (contiguous) dimension of the tensor layout."
            ),
        ),
        expected_impact="2-10x for severely uncoalesced access patterns",
        conflicts_with=[],
    ),

    # ── Fusion (3) ─────────────────────────────────────────────────────────

    OptimizationTechnique(
        name="gemm_epilogue_fusion",
        category="structural",
        description=(
            "Fuse matmul epilogue (bias, activation, quantization) into the "
            "GEMM kernel's accumulation loop. After the tl.dot accumulation "
            "finishes, apply the epilogue operations in-register before "
            "storing to global memory, eliminating the intermediate write-back."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound", "compute_bound"],
            kernel_types=["matmul"],
            min_compute_capability="",
            source_contains=["tl.dot"],
            source_excludes=["silu", "gelu", "sigmoid"],  # already has epilogue
        ),
        code_pattern=CodePattern(
            before="""\
# Separate GEMM + epilogue: two kernels, intermediate buffer
# Kernel 1: matmul
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    acc += tl.dot(a, b)
tl.store(c_ptr + ..., acc)  # write full matrix to global memory

# Kernel 2: bias + activation (separate launch)
c = tl.load(c_ptr + ...)    # read it back from global memory
result = tl.maximum(c + bias, 0.0)  # ReLU
tl.store(out_ptr + ..., result)""",
            after="""\
# Fused GEMM + epilogue: one kernel, no intermediate buffer
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    acc += tl.dot(a, b)

# Epilogue: apply bias + activation in-register before storing
bias = tl.load(bias_ptr + cols)
result = tl.maximum(acc + bias[None, :], 0.0)  # fused bias + ReLU
tl.store(out_ptr + ..., result)  # single write to global memory""",
            explanation=(
                "The unfused version writes the full GEMM output to global "
                "memory then reads it back for the epilogue — 2x memory "
                "traffic for the output matrix plus an extra kernel launch. "
                "By computing the epilogue in the same kernel immediately "
                "after the accumulation loop, the result stays in registers "
                "and is written to global memory only once. This pattern "
                "is used by all production GEMM libraries (cuBLAS, cuDNN) "
                "and is the canonical way to fuse bias+activation into matmul."
            ),
        ),
        expected_impact="1.3-2x for matmul kernels with post-processing epilogues",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="online_softmax_attention",
        category="structural",
        description=(
            "Replace naive multi-pass attention (score → softmax → weighted sum) "
            "with online softmax that computes attention in a single streaming "
            "pass over the KV sequence. Maintains running max and normalizer "
            "to avoid materializing the full attention matrix."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["memory_bound", "compute_bound", "underutilized"],
            kernel_types=["attention", "softmax"],
            min_compute_capability="",
            source_contains=["tl.dot", "tl.exp"],
            source_excludes=["correction", "rescale"],  # already uses online softmax
        ),
        code_pattern=CodePattern(
            before="""\
# Naive attention: materializes full N×N score matrix
scores = tl.zeros([BLOCK_M, SEQ_LEN], dtype=tl.float32)
for kv in range(0, SEQ_LEN, BLOCK_N):
    k_tile = tl.load(k_ptr + ...)
    scores[:, kv:kv+BLOCK_N] = tl.dot(q_tile, k_tile)
# Full softmax over materialized scores
max_s = tl.max(scores, axis=1)
exp_s = tl.exp(scores - max_s[:, None])
attn = exp_s / tl.sum(exp_s, axis=1)[:, None]
# Weighted sum
out = tl.dot(attn, v_matrix)  # requires full attn matrix""",
            after="""\
# Online softmax attention: streaming, O(1) memory per row
row_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
for kv in range(0, SEQ_LEN, BLOCK_N):
    k_tile = tl.load(k_ptr + ...)
    v_tile = tl.load(v_ptr + ...)
    scores = tl.dot(q_tile, k_tile)  # [BLOCK_M, BLOCK_N]
    new_max = tl.maximum(row_max, tl.max(scores, axis=1))
    # Rescale running accumulator for updated max
    correction = tl.exp(row_max - new_max)
    row_sum = row_sum * correction + tl.sum(tl.exp(scores - new_max[:, None]), axis=1)
    acc = acc * correction[:, None] + tl.dot(
        tl.exp(scores - new_max[:, None]).to(v_tile.dtype), v_tile
    )
    row_max = new_max
out = acc / row_sum[:, None]""",
            explanation=(
                "Naive attention materializes the full N×N score matrix in "
                "global memory (O(N^2) memory). Online softmax processes KV "
                "tiles in a streaming fashion, maintaining only a running max "
                "and sum per query row (O(1) extra memory). When the max "
                "changes between tiles, the existing accumulator is rescaled "
                "by exp(old_max - new_max), preserving numerical correctness. "
                "This is the core algorithm behind Flash Attention and is "
                "the canonical approach for attention in production models."
            ),
        ),
        expected_impact="2-10x for attention with seq_len > 1024, linear memory savings",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="persistent_tile_scheduling",
        category="parallelism",
        description=(
            "Use a persistent kernel where each SM processes multiple tiles "
            "in a loop instead of launching one CTA per tile. Reduces kernel "
            "launch overhead and improves SM utilization for workloads with "
            "many small tiles (MoE, grouped GEMM, batched small matrices)."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["underutilized", "compute_bound"],
            kernel_types=["matmul", "pointwise"],
            min_compute_capability="",
            source_contains=["tl.program_id"],
            source_excludes=["tl.num_programs", "persistent"],  # already persistent
        ),
        code_pattern=CodePattern(
            before="""\
# Standard: one CTA per tile, grid = (num_tiles,)
pid = tl.program_id(0)
tile_m = pid // num_tiles_n
tile_n = pid % num_tiles_n
# Process single tile
offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
# ... compute and store ...""",
            after="""\
# Persistent: each SM loops over multiple tiles
pid = tl.program_id(0)
num_sms = tl.num_programs(0)  # grid = (num_SMs,)
num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
for tile_id in tl.range(pid, num_tiles, num_sms):
    tile_m = tile_id // tl.cdiv(N, BLOCK_N)
    tile_n = tile_id % tl.cdiv(N, BLOCK_N)
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # ... compute and store (same inner loop) ...""",
            explanation=(
                "When the number of tiles greatly exceeds the SM count (e.g., "
                "batched small GEMMs, MoE with many experts), each kernel "
                "launch has fixed overhead (~5-10 µs). A persistent kernel "
                "launches exactly num_SMs CTAs and each loops over its share "
                "of tiles, eliminating per-tile launch overhead. Each SM "
                "stays busy until all tiles are processed. This pattern is "
                "used by cuDNN's MoE grouped GEMM kernels for maximum "
                "SM utilization across expert dispatches."
            ),
        ),
        expected_impact="1.2-2x for workloads with > 1000 small tiles",
        conflicts_with=[],
    ),

    OptimizationTechnique(
        name="moe_expert_routing",
        category="structural",
        description=(
            "For Mixture of Experts (MoE) workloads, route tokens to "
            "per-expert weight matrices using pointer indirection. Each "
            "program tile computes one expert's GEMM by loading the expert "
            "assignment, resolving the weight pointer, and computing the "
            "tile as a normal GEMM. Combined with persistent scheduling "
            "for SM utilization across variable-size expert batches."
        ),
        when_to_use=TechniqueConditions(
            bottleneck_types=["underutilized", "compute_bound", "memory_bound"],
            kernel_types=["matmul"],
            min_compute_capability="",
            source_contains=["tl.dot"],
            source_excludes=["expert", "routing"],  # already has MoE routing
        ),
        code_pattern=CodePattern(
            before="""\
# Naive MoE: loop over experts, launch one GEMM per expert
for expert_idx in range(num_experts):
    expert_tokens = tokens[assignments == expert_idx]
    if len(expert_tokens) > 0:
        output[assignments == expert_idx] = expert_tokens @ weights[expert_idx]
# Problem: num_experts kernel launches, poor utilization for small expert batches""",
            after="""\
# Fused MoE: persistent kernel routes tokens to experts via pointer indirection
pid = tl.program_id(0)
num_sms = tl.num_programs(0)
# Expert token ranges from cumulative sum (precomputed on CPU)
# offs[i] = cumulative token count through expert i
for tile_id in tl.range(pid, total_tiles, num_sms):
    # Determine which expert this tile belongs to
    expert_idx = _find_expert(tile_id, expert_tile_offsets)
    token_start = tl.load(offs_ptr + expert_idx)
    # Load per-expert weight pointer
    w_ptr = tl.load(weight_ptrs + expert_idx)
    # Standard GEMM tile within this expert's token range
    tile_m = (tile_id - expert_tile_start) // tiles_n
    offs_m = token_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(input_ptr + offs_m[:, None] * K + k + tl.arange(0, BLOCK_K)[None, :])
        b = tl.load(w_ptr + (k + tl.arange(0, BLOCK_K)[:, None]) * N + tl.arange(0, BLOCK_N)[None, :])
        acc += tl.dot(a, b)
    tl.store(out_ptr + offs_m[:, None] * N + tl.arange(0, BLOCK_N)[None, :], acc)""",
            explanation=(
                "Naive MoE launches one GEMM kernel per expert, causing high "
                "launch overhead and poor SM utilization (experts with few "
                "tokens leave SMs idle). A fused persistent kernel processes "
                "ALL experts' tiles in a single launch. Each SM loops over "
                "tiles, determines which expert the tile belongs to using "
                "precomputed cumulative token counts, loads that expert's "
                "weight pointer, and computes the GEMM tile normally. This "
                "pattern is used by cuDNN's grouped GEMM MoE kernels with "
                "per-expert pointer arrays for weight dispatch."
            ),
        ),
        expected_impact="2-5x for MoE with many experts and variable batch sizes",
        conflicts_with=["persistent_tile_scheduling"],  # superset — use one or the other
    ),
]

# Build a name→technique lookup for O(1) access
_TECHNIQUE_INDEX: dict[str, OptimizationTechnique] = {
    t.name: t for t in TECHNIQUE_LIBRARY
}


# ---------------------------------------------------------------------------
# Layer 2: Technique selection
# ---------------------------------------------------------------------------

# Category-to-bottleneck affinity bonuses.
# A memory technique applied to a memory-bound kernel gets a boost.
_CATEGORY_AFFINITY: dict[str, dict[str, float]] = {
    "memory": {
        "memory_bound": 0.5,
        "compute_bound": 0.0,
        "underutilized": 0.2,
        "near_optimal": 0.0,
    },
    "compute": {
        "memory_bound": 0.0,
        "compute_bound": 0.5,
        "underutilized": 0.3,
        "near_optimal": 0.1,
    },
    "parallelism": {
        "memory_bound": 0.2,
        "compute_bound": 0.1,
        "underutilized": 0.5,
        "near_optimal": 0.0,
    },
    "structural": {
        "memory_bound": 0.3,
        "compute_bound": 0.1,
        "underutilized": 0.4,
        "near_optimal": 0.0,
    },
}


def select_techniques(
    profiling_context: dict,
    kernel_source: str,
    max_techniques: int = 3,
) -> list[tuple[OptimizationTechnique, float]]:
    """Deterministically select the top N applicable optimization techniques.

    Selection algorithm:
      1. Extract tier (bottleneck type), compute capability, and kernel type.
      2. Hard-filter each technique on: cc, bottleneck, kernel type,
         source_contains, source_excludes.
      3. Score: base 1.0 + category-bottleneck affinity bonus.
      4. Deduplicate: remove conflicting techniques (keep higher score).
      5. Return top N by score (descending), with ties broken by technique
         order in the library (earlier = higher priority).

    Args:
        profiling_context: Dict with "sections", "compute_capability", etc.
            Follows the structure produced by csv_parser.parse_ncu_csv() and
            stored in ProfilingContext.
        kernel_source: The Triton kernel source code.
        max_techniques: Maximum number of techniques to return.

    Returns:
        List of (OptimizationTechnique, score) tuples, sorted by score desc.
    """
    # -- Step 1: extract classification signals --------------------------------
    tier = _get_ncu_tier(profiling_context)

    cc_raw = profiling_context.get("compute_capability", "")
    if not cc_raw:
        # Fallback: check launch dict (csv_parser stores it as "cc")
        launch = profiling_context.get("launch", {})
        cc_raw = launch.get("cc", "")
    device_cc = _parse_compute_capability(str(cc_raw))

    kernel_type = _classify_kernel(kernel_source)
    source_lower = kernel_source.lower()

    # -- Step 2: hard filter ---------------------------------------------------
    candidates: list[tuple[OptimizationTechnique, float, int]] = []

    for idx, tech in enumerate(TECHNIQUE_LIBRARY):
        cond = tech.when_to_use

        # Check compute capability
        if cond.min_compute_capability:
            required_cc = _parse_compute_capability(cond.min_compute_capability)
            if device_cc > 0 and device_cc < required_cc:
                continue

        # Check bottleneck type
        if tier not in cond.bottleneck_types:
            continue

        # Check kernel type (empty list = matches all)
        if cond.kernel_types and kernel_type not in cond.kernel_types:
            continue

        # Check source_contains (at least one must match)
        if cond.source_contains:
            if not any(pat.lower() in source_lower for pat in cond.source_contains):
                continue

        # Check source_excludes (none must match — already applied)
        if cond.source_excludes:
            if any(pat.lower() in source_lower for pat in cond.source_excludes):
                continue

        # -- Step 3: score -----------------------------------------------------
        score = 1.0
        affinity = _CATEGORY_AFFINITY.get(tech.category, {})
        score += affinity.get(tier, 0.0)

        # TF32 is high-impact for any matmul regardless of bottleneck type
        if tech.name == "tf32_tensor_cores" and kernel_type == "matmul":
            score += 0.8

        candidates.append((tech, score, idx))

    # -- Step 4: deduplicate conflicts -----------------------------------------
    # Sort by score desc, then library order asc (for stable tie-breaking)
    candidates.sort(key=lambda x: (-x[1], x[2]))

    selected: list[tuple[OptimizationTechnique, float]] = []
    excluded_names: set[str] = set()

    for tech, score, _idx in candidates:
        if tech.name in excluded_names:
            continue
        selected.append((tech, score))
        # Mark all techniques this one conflicts with
        for conflict in tech.conflicts_with:
            excluded_names.add(conflict)

    # -- Step 5: top N ---------------------------------------------------------
    return selected[:max_techniques]


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------

def format_techniques_for_executor(
    techniques: list[tuple[OptimizationTechnique, float]],
) -> str:
    """Render selected techniques with before/after code patterns for the executor.

    Produces ~500 tokens of context that the executor LLM can use to apply
    the optimization. Each technique includes the concrete code transformation
    and the hardware rationale.

    Args:
        techniques: Output of select_techniques().

    Returns:
        Markdown-formatted string with code blocks.
    """
    if not techniques:
        return ""

    parts = ["## Applicable Optimization Techniques\n"]

    for i, (tech, score) in enumerate(techniques, 1):
        pat = tech.code_pattern
        parts.append(f"### {i}. {tech.name} (score: {score:.2f})")
        parts.append(f"**Category:** {tech.category} | **Impact:** {tech.expected_impact}")
        parts.append(f"{tech.description}\n")
        parts.append(f"**Before:**\n```python\n{pat.before}\n```\n")
        parts.append(f"**After:**\n```python\n{pat.after}\n```\n")
        parts.append(f"**Why this works:** {pat.explanation}\n")

    return "\n".join(parts)


def format_techniques_for_planner(
    techniques: list[tuple[OptimizationTechnique, float]],
) -> str:
    """Compact technique list for the planner — names, descriptions, impact.

    The planner needs to know what is available and how much it helps, but
    does NOT need the full code patterns (those go to the executor).

    Args:
        techniques: Output of select_techniques().

    Returns:
        Compact markdown list.
    """
    if not techniques:
        return ""

    lines = ["## Candidate Techniques\n"]
    for i, (tech, score) in enumerate(techniques, 1):
        lines.append(
            f"{i}. **{tech.name}** ({tech.category}) — {tech.description} "
            f"Expected: {tech.expected_impact}. Score: {score:.2f}"
        )
        if tech.conflicts_with:
            lines.append(f"   Conflicts with: {', '.join(tech.conflicts_with)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_technique_by_name(name: str) -> OptimizationTechnique | None:
    """Look up a technique by its unique name. Returns None if not found."""
    return _TECHNIQUE_INDEX.get(name)


def get_all_technique_names() -> list[str]:
    """Return all technique names in library order."""
    return [t.name for t in TECHNIQUE_LIBRARY]
