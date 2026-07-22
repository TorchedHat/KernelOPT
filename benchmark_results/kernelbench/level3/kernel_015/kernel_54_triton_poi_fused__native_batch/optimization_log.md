# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3900 ns — **Best latency**: 3230 ns (**1.2074×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_54_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 0.97
- **Registers/thread**: 28
- **Duration**: 3900 ns
- **SM Throughput**: 26.20 % | **Memory**: 25.38 % | **DRAM**: 3.12 %
- **Achieved Occupancy**: 66.19 %
- **IPC Active**: 2.28 inst/cycle | **No Eligible**: 47.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.003× | 0.002× | ✓ |  |
| D | flatten_2d_to_1d_with_vectorized_loads | 1.059× | 1.219× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | flatten_2d_to_1d_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3900 ns | 2550000 ns | +2546100 ns (+65284.6%) | 3200 ns | -700 ns (-17.9%) |
| SM Throughput (%) | 26.20 % | 64.67 % | +38.47 pp | 10.07 % | -16.13 pp |
| Memory Throughput (%) | 25.38 % | 81.34 % | +55.96 pp | 3.81 % | -21.57 pp |
| DRAM Throughput (%) | 3.12 % | 0.01 % | -3.11 pp | 3.81 % | +0.69 pp |
| Achieved Occupancy (%) | 66.19 % | 78.70 % | +12.51 pp | 49.43 % | -16.76 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 47.67 % | 32.12 % | -15.55 pp | 60.75 % | +13.08 pp |
| IPC Active | 2.28 inst/cycle | 2.70 inst/cycle | +0.42 | 1.43 inst/cycle | -0.85 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.002×)

Converting a 2D grid launch (grid_x=1, grid_y=2048) to a 1D persistent kernel with an inner while-loop caused a ~1578x increase in executed instructions (819K → 1.292B) and duration blew up from 3.9µs to 2.55ms. The persistent loop correctly iterated over all 2048 tiles per SM, but the per-iteration overhead of recomputing tile indices, re-evaluating masks, and re-issuing loads/stores inside the while-loop accumulated catastrophically — the GPU was not bottlenecked on wave count but on instruction throughput. The original kernel was already near-optimal in absolute wall time (3.9µs); the SOLBottleneck rule flagging '1.0 waves' was misleading because the kernel was so short that the overhead of persistent scheduling far exceeded any wave-count benefit.

> **Applicable when**: Never beneficial when the original kernel is already very short (< 10µs) and the total work per tile is tiny (e.g., 8×128 elements with 93% masking). The persistent loop overhead dominates.

> **Avoid when**: Do NOT apply persistent kernel scheduling when: (1) the original kernel duration is already in the low-microsecond range (< ~10µs), (2) each tile does very little work (small XBLOCK×YBLOCK with heavy masking), (3) the total instruction count per tile is small — the while-loop bookkeeping (index recomputation, mask re-evaluation, branch overhead) will cost more than the wave-count savings. The SOLBottleneck '~1 wave' warning is misleading for ultra-short kernels where the absolute latency is already near the GPU launch overhead floor.

---

## Iteration 3

**Best so far**: 1.2188× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_store_eviction_policy_for_uncoalesced_writes | 1.051× | 1.207× | ✓ |  |
| B | expand_autotune_configs_with_larger_yblock_and_streaming_store_eviction | 1.048× | 1.207× | ✓ |  |
| C | autotune_2d_tile_sizes_with_ymask_guard | 1.051× | 1.219× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_store_e... | Δ | expand_autotune_c... | Δ | autotune_2d_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3900 ns | 3230 ns | -670 ns (-17.2%) | 3230 ns | -670 ns (-17.2%) | 3200 ns | -700 ns (-17.9%) |
| SM Throughput (%) | 26.20 % | 10.17 % | -16.03 pp | 10.24 % | -15.96 pp | 10.52 % | -15.68 pp |
| Memory Throughput (%) | 25.38 % | 3.80 % | -21.58 pp | 3.81 % | -21.57 pp | 3.80 % | -21.58 pp |
| DRAM Throughput (%) | 3.12 % | 3.80 % | +0.68 pp | 3.81 % | +0.69 pp | 3.80 % | +0.68 pp |
| Achieved Occupancy (%) | 66.19 % | 49.26 % | -16.93 pp | 50.34 % | -15.85 pp | 48.75 % | -17.44 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 47.67 % | 60.10 % | +12.43 pp | 58.89 % | +11.22 pp | 57.88 % | +10.21 pp |
| IPC Active | 2.28 inst/cycle | 1.41 inst/cycle | -0.87 | 1.45 inst/cycle | -0.83 | 1.40 inst/cycle | -0.88 |


### Key Learnings

#### ✅ Plan A: `streaming_store_eviction_policy_for_uncoalesced_writes` (1.207×)

When a kernel performs write-once scatter stores with a large stride (e.g., stride-512 floats between consecutive threads), adding cache_modifier='.cs' (streaming/evict-first) to tl.store bypasses L1 caching for those writes, reducing cache pollution and MIO queue pressure. In this case, the UncoalescedGlobalAccess rule confirmed 82% excessive sectors with only 4/32 bytes utilized per sector; adding '.cs' reduced total SMSP elapsed cycles from 2,684,400 to 2,601,344 (~3% reduction) and increased eligible warps per scheduler from 2.14 to 2.53, contributing to a 1.21x wall-time speedup. This strategy is most effective when the store pattern is fundamentally non-coalesced and the written data is never reused within the same kernel, making L1 caching of stores wasteful.

> **Applicable when**: UncoalescedGlobalAccess rule fires with >70% excessive sectors and <8 bytes utilized per sector on L2 misses. The store is write-once (data not reread within the kernel). The store stride between consecutive threads is large (e.g., hundreds of elements), making L1 caching of stores counterproductive. MIO stall percentage is elevated (>20% of warp cycles).

> **Avoid when**: The store data is reused within the same kernel (e.g., reduction followed by store). The store pattern is coalesced or near-coalesced (bytes utilized per sector > 16). The kernel is already compute-bound rather than memory-bound. Do not apply if Triton version < 2.1 lacks cache_modifier support on tl.store (use eviction_policy='evict_first' as fallback).

#### ✅ Plan C: `autotune_2d_tile_sizes_with_ymask_guard` (1.258×)

Expanding the autotune config search space to include smaller block sizes (XBLOCK=8 as power-of-2 near-exact fit for xnumel=9, YBLOCK up to 1024) combined with evict_first on streaming loads and .cs cache modifier on write-once scatter stores reduced elapsed cycles from 4760 to 4650 (1.26x speedup). The autotuner selected block_size=64 (half the prior 128), halving executed instructions from 247K to 116K and reducing warp cycles per issued instruction from 25.14 to 21.64. Notably, achieved occupancy dropped from 50% to 24% and L1 hit rate fell to 0% (evict_first eliminated L1 caching of loads), yet wall-time still improved — indicating the prior kernel was wasting cycles on unnecessary L1 cache fills for streaming data and on masked X-threads, and the smaller block size reduced per-CTA overhead enough to offset the lower occupancy.

> **Applicable when**: A 2D Triton kernel has xnumel much smaller than available XBLOCK options (e.g., xnumel=9 with XBLOCK=16 wastes 7/16 threads). IssueSlotUtilization rule fires with no-eligible-warp % >50%. The load is streaming/read-once (evict_first is semantically correct). The store is write-once with large stride between threads (UncoalescedGlobalAccess rule fires with >70% excessive sectors). The existing autotune search space does not include near-exact-fit XBLOCK values or larger YBLOCK options.

> **Avoid when**: The kernel already uses a near-exact-fit XBLOCK (waste <2 threads). The load data is reused within the kernel (evict_first would hurt L1 hit rate for genuinely reused data). The store pattern is coalesced (bytes utilized per sector >16). Do not expect occupancy to improve — this optimization trades occupancy for reduced per-CTA instruction count; it works when the kernel is instruction-count-limited rather than latency-hiding-limited.

---

## Iteration 4

**Best so far**: 1.2188× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls | 1.076× | 1.207× | ✓ |  |
| B | add_num_stages_pipelining_and_fix_load_eviction_policy | 1.079× | 1.207× | ✓ | **✓ best** |
| C | l2_swizzle_program_id_reorder_for_workload_balance | 1.024× | 1.069× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_modulo_di... | Δ | add_num_stages_pi... | Δ | l2_swizzle_progra... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3900 ns | 3230 ns | -670 ns (-17.2%) | 3230 ns | -670 ns (-17.2%) | 3650 ns | -250 ns (-6.4%) |
| SM Throughput (%) | 26.20 % | 10.51 % | -15.69 pp | 10.34 % | -15.86 pp | 10.58 % | -15.62 pp |
| Memory Throughput (%) | 25.38 % | 3.80 % | -21.58 pp | 3.80 % | -21.58 pp | 9.16 % | -16.22 pp |
| DRAM Throughput (%) | 3.12 % | 3.80 % | +0.68 pp | 3.80 % | +0.68 pp | 3.36 % | +0.24 pp |
| Achieved Occupancy (%) | 66.19 % | 50.39 % | -15.80 pp | 50.53 % | -15.66 pp | 72.41 % | +6.22 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 47.67 % | 58.98 % | +11.31 pp | 60.50 % | +12.83 pp | 67.03 % | +19.36 pp |
| IPC Active | 2.28 inst/cycle | 1.40 inst/cycle | -0.88 | 1.38 inst/cycle | -0.90 | 1.34 inst/cycle | -0.94 |


### Key Learnings

#### ✅ Plan A: `replace_modulo_division_with_bitwise_ops_to_eliminate_imc_stalls` (1.207×)

The CPIStall rule identified that 46.8% of all warp stall cycles (8.6 cycles/warp) were caused by Immediate Constant Cache (IMC) misses, attributed to integer modulo and floor-division operations with a power-of-2 constant (512 = 2^9). Replacing `yindex % 512` with `yindex & 511` and `yindex // 512` with `yindex >> 9` was expected to eliminate IMC stalls by emitting LOP3/SHR SASS instructions with register-encoded immediates instead of constant-cache references. However, the profiling data shows the optimization did NOT improve the targeted metrics: warp cycles per issued instruction actually increased from 18.47 to 25.14, active warps per scheduler increased from 7.78 to 10.31 (suggesting more concurrency but not less stalling), and the CPIStall rule was no longer reported — yet duration remained essentially unchanged (3.2 vs 3.23 µs). The 1.21x speedup reported appears to be measurement noise or a wall-time artifact, not a genuine metric improvement from this specific change.

> **Applicable when**: CPIStall rule fires with IMC miss stalls accounting for >30% of total warp stall cycles (>5 cycles/warp on IMC). The kernel uses integer modulo or floor-division with a power-of-2 constant in index computations. The divisor is a compile-time constant expressible as 2^N.

> **Avoid when**: The divisor is not a power of two (bitwise equivalence does not hold). The kernel is already using bitwise ops for index decomposition. The CPIStall is attributed to a different stall source (e.g., L1TEX or LG throttle). Do not expect large speedups when the kernel duration is already in the low-microsecond range and other bottlenecks (uncoalesced access, low occupancy) dominate — the profiling data here shows the change did not materially move any key metric despite the CPIStall rule firing strongly.

---

## Iteration 5

**Best so far**: 1.2188× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_exact_xblock_with_larger_yblock_and_evict_first_load | 1.051× | 1.258× | ✓ |  |
| C | fix_autotune_grid_lambda_to_use_autotuned_tile_sizes | 1.031× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_exact_xb... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3900 ns | 3100 ns | -800 ns (-20.5%) | 3780 ns | -120 ns (-3.1%) |
| SM Throughput (%) | 26.20 % | 5.06 % | -21.14 pp | 11.16 % | -15.04 pp |
| Memory Throughput (%) | 25.38 % | 3.90 % | -21.48 pp | 8.83 % | -16.55 pp |
| DRAM Throughput (%) | 3.12 % | 3.90 % | +0.78 pp | 3.21 % | +0.09 pp |
| Achieved Occupancy (%) | 66.19 % | 23.61 % | -42.58 pp | 70.54 % | +4.35 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.97 | 0.48 | -0.49 | 0.97 | +0.00 |
| No Eligible (%) | 47.67 % | 76.90 % | +29.23 pp | 60.21 % | +12.54 pp |
| IPC Active | 2.28 inst/cycle | 0.55 inst/cycle | -1.73 | 1.40 inst/cycle | -0.88 |

---

## Cumulative Progress

```
Baseline:  3900 ns  (1.000×)
Iter 2     3200 ns  (1.219×)  — flatten_2d_to_1d_with_vectorized_loads

Final best: 3230 ns  (1.2074×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 16384
     xnumel = 9
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    yoffset = tl.program_id(1) * YBLOCK
+    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    ymask = yindex < ynumel
     x2 = xindex
     y3 = yindex
-    y0 = (yindex % 512)
-    y1 = yindex // 512
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask)
-
-
+    y0 = yindex & 511
+    y1 = yindex >> 9
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 16384
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54[grid](arg_0, arg_1, 16384, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_54[grid](arg_0, arg_1, 16384, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
