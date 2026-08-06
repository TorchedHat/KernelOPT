# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_clone_transpose_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22940 ns — **Best latency**: 22590 ns (**1.0155×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_k_dimension_into_flat_tile_eliminate_static_range_loop`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_clone_transpose_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_priority/kernel_031/kernel_8_triton_poi_fused_clone_transpo/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32768,1,1), Waves/SM: 15.52
- **Registers/thread**: 16
- **Duration**: 22940 ns
- **SM Throughput**: 20.74 % | **Memory**: 33.23 % | **DRAM**: 33.23 %
- **Achieved Occupancy**: 45.99 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.79 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_transpose | 0.999× | 1.016× | ✓ |  |
| B | vectorized_loads_with_larger_xblock | 0.998× | 0.868× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_latency_hiding | 0.414× | 1.016× | ✓ |  |
| D | increase_xblock_tile_size_with_autotune | 0.891× | 0.918× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22940 ns | 22590 ns | -350 ns (-1.5%) | 26430 ns | +3490 ns (+15.2%) | 22590 ns | -350 ns (-1.5%) | 24990 ns | +2050 ns (+8.9%) |
| SM Throughput (%) | 20.74 % | 26.67 % | +5.93 pp | 43.03 % | +22.29 pp | 25.03 % | +4.29 pp | 35.27 % | +14.53 pp |
| Memory Throughput (%) | 33.23 % | 18.80 % | -14.43 pp | 16.99 % | -16.24 pp | 18.81 % | -14.42 pp | 16.49 % | -16.74 pp |
| DRAM Throughput (%) | 33.23 % | 15.80 % | -17.43 pp | 13.77 % | -19.46 pp | 15.98 % | -17.25 pp | 14.30 % | -18.93 pp |
| Achieved Occupancy (%) | 45.99 % | 38.36 % | -7.63 pp | 38.12 % | -7.87 pp | 38.07 % | -7.92 pp | 32.57 % | -13.42 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 31.03 | +15.51 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 76.79 % | 70.18 % | -6.61 pp | 57.84 % | -18.95 pp | 71.69 % | -5.10 pp | 61.41 % | -15.38 pp |
| IPC Active | 0.92 inst/cycle | 1.17 inst/cycle | +0.25 | 1.80 inst/cycle | +0.88 | 1.12 inst/cycle | +0.20 | 1.50 inst/cycle | +0.58 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_load_evict_last_for_l2_reuse | 0.996× | 1.014× | ✓ |  |
| B | eviction_policies_streaming_load_evict_first_store_evict_last | 1.006× | 1.003× | ✓ | **✓ best** |
| C | eviction_policies_streaming_load_evict_first | 0.883× | 0.918× | ✓ |  |
| D | eviction_policies_load_evict_last_store_evict_first | 0.997× | 0.878× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_l... | Δ | eviction_policies... | Δ | eviction_policies... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22940 ns | 22620 ns | -320 ns (-1.4%) | 22880 ns | -60 ns (-0.3%) | 24990 ns | +2050 ns (+8.9%) | 26140 ns | +3200 ns (+13.9%) |
| SM Throughput (%) | 20.74 % | 30.91 % | +10.17 pp | 30.94 % | +10.20 pp | 38.26 % | +17.52 pp | 47.23 % | +26.49 pp |
| Memory Throughput (%) | 33.23 % | 18.80 % | -14.43 pp | 18.65 % | -14.58 pp | 16.55 % | -16.68 pp | 16.21 % | -17.02 pp |
| DRAM Throughput (%) | 33.23 % | 15.80 % | -17.43 pp | 15.68 % | -17.55 pp | 14.34 % | -18.89 pp | 13.79 % | -19.44 pp |
| Achieved Occupancy (%) | 45.99 % | 37.70 % | -8.29 pp | 38.41 % | -7.58 pp | 32.39 % | -13.60 pp | 39.61 % | -6.38 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 18.00 register/thread | +2 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 | 31.03 | +15.51 |
| No Eligible (%) | 76.79 % | 65.18 % | -11.61 pp | 65.18 % | -11.61 pp | 54.00 % | -22.79 pp | 45.50 % | -31.29 pp |
| IPC Active | 0.92 inst/cycle | 1.38 inst/cycle | +0.46 | 1.37 inst/cycle | +0.45 | 1.80 inst/cycle | +0.88 | 2.12 inst/cycle | +1.20 |

---

## Iteration 3

**Best so far**: 1.0026× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_tiled_transpose_for_coalesced_access | 0.087× | 0.081× | ✓ |  |
| B | fix_autotune_grid_to_use_meta_xblock | 1.008× | 1.014× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 22940 ns | 283650 ns | +260710 ns (+1136.5%) | 22620 ns | -320 ns (-1.4%) |
| SM Throughput (%) | 20.74 % | 59.69 % | +38.95 pp | 30.95 % | +10.21 pp |
| Memory Throughput (%) | 33.23 % | 82.55 % | +49.32 pp | 18.85 % | -14.38 pp |
| DRAM Throughput (%) | 33.23 % | 3.07 % | -30.16 pp | 15.81 % | -17.42 pp |
| Achieved Occupancy (%) | 45.99 % | 97.51 % | +51.52 pp | 38.17 % | -7.82 pp |
| Registers/Thread | 16.00 register/thread | 30.00 register/thread | +14 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 76.79 % | 74.66 % | -2.13 pp | 65.18 % | -11.61 pp |
| IPC Active | 0.92 inst/cycle | 1.01 inst/cycle | +0.09 | 1.37 inst/cycle | +0.45 |


### Key Learnings

#### ❌ Plan A: `shared_memory_tiled_transpose_for_coalesced_access` (0.081×)

Replacing a flat 1D transpose kernel with a 2D tiled kernel that unrolls K=32 via tl.static_range caused a 12x regression (22.6us → 283.65us). The static_range(32) unroll exploded instruction count from 5.5M to 56.4M (10x), doubled register pressure (16→30 registers/thread), and introduced MIO queue stalls (23.6 cycles/warp, 38.5% of total stall cycles) due to excessive shared memory and L1TEX traffic. Although occupancy improved from 37.7% to 97.5% and L1 hit rate rose from 0% to 47%, the massive instruction count increase and MIO pipeline saturation completely overwhelmed any coalescing benefit. When a transpose kernel has a small K dimension that is fully unrolled at compile time, the instruction explosion from static_range unrolling dominates all other effects.

> **Applicable when**: Never: this pattern (static_range unroll over a small K dimension in a 2D tiled transpose) causes catastrophic instruction count explosion regardless of the coalescing improvement.

> **Avoid when**: Any transpose kernel where K is a compile-time constant and tl.static_range would be used to unroll it. The instruction count explosion (10x in this case) and MIO queue saturation from 32 unrolled load+store pairs per thread completely negates coalescing gains. Also avoid when register pressure increase (16→30 regs/thread) would reduce Block Limit Registers occupancy ceiling. The correct fix for strided transpose access is to use a single vectorized 2D tile load without K-loop unrolling, or to fuse K into the tile dimensions.

---

## Iteration 4

**Best so far**: 1.0141× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fuse_k_dimension_into_tile_eliminate_static_range_loop | 1.002× | 1.008× | ✓ |  |
| D | fuse_k_dimension_into_flat_tile_eliminate_static_range_loop | 1.010× | 1.016× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.006× | 1.017× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_k_dimension_... | Δ | fuse_k_dimension_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22940 ns | 22750 ns | -190 ns (-0.8%) | 22590 ns | -350 ns (-1.5%) | 22560 ns | -380 ns (-1.7%) |
| SM Throughput (%) | 20.74 % | 25.82 % | +5.08 pp | 25.81 % | +5.07 pp | 30.91 % | +10.17 pp |
| Memory Throughput (%) | 33.23 % | 18.72 % | -14.51 pp | 18.88 % | -14.35 pp | 18.89 % | -14.34 pp |
| DRAM Throughput (%) | 33.23 % | 15.83 % | -17.40 pp | 15.96 % | -17.27 pp | 15.82 % | -17.41 pp |
| Achieved Occupancy (%) | 45.99 % | 37.84 % | -8.15 pp | 38.22 % | -7.77 pp | 37.62 % | -8.37 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 | 15.52 | +0.00 | 15.52 | +0.00 |
| No Eligible (%) | 76.79 % | 70.94 % | -5.85 pp | 71.13 % | -5.66 pp | 65.17 % | -11.62 pp |
| IPC Active | 0.92 inst/cycle | 1.15 inst/cycle | +0.23 | 1.15 inst/cycle | +0.23 | 1.38 inst/cycle | +0.46 |

---

## Iteration 5

**Best so far**: 1.0155× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_triton_autotune | 0.996× | 1.012× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 22940 ns | 22660 ns | -280 ns (-1.2%) |
| SM Throughput (%) | 20.74 % | 30.79 % | +10.05 pp |
| Memory Throughput (%) | 33.23 % | 18.74 % | -14.49 pp |
| DRAM Throughput (%) | 33.23 % | 15.85 % | -17.38 pp |
| Achieved Occupancy (%) | 45.99 % | 37.83 % | -8.16 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 15.52 | 15.52 | +0.00 |
| No Eligible (%) | 76.79 % | 65.27 % | -11.52 pp |
| IPC Active | 0.92 inst/cycle | 1.38 inst/cycle | +0.46 |

---

## Cumulative Progress

```
Baseline:  22940 ns  (1.000×)
Iter 2     22880 ns  (1.003×)  — eviction_policies_streaming_load_evict_first_store_evict_last
Iter 3     22620 ns  (1.014×)  — fix_autotune_grid_to_use_meta_xblock
Iter 4     22590 ns  (1.016×)  — fuse_k_dimension_into_flat_tile_eliminate_static_range_loop

Final best: 22590 ns  (1.0155×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,27 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_clone_transpose_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4194304
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
     x1 = ((xindex // 32) % 8)
     x2 = xindex // 256
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), None)
-    tl.store(out_ptr0 + (x3), tmp0, None)
-
-
+    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 524288*x1), xmask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (x3), tmp0, xmask, eviction_policy='evict_last')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 4194304
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304, XBLOCK=128)
+    triton_poi_fused_clone_transpose_8[grid](arg_0, arg_1, 4194304)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
