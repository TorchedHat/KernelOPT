# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3330 ns — **Best latency**: 2590 ns (**1.2857×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_triton_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_7_triton_poi_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 28
- **Duration**: 3330 ns
- **SM Throughput**: 16.35 % | **Memory**: 15.48 % | **DRAM**: 1.85 %
- **Achieved Occupancy**: 36.70 %
- **IPC Active**: 1.61 inst/cycle | **No Eligible**: 55.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_xblock_yblock_with_correct_masking | 1.038× | 1.074× | ✓ |  |
| D | flatten_and_autotune_layout_permutation_kernel | 1.050× | 1.156× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_xblock_yblock... | Δ | flatten_and_autot... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 3100 ns | -230 ns (-6.9%) | 2880 ns | -450 ns (-13.5%) |
| SM Throughput (%) | 16.35 % | 16.67 % | +0.32 pp | 22.64 % | +6.29 pp |
| Memory Throughput (%) | 15.48 % | 15.68 % | +0.20 pp | 15.39 % | -0.09 pp |
| DRAM Throughput (%) | 1.85 % | 2.00 % | +0.15 pp | 2.14 % | +0.29 pp |
| Achieved Occupancy (%) | 36.70 % | 36.51 % | -0.19 pp | 39.78 % | +3.08 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 27.00 register/thread | -1 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 55.49 % | 64.02 % | +8.53 pp | 37.87 % | -17.62 pp |
| IPC Active | 1.61 inst/cycle | 1.66 inst/cycle | +0.05 | 2.03 inst/cycle | +0.42 |

---

## Iteration 3

**Best so far**: 1.1562× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | correct_grid_launch_with_small_xblock_large_yblock | 1.135× | 1.354× | ✓ |  |
| B | fix_grid_launch_to_use_autotuned_tile_blocks | 1.092× | 1.271× | ✓ |  |
| D | autotune_extended_configs_small_x_layout_permutation | 1.140× | 1.286× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | correct_grid_laun... | Δ | fix_grid_launch_t... | Δ | autotune_extended... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 2460 ns | -870 ns (-26.1%) | 2620 ns | -710 ns (-21.3%) | 2590 ns | -740 ns (-22.2%) |
| SM Throughput (%) | 16.35 % | 3.46 % | -12.89 pp | 3.45 % | -12.90 pp | 3.24 % | -13.11 pp |
| Memory Throughput (%) | 15.48 % | 2.50 % | -12.98 pp | 2.35 % | -13.13 pp | 2.39 % | -13.09 pp |
| DRAM Throughput (%) | 1.85 % | 2.50 % | +0.65 pp | 2.35 % | +0.50 pp | 2.39 % | +0.54 pp |
| Achieved Occupancy (%) | 36.70 % | 10.87 % | -25.83 pp | 16.16 % | -20.54 pp | 10.02 % | -26.68 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.24 | -0.24 | 0.24 | -0.24 |
| No Eligible (%) | 55.49 % | 86.10 % | +30.61 pp | 86.28 % | +30.79 pp | 83.32 % | +27.83 pp |
| IPC Active | 1.61 inst/cycle | 0.40 inst/cycle | -1.21 | 0.32 inst/cycle | -1.29 | 0.39 inst/cycle | -1.22 |

---

## Iteration 4

**Best so far**: 1.2857× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_correct_grid_launch | 0.042× | 0.018× | ✓ |  |
| B | persistent_tile_scheduling_via_correct_autotune_grid_launch | 1.083× | 1.286× | ✓ |  |
| C | persistent_tile_scheduling_for_layout_permutation | 0.033× | 0.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 180510 ns | +177180 ns (+5320.7%) | 2590 ns | -740 ns (-22.2%) | 458210 ns | +454880 ns (+13660.1%) |
| SM Throughput (%) | 16.35 % | 31.05 % | +14.70 pp | 3.00 % | -13.35 pp | 16.14 % | -0.21 pp |
| Memory Throughput (%) | 15.48 % | 45.01 % | +29.53 pp | 2.35 % | -13.13 pp | 25.97 % | +10.49 pp |
| DRAM Throughput (%) | 1.85 % | 0.05 % | -1.80 pp | 2.35 % | +0.50 pp | 0.01 % | -1.84 pp |
| Achieved Occupancy (%) | 36.70 % | 24.23 % | -12.47 pp | 10.85 % | -25.85 pp | 24.23 % | -12.47 pp |
| Registers/Thread | 28.00 register/thread | 31.00 register/thread | +3 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.24 | -0.24 | 0.24 | -0.24 |
| No Eligible (%) | 55.49 % | 68.03 % | +12.54 pp | 87.47 % | +31.98 pp | 83.81 % | +28.32 pp |
| IPC Active | 1.61 inst/cycle | 1.29 inst/cycle | -0.32 | 0.40 inst/cycle | -1.21 | 0.66 inst/cycle | -0.95 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_correct_grid_launch` (0.018×)

Converting a simple 2D one-CTA-per-tile Triton kernel into a persistent while-loop scheduler (each of num_sm CTAs loops over all tiles) caused a 54x slowdown: kernel duration jumped from 2.46 µs to 180.51 µs and executed instructions exploded from 62K to 45.7M. The persistent loop introduced a dynamic while-loop with runtime-computed tile counts (tl.cdiv inside the kernel), which Triton cannot unroll or optimize — each iteration requires recomputing tile indices, reloading tl.arange offsets, and re-evaluating masks, turning a trivially vectorized single-pass kernel into a serial loop. Additionally, registers per thread nearly doubled (16→31) and dynamic shared memory was allocated (0→2KB/block), further reducing occupancy headroom. The anti-pattern is applying persistent scheduling to kernels that are already small enough to complete in a single pass — persistent loops only help when the kernel is genuinely compute-bound and the loop body amortizes significant reuse; for pure memory-copy/transpose kernels with tiny total work, the loop overhead dominates.

> **Applicable when**: Never apply persistent while-loop tile scheduling to small, short-duration Triton kernels (< 10 µs) that perform a single pass over data with no reuse. The loop overhead and register pressure increase will dominate.

> **Avoid when**: Do not apply persistent scheduling when: (1) the kernel is a pure memory-copy or layout-permutation with no compute reuse, (2) total kernel duration is already < 10 µs, (3) the kernel body is trivially vectorized in a single pass, (4) tl.cdiv and modulo/division operations would be computed inside the loop body at runtime (not compile-time constexpr). Persistent scheduling is only beneficial for large compute-bound kernels (e.g., GEMM, attention) where the loop body amortizes significant shared memory reuse.

#### ❌ Plan C: `persistent_tile_scheduling_for_layout_permutation` (0.007×)

Converting a 2D Triton kernel to a persistent 1D loop (using tl.range(pid, num_tiles, num_programs) inside the kernel body while keeping the same grid shape) caused a catastrophic 137x slowdown: elapsed cycles exploded from 3,647 to 686,583 and executed instructions grew from 62K to 59M. The root cause is that the grid was NOT changed to min(132, num_tiles) CTAs — it remained at 1,024 CTAs, each now iterating over ~144/1024 ≈ 0.14 tiles, meaning most CTAs do nearly zero useful work while still paying the loop overhead. Additionally, the persistent loop introduced a CPIStall on L1TEX (78.4% of stall cycles) because the sequential flat-index decomposition (integer division/modulo inside the loop) serializes memory access patterns, destroying the coalescing that the original 2D tiled layout provided. A persistent kernel strategy only helps when the grid is simultaneously reduced to match SM count; applying the loop without fixing the grid is strictly worse.

> **Applicable when**: Never apply this pattern (persistent loop without grid reduction). This is a pure anti-pattern.

> **Avoid when**: Always avoid adding a persistent tl.range loop inside a Triton kernel without simultaneously reducing the grid to min(SM_count, num_tiles). If the grid remains large (e.g., 1024 CTAs) while each CTA iterates over <1 tile on average, the overhead of the loop and the integer division/modulo index reconstruction dominates, causing orders-of-magnitude slowdown. The CPIStall on L1TEX (78% of stall cycles) confirms that flat-index decomposition via integer division inside a hot loop serializes memory access and destroys coalescing.

---

## Iteration 5

**Best so far**: 1.2857× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_lambda_grid_with_num_stages_pipelining | 1.123× | 1.156× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_with_small_xblock_configs | 1.124× | 1.316× | ✓ |  |
| D | add_triton_autotune | 1.093× | 1.286× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_lambda_g... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 2880 ns | -450 ns (-13.5%) | 2530 ns | -800 ns (-24.0%) | 2590 ns | -740 ns (-22.2%) |
| SM Throughput (%) | 16.35 % | 1.60 % | -14.75 pp | 3.52 % | -12.83 pp | 3.04 % | -13.31 pp |
| Memory Throughput (%) | 15.48 % | 2.13 % | -13.35 pp | 2.44 % | -13.04 pp | 2.35 % | -13.13 pp |
| DRAM Throughput (%) | 1.85 % | 2.13 % | +0.28 pp | 2.44 % | +0.59 pp | 2.35 % | +0.50 pp |
| Achieved Occupancy (%) | 36.70 % | 6.15 % | -30.55 pp | 11.45 % | -25.25 pp | 10.25 % | -26.45 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.24 | -0.24 | 0.24 | -0.24 | 0.24 | -0.24 |
| No Eligible (%) | 55.49 % | 90.20 % | +34.71 pp | 88.90 % | +33.41 pp | 86.72 % | +31.23 pp |
| IPC Active | 1.61 inst/cycle | 0.17 inst/cycle | -1.44 | 0.43 inst/cycle | -1.18 | 0.39 inst/cycle | -1.22 |

---

## Cumulative Progress

```
Baseline:  3330 ns  (1.000×)
Iter 2     2880 ns  (1.156×)  — flatten_and_autotune_layout_permutation_kernel
Iter 3     2590 ns  (1.286×)  — autotune_extended_configs_small_x_layout_permutation

Final best: 2590 ns  (1.2857×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +37,8 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +46,7 @@

     ynumel = 8192
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
