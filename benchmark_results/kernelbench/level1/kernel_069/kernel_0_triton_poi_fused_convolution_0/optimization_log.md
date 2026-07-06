# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5890 ns — **Best latency**: 3300 ns (**1.7848×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`vectorized_loads_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5890 ns
- **SM Throughput**: 16.51 % | **Memory**: 42.39 % | **DRAM**: 20.97 %
- **Achieved Occupancy**: 54.91 %
- **IPC Active**: 1.15 inst/cycle | **No Eligible**: 70.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.413× | 1.785× | ✓ | **✓ best** |
| B | autotune_tile_sizes_fix_uncoalesced_stores | 1.283× | 1.451× | ✓ |  |
| C | memory_coalescing_via_tile_dimension_reorder | 1.251× | 0.555× | ✓ |  |
| D | autotune_tile_sizes_and_improve_coalescing | 1.086× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 3300 ns | -2590 ns (-44.0%) | 4060 ns | -1830 ns (-31.1%) | 10620 ns | +4730 ns (+80.3%) | 6050 ns | +160 ns (+2.7%) |
| SM Throughput (%) | 16.51 % | 17.57 % | +1.06 pp | 17.55 % | +1.04 pp | 32.20 % | +15.69 pp | 17.42 % | +0.91 pp |
| Memory Throughput (%) | 42.39 % | 20.83 % | -21.56 pp | 28.36 % | -14.03 pp | 37.36 % | -5.03 pp | 41.26 % | -1.13 pp |
| DRAM Throughput (%) | 20.97 % | 5.04 % | -15.93 pp | 15.26 % | -5.71 pp | 11.57 % | -9.40 pp | 20.43 % | -0.54 pp |
| Achieved Occupancy (%) | 54.91 % | 44.96 % | -9.95 pp | 57.51 % | +2.60 pp | 57.82 % | +2.91 pp | 56.04 % | +1.13 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 21.00 register/thread | -5 | 40.00 register/thread | +14 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.99 | +0.25 | 0.74 | +0.00 |
| No Eligible (%) | 70.77 % | 55.41 % | -15.36 pp | 70.77 % | +0.00 pp | 49.27 % | -21.50 pp | 71.92 % | +1.15 pp |
| IPC Active | 1.15 inst/cycle | 1.64 inst/cycle | +0.49 | 1.37 inst/cycle | +0.22 | 2.01 inst/cycle | +0.86 | 1.09 inst/cycle | -0.06 |


### Key Learnings

#### ❌ Plan C: `memory_coalescing_via_tile_dimension_reorder` (0.555×)

Swapping the program_id axis assignment to make the stride-1 output dimension innermost (column) while making the spatial dimension outermost (row) was intended to coalesce stores, but instead caused a 3.25x increase in executed instructions (771K → 2.5M), a 54% increase in registers per thread (26 → 40), and a drop in theoretical occupancy from 100% to 75%. The excessive sectors actually worsened from 112K to 376K (23% → 50%), indicating the transposition of arange shapes ([YBLOCK, 1] vs [1, YBLOCK]) created a fundamentally different and worse access pattern rather than fixing coalescing. The net effect was a 1.8x slowdown in kernel duration (5.89 → 10.62 µs).

> **Applicable when**: Never — this specific axis swap made things worse. Do not apply when the kernel has a small Y dimension (ynumel=30) relative to X (xnumel=50176), as the autotune grid becomes degenerate and register pressure increases.

> **Avoid when**: Do not swap tile axis shapes ([:, None] vs [None, :]) for coalescing when: (1) the Y dimension is very small (e.g., ynumel=30) causing the Y-block to cover all elements in 1-2 blocks, reducing grid size and parallelism; (2) the autotune decorator is added simultaneously, as it can increase register usage and reduce theoretical occupancy; (3) the store address formula has both X and Y terms interleaved (y0 + 3*x2), making simple axis swaps insufficient to achieve coalescing.

---

## Iteration 2

**Best so far**: 1.7848× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_lambda_grid_to_activate_tile_exploration | 1.352× | 1.558× | ✓ |  |
| B | vectorized_loads_with_max_contiguous_hints | 1.361× | 1.324× | ✓ |  |
| C | vectorized_loads_with_max_contiguous_hints | 1.401× | 0.964× | ✓ |  |
| D | vectorized_loads_with_contiguity_hints_and_large_xblock_configs | 1.388× | 1.406× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_lamb... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 3780 ns | -2110 ns (-35.8%) | 4450 ns | -1440 ns (-24.4%) | 6110 ns | +220 ns (+3.7%) | 4190 ns | -1700 ns (-28.9%) |
| SM Throughput (%) | 16.51 % | 17.63 % | +1.12 pp | 27.09 % | +10.58 pp | 16.65 % | +0.14 pp | 26.61 % | +10.10 pp |
| Memory Throughput (%) | 42.39 % | 18.23 % | -24.16 pp | 21.51 % | -20.88 pp | 41.53 % | -0.86 pp | 21.10 % | -21.29 pp |
| DRAM Throughput (%) | 20.97 % | 4.40 % | -16.57 pp | 3.73 % | -17.24 pp | 20.24 % | -0.73 pp | 3.95 % | -17.02 pp |
| Achieved Occupancy (%) | 54.91 % | 44.89 % | -10.02 pp | 67.65 % | +12.74 pp | 56.01 % | +1.10 pp | 69.40 % | +14.49 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 26.00 register/thread | +0 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.77 % | 54.68 % | -16.09 pp | 36.97 % | -33.80 pp | 71.83 % | +1.06 pp | 37.37 % | -33.40 pp |
| IPC Active | 1.15 inst/cycle | 1.66 inst/cycle | +0.51 | 2.37 inst/cycle | +1.22 | 1.09 inst/cycle | -0.06 | 2.42 inst/cycle | +1.27 |


### Key Learnings

#### ✅ Plan A: `fix_autotune_lambda_grid_to_activate_tile_exploration` (1.738×)

Replacing a hardcoded static grid tuple with a lambda grid in a Triton kernel that has @triton.autotune allows the autotuner to correctly size the grid per config, yielding a modest NCU-measured duration reduction (3.78µs → 3.39µs, ~10%). However, the core bottlenecks — 50% uncoalesced global store sectors, ~55% no-eligible scheduler cycles, and 0.74 waves/SM — remain entirely unchanged, indicating the lambda grid fix is a necessary prerequisite for autotune to function but does not itself resolve the underlying access pattern or occupancy issues. The reported wall-clock speedup (1.74x) substantially exceeds the NCU kernel duration improvement, suggesting the majority of the measured speedup comes from reduced autotune overhead or framework-level caching effects rather than kernel execution improvement.

> **Applicable when**: Triton kernel has @triton.autotune but the launch site uses a hardcoded static grid tuple instead of a lambda; the static grid locks in a specific XBLOCK/YBLOCK regardless of which config the autotuner selects; wall-clock speedup is expected from allowing the autotuner to cache the best config across repeated calls.

> **Avoid when**: When the bottleneck is uncoalesced global stores (UncoalescedGlobalAccess >40% excess sectors) or low SM occupancy — the lambda grid fix alone will not resolve these issues and the NCU kernel duration improvement will be marginal (~10%). Combine with tile config changes or store access pattern fixes to address the root cause.

---

## Iteration 3

**Best so far**: 1.7848× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_grid_enabling_autotune | 1.406× | 1.738× | ✓ |  |
| B | fix_hardcoded_grid_and_add_yblock32_configs_for_full_channel_coverage | 1.402× | 1.354× | ✓ |  |
| C | eliminate_partial_wave_with_grid_aligned_autotune_configs | 1.405× | 1.017× | ✓ |  |
| D | fix_grid_lambda_prune_configs_and_eviction_policy | 1.306× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_hardcoded_gri... | Δ | eliminate_partial... | Δ | fix_grid_lambda_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5890 ns | 3390 ns | -2500 ns (-42.4%) | 4350 ns | -1540 ns (-26.1%) | 5790 ns | -100 ns (-1.7%) | 5890 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 16.51 % | 17.65 % | +1.14 pp | 27.57 % | +11.06 pp | 28.27 % | +11.76 pp | 16.99 % | +0.48 pp |
| Memory Throughput (%) | 42.39 % | 20.42 % | -21.97 pp | 21.90 % | -20.49 pp | 30.92 % | -11.47 pp | 40.83 % | -1.56 pp |
| DRAM Throughput (%) | 20.97 % | 4.91 % | -16.06 pp | 3.80 % | -17.17 pp | 2.86 % | -18.11 pp | 20.95 % | -0.02 pp |
| Achieved Occupancy (%) | 54.91 % | 44.14 % | -10.77 pp | 71.33 % | +16.42 pp | 75.03 % | +20.12 pp | 56.60 % | +1.69 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 | 0.74 | +0.00 |
| No Eligible (%) | 70.77 % | 54.62 % | -16.15 pp | 41.10 % | -29.67 pp | 37.86 % | -32.91 pp | 69.24 % | -1.53 pp |
| IPC Active | 1.15 inst/cycle | 1.69 inst/cycle | +0.54 | 2.15 inst/cycle | +1.00 | 2.22 inst/cycle | +1.07 | 1.17 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  5890 ns  (1.000×)
Iter 1     3300 ns  (1.785×)  — vectorized_loads_with_autotune

Final best: 3300 ns  (1.7848×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512,  'YBLOCK': 1},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 2},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256,  'YBLOCK': 4},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 8},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=4),
+        triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=4),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1},  num_warps=8, num_stages=2),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -13,7 +26,9 @@

     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
+    # Apply vectorization hints to 1D arange before broadcasting
+    xarange = tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), 16), 16)
+    xindex = xoffset + xarange[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
@@ -22,15 +37,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
