# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_47` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2500 ns (**1.0360×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_store_coalescing_by_swapping_tile_axes_and_grid_meta`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_47`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_47_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (192,1,1), Waves/SM: 0.09
- **Registers/thread**: 28
- **Duration**: 2590 ns
- **SM Throughput**: 4.00 % | **Memory**: 3.52 % | **DRAM**: 1.26 %
- **Achieved Occupancy**: 8.79 %
- **IPC Active**: 0.40 inst/cycle | **No Eligible**: 86.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 1.032× | 1.012× | ✓ | **✓ best** |
| B | autotune_tile_sizes_for_small_grid_utilization | 0.951× | 1.036× | ✓ |  |
| C | add_autotune_increase_tile_sizes_for_underutilized_kernel | 0.958× | 0.952× | ✓ |  |
| D | fix_tile_sizes_to_match_problem_dimensions_and_add_autotune | 1.020× | 0.918× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | add_autotune_incr... | Δ | fix_tile_sizes_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2560 ns | -30 ns (-1.2%) | 2500 ns | -90 ns (-3.5%) | 2720 ns | +130 ns (+5.0%) | 2820 ns | +230 ns (+8.9%) |
| SM Throughput (%) | 4.00 % | 0.65 % | -3.35 pp | 3.83 % | -0.17 pp | 4.12 % | +0.12 pp | 2.28 % | -1.72 pp |
| Memory Throughput (%) | 3.52 % | 1.06 % | -2.46 pp | 3.63 % | +0.11 pp | 3.91 % | +0.39 pp | 2.61 % | -0.91 pp |
| DRAM Throughput (%) | 1.26 % | 0.82 % | -0.44 pp | 1.31 % | +0.05 pp | 1.20 % | -0.06 pp | 1.15 % | -0.11 pp |
| Achieved Occupancy (%) | 8.79 % | 4.00 % | -4.79 pp | 8.54 % | -0.25 pp | 8.55 % | -0.24 pp | 4.07 % | -4.72 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 30.00 register/thread | +2 |
| Waves/SM | 0.09 | 0.05 | -0.04 | 0.09 | +0.00 | 0.09 | +0.00 | 0.05 | -0.04 |
| No Eligible (%) | 86.58 % | 94.21 % | +7.63 pp | 81.37 % | -5.21 pp | 82.51 % | -4.07 pp | 87.85 % | +1.27 pp |
| IPC Active | 0.40 inst/cycle | 0.15 inst/cycle | -0.25 | 0.69 inst/cycle | +0.29 | 0.68 inst/cycle | +0.28 | 0.32 inst/cycle | -0.08 |

---

## Iteration 2

**Best so far**: 1.0117× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_use_persistent_warp_tiling | 1.008× | 0.952× | ✓ |  |
| B | fix_grid_launch_to_use_autotuned_tile_sizes_and_improve_coalescing | 0.995× | 1.093× | ✓ |  |
| C | fix_grid_launch_to_match_autotune_tile_sizes_with_persistent_scheduling | 1.036× | 0.974× | ✓ | **✓ best** |
| D | fix_autotune_grid_to_use_meta_block_sizes | 0.985× | 1.107× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_launch_t... | Δ | fix_grid_launch_t... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2720 ns | +130 ns (+5.0%) | 2370 ns | -220 ns (-8.5%) | 2660 ns | +70 ns (+2.7%) | 2340 ns | -250 ns (-9.7%) |
| SM Throughput (%) | 4.00 % | 4.20 % | +0.20 pp | 1.10 % | -2.90 pp | 0.39 % | -3.61 pp | 0.66 % | -3.34 pp |
| Memory Throughput (%) | 3.52 % | 3.99 % | +0.47 pp | 0.98 % | -2.54 pp | 1.08 % | -2.44 pp | 1.01 % | -2.51 pp |
| DRAM Throughput (%) | 1.26 % | 1.20 % | -0.06 pp | 0.89 % | -0.37 pp | 0.79 % | -0.47 pp | 0.90 % | -0.36 pp |
| Achieved Occupancy (%) | 8.79 % | 8.55 % | -0.24 pp | 8.41 % | -0.38 pp | 2.21 % | -6.58 pp | 4.07 % | -4.72 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.09 | 0.09 | +0.00 | 0.09 | +0.00 | 0.05 | -0.04 | 0.05 | -0.04 |
| No Eligible (%) | 86.58 % | 81.36 % | -5.22 pp | 90.63 % | +4.05 pp | 95.92 % | +9.34 pp | 93.48 % | +6.90 pp |
| IPC Active | 0.40 inst/cycle | 0.65 inst/cycle | +0.25 | 0.30 inst/cycle | -0.10 | 0.06 inst/cycle | -0.34 | 0.15 inst/cycle | -0.25 |

---

## Iteration 3

**Best so far**: 1.0117× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_use_meta_tile_sizes_and_coalesce_store_access | 0.978× | 0.932× | ✓ |  |
| B | fix_grid_launch_use_meta_tile_sizes_and_coalesced_store_configs | 0.959× | 0.918× | ✓ |  |
| C | fix_grid_launch_meta_and_improve_store_coalescing_with_eviction_hint | 0.991× | 0.942× | ✓ |  |
| D | reduce_register_pressure_with_num_stages1_and_streaming_store_eviction | 1.039× | 1.036× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_launch_u... | Δ | fix_grid_launch_m... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2780 ns | +190 ns (+7.3%) | 2820 ns | +230 ns (+8.9%) | 2750 ns | +160 ns (+6.2%) | 2500 ns | -90 ns (-3.5%) |
| SM Throughput (%) | 4.00 % | 3.82 % | -0.18 pp | 1.24 % | -2.76 pp | 0.62 % | -3.38 pp | 0.67 % | -3.33 pp |
| Memory Throughput (%) | 3.52 % | 4.40 % | +0.88 pp | 1.00 % | -2.52 pp | 0.95 % | -2.57 pp | 0.97 % | -2.55 pp |
| DRAM Throughput (%) | 1.26 % | 1.18 % | -0.08 pp | 0.75 % | -0.51 pp | 0.76 % | -0.50 pp | 0.85 % | -0.41 pp |
| Achieved Occupancy (%) | 8.79 % | 4.24 % | -4.55 pp | 8.55 % | -0.24 pp | 4.12 % | -4.67 pp | 4.15 % | -4.64 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.09 | 0.05 | -0.04 | 0.09 | +0.00 | 0.05 | -0.04 | 0.05 | -0.04 |
| No Eligible (%) | 86.58 % | 78.93 % | -7.65 pp | 90.06 % | +3.48 pp | 92.78 % | +6.20 pp | 94.40 % | +7.82 pp |
| IPC Active | 0.40 inst/cycle | 0.52 inst/cycle | +0.12 | 0.28 inst/cycle | -0.12 | 0.14 inst/cycle | -0.26 | 0.15 inst/cycle | -0.25 |

---

## Iteration 4

**Best so far**: 1.036× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_meta_tile_sizes_with_smem_transpose_for_coalescing | 1.000× | 1.024× | ✓ |  |
| B | fix_store_coalescing_by_swapping_tile_axes_and_grid_meta | 1.010× | 1.036× | ✓ |  |
| C | fix_grid_lambda_use_meta_tile_sizes_with_num_stages1_and_coalesced_load | 1.033× | 0.989× | ✓ |  |
| D | coalesce_strided_store_via_evict_first_and_fix_grid_lambda | 0.984× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_store_coalesc... | Δ | fix_grid_lambda_u... | Δ | coalesce_strided_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2530 ns | -60 ns (-2.3%) | 2500 ns | -90 ns (-3.5%) | 2620 ns | +30 ns (+1.2%) | 2720 ns | +130 ns (+5.0%) |
| SM Throughput (%) | 4.00 % | 0.69 % | -3.31 pp | 0.66 % | -3.34 pp | 3.58 % | -0.42 pp | 1.28 % | -2.72 pp |
| Memory Throughput (%) | 3.52 % | 0.99 % | -2.53 pp | 0.93 % | -2.59 pp | 4.13 % | +0.61 pp | 0.85 % | -2.67 pp |
| DRAM Throughput (%) | 1.26 % | 0.83 % | -0.43 pp | 0.84 % | -0.42 pp | 1.26 % | +0.00 pp | 0.77 % | -0.49 pp |
| Achieved Occupancy (%) | 8.79 % | 3.91 % | -4.88 pp | 4.03 % | -4.76 pp | 4.22 % | -4.57 pp | 8.60 % | -0.19 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.09 | 0.05 | -0.04 | 0.05 | -0.04 | 0.05 | -0.04 | 0.09 | +0.00 |
| No Eligible (%) | 86.58 % | 95.68 % | +9.10 pp | 93.65 % | +7.07 pp | 80.16 % | -6.42 pp | 90.60 % | +4.02 pp |
| IPC Active | 0.40 inst/cycle | 0.09 inst/cycle | -0.31 | 0.14 inst/cycle | -0.26 | 0.51 inst/cycle | +0.11 | 0.20 inst/cycle | -0.20 |

---

## Iteration 5

**Best so far**: 1.036× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.170× | 0.016× | ✓ |  |
| B | flatten_2d_to_1d_with_persistent_loop_over_tiles | 0.043× | 0.005× | ✓ |  |
| C | fix_grid_lambda_use_meta_tile_sizes_and_fix_hardcoded_kernel_dims | 0.989× | 0.899× | ✓ |  |
| D | persistent_tile_loop_with_coalesced_store_reindex | 0.171× | 0.033× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | flatten_2d_to_1d_... | Δ | fix_grid_lambda_u... | Δ | persistent_tile_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 160800 ns | +158210 ns (+6108.5%) | 521280 ns | +518690 ns (+20026.6%) | 2880 ns | +290 ns (+11.2%) | 78180 ns | +75590 ns (+2918.5%) |
| SM Throughput (%) | 4.00 % | 3.95 % | -0.05 pp | 3.91 % | -0.09 pp | 3.69 % | -0.31 pp | 10.82 % | +6.82 pp |
| Memory Throughput (%) | 3.52 % | 3.46 % | -0.06 pp | 8.79 % | +5.27 pp | 4.25 % | +0.73 pp | 7.20 % | +3.68 pp |
| DRAM Throughput (%) | 1.26 % | 0.02 % | -1.24 pp | 0.01 % | -1.25 pp | 1.15 % | -0.11 pp | 0.04 % | -1.22 pp |
| Achieved Occupancy (%) | 8.79 % | 4.50 % | -4.29 pp | 4.55 % | -4.24 pp | 4.26 % | -4.53 pp | 8.96 % | +0.17 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.09 | 0.05 | -0.04 | 0.05 | -0.04 | 0.05 | -0.04 | 0.09 | +0.00 |
| No Eligible (%) | 86.58 % | 94.55 % | +7.97 pp | 94.59 % | +8.01 pp | 80.26 % | -6.32 pp | 88.79 % | +2.21 pp |
| IPC Active | 0.40 inst/cycle | 0.16 inst/cycle | -0.24 | 0.16 inst/cycle | -0.24 | 0.48 inst/cycle | +0.08 | 0.44 inst/cycle | +0.04 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.016×)

Replacing a single-pass 2D tile dispatch with a persistent while-loop (each SM iterates over multiple tiles via stride-NUM_SMS) caused a 62x regression: kernel duration exploded from 2.5 µs to 160.8 µs and executed instructions grew from 11,136 to 4,846,464 (~435x). Despite the grid remaining at 192 CTAs (waves_per_sm unchanged at 0.05), the while-loop body executed hundreds of iterations per SM, generating massive L1TEX stall pressure (CPIStall: 74.2% of cycles stalled on L1TEX, 13.7 cycles per warp). The persistent loop pattern is counterproductive for tiny kernels where the total work is already small: the loop overhead and repeated memory accesses dwarf the original kernel cost, and the uncoalesced access pattern (78% excessive sectors) is amplified across all loop iterations.

> **Applicable when**: Never applicable for tiny kernels (total_tiles < 1000, kernel duration < 10 µs). Persistent tile scheduling is only beneficial when the kernel is large enough that the loop overhead is negligible compared to the work per tile, and when the grid is severely undersized relative to SM count (waves_per_sm << 1 AND total work is large).

> **Avoid when**: Kernel duration is already in the 1-10 µs range; total tile count is small (< a few hundred); the kernel has uncoalesced memory access patterns (UncoalescedGlobalAccess rule firing at >50% excessive sectors) — the loop amplifies the uncoalesced penalty across all iterations; the grid size is already close to or exceeds NUM_SMS.

#### ❌ Plan B: `flatten_2d_to_1d_with_persistent_loop_over_tiles` (0.005×)

Replacing a 2D grid kernel with a 1D persistent loop caused a 200x slowdown (2.5 µs → 521 µs) on a tiny 38400-element kernel. The persistent loop introduced a Python-level `range()` loop inside the Triton kernel body that Triton compiled into a sequential scalar loop rather than a vectorized tile loop, exploding executed instructions from 11136 to 15,674,496 (~1408x). The CPIStall rule fired with 80.5% of cycles stalled on L1TEX, and branch instructions jumped from 432 to 461,568, confirming the loop overhead dominated. For kernels with numel < ~100K elements, persistent loops add catastrophic loop-control overhead that dwarfs any SM utilization benefit.

> **Applicable when**: Never applicable for tiny kernels (numel < 100K). Persistent loops in Triton are only beneficial when numel is large enough (>1M elements) that the loop overhead is amortized across many vectorized tile iterations.

> **Avoid when**: Any kernel with numel < ~500K elements; kernels where the grid is already small (waves_per_sm < 0.5) — the persistent loop adds loop-control overhead that explodes instruction count without improving SM utilization; kernels where the baseline duration is already in the 1-5 µs range (launch-latency dominated).

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)
Iter 1     2560 ns  (1.012×)  — persistent_tile_scheduling_with_autotune
Iter 3     2500 ns  (1.036×)  — reduce_register_pressure_with_num_stages1_and_streaming_store_eviction

Final best: 2500 ns  (1.0360×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,44 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 512}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':1024}, num_warps=16, num_stages=1),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK':  4, 'YBLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':  64}, num_warps=2, num_stages=1),
+        triton.Config({'XBLOCK':  8, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  32}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK':  64}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 256}, num_warps=8, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1536
     xnumel = 25
+    xoffset = tl.program_id(0) * XBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
+    xmask = xindex < xnumel
     yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
+    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
     ymask = yindex < ynumel
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 24)
     y1 = yindex // 24
     tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 24*x2 + 600*y1), tmp0, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 24*x2 + 600*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +50,7 @@

     ynumel = 1536
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_47[grid](arg_0, arg_1, 1536, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_47[grid](arg_0, arg_1, 1536, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
