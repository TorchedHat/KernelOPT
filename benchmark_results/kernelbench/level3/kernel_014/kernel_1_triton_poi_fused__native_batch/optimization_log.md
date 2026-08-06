# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2460 ns — **Best latency**: 2400 ns (**1.0250×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_grid_lambda_to_activate_autotune_without_index_changes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.06
- **Registers/thread**: 28
- **Duration**: 2460 ns
- **SM Throughput**: 2.45 % | **Memory**: 2.29 % | **DRAM**: 0.34 %
- **Achieved Occupancy**: 6.07 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 90.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.445× | 0.086× | ✓ |  |
| B | fix_tile_sizes_to_eliminate_masked_waste_and_improve_occupancy | — | fail | ✓ |  |
| C | right_size_tile_dimensions_for_small_xnumel | 1.073× | 0.961× | ✓ |  |
| D | fix_tile_sizes_for_underutilized_small_kernel | 1.032× | 0.885× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | right_size_tile_d... | Δ | fix_tile_sizes_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 28640 ns | +26180 ns (+1064.2%) | 2560 ns | +100 ns (+4.1%) | 2780 ns | +320 ns (+13.0%) |
| SM Throughput (%) | 2.45 % | 3.41 % | +0.96 pp | 0.85 % | -1.60 pp | 2.28 % | -0.17 pp |
| Memory Throughput (%) | 2.29 % | 5.23 % | +2.94 pp | 4.76 % | +2.47 pp | 3.71 % | +1.42 pp |
| DRAM Throughput (%) | 0.34 % | 0.03 % | -0.31 pp | 4.76 % | +4.42 pp | 2.22 % | +1.88 pp |
| Achieved Occupancy (%) | 6.07 % | 3.12 % | -2.95 pp | 5.78 % | -0.29 pp | 3.12 % | -2.95 pp |
| Registers/Thread | 28.00 register/thread | 27.00 register/thread | -1 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.06 | 0.03 | -0.03 | 0.06 | +0.00 | 0.03 | -0.03 |
| No Eligible (%) | 90.30 % | 92.16 % | +1.86 pp | 96.58 % | +6.28 pp | 84.46 % | -5.84 pp |
| IPC Active | 0.34 inst/cycle | 0.16 inst/cycle | -0.18 | 0.11 inst/cycle | -0.23 | 0.30 inst/cycle | -0.04 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.086×)

Adding a persistent while-loop inside a Triton kernel to reuse CTAs across tiles caused a 11.6x slowdown despite the kernel being grid-size limited. The instruction count exploded from 51,200 to 796,160 (15.5x more), and the dominant stall shifted from barrier/instruction-fetch stalls to L1TEX memory stalls (77.7% of cycles). The grid was not updated to cap at SM count (still used the old 2D grid formula), so the persistent loop ran redundant iterations while also introducing per-tile index arithmetic overhead that dominated the tiny workload.

> **Applicable when**: Never: persistent tile scheduling is harmful for tiny kernels where the total work (tiles) is already smaller than or comparable to the number of SMs. The per-tile index arithmetic in the while-loop body dominates over actual memory work.

> **Avoid when**: Kernel has very few total tiles (e.g., <200 tiles for a 132-SM GPU). The grid is already small but the kernel duration is already in the low microsecond range (< 5 us). The workload is so small that adding loop overhead and tile-index arithmetic (modulo, division) per iteration costs more than the occupancy gain. Also avoid when the grid launch formula is not updated to actually cap at SM count — the persistent loop then runs with the same number of CTAs as before, adding overhead with no benefit.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_formula_and_maximize_coalescing_via_autotune | 1.041× | 0.904× | ✓ |  |
| B | fix_grid_launch_formula_and_eviction_policy_for_streaming_copy | 0.927× | 0.894× | ✓ |  |
| C | revert_persistent_loop_to_simple_2d_grid_dispatch | 1.004× | 1.025× | ✓ | **✓ best** |
| D | add_triton_autotune | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_formula_... | Δ | fix_grid_launch_f... | Δ | revert_persistent... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2720 ns | +260 ns (+10.6%) | 2750 ns | +290 ns (+11.8%) | 2400 ns | -60 ns (-2.4%) |
| SM Throughput (%) | 2.45 % | 0.78 % | -1.67 pp | 2.28 % | -0.17 pp | 1.01 % | -1.44 pp |
| Memory Throughput (%) | 2.29 % | 4.50 % | +2.21 pp | 3.52 % | +1.23 pp | 1.60 % | -0.69 pp |
| DRAM Throughput (%) | 0.34 % | 4.50 % | +4.16 pp | 2.24 % | +1.90 pp | 0.66 % | +0.32 pp |
| Achieved Occupancy (%) | 6.07 % | 5.85 % | -0.22 pp | 3.12 % | -2.95 pp | 3.10 % | -2.97 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 21.00 register/thread | -7 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.03 | -0.03 | 0.03 | -0.03 |
| No Eligible (%) | 90.30 % | 96.62 % | +6.32 pp | 84.78 % | -5.52 pp | 93.89 % | +3.59 pp |
| IPC Active | 0.34 inst/cycle | 0.11 inst/cycle | -0.23 | 0.30 inst/cycle | -0.04 | 0.14 inst/cycle | -0.20 |

---

## Iteration 3

**Best so far**: 1.025× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_formula_and_eviction_policy_for_streaming_transpose | 0.982× | 0.972× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_formula_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2460 ns | 2530 ns | +70 ns (+2.8%) |
| SM Throughput (%) | 2.45 % | 0.44 % | -2.01 pp |
| Memory Throughput (%) | 2.29 % | 2.40 % | +0.11 pp |
| DRAM Throughput (%) | 0.34 % | 2.40 % | +2.06 pp |
| Achieved Occupancy (%) | 6.07 % | 3.05 % | -3.02 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 |
| Waves/SM | 0.06 | 0.03 | -0.03 |
| No Eligible (%) | 90.30 % | 96.09 % | +5.79 pp |
| IPC Active | 0.34 inst/cycle | 0.06 inst/cycle | -0.28 |

---

## Iteration 4

**Best so far**: 1.025× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_with_correct_index_arithmetic | 0.997× | 0.972× | ✓ |  |
| C | fix_hardcoded_grid_to_restore_autotune_effectiveness | 0.958× | 0.961× | ✓ |  |
| D | fix_grid_lambda_to_activate_autotune_without_index_changes | 0.995× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2460 ns | 2530 ns | +70 ns (+2.8%) | 2560 ns | +100 ns (+4.1%) | 2400 ns | -60 ns (-2.4%) |
| SM Throughput (%) | 2.45 % | 0.37 % | -2.08 pp | 0.43 % | -2.02 pp | 0.20 % | -2.25 pp |
| Memory Throughput (%) | 2.29 % | 2.43 % | +0.14 pp | 2.39 % | +0.10 pp | 1.30 % | -0.99 pp |
| DRAM Throughput (%) | 0.34 % | 2.43 % | +2.09 pp | 2.39 % | +2.05 pp | 1.30 % | +0.96 pp |
| Achieved Occupancy (%) | 6.07 % | 3.00 % | -3.07 pp | 3.00 % | -3.07 pp | 1.56 % | -4.51 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.06 | 0.03 | -0.03 | 0.03 | -0.03 | 0.03 | -0.03 |
| No Eligible (%) | 90.30 % | 96.36 % | +6.06 pp | 96.17 % | +5.87 pp | 96.26 % | +5.96 pp |
| IPC Active | 0.34 inst/cycle | 0.06 inst/cycle | -0.28 | 0.06 inst/cycle | -0.28 | 0.03 inst/cycle | -0.31 |

---

## Cumulative Progress

```
Baseline:  2460 ns  (1.000×)
Iter 2     2400 ns  (1.025×)  — revert_persistent_loop_to_simple_2d_grid_dispatch

Final best: 2400 ns  (1.0250×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 64}, num_warps=2),
+    triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=2),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=4),
+    triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+], key=['ynumel', 'xnumel'])
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
     ynumel = 1024
     xnumel = 9
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_first')
     tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +36,7 @@

     ynumel = 1024
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
