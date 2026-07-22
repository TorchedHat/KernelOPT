# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5310 ns — **Best latency**: 5380 ns (**0.9870×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_11_triton_poi_fused__softmax_conv/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 5310 ns
- **SM Throughput**: 40.27 % | **Memory**: 39.70 % | **DRAM**: 4.58 %
- **Achieved Occupancy**: 69.50 %
- **IPC Active**: 2.74 inst/cycle | **No Eligible**: 31.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_to_eliminate_partial_wave_and_improve_coalescing | — | 0.987× | ✓ | **✓ best** |
| C | autotune_tile_sizes_to_eliminate_partial_wave_and_improve_coalescing | — | 0.701× | ✓ |  |
| D | add_autotune_fix_tile_sizes_for_coalesced_access | — | 1.193× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_to... | Δ | autotune_tile_siz... | Δ | add_autotune_fix_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 5380 ns | +70 ns (+1.3%) | 7580 ns | +2270 ns (+42.7%) | 4450 ns | -860 ns (-16.2%) |
| SM Throughput (%) | 40.27 % | 50.58 % | +10.31 pp | 63.19 % | +22.92 pp | 12.46 % | -27.81 pp |
| Memory Throughput (%) | 39.70 % | 34.96 % | -4.74 pp | 53.91 % | +14.21 pp | 5.45 % | -34.25 pp |
| DRAM Throughput (%) | 4.58 % | 4.49 % | -0.09 pp | 3.19 % | -1.39 pp | 5.45 % | +0.87 pp |
| Achieved Occupancy (%) | 69.50 % | 71.37 % | +1.87 pp | 76.70 % | +7.20 pp | 32.76 % | -36.74 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.11 % | 26.52 % | -4.59 pp | 15.11 % | -16.00 pp | 67.03 % | +35.92 pp |
| IPC Active | 2.74 inst/cycle | 3.06 inst/cycle | +0.32 | 3.36 inst/cycle | +0.62 | 1.23 inst/cycle | -1.51 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_persistent_tile_scheduling | — | 1.185× | ✓ |  |
| B | persistent_tile_scheduling_via_small_xblock_large_yblock_grid_lambda | — | 1.177× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 4480 ns | -830 ns (-15.6%) | 4510 ns | -800 ns (-15.1%) |
| SM Throughput (%) | 40.27 % | 12.60 % | -27.67 pp | 15.97 % | -24.30 pp |
| Memory Throughput (%) | 39.70 % | 5.43 % | -34.27 pp | 6.11 % | -33.59 pp |
| DRAM Throughput (%) | 4.58 % | 5.43 % | +0.85 pp | 5.39 % | +0.81 pp |
| Achieved Occupancy (%) | 69.50 % | 29.31 % | -40.19 pp | 31.65 % | -37.85 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.11 % | 62.75 % | +31.64 pp | 56.53 % | +25.42 pp |
| IPC Active | 2.74 inst/cycle | 1.29 inst/cycle | -1.45 | 1.58 inst/cycle | -1.16 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_autotune_meta_tile_sizes | — | 1.185× | ✓ |  |
| B | fix_grid_lambda_to_honor_autotune_tile_configs | — | 1.159× | ✓ |  |
| C | fix_grid_lambda_and_small_xblock_autotune_configs | — | 1.185× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5310 ns | 4480 ns | -830 ns (-15.6%) | 4580 ns | -730 ns (-13.7%) | 4480 ns | -830 ns (-15.6%) |
| SM Throughput (%) | 40.27 % | 12.53 % | -27.74 pp | 12.22 % | -28.05 pp | 15.75 % | -24.52 pp |
| Memory Throughput (%) | 39.70 % | 5.43 % | -34.27 pp | 5.30 % | -34.40 pp | 6.13 % | -33.57 pp |
| DRAM Throughput (%) | 4.58 % | 5.43 % | +0.85 pp | 5.30 % | +0.72 pp | 5.42 % | +0.84 pp |
| Achieved Occupancy (%) | 69.50 % | 28.93 % | -40.57 pp | 29.10 % | -40.40 pp | 32.06 % | -37.44 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.11 % | 64.41 % | +33.30 pp | 62.86 % | +31.75 pp | 55.62 % | +24.51 pp |
| IPC Active | 2.74 inst/cycle | 1.30 inst/cycle | -1.44 | 1.29 inst/cycle | -1.45 | 1.60 inst/cycle | -1.14 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_use_autotune_meta_tile_sizes` (1.185×)

When a Triton kernel uses @triton.autotune but the grid in the launch wrapper is hardcoded with constant tile sizes (e.g., cdiv(xnumel, 128) instead of cdiv(xnumel, META['XBLOCK'])), the autotune decorator selects different configs but the actual launch grid never changes, making the tuning completely ineffective. Replacing the static grid tuple with a lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK'])) allows autotune to actually vary the launch configuration. However, in this iteration the profiling metrics (duration, grid size, waves_per_sm, executed instructions) remained essentially unchanged between before and after, suggesting the diff was either not applied or the autotune selected a config equivalent to the original — the 1.19x speedup may reflect measurement noise or a different kernel being optimized rather than this specific NCU-profiled kernel.

> **Applicable when**: A Triton kernel has @triton.autotune but the grid in the launch wrapper uses hardcoded tile-size constants instead of META['BLOCK_PARAM']; NCU LaunchConfiguration rule fires indicating a large partial-wave tail; the autotune configs include tile sizes that would produce a smaller grid than the hardcoded one.

> **Avoid when**: The grid is already a META lambda; the kernel is not autotuned; the autotune configs all produce the same grid size as the hardcoded constants; the kernel is compute-bound rather than latency-bound from partial waves.

#### ✅ Plan B: `fix_grid_lambda_to_honor_autotune_tile_configs` (1.159×)

When a Triton kernel uses @triton.autotune but the grid in kernel_function is hardcoded with fixed constants (e.g., cdiv(xnumel, 128)) instead of a META lambda (lambda META: (cdiv(xnumel, META['XBLOCK']), cdiv(ynumel, META['YBLOCK']))), the autotune decorator selects tile configs but the launch grid never changes, making autotuning completely ineffective. Fixing the grid to a META lambda allows autotune to actually vary the number of launched blocks, which can reduce partial-wave tail overhead flagged by the LaunchConfiguration NCU rule. Note: in this specific case the profiling metrics (duration, occupancy, warp cycles) did not improve — the 1.16x speedup was measured at the benchmark level, suggesting the benefit may come from reduced kernel launch overhead or better autotune config selection across repeated calls rather than per-kernel metric improvement.

> **Applicable when**: A Triton kernel has @triton.autotune with multiple configs but the grid in kernel_function uses hardcoded tile-size constants instead of META['BLOCK_PARAM']. NCU flags LaunchConfiguration (partial wave overhead) and/or UncoalescedGlobalAccess. The autotune key includes dimension parameters (e.g., key=['ynumel','xnumel']).

> **Avoid when**: The grid is already a META lambda (autotune is already effective). Do not apply if the kernel has no @triton.autotune decorator. Be cautious if the kernel body uses hardcoded tile-size constants internally (not just in the grid) — those would also need updating. Note that per-kernel NCU metrics may not show improvement even when benchmark-level speedup is positive.

---

## Cumulative Progress

```
Baseline:  5310 ns  (1.000×)

Final best: 5380 ns  (0.9870×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 2,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 32768
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +35,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

     ynumel = 32768
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_11[grid](arg_0, arg_1, 32768, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
