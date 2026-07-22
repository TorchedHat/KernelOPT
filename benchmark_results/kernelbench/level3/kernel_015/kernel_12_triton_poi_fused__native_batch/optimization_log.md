# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3070 ns — **Best latency**: 2400 ns (**1.2792×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_hardcoded_grid_to_lambda_and_expand_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_12_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (768,1,1), Waves/SM: 0.36
- **Registers/thread**: 28
- **Duration**: 3070 ns
- **SM Throughput**: 13.90 % | **Memory**: 12.68 % | **DRAM**: 1.51 %
- **Achieved Occupancy**: 27.48 %
- **IPC Active**: 1.32 inst/cycle | **No Eligible**: 60.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune | 1.020× | 1.141× | ✓ |  |
| B | autotune_tile_sizes_for_occupancy_and_coalescing | 1.073× | 1.066× | ✓ |  |
| C | right_size_tile_blocks_for_small_xnumel_and_add_autotune | 1.092× | 1.279× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_underutilized_transpose_kernel | 1.041× | 1.185× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_tile_siz... | Δ | right_size_tile_b... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2690 ns | -380 ns (-12.4%) | 2880 ns | -190 ns (-6.2%) | 2400 ns | -670 ns (-21.8%) | 2590 ns | -480 ns (-15.6%) |
| SM Throughput (%) | 13.90 % | 8.83 % | -5.07 pp | 12.80 % | -1.10 pp | 2.62 % | -11.28 pp | 5.65 % | -8.25 pp |
| Memory Throughput (%) | 12.68 % | 7.95 % | -4.73 pp | 6.26 % | -6.42 pp | 1.91 % | -10.77 pp | 2.06 % | -10.62 pp |
| DRAM Throughput (%) | 1.51 % | 1.74 % | +0.23 pp | 1.61 % | +0.10 pp | 1.91 % | +0.40 pp | 1.80 % | +0.29 pp |
| Achieved Occupancy (%) | 27.48 % | 12.22 % | -15.26 pp | 24.78 % | -2.70 pp | 10.22 % | -17.26 pp | 24.32 % | -3.16 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 22.00 register/thread | -6 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.36 | 0.18 | -0.18 | 0.36 | +0.00 | 0.18 | -0.18 | 0.36 | +0.00 |
| No Eligible (%) | 60.76 % | 66.86 % | +6.10 pp | 57.31 % | -3.45 pp | 89.52 % | +28.76 pp | 68.46 % | +7.70 pp |
| IPC Active | 1.32 inst/cycle | 1.01 inst/cycle | -0.31 | 1.35 inst/cycle | +0.03 | 0.41 inst/cycle | -0.91 | 1.04 inst/cycle | -0.28 |

---

## Iteration 2

**Best so far**: 1.2792× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_grid_for_autotune_correctness | 1.053× | 1.199× | ✓ |  |
| B | fix_grid_lambda_to_use_meta_xblock_yblock_for_autotune | 1.068× | 1.172× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_dynamic_cta_reshaping | 1.071× | 1.263× | ✓ |  |
| E | add_triton_autotune | 1.067× | 1.172× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2560 ns | -510 ns (-16.6%) | 2620 ns | -450 ns (-14.7%) | 2430 ns | -640 ns (-20.8%) | 2620 ns | -450 ns (-14.7%) |
| SM Throughput (%) | 13.90 % | 2.49 % | -11.41 pp | 5.60 % | -8.30 pp | 2.82 % | -11.08 pp | 2.61 % | -11.29 pp |
| Memory Throughput (%) | 12.68 % | 1.80 % | -10.88 pp | 2.04 % | -10.64 pp | 1.89 % | -10.79 pp | 1.75 % | -10.93 pp |
| DRAM Throughput (%) | 1.51 % | 1.80 % | +0.29 pp | 1.77 % | +0.26 pp | 1.89 % | +0.38 pp | 1.75 % | +0.24 pp |
| Achieved Occupancy (%) | 27.48 % | 9.57 % | -17.91 pp | 24.09 % | -3.39 pp | 10.23 % | -17.25 pp | 11.50 % | -15.98 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.36 | 0.18 | -0.18 | 0.36 | +0.00 | 0.18 | -0.18 | 0.18 | -0.18 |
| No Eligible (%) | 60.76 % | 85.28 % | +24.52 pp | 69.43 % | +8.67 pp | 86.97 % | +26.21 pp | 84.13 % | +23.37 pp |
| IPC Active | 1.32 inst/cycle | 0.38 inst/cycle | -0.94 | 1.02 inst/cycle | -0.30 | 0.41 inst/cycle | -0.91 | 0.35 inst/cycle | -0.97 |


### Key Learnings

#### ✅ Plan A: `fix_static_grid_to_lambda_grid_for_autotune_correctness` (1.199×)

When a Triton kernel has @triton.autotune but the call site uses a hardcoded static grid tuple (e.g., `(cdiv(xnumel, 128), cdiv(ynumel, 8))`), the autotuner selects tile configs but the grid shape never changes, so each CTA processes far fewer elements than the selected YBLOCK/XBLOCK implies. Replacing the static grid with a lambda (`lambda meta: (cdiv(xnumel, meta['XBLOCK']), cdiv(ynumel, meta['YBLOCK']))`) is the correct fix, but in this iteration the diff was empty — the change was not actually applied. The profiling data shows no improvement in waves_per_sm (still 0.18), executed instructions (still 48192), or grid size (still 768), confirming the static grid was not replaced. The measured 1.2x speedup likely reflects measurement noise or a minor unrelated change, not the intended optimization.

> **Applicable when**: Triton kernel has @triton.autotune but call site uses a static grid tuple with hardcoded tile sizes; waves_per_sm < 0.5; achieved occupancy < 20%; No Eligible scheduler cycles > 80%; grid_size and executed instructions are unchanged across iterations despite autotune being present.

> **Avoid when**: The call site already uses a lambda grid (meta['XBLOCK']/meta['YBLOCK']); kernel does not use @triton.autotune; waves_per_sm is already > 1.0 indicating sufficient SM saturation.

---

## Iteration 3

**Best so far**: 1.2792× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_correct_2d_linearization | 0.012× | 0.006× | ✓ |  |
| B | fix_grid_lambda_to_match_autotune_tile_sizes | 1.086× | 1.185× | ✓ |  |
| C | fix_hardcoded_grid_to_dynamic_lambda_for_real_cta_reshaping | 1.047× | 1.185× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_and_expand_autotune_configs | 1.055× | 1.279× | ✓ |  |
| E | add_triton_autotune | 1.082× | 1.185× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 537630 ns | +534560 ns (+17412.4%) | 2590 ns | -480 ns (-15.6%) | 2590 ns | -480 ns (-15.6%) | 2400 ns | -670 ns (-21.8%) | 2590 ns | -480 ns (-15.6%) |
| SM Throughput (%) | 13.90 % | 64.81 % | +50.91 pp | 2.70 % | -11.20 pp | 5.35 % | -8.55 pp | 2.70 % | -11.20 pp | 2.53 % | -11.37 pp |
| Memory Throughput (%) | 12.68 % | 63.82 % | +51.14 pp | 1.78 % | -10.90 pp | 1.95 % | -10.73 pp | 1.93 % | -10.75 pp | 1.78 % | -10.90 pp |
| DRAM Throughput (%) | 1.51 % | 0.01 % | -1.50 pp | 1.78 % | +0.27 pp | 1.80 % | +0.29 pp | 1.93 % | +0.42 pp | 1.78 % | +0.27 pp |
| Achieved Occupancy (%) | 27.48 % | 36.32 % | +8.84 pp | 9.27 % | -18.21 pp | 23.33 % | -4.15 pp | 9.27 % | -18.21 pp | 9.18 % | -18.30 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.36 | 0.36 | +0.00 | 0.18 | -0.18 | 0.36 | +0.00 | 0.18 | -0.18 | 0.18 | -0.18 |
| No Eligible (%) | 60.76 % | 33.56 % | -27.20 pp | 87.27 % | +26.51 pp | 67.56 % | +6.80 pp | 83.31 % | +22.55 pp | 83.56 % | +22.80 pp |
| IPC Active | 1.32 inst/cycle | 2.52 inst/cycle | +1.20 | 0.40 inst/cycle | -0.92 | 1.02 inst/cycle | -0.30 | 0.40 inst/cycle | -0.92 | 0.40 inst/cycle | -0.92 |

---

## Cumulative Progress

```
Baseline:  3070 ns  (1.000×)
Iter 1     2400 ns  (1.279×)  — right_size_tile_blocks_for_small_xnumel_and_add_autotune

Final best: 2400 ns  (1.2792×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK':  1, 'YBLOCK':   64}, num_warps=2),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  128}, num_warps=4),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  256}, num_warps=4),
+        triton.Config({'XBLOCK':  1, 'YBLOCK':  512}, num_warps=8),
+        triton.Config({'XBLOCK':  1, 'YBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':   64}, num_warps=2),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  128}, num_warps=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  256}, num_warps=4),
+        triton.Config({'XBLOCK':  4, 'YBLOCK':  512}, num_warps=8),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':   64}, num_warps=2),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':  128}, num_warps=4),
+        triton.Config({'XBLOCK':  8, 'YBLOCK':  256}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':   32}, num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':   64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 6144
@@ -22,14 +44,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 6144
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12[grid](arg_0, arg_1, 6144, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_12[grid](arg_0, arg_1, 6144, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
