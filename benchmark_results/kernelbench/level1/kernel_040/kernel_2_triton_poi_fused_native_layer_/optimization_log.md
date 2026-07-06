# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_native_layer_norm_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 325920 ns — **Best latency**: 326020 ns (**0.9997×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_native_layer_norm_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_040/kernel_2_triton_poi_fused_native_layer_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524288,1,1), Waves/SM: 248.24
- **Registers/thread**: 18
- **Duration**: 325920 ns
- **SM Throughput**: 53.51 % | **Memory**: 66.55 % | **DRAM**: 66.55 %
- **Achieved Occupancy**: 56.78 %
- **IPC Active**: 1.42 inst/cycle | **No Eligible**: 64.49 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.002× | 0.267× | ✓ | **✓ best** |
| C | autotune_xblock_for_memory_coalescing_and_l2_reuse | 0.810× | 0.238× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 325920 ns | 1220000 ns | +894080 ns (+274.3%) | 1370000 ns | +1044080 ns (+320.3%) |
| SM Throughput (%) | 53.51 % | 74.59 % | +21.08 pp | 75.33 % | +21.82 pp |
| Memory Throughput (%) | 66.55 % | 45.36 % | -21.19 pp | 40.70 % | -25.85 pp |
| DRAM Throughput (%) | 66.55 % | 17.76 % | -48.79 pp | 15.77 % | -50.78 pp |
| Achieved Occupancy (%) | 56.78 % | 87.50 % | +30.72 pp | 79.26 % | +22.48 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 248.24 | 496.48 | +248.24 | 992.97 | +744.73 |
| No Eligible (%) | 64.49 % | 24.90 % | -39.59 pp | 24.51 % | -39.98 pp |
| IPC Active | 1.42 inst/cycle | 3.01 inst/cycle | +1.59 | 3.01 inst/cycle | +1.59 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_incorrect_vectorization_hints_and_predication | 1.001× | 0.267× | ✓ |  |
| B | remove_invalid_vectorization_hints_and_unnecessary_predication | 1.001× | 0.267× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_incorrect_... | Δ | remove_invalid_ve... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 325920 ns | 1220000 ns | +894080 ns (+274.3%) | 1220000 ns | +894080 ns (+274.3%) |
| SM Throughput (%) | 53.51 % | 74.46 % | +20.95 pp | 74.86 % | +21.35 pp |
| Memory Throughput (%) | 66.55 % | 45.29 % | -21.26 pp | 45.53 % | -21.02 pp |
| DRAM Throughput (%) | 66.55 % | 17.77 % | -48.78 pp | 17.76 % | -48.79 pp |
| Achieved Occupancy (%) | 56.78 % | 87.68 % | +30.90 pp | 87.70 % | +30.92 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 248.24 | 496.48 | +248.24 | 496.48 | +248.24 |
| No Eligible (%) | 64.49 % | 24.83 % | -39.66 pp | 25.21 % | -39.28 pp |
| IPC Active | 1.42 inst/cycle | 3.01 inst/cycle | +1.59 | 3.01 inst/cycle | +1.59 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | revert_instruction_explosion_remove_hints_and_fix_mask | 1.002× | 1.000× | ✓ | **✓ best** |
| B | remove_flawed_vectorization_hints_and_unnecessary_predication | 1.002× | 0.267× | ✓ |  |
| D | fix_autotune_meta_lambda_grid_and_remove_unnecessary_predication | 1.001× | 0.267× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_instructio... | Δ | remove_flawed_vec... | Δ | fix_autotune_meta... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 325920 ns | 326020 ns | +100 ns (+0.0%) | 1220000 ns | +894080 ns (+274.3%) | 1220000 ns | +894080 ns (+274.3%) |
| SM Throughput (%) | 53.51 % | 53.33 % | -0.18 pp | 75.01 % | +21.50 pp | 74.68 % | +21.17 pp |
| Memory Throughput (%) | 66.55 % | 66.49 % | -0.06 pp | 45.62 % | -20.93 pp | 45.42 % | -21.13 pp |
| DRAM Throughput (%) | 66.55 % | 66.49 % | -0.06 pp | 17.73 % | -48.82 pp | 17.73 % | -48.82 pp |
| Achieved Occupancy (%) | 56.78 % | 56.58 % | -0.20 pp | 87.39 % | +30.61 pp | 87.62 % | +30.84 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 496.48 | +248.24 | 496.48 | +248.24 |
| No Eligible (%) | 64.49 % | 64.48 % | -0.01 pp | 25.32 % | -39.17 pp | 24.86 % | -39.63 pp |
| IPC Active | 1.42 inst/cycle | 1.43 inst/cycle | +0.01 | 3.00 inst/cycle | +1.58 | 3.00 inst/cycle | +1.58 |


### Key Learnings

#### ❌ Plan B: `remove_flawed_vectorization_hints_and_unnecessary_predication` (0.267×)

Removing only the tl.max_contiguous/tl.multiple_of hint on xindex (before the modulo) while leaving the xmask predication and the fixed grid in place produced zero measurable improvement: executed instructions remained at 713M, duration stayed at 1.22ms, and all throughput metrics were statistically identical before and after. The diff shows only the vectorization hint line was removed, but the xmask guard (xindex < xnumel) and the fixed grid were not touched, meaning the root causes of the regression (unnecessary predication on every load/store, and grid mismatch with autotuned XBLOCK) were left intact. This confirms that the instruction explosion is driven by the combination of flaws, not the hint alone — partial fixes are insufficient and the full set of changes (remove xmask, remove hint, fix grid to meta-lambda) must be applied atomically.

> **Applicable when**: Kernel is in the regressed state with 713M executed instructions, 17.7% DRAM throughput, 74-75% compute throughput, 1.22ms duration, and xmask predication + vectorization hints present. The fix must be applied as a complete atomic change.

> **Avoid when**: Do not apply only a subset of the required fixes (e.g., removing only the vectorization hint while leaving xmask and fixed grid). Partial application leaves instruction count and duration completely unchanged, wasting an optimization iteration.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_for_l1tex_latency_hiding | 1.001× | 0.265× | ✓ |  |
| B | atomic_fix_grid_mask_and_hints_regression | 1.001× | 0.267× | ✓ |  |
| C | remove_autotune_fix_invalid_hints_and_mask | 1.001× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | atomic_fix_grid_m... | Δ | remove_autotune_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 325920 ns | 1230000 ns | +904080 ns (+277.4%) | 1220000 ns | +894080 ns (+274.3%) | 326940 ns | +1020 ns (+0.3%) |
| SM Throughput (%) | 53.51 % | 74.86 % | +21.35 pp | 74.97 % | +21.46 pp | 53.60 % | +0.09 pp |
| Memory Throughput (%) | 66.55 % | 45.53 % | -21.02 pp | 45.59 % | -20.96 pp | 66.29 % | -0.26 pp |
| DRAM Throughput (%) | 66.55 % | 17.50 % | -49.05 pp | 17.75 % | -48.80 pp | 66.29 % | -0.26 pp |
| Achieved Occupancy (%) | 56.78 % | 87.39 % | +30.61 pp | 87.63 % | +30.85 pp | 56.65 % | -0.13 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 248.24 | 496.48 | +248.24 | 496.48 | +248.24 | 248.24 | +0.00 |
| No Eligible (%) | 64.49 % | 24.82 % | -39.67 pp | 25.14 % | -39.35 pp | 64.24 % | -0.25 pp |
| IPC Active | 1.42 inst/cycle | 2.99 inst/cycle | +1.57 | 3.00 inst/cycle | +1.58 | 1.42 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_for_l1tex_latency_hiding | 1.001× | 0.999× | ✓ |  |
| B | fix_autotune_meta_lambda_grid_and_remove_unnecessary_mask | 1.002× | 0.267× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_autotune_meta... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 325920 ns | 326340 ns | +420 ns (+0.1%) | 1220000 ns | +894080 ns (+274.3%) |
| SM Throughput (%) | 53.51 % | 53.92 % | +0.41 pp | 74.98 % | +21.47 pp |
| Memory Throughput (%) | 66.55 % | 66.42 % | -0.13 pp | 45.59 % | -20.96 pp |
| DRAM Throughput (%) | 66.55 % | 66.42 % | -0.13 pp | 17.71 % | -48.84 pp |
| Achieved Occupancy (%) | 56.78 % | 57.84 % | +1.06 pp | 87.37 % | +30.59 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 32.00 register/thread | +14 |
| Waves/SM | 248.24 | 248.24 | +0.00 | 496.48 | +248.24 |
| No Eligible (%) | 64.49 % | 56.01 % | -8.48 pp | 25.32 % | -39.17 pp |
| IPC Active | 1.42 inst/cycle | 1.76 inst/cycle | +0.34 | 2.99 inst/cycle | +1.57 |

---

## Cumulative Progress

```
Baseline:  325920 ns  (1.000×)

Final best: 326020 ns  (0.9997×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,20 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 67108864
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x2 = xindex
     x1 = xindex // 4194304
-    x0 = (xindex % 4194304)
-    tmp0 = tl.load(in_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
-    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    x0 = xindex % 4194304
+    tmp0 = tl.load(in_ptr0 + x2, None)
+    tmp1 = tl.load(in_ptr1 + x1, None, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + x1, None, eviction_policy='evict_last')
+    tmp10 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 4194304.0, tl.float32)
     tmp5 = (tmp3 / tmp4)
@@ -28,9 +28,7 @@

     tmp9 = tmp2 * tmp8
     tmp11 = tmp9 * tmp10
     tmp13 = tmp11 + tmp12
-    tl.store(out_ptr0 + (x2), tmp13, None)
-
-
+    tl.store(out_ptr0 + x2, tmp13, None)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
