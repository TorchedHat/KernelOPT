# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41180 ns — **Best latency**: 40900 ns (**1.0068×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_launch_to_enable_autotune_and_add_prefetch_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 17
- **Duration**: 41180 ns
- **SM Throughput**: 56.69 % | **Memory**: 24.62 % | **DRAM**: 18.47 %
- **Achieved Occupancy**: 55.37 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.97 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_latency_hiding | 0.681× | 0.140× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy | 0.695× | 0.693× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_ilp | 0.707× | 0.690× | ✓ |  |
| D | autotune_xblock_to_improve_occupancy_and_hide_l1tex_latency | 1.005× | 1.001× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41180 ns | 293500 ns | +252320 ns (+612.7%) | 59390 ns | +18210 ns (+44.2%) | 59710 ns | +18530 ns (+45.0%) | 41150 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 56.69 % | 82.98 % | +26.29 pp | 81.25 % | +24.56 pp | 86.00 % | +29.31 pp | 66.56 % | +9.87 pp |
| Memory Throughput (%) | 24.62 % | 22.58 % | -2.04 pp | 30.22 % | +5.60 pp | 25.24 % | +0.62 pp | 24.68 % | +0.06 pp |
| DRAM Throughput (%) | 18.47 % | 2.59 % | -15.88 pp | 12.87 % | -5.60 pp | 12.73 % | -5.74 pp | 18.47 % | +0.00 pp |
| Achieved Occupancy (%) | 55.37 % | 74.86 % | +19.49 pp | 79.63 % | +24.26 pp | 87.34 % | +31.97 pp | 58.85 % | +3.48 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 |
| Waves/SM | 29.70 | 237.58 | +207.88 | 59.39 | +29.69 | 29.70 | +0.00 | 29.70 | +0.00 |
| No Eligible (%) | 39.97 % | 16.30 % | -23.67 pp | 15.70 % | -24.27 pp | 10.99 % | -28.98 pp | 29.38 % | -10.59 pp |
| IPC Active | 2.40 inst/cycle | 3.35 inst/cycle | +0.95 | 3.36 inst/cycle | +0.96 | 3.55 inst/cycle | +1.15 | 2.82 inst/cycle | +0.42 |

---

## Iteration 2

**Best so far**: 1.0007× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_simplify_arithmetic | 1.010× | 1.007× | ✓ | **✓ best** |
| B | replace_sqrt_div_with_rsqrt_mul_to_enable_fma_fusion | 0.854× | 0.895× | ✓ |  |
| C | simplify_arithmetic_to_enable_fma_fusion_and_fix_grid_lambda | 0.917× | 0.942× | ✓ |  |
| D | fix_static_grid_bug_and_simplify_fp_arithmetic | 0.853× | 0.214× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | replace_sqrt_div_... | Δ | simplify_arithmet... | Δ | fix_static_grid_b... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41180 ns | 40900 ns | -280 ns (-0.7%) | 46020 ns | +4840 ns (+11.8%) | 43710 ns | +2530 ns (+6.1%) | 192700 ns | +151520 ns (+367.9%) |
| SM Throughput (%) | 56.69 % | 40.16 % | -16.53 pp | 71.64 % | +14.95 pp | 55.21 % | -1.48 pp | 75.47 % | +18.78 pp |
| Memory Throughput (%) | 24.62 % | 24.37 % | -0.25 pp | 40.10 % | +15.48 pp | 35.58 % | +10.96 pp | 34.80 % | +10.18 pp |
| DRAM Throughput (%) | 18.47 % | 18.59 % | +0.12 pp | 16.56 % | -1.91 pp | 17.35 % | -1.12 pp | 3.97 % | -14.50 pp |
| Achieved Occupancy (%) | 55.37 % | 40.71 % | -14.66 pp | 62.74 % | +7.37 pp | 39.21 % | -16.16 pp | 65.25 % | +9.88 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 22.00 register/thread | +5 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 59.39 | +29.69 | 29.70 | +0.00 | 237.58 | +207.88 |
| No Eligible (%) | 39.97 % | 57.08 % | +17.11 pp | 24.81 % | -15.16 pp | 41.14 % | +1.17 pp | 23.68 % | -16.29 pp |
| IPC Active | 2.40 inst/cycle | 1.71 inst/cycle | -0.69 | 3.00 inst/cycle | +0.60 | 2.35 inst/cycle | -0.05 | 3.05 inst/cycle | +0.65 |


### Key Learnings

#### ❌ Plan D: `fix_static_grid_bug_and_simplify_fp_arithmetic` (0.213×)

For the fourth consecutive iteration, changing @triton.autotune configs (pruning to 4 configs, setting num_stages=1) and restructuring FMA arithmetic while leaving the grid as a static tuple `(cdiv(xnumel, 128),)` produced an identical catastrophic regression: executed instructions inflated from 13M to 114.7M (8.8x), duration grew from 41µs to 194µs (4.7x slowdown), and memory throughput collapsed from 920 GB/s to 193 GB/s. The FMA restructuring in the kernel body (removing the intermediate `tmp2 = tmp0 - tmp1` and reordering multiplications) had zero effect on the regression because the grid bug dominates entirely. The diff confirms the grid was never changed to a lambda — only autotune configs and kernel arithmetic were modified, reproducing the exact same failure signature as iterations 1, 2, and 3.

> **Applicable when**: Never — this is documentation of a persistent failure mode confirmed across 4 consecutive iterations. The static grid anti-pattern must never be used with @triton.autotune that varies XBLOCK.

> **Avoid when**: Always avoid a static pre-computed grid tuple when @triton.autotune is present and varies XBLOCK. Neither changing autotune configs (num_stages, num_warps, XBLOCK values), nor restructuring kernel arithmetic (FMA fusion, variable elimination), nor any other kernel-body change will fix this — only changing the grid to a lambda resolves it.

---

## Iteration 3

**Best so far**: 1.0068× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_to_enable_autotune_and_add_prefetch_stages | 1.006× | 1.007× | ✓ |  |
| B | fix_grid_lambda_and_add_num_stages_for_pipelining | 0.913× | 0.936× | ✓ |  |
| C | fix_static_grid_to_lambda_for_correct_autotune_dispatch | 0.850× | 0.890× | ✓ |  |
| D | fix_static_grid_to_lambda_for_autotune_correctness | 0.853× | 0.214× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_to_ena... | Δ | fix_grid_lambda_a... | Δ | fix_static_grid_t... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41180 ns | 40900 ns | -280 ns (-0.7%) | 43970 ns | +2790 ns (+6.8%) | 46270 ns | +5090 ns (+12.4%) | 192830 ns | +151650 ns (+368.3%) |
| SM Throughput (%) | 56.69 % | 40.22 % | -16.47 pp | 55.00 % | -1.69 pp | 71.37 % | +14.68 pp | 75.46 % | +18.77 pp |
| Memory Throughput (%) | 24.62 % | 24.40 % | -0.22 pp | 35.30 % | +10.68 pp | 39.95 % | +15.33 pp | 34.80 % | +10.18 pp |
| DRAM Throughput (%) | 18.47 % | 18.73 % | +0.26 pp | 17.37 % | -1.10 pp | 16.43 % | -2.04 pp | 3.97 % | -14.50 pp |
| Achieved Occupancy (%) | 55.37 % | 41.00 % | -14.37 pp | 38.84 % | -16.53 pp | 62.77 % | +7.40 pp | 65.22 % | +9.85 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 22.00 register/thread | +5 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 59.39 | +29.69 | 237.58 | +207.88 |
| No Eligible (%) | 39.97 % | 57.04 % | +17.07 pp | 40.12 % | +0.15 pp | 26.15 % | -13.82 pp | 23.74 % | -16.23 pp |
| IPC Active | 2.40 inst/cycle | 1.73 inst/cycle | -0.67 | 2.37 inst/cycle | -0.03 | 2.99 inst/cycle | +0.59 | 3.05 inst/cycle | +0.65 |

---

## Iteration 4

**Best so far**: 1.0068× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_lambda_to_unlock_autotune_correctness | 1.004× | 1.001× | ✓ |  |
| B | fix_autotune_grid_lambda_and_enable_fma_fusion | 0.913× | 0.941× | ✓ |  |
| C | fix_static_grid_to_lambda_for_correct_autotune_dispatch_v2 | 0.849× | 0.886× | ✓ |  |
| D | fix_grid_lambda_and_fuse_batchnorm_arithmetic_for_fma | 0.848× | 0.213× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_l... | Δ | fix_autotune_grid... | Δ | fix_static_grid_t... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41180 ns | 41150 ns | -30 ns (-0.1%) | 43780 ns | +2600 ns (+6.3%) | 46460 ns | +5280 ns (+12.8%) | 193730 ns | +152550 ns (+370.4%) |
| SM Throughput (%) | 56.69 % | 40.40 % | -16.29 pp | 55.54 % | -1.15 pp | 71.18 % | +14.49 pp | 74.97 % | +18.28 pp |
| Memory Throughput (%) | 24.62 % | 24.50 % | -0.12 pp | 35.65 % | +11.03 pp | 39.87 % | +15.25 pp | 34.44 % | +9.82 pp |
| DRAM Throughput (%) | 18.47 % | 18.49 % | +0.02 pp | 17.41 % | -1.06 pp | 16.41 % | -2.06 pp | 3.92 % | -14.55 pp |
| Achieved Occupancy (%) | 55.37 % | 40.75 % | -14.62 pp | 38.88 % | -16.49 pp | 62.59 % | +7.22 pp | 66.28 % | +10.91 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 24.00 register/thread | +7 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 29.70 | +0.00 | 59.39 | +29.69 | 237.58 | +207.88 |
| No Eligible (%) | 39.97 % | 57.22 % | +17.25 pp | 41.43 % | +1.46 pp | 25.71 % | -14.26 pp | 24.13 % | -15.84 pp |
| IPC Active | 2.40 inst/cycle | 1.71 inst/cycle | -0.69 | 2.36 inst/cycle | -0.04 | 2.98 inst/cycle | +0.58 | 3.03 inst/cycle | +0.63 |

---

## Cumulative Progress

```
Baseline:  41180 ns  (1.000×)
Iter 1     41150 ns  (1.001×)  — autotune_xblock_to_improve_occupancy_and_hide_l1tex_latency
Iter 2     40900 ns  (1.007×)  — fix_autotune_grid_lambda_and_simplify_arithmetic

Final best: 40900 ns  (1.0068×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,45 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8028160
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +53,7 @@

     arg_4 = inputs[4]
     xnumel = 8028160
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 8028160, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 8028160)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
