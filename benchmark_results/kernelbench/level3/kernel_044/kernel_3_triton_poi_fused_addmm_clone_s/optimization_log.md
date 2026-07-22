# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2720 ns — **Best latency**: 2690 ns (**1.0112×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_launch_with_yblock_aligned_to_768_stride`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_3_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2720 ns
- **SM Throughput**: 1.28 % | **Memory**: 1.22 % | **DRAM**: 0.28 %
- **Achieved Occupancy**: 6.22 %
- **IPC Active**: 0.29 inst/cycle | **No Eligible**: 90.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_match_kernel_3d_indexing | 1.001× | 0.989× | ✓ | **✓ best** |
| C | fix_grid_dimensionality_to_parallelize_y_dimension | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2720 ns | 2750 ns | +30 ns (+1.1%) | 2720 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 1.28 % | 1.19 % | -0.09 pp | 1.28 % | +0.00 pp |
| Memory Throughput (%) | 1.22 % | 1.14 % | -0.08 pp | 1.22 % | +0.00 pp |
| DRAM Throughput (%) | 0.28 % | 0.27 % | -0.01 pp | 0.27 % | -0.01 pp |
| Achieved Occupancy (%) | 6.22 % | 6.21 % | -0.01 pp | 6.21 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.03 % | 92.15 % | +2.12 pp | 89.59 % | -0.44 pp |
| IPC Active | 0.29 inst/cycle | 0.37 inst/cycle | +0.08 | 0.38 inst/cycle | +0.09 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_1d_grid_to_proper_2d_grid_with_correct_xblock_yblock | 1.015× | 0.989× | ✓ | **✓ best** |
| C | fix_grid_launch_and_kernel_body_for_correct_2d_parallelism | 1.013× | 0.935× | ✓ |  |
| D | rewrite_kernel_with_flat_2d_grid_and_correct_yoffset_formula | 1.008× | 1.023× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_1d_grid_to_pr... | Δ | fix_grid_launch_a... | Δ | rewrite_kernel_wi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2720 ns | 2750 ns | +30 ns (+1.1%) | 2910 ns | +190 ns (+7.0%) | 2660 ns | -60 ns (-2.2%) |
| SM Throughput (%) | 1.28 % | 1.25 % | -0.03 pp | 1.24 % | -0.04 pp | 1.26 % | -0.02 pp |
| Memory Throughput (%) | 1.22 % | 1.19 % | -0.03 pp | 1.24 % | +0.02 pp | 1.26 % | +0.04 pp |
| DRAM Throughput (%) | 0.28 % | 0.27 % | -0.01 pp | 0.27 % | -0.01 pp | 0.28 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.21 % | -0.01 pp | 6.20 % | -0.02 pp | 6.21 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.03 % | 89.31 % | -0.72 pp | 89.04 % | -0.99 pp | 89.22 % | -0.81 pp |
| IPC Active | 0.29 inst/cycle | 0.37 inst/cycle | +0.08 | 0.41 inst/cycle | +0.12 | 0.35 inst/cycle | +0.06 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_cdiv_mismatch_and_add_y_dimension | 1.026× | 1.011× | ✓ | **✓ best** |
| C | fix_1d_grid_to_2d_grid_preserving_exact_kernel_constexpr_values | 1.009× | 1.023× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | fix_1d_grid_to_2d... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2720 ns | 2690 ns | -30 ns (-1.1%) | 2660 ns | -60 ns (-2.2%) |
| SM Throughput (%) | 1.28 % | 1.27 % | -0.01 pp | 1.20 % | -0.08 pp |
| Memory Throughput (%) | 1.22 % | 1.26 % | +0.04 pp | 1.20 % | -0.02 pp |
| DRAM Throughput (%) | 0.28 % | 0.30 % | +0.02 pp | 0.30 % | +0.02 pp |
| Achieved Occupancy (%) | 6.22 % | 6.21 % | -0.01 pp | 6.21 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.03 % | 91.91 % | +1.88 pp | 89.21 % | -0.82 pp |
| IPC Active | 0.29 inst/cycle | 0.34 inst/cycle | +0.05 | 0.35 inst/cycle | +0.06 |

---

## Iteration 5

**Best so far**: 1.0112× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_xblock_cdiv_and_add_y_dimension | 1.026× | 1.023× | ✓ |  |
| B | fix_grid_launch_with_yblock_aligned_to_768_stride | 1.009× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_x... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2720 ns | 2660 ns | -60 ns (-2.2%) | 2690 ns | -30 ns (-1.1%) |
| SM Throughput (%) | 1.28 % | 1.24 % | -0.04 pp | 1.30 % | +0.02 pp |
| Memory Throughput (%) | 1.22 % | 1.24 % | +0.02 pp | 1.30 % | +0.08 pp |
| DRAM Throughput (%) | 0.28 % | 0.29 % | +0.01 pp | 0.29 % | +0.01 pp |
| Achieved Occupancy (%) | 6.22 % | 6.20 % | -0.02 pp | 6.20 % | -0.02 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.03 % | 89.25 % | -0.78 pp | 88.84 % | -1.19 pp |
| IPC Active | 0.29 inst/cycle | 0.41 inst/cycle | +0.12 | 0.42 inst/cycle | +0.13 |

---

## Cumulative Progress

```
Baseline:  2720 ns  (1.000×)
Iter 4     2690 ns  (1.011×)  — fix_grid_launch_cdiv_mismatch_and_add_y_dimension

Final best: 2690 ns  (1.0112×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,11 +5,12 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused_addmm_clone_split_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 98304
     xnumel = 512
-    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
+    yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
     ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
@@ -19,12 +20,10 @@

     x2 = xindex
     y1 = yindex // 768
     y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (768 + y0), ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), xmask & ymask, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (768 + y0), mask=ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), mask=xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
-    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 512*y3), tmp2, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
