# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_addmm_clone_split_transpose_view_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2750 ns — **Best latency**: 2620 ns (**1.0496×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`baseline`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_addmm_clone_split_transpose_view_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_2_triton_poi_fused_addmm_clone_s/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.03
- **Registers/thread**: 28
- **Duration**: 2750 ns
- **SM Throughput**: 1.34 % | **Memory**: 1.27 % | **DRAM**: 0.27 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.38 inst/cycle | **No Eligible**: 90.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_preserve_kernel_body_exactly | 1.002× | 1.000× | ✓ | **✓ best** |
| B | fix_grid_dimensionality_for_full_parallelism | 1.001× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_p... | Δ | fix_grid_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2750 ns | 2750 ns | +0 ns (+0.0%) | 2720 ns | -30 ns (-1.1%) |
| SM Throughput (%) | 1.34 % | 1.30 % | -0.04 pp | 1.32 % | -0.02 pp |
| Memory Throughput (%) | 1.27 % | 1.24 % | -0.03 pp | 1.26 % | -0.01 pp |
| DRAM Throughput (%) | 0.27 % | 0.27 % | +0.00 pp | 0.28 % | +0.01 pp |
| Achieved Occupancy (%) | 6.21 % | 6.21 % | +0.00 pp | 6.21 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.34 % | 89.33 % | -1.01 pp | 89.82 % | -0.52 pp |
| IPC Active | 0.38 inst/cycle | 0.36 inst/cycle | -0.02 | 0.39 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_launch_with_2d_grid_yblock_divisor_of_768 | 1.029× | 1.022× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2750 ns | 2690 ns | -60 ns (-2.2%) |
| SM Throughput (%) | 1.34 % | 1.26 % | -0.08 pp |
| Memory Throughput (%) | 1.27 % | 1.26 % | -0.01 pp |
| DRAM Throughput (%) | 0.27 % | 0.29 % | +0.02 pp |
| Achieved Occupancy (%) | 6.21 % | 6.20 % | -0.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 90.34 % | 89.15 % | -1.19 pp |
| IPC Active | 0.38 inst/cycle | 0.42 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.0223× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 5

**Best so far**: 1.0223× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_3d_kernel_body_untouched | 1.018× | 0.989× | ✓ |  |
| B | fix_grid_parallelism_yblock_768_exact_divisor | 1.029× | 1.022× | ✓ |  |
| C | fix_3d_grid_launch_yblock_128_divisor_of_768 | 1.017× | 1.000× | ✓ |  |
| D | vectorized_stores_with_correct_3d_grid_xblock512_yblock128 | 1.018× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_3d_kerne... | Δ | fix_grid_parallel... | Δ | fix_3d_grid_launc... | Δ | vectorized_stores... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2750 ns | 2780 ns | +30 ns (+1.1%) | 2690 ns | -60 ns (-2.2%) | 2750 ns | +0 ns (+0.0%) | 2820 ns | +70 ns (+2.5%) |
| SM Throughput (%) | 1.34 % | 1.28 % | -0.06 pp | 1.26 % | -0.08 pp | 1.20 % | -0.14 pp | 1.28 % | -0.06 pp |
| Memory Throughput (%) | 1.27 % | 1.22 % | -0.05 pp | 1.26 % | -0.01 pp | 1.15 % | -0.12 pp | 1.22 % | -0.05 pp |
| DRAM Throughput (%) | 0.27 % | 0.27 % | +0.00 pp | 0.28 % | +0.01 pp | 0.27 % | +0.00 pp | 0.28 % | +0.01 pp |
| Achieved Occupancy (%) | 6.21 % | 6.21 % | +0.00 pp | 6.20 % | -0.01 pp | 6.21 % | +0.00 pp | 6.21 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 | 0.03 | +0.00 |
| No Eligible (%) | 90.34 % | 89.23 % | -1.11 pp | 91.51 % | +1.17 pp | 91.02 % | +0.68 pp | 89.08 % | -1.26 pp |
| IPC Active | 0.38 inst/cycle | 0.38 inst/cycle | +0.00 | 0.41 inst/cycle | +0.03 | 0.39 inst/cycle | +0.01 | 0.31 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  2750 ns  (1.000×)
Iter 3     2690 ns  (1.022×)  — fix_grid_launch_with_2d_grid_yblock_divisor_of_768

Final best: 2620 ns  (1.0496×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_addmm_clone_split_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -23,7 +24,6 @@

     tmp1 = tl.load(in_ptr1 + (768 + y0 + 2304*x2 + 1179648*y1), mask=xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 + tmp1
     tl.store(out_ptr0 + (x2 + 512*y3), tmp2, mask=xmask & ymask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
