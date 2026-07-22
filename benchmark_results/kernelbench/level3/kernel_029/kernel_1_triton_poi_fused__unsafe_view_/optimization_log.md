# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2820 ns — **Best latency**: 2880 ns (**0.9792×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_1_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 2820 ns
- **SM Throughput**: 0.20 % | **Memory**: 0.42 % | **DRAM**: 0.09 %
- **Achieved Occupancy**: 6.20 %
- **IPC Active**: 0.77 inst/cycle | **No Eligible**: 81.95 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_with_correct_3d_indexing | 1.040× | 0.979× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2820 ns | 2880 ns | +60 ns (+2.1%) |
| SM Throughput (%) | 0.20 % | 0.23 % | +0.03 pp |
| Memory Throughput (%) | 0.42 % | 0.44 % | +0.02 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.21 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 81.95 % | 75.35 % | -6.60 pp |
| IPC Active | 0.77 inst/cycle | 0.97 inst/cycle | +0.20 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_3d_with_correct_pid_mapping | 0.996× | 0.881× | ✓ |  |
| B | fix_grid_launch_correct_tile_sizes_xblock32 | 1.034× | 0.928× | ✓ |  |
| D | fix_grid_launch_with_vectorized_loads_via_xblock32_3d_grid | 1.035× | 0.959× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_3... | Δ | fix_grid_launch_c... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2820 ns | 3200 ns | +380 ns (+13.5%) | 3040 ns | +220 ns (+7.8%) | 2940 ns | +120 ns (+4.3%) |
| SM Throughput (%) | 0.20 % | 0.23 % | +0.03 pp | 0.24 % | +0.04 pp | 0.22 % | +0.02 pp |
| Memory Throughput (%) | 0.42 % | 0.39 % | -0.03 pp | 0.43 % | +0.01 pp | 0.43 % | +0.01 pp |
| DRAM Throughput (%) | 0.09 % | 0.08 % | -0.01 pp | 0.08 % | -0.01 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.23 % | +0.03 pp | 6.23 % | +0.03 pp | 6.23 % | +0.03 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 81.95 % | 74.67 % | -7.28 pp | 75.42 % | -6.53 pp | 74.93 % | -7.02 pp |
| IPC Active | 0.77 inst/cycle | 1.03 inst/cycle | +0.26 | 1.03 inst/cycle | +0.26 | 0.90 inst/cycle | +0.13 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_launch_flat_2d_with_xblock32_yblock32 | 1.032× | 0.898× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_f... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2820 ns | 3140 ns | +320 ns (+11.3%) |
| SM Throughput (%) | 0.20 % | 0.23 % | +0.03 pp |
| Memory Throughput (%) | 0.42 % | 0.40 % | -0.02 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.20 % | 6.23 % | +0.03 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 81.95 % | 75.25 % | -6.70 pp |
| IPC Active | 0.77 inst/cycle | 1.04 inst/cycle | +0.27 |

---

## Cumulative Progress

```
Baseline:  2820 ns  (1.000×)

Final best: 2880 ns  (0.9792×)
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
 def triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -20,8 +21,8 @@

     y1 = yindex // 147
     y3 = yindex
     tmp0 = tl.load(in_ptr0 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(((y1 // 8) % 8)) + 3136*x2 + 100352*(y0 // 49) + 301056*(y1 // 64) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr1 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), ymask, eviction_policy='evict_last')
+    tmp1 = tl.load(in_ptr1 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (7*((y1 % 8)) + 56*(((y0 % 49)) // 7) + 392*(y1 // 8) + ((((y0 % 49)) % 7))), xmask & ymask, eviction_policy='evict_last')
     tmp10 = tl.load(in_ptr3 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x2 + 32*(y0 // 49)), xmask & ymask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
@@ -34,8 +35,6 @@

     tmp11 = tmp9 * tmp10
     tmp13 = tmp11 + tmp12
     tl.store(out_ptr0 + (x2 + 32*y3), tmp13, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
