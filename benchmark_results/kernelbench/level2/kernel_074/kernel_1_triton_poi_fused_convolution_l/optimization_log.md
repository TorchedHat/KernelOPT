# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 68770 ns — **Best latency**: 67260 ns (**1.0225×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`reorder_index_decomposition_for_warp_coalescing`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_074/kernel_1_triton_poi_fused_convolution_l/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 20
- **Duration**: 68770 ns
- **SM Throughput**: 34.53 % | **Memory**: 88.60 % | **DRAM**: 88.60 %
- **Achieved Occupancy**: 84.14 %
- **IPC Active**: 1.44 inst/cycle | **No Eligible**: 64.12 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_coalesced_access_pattern | 1.001× | 0.402× | ✓ |  |
| B | memory_coalescing_reorder_index_decomposition | 1.024× | 1.011× | ✓ |  |
| C | software_pipelining_and_autotune_xblock | 1.000× | 1.000× | ✓ |  |
| D | fix_eviction_policy_and_autotune_xblock | 1.024× | 0.236× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | software_pipelini... | Δ | fix_eviction_poli... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 171300 ns | +102530 ns (+149.1%) | 68030 ns | -740 ns (-1.1%) | 68800 ns | +30 ns (+0.0%) | 290780 ns | +222010 ns (+322.8%) |
| SM Throughput (%) | 34.53 % | 58.30 % | +23.77 pp | 36.02 % | +1.49 pp | 46.13 % | +11.60 pp | 70.39 % | +35.86 pp |
| Memory Throughput (%) | 88.60 % | 72.85 % | -15.75 pp | 83.32 % | -5.28 pp | 88.56 % | -0.04 pp | 57.72 % | -30.88 pp |
| DRAM Throughput (%) | 88.60 % | 36.26 % | -52.34 pp | 83.32 % | -5.28 pp | 88.56 % | -0.04 pp | 19.40 % | -69.20 pp |
| Achieved Occupancy (%) | 84.14 % | 91.52 % | +7.38 pp | 83.64 % | -0.50 pp | 84.18 % | +0.04 pp | 68.79 % | -15.35 pp |
| Registers/Thread | 20.00 register/thread | 32.00 register/thread | +12 | 16.00 register/thread | -4 | 22.00 register/thread | +2 | 40.00 register/thread | +20 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 82.75 | +51.72 |
| No Eligible (%) | 64.12 % | 40.89 % | -23.23 pp | 63.04 % | -1.08 pp | 52.62 % | -11.50 pp | 29.06 % | -35.06 pp |
| IPC Active | 1.44 inst/cycle | 2.36 inst/cycle | +0.92 | 1.48 inst/cycle | +0.04 | 1.91 inst/cycle | +0.47 | 2.84 inst/cycle | +1.40 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_to_unlock_xblock_exploration | 0.999× | 0.984× | ✓ |  |
| C | revert_vectorized_hint_regression_to_clean_baseline | 1.000× | 0.985× | ✓ |  |
| D | fix_autotune_grid_lambda_and_reduce_register_pressure | 1.710× | 1.022× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | revert_vectorized... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 69860 ns | +1090 ns (+1.6%) | 69790 ns | +1020 ns (+1.5%) | 67260 ns | -1510 ns (-2.2%) |
| SM Throughput (%) | 34.53 % | 45.48 % | +10.95 pp | 33.15 % | -1.38 pp | 47.68 % | +13.15 pp |
| Memory Throughput (%) | 88.60 % | 87.26 % | -1.34 pp | 87.33 % | -1.27 pp | 84.30 % | -4.30 pp |
| DRAM Throughput (%) | 88.60 % | 87.26 % | -1.34 pp | 87.33 % | -1.27 pp | 84.30 % | -4.30 pp |
| Achieved Occupancy (%) | 84.14 % | 84.30 % | +0.16 pp | 84.38 % | +0.24 pp | 83.68 % | -0.46 pp |
| Registers/Thread | 20.00 register/thread | 22.00 register/thread | +2 | 20.00 register/thread | +0 | 22.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 64.12 % | 52.34 % | -11.78 pp | 65.86 % | +1.74 pp | 50.69 % | -13.43 pp |
| IPC Active | 1.44 inst/cycle | 1.91 inst/cycle | +0.47 | 1.35 inst/cycle | -0.09 | 1.99 inst/cycle | +0.55 |

---

## Iteration 3

**Best so far**: 1.0225× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reorder_index_decomposition_for_warp_coalescing | 1.022× | 1.022× | ✓ |  |
| B | autotune_xblock_with_correct_lambda_grid_for_l1_reuse | 1.001× | 0.978× | ✓ |  |
| C | fix_autotune_grid_lambda_enable_xblock_exploration | 0.999× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reorder_index_dec... | Δ | autotune_xblock_w... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 67260 ns | -1510 ns (-2.2%) | 70300 ns | +1530 ns (+2.2%) | 70300 ns | +1530 ns (+2.2%) |
| SM Throughput (%) | 34.53 % | 47.46 % | +12.93 pp | 44.26 % | +9.73 pp | 46.41 % | +11.88 pp |
| Memory Throughput (%) | 88.60 % | 84.22 % | -4.38 pp | 86.69 % | -1.91 pp | 86.70 % | -1.90 pp |
| DRAM Throughput (%) | 88.60 % | 84.22 % | -4.38 pp | 86.69 % | -1.91 pp | 86.70 % | -1.90 pp |
| Achieved Occupancy (%) | 84.14 % | 83.77 % | -0.37 pp | 84.14 % | +0.00 pp | 84.34 % | +0.20 pp |
| Registers/Thread | 20.00 register/thread | 22.00 register/thread | +2 | 23.00 register/thread | +3 | 22.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 64.12 % | 49.29 % | -14.83 pp | 54.82 % | -9.30 pp | 52.83 % | -11.29 pp |
| IPC Active | 1.44 inst/cycle | 1.98 inst/cycle | +0.54 | 1.83 inst/cycle | +0.39 | 1.86 inst/cycle | +0.42 |

---

## Iteration 4

**Best so far**: 1.0225× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_expand_autotune_num_warps_for_latency_hiding | 1.024× | 1.032× | ✓ |  |
| B | evict_first_streaming_loads_with_correct_autotune_grid_lambda | 1.024× | 1.040× | ✓ |  |
| C | fix_autotune_grid_lambda_and_remove_hardcoded_xblock_kwarg | 1.001× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | evict_first_strea... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 68770 ns | 66660 ns | -2110 ns (-3.1%) | 66140 ns | -2630 ns (-3.8%) | 69980 ns | +1210 ns (+1.8%) |
| SM Throughput (%) | 34.53 % | 48.31 % | +13.78 pp | 46.11 % | +11.58 pp | 46.13 % | +11.60 pp |
| Memory Throughput (%) | 88.60 % | 85.02 % | -3.58 pp | 85.57 % | -3.03 pp | 87.08 % | -1.52 pp |
| DRAM Throughput (%) | 88.60 % | 85.02 % | -3.58 pp | 85.57 % | -3.03 pp | 87.08 % | -1.52 pp |
| Achieved Occupancy (%) | 84.14 % | 84.04 % | -0.10 pp | 84.05 % | -0.09 pp | 84.36 % | +0.22 pp |
| Registers/Thread | 20.00 register/thread | 22.00 register/thread | +2 | 23.00 register/thread | +3 | 22.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 64.12 % | 49.78 % | -14.34 pp | 51.95 % | -12.17 pp | 52.41 % | -11.71 pp |
| IPC Active | 1.44 inst/cycle | 1.96 inst/cycle | +0.52 | 1.93 inst/cycle | +0.49 | 1.90 inst/cycle | +0.46 |

---

## Cumulative Progress

```
Baseline:  68770 ns  (1.000×)
Iter 2     67260 ns  (1.022×)  — fix_autotune_grid_lambda_and_reduce_register_pressure

Final best: 67260 ns  (1.0225×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 8388608
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 32)
     x1 = ((xindex // 32) % 32)
     x2 = xindex // 1024
     x3 = xindex
-    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp7 = tl.load(in_ptr0 + (4096 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp9 = tl.load(in_ptr0 + (4097 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp11 = tl.load(in_ptr0 + (4160 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
+    tmp13 = tl.load(in_ptr0 + (4161 + 2*x0 + 128*x1 + 8192*x2), xmask, eviction_policy='evict_first')
     tmp2 = tl.maximum(tmp0, tmp1)
     tmp4 = tl.maximum(tmp2, tmp3)
     tmp6 = tl.maximum(tmp4, tmp5)
@@ -30,16 +40,14 @@

     tmp10 = tl.maximum(tmp8, tmp9)
     tmp12 = tl.maximum(tmp10, tmp11)
     tmp14 = tl.maximum(tmp12, tmp13)
-    tl.store(out_ptr0 + (x3), tmp14, None)
-
-
+    tl.store(out_ptr0 + (x3), tmp14, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 8388608
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1[grid](arg_0, arg_1, 8388608, XBLOCK=128)
+    triton_poi_fused_convolution_leaky_relu_max_pool3d_with_indices_mul_1[grid](arg_0, arg_1, 8388608)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
