# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5790 ns — **Best latency**: 3420 ns (**1.6930×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_coalescing_and_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 5790 ns
- **SM Throughput**: 16.46 % | **Memory**: 42.97 % | **DRAM**: 21.32 %
- **Achieved Occupancy**: 55.82 %
- **IPC Active**: 1.11 inst/cycle | **No Eligible**: 70.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.278× | 1.000× | ✓ |  |
| B | autotune_tile_sizes_for_coalescing_and_occupancy | 1.440× | 1.693× | ✓ | **✓ best** |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.380× | 1.350× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 5790 ns | +0 ns (+0.0%) | 3420 ns | -2370 ns (-40.9%) | 4290 ns | -1500 ns (-25.9%) |
| SM Throughput (%) | 16.46 % | 24.46 % | +8.00 pp | 17.63 % | +1.17 pp | 25.69 % | +9.23 pp |
| Memory Throughput (%) | 42.97 % | 36.03 % | -6.94 pp | 19.82 % | -23.15 pp | 20.37 % | -22.60 pp |
| DRAM Throughput (%) | 21.32 % | 11.36 % | -9.96 pp | 4.82 % | -16.50 pp | 3.84 % | -17.48 pp |
| Achieved Occupancy (%) | 55.82 % | 51.54 % | -4.28 pp | 43.39 % | -12.43 pp | 67.60 % | +11.78 pp |
| Registers/Thread | 26.00 register/thread | 38.00 register/thread | +12 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.99 | +0.25 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.86 % | 56.80 % | -14.06 pp | 54.25 % | -16.61 pp | 39.08 % | -31.78 pp |
| IPC Active | 1.11 inst/cycle | 1.80 inst/cycle | +0.69 | 1.67 inst/cycle | +0.56 | 2.39 inst/cycle | +1.28 |

---

## Iteration 2

**Best so far**: 1.693× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_eviction_policy | 1.395× | 1.708× | ✓ |  |
| B | fix_autotune_lambda_grid_to_enable_store_coalescing | 1.402× | 1.392× | ✓ |  |
| C | add_yblock1_configs_to_fix_strided_store_uncoalescing | 1.403× | 1.372× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_lamb... | Δ | add_yblock1_confi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3390 ns | -2400 ns (-41.5%) | 4160 ns | -1630 ns (-28.2%) | 4220 ns | -1570 ns (-27.1%) |
| SM Throughput (%) | 16.46 % | 17.34 % | +0.88 pp | 27.43 % | +10.97 pp | 27.52 % | +11.06 pp |
| Memory Throughput (%) | 42.97 % | 20.32 % | -22.65 pp | 21.77 % | -21.20 pp | 21.87 % | -21.10 pp |
| DRAM Throughput (%) | 21.32 % | 4.90 % | -16.42 pp | 3.96 % | -17.36 pp | 3.92 % | -17.40 pp |
| Achieved Occupancy (%) | 55.82 % | 47.43 % | -8.39 pp | 69.32 % | +13.50 pp | 67.94 % | +12.12 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 70.86 % | 55.44 % | -15.42 pp | 37.45 % | -33.41 pp | 37.74 % | -33.12 pp |
| IPC Active | 1.11 inst/cycle | 1.50 inst/cycle | +0.39 | 2.30 inst/cycle | +1.19 | 2.39 inst/cycle | +1.28 |

---

## Iteration 3

**Best so far**: 1.693× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_max_contiguous_multiple_of_hints | 1.364× | 1.739× | ✓ |  |
| C | fix_static_grid_to_lambda_grid_to_activate_autotune | 1.398× | 1.340× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5790 ns | 3330 ns | -2460 ns (-42.5%) | 4320 ns | -1470 ns (-25.4%) |
| SM Throughput (%) | 16.46 % | 17.08 % | +0.62 pp | 27.76 % | +11.30 pp |
| Memory Throughput (%) | 42.97 % | 20.58 % | -22.39 pp | 22.02 % | -20.95 pp |
| DRAM Throughput (%) | 21.32 % | 4.96 % | -16.36 pp | 3.83 % | -17.49 pp |
| Achieved Occupancy (%) | 55.82 % | 44.53 % | -11.29 pp | 69.63 % | +13.81 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 70.86 % | 55.10 % | -15.76 pp | 38.00 % | -32.86 pp |
| IPC Active | 1.11 inst/cycle | 1.67 inst/cycle | +0.56 | 2.42 inst/cycle | +1.31 |

---

## Cumulative Progress

```
Baseline:  5790 ns  (1.000×)
Iter 1     3420 ns  (1.693×)  — autotune_tile_sizes_for_coalescing_and_occupancy

Final best: 3420 ns  (1.6930×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +38,8 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +47,7 @@

     ynumel = 30
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
