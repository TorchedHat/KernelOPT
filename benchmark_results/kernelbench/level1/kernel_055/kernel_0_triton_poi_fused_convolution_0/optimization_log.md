# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6080 ns — **Best latency**: 4350 ns (**1.3977×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`vectorized_loads_with_fixed_grid_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_055/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6080 ns
- **SM Throughput**: 17.00 % | **Memory**: 40.20 % | **DRAM**: 20.30 %
- **Achieved Occupancy**: 54.92 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 72.41 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.311× | 1.357× | ✓ |  |
| B | autotune_tile_sizes_and_warps_to_improve_occupancy_and_coalescing | 1.455× | 1.742× | ✓ | **✓ best** |
| C | memory_coalescing_via_autotune_tile_swap | 1.450× | 1.000× | ✓ |  |
| D | autotune_tile_sizes_for_occupancy_and_coalescing | 1.411× | 1.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4480 ns | -1600 ns (-26.3%) | 3490 ns | -2590 ns (-42.6%) | 6080 ns | +0 ns (+0.0%) | 4160 ns | -1920 ns (-31.6%) |
| SM Throughput (%) | 17.00 % | 20.83 % | +3.83 pp | 9.67 % | -7.33 pp | 14.35 % | -2.65 pp | 28.05 % | +11.05 pp |
| Memory Throughput (%) | 40.20 % | 31.48 % | -8.72 pp | 19.98 % | -20.22 pp | 26.18 % | -14.02 pp | 22.30 % | -17.90 pp |
| DRAM Throughput (%) | 20.30 % | 14.73 % | -5.57 pp | 4.78 % | -15.52 pp | 5.08 % | -15.22 pp | 3.98 % | -16.32 pp |
| Achieved Occupancy (%) | 54.92 % | 49.61 % | -5.31 pp | 52.20 % | -2.72 pp | 43.28 % | -11.64 pp | 68.04 % | +13.12 pp |
| Registers/Thread | 26.00 register/thread | 27.00 register/thread | +1 | 16.00 register/thread | -10 | 24.00 register/thread | -2 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 1.48 | +0.74 |
| No Eligible (%) | 72.41 % | 55.99 % | -16.42 pp | 73.31 % | +0.90 pp | 67.88 % | -4.53 pp | 38.56 % | -33.85 pp |
| IPC Active | 1.14 inst/cycle | 1.57 inst/cycle | +0.43 | 1.07 inst/cycle | -0.07 | 1.11 inst/cycle | -0.03 | 2.35 inst/cycle | +1.21 |

---

## Iteration 2

**Best so far**: 1.7421× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_with_fixed_grid_lambda | 1.456× | 1.398× | ✓ | **✓ best** |
| C | coalesced_store_via_yblock1_configs | 1.390× | 1.105× | ✓ |  |
| D | vectorized_loads_with_contiguous_hint_and_expanded_autotune | 1.396× | 1.509× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | coalesced_store_v... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4350 ns | -1730 ns (-28.5%) | 5500 ns | -580 ns (-9.5%) | 4030 ns | -2050 ns (-33.7%) |
| SM Throughput (%) | 17.00 % | 26.35 % | +9.35 pp | 28.83 % | +11.83 pp | 26.96 % | +9.96 pp |
| Memory Throughput (%) | 40.20 % | 20.91 % | -19.29 pp | 31.60 % | -8.60 pp | 21.40 % | -18.80 pp |
| DRAM Throughput (%) | 20.30 % | 3.81 % | -16.49 pp | 2.99 % | -17.31 pp | 4.11 % | -16.19 pp |
| Achieved Occupancy (%) | 54.92 % | 68.88 % | +13.96 pp | 73.10 % | +18.18 pp | 70.28 % | +15.36 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 25.00 register/thread | -1 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.41 % | 37.99 % | -34.42 pp | 37.86 % | -34.55 pp | 36.17 % | -36.24 pp |
| IPC Active | 1.14 inst/cycle | 2.40 inst/cycle | +1.26 | 2.43 inst/cycle | +1.29 | 2.41 inst/cycle | +1.27 |

---

## Iteration 3

**Best so far**: 1.7421× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_use_autotune_selected_tile_sizes | 1.389× | 1.451× | ✓ |  |
| B | shared_memory_transpose_for_coalesced_store | 1.437× | 1.407× | ✓ |  |
| D | add_triton_autotune | 1.404× | 1.498× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | shared_memory_tra... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4190 ns | -1890 ns (-31.1%) | 4320 ns | -1760 ns (-28.9%) | 4060 ns | -2020 ns (-33.2%) |
| SM Throughput (%) | 17.00 % | 25.92 % | +8.92 pp | 26.68 % | +9.68 pp | 27.12 % | +10.12 pp |
| Memory Throughput (%) | 40.20 % | 20.61 % | -19.59 pp | 20.13 % | -20.07 pp | 21.51 % | -18.69 pp |
| DRAM Throughput (%) | 20.30 % | 3.98 % | -16.32 pp | 3.83 % | -16.47 pp | 4.08 % | -16.22 pp |
| Achieved Occupancy (%) | 54.92 % | 73.47 % | +18.55 pp | 69.50 % | +14.58 pp | 68.97 % | +14.05 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.41 % | 40.36 % | -32.05 pp | 36.41 % | -36.00 pp | 39.33 % | -33.08 pp |
| IPC Active | 1.14 inst/cycle | 1.96 inst/cycle | +0.82 | 2.42 inst/cycle | +1.28 | 2.42 inst/cycle | +1.28 |

---

## Iteration 4

**Best so far**: 1.7421× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | store_side_multiple_of_hint_with_evict_first_and_grid_lambda_fix | 1.399× | 1.462× | ✓ |  |
| C | fix_grid_lambda_and_evict_first_store_for_uncoalesced_writes | 0.589× | 1.388× | ✓ |  |


### Metric Comparison

| Metric | Baseline | store_side_multip... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6080 ns | 4160 ns | -1920 ns (-31.6%) | 4380 ns | -1700 ns (-28.0%) |
| SM Throughput (%) | 17.00 % | 27.63 % | +10.63 pp | 28.22 % | +11.22 pp |
| Memory Throughput (%) | 40.20 % | 22.06 % | -18.14 pp | 21.23 % | -18.97 pp |
| DRAM Throughput (%) | 20.30 % | 3.97 % | -16.33 pp | 3.78 % | -16.52 pp |
| Achieved Occupancy (%) | 54.92 % | 67.74 % | +12.82 pp | 69.47 % | +14.55 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 0.74 | 1.48 | +0.74 | 1.48 | +0.74 |
| No Eligible (%) | 72.41 % | 37.21 % | -35.20 pp | 36.84 % | -35.57 pp |
| IPC Active | 1.14 inst/cycle | 2.34 inst/cycle | +1.20 | 2.48 inst/cycle | +1.34 |

---

## Cumulative Progress

```
Baseline:  6080 ns  (1.000×)
Iter 1     3490 ns  (1.742×)  — autotune_tile_sizes_and_warps_to_improve_occupancy_and_coalescing

Final best: 4350 ns  (1.3977×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8, num_stages=3),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 1}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 2}, num_warps=4, num_stages=3),
+    triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 4}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 64,   'YBLOCK': 8}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 64,   'YBLOCK': 16}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 32,   'YBLOCK': 16}, num_warps=4, num_stages=2),
+    triton.Config({'XBLOCK': 128,  'YBLOCK': 16}, num_warps=8, num_stages=3),
+], key=['ynumel', 'xnumel'])
 @triton.jit
 def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 30
@@ -19,10 +34,9 @@

     y3 = yindex
     y0 = (yindex % 3)
     y1 = yindex // 3
-    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
+    load_idx = tl.max_contiguous(tl.multiple_of(x2 + 50176 * y3, [1, 16]), [1, 16])
+    tmp0 = tl.load(in_ptr0 + load_idx, xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +44,7 @@

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
