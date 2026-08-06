# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2750 ns — **Best latency**: 2560 ns (**1.0742×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`autotune_tile_sizes_for_small_grid_utilization`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.06
- **Registers/thread**: 28
- **Duration**: 2750 ns
- **SM Throughput**: 2.46 % | **Memory**: 2.30 % | **DRAM**: 0.31 %
- **Achieved Occupancy**: 6.07 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 90.55 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_small_grid_utilization | 1.013× | 1.074× | ✓ | **✓ best** |
| B | rightsize_tile_dimensions_to_eliminate_wasted_threads | 0.978× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | rightsize_tile_di... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2750 ns | 2560 ns | -190 ns (-6.9%) | 2720 ns | -30 ns (-1.1%) |
| SM Throughput (%) | 2.46 % | 0.42 % | -2.04 pp | 2.68 % | +0.22 pp |
| Memory Throughput (%) | 2.30 % | 1.23 % | -1.07 pp | 2.50 % | +0.20 pp |
| DRAM Throughput (%) | 0.31 % | 1.23 % | +0.92 pp | 0.31 % | +0.00 pp |
| Achieved Occupancy (%) | 6.07 % | 3.07 % | -3.00 pp | 6.07 % | +0.00 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.03 | -0.03 | 0.06 | +0.00 |
| No Eligible (%) | 90.55 % | 96.12 % | +5.57 pp | 90.99 % | +0.44 pp |
| IPC Active | 0.33 inst/cycle | 0.06 inst/cycle | -0.27 | 0.35 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  2750 ns  (1.000×)
Iter 1     2560 ns  (1.074×)  — autotune_tile_sizes_for_small_grid_utilization

Final best: 2560 ns  (1.0742×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 1024
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = tl.full([YBLOCK, 1], True, tl.int1)
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +34,8 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +43,7 @@

     ynumel = 1024
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1[grid](arg_0, arg_1, 1024, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
